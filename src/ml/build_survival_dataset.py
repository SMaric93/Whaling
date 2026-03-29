"""
ML Layer — Survival Dataset Builder.

Unit: patch-spell-day (person-period data for hazard models).

Reuses:
- patch_spells.expand_to_patch_days()
- data_builder.build_analysis_panel() for voyage-level merges
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from src.ml.config import ML_CFG, ML_DATA_DIR, DATA_DIR

logger = logging.getLogger(__name__)

OUTPUT_PATH = ML_DATA_DIR / "survival_dataset.parquet"


def build_survival_dataset(
    *,
    force_rebuild: bool = False,
    save: bool = True,
) -> pd.DataFrame:
    """
    Build the survival (patch-spell-day) dataset for hazard models.

    Returns
    -------
    pd.DataFrame
        One row per day within a patch spell, with exit indicator,
        negative-signal state variables, and held-out type estimates.
    """
    if OUTPUT_PATH.exists() and not force_rebuild:
        logger.info("Loading cached survival dataset from %s", OUTPUT_PATH)
        return pd.read_parquet(OUTPUT_PATH)

    t0 = time.time()
    logger.info("Building survival dataset...")

    from src.reinforcement.data_builder import (
        load_logbook_positions,
        build_analysis_panel,
    )
    from src.reinforcement.patch_spells import build_patch_spells, expand_to_patch_days

    # Load positions and build patches
    positions = load_logbook_positions()
    patches = build_patch_spells(positions)
    logger.info("Built %d patch spells", len(patches))

    # Expand to patch-day panel (cap for tractability)
    MAX_SPELLS = 10000  # Cap to keep expansion tractable
    if len(patches) > MAX_SPELLS:
        logger.info("Sampling %d of %d patch spells for tractability", MAX_SPELLS, len(patches))
        patches = patches.sample(n=MAX_SPELLS, random_state=ML_CFG.random_seed)

    try:
        patch_days = expand_to_patch_days(patches, positions)
        logger.info("Expanded to %d patch-day observations", len(patch_days))
    except Exception as e:
        logger.warning("expand_to_patch_days failed (%s); building from spell-level data", e)
        patch_days = _build_patch_days_simple(patches)
        logger.info("Built %d patch-day observations from spell data", len(patch_days))

    # Load voyage data for merging
    voyages = build_analysis_panel(require_akm=True, require_logbook=False)

    # ── Merge voyage-level features ─────────────────────────────────
    merge_cols = []
    for c in ["captain_id", "agent_id", "vessel_id",
              "theta", "psi", "theta_heldout", "psi_heldout",
              "captain_experience", "captain_voyage_num", "novice",
              "tonnage", "rig", "crew_count",
              "home_port", "year_out"]:
        if c in voyages.columns:
            merge_cols.append(c)

    if merge_cols and "voyage_id" in patch_days.columns:
        patch_days = patch_days.merge(
            voyages[["voyage_id"] + merge_cols].drop_duplicates("voyage_id"),
            on="voyage_id",
            how="left",
        )

    # Rename holdout columns (prefer heldout, fall back to in-sample)
    for old, new in [("theta_heldout", "theta_hat_holdout"),
                     ("psi_heldout", "psi_hat_holdout")]:
        if old in patch_days.columns and new not in patch_days.columns:
            patch_days.rename(columns={old: new}, inplace=True)

    # Fall back to in-sample theta/psi if holdout not available
    if "theta_hat_holdout" not in patch_days.columns or patch_days.get("theta_hat_holdout", pd.Series(dtype=float)).isna().all():
        if "theta" in patch_days.columns and patch_days["theta"].notna().any():
            patch_days["theta_hat_holdout"] = patch_days["theta"]
            logger.warning("Survival dataset: using in-sample theta as holdout fallback")
    if "psi_hat_holdout" not in patch_days.columns or patch_days.get("psi_hat_holdout", pd.Series(dtype=float)).isna().all():
        if "psi" in patch_days.columns and patch_days["psi"].notna().any():
            patch_days["psi_hat_holdout"] = patch_days["psi"]
            logger.warning("Survival dataset: using in-sample psi as holdout fallback")

    # ── Derive year column ──────────────────────────────────────────
    if "year" not in patch_days.columns:
        if "year_out" in patch_days.columns:
            patch_days["year"] = patch_days["year_out"]
        elif "obs_date" in patch_days.columns:
            patch_days["year"] = pd.to_datetime(patch_days["obs_date"]).dt.year

    # ── Season remaining (approximate) ─────────────────────────────
    if "season_remaining" not in patch_days.columns and "duration_day" in patch_days.columns:
        patch_days["season_remaining"] = np.clip(180 - patch_days.get("duration_day", 0), 0, 180)

    # ── Median imputation for tonnage ───────────────────────────────
    if "tonnage" in patch_days.columns:
        tonnage_median = patch_days["tonnage"].median()
        n_imp = patch_days["tonnage"].isna().sum()
        if n_imp > 0 and pd.notna(tonnage_median):
            patch_days["tonnage"] = patch_days["tonnage"].fillna(tonnage_median)
            logger.info("Imputed %d missing tonnage values with median (%.1f)", n_imp, tonnage_median)

    # ── Construct patch spell id ────────────────────────────────────
    if "patch_spell_id" not in patch_days.columns:
        id_cols = [c for c in ["voyage_id", "patch_id"] if c in patch_days.columns]
        if id_cols:
            patch_days["patch_spell_id"] = (
                patch_days[id_cols].astype(str).agg("_".join, axis=1)
            )
        else:
            patch_days["patch_spell_id"] = patch_days.index.astype(str)

    # ── Duration day within spell ───────────────────────────────────
    if "duration_day" not in patch_days.columns:
        if "day_in_patch" in patch_days.columns:
            patch_days["duration_day"] = patch_days["day_in_patch"]
        else:
            patch_days["duration_day"] = patch_days.groupby(
                "patch_spell_id"
            ).cumcount() + 1

    # ── Event exit ──────────────────────────────────────────────────
    if "event_exit" not in patch_days.columns:
        if "exit_tomorrow" in patch_days.columns:
            patch_days["event_exit"] = patch_days["exit_tomorrow"]
        else:
            # Last day in each spell is exit
            patch_days["event_exit"] = 0
            last_day = patch_days.groupby("patch_spell_id")["duration_day"].transform("max")
            patch_days.loc[patch_days["duration_day"] == last_day, "event_exit"] = 1

    # ── Scarcity proxy ──────────────────────────────────────────────
    if "scarcity" not in patch_days.columns:
        if "encounter_rate" in patch_days.columns:
            patch_days["scarcity"] = 1 - patch_days["encounter_rate"]
        elif "ground_id" in patch_days.columns and "year" in patch_days.columns:
            # Use spell-level encounter rates
            enc_cols = [c for c in patch_days.columns if "encounter" in c.lower()]
            if enc_cols:
                enc_col = enc_cols[0]
                gy_enc = patch_days.groupby(["ground_id", "year"])[enc_col].transform("mean")
                patch_days["scarcity"] = 1 - gy_enc
            else:
                patch_days["scarcity"] = np.nan
        else:
            patch_days["scarcity"] = np.nan

    # ── Sanity checks ───────────────────────────────────────────────
    n_spells = patch_days["patch_spell_id"].nunique()
    n_events = patch_days["event_exit"].sum()
    logger.info("Survival dataset: %d spells, %d exits (%.1f%%)",
                n_spells, n_events, 100 * n_events / max(len(patch_days), 1))

    elapsed = time.time() - t0
    logger.info(
        "Survival dataset built: %d rows, %d columns, %.1fs",
        len(patch_days), len(patch_days.columns), elapsed,
    )

    if save:
        patch_days.to_parquet(OUTPUT_PATH, index=False)
        logger.info("Saved to %s", OUTPUT_PATH)

    return patch_days


def _build_patch_days_simple(patches: pd.DataFrame) -> pd.DataFrame:
    """
    Simple fallback: expand each patch spell into daily rows using duration only.

    Avoids position-level joins when expand_to_patch_days is too slow.
    """
    rows = []
    entry_col = "entry_date" if "entry_date" in patches.columns else "start_date"
    exit_col = "exit_date" if "exit_date" in patches.columns else "end_date"

    for _, spell in patches.iterrows():
        vid = spell.get("voyage_id", None)
        sid = spell.get("patch_spell_id", spell.get("patch_id", spell.name))
        try:
            start = pd.to_datetime(spell[entry_col])
            end = pd.to_datetime(spell[exit_col])
        except Exception:
            continue

        duration = max(1, (end - start).days + 1)
        for d in range(duration):
            rows.append({
                "voyage_id": vid,
                "patch_spell_id": sid,
                "day_in_patch": d + 1,
                "duration_day": d + 1,
                "obs_date": start + pd.Timedelta(days=d),
                "event_exit": 1 if d == duration - 1 else 0,
            })

    df = pd.DataFrame(rows)
    if "obs_date" in df.columns:
        df["year"] = pd.to_datetime(df["obs_date"]).dt.year
    return df
