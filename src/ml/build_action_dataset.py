"""
ML Layer — Action Dataset Builder.

Unit: ship-day (or ship-observation) within a ground,
restricted to active search segments.

Reuses:
- data_builder.load_logbook_positions()
- ground_spells.classify_ground_vectorized(), build_ground_spells()
- patch_spells.build_patch_spells()
- search_metrics (movement geometry)
- type_estimation (theta/psi holdout)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.ml.config import ML_CFG, ML_DATA_DIR, DATA_DIR

logger = logging.getLogger(__name__)

OUTPUT_PATH = ML_DATA_DIR / "action_dataset.parquet"


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _haversine_nm(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in nautical miles."""
    R_NM = 3440.065
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R_NM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _bearing_deg(lat1, lon1, lat2, lon2):
    """Vectorized initial bearing in degrees [0, 360)."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(x, y))
    return bearing % 360


def _turn_angle(bearings):
    """Compute turn angles from a bearing series. Returns degrees [-180, 180]."""
    diff = np.diff(bearings)
    # Normalize to [-180, 180]
    diff = (diff + 180) % 360 - 180
    return np.concatenate([[np.nan], diff])


# ═══════════════════════════════════════════════════════════════════════════
# Main Builder
# ═══════════════════════════════════════════════════════════════════════════

def build_action_dataset(
    *,
    force_rebuild: bool = False,
    save: bool = True,
) -> pd.DataFrame:
    """
    Build the ship-day action dataset.

    Returns
    -------
    pd.DataFrame
        One row per ship-day within a ground, with movement geometry,
        search state, and action targets.
    """
    if OUTPUT_PATH.exists() and not force_rebuild:
        logger.info("Loading cached action dataset from %s", OUTPUT_PATH)
        return pd.read_parquet(OUTPUT_PATH)

    t0 = time.time()
    logger.info("Building action dataset...")

    # ── Load positions ──────────────────────────────────────────────────
    from src.reinforcement.data_builder import (
        load_logbook_positions,
        build_analysis_panel,
    )
    from src.reinforcement.ground_spells import (
        classify_ground_vectorized,
        build_ground_spells,
    )
    from src.reinforcement.patch_spells import build_patch_spells

    positions = load_logbook_positions()
    logger.info("Loaded %d logbook positions", len(positions))

    # ── Load voyage panel for merging voyage-level info ──────────────
    voyages = build_analysis_panel(require_akm=True, require_logbook=False)
    logger.info("Loaded %d voyages", len(voyages))

    # ── Classify positions into grounds ─────────────────────────────
    if "ground_id" not in positions.columns:
        positions["ground_id"] = classify_ground_vectorized(
            positions["lat"].values, positions["lon"].values
        )

    # ── Build ground spells for ground/patch identifiers ────────────
    ground_spells = build_ground_spells(positions)
    patch_spells = build_patch_spells(positions)

    # ── Sort within voyage by date ──────────────────────────────────
    positions = positions.sort_values(["voyage_id", "obs_date"]).reset_index(drop=True)

    # ── Movement geometry ───────────────────────────────────────────
    # Previous position
    positions["prev_lat"] = positions.groupby("voyage_id")["lat"].shift(1)
    positions["prev_lon"] = positions.groupby("voyage_id")["lon"].shift(1)

    # Move length (NM)
    mask = positions["prev_lat"].notna()
    positions.loc[mask, "move_length"] = _haversine_nm(
        positions.loc[mask, "prev_lat"].values,
        positions.loc[mask, "prev_lon"].values,
        positions.loc[mask, "lat"].values,
        positions.loc[mask, "lon"].values,
    )

    # Bearing
    positions.loc[mask, "bearing"] = _bearing_deg(
        positions.loc[mask, "prev_lat"].values,
        positions.loc[mask, "prev_lon"].values,
        positions.loc[mask, "lat"].values,
        positions.loc[mask, "lon"].values,
    )

    # Turn angle
    positions["turn_angle"] = np.nan
    for vid, grp in positions.groupby("voyage_id"):
        if len(grp) > 1 and "bearing" in grp.columns:
            ta = _turn_angle(grp["bearing"].values)
            positions.loc[grp.index, "turn_angle"] = ta

    # Speed (NM/day) — approximate as move_length per day gap
    positions["_date_num"] = pd.to_datetime(positions["obs_date"]).astype(np.int64) // 10**9 / 86400
    positions["_day_gap"] = positions.groupby("voyage_id")["_date_num"].diff()
    positions["speed"] = positions["move_length"] / positions["_day_gap"].clip(lower=0.5)

    # Net displacement from voyage start
    first_pos = positions.groupby("voyage_id")[["lat", "lon"]].first()
    first_pos.columns = ["start_lat", "start_lon"]
    positions = positions.merge(first_pos, on="voyage_id", how="left")
    positions["net_displacement"] = _haversine_nm(
        positions["start_lat"].values,
        positions["start_lon"].values,
        positions["lat"].values,
        positions["lon"].values,
    )

    # ── Revisit indicator (same 0.5° cell visited before) ──────────
    positions["_cell"] = (
        (positions["lat"] * 2).round() / 2).astype(str) + "_" + (
        (positions["lon"] * 2).round() / 2).astype(str)
    positions["revisit_indicator"] = (
        positions.groupby("voyage_id")["_cell"]
        .transform(lambda s: s.duplicated())
        .astype(int)
    )

    # ── Encounter-based features ────────────────────────────────────
    enc_col = "encounter" if "encounter" in positions.columns else None
    if enc_col:
        # Handle string encounter column (e.g. 'NoEnc', 'Sight', 'Strike', 'Spoke')
        if positions[enc_col].dtype == "object" or positions[enc_col].dtype.name == "category":
            positions["_enc_flag"] = (~positions[enc_col].isin(["NoEnc", "", "None"])).astype(int)
        elif positions[enc_col].dtype == bool:
            positions["_enc_flag"] = positions[enc_col].astype(int)
        else:
            positions["_enc_flag"] = (pd.to_numeric(positions[enc_col], errors="coerce").fillna(0) > 0).astype(int)
    else:
        positions["_enc_flag"] = 0

    # Days since last success (encounter)
    positions["days_since_last_success"] = np.nan
    for vid, grp in positions.groupby("voyage_id"):
        idx = grp.index.values
        enc_flags = grp["_enc_flag"].values
        dsls = np.full(len(enc_flags), np.nan)
        last_success = -999
        for i, ef in enumerate(enc_flags):
            if ef:
                last_success = i
                dsls[i] = 0
            elif last_success >= 0:
                dsls[i] = i - last_success
        positions.loc[idx, "days_since_last_success"] = dsls

    # Consecutive empty days
    positions["consecutive_empty_days"] = 0
    for vid, grp in positions.groupby("voyage_id"):
        idx = grp.index.values
        enc = grp["_enc_flag"].values
        consec = np.zeros(len(enc), dtype=int)
        for i in range(1, len(enc)):
            if enc[i] == 0:
                consec[i] = consec[i - 1] + 1
            else:
                consec[i] = 0
        positions.loc[idx, "consecutive_empty_days"] = consec

    # ── Time features ───────────────────────────────────────────────
    positions["date"] = pd.to_datetime(positions["obs_date"])
    positions["year"] = positions["date"].dt.year
    positions["voyage_day"] = positions.groupby("voyage_id").cumcount() + 1

    # Days in ground
    positions["days_in_ground"] = positions.groupby(
        ["voyage_id", "ground_id"]
    ).cumcount() + 1

    # ── Phase flags (heuristic) ─────────────────────────────────────
    # Active search: within a whaling ground and not in first/last 10% of voyage
    voyage_lengths = positions.groupby("voyage_id")["voyage_day"].transform("max")
    positions["_frac"] = positions["voyage_day"] / voyage_lengths.clip(lower=1)
    positions["active_search_flag"] = (
        positions["ground_id"].notna() &
        (positions["_frac"] > 0.05) &
        (positions["_frac"] < 0.90)
    ).astype(int)
    positions["transit_flag"] = (positions["ground_id"].isna()).astype(int)
    positions["homebound_flag"] = (positions["_frac"] >= 0.90).astype(int)

    # Season remaining (approximate: 180-day season)
    positions["season_remaining"] = np.clip(
        180 - positions["voyage_day"], 0, 180
    )

    # ── Merge voyage-level features ─────────────────────────────────
    merge_cols = ["voyage_id"]
    available_merge = []
    for c in ["captain_id", "agent_id", "vessel_id", "home_port",
              "theta", "psi", "theta_heldout", "psi_heldout",
              "tonnage", "rig", "crew_count",
              "switch_agent", "switch_vessel",
              "captain_experience", "captain_voyage_num",
              "novice"]:
        # Check actual column name from config
        actual = c
        if actual in voyages.columns:
            available_merge.append(actual)

    if available_merge:
        positions = positions.merge(
            voyages[["voyage_id"] + available_merge].drop_duplicates("voyage_id"),
            on="voyage_id",
            how="left",
        )

    # Rename holdout columns to standard names
    for old, new in [("theta_heldout", "theta_hat_holdout"),
                     ("psi_heldout", "psi_hat_holdout")]:
        if old in positions.columns and new not in positions.columns:
            positions.rename(columns={old: new}, inplace=True)

    # ── Patch assignment ────────────────────────────────────────────
    if "patch_id" not in positions.columns and len(patch_spells) > 0:
        # Vectorized patch assignment via merge_asof on voyage + date
        positions["patch_id"] = np.nan
        positions["days_in_patch"] = np.nan
        try:
            entry_col = "entry_date" if "entry_date" in patch_spells.columns else "start_date"
            exit_col = "exit_date" if "exit_date" in patch_spells.columns else "end_date"
            ps_id_col = "patch_spell_id" if "patch_spell_id" in patch_spells.columns else "patch_id"
            if entry_col in patch_spells.columns and exit_col in patch_spells.columns:
                ps = patch_spells[["voyage_id", entry_col, exit_col, ps_id_col]].copy()
                ps[entry_col] = pd.to_datetime(ps[entry_col])
                ps[exit_col] = pd.to_datetime(ps[exit_col])
                # Group positions and patches by voyage for fast subset matching
                pos_grouped = dict(list(positions.groupby("voyage_id")))
                for vid, vid_patches in ps.groupby("voyage_id"):
                    if vid not in pos_grouped:
                        continue
                    vid_pos = pos_grouped[vid]
                    vid_dates = vid_pos["date"]
                    for _, patch_row in vid_patches.iterrows():
                        date_mask = (vid_dates >= patch_row[entry_col]) & (vid_dates <= patch_row[exit_col])
                        positions.loc[vid_pos.index[date_mask], "patch_id"] = patch_row[ps_id_col]
            logger.info("Assigned patch_id to %d positions", positions["patch_id"].notna().sum())
        except Exception as e:
            logger.warning("Patch assignment failed: %s", e)

    # ── Action targets ──────────────────────────────────────────────
    # Next move length & turn angle
    positions["next_move_length"] = positions.groupby("voyage_id")["move_length"].shift(-1)
    positions["next_turn_angle"] = positions.groupby("voyage_id")["turn_angle"].shift(-1)

    # Exit patch next: did ground_id or patch_id change tomorrow?
    positions["_next_ground"] = positions.groupby("voyage_id")["ground_id"].shift(-1)
    positions["exit_patch_next"] = (
        positions["ground_id"] != positions["_next_ground"]
    ).astype(int)

    positions["switch_ground_next"] = (
        positions["ground_id"].notna() &
        positions["_next_ground"].notna() &
        (positions["ground_id"] != positions["_next_ground"])
    ).astype(int)

    # Next action class (simplified 5-way)
    positions["next_action_class"] = _classify_action(positions)

    # ── Scarcity proxy ──────────────────────────────────────────────
    if "scarcity" not in positions.columns:
        # Use ground-year encounter rate as scarcity proxy
        if enc_col and "ground_id" in positions.columns:
            enc_rates = positions.groupby(["ground_id", "year"])["_enc_flag"].mean()
            enc_rates.name = "scarcity"
            positions = positions.merge(
                enc_rates.reset_index(),
                on=["ground_id", "year"],
                how="left",
            )
        else:
            positions["scarcity"] = np.nan

    # ── Clean up temp columns ───────────────────────────────────────
    drop_cols = [c for c in positions.columns if c.startswith("_")]
    drop_cols += ["start_lat", "start_lon"]
    positions.drop(columns=drop_cols, inplace=True, errors="ignore")

    # ── Sanity checks ───────────────────────────────────────────────
    _sanity_check(positions)

    # ── Save ────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    logger.info(
        "Action dataset built: %d rows, %d columns, %.1fs",
        len(positions), len(positions.columns), elapsed,
    )

    if save:
        positions.to_parquet(OUTPUT_PATH, index=False)
        logger.info("Saved to %s", OUTPUT_PATH)

    return positions


def _classify_action(df: pd.DataFrame) -> pd.Series:
    """
    Classify next action into 5 categories.

    0: stay_in_patch (move < 10nm)
    1: move_local (10-50nm, same ground)
    2: move_long_within_ground (>50nm, same ground)
    3: exit_patch (ground changes or large move out)
    4: switch_ground (ground changes to different named ground)
    """
    next_ml = df.get("next_move_length", pd.Series(dtype=float))
    switch = df.get("switch_ground_next", pd.Series(0, index=df.index))

    action = pd.Series(np.nan, index=df.index, dtype="float64")

    # Switch ground
    action[switch == 1] = 4

    # Exit patch (but not switch ground): large move or patch exit
    exit_mask = (df.get("exit_patch_next", pd.Series(0, index=df.index)) == 1) & (switch != 1)
    action[exit_mask] = 3

    # Move categories for non-exit, non-switch
    remaining = action.isna() & next_ml.notna()
    action[remaining & (next_ml < 10)] = 0  # stay in patch
    action[remaining & (next_ml >= 10) & (next_ml < 50)] = 1  # move local
    action[remaining & (next_ml >= 50)] = 2  # move long within ground

    return action


def _sanity_check(df: pd.DataFrame) -> None:
    """Run basic sanity checks on the action dataset."""
    # No duplicate rows
    dup_cols = ["voyage_id", "obs_date"]
    available = [c for c in dup_cols if c in df.columns]
    if available:
        n_dup = df.duplicated(subset=available).sum()
        if n_dup > 0:
            logger.warning("Action dataset has %d duplicate voyage-date rows", n_dup)

    # Check coordinate ranges
    if "lat" in df.columns:
        bad_lat = (df["lat"].abs() > 90).sum()
        if bad_lat > 0:
            logger.warning("%d positions with |lat| > 90", bad_lat)

    if "lon" in df.columns:
        bad_lon = (df["lon"].abs() > 180).sum()
        if bad_lon > 0:
            logger.warning("%d positions with |lon| > 180", bad_lon)

    # Check impossible jumps (>500nm/day)
    if "speed" in df.columns:
        bad_speed = (df["speed"] > 500).sum()
        if bad_speed > 0:
            logger.warning("%d observations with speed > 500 nm/day", bad_speed)

    logger.info("Sanity checks passed for action dataset")
