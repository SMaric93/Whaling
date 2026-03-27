"""
ML Layer — Outcome ML Dataset Builder.

Unit: voyage (or voyage-ground spell).

Reuses:
- data_builder.build_analysis_panel() for the full voyage panel
- type_estimation for held-out theta/psi
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from src.ml.config import ML_CFG, ML_DATA_DIR, DATA_DIR

logger = logging.getLogger(__name__)

OUTPUT_PATH = ML_DATA_DIR / "outcome_ml_dataset.parquet"


def build_outcome_ml_dataset(
    *,
    force_rebuild: bool = False,
    save: bool = True,
) -> pd.DataFrame:
    """
    Build the voyage-level outcome dataset for production surface and
    heterogeneity models.

    Returns
    -------
    pd.DataFrame
        One row per voyage with output, type estimates, controls, and
        switch indicators.
    """
    if OUTPUT_PATH.exists() and not force_rebuild:
        logger.info("Loading cached outcome ML dataset from %s", OUTPUT_PATH)
        return pd.read_parquet(OUTPUT_PATH)

    t0 = time.time()
    logger.info("Building outcome ML dataset...")

    from src.reinforcement.data_builder import build_analysis_panel

    df = build_analysis_panel(require_akm=True, require_logbook=False)

    # ── Standardize column names ────────────────────────────────────
    for old, new in [("theta_heldout", "theta_hat_holdout"),
                     ("psi_heldout", "psi_hat_holdout")]:
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)

    # ── Derived outcome variables ───────────────────────────────────
    if "log_q" not in df.columns and "q_total_index" in df.columns:
        df["log_q"] = np.log(df["q_total_index"].clip(lower=1))
    elif "log_q" not in df.columns and "q_oil_bbl" in df.columns:
        df["log_q"] = np.log(df["q_oil_bbl"].clip(lower=1))

    # Novice / expert indicators
    if "novice" not in df.columns and "captain_voyage_num" in df.columns:
        df["novice"] = (df["captain_voyage_num"] <= ML_CFG.experience_bins["novice_max"]).astype(int)
    if "expert" not in df.columns and "captain_voyage_num" in df.columns:
        df["expert"] = (df["captain_voyage_num"] >= ML_CFG.experience_bins["expert_min"]).astype(int)

    # Bottom-decile indicators for downside risk
    if "log_q" in df.columns:
        q10 = df["log_q"].quantile(0.10)
        q05 = df["log_q"].quantile(0.05)
        df["bottom_decile"] = (df["log_q"] <= q10).astype(int)
        df["bottom_5pct"] = (df["log_q"] <= q05).astype(int)

    # Scarcity proxy at voyage level
    if "scarcity" not in df.columns:
        if "ground_or_route" in df.columns and "year_out" in df.columns:
            gy_mean = df.groupby(["ground_or_route", "year_out"])["log_q"].transform("mean")
            df["scarcity"] = -gy_mean  # Lower average output = more scarce
        else:
            df["scarcity"] = np.nan

    # ── Select and order columns ────────────────────────────────────
    desired_cols = [
        # Identifiers
        "voyage_id", "captain_id", "agent_id", "vessel_id",
        "year_out", "year_in", "home_port", "ground_or_route",
        # Outcomes
        "log_q", "q_total_index", "q_oil_bbl",
        "bottom_decile", "bottom_5pct",
        # Type estimates
        "theta", "psi", "theta_hat_holdout", "psi_hat_holdout",
        # Experience
        "captain_experience", "captain_voyage_num", "novice", "expert",
        # Scarcity
        "scarcity",
        # Controls
        "tonnage", "rig", "crew_count",
        # Switch indicators
        "switch_agent", "switch_vessel",
        # Time
        "decade",
    ]

    # Keep columns that actually exist
    available = [c for c in desired_cols if c in df.columns]
    # Add any remaining columns not in the desired list
    extra = [c for c in df.columns if c not in desired_cols]
    result = df[available + extra].copy()

    # ── Sanity checks ───────────────────────────────────────────────
    n_voy = result["voyage_id"].nunique() if "voyage_id" in result.columns else len(result)
    logger.info("Outcome ML dataset: %d voyages", n_voy)

    if "theta_hat_holdout" in result.columns:
        n_missing_theta = result["theta_hat_holdout"].isna().sum()
        if n_missing_theta > 0:
            logger.warning("%d voyages missing theta_hat_holdout", n_missing_theta)

    elapsed = time.time() - t0
    logger.info(
        "Outcome ML dataset built: %d rows, %d columns, %.1fs",
        len(result), len(result.columns), elapsed,
    )

    if save:
        result.to_parquet(OUTPUT_PATH, index=False)
        logger.info("Saved to %s", OUTPUT_PATH)

    return result
