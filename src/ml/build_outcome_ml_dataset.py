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

    # Use require_akm=False to keep all voyages; theta/psi availability
    # is handled below via cross-fitting.  The fixed merge_akm_effects()
    # now loads from output/tables/r1_*_effects.csv.
    df = build_analysis_panel(require_akm=False, require_logbook=False)

    # ── Cross-fitted type estimation ────────────────────────────────
    # Produce held-out theta/psi via time-split cross-fitting to avoid
    # leaking in-sample information into downstream ML models.
    has_theta = "theta" in df.columns and df["theta"].notna().sum() > 50
    if has_theta and "theta_heldout" not in df.columns:
        try:
            from src.reinforcement.type_estimation import run_type_estimation
            logger.info("Running cross-fitted type estimation (time_split)...")
            df = run_type_estimation(df, method="time_split")
        except Exception as e:
            logger.warning(
                "Cross-fitted type estimation failed (%s); "
                "theta_hat_holdout will be NaN.", e,
            )
    elif not has_theta:
        logger.warning(
            "In-sample theta not available (merged %d non-null). "
            "Cross-fitting cannot run — theta_hat_holdout will be NaN.",
            df["theta"].notna().sum() if "theta" in df.columns else 0,
        )

    # ── Standardize column names ────────────────────────────────────
    # Rename cross-fitted columns to the ML-standard names.
    # NOTE: We intentionally do NOT fall back to in-sample theta/psi.
    # If cross-fitting failed, theta_hat_holdout stays NaN so the
    # problem is visible rather than silently using leaked estimates.
    for old, new in [("theta_heldout", "theta_hat_holdout"),
                     ("psi_heldout", "psi_hat_holdout")]:
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    # Sanity: warn if holdout is identical to in-sample (regression guard)
    if (
        "theta_hat_holdout" in df.columns
        and "theta" in df.columns
        and df["theta_hat_holdout"].notna().sum() > 100
    ):
        both = df[["theta", "theta_hat_holdout"]].dropna()
        if len(both) > 100:
            corr = both.corr().iloc[0, 1]
            if abs(corr - 1.0) < 1e-10:
                logger.warning(
                    "theta_hat_holdout is identical to in-sample theta "
                    "(r=%.6f). Cross-fitting may not have run correctly.",
                    corr,
                )

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

    # ── Median imputation for continuous features ───────────────────
    for impute_col in ["tonnage", "scarcity"]:
        if impute_col in df.columns:
            col_median = df[impute_col].median()
            n_imp = df[impute_col].isna().sum()
            if n_imp > 0 and pd.notna(col_median):
                df[impute_col] = df[impute_col].fillna(col_median)
                logger.info(
                    "Imputed %d missing %s values with median (%.3f)",
                    n_imp, impute_col, col_median,
                )

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
