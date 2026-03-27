"""
Repair 4: Leave-One-Out Ground Quality Controls.

Rebuilds quality/success controls so that no observation uses
contemporaneous realized output from the same voyage or captain-agent pair.

Creates:
  - lagged ground-year quality
  - leave-one-out ground-year quality
  - rolling historical quality
  - ground volatility
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

from src.next_round.config import DATA_FINAL, DATA_DERIVED

logger = logging.getLogger(__name__)


def build_ground_quality_loo(*, save: bool = True) -> pd.DataFrame:
    """
    Build leave-one-out ground quality controls.

    Returns DataFrame with voyage_id and LOO quality measures.
    """
    logger.info("=" * 60)
    logger.info("Repair 4: Leave-One-Out Ground Quality Controls")
    logger.info("=" * 60)

    df = pd.read_parquet(DATA_FINAL / "analysis_voyage_augmented.parquet")

    outcome_col = "q_total_index"
    if outcome_col not in df.columns:
        logger.warning("q_total_index not found, trying log transform of q_oil_bbl")
        df["q_total_index"] = np.log1p(df["q_oil_bbl"].fillna(0))
        outcome_col = "q_total_index"

    ground_col = "ground_or_route"
    year_col = "year_out"

    # Ensure we have the necessary columns
    for col in [ground_col, year_col, outcome_col, "voyage_id"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    logger.info("Building LOO quality from %d voyages", len(df))

    # ── 1. Leave-one-out ground-year quality ──────────────────────────
    gy = df.groupby([ground_col, year_col])
    ground_year_sum = gy[outcome_col].transform("sum")
    ground_year_n = gy[outcome_col].transform("count")

    # LOO: exclude self
    df["quality_loo_ground_year"] = np.where(
        ground_year_n > 1,
        (ground_year_sum - df[outcome_col]) / (ground_year_n - 1),
        np.nan,
    )

    # ── 2. Lagged ground-year quality (t-1) ───────────────────────────
    ground_year_avg = (
        df.groupby([ground_col, year_col])[outcome_col]
        .mean()
        .reset_index()
        .rename(columns={outcome_col: "quality_ground_year_avg"})
    )
    ground_year_avg["year_out_next"] = ground_year_avg[year_col] + 1

    df = df.merge(
        ground_year_avg[[ground_col, "year_out_next", "quality_ground_year_avg"]]
        .rename(columns={"year_out_next": year_col, "quality_ground_year_avg": "quality_lagged_1yr"}),
        on=[ground_col, year_col],
        how="left",
    )

    # ── 3. Rolling historical quality (all prior years) ───────────────
    # For each ground-year, compute mean of all strictly prior years
    ground_history = (
        df.groupby([ground_col, year_col])[outcome_col]
        .mean()
        .reset_index()
        .sort_values([ground_col, year_col])
    )

    rolling_quality = []
    for ground, gdf in ground_history.groupby(ground_col):
        gdf = gdf.sort_values(year_col)
        gdf["quality_rolling_hist"] = (
            gdf[outcome_col]
            .expanding(min_periods=1)
            .mean()
            .shift(1)  # Strictly prior
        )
        rolling_quality.append(gdf[[ground_col, year_col, "quality_rolling_hist"]])

    rolling_df = pd.concat(rolling_quality, ignore_index=True)
    df = df.merge(rolling_df, on=[ground_col, year_col], how="left")

    # ── 4. Ground volatility (SD of quality across years) ─────────────
    ground_vol = (
        df.groupby(ground_col)[outcome_col]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "quality_ground_mean",
                         "std": "quality_ground_vol",
                         "count": "quality_ground_n"})
        .reset_index()
    )
    df = df.merge(ground_vol, on=ground_col, how="left")

    # ── 5. Leave-one-out captain-agent quality ────────────────────────
    if "captain_id" in df.columns and "agent_id" in df.columns:
        ca = df.groupby(["captain_id", "agent_id"])
        ca_sum = ca[outcome_col].transform("sum")
        ca_n = ca[outcome_col].transform("count")
        df["quality_loo_captain_agent"] = np.where(
            ca_n > 1,
            (ca_sum - df[outcome_col]) / (ca_n - 1),
            np.nan,
        )
    else:
        df["quality_loo_captain_agent"] = np.nan

    # ── Select output columns ─────────────────────────────────────────
    quality_cols = [
        "voyage_id", ground_col, year_col,
        "quality_loo_ground_year",
        "quality_lagged_1yr",
        "quality_rolling_hist",
        "quality_ground_mean",
        "quality_ground_vol",
        "quality_ground_n",
        "quality_loo_captain_agent",
    ]
    out = df[[c for c in quality_cols if c in df.columns]].copy()

    # Summary
    logger.info("LOO quality coverage:")
    for col in out.columns:
        if col.startswith("quality_"):
            logger.info("  %s: %.1f%% non-null", col, out[col].notna().mean() * 100)

    if save:
        out_path = DATA_DERIVED / "ground_quality_loo.parquet"
        out.to_parquet(out_path, index=False)
        logger.info("Saved %s (%d rows)", out_path, len(out))

    return out
