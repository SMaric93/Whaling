"""
Compass Pipeline — Econometric Exports (Step 10).

Merges voyage-level compass outputs with voyage metadata and produces
panel-ready datasets for causal/movers designs.

Outputs:
    * ``panel_voyage_compass.parquet`` — voyage-level
    * ``panel_captain_year_compass.parquet`` — captain × year aggregated
    * ``diagnostics_compass.parquet`` — quality / diagnostic columns
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import CompassConfig

logger = logging.getLogger(__name__)


# ── merge with metadata ────────────────────────────────────────────────────

def merge_with_metadata(
    compass_df: pd.DataFrame,
    meta_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join compass features onto voyage metadata.

    Expects both to have ``voyage_id`` column.
    """
    out = compass_df.merge(meta_df, on="voyage_id", how="left")
    logger.info(
        "Merged compass (%d) with metadata (%d) → %d rows.",
        len(compass_df), len(meta_df), len(out),
    )
    return out


# ── aggregation ─────────────────────────────────────────────────────────────

def _weighted_compass_means(
    df: pd.DataFrame,
    index_cols: list[str],
    compass_cols: list[str],
    weight_col: str,
) -> pd.DataFrame:
    """Compute weighted means for compass columns without per-column groupby.apply."""
    weights = pd.to_numeric(df[weight_col], errors="coerce")
    valid_weight = weights.where(np.isfinite(weights) & (weights > 0))
    compass = df[compass_cols].apply(pd.to_numeric, errors="coerce")
    valid_values = compass.where(np.isfinite(compass))

    weighted_values = valid_values.mul(valid_weight, axis=0)
    weight_contrib = valid_values.notna().mul(valid_weight, axis=0)

    group_keys = [df[col] for col in index_cols]
    weighted_sum = weighted_values.groupby(group_keys, sort=True).sum(min_count=1)
    weight_sum = weight_contrib.groupby(group_keys, sort=True).sum(min_count=1)

    return weighted_sum.div(weight_sum).add_suffix("_wtd").reset_index()

def aggregate_captain_year(
    df: pd.DataFrame,
    index_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Aggregate voyage-level compass to captain × year, weighted by
    ``n_search_steps``.

    Parameters
    ----------
    df : pd.DataFrame
        Voyage-level panel (must contain ``captain_id``, ``year``).
    index_cols : list[str], optional
        Override grouping columns (default: ``["captain_id", "year"]``).
    """
    if index_cols is None:
        index_cols = ["captain_id", "year"]

    for c in index_cols:
        if c not in df.columns:
            logger.warning("Column '%s' not in df — cannot aggregate.", c)
            return pd.DataFrame()

    numeric = df.select_dtypes(include="number").columns.tolist()
    weight_col = "n_search_steps"

    agg: dict = {}
    for col in numeric:
        if col in index_cols:
            continue
        agg[col] = "mean"

    out = df.groupby(index_cols, as_index=False).agg(agg)

    # additionally, compute weighted means if weights are present
    compass_cols = [c for c in numeric if c.startswith("Compass")]
    if weight_col in df.columns and compass_cols:
        weighted = _weighted_compass_means(df, index_cols, compass_cols, weight_col)
        out = out.merge(weighted, on=index_cols, how="left")

    logger.info(
        "Aggregated to captain×year: %d rows.", len(out),
    )
    return out


# ── diagnostics ─────────────────────────────────────────────────────────────

def build_diagnostics(
    steps_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build per-voyage diagnostic columns:
    ``n_steps_total, n_steps_search, missing_gap_share, regime_confidence``.
    """
    if steps_df.empty:
        return pd.DataFrame(
            columns=[
                "voyage_id",
                "n_steps_total",
                "n_steps_search",
                "missing_gap_share",
                "regime_confidence",
            ]
        )

    work = pd.DataFrame({"voyage_id": steps_df["voyage_id"], "n_steps_total": 1})
    work["n_steps_search"] = (
        steps_df["regime_label"].eq("search").astype("int64")
        if "regime_label" in steps_df.columns
        else 0
    )
    work["missing_gap_share"] = steps_df["gap_flag"] if "gap_flag" in steps_df.columns else 0.0
    work["regime_confidence"] = steps_df["p_search"] if "p_search" in steps_df.columns else np.nan

    return (
        work.groupby("voyage_id", sort=False, as_index=False)
        .agg(
            n_steps_total=("n_steps_total", "sum"),
            n_steps_search=("n_steps_search", "sum"),
            missing_gap_share=("missing_gap_share", "mean"),
            regime_confidence=("regime_confidence", "mean"),
        )
    )


# ── export orchestrator ─────────────────────────────────────────────────────

def export_panels(
    compass_df: pd.DataFrame,
    steps_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    cfg: CompassConfig,
    output_dir: Optional[Path] = None,
) -> dict[str, pd.DataFrame]:
    """
    Produce and save all panel-ready outputs.

    Returns dict of DataFrames for testing / downstream use.
    """
    od = Path(output_dir) if output_dir else cfg.output_path
    od.mkdir(parents=True, exist_ok=True)

    # 1. voyage panel
    panel_voyage = merge_with_metadata(compass_df, meta_df)
    diag = build_diagnostics(steps_df)
    panel_voyage = panel_voyage.merge(diag, on="voyage_id", how="left")
    panel_voyage.to_parquet(od / "panel_voyage_compass.parquet", index=False)
    logger.info("Saved panel_voyage_compass.parquet (%d rows).", len(panel_voyage))

    # 2. captain × year
    panel_cy = pd.DataFrame()
    if "captain_id" in panel_voyage.columns and "year" in panel_voyage.columns:
        panel_cy = aggregate_captain_year(panel_voyage)
        panel_cy.to_parquet(od / "panel_captain_year_compass.parquet", index=False)
        logger.info(
            "Saved panel_captain_year_compass.parquet (%d rows).", len(panel_cy),
        )

    # 3. diagnostics
    diag.to_parquet(od / "diagnostics_compass.parquet", index=False)
    logger.info("Saved diagnostics_compass.parquet (%d rows).", len(diag))

    return {
        "panel_voyage": panel_voyage,
        "panel_captain_year": panel_cy,
        "diagnostics": diag,
    }
