"""Build a weekly observation panel from canonical event rows.

Transforms irregular, row-level WSL event data into a regular weekly grid
per voyage.  This is mandated by the build spec's non-negotiable rule:
"Do not fit the latent state model directly on raw extracted rows."

**Performance note**: Uses fully vectorized pandas aggregation (no Python
loops over groups) so it scales to 500K+ events without bottleneck.

Usage::

    from .build_weekly_panel import build_weekly_observation_panel
    weekly_df = build_weekly_observation_panel(events_df, linkage_df, voyage_ref_df, config)
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd

from .state_space import STATE_NAMES
from .utils import (
    WSLReliabilityConfig,
    attach_voyage_linkage,
    PerfTracer,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Week index computation
# ---------------------------------------------------------------------------


def compute_report_week(
    event_date: pd.Series,
    issue_date: pd.Series,
    departure_date: pd.Series,
) -> pd.Series:
    """Compute ``week_idx = floor((report_date_effective - departure_date) / 7)``.

    ``report_date_effective`` = ``event_date`` if available, else ``issue_date``.
    """
    report_date = event_date.combine_first(issue_date)
    delta_days = (report_date - departure_date).dt.days
    week_idx = (delta_days // 7).astype("Int64")
    # Clamp negatives to 0 (pre-departure mentions), keep NaN
    week_idx = week_idx.clip(lower=0)
    return week_idx


# ---------------------------------------------------------------------------
# Vectorized aggregation helpers
# ---------------------------------------------------------------------------


def _vectorized_event_type_counts(events_df: pd.DataFrame) -> pd.Series:
    """Return a Series of JSON-encoded event type count dicts, indexed by (voyage_id, week_idx)."""
    if "event_type" not in events_df.columns:
        return pd.Series({}, dtype=object)
    dummies = pd.get_dummies(events_df["event_type"].fillna("missing"), dtype=int)
    dummies["voyage_id"] = events_df["voyage_id"].values
    dummies["week_idx"] = events_df["week_idx"].values
    grouped = dummies.groupby(["voyage_id", "week_idx"], sort=True).sum()
    count_dicts = grouped.apply(lambda row: json.dumps(row[row > 0].to_dict()), axis=1)
    return count_dicts


def _vectorized_source_mode(events_df: pd.DataFrame) -> pd.DataFrame:
    """Compute source_mode stats vectorized."""
    df = events_df.copy()
    df["is_flow"] = (df["page_type"] == "weekly_event_flow").astype(int)
    df["is_snapshot"] = (df["page_type"] == "fleet_registry_stock").astype(int)
    df["is_total"] = 1

    g = df.groupby(["voyage_id", "week_idx"], sort=True)
    flow_sum = g["is_flow"].sum()
    snap_sum = g["is_snapshot"].sum()
    total_count = g["is_total"].sum()
    frac = (flow_sum / total_count.clip(lower=1)).rename("source_mode_flow_fraction")
    return pd.DataFrame({
        "source_mode_flow_count": flow_sum,
        "source_mode_snapshot_count": snap_sum,
        "source_mode_flow_fraction": frac,
        "n_events": total_count,
    })


def _vectorized_numeric(events_df: pd.DataFrame) -> pd.DataFrame:
    """Compute max and missing flags for numeric cargo columns, vectorized."""
    g = events_df.groupby(["voyage_id", "week_idx"], sort=True)
    results = {}
    for col, result_col in [
        ("oil_sperm_bbls", "oil_sperm_bbls"),
        ("oil_whale_bbls", "oil_whale_bbls"),
        ("bone_lbs", "bone_lbs"),
        ("days_out", "days_out"),
    ]:
        if col in events_df.columns:
            series = pd.to_numeric(events_df[col], errors="coerce")
            results[result_col] = g[col].apply(lambda x: pd.to_numeric(x, errors="coerce").max())
        else:
            results[result_col] = pd.Series(np.nan, dtype=float)

    df = pd.DataFrame(results)

    # Compute oil_total = max(sperm + whale) per group
    if "oil_sperm_bbls" in df.columns and "oil_whale_bbls" in df.columns:
        df["oil_total"] = df["oil_sperm_bbls"].fillna(0) + df["oil_whale_bbls"].fillna(0)
        # If both were NaN, oil_total should be NaN
        both_nan = df["oil_sperm_bbls"].isna() & df["oil_whale_bbls"].isna()
        df.loc[both_nan, "oil_total"] = np.nan
    else:
        df["oil_total"] = np.nan

    df["oil_total_missing"] = df["oil_total"].isna().astype(int)
    df["days_out_missing"] = df["days_out"].isna().astype(int)
    return df


def _vectorized_quality(events_df: pd.DataFrame) -> pd.DataFrame:
    """Compute quality features vectorized."""
    g = events_df.groupby(["voyage_id", "week_idx"], sort=True)
    result = {}
    if "row_weight" in events_df.columns:
        result["quality_weight"] = g["row_weight"].mean()
    if "_confidence" in events_df.columns:
        conf = pd.to_numeric(events_df["_confidence"], errors="coerce")
        conf_g = g["_confidence"].agg(lambda x: pd.to_numeric(x, errors="coerce").mean())
        result["mean_confidence"] = conf_g
        conf_min = g["_confidence"].agg(lambda x: pd.to_numeric(x, errors="coerce").min())
        result["min_confidence"] = conf_min
    if "linkage_confidence" in events_df.columns:
        result["link_probability_mean"] = g["linkage_confidence"].mean()
    return pd.DataFrame(result)


def _vectorized_remarks(events_df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized mean-aggregation of remarks probability columns."""
    prob_cols = [c for c in events_df.columns if c.startswith("p_primary__") or c.startswith("p_tag__")]
    if not prob_cols:
        return pd.DataFrame()
    for col in prob_cols:
        events_df[col] = pd.to_numeric(events_df[col], errors="coerce")
    g = events_df.groupby(["voyage_id", "week_idx"], sort=True)
    return g[prob_cols].mean()


def _vectorized_severity(events_df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized distress_severity_max and productivity_polarity_mean."""
    result = {}
    g = events_df.groupby(["voyage_id", "week_idx"], sort=True)
    if "distress_severity_0_4" in events_df.columns:
        result["distress_severity_max"] = g["distress_severity_0_4"].agg(
            lambda x: pd.to_numeric(x, errors="coerce").max()
        )
    if "productivity_polarity_m2_p2" in events_df.columns:
        result["productivity_polarity_mean"] = g["productivity_polarity_m2_p2"].agg(
            lambda x: pd.to_numeric(x, errors="coerce").mean()
        )
    return pd.DataFrame(result).fillna(0.0)


def _vectorized_mode_fields(events_df: pd.DataFrame) -> pd.DataFrame:
    """Compute mode (most frequent value) for categorical fields, vectorized."""
    result = {}
    g = events_df.groupby(["voyage_id", "week_idx"], sort=True)

    def safe_mode(x: pd.Series) -> Any:
        m = x.dropna().mode()
        return m.iloc[0] if not m.empty else None

    for col, out_name in [
        ("primary_class", "primary_class_mode"),
        ("home_port_norm", "home_port_mode"),
        ("port_norm", "report_port_mode"),
        ("destination_basin", "basin_mode"),
    ]:
        if col in events_df.columns:
            result[out_name] = g[col].agg(safe_mode)

    return pd.DataFrame(result)


# ---------------------------------------------------------------------------
# Main panel builder
# ---------------------------------------------------------------------------


def build_weekly_observation_panel(
    events_df: pd.DataFrame,
    linkage_df: pd.DataFrame,
    voyage_ref_df: pd.DataFrame,
    config: WSLReliabilityConfig,
    *,
    tracer: PerfTracer | None = None,
) -> pd.DataFrame:
    """Build a weekly observation panel: one row per ``voyage_id × week_idx``.

    Fully vectorized — no Python-level loops over groups.

    Parameters
    ----------
    events_df : pd.DataFrame
        Flattened canonical event rows (from ``load_wsl_cleaned_events``).
    linkage_df : pd.DataFrame
        Voyage linkage table (from ``build_voyage_linkage``).
    voyage_ref_df : pd.DataFrame
        Voyage master reference.
    config : WSLReliabilityConfig
        Pipeline configuration.
    tracer : PerfTracer, optional
        Performance tracer.

    Returns
    -------
    pd.DataFrame
        Weekly panel with mixed-type features and quality metadata.
    """
    # --- 1. Merge linkage and filter to linked voyages ---
    merged = attach_voyage_linkage(events_df, linkage_df, config)
    merged = merged[~merged["drop_hard"]].copy()
    merged = merged[merged["voyage_id"].notna()].copy()
    merged["issue_date"] = pd.to_datetime(merged["issue_date"], errors="coerce")
    merged = merged[merged["issue_date"].notna()].copy()

    logger.info(
        "[weekly_panel] Starting with %d linked event rows across %d voyages",
        len(merged),
        merged["voyage_id"].nunique(),
    )

    # --- 2. Get departure dates from voyage reference ---
    departure_info = voyage_ref_df[["voyage_id", "date_out"]].dropna(subset=["voyage_id", "date_out"]).copy()
    departure_info = departure_info.drop_duplicates("voyage_id")
    departure_info["departure_date"] = pd.to_datetime(departure_info["date_out"], errors="coerce")
    merged = merged.merge(departure_info[["voyage_id", "departure_date"]], on="voyage_id", how="left")

    # Fallback: use earliest dep-event issue_date within each voyage
    dep_dates = (
        merged[merged["event_type"] == "dep"]
        .groupby("voyage_id")["issue_date"]
        .min()
        .rename("departure_date_fallback")
    )
    merged = merged.merge(dep_dates, on="voyage_id", how="left")
    merged["departure_date"] = merged["departure_date"].combine_first(merged["departure_date_fallback"])
    merged = merged.drop(columns=["departure_date_fallback"], errors="ignore")

    has_departure = merged["departure_date"].notna()
    n_dropped = (~has_departure).sum()
    if n_dropped > 0:
        logger.warning("[weekly_panel] Dropping %d rows with no departure date", n_dropped)
    merged = merged[has_departure].copy()

    # --- 3. Compute week index (vectorized) ---
    event_date = pd.to_datetime(
        merged.get("parsed_event_date_if_available"), errors="coerce"
    )
    merged["week_idx"] = compute_report_week(
        event_date,
        merged["issue_date"],
        merged["departure_date"],
    )
    merged = merged[merged["week_idx"].notna()].copy()
    merged["week_idx"] = merged["week_idx"].astype(int)

    logger.info(
        "[weekly_panel] %d events mapped to week indices, %d voyages",
        len(merged),
        merged["voyage_id"].nunique(),
    )

    # --- 4. Vectorized aggregations ---
    with tracer.span("panel_event_counts") if tracer else _nullctx():
        event_counts_s = _vectorized_event_type_counts(merged)

    with tracer.span("panel_source_mode") if tracer else _nullctx():
        source_df = _vectorized_source_mode(merged)

    with tracer.span("panel_numeric") if tracer else _nullctx():
        numeric_df = _vectorized_numeric(merged)

    with tracer.span("panel_quality") if tracer else _nullctx():
        quality_df = _vectorized_quality(merged)

    with tracer.span("panel_remarks") if tracer else _nullctx():
        remarks_df = _vectorized_remarks(merged)

    with tracer.span("panel_severity") if tracer else _nullctx():
        severity_df = _vectorized_severity(merged)

    with tracer.span("panel_modes") if tracer else _nullctx():
        mode_df = _vectorized_mode_fields(merged)

    # Date range per group
    g = merged.groupby(["voyage_id", "week_idx"], sort=True)
    date_min = g["issue_date"].min().rename("issue_date_min")
    date_max = g["issue_date"].max().rename("issue_date_max")
    dep_date = g["departure_date"].first().rename("departure_date")

    # --- 5. Assemble panel ---
    base = pd.DataFrame({
        "issue_date_min": date_min,
        "issue_date_max": date_max,
        "departure_date": dep_date,
    })

    base["event_type_counts_json"] = event_counts_s.reindex(base.index)

    for df in [source_df, numeric_df, quality_df, severity_df, mode_df]:
        if not df.empty:
            for col in df.columns:
                base[col] = df[col].reindex(base.index)

    if not remarks_df.empty:
        for col in remarks_df.columns:
            base[col] = remarks_df[col].reindex(base.index)

    base = base.reset_index()

    # weeks_since_departure = week_idx
    base["weeks_since_departure"] = base["week_idx"].astype(int)
    base["missing_fraction"] = (
        base.get("oil_total_missing", 0).fillna(0)
        + base.get("days_out_missing", 0).fillna(0)
    ) / 2.0

    # --- 6. Oil deltas (per-voyage difference, vectorized) ---
    if "oil_total" in base.columns:
        base = base.sort_values(["voyage_id", "week_idx"])
        base["oil_delta"] = base.groupby("voyage_id")["oil_total"].diff().fillna(0.0)
        week_diff = base.groupby("voyage_id")["week_idx"].diff().fillna(1).clip(lower=1)
        base["oil_delta_per_elapsed_week"] = (base["oil_delta"] / week_diff).fillna(0.0)
    else:
        base["oil_delta"] = 0.0
        base["oil_delta_per_elapsed_week"] = 0.0

    base["weeks_since_last_observation"] = (
        base.groupby("voyage_id")["week_idx"].diff().fillna(1).astype(int)
    )

    logger.info(
        "[weekly_panel] Built panel: %d rows, %d voyages, weeks range [%s, %s]",
        len(base),
        base["voyage_id"].nunique(),
        base["week_idx"].min() if not base.empty else "?",
        base["week_idx"].max() if not base.empty else "?",
    )

    return base.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Null context manager for optional span usage
# ---------------------------------------------------------------------------

from contextlib import contextmanager


@contextmanager
def _nullctx():
    yield


def export_weekly_panel(panel_df: pd.DataFrame, output_path: Any) -> None:
    """Save the weekly panel to parquet."""
    from pathlib import Path

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    panel_df.to_parquet(path, index=False)
    logger.info("[weekly_panel] Exported to %s (%d rows)", path, len(panel_df))
