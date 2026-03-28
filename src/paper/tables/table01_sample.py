from __future__ import annotations

import numpy as np
import pandas as pd

from .._table_common import save_table_outputs
from ..config import BuildContext
from ..data import (
    load_action_dataset,
    load_connected_sample,
    load_patch_sample,
    load_positions,
    load_survival_dataset,
    load_universe,
    years_label,
)
from ..utils.footnotes import standard_footnote


def _sample_flow_rows(
    universe: pd.DataFrame,
    connected: pd.DataFrame,
    positions: pd.DataFrame,
    patches: pd.DataFrame,
    survival: pd.DataFrame,
    action: pd.DataFrame,
) -> list[dict]:
    switcher = connected[connected.get("switch_agent", 0).fillna(0).astype(float) > 0].copy()
    archival = universe[
        universe.get("has_labor_data", False).fillna(False)
        | universe.get("has_vqi_data", False).fillna(False)
        | universe.get("has_logbook_data", False).fillna(False)
        | universe.get("logbook_source_count", 0).fillna(0).gt(0)
    ].copy()
    coordinate_sample = universe[universe.get("has_route_data", False).fillna(False)].copy()
    ground_sample = universe[universe["ground_or_route"].notna()].copy()

    samples = [
        ("Universe of voyages", universe, len(universe)),
        ("Connected set", connected, len(connected)),
        ("Daily-coordinate sample", coordinate_sample, len(positions)),
        ("Ground-labeled sample", ground_sample, len(ground_sample)),
        ("Patch sample", universe[universe["voyage_id"].isin(patches["voyage_id"].unique())], len(patches)),
        ("Patch-day sample", universe[universe["voyage_id"].isin(survival["voyage_id"].unique())], len(survival)),
        ("Encounter subsample", universe[universe["voyage_id"].isin(action["voyage_id"].unique())], len(action)),
        ("Switcher sample", switcher, len(switcher)),
        ("Archival-mechanism sample", archival, len(archival)),
    ]

    rows = []
    for label, sample, n_obs in samples:
        rows.append(
            {
                "panel": "Panel A",
                "row_label": label,
                "n_voyages": int(sample["voyage_id"].nunique()) if "voyage_id" in sample.columns else 0,
                "n_captains": int(sample["captain_id"].nunique()) if "captain_id" in sample.columns else 0,
                "n_agents": int(sample["agent_id"].nunique()) if "agent_id" in sample.columns else 0,
                "n_vessels": int(sample["vessel_id"].nunique()) if "vessel_id" in sample.columns else 0,
                "n_observations": int(n_obs),
                "years_covered": years_label(sample.get("year_out", pd.Series(dtype=float))),
            }
        )
    return rows


def _describe(series: pd.Series, base_n: int) -> dict[str, float]:
    clean = pd.to_numeric(series, errors="coerce")
    stats = clean.dropna()
    return {
        "mean": float(stats.mean()) if not stats.empty else np.nan,
        "sd": float(stats.std(ddof=1)) if len(stats) > 1 else np.nan,
        "p25": float(stats.quantile(0.25)) if not stats.empty else np.nan,
        "median": float(stats.median()) if not stats.empty else np.nan,
        "p75": float(stats.quantile(0.75)) if not stats.empty else np.nan,
        "missing_pct": float(clean.isna().mean() * 100) if base_n else np.nan,
    }


def _descriptive_rows(connected: pd.DataFrame, action: pd.DataFrame, patches: pd.DataFrame) -> list[dict]:
    action_agg = pd.DataFrame(columns=["voyage_id", "days_since_last_success", "consecutive_empty_days", "scarcity"])
    if not action.empty:
        action_agg = (
            action.groupby("voyage_id")
            .agg(
                days_since_last_success=("days_since_last_success", "mean"),
                consecutive_empty_days=("consecutive_empty_days", "mean"),
                scarcity=("scarcity", "mean"),
            )
            .reset_index()
        )

    patch_agg = pd.DataFrame(columns=["voyage_id", "patch_residence_time"])
    if not patches.empty:
        patch_agg = (
            patches.groupby("voyage_id")
            .agg(patch_residence_time=("duration_days", "mean"))
            .reset_index()
        )

    sample = connected.merge(action_agg, on="voyage_id", how="left", suffixes=("", "_action")).merge(patch_agg, on="voyage_id", how="left")
    if "scarcity_action" in sample.columns:
        sample["scarcity"] = sample["scarcity"].fillna(sample["scarcity_action"])
    sample["log_revenue_proxy"] = np.log(sample["q_total_index"].clip(lower=1))
    sample["log_tonnage"] = np.log(sample["tonnage"].clip(lower=1))

    variables = [
        ("Log output", sample["log_q"]),
        ("Log revenue proxy (q_total_index)", sample["log_revenue_proxy"]),
        ("Log tonnage", sample["log_tonnage"]),
        ("Crew size", sample["crew_count"]),
        ("Voyage duration", sample["duration_days"]),
        ("Captain experience", sample["captain_experience"]),
        ("Captain skill (theta_hat)", sample["theta"]),
        ("Agent capability (psi_hat)", sample["psi"]),
        ("Scarcity", sample["scarcity"]),
        ("Days since last success", sample["days_since_last_success"]),
        ("Consecutive empty days", sample["consecutive_empty_days"]),
        ("Patch residence time", sample["patch_residence_time"]),
        ("Number of grounds visited", sample["n_grounds_visited"]),
    ]

    rows = []
    for label, series in variables:
        rows.append({"panel": "Panel B", "row_label": label, **_describe(series, len(sample))})
    return rows


def build(context: BuildContext):
    universe = load_universe(context)
    connected = load_connected_sample(context)
    positions = load_positions(context)
    patches = load_patch_sample(context)
    survival = load_survival_dataset(context)
    action = load_action_dataset(context)

    frame = pd.DataFrame(
        _sample_flow_rows(universe, connected, positions, patches, survival, action)
        + _descriptive_rows(connected, action, patches)
    )

    memo = standard_footnote(
        sample="Panel A reports sample lineage across the full voyage universe; Panel B reports connected-set voyages with AKM types.",
        unit="Voyage in Panel B; row-specific sample unit in Panel A.",
        types_note="theta_hat and psi_hat use the repository's connected-set AKM estimates from outcome_ml_dataset.parquet.",
        fe="None for descriptive statistics.",
        cluster="None for descriptive statistics.",
        controls="No regression controls; state variables are voyage-level means across logged search days when only day-level inputs exist.",
        interpretation="The paper's usable identification sample is materially smaller than the raw universe because AKM, logbook, and patch-day coverage intersect only for a subset of voyages.",
        caution="Log revenue is proxied with log(q_total_index), and signal variables are voyage-level averages constructed from action logs rather than native voyage scalars.",
    )
    memo = (
        "# table01_sample\n\n"
        + memo
        + "\n\nProxy notes:\n"
        + "- `Log revenue` uses `q_total_index` because a direct revenue variable is not present in the final voyage panel.\n"
        + "- `Days since last success`, `Consecutive empty days`, and `Scarcity` are aggregated from the action dataset to the voyage level.\n"
        + "- `Patch residence time` is the within-voyage mean patch duration from `output/stopping_rule/patches.csv`.\n"
    )

    return save_table_outputs(
        name="table01_sample",
        frame=frame,
        out_dir=context.outputs / "tables",
        context=context,
        memo=memo,
        title="Table 1. Sample Construction and Descriptive Statistics",
    )
