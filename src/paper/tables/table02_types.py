from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from src.reinforcement.type_estimation import cross_fit_time_split

from .._table_common import save_table_outputs
from ..config import BuildContext
from ..data import (
    infer_basin,
    load_akm_variance_decomposition,
    load_connected_sample,
    load_split_sample_stability,
)
from ..utils.footnotes import standard_footnote


def _fisher_ci(corr: float, n: int) -> str:
    if not np.isfinite(corr) or n <= 3 or abs(corr) >= 1:
        return ""
    z = np.arctanh(corr)
    se = 1.0 / math.sqrt(n - 3)
    lo, hi = np.tanh([z - 1.96 * se, z + 1.96 * se])
    return f"[{lo:.3f}, {hi:.3f}]"


def _panel_a(variance_df: pd.DataFrame) -> list[dict]:
    component_map = {
        "Var(θ) - Captain": "Captain variance",
        "Var(ψ) - Agent": "Agent variance",
        "2×Cov(θ,ψ) - Sorting": "Sorting covariance",
        "Var(ε) - Residual": "Residual variance",
    }
    rows = []
    for _, row in variance_df.iterrows():
        if row["Component"] not in component_map:
            continue
        rows.append(
            {
                "panel": "Panel A",
                "row_label": component_map[row["Component"]],
                "model": row["Type"],
                "estimate": row["Variance"],
                "share_of_total_variance": row["Share"],
                "se_or_ci": "",
            }
        )
    return rows


def _panel_b(connected: pd.DataFrame, stability_df: pd.DataFrame) -> list[dict]:
    base = connected[
        ["voyage_id", "captain_id", "agent_id", "year_out", "log_q", "tonnage", "theta", "psi"]
    ].dropna(subset=["voyage_id", "captain_id", "agent_id", "year_out", "log_q", "tonnage", "theta", "psi"]).copy()
    base["year_out"] = pd.to_numeric(base["year_out"], errors="coerce")
    base = base.dropna(subset=["year_out"])
    heldout = cross_fit_time_split(base, outcome_col="log_q", controls=["tonnage"], year_col="year_out")

    captain_corr = stability_df.loc[
        (stability_df["entity_type"] == "captain") & (stability_df["n_bin"] == "all"),
        "split_corr",
    ]
    agent_corr = stability_df.loc[
        (stability_df["entity_type"] == "agent") & (stability_df["n_bin"] == "all"),
        "split_corr",
    ]

    captain_compare = (
        heldout.dropna(subset=["theta_heldout"])
        .groupby("captain_id")[["theta", "theta_heldout"]]
        .mean()
    )
    agent_compare = (
        heldout.dropna(subset=["psi_heldout"])
        .groupby("agent_id")[["psi", "psi_heldout"]]
        .mean()
    )

    theta_holdout_corr = float(captain_compare["theta"].corr(captain_compare["theta_heldout"])) if len(captain_compare) > 1 else np.nan
    psi_holdout_corr = float(agent_compare["psi"].corr(agent_compare["psi_heldout"])) if len(agent_compare) > 1 else np.nan

    rows = [
        {
            "panel": "Panel B",
            "row_label": "theta split-half correlation",
            "estimate": float(captain_corr.iloc[0]) if not captain_corr.empty else np.nan,
            "n": int(stability_df.loc[(stability_df["entity_type"] == "captain") & (stability_df["n_bin"] == "all"), "n_entities"].iloc[0])
            if not captain_corr.empty
            else 0,
            "ci": "",
        },
        {
            "panel": "Panel B",
            "row_label": "theta in-sample vs held-out",
            "estimate": theta_holdout_corr,
            "n": int(len(captain_compare)),
            "ci": _fisher_ci(theta_holdout_corr, len(captain_compare)),
        },
        {
            "panel": "Panel B",
            "row_label": "psi split-half correlation",
            "estimate": float(agent_corr.iloc[0]) if not agent_corr.empty else np.nan,
            "n": int(stability_df.loc[(stability_df["entity_type"] == "agent") & (stability_df["n_bin"] == "all"), "n_entities"].iloc[0])
            if not agent_corr.empty
            else 0,
            "ci": "",
        },
        {
            "panel": "Panel B",
            "row_label": "psi in-sample vs held-out",
            "estimate": psi_holdout_corr,
            "n": int(len(agent_compare)),
            "ci": _fisher_ci(psi_holdout_corr, len(agent_compare)),
        },
    ]
    return rows


def _corr_rows(df: pd.DataFrame, group_type: str, group_label: str) -> list[dict]:
    sample = df.dropna(subset=["theta", "psi"])
    if len(sample) < 4:
        return []
    pearson = float(sample["theta"].corr(sample["psi"]))
    spearman = float(stats.spearmanr(sample["theta"], sample["psi"], nan_policy="omit").statistic)
    return [
        {
            "panel": "Panel C",
            "group_type": group_type,
            "group_label": group_label,
            "metric": "Pearson",
            "estimate": pearson,
            "n": int(len(sample)),
            "ci": _fisher_ci(pearson, len(sample)),
        },
        {
            "panel": "Panel C",
            "group_type": group_type,
            "group_label": group_label,
            "metric": "Spearman",
            "estimate": spearman,
            "n": int(len(sample)),
            "ci": _fisher_ci(spearman, len(sample)),
        },
    ]


def _panel_c(connected: pd.DataFrame) -> list[dict]:
    df = connected.copy()
    df["era"] = pd.cut(
        pd.to_numeric(df["year_out"], errors="coerce"),
        bins=[-np.inf, 1829, 1869, np.inf],
        labels=["Pre-1830", "1830-1869", "1870+"],
    )
    scarcity_sample = df.dropna(subset=["scarcity"]).copy()
    if not scarcity_sample.empty:
        scarcity_sample["scarcity_tercile"] = pd.qcut(
            scarcity_sample["scarcity"].rank(method="first"),
            3,
            labels=["Tercile 1", "Tercile 2", "Tercile 3"],
        )
    df["basin"] = infer_basin(df["ground_or_route"])

    rows = []
    rows.extend(_corr_rows(df, "overall", "All voyages"))
    for era, subset in df.groupby("era", observed=True):
        rows.extend(_corr_rows(subset, "era", str(era)))
    if not scarcity_sample.empty:
        for tercile, subset in scarcity_sample.groupby("scarcity_tercile", observed=True):
            rows.extend(_corr_rows(subset, "scarcity_tercile", str(tercile)))
    basin_counts = df["basin"].value_counts(dropna=True)
    for basin in basin_counts[basin_counts >= 100].index.tolist():
        rows.extend(_corr_rows(df[df["basin"] == basin], "basin", basin))
    return rows


def build(context: BuildContext):
    connected = load_connected_sample(context)
    variance_df = load_akm_variance_decomposition(context)
    stability_df = load_split_sample_stability(context)

    frame = pd.DataFrame(_panel_a(variance_df) + _panel_b(connected, stability_df) + _panel_c(connected))

    memo = standard_footnote(
        sample="Connected-set voyages with non-missing theta_hat and psi_hat from the paper ML outcome dataset.",
        unit="Voyage for sorting moments; captain or agent entity for reliability rows.",
        types_note="Full-sample AKM estimates are reported alongside split-half and time-split held-out reliability diagnostics.",
        fe="AKM two-way captain and agent effects in the upstream estimator.",
        cluster="Not applicable for descriptive reliability summaries.",
        controls="Time-split held-out comparison conditions on tonnage in the AKM cross-fit to match the existing repository routine.",
        interpretation="Captain and agent effects are both sizable, while observed sorting moments vary materially across eras and basins.",
        caution="The repository ships split-half stability directly, but full held-out cross-fit coverage is incomplete for captains, so held-out correlations should be read as lower-bound diagnostics.",
    )
    memo = (
        "# table02_types\n\n"
        + memo
        + "\n\nImplementation notes:\n"
        + "- Panel A uses `data/final/akm_variance_decomposition.csv`.\n"
        + "- Panel B combines the shipped split-sample stability file with a fresh time-split cross-fit on the connected outcome sample.\n"
        + "- Basin labels in Panel C are inferred from `ground_or_route` strings because the repository does not yet ship a fully cleaned basin ontology table.\n"
    )

    return save_table_outputs(
        name="table02_types",
        frame=frame,
        out_dir=context.outputs / "tables",
        context=context,
        memo=memo,
        title="Table 2. Type Estimation, Sorting, and Identification",
    )
