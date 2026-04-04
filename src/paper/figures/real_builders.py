from __future__ import annotations

import importlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import BuildContext
from ..sample_lineage import build_master_sample_lineage
from ..tables.table05_state_switching import _prepare_state_transitions
from ..data import load_connected_sample, load_state_dataset
from ..utils.footnotes import standard_footnote


COLORS = {
    "captain": "#1f4e79",
    "agent": "#d17b0f",
    "accent": "#2a9d8f",
    "risk": "#b23a48",
    "neutral": "#6c757d",
}


def _load_table(context: BuildContext, name: str) -> pd.DataFrame:
    path = context.outputs / "tables" / f"{name}.csv"
    if not path.exists():
        mod = importlib.import_module(f"src.paper.tables.{name}")
        mod.build(context)
    return pd.read_csv(path)


def _save_figure(context: BuildContext, name: str, fig: plt.Figure, memo: str) -> dict[str, str]:
    path = context.outputs / "figures" / f"{name}.png"
    memo_path = context.outputs / "memos" / f"{name}.md"
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    memo_path.write_text(memo.rstrip() + "\n", encoding="utf-8")
    return {"name": name, "figure": str(path), "memo": str(memo_path)}


def _empty_figure(context: BuildContext, name: str, title: str) -> dict[str, str]:
    """Create a placeholder figure when data is unavailable."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, "No data available — run the full pipeline",
            ha="center", va="center", fontsize=14, color=COLORS["neutral"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title)
    memo = f"# {name}\n\nPlaceholder: upstream data not yet generated.\n"
    return _save_figure(context, name, fig, memo)


def _memo(name: str, sample: str, unit: str, interpretation: str, caution: str, notes: list[str]) -> str:
    return (
        f"# {name}\n\n"
        + standard_footnote(
            sample=sample,
            unit=unit,
            types_note="Figure uses the paper-layer table outputs and shipped AKM/search datasets; theta_hat and psi_hat are paper-facing notation.",
            fe="As inherited from the underlying table or transition builder.",
            cluster="As inherited from the underlying table or transition builder.",
            controls="No additional figure-only controls.",
            interpretation=interpretation,
            caution=caution,
        )
        + "\n\nImplementation notes:\n"
        + "\n".join(f"- {note}" for note in notes)
        + "\n"
    )


def _fig01(context: BuildContext):
    build_master_sample_lineage(context)
    lineage = pd.read_parquet(context.outputs / "manifests" / "master_sample_lineage.parquet")
    stages = [
        ("Universe", int(lineage["in_universe"].sum())),
        ("Connected set", int(lineage["in_connected_set"].sum())),
        ("Coordinates", int(lineage["has_coordinates"].sum())),
        ("Ground labels", int(lineage["has_ground_labels"].sum())),
        ("Patch data", int(lineage["has_patch_data"].sum())),
        ("Encounter data", int(lineage["has_encounter_data"].sum())),
        ("Switch event", int(lineage["has_switch_event"].sum())),
        ("Archival proxy", int(lineage["has_archival_data"].sum())),
    ]
    labels = [stage for stage, _ in stages]
    values = [value for _, value in stages]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels, values, color=COLORS["accent"])
    ax.invert_yaxis()
    ax.set_xlabel("Voyages")
    ax.set_title("Sample Flow Across Paper Layers")
    for idx, value in enumerate(values):
        ax.text(value, idx, f" {value:,}", va="center", fontsize=9)
    return _save_figure(
        context,
        "fig01_sample_flow",
        fig,
        _memo(
            "fig01_sample_flow",
            "Master sample-lineage manifest.",
            "Voyage count by inclusion stage.",
            "The figure makes the replication sample architecture explicit from the universe through the main analytic subsamples.",
            "Counts can overlap because later stages are capability flags rather than a single strict filtration ladder.",
            ["Built directly from `master_sample_lineage.parquet`."],
        ),
    )


def _fig02(context: BuildContext):
    table = _load_table(context, "table03_hierarchical_map")
    if table.empty or "model" not in table.columns or "captain_marginal_contribution" not in table.columns:
        return _empty_figure(context, "fig02_map_hierarchy", "Hierarchical Destination Contributions")
    subset = table[(table["model"] == "multinomial_logit") & (table["specification"] == "4. + captain + agent")].copy()
    if subset.empty:
        return _empty_figure(context, "fig02_map_hierarchy", "Hierarchical Destination Contributions")
    x = np.arange(len(subset))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - 0.17, subset["captain_marginal_contribution"], width=0.34, label="Captain contribution", color=COLORS["captain"])
    ax.bar(x + 0.17, subset["agent_marginal_contribution"], width=0.34, label="Agent contribution", color=COLORS["agent"])
    ax.set_xticks(x)
    ax.set_xticklabels(subset["level"], rotation=15, ha="right")
    ax.set_ylabel("Log-loss improvement")
    ax.set_title("Hierarchical Destination Contributions")
    ax.legend(frameon=False)
    return _save_figure(
        context,
        "fig02_map_hierarchy",
        fig,
        _memo(
            "fig02_map_hierarchy",
            "Paper Table 3, multinomial logit hierarchy rows.",
            "Destination level.",
            "Captain and agent contributions vary by destination level rather than collapsing into a single maps-versus-compasses contrast.",
            "The figure uses the time-split table output and does not yet add the captain-group or agent-group holdout overlays.",
            ["Uses the `4. + captain + agent` rows from Table 3."],
        ),
    )


def _fig03(context: BuildContext):
    table = _load_table(context, "table04_stopping")
    subset = table[table["panel"] == "Panel B"].copy()
    if subset.empty or "coefficient" not in subset.columns:
        return _empty_figure(context, "fig03_stopping_margins", "Stopping Margins Under Alternative Negative Signals")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(np.arange(len(subset)), subset["coefficient"], color=COLORS["risk"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(subset)))
    ax.set_xticklabels(subset["row_label"], rotation=25, ha="right")
    ax.set_ylabel("psi × signal coefficient")
    ax.set_title("Stopping Margins Under Alternative Negative Signals")
    return _save_figure(
        context,
        "fig03_stopping_margins",
        fig,
        _memo(
            "fig03_stopping_margins",
            "Paper Table 4 robustness panel.",
            "Signal-definition row.",
            "The sign and magnitude of the stopping-rule interaction can be compared across the alternative negative-signal definitions used in the paper layer.",
            "These coefficients are clustered-LPM robustness rows rather than re-estimated logit hazards.",
            ["Built from Table 4 Panel B."],
        ),
    )


def _fig04(context: BuildContext):
    state_df = load_state_dataset(context)
    connected = load_connected_sample(context)

    transitions, _ = _prepare_state_transitions(state_df, connected)
    if transitions.empty or "state_label" not in transitions.columns or "next_state" not in transitions.columns:
        return _empty_figure(context, "fig04_state_transitions", "State-Transition Heatmap")
    matrix = (
        transitions.groupby(["state_label", "next_state"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    share = matrix.div(matrix.sum(axis=1).replace(0, np.nan), axis=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(share.to_numpy(), cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(len(share.columns)))
    ax.set_xticklabels(share.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(share.index)))
    ax.set_yticklabels(share.index)
    ax.set_title("State-Transition Heatmap")
    fig.colorbar(im, ax=ax, label="Transition share")
    return _save_figure(
        context,
        "fig04_state_transitions",
        fig,
        _memo(
            "fig04_state_transitions",
            "State dataset rebuilt with the Table 5 latent-state labels.",
            "Current-state to next-state transition share.",
            "The latent-state reconstruction yields an interpretable transition matrix centered on barren search, exploitation, and transit dynamics.",
            "The heatmap summarizes empirical transition frequencies rather than regression coefficients.",
            ["Transitions are rebuilt with the same helper used by Table 5."],
        ),
    )


def _fig05(context: BuildContext):
    table = _load_table(context, "table05_state_switching")
    subset = table[table["panel"] == "Panel C"].copy()
    if subset.empty or "coefficient" not in subset.columns:
        return _empty_figure(context, "fig05_switch_event_study", "Switch Event Study")
    x = np.arange(len(subset))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.errorbar(x, subset["coefficient"], yerr=subset["std_error"].fillna(0), marker="o", color=COLORS["captain"], linewidth=2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(subset["row_label"])
    ax.set_ylabel("Change relative to t-1")
    ax.set_title("Switch Event Study")
    return _save_figure(
        context,
        "fig05_switch_event_study",
        fig,
        _memo(
            "fig05_switch_event_study",
            "Paper Table 5 event-study panel.",
            "Relative event time around the first observed switch.",
            "The figure shows how barren-state exit behavior shifts around the first observed agent switch within captain histories.",
            "Confidence intervals inherit the simplified event-study construction used in Table 5.",
            ["Built from Table 5 Panel C."],
        ),
    )


def _fig06(context: BuildContext):
    table = _load_table(context, "table06_search_execution_exitvalue")
    subset = table[table["panel"] == "Panel B"].copy()
    if subset.empty or "simple_difference" not in subset.columns:
        return _empty_figure(context, "fig06_exit_value", "Value of Exit from Barren Search")
    methods = ["simple_difference", "matched_estimate", "ipw_estimate", "doubly_robust_estimate"]
    x = np.arange(len(subset))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, method in enumerate(methods):
        ax.bar(x + (idx - 1.5) * width, subset[method], width=width, label=method.replace("_", " "))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(subset["row_label"], rotation=25, ha="right")
    ax.set_ylabel("Exit minus stay")
    ax.set_title("Value of Exit from Barren Search")
    ax.legend(frameon=False, ncols=2)
    return _save_figure(
        context,
        "fig06_exit_value",
        fig,
        _memo(
            "fig06_exit_value",
            "Paper Table 6 value-of-exit panel.",
            "Outcome-by-estimator comparison.",
            "The figure compares the exit-versus-stay value estimates across simple, matched, IPW, and doubly robust estimators.",
            "The estimators are paper-layer reconstructions on the shipped action panel rather than a standalone causal design.",
            ["Built from Table 6 Panel B."],
        ),
    )


def _fig07(context: BuildContext):
    table = _load_table(context, "table06_search_execution_exitvalue")
    subset = table[table["panel"] == "Panel A"].copy()
    if subset.empty or "psi_hat_coefficient" not in subset.columns:
        return _empty_figure(context, "fig07_search_vs_execution", "Search Versus Execution Decomposition")
    x = np.arange(len(subset))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - 0.18, subset["psi_hat_coefficient"], width=0.36, label="psi_hat", color=COLORS["agent"])
    ax.bar(x + 0.18, subset["theta_hat_coefficient"], width=0.36, label="theta_hat", color=COLORS["captain"])
    ax.set_xticks(x)
    ax.set_xticklabels(subset["row_label"], rotation=25, ha="right")
    ax.set_ylabel("Coefficient")
    ax.set_title("Search Versus Execution Decomposition")
    ax.legend(frameon=False)
    return _save_figure(
        context,
        "fig07_search_vs_execution",
        fig,
        _memo(
            "fig07_search_vs_execution",
            "Paper Table 6 production-chain panel.",
            "Stage of the production chain.",
            "The organizational capability story is strongest in cumulative search governance rather than at every single execution stage.",
            "These coefficients come from mixed model families collapsed into a common table format.",
            ["Built from Table 6 Panel A."],
        ),
    )


def _fig08(context: BuildContext):
    table = _load_table(context, "table08_floor_raising")
    subset = table[(table["panel"] == "Panel B") & table["row_label"].isin(["novice", "expert", "low theta", "high theta"])].copy()
    if subset.empty or "marginal_effect_of_psi" not in subset.columns:
        return _empty_figure(context, "fig08_floor_raising", "Downside-Risk Effects by Experience and Skill")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(np.arange(len(subset)), subset["marginal_effect_of_psi"], color=COLORS["risk"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(subset)))
    ax.set_xticklabels(subset["row_label"], rotation=15, ha="right")
    ax.set_ylabel("psi effect on bottom-decile risk")
    ax.set_title("Downside-Risk Effects by Experience and Skill")
    return _save_figure(
        context,
        "fig08_floor_raising",
        fig,
        _memo(
            "fig08_floor_raising",
            "Paper Table 8 heterogeneity panel.",
            "Subgroup.",
            "The figure highlights where the downside-risk reduction from organizational capability is strongest across novice/expert and low/high-skill captains.",
            "These are subgroup slopes from the appendix-style heterogeneity rows rather than a single interaction regression.",
            ["Built from Table 8 Panel B."],
        ),
    )


def _fig09(context: BuildContext):
    table = _load_table(context, "table10_tail_matching")
    subset = table[(table["panel"] == "Panel A") & table["row_label"].isin(["mean output: theta × psi", "bottom-decile risk: theta × psi", "expected shortfall: theta × psi", "long dry spell: theta × psi"])].copy()
    if subset.empty or "coefficient" not in subset.columns:
        return _empty_figure(context, "fig09_tail_submodularity", "Tail Submodularity and Risk Interactions")
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(np.arange(len(subset)), subset["coefficient"], color=[COLORS["captain"], COLORS["risk"], COLORS["risk"], COLORS["risk"]])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(subset)))
    ax.set_xticklabels(subset["row_label"], rotation=20, ha="right")
    ax.set_ylabel("Interaction coefficient")
    ax.set_title("Tail Submodularity and Risk Interactions")
    return _save_figure(
        context,
        "fig09_tail_submodularity",
        fig,
        _memo(
            "fig09_tail_submodularity",
            "Paper Table 10 interaction panel.",
            "Outcome.",
            "The interaction story is more compelling in the left tail than in mean output alone, which is why the matching section is framed around downside-risk objectives.",
            "Different rows are on different scales because the outcomes differ.",
            ["Built from Table 10 Panel A."],
        ),
    )


def _fig10(context: BuildContext):
    table = _load_table(context, "table10_tail_matching")
    subset = table[(table["panel"] == "Panel C") & table["row_label"].isin(["observed assignment", "PAM", "AAM/NAM", "constrained mean-optimal", "constrained CVaR-optimal", "constrained certainty-equivalent-optimal"])].copy()
    if subset.empty or "mean_output" not in subset.columns:
        return _empty_figure(context, "fig10_matching_welfare", "Matching Welfare Under Mean and Risk Objectives")
    x = np.arange(len(subset))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.18, subset["mean_output"], width=0.36, label="Mean output", color=COLORS["captain"])
    ax.bar(x + 0.18, subset["cvar_10"], width=0.36, label="CVaR(10%)", color=COLORS["risk"])
    ax.set_xticks(x)
    ax.set_xticklabels(subset["row_label"], rotation=25, ha="right")
    ax.set_ylabel("Predicted welfare")
    ax.set_title("Matching Welfare Under Mean and Risk Objectives")
    ax.legend(frameon=False)
    return _save_figure(
        context,
        "fig10_matching_welfare",
        fig,
        _memo(
            "fig10_matching_welfare",
            "Paper Table 10 matching panel.",
            "Assignment rule.",
            "Different assignment objectives move mean output and tail protection in different ways, which is the core managerial point of the matching section.",
            "The welfare numbers come from the paper-layer predicted output surface rather than a full equilibrium assignment model.",
            ["Built from Table 10 Panel C."],
        ),
    )


def build_figure(name: str, context: BuildContext):
    builders = {
        "fig01_sample_flow": _fig01,
        "fig02_map_hierarchy": _fig02,
        "fig03_stopping_margins": _fig03,
        "fig04_state_transitions": _fig04,
        "fig05_switch_event_study": _fig05,
        "fig06_exit_value": _fig06,
        "fig07_search_vs_execution": _fig07,
        "fig08_floor_raising": _fig08,
        "fig09_tail_submodularity": _fig09,
        "fig10_matching_welfare": _fig10,
    }
    return builders[name](context)
