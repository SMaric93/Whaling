"""
Test 1: Same-Captain, Same-Ground, Different-Agent Design.

Tests whether organizational capability (ψ) affects within-ground search
behavior, holding captain skill constant through the absorbing fixed effect
α_{c×g} (captain × ground).

Key specification:
    y_{cgat} = α_{c×g} + δ_t + β·ψ_a + Γ·X + ε

Where y is a search behavior metric (straightness, turning, Lévy μ, etc.)
β > 0 means higher organizational capability improves search behavior.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import CFG, COLS, TABLES_DIR, FIGURES_DIR
from .type_measures import build_switch_sample, classify_movers_stayers
from .utils import absorb_fixed_effects, cluster_se, make_table, make_figure, save_figure, write_memo

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Core Regression
# ═══════════════════════════════════════════════════════════════════════════

def run_test1(
    df: pd.DataFrame,
    *,
    outcomes: Optional[List[str]] = None,
    use_heldout: bool = True,
    save_outputs: bool = True,
) -> Dict:
    """
    Run Test 1: Same-captain, same-ground, different-agent design.

    Parameters
    ----------
    df : pd.DataFrame
        Voyage panel with theta, psi, ground_or_route, search metrics.
    outcomes : list of str
        Search behavior outcome variables.
    use_heldout : bool
        Use held-out psi (preferred to avoid circularity).
    save_outputs : bool
        Save tables and figures.

    Returns
    -------
    dict with regression results, switch sample stats, and audit info.
    """
    psi_col = "psi_heldout" if (use_heldout and "psi_heldout" in df.columns) else "psi"

    if outcomes is None:
        outcomes = _find_outcome_vars(df)

    logger.info("Test 1: %d outcome variables, psi_col=%s", len(outcomes), psi_col)

    # ── 1. Build switch sample ─────────────────────────────────────────
    switch_df = build_switch_sample(df, require_same_ground=True)
    switch_df = switch_df.dropna(subset=[psi_col])

    audit = {
        "n_voyages": len(switch_df),
        "n_captains": switch_df[COLS.captain_id].nunique(),
        "n_agents": switch_df[COLS.agent_id].nunique(),
        "n_captain_ground_pairs": switch_df["captain_ground"].nunique() if "captain_ground" in switch_df.columns else 0,
        "psi_col": psi_col,
    }
    logger.info("Switch sample: %s", audit)

    if len(switch_df) < 50:
        logger.warning("Test 1: switch sample too small (%d), aborting", len(switch_df))
        return {"audit": audit, "results": [], "status": "insufficient_data"}

    # ── 2. Run regressions ─────────────────────────────────────────────
    results = []
    for outcome in outcomes:
        if outcome not in switch_df.columns or switch_df[outcome].isna().all():
            continue

        for fe_spec_name, fe_col_names in _get_fe_specifications(switch_df):
            res = _run_single_regression(
                switch_df, outcome, psi_col, fe_col_names, fe_spec_name,
            )
            if res is not None:
                results.append(res)

    # ── 3. Event study ─────────────────────────────────────────────────
    event_results = _run_event_study(switch_df, outcomes, psi_col)

    # ── 4. Save outputs ────────────────────────────────────────────────
    if save_outputs and results:
        _save_test1_outputs(results, event_results, audit)

    return {
        "audit": audit,
        "results": results,
        "event_study": event_results,
        "status": "complete",
    }


def _find_outcome_vars(df: pd.DataFrame) -> List[str]:
    """Find available search behavior outcome variables."""
    candidates = [
        "straightness_index", "turning_concentration", "levy_mu",
        "revisit_rate", "first_passage_time_50nm", "avg_daily_distance_nm",
        "local_redundancy", "route_efficiency", "ground_switching_count",
    ]
    return [c for c in candidates if c in df.columns and df[c].notna().sum() > 50]


def _get_fe_specifications(df: pd.DataFrame) -> List:
    """Define FE specifications (returns column name lists, not arrays)."""
    specs = []
    if "captain_ground" in df.columns:
        specs.append(("captain_ground + year", ["captain_ground", COLS.year_out]))
    specs.append(("captain + ground + year",
                   [COLS.captain_id, COLS.ground_or_route, COLS.year_out]))
    if "captain_ground" in df.columns and "tonnage" in df.columns:
        specs.append(("captain_ground + year + controls",
                       ["captain_ground", COLS.year_out]))
    return specs


def _run_single_regression(
    df: pd.DataFrame,
    outcome: str,
    psi_col: str,
    fe_col_names: List[str],
    fe_spec_name: str,
) -> Optional[Dict]:
    """Run a single FE regression and compute clustered SEs."""
    clean = df.dropna(subset=[outcome, psi_col]).copy()
    if len(clean) < 30:
        return None

    y = clean[outcome].values.astype(float)
    X = clean[[psi_col]].values.astype(float)

    # Build FE groups from cleaned data (matching dimensions)
    fe_groups = [clean[col].fillna("UNK").values for col in fe_col_names]

    result = absorb_fixed_effects(y, X, fe_groups, return_residuals=True)

    if result["dof"] < 10:
        return None

    # Clustered SE
    mask = result["_mask"]
    residuals = result["residuals"]
    clusters = clean[COLS.captain_id].values[mask]
    se = cluster_se(X[mask], residuals, clusters)

    coef = result["coefficients"][0]
    t_stat = coef / se[0] if se[0] > 0 else 0
    from scipy.stats import t as t_dist
    p_val = 2 * (1 - t_dist.cdf(abs(t_stat), df=result["dof"]))

    return {
        "name": f"{outcome} | {fe_spec_name}",
        "outcome": outcome,
        "fe_structure": fe_spec_name,
        "coefficients": {psi_col: coef},
        "se": {psi_col: se[0]},
        "pvalues": {psi_col: p_val},
        "t_stat": t_stat,
        "n_obs": result["n_obs"],
        "r_squared": result["r_squared"],
        "n_clusters": len(np.unique(clusters)),
        "cluster_var": COLS.captain_id,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Event Study
# ═══════════════════════════════════════════════════════════════════════════

def _run_event_study(
    df: pd.DataFrame,
    outcomes: List[str],
    psi_col: str,
) -> Optional[Dict]:
    """
    Event study around agent switches.

    Creates event-time dummies around the first agent switch for each captain.
    """
    if COLS.switch_agent not in df.columns:
        return None

    df = df.sort_values([COLS.captain_id, COLS.year_out])

    # Find first switch year for each captain
    switches = df[df[COLS.switch_agent] == 1]
    if len(switches) == 0:
        return None

    first_switch = switches.groupby(COLS.captain_id)[COLS.year_out].min()
    first_switch.name = "switch_year"
    df = df.merge(first_switch, on=COLS.captain_id, how="left")
    df = df[df["switch_year"].notna()]

    # Event time
    df["event_time"] = df[COLS.year_out] - df["switch_year"]
    window = CFG.event_study_window
    df = df[(df["event_time"] >= -window) & (df["event_time"] <= window)]

    event_results = {}
    for outcome in outcomes:
        if outcome not in df.columns or df[outcome].isna().all():
            continue

        # Simple event-time means by psi group
        df["high_psi"] = (df[psi_col] >= df[psi_col].median()).astype(int)

        means = df.groupby(["event_time", "high_psi"])[outcome].agg(
            ["mean", "std", "count"]
        ).reset_index()
        means["se"] = means["std"] / np.sqrt(means["count"].clip(lower=1))

        event_results[outcome] = means

    return event_results


# ═══════════════════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════════════════

def _save_test1_outputs(results, event_results, audit):
    """Save Test 1 tables, figures, and memo."""
    # Regression table
    table = make_table(
        results, "Search Behavior", "test1_map_compass",
    )

    # Audit table
    audit_df = pd.DataFrame([audit])
    audit_df.to_csv(TABLES_DIR / "test1_switch_sample_audit.csv", index=False)

    # Event study figures
    if event_results:
        for outcome, means in event_results.items():
            try:
                fig, ax = make_figure("test1", f"event_study_{outcome}")
                for psi_group, gdf in means.groupby("high_psi"):
                    label = "High ψ" if psi_group else "Low ψ"
                    color = "#2196F3" if psi_group else "#FF5722"
                    ax.plot(
                        gdf["event_time"], gdf["mean"],
                        marker="o", label=label, color=color,
                    )
                    ax.fill_between(
                        gdf["event_time"],
                        gdf["mean"] - 1.96 * gdf["se"],
                        gdf["mean"] + 1.96 * gdf["se"],
                        alpha=0.15, color=color,
                    )
                ax.axvline(0, color="gray", linestyle="--", alpha=0.5, label="Switch")
                ax.set_xlabel("Years Relative to Agent Switch")
                ax.set_ylabel(outcome.replace("_", " ").title())
                ax.legend()
                ax.set_title(f"Event Study: {outcome.replace('_', ' ').title()}")
                save_figure(fig, "test1", f"event_study_{outcome}")
            except Exception as e:
                logger.warning("Failed to save event study figure for %s: %s", outcome, e)

    # Memo
    n_sig = sum(1 for r in results if r["pvalues"].get(list(r["pvalues"].keys())[0], 1) < 0.05)
    write_memo(
        "test1_map_compass",
        f"## Same-Captain, Same-Ground, Different-Agent\n\n"
        f"- Switch sample: {audit['n_voyages']} voyages, {audit['n_captains']} captains\n"
        f"- Significant at 5%: {n_sig} / {len(results)} specifications\n"
        f"- FE structure absorbs captain×ground, so β on ψ is identified from "
        f"within-captain, within-ground variation in organizational affiliation.\n",
        threats=[
            "Agent switches may be endogenous (selection into better agents)",
            "Logbook coverage is ~10% of all voyages",
            "Ground classification is approximate",
        ],
    )

    logger.info("Test 1 outputs saved")
