"""
Test 2: Same-Captain, Same-State, Different-Agent Switch Design.

Estimates whether the same captain changes decision policy after
joining a different agent, holding the decision state as constant as possible.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from src.next_round.config import (
    OUTPUTS_TABLES, OUTPUTS_FIGURES, PSI_COL, THETA_COL,
    RANDOM_SEED, FIGURE_DPI, FIGURE_FORMAT,
)

logger = logging.getLogger(__name__)

STATE_MATCH_VARS = [
    "consecutive_empty_days", "days_since_last_success",
    "season_remaining", "scarcity", "duration_day",
]


def run_switch_policy_change(*, save_outputs: bool = True) -> Dict:
    """
    Test whether captains change policy after switching agents.

    Uses within-captain variation with matched-state controls.
    """
    logger.info("=" * 60)
    logger.info("Test 2: Switch Policy Change")
    logger.info("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────
    from src.ml.build_action_dataset import build_action_dataset
    from src.reinforcement.data_builder import build_analysis_panel

    actions = build_action_dataset(force_rebuild=False, save=False)
    voyages = build_analysis_panel(require_akm=True)

    # Identify switching captains
    switch_captains = _identify_switchers(voyages)
    logger.info("Found %d captains who switched agents", len(switch_captains))

    if len(switch_captains) < 10:
        logger.warning("Too few switching captains (%d) for reliable analysis",
                       len(switch_captains))
        return {"error": "insufficient_switchers", "n_switchers": len(switch_captains)}

    # Merge voyage-level switch info into action data
    info_cols = ["voyage_id", "captain_id", "agent_id"]
    for c in [PSI_COL, THETA_COL]:
        if c in voyages.columns:
            info_cols.append(c)
    voyage_info = voyages[info_cols].drop_duplicates(subset=["voyage_id"]).copy()
    voyage_info = voyage_info.dropna(subset=["captain_id"])

    # Tag pre/post switch
    df = _tag_pre_post_switch(actions, voyage_info, switch_captains)
    logger.info("Tagged %d observations as pre/post switch", len(df))

    results = {}

    # ── 1. Event study ────────────────────────────────────────────────
    event_study = _run_event_study(df)
    results["event_study"] = event_study

    # ── 2. Captain FE with matched-state controls ─────────────────────
    captain_fe = _run_captain_fe(df)
    results["captain_fe"] = captain_fe

    # ── 3. Simple pre/post comparison ─────────────────────────────────
    pre_post = _run_pre_post(df)
    results["pre_post"] = pre_post

    # ── Save ──────────────────────────────────────────────────────────
    if save_outputs:
        _save_outputs(results, df)

    return results


def _identify_switchers(voyages: pd.DataFrame) -> pd.DataFrame:
    """Identify captains who switched agents between voyages."""
    v = voyages.sort_values(["captain_id", "year_out"]).copy()
    v["prev_agent"] = v.groupby("captain_id")["agent_id"].shift(1)
    v["switched"] = (v["agent_id"] != v["prev_agent"]) & v["prev_agent"].notna()

    switchers = v[v["switched"]]["captain_id"].unique()
    return pd.DataFrame({"captain_id": switchers})


def _tag_pre_post_switch(actions, voyage_info, switch_captains):
    """Tag action-level data as pre or post switch."""
    # Merge, drop any conflicting columns first
    drop_cols = [c for c in ["captain_id", "agent_id", PSI_COL, THETA_COL]
                 if c in actions.columns]
    actions_clean = actions.drop(columns=drop_cols, errors='ignore')
    df = actions_clean.merge(voyage_info, on="voyage_id", how="inner")

    # Keep only switching captains
    df = df[df["captain_id"].isin(switch_captains["captain_id"])].copy()

    # For each captain, identify switch points
    captain_agents = (
        df.groupby(["captain_id", "voyage_id"])["agent_id"]
        .first()
        .reset_index()
        .sort_values(["captain_id", "voyage_id"])
    )
    captain_agents["prev_agent"] = captain_agents.groupby("captain_id")["agent_id"].shift(1)
    captain_agents["is_post_switch"] = (
        (captain_agents["agent_id"] != captain_agents["prev_agent"]) &
        captain_agents["prev_agent"].notna()
    ).astype(int)

    # Propagate to action-level
    df = df.merge(
        captain_agents[["captain_id", "voyage_id", "is_post_switch"]],
        on=["captain_id", "voyage_id"],
        how="left",
    )
    df["is_post_switch"] = df["is_post_switch"].fillna(0).astype(int)

    return df


def _run_event_study(df: pd.DataFrame) -> Dict:
    """Run event study around switch date."""
    outcomes = {}

    for outcome in ["exit_patch_next", "next_action_class"]:
        if outcome not in df.columns:
            continue

        # Group by pre/post and compute means
        summary = (
            df.groupby("is_post_switch")[outcome]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        summary.columns = ["is_post_switch", "mean", "std", "n"]
        outcomes[outcome] = summary

    return outcomes


def _run_captain_fe(df: pd.DataFrame) -> Dict:
    """Run captain FE regression with state controls."""
    try:
        import statsmodels.formula.api as smf

        # Build state control string
        available_controls = [c for c in STATE_MATCH_VARS if c in df.columns]
        if not available_controls:
            return {"error": "no_state_controls"}

        results = {}
        for outcome in ["exit_patch_next"]:
            if outcome not in df.columns:
                continue

            # Simple OLS with captain FE (demeaned)
            df_reg = df[[outcome, "is_post_switch", "captain_id"] + available_controls].dropna()
            if len(df_reg) < 100:
                continue

            # Demean by captain
            for col in [outcome, "is_post_switch"] + available_controls:
                if df_reg[col].dtype in (float, int, np.float64, np.int64):
                    captain_mean = df_reg.groupby("captain_id")[col].transform("mean")
                    df_reg[f"{col}_dm"] = df_reg[col] - captain_mean

            controls_dm = " + ".join(f"{c}_dm" for c in available_controls
                                     if f"{c}_dm" in df_reg.columns)
            formula = f"{outcome}_dm ~ is_post_switch_dm + {controls_dm}" if controls_dm else f"{outcome}_dm ~ is_post_switch_dm"

            try:
                model = smf.ols(formula, data=df_reg).fit(
                    cov_type="cluster", cov_kwds={"groups": df_reg["captain_id"]}
                )
                results[outcome] = {
                    "coef_post_switch": model.params.get("is_post_switch_dm", np.nan),
                    "se_post_switch": model.bse.get("is_post_switch_dm", np.nan),
                    "pval_post_switch": model.pvalues.get("is_post_switch_dm", np.nan),
                    "n_obs": int(model.nobs),
                    "r_squared": model.rsquared,
                }
            except Exception as e:
                logger.warning("Captain FE regression failed: %s", e)
                results[outcome] = {"error": str(e)}

        return results

    except ImportError:
        logger.warning("statsmodels not available for captain FE")
        return {"error": "no_statsmodels"}


def _run_pre_post(df: pd.DataFrame) -> Dict:
    """Simple pre/post comparison."""
    results = {}
    for outcome in ["exit_patch_next", "next_action_class"]:
        if outcome not in df.columns:
            continue

        pre = df[df["is_post_switch"] == 0][outcome]
        post = df[df["is_post_switch"] == 1][outcome]

        results[outcome] = {
            "pre_mean": pre.mean(),
            "post_mean": post.mean(),
            "diff": post.mean() - pre.mean(),
            "pre_n": len(pre),
            "post_n": len(post),
        }

    return results


def _save_outputs(results: Dict, df: pd.DataFrame):
    """Save results table, figure, and memo."""
    # Table
    rows = []
    for key, val in results.get("pre_post", {}).items():
        if isinstance(val, dict) and "error" not in val:
            val["outcome"] = key
            rows.append(val)

    for key, val in results.get("captain_fe", {}).items():
        if isinstance(val, dict) and "error" not in val:
            val["outcome"] = key
            val["method"] = "captain_fe"
            rows.append(val)

    if rows:
        pd.DataFrame(rows).to_csv(
            OUTPUTS_TABLES / "switch_policy_change.csv", index=False)

    # Figure
    try:
        import matplotlib.pyplot as plt

        event_study = results.get("event_study", {})
        if "exit_patch_next" in event_study:
            es = event_study["exit_patch_next"]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(es["is_post_switch"].astype(str), es["mean"],
                   yerr=es["std"] / np.sqrt(es["n"]),
                   color=["#2196F3", "#FF5722"], capsize=5)
            ax.set_xticklabels(["Pre-Switch", "Post-Switch"])
            ax.set_ylabel("P(Exit Patch)")
            ax.set_title("Exit Propensity Around Agent Switch")
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(OUTPUTS_FIGURES / f"switch_policy_change.{FIGURE_FORMAT}",
                       dpi=FIGURE_DPI, bbox_inches="tight")
            plt.close(fig)
    except ImportError:
        pass

    # Memo
    memo = [
        "# Test 2: Switch Policy Change — Memo",
        "",
        "## What this test identifies",
        "Whether the same captain changes stopping/action policy after switching",
        "agents, conditional on being in the same navigational state.",
        "",
        "## What this does NOT identify",
        "- Cannot distinguish instruction from selection into different voyages",
        "- Switch may coincide with other changes (aging, market conditions)",
        "",
        "## Key Results",
    ]
    for outcome, val in results.get("pre_post", {}).items():
        if isinstance(val, dict) and "diff" in val:
            memo.append(f"- **{outcome}**: pre={val['pre_mean']:.4f}, "
                       f"post={val['post_mean']:.4f}, diff={val['diff']:.4f}")

    (OUTPUTS_TABLES / "switch_policy_change_memo.md").write_text("\n".join(memo))
