"""
Test 4: Value of Exit.

Estimates whether leaving a barren state earlier creates downstream value.
Matches decision episodes where ships are in similar barren states,
compares exit-now vs stay-longer on future outcomes.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

from src.next_round.config import (
    OUTPUTS_TABLES, OUTPUTS_FIGURES, PSI_COL, THETA_COL,
    RANDOM_SEED, FIGURE_DPI, FIGURE_FORMAT,
)

logger = logging.getLogger(__name__)

STATE_MATCH_VARS = [
    "consecutive_empty_days", "days_since_last_success",
    "duration_day", "scarcity",
]

OUTCOME_HORIZONS = [30, 60, 90]


def run_exit_value_eval(*, save_outputs: bool = True) -> Dict:
    """
    Estimate the downstream value of exiting barren states early.
    """
    logger.info("=" * 60)
    logger.info("Test 4: Value of Exit Evaluation")
    logger.info("=" * 60)

    from src.ml.build_action_dataset import build_action_dataset
    from src.reinforcement.data_builder import build_analysis_panel

    actions = build_action_dataset(force_rebuild=False, save=False)
    voyages = build_analysis_panel(require_akm=True)

    # Merge psi
    voyage_info = voyages[["voyage_id", PSI_COL, THETA_COL, "captain_id"]].dropna(subset=[PSI_COL])
    df = actions.merge(voyage_info, on="voyage_id", how="inner")

    # Identify barren episodes
    barren = _identify_barren_episodes(df)
    logger.info("Identified %d barren episodes", len(barren))

    if len(barren) < 100:
        logger.warning("Too few barren episodes for reliable analysis")
        return {"error": "insufficient_barren", "n_episodes": len(barren)}

    # ── Build future outcomes ─────────────────────────────────────────
    barren = _compute_future_outcomes(barren, df)

    # ── Exit vs stay comparison ───────────────────────────────────────
    results = {}

    # 1. Simple comparison
    results["simple"] = _simple_comparison(barren)

    # 2. Matched comparison (coarsened exact matching)
    results["matched"] = _matched_comparison(barren)

    # 3. IPW estimator
    results["ipw"] = _ipw_estimator(barren)

    if save_outputs:
        _save_outputs(results, barren)

    return results


def _identify_barren_episodes(df: pd.DataFrame) -> pd.DataFrame:
    """Identify episodes where ship is in a barren search state."""
    barren_mask = pd.Series(False, index=df.index)

    # Use multiple indicators for barren state
    if "consecutive_empty_days" in df.columns:
        barren_mask |= df["consecutive_empty_days"] >= 5

    if "hmm_state_label" in df.columns:
        barren_mask |= df["hmm_state_label"].isin(["barren_search"])

    if "state_label" in df.columns:
        barren_mask |= df["state_label"].isin(["barren_search"])

    # Need exit decision
    if "exit_patch_next" in df.columns:
        barren = df[barren_mask & df["exit_patch_next"].notna()].copy()
        barren["exited"] = barren["exit_patch_next"].astype(int)
    else:
        barren = df[barren_mask].copy()
        barren["exited"] = 0  # placeholder

    return barren


def _compute_future_outcomes(barren: pd.DataFrame, full_df: pd.DataFrame) -> pd.DataFrame:
    """Compute forward-looking outcomes for each barren episode."""
    # Sort by voyage and time
    full_df = full_df.sort_values(["voyage_id", "obs_date" if "obs_date" in full_df.columns else full_df.columns[1]])

    for horizon in OUTCOME_HORIZONS:
        col_name = f"future_output_{horizon}d"
        barren[col_name] = np.nan  # placeholder

    # For each barren observation, look ahead
    # Precompute binary encounter for performance
    encounter_col = None
    for col in ["encounter", "whale_encounter", "sighting"]:
        if col in full_df.columns:
            encounter_col = col
            break

    if encounter_col:
        # Convert string encounter to binary
        full_df = full_df.copy()
        full_df["_enc_binary"] = (~full_df[encounter_col].isin(["NoEnc", "No", "", "0", 0, False]) & full_df[encounter_col].notna()).astype(int)

    # Sample to avoid O(N^2) loop on 238K rows
    barren_sample = barren.sample(min(5000, len(barren)), random_state=42) if len(barren) > 5000 else barren

    for voyage_id, vdf in full_df.groupby("voyage_id"):
        v_barren = barren_sample[barren_sample["voyage_id"] == voyage_id]
        if v_barren.empty:
            continue

        vdf = vdf.reset_index(drop=True)

        for idx in v_barren.index:
            row = barren.loc[idx]
            try:
                pos = vdf.index[vdf["obs_date"] == row.get("obs_date")].tolist()
                if not pos:
                    continue
                pos = pos[0]

                for horizon in OUTCOME_HORIZONS:
                    future = vdf.iloc[pos:pos + horizon]
                    if "_enc_binary" in future.columns:
                        barren.loc[idx, f"future_output_{horizon}d"] = float(future["_enc_binary"].sum())
                    elif "daily_output" in future.columns:
                        barren.loc[idx, f"future_output_{horizon}d"] = float(future["daily_output"].sum())
            except Exception:
                continue

    return barren


def _simple_comparison(barren: pd.DataFrame) -> Dict:
    """Simple exit vs stay comparison."""
    results = {}
    exit_group = barren[barren["exited"] == 1]
    stay_group = barren[barren["exited"] == 0]

    for horizon in OUTCOME_HORIZONS:
        col = f"future_output_{horizon}d"
        if col not in barren.columns:
            continue

        exit_mean = exit_group[col].mean()
        stay_mean = stay_group[col].mean()

        results[f"horizon_{horizon}d"] = {
            "exit_mean": exit_mean,
            "stay_mean": stay_mean,
            "diff": exit_mean - stay_mean,
            "n_exit": len(exit_group),
            "n_stay": len(stay_group),
        }

    return results


def _matched_comparison(barren: pd.DataFrame) -> Dict:
    """Coarsened exact matching on state variables."""
    available_match = [c for c in STATE_MATCH_VARS if c in barren.columns]
    if not available_match:
        return {"error": "no_match_vars"}

    # Coarsen continuous variables into bins
    df_match = barren.copy()
    for col in available_match:
        if df_match[col].dtype in (float, np.float64):
            try:
                df_match[f"{col}_bin"] = pd.qcut(
                    df_match[col].rank(method="first"), 5,
                    labels=False, duplicates="drop"
                )
            except Exception:
                df_match[f"{col}_bin"] = pd.cut(df_match[col], 5, labels=False)

    bin_cols = [f"{c}_bin" for c in available_match if f"{c}_bin" in df_match.columns]
    if not bin_cols:
        return {"error": "binning_failed"}

    # Create strata
    df_match["stratum"] = df_match[bin_cols].astype(str).agg("_".join, axis=1)

    # Keep strata with both exiters and stayers
    strata_counts = df_match.groupby(["stratum", "exited"]).size().unstack(fill_value=0)
    valid_strata = strata_counts[(strata_counts.get(0, 0) > 0) & (strata_counts.get(1, 0) > 0)].index
    df_matched = df_match[df_match["stratum"].isin(valid_strata)]

    results = {"n_matched": len(df_matched), "n_strata": len(valid_strata)}

    for horizon in OUTCOME_HORIZONS:
        col = f"future_output_{horizon}d"
        if col not in df_matched.columns:
            continue

        exit_mean = df_matched[df_matched["exited"] == 1][col].mean()
        stay_mean = df_matched[df_matched["exited"] == 0][col].mean()
        results[f"matched_diff_{horizon}d"] = exit_mean - stay_mean

    return results


def _ipw_estimator(barren: pd.DataFrame) -> Dict:
    """Inverse probability weighting estimator."""
    try:
        from sklearn.linear_model import LogisticRegression

        available = [c for c in STATE_MATCH_VARS + [PSI_COL, THETA_COL]
                     if c in barren.columns]
        if not available or "exited" not in barren.columns:
            return {"error": "missing_data"}

        df_ipw = barren[available + ["exited"]].dropna()
        if len(df_ipw) < 50:
            return {"error": "insufficient_data"}

        X = df_ipw[available].values
        T = df_ipw["exited"].values

        # Propensity score
        ps_model = LogisticRegression(max_iter=300, random_state=RANDOM_SEED)
        ps_model.fit(X, T)
        ps = ps_model.predict_proba(X)[:, 1]
        ps = np.clip(ps, 0.05, 0.95)  # Trim extreme propensities

        results = {"n_obs": len(df_ipw), "mean_propensity": ps.mean()}

        for horizon in OUTCOME_HORIZONS:
            col = f"future_output_{horizon}d"
            if col not in barren.columns:
                continue

            y = barren.loc[df_ipw.index, col].values
            valid = ~np.isnan(y)
            if valid.sum() < 20:
                continue

            # IPW ATE
            y_v, T_v, ps_v = y[valid], T[valid], ps[valid]
            ate = np.mean(T_v * y_v / ps_v) - np.mean((1 - T_v) * y_v / (1 - ps_v))
            results[f"ipw_ate_{horizon}d"] = ate

        return results

    except Exception as e:
        return {"error": str(e)}


def _save_outputs(results: Dict, barren: pd.DataFrame):
    """Save all outputs."""
    rows = []
    for method, method_results in results.items():
        if isinstance(method_results, dict):
            method_results["method"] = method
            rows.append(method_results)

    if rows:
        pd.DataFrame(rows).to_csv(
            OUTPUTS_TABLES / "exit_value_eval.csv", index=False)

    # Figure
    try:
        import matplotlib.pyplot as plt

        simple = results.get("simple", {})
        horizons = []
        diffs = []
        for h in OUTCOME_HORIZONS:
            key = f"horizon_{h}d"
            if key in simple and isinstance(simple[key], dict):
                horizons.append(h)
                diffs.append(simple[key].get("diff", 0))

        if horizons:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(range(len(horizons)), diffs, color="#4CAF50")
            ax.set_xticks(range(len(horizons)))
            ax.set_xticklabels([f"{h}d" for h in horizons])
            ax.set_ylabel("Future Output Gain (Exit − Stay)")
            ax.set_title("Downstream Value of Exiting Barren States")
            ax.axhline(0, color="black", lw=0.5)
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(OUTPUTS_FIGURES / f"exit_value_eval.{FIGURE_FORMAT}",
                       dpi=FIGURE_DPI, bbox_inches="tight")
            plt.close(fig)
    except ImportError:
        pass

    # Memo
    memo = [
        "# Test 4: Value of Exit — Memo",
        "",
        "## What this identifies",
        "Whether leaving a barren state earlier creates measurable downstream value",
        "(faster return to productive states, more catches over 30/60/90 days).",
        "",
        "## What this does NOT identify",
        "- Cannot fully rule out that exiters had better outside options ex ante",
        "- Matching quality depends on the richness of state variables",
        "- Forward outcomes may be confounded by unobserved ground conditions",
    ]
    (OUTPUTS_TABLES / "exit_value_eval_memo.md").write_text("\n".join(memo))
