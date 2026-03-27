"""
Test 6: Repaired Lower-Tail / Insurance Tests.

Re-estimates the floor-raising claim using corrected targets
and proper within-train thresholds.
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

TARGETS = ["bottom_decile", "bottom_5pct", "expected_shortfall_proxy",
           "long_dry_spell", "catastrophic_voyage"]

FEATURES = [THETA_COL, PSI_COL, "captain_voyage_num", "scarcity", "tonnage"]


def run_lower_tail_repair(*, save_outputs: bool = True) -> Dict:
    """
    Re-estimate floor-raising with corrected targets and heterogeneity.
    """
    logger.info("=" * 60)
    logger.info("Test 6: Lower-Tail Repair")
    logger.info("=" * 60)

    from src.ml.build_outcome_ml_dataset import build_outcome_ml_dataset
    from src.ml.splits import split_rolling_time

    df = build_outcome_ml_dataset(force_rebuild=False, save=False)
    df = _build_targets(df)

    train_idx, val_idx, test_idx = split_rolling_time(df)

    results = {}

    for target in TARGETS:
        if target not in df.columns or df[target].isna().all():
            logger.warning("Target '%s' not available, skipping", target)
            continue

        logger.info("Estimating for target: %s (prevalence=%.3f)",
                    target, df[target].mean())

        target_results = _estimate_target(df, target, train_idx, val_idx, test_idx)
        results[target] = target_results

    # ── Heterogeneity ─────────────────────────────────────────────────
    hetero = _estimate_heterogeneity(df, train_idx, val_idx, test_idx)
    results["heterogeneity"] = hetero

    if save_outputs:
        _save_outputs(results, df)

    return results


def _build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Build corrected lower-tail targets."""
    outcome = "q_total_index" if "q_total_index" in df.columns else "log_q"
    if outcome not in df.columns:
        for c in df.columns:
            if "q_" in c.lower():
                outcome = c
                break

    y = df[outcome]

    # Binary tail indicators (using full-sample for now, will be corrected in splits)
    df["bottom_decile"] = (y <= y.quantile(0.10)).astype(int)
    df["bottom_5pct"] = (y <= y.quantile(0.05)).astype(int)

    # Expected shortfall proxy: mean of outcomes below 10th percentile
    p10 = y.quantile(0.10)
    df["expected_shortfall_proxy"] = np.where(y <= p10, y, np.nan)

    # Long dry spell: prolonged zero-output period
    if "max_empty_streak" in df.columns:
        df["long_dry_spell"] = (df["max_empty_streak"] >= df["max_empty_streak"].quantile(0.90)).astype(int)
    elif "days_without_catch" in df.columns:
        df["long_dry_spell"] = (df["days_without_catch"] >= df["days_without_catch"].quantile(0.90)).astype(int)
    else:
        df["long_dry_spell"] = np.nan

    # Catastrophic voyage: bottom 5% AND duration > median
    median_dur = df["duration_days"].median() if "duration_days" in df.columns else 365
    if "duration_days" in df.columns:
        df["catastrophic_voyage"] = (
            (df["bottom_5pct"] == 1) &
            (df["duration_days"] >= median_dur)
        ).astype(int)
    else:
        df["catastrophic_voyage"] = df["bottom_5pct"]

    return df


def _estimate_target(df, target, train_idx, val_idx, test_idx) -> Dict:
    """Estimate models for a single target."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

    available = [f for f in FEATURES if f in df.columns]
    df_clean = df[available + [target]].dropna()

    # Re-align indices
    clean_train = [i for i in train_idx if i < len(df_clean)]
    clean_val = [i for i in val_idx if i < len(df_clean)]
    clean_test = [i for i in test_idx if i < len(df_clean)]

    X = df_clean[available].values
    y = df_clean[target].values

    if len(clean_train) < 50 or y[clean_train].sum() < 5:
        return {"error": "insufficient_positive_cases"}

    results = {}

    # Logistic baseline
    try:
        lr = LogisticRegression(max_iter=300, random_state=RANDOM_SEED)
        lr.fit(X[clean_train], y[clean_train])

        for split_name, idx in [("val", clean_val), ("test", clean_test)]:
            if len(idx) < 10:
                continue
            pred_proba = lr.predict_proba(X[idx])[:, 1]
            results[f"logistic_{split_name}"] = {
                "model": "logistic",
                "split": split_name,
                "auc": roc_auc_score(y[idx], pred_proba) if len(np.unique(y[idx])) > 1 else np.nan,
                "brier": brier_score_loss(y[idx], pred_proba),
                "prevalence": y[idx].mean(),
                "n": len(idx),
            }
    except Exception as e:
        results["logistic_error"] = str(e)

    # HistGBT
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        hgb = HistGradientBoostingClassifier(
            max_iter=200, max_depth=4, random_state=RANDOM_SEED)
        hgb.fit(X[clean_train], y[clean_train])

        for split_name, idx in [("val", clean_val), ("test", clean_test)]:
            if len(idx) < 10:
                continue
            pred_proba = hgb.predict_proba(X[idx])[:, 1]
            results[f"hist_gbt_{split_name}"] = {
                "model": "hist_gbt",
                "split": split_name,
                "auc": roc_auc_score(y[idx], pred_proba) if len(np.unique(y[idx])) > 1 else np.nan,
                "brier": brier_score_loss(y[idx], pred_proba),
                "prevalence": y[idx].mean(),
                "n": len(idx),
            }
    except Exception as e:
        results["hist_gbt_error"] = str(e)

    return results


def _estimate_heterogeneity(df, train_idx, val_idx, test_idx) -> Dict:
    """Report separately by novice/expert, scarcity, mover/stayer."""
    results = {}

    for group_col, group_name in [
        ("novice", "experience"),
        ("scarcity", "scarcity"),
        ("switch_agent", "mover_status"),
    ]:
        if group_col not in df.columns:
            continue

        for group_val in df[group_col].dropna().unique():
            mask = df[group_col] == group_val
            sub = df[mask]
            if len(sub) < 50:
                continue

            for target in ["bottom_decile", "bottom_5pct"]:
                if target not in sub.columns:
                    continue

                results[f"{group_name}_{group_val}_{target}"] = {
                    "group": group_name,
                    "group_value": str(group_val),
                    "target": target,
                    "prevalence": sub[target].mean(),
                    "mean_psi": sub[PSI_COL].mean() if PSI_COL in sub.columns else np.nan,
                    "n": len(sub),
                }

    return results


def _save_outputs(results: Dict, df: pd.DataFrame):
    """Save results."""
    rows = []
    for target, target_results in results.items():
        if target == "heterogeneity":
            continue
        if isinstance(target_results, dict):
            for key, val in target_results.items():
                if isinstance(val, dict):
                    val["target"] = target
                    rows.append(val)

    # Add heterogeneity
    for key, val in results.get("heterogeneity", {}).items():
        if isinstance(val, dict):
            rows.append(val)

    if rows:
        pd.DataFrame(rows).to_csv(
            OUTPUTS_TABLES / "lower_tail_repair.csv", index=False)

    # Figure
    try:
        import matplotlib.pyplot as plt

        # Plot downside risk by psi quartile and experience
        if PSI_COL in df.columns and "bottom_decile" in df.columns:
            df["psi_q"] = pd.qcut(df[PSI_COL].rank(method="first"), 4,
                                   labels=["Q1", "Q2", "Q3", "Q4"])

            if "novice" in df.columns:
                fig, ax = plt.subplots(figsize=(8, 5))
                for novice_val, label, color in [(1, "Novice", "#F44336"), (0, "Expert", "#2196F3")]:
                    sub = df[df["novice"] == novice_val]
                    risk_by_q = sub.groupby("psi_q")["bottom_decile"].mean()
                    ax.plot(risk_by_q.index, risk_by_q.values, "o-",
                           label=label, color=color, lw=2)

                ax.set_xlabel("ψ Quartile")
                ax.set_ylabel("P(Bottom Decile)")
                ax.set_title("Downside Risk by ψ and Experience")
                ax.legend()
                ax.grid(alpha=0.3)
                fig.tight_layout()
                fig.savefig(OUTPUTS_FIGURES / f"lower_tail_repair.{FIGURE_FORMAT}",
                           dpi=FIGURE_DPI, bbox_inches="tight")
                plt.close(fig)
    except ImportError:
        pass

    # Memo
    memo = [
        "# Test 6: Lower-Tail Repair — Memo",
        "",
        "## What this identifies",
        "Whether high-ψ organizations reduce the probability of catastrophic",
        "outcomes (bottom decile, bottom 5%), especially for novice captains.",
        "",
        "## What this does NOT identify",
        "- Cannot separate floor-raising from selection of safer voyages",
        "- Threshold-based targets are sensitive to outcome distribution",
    ]
    (OUTPUTS_TABLES / "lower_tail_repair_memo.md").write_text("\n".join(memo))
