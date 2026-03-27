"""
Repair 2: Lower-Tail Target Audit.

Verifies that bottom_decile and bottom_5pct are distinct targets,
prevalence differs in train/val/test, and thresholds are computed correctly.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

from src.next_round.config import DEBUG_TABLES, DEBUG_DOCS

logger = logging.getLogger(__name__)


def run_lower_tail_audit(*, save_outputs: bool = True) -> Dict:
    """
    Audit lower-tail target construction.

    Checks:
    1. bottom_decile and bottom_5pct are distinct binary targets
    2. Prevalence differs across train/val/test splits
    3. Thresholds are within-sample (train-only) not leaking from full sample
    """
    from src.ml.build_outcome_ml_dataset import build_outcome_ml_dataset
    from src.ml.splits import split_rolling_time

    logger.info("=" * 60)
    logger.info("Repair 2: Lower-Tail Target Audit")
    logger.info("=" * 60)

    df = build_outcome_ml_dataset(force_rebuild=False, save=False)

    outcome_col = "log_q" if "log_q" in df.columns else "q_total_index"
    if outcome_col not in df.columns:
        for c in df.columns:
            if "q_" in c.lower() or "output" in c.lower():
                outcome_col = c
                break

    logger.info("Using outcome column: %s", outcome_col)

    audit_rows = []

    # ── Global thresholds (potential leakage) ─────────────────────────
    global_p10 = df[outcome_col].quantile(0.10)
    global_p05 = df[outcome_col].quantile(0.05)

    df["bottom_decile_global"] = (df[outcome_col] <= global_p10).astype(int)
    df["bottom_5pct_global"] = (df[outcome_col] <= global_p05).astype(int)

    # ── Split and compute within-train thresholds ─────────────────────
    try:
        train_idx, val_idx, test_idx = split_rolling_time(df)

        train_p10 = df.iloc[train_idx][outcome_col].quantile(0.10)
        train_p05 = df.iloc[train_idx][outcome_col].quantile(0.05)

        df["bottom_decile_train"] = (df[outcome_col] <= train_p10).astype(int)
        df["bottom_5pct_train"] = (df[outcome_col] <= train_p05).astype(int)

        # Check distinctness
        for target_name, global_col, train_col in [
            ("bottom_decile", "bottom_decile_global", "bottom_decile_train"),
            ("bottom_5pct", "bottom_5pct_global", "bottom_5pct_train"),
        ]:
            for split_name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
                subset = df.iloc[idx]
                audit_rows.append({
                    "target": target_name,
                    "split": split_name,
                    "n": len(subset),
                    "prevalence_global_threshold": subset[global_col].mean(),
                    "prevalence_train_threshold": subset[train_col].mean(),
                    "global_threshold": global_p10 if "decile" in target_name else global_p05,
                    "train_threshold": train_p10 if "decile" in target_name else train_p05,
                    "threshold_diff_pct": abs(
                        (train_p10 if "decile" in target_name else train_p05) -
                        (global_p10 if "decile" in target_name else global_p05)
                    ) / max(abs(global_p10 if "decile" in target_name else global_p05), 1e-12) * 100,
                })

        # ── Check: are the two targets actually distinct? ─────────────
        overlap = (df["bottom_decile_global"] == df["bottom_5pct_global"]).mean()
        targets_distinct = overlap < 0.99

    except Exception as e:
        logger.warning("Split-based audit failed: %s", e)
        overlap = np.nan
        targets_distinct = None

    # Check existing target columns in the dataset
    existing_targets = {}
    for col in ["bottom_decile", "bottom_5pct"]:
        if col in df.columns:
            existing_targets[col] = {
                "prevalence": df[col].mean(),
                "n_positive": df[col].sum(),
                "n_total": len(df),
            }

    audit_df = pd.DataFrame(audit_rows)

    results = {
        "audit_table": audit_df,
        "targets_distinct": targets_distinct,
        "target_overlap_pct": overlap * 100 if not np.isnan(overlap) else np.nan,
        "existing_targets": existing_targets,
        "outcome_col": outcome_col,
        "global_p10": global_p10,
        "global_p05": global_p05,
    }

    if save_outputs:
        _save_outputs(results)

    logger.info("Lower-tail audit complete: targets_distinct=%s, overlap=%.1f%%",
                targets_distinct, results["target_overlap_pct"])

    return results


def _save_outputs(results: Dict):
    """Save audit table and memo."""
    out_csv = DEBUG_TABLES / "lower_tail_target_audit.csv"
    results["audit_table"].to_csv(out_csv, index=False)
    logger.info("Saved %s", out_csv)

    memo_path = DEBUG_DOCS / "lower_tail_audit.md"
    lines = [
        "# Lower-Tail Target Audit",
        "",
        "## Summary",
        f"- Outcome column: `{results['outcome_col']}`",
        f"- Global 10th percentile: **{results['global_p10']:.4f}**",
        f"- Global 5th percentile: **{results['global_p05']:.4f}**",
        f"- Targets are distinct: **{results['targets_distinct']}** (overlap: {results['target_overlap_pct']:.1f}%)",
        "",
        "## Prevalence by Split",
        "",
        results["audit_table"].to_markdown(index=False),
        "",
        "## Recommendation",
        "",
        "Use **train-only thresholds** to define targets. This prevents leakage",
        "from future observations into the target definition.",
    ]

    if results["existing_targets"]:
        lines.extend(["", "## Existing Target Columns in Dataset"])
        for col, info in results["existing_targets"].items():
            lines.append(f"- `{col}`: prevalence={info['prevalence']:.4f}, "
                        f"n_pos={info['n_positive']}, n_total={info['n_total']}")

    memo_path.write_text("\n".join(lines))
    logger.info("Saved %s", memo_path)
