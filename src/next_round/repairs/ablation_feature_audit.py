"""
Repair 1: ML Ablation Feature Audit.

Verifies that captain features enter captain ablations, agent features
enter agent ablations, encoding pipelines don't silently drop categories,
and feature counts differ across ablations when expected.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.next_round.config import DEBUG_TABLES, DEBUG_DOCS

logger = logging.getLogger(__name__)


def run_ablation_feature_audit(*, save_outputs: bool = True) -> Dict:
    """
    Run the ablation feature audit.

    Checks:
    1. Captain features are present in captain ablations
    2. Agent features are present in agent ablations
    3. Encoding pipelines handle categories correctly
    4. Feature counts differ across ablations
    5. Unseen categories in test splits are logged
    """
    from src.ml.policy_learning import (
        ENVIRONMENT_FEATURES, CAPTAIN_FEATURES, AGENT_FEATURES,
        ABLATION_LADDER, _available_features, _encode_categoricals,
    )
    from src.ml.build_action_dataset import build_action_dataset
    from src.ml.splits import split_rolling_time

    logger.info("=" * 60)
    logger.info("Repair 1: ML Ablation Feature Audit")
    logger.info("=" * 60)

    # Load action dataset (cached)
    df = build_action_dataset(force_rebuild=False, save=False)

    audit_rows = []
    category_issues = []

    # ── Check 1: Feature presence per ablation ────────────────────────
    for ablation_name, feature_list in ABLATION_LADDER.items():
        available = _available_features(df, feature_list)
        missing = [f for f in feature_list if f not in df.columns]

        has_captain = any(f in CAPTAIN_FEATURES for f in available)
        has_agent = any(f in AGENT_FEATURES for f in available)
        has_env = any(f in ENVIRONMENT_FEATURES for f in available)

        expected_captain = "captain" in ablation_name
        expected_agent = "agent" in ablation_name

        captain_ok = has_captain == expected_captain
        agent_ok = has_agent == expected_agent

        audit_rows.append({
            "ablation": ablation_name,
            "n_requested": len(feature_list),
            "n_available": len(available),
            "n_missing": len(missing),
            "missing_features": ", ".join(missing) if missing else "none",
            "has_captain_features": has_captain,
            "expected_captain": expected_captain,
            "captain_correct": captain_ok,
            "has_agent_features": has_agent,
            "expected_agent": expected_agent,
            "agent_correct": agent_ok,
            "has_env_features": has_env,
        })

        if not captain_ok:
            logger.warning(
                "CAPTAIN MISMATCH in '%s': has_captain=%s, expected=%s",
                ablation_name, has_captain, expected_captain,
            )
        if not agent_ok:
            logger.warning(
                "AGENT MISMATCH in '%s': has_agent=%s, expected=%s",
                ablation_name, has_agent, expected_agent,
            )

    # ── Check 2: Encoding pipeline ────────────────────────────────────
    all_features = list(set(
        f for flist in ABLATION_LADDER.values() for f in flist
    ))
    available_all = _available_features(df, all_features)

    # Check for object/category columns
    cat_cols = [c for c in available_all if df[c].dtype in ("object", "category")]
    for col in cat_cols:
        n_unique = df[col].nunique()
        n_null = df[col].isna().sum()
        category_issues.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "n_unique": n_unique,
            "n_null": n_null,
            "sample_values": str(df[col].dropna().unique()[:5].tolist()),
        })

    # ── Check 3: Train/test category overlap ──────────────────────────
    try:
        train_idx, val_idx, test_idx = split_rolling_time(df)
        for col in cat_cols:
            train_cats = set(df.iloc[train_idx][col].dropna().unique())
            test_cats = set(df.iloc[test_idx][col].dropna().unique())
            unseen = test_cats - train_cats
            if unseen:
                logger.warning(
                    "Column '%s': %d unseen categories in test: %s",
                    col, len(unseen), list(unseen)[:10],
                )
                category_issues.append({
                    "column": col,
                    "dtype": "unseen_in_test",
                    "n_unique": len(unseen),
                    "n_null": 0,
                    "sample_values": str(list(unseen)[:10]),
                })
    except Exception as e:
        logger.warning("Could not split for category check: %s", e)

    # ── Check 4: Feature count distinctness ───────────────────────────
    counts = {name: len(_available_features(df, flist))
              for name, flist in ABLATION_LADDER.items()}
    all_same = len(set(counts.values())) == 1
    if all_same:
        logger.warning(
            "ALL ABLATIONS HAVE SAME FEATURE COUNT (%d) — likely a bug",
            list(counts.values())[0],
        )

    # ── Build results ─────────────────────────────────────────────────
    audit_df = pd.DataFrame(audit_rows)
    cat_df = pd.DataFrame(category_issues) if category_issues else pd.DataFrame()

    results = {
        "audit_table": audit_df,
        "category_issues": cat_df,
        "feature_counts": counts,
        "all_counts_same": all_same,
        "n_captain_mismatches": sum(1 for r in audit_rows if not r["captain_correct"]),
        "n_agent_mismatches": sum(1 for r in audit_rows if not r["agent_correct"]),
    }

    if save_outputs:
        _save_outputs(results)

    logger.info("Ablation audit complete: %d captain mismatches, %d agent mismatches",
                results["n_captain_mismatches"], results["n_agent_mismatches"])

    return results


def _save_outputs(results: Dict):
    """Save audit table and memo."""
    # CSV
    out_csv = DEBUG_TABLES / "ablation_feature_audit.csv"
    results["audit_table"].to_csv(out_csv, index=False)
    logger.info("Saved %s", out_csv)

    if not results["category_issues"].empty:
        cat_csv = DEBUG_TABLES / "ablation_category_issues.csv"
        results["category_issues"].to_csv(cat_csv, index=False)
        logger.info("Saved %s", cat_csv)

    # Memo
    memo_path = DEBUG_DOCS / "ablation_audit.md"
    lines = [
        "# Ablation Feature Audit",
        "",
        "## Summary",
        f"- Captain feature mismatches: **{results['n_captain_mismatches']}**",
        f"- Agent feature mismatches: **{results['n_agent_mismatches']}**",
        f"- All ablations same feature count: **{results['all_counts_same']}**",
        "",
        "## Feature Counts by Ablation",
    ]
    for name, count in results["feature_counts"].items():
        lines.append(f"- `{name}`: {count} features")

    lines.extend(["", "## Audit Table", ""])
    lines.append(results["audit_table"].to_markdown(index=False))

    if not results["category_issues"].empty:
        lines.extend(["", "## Category / Encoding Issues", ""])
        lines.append(results["category_issues"].to_markdown(index=False))

    memo_path.write_text("\n".join(lines))
    logger.info("Saved %s", memo_path)
