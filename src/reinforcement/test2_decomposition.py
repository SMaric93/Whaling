"""
Test 2: Conditional Decomposition — Map vs Compass.

Decomposes captain and agent contributions to:
1. Destination choice ("map") — ground_or_route
2. Within-ground search behavior ("compass") — search metrics

Uses cross-validated prediction to measure how much each
identity contributes to each dimension.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from .config import CFG, COLS, TABLES_DIR, FIGURES_DIR
from .utils import make_table, make_figure, save_figure, write_memo

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Core Decomposition
# ═══════════════════════════════════════════════════════════════════════════

def run_test2(
    df: pd.DataFrame,
    *,
    search_outcomes: Optional[List[str]] = None,
    n_cv_folds: int = 5,
    save_outputs: bool = True,
) -> Dict:
    """
    Run Test 2: Map vs Compass conditional decomposition.

    For destination choice (classification):
    - Predict ground_or_route from captain_id, then agent_id, then both
    - Shapley-style decomposition

    For within-ground behavior (regression):
    - Predict search metrics from captain_id, then agent_id, then both
    - Measure R² contribution

    Parameters
    ----------
    df : pd.DataFrame
        Voyage panel with identifiers and search metrics.
    search_outcomes : list of str
        Within-ground search behavior variables.
    n_cv_folds : int
        Number of cross-validation folds.
    save_outputs : bool
        Save tables and figures.

    Returns
    -------
    dict with destination R², behavior R², and decomposition table.
    """
    if search_outcomes is None:
        search_outcomes = [
            c for c in [
                "straightness_index", "turning_concentration", "levy_mu",
                "revisit_rate", "route_efficiency", "avg_daily_distance_nm",
            ] if c in df.columns and df[c].notna().sum() > 100
        ]

    logger.info("Test 2: %d search outcomes", len(search_outcomes))

    # ── 1. Destination prediction ──────────────────────────────────────
    dest_results = _destination_decomposition(df, n_cv_folds)

    # ── 2. Search behavior prediction ──────────────────────────────────
    behavior_results = _behavior_decomposition(df, search_outcomes, n_cv_folds)

    # ── 3. Combine ─────────────────────────────────────────────────────
    decomp_table = _build_decomposition_table(dest_results, behavior_results)

    if save_outputs:
        _save_test2_outputs(decomp_table, dest_results, behavior_results)

    return {
        "destination": dest_results,
        "behavior": behavior_results,
        "decomposition_table": decomp_table,
        "status": "complete",
    }


def _destination_decomposition(df: pd.DataFrame, n_folds: int) -> Dict:
    """Predict destination choice from captain vs agent identity."""
    clean = df.dropna(subset=[COLS.ground_or_route, COLS.captain_id, COLS.agent_id])

    # Encode IDs as integers
    le_cap = LabelEncoder()
    le_agt = LabelEncoder()
    le_dest = LabelEncoder()

    X_captain = le_cap.fit_transform(clean[COLS.captain_id]).reshape(-1, 1)
    X_agent = le_agt.fit_transform(clean[COLS.agent_id]).reshape(-1, 1)
    X_both = np.hstack([X_captain, X_agent])
    y = le_dest.fit_transform(clean[COLS.ground_or_route])

    n_classes = len(le_dest.classes_)
    logger.info("Destination: %d classes, %d obs", n_classes, len(clean))

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=CFG.random_seed)

    results = {}
    for label, X in [
        ("captain_only", X_captain),
        ("agent_only", X_agent),
        ("both", X_both),
    ]:
        try:
            clf = RandomForestClassifier(
                n_estimators=100, max_depth=8, random_state=CFG.random_seed,
                n_jobs=-1,
            )
            scores = cross_val_score(clf, X, y, cv=kf, scoring="accuracy")
            results[label] = {
                "accuracy_mean": scores.mean(),
                "accuracy_std": scores.std(),
            }
            logger.info("Destination %s: accuracy=%.3f ± %.3f",
                        label, scores.mean(), scores.std())
        except Exception as e:
            logger.warning("Destination %s failed: %s", label, e)
            results[label] = {"accuracy_mean": np.nan, "accuracy_std": np.nan}

    # Shapley decomposition
    both_acc = results.get("both", {}).get("accuracy_mean", 0)
    cap_acc = results.get("captain_only", {}).get("accuracy_mean", 0)
    agt_acc = results.get("agent_only", {}).get("accuracy_mean", 0)

    # Marginal contributions
    results["captain_marginal"] = both_acc - agt_acc
    results["agent_marginal"] = both_acc - cap_acc
    results["shapley_captain"] = 0.5 * (cap_acc + (both_acc - agt_acc))
    results["shapley_agent"] = 0.5 * (agt_acc + (both_acc - cap_acc))

    return results


def _behavior_decomposition(
    df: pd.DataFrame,
    outcomes: List[str],
    n_folds: int,
) -> Dict:
    """Predict search behavior from captain vs agent identity."""
    results = {}

    for outcome in outcomes:
        clean = df.dropna(subset=[outcome, COLS.captain_id, COLS.agent_id])
        if len(clean) < 100:
            continue

        le_cap = LabelEncoder()
        le_agt = LabelEncoder()

        X_captain = le_cap.fit_transform(clean[COLS.captain_id]).reshape(-1, 1)
        X_agent = le_agt.fit_transform(clean[COLS.agent_id]).reshape(-1, 1)
        X_both = np.hstack([X_captain, X_agent])
        y = clean[outcome].values.astype(float)

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=CFG.random_seed)

        out_results = {}
        for label, X in [
            ("captain_only", X_captain),
            ("agent_only", X_agent),
            ("both", X_both),
        ]:
            try:
                reg = RandomForestRegressor(
                    n_estimators=100, max_depth=8, random_state=CFG.random_seed,
                    n_jobs=-1,
                )
                scores = cross_val_score(reg, X, y, cv=kf, scoring="r2")
                out_results[label] = {
                    "r2_mean": max(scores.mean(), 0),
                    "r2_std": scores.std(),
                }
            except Exception as e:
                logger.warning("Behavior %s/%s failed: %s", outcome, label, e)
                out_results[label] = {"r2_mean": np.nan, "r2_std": np.nan}

        # Attribution
        both_r2 = out_results.get("both", {}).get("r2_mean", 0)
        cap_r2 = out_results.get("captain_only", {}).get("r2_mean", 0)
        agt_r2 = out_results.get("agent_only", {}).get("r2_mean", 0)

        out_results["captain_marginal"] = both_r2 - agt_r2
        out_results["agent_marginal"] = both_r2 - cap_r2

        results[outcome] = out_results
        logger.info(
            "Behavior %s: cap_R²=%.3f, agt_R²=%.3f, both_R²=%.3f",
            outcome, cap_r2, agt_r2, both_r2,
        )

    return results


def _build_decomposition_table(dest_results, behavior_results) -> pd.DataFrame:
    """Build combined decomposition table."""
    rows = []

    # Destination row
    rows.append({
        "dimension": "Destination Choice (Map)",
        "metric": "Classification Accuracy",
        "captain_only": dest_results.get("captain_only", {}).get("accuracy_mean", np.nan),
        "agent_only": dest_results.get("agent_only", {}).get("accuracy_mean", np.nan),
        "both": dest_results.get("both", {}).get("accuracy_mean", np.nan),
        "captain_marginal": dest_results.get("captain_marginal", np.nan),
        "agent_marginal": dest_results.get("agent_marginal", np.nan),
    })

    # Behavior rows
    for outcome, bres in behavior_results.items():
        rows.append({
            "dimension": f"Search Behavior (Compass): {outcome}",
            "metric": "CV R²",
            "captain_only": bres.get("captain_only", {}).get("r2_mean", np.nan),
            "agent_only": bres.get("agent_only", {}).get("r2_mean", np.nan),
            "both": bres.get("both", {}).get("r2_mean", np.nan),
            "captain_marginal": bres.get("captain_marginal", np.nan),
            "agent_marginal": bres.get("agent_marginal", np.nan),
        })

    return pd.DataFrame(rows)


def _save_test2_outputs(decomp_table, dest_results, behavior_results):
    """Save Test 2 outputs."""
    decomp_table.to_csv(TABLES_DIR / "test2_decomposition.csv", index=False)

    # Summary figure
    try:
        fig, ax = make_figure("test2", "decomposition_summary", figsize=(10, 5))
        if len(decomp_table) > 0:
            x = np.arange(len(decomp_table))
            width = 0.3
            captain_vals = decomp_table["captain_marginal"].fillna(0).values
            agent_vals = decomp_table["agent_marginal"].fillna(0).values
            ax.barh(x - width/2, captain_vals, width, label="Captain (Map)",
                    color="#2196F3", alpha=0.8)
            ax.barh(x + width/2, agent_vals, width, label="Agent (Compass)",
                    color="#FF9800", alpha=0.8)
            ax.set_yticks(x)
            labels = decomp_table["dimension"].str[:40].values
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlabel("Marginal Contribution")
            ax.set_title("Map vs Compass Decomposition")
            ax.legend()
        save_figure(fig, "test2", "decomposition_summary")
    except Exception as e:
        logger.warning("Failed to save decomposition figure: %s", e)

    write_memo(
        "test2_decomposition",
        "## Map vs Compass Conditional Decomposition\n\n"
        "Cross-validated prediction of destination choice (map) and within-ground "
        "search behavior (compass) from captain and agent identities.\n\n"
        "If captains determine 'where to go' (the map), captain identity should "
        "predict destination better. If organizations shape 'how to search' "
        "(the compass), agent identity should predict search behavior better.\n",
        threats=[
            "GBM may overfit with high-cardinality IDs",
            "Only 1,449 voyages have logbook data for search metrics",
            "Ground classification is approximate",
        ],
    )
    logger.info("Test 2 outputs saved")
