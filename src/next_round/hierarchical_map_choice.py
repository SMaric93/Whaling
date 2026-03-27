"""
Test 1: Hierarchical Map Choice.

Resolves the destination-choice contradiction by estimating at multiple levels:
  1. basin / theater
  2. major ground conditional on basin
  3. local ground conditional on major ground

Key question: Does agent matter more for broad theater choice while
captain matters more for conditional fine-grained choice?
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from src.next_round.config import (
    DATA_FINAL, DATA_DERIVED, OUTPUTS_TABLES, OUTPUTS_FIGURES,
    PSI_COL, THETA_COL, RANDOM_SEED, FIGURE_DPI, FIGURE_FORMAT,
)

logger = logging.getLogger(__name__)

ENV_FEATURES = ["year_out", "tonnage", "duration_days"]
CAPTAIN_FEATURES = [THETA_COL, "captain_voyage_num"]
AGENT_FEATURES = [PSI_COL]

ABLATION_LADDER = {
    "env_only": ENV_FEATURES,
    "env_captain": ENV_FEATURES + CAPTAIN_FEATURES,
    "env_agent": ENV_FEATURES + AGENT_FEATURES,
    "env_captain_agent": ENV_FEATURES + CAPTAIN_FEATURES + AGENT_FEATURES,
}


def run_hierarchical_map_choice(*, save_outputs: bool = True) -> Dict:
    """
    Estimate destination choice at basin → theater → major ground levels.

    Uses multinomial logit baselines + regularized alternatives.
    Standard ablation ladder at each level.
    """
    logger.info("=" * 60)
    logger.info("Test 1: Hierarchical Map Choice")
    logger.info("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────
    df = _load_data()
    results = {}

    # ── Estimate at each level ────────────────────────────────────────
    for level in ["basin", "theater", "major_ground"]:
        target_col = level
        if target_col not in df.columns:
            logger.warning("Missing level column: %s", target_col)
            continue

        # Drop unknown/rare
        valid = df[target_col].notna() & (df[target_col] != "Unknown")
        df_level = df[valid].copy()

        # Need at least 2 classes
        n_classes = df_level[target_col].nunique()
        if n_classes < 2:
            logger.warning("Only %d classes for %s, skipping", n_classes, level)
            continue

        logger.info("Level '%s': %d classes, %d observations",
                    level, n_classes, len(df_level))

        level_results = _estimate_level(df_level, target_col, level)
        results[level] = level_results

    # ── Build benchmark table ─────────────────────────────────────────
    all_rows = []
    for level, lr in results.items():
        for ablation, ar in lr.items():
            for row in ar:
                row["level"] = level
                row["ablation"] = ablation
                all_rows.append(row)

    benchmark = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()

    if save_outputs and not benchmark.empty:
        out_csv = OUTPUTS_TABLES / "hierarchical_map_benchmark.csv"
        benchmark.to_csv(out_csv, index=False)
        logger.info("Saved %s", out_csv)

        _plot_contribution(results)
        _save_memo(results, benchmark)

    return {"benchmark": benchmark, "level_results": results}


def _load_data() -> pd.DataFrame:
    """Load voyage data with ontology and AKM effects."""
    from src.reinforcement.data_builder import build_analysis_panel

    df = build_analysis_panel(require_akm=True, require_logbook=False)

    # Merge ontology
    ontology_path = DATA_DERIVED / "destination_ontology.parquet"
    if ontology_path.exists():
        ont = pd.read_parquet(ontology_path)
        ont_cols = ["ground_or_route", "basin", "theater", "major_ground", "ground_for_model"]
        available_cols = [c for c in ont_cols if c in ont.columns]
        df = df.merge(ont[available_cols].drop_duplicates(), on="ground_or_route", how="left")
    else:
        logger.warning("Ontology not found; using raw ground_or_route labels")
        df["basin"] = "Unknown"
        df["theater"] = df["ground_or_route"]
        df["major_ground"] = df["ground_or_route"]

    return df


def _estimate_level(df: pd.DataFrame, target_col: str, level_name: str) -> Dict:
    """Estimate multinomial logit at one hierarchical level."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder

    results = {}
    le = LabelEncoder()
    y_all = le.fit_transform(df[target_col].astype(str))
    n_classes = len(le.classes_)

    # Time-based split
    years = df["year_out"]
    p60, p80 = years.quantile([0.6, 0.8])
    train_mask = years <= p60
    val_mask = (years > p60) & (years <= p80)
    test_mask = years > p80

    for ablation_name, feature_list in ABLATION_LADDER.items():
        available = [f for f in feature_list if f in df.columns]
        if not available:
            continue

        X = df[available].fillna(0).values
        y = y_all

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_te, y_te = X[test_mask], y[test_mask]

        ablation_rows = []

        # Multinomial logit baseline
        try:
            from sklearn.metrics import log_loss, accuracy_score
            lr = LogisticRegression(
                max_iter=500, solver="lbfgs",
                multi_class="multinomial", random_state=RANDOM_SEED,
                C=1.0,
            )
            lr.fit(X_tr, y_tr)

            for split_name, Xs, ys in [("val", X_val, y_val), ("test", X_te, y_te)]:
                pred_proba = lr.predict_proba(Xs)
                pred = lr.predict(Xs)
                ablation_rows.append({
                    "model": "multinomial_logit",
                    "split": split_name,
                    "n_classes": n_classes,
                    "n_features": len(available),
                    "accuracy": accuracy_score(ys, pred),
                    "log_loss": log_loss(ys, pred_proba, labels=list(range(n_classes))),
                    "n_obs": len(ys),
                })
        except Exception as e:
            logger.warning("MNL failed for %s/%s: %s", level_name, ablation_name, e)

        # Regularized multinomial logit
        try:
            lr_reg = LogisticRegression(
                max_iter=500, solver="lbfgs",
                multi_class="multinomial", random_state=RANDOM_SEED,
                C=0.1, penalty="l2",
            )
            lr_reg.fit(X_tr, y_tr)

            for split_name, Xs, ys in [("val", X_val, y_val), ("test", X_te, y_te)]:
                pred_proba = lr_reg.predict_proba(Xs)
                pred = lr_reg.predict(Xs)
                ablation_rows.append({
                    "model": "regularized_mnl",
                    "split": split_name,
                    "n_classes": n_classes,
                    "n_features": len(available),
                    "accuracy": accuracy_score(ys, pred),
                    "log_loss": log_loss(ys, pred_proba, labels=list(range(n_classes))),
                    "n_obs": len(ys),
                })
        except Exception as e:
            logger.warning("Regularized MNL failed for %s/%s: %s", level_name, ablation_name, e)

        results[ablation_name] = ablation_rows

    return results


def _plot_contribution(results: Dict):
    """Plot captain vs agent contribution by decision level."""
    try:
        import matplotlib.pyplot as plt

        levels = []
        captain_gains = []
        agent_gains = []

        for level, lr in results.items():
            # Get test log_loss for env_only vs env_captain vs env_agent
            def _get_test_ll(ablation):
                if ablation in lr:
                    for r in lr[ablation]:
                        if r.get("split") == "test" and r.get("model") == "multinomial_logit":
                            return r.get("log_loss", np.nan)
                return np.nan

            base = _get_test_ll("env_only")
            with_captain = _get_test_ll("env_captain")
            with_agent = _get_test_ll("env_agent")

            if not np.isnan(base):
                levels.append(level)
                captain_gains.append(max(0, base - with_captain) if not np.isnan(with_captain) else 0)
                agent_gains.append(max(0, base - with_agent) if not np.isnan(with_agent) else 0)

        if levels:
            x = np.arange(len(levels))
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(x - 0.2, captain_gains, 0.35, label="Captain (θ)", color="#2196F3")
            ax.bar(x + 0.2, agent_gains, 0.35, label="Agent (ψ)", color="#FF5722")
            ax.set_xticks(x)
            ax.set_xticklabels(levels)
            ax.set_ylabel("Log-Loss Reduction vs Env-Only")
            ax.set_title("Captain vs Agent Contribution by Destination Level")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(OUTPUTS_FIGURES / f"hierarchical_map_contribution.{FIGURE_FORMAT}",
                       dpi=FIGURE_DPI, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved contribution figure")
    except ImportError:
        logger.warning("matplotlib not available for plotting")


def _save_memo(results: Dict, benchmark: pd.DataFrame):
    """Save identification memo."""
    memo_path = OUTPUTS_TABLES / "hierarchical_map_memo.md"
    lines = [
        "# Test 1: Hierarchical Map Ownership — Memo",
        "",
        "## What this test identifies",
        "Whether 'map ownership' is hierarchical: agents control broad theater",
        "selection (information advantage) while captains control fine-grained",
        "ground choice (experience advantage).",
        "",
        "## What this test does NOT identify",
        "- Cannot distinguish agent information from agent constraint/instruction",
        "- Cannot rule out that agents simply assign captains to theaters",
        "  (selection vs treatment)",
        "",
        "## Results Summary",
        "",
    ]
    for level in results:
        lines.append(f"### Level: {level}")
        level_rows = benchmark[benchmark["level"] == level]
        if not level_rows.empty:
            lines.append(level_rows.to_markdown(index=False))
        lines.append("")

    memo_path.write_text("\n".join(lines))
