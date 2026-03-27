"""
ML Layer — Phase ML-5b: Assignment Optimizer.

Use the learned production surface to solve realistic assignment counterfactuals.

Assignment rules:
1. Observed matching
2. Positive assortative matching (PAM)
3. Anti-assortative matching (AAM)
4. Constrained optimal matching
5. Robust matching under prediction uncertainty
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.ml.config import ML_CFG, ML_TABLES_DIR, ML_FIGURES_DIR

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Assignment Optimization
# ═══════════════════════════════════════════════════════════════════════════

def solve_assignment(
    predict_fn,
    captains: pd.DataFrame,
    agents: pd.DataFrame,
    *,
    controls: Dict[str, float] = None,
    feature_names: List[str] = None,
    save_outputs: bool = True,
) -> Dict[str, Any]:
    """
    Solve assignment counterfactuals using learned production surface.

    Parameters
    ----------
    predict_fn : callable
        f(X) -> predicted outcomes. X is an array with columns matching feature_names.
    captains : pd.DataFrame
        Captain characteristics (theta_hat_holdout, captain_voyage_num, etc.)
    agents : pd.DataFrame
        Agent characteristics (psi_hat_holdout, etc.)
    controls : dict
        Control values to hold fixed (e.g., tonnage, scarcity).
    feature_names : list
        Feature names expected by predict_fn.
    """
    t0 = time.time()
    logger.info("Solving assignment counterfactuals...")

    n_captains = len(captains)
    n_agents = len(agents)
    logger.info("Assignment: %d captains × %d agents", n_captains, n_agents)

    if feature_names is None:
        feature_names = ["theta_hat_holdout", "psi_hat_holdout", "scarcity",
                        "captain_voyage_num", "tonnage"]

    controls = controls or {}

    # ── Build payoff matrix ─────────────────────────────────────────
    payoff_matrix = np.zeros((n_captains, n_agents))

    theta_col = "theta_hat_holdout"
    psi_col = "psi_hat_holdout"

    theta_vals = captains[theta_col].values if theta_col in captains.columns else np.zeros(n_captains)
    psi_vals = agents[psi_col].values if psi_col in agents.columns else np.zeros(n_agents)

    for i in range(n_captains):
        X_row = np.zeros((n_agents, len(feature_names)))
        for j in range(n_agents):
            x = {}
            # Captain features
            for f in feature_names:
                if f in captains.columns:
                    x[f] = captains.iloc[i][f]
                elif f in agents.columns:
                    x[f] = agents.iloc[j][f]
                elif f in controls:
                    x[f] = controls[f]
                else:
                    x[f] = 0

            X_row[j] = [x.get(f, 0) for f in feature_names]

        payoff_matrix[i] = predict_fn(X_row)

    # ── Assignment rules ────────────────────────────────────────────
    results = {}

    # 1. Observed (diagonal-ish, or random)
    n_match = min(n_captains, n_agents)
    observed_idx = np.arange(n_match)
    results["observed"] = _welfare(payoff_matrix, observed_idx, observed_idx)

    # 2. PAM: sort both by type, match rank-to-rank
    theta_order = np.argsort(theta_vals)[::-1]  # high to low
    psi_order = np.argsort(psi_vals)[::-1]
    pam_c = theta_order[:n_match]
    pam_a = psi_order[:n_match]
    results["pam"] = _welfare(payoff_matrix, pam_c, pam_a)

    # 3. AAM: reverse one ordering
    aam_a = psi_order[::-1][:n_match]
    results["aam"] = _welfare(payoff_matrix, pam_c, aam_a)

    # 4. Constrained optimal (linear sum assignment)
    try:
        from scipy.optimize import linear_sum_assignment
        # Minimize negative payoff = maximize payoff
        cost = -payoff_matrix[:n_match, :n_match]
        row_ind, col_ind = linear_sum_assignment(cost)
        results["constrained_optimal"] = _welfare(payoff_matrix, row_ind, col_ind)
    except ImportError:
        logger.warning("scipy not available; skipping optimal assignment")
    except Exception as e:
        logger.warning("Optimal assignment failed: %s", e)

    # 5. Robust optimal (with noise in predictions)
    try:
        from scipy.optimize import linear_sum_assignment
        rng = np.random.RandomState(ML_CFG.random_seed)
        n_bootstrap = 50
        robust_assignments = []

        for b in range(n_bootstrap):
            noise = rng.normal(0, payoff_matrix.std() * 0.1, payoff_matrix[:n_match, :n_match].shape)
            cost_b = -(payoff_matrix[:n_match, :n_match] + noise)
            row_b, col_b = linear_sum_assignment(cost_b)
            robust_assignments.append((row_b, col_b))

        # Use most frequent assignment as robust (mode)
        # Simplification: use mean payoff across bootstraps
        mean_payoffs = []
        for row_b, col_b in robust_assignments:
            mp = np.mean(payoff_matrix[row_b, col_b])
            mean_payoffs.append(mp)

        best_b = np.argmax(mean_payoffs)
        row_rob, col_rob = robust_assignments[best_b]
        results["robust_optimal"] = _welfare(payoff_matrix, row_rob, col_rob)
    except Exception as e:
        logger.warning("Robust assignment failed: %s", e)

    # ── Welfare table ───────────────────────────────────────────────
    welfare_table = pd.DataFrame(results).T
    welfare_table.index.name = "assignment_rule"

    if save_outputs:
        welfare_table.to_csv(ML_TABLES_DIR / "assignment_welfare.csv")
        _plot_assignment_gains(results, save=True)

    elapsed = time.time() - t0
    logger.info("Assignment optimization complete in %.1fs", elapsed)

    return {
        "welfare_table": welfare_table,
        "payoff_matrix": payoff_matrix,
        "results": results,
    }


def _welfare(payoff_matrix, captain_idx, agent_idx) -> Dict[str, float]:
    """Compute welfare metrics for an assignment."""
    n = min(len(captain_idx), len(agent_idx))
    payoffs = np.array([payoff_matrix[captain_idx[i], agent_idx[i]] for i in range(n)])

    return {
        "aggregate": float(np.mean(payoffs)),
        "bottom_10pct": float(np.percentile(payoffs, 10)),
        "std": float(np.std(payoffs)),
        "n_assigned": int(n),
    }


def _plot_assignment_gains(results, *, save=False):
    """Plot gains/losses under alternative assignment rules."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    rules = list(results.keys())
    agg = [results[r]["aggregate"] for r in rules]
    bottom = [results[r]["bottom_10pct"] for r in rules]

    x = np.arange(len(rules))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, agg, width, label="Aggregate Mean", color="#3498db", alpha=0.8)
    ax.bar(x + width / 2, bottom, width, label="Bottom 10%", color="#e74c3c", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([r.replace("_", " ").title() for r in rules], rotation=15)
    ax.set_ylabel("Predicted Output")
    ax.set_title("Welfare Under Alternative Matching Rules")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    for i in range(len(rules)):
        ax.text(i - width / 2, agg[i], f"{agg[i]:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + width / 2, bottom[i], f"{bottom[i]:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()

    if save:
        path = ML_FIGURES_DIR / f"assignment_welfare.{ML_CFG.figure_format}"
        fig.savefig(path, dpi=ML_CFG.figure_dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: Run from production surface
# ═══════════════════════════════════════════════════════════════════════════

def run_assignment_from_surface(
    surface_results: Dict[str, Any] = None,
    *,
    save_outputs: bool = True,
) -> Dict[str, Any]:
    """
    Run assignment optimizer using the best model from production surface estimation.
    """
    logger.info("=" * 60)
    logger.info("Phase ML-5b: Assignment Optimizer")
    logger.info("=" * 60)

    if surface_results is None:
        from src.ml.production_surface_ml import estimate_production_surface
        surface_results = estimate_production_surface(save_outputs=False)

    # Get best model
    best_name = surface_results.get("best_model_name", "")
    benchmark = surface_results.get("benchmark")

    # Load data for captain/agent characteristics
    from src.ml.build_outcome_ml_dataset import build_outcome_ml_dataset
    df = build_outcome_ml_dataset()

    # Extract unique captains and agents
    captains = df.groupby("captain_id").agg({
        "theta_hat_holdout": "mean",
        "captain_voyage_num": "max",
    }).reset_index() if "captain_id" in df.columns else pd.DataFrame()

    agents = df.groupby("agent_id").agg({
        "psi_hat_holdout": "mean",
    }).reset_index() if "agent_id" in df.columns else pd.DataFrame()

    if len(captains) == 0 or len(agents) == 0:
        return {"error": "insufficient_captain_agent_data"}

    # Use production surface best model for predictions
    # Find the fitted model
    models = surface_results.get("models", {})
    predict_fn = None
    feature_names = ["theta_hat_holdout", "psi_hat_holdout", "scarcity",
                    "captain_voyage_num", "tonnage"]

    # Try to get the HGB model
    for model_key in ["hist_gradient_boosting", "random_forest", "spline_linear"]:
        if model_key in models and "model" in models[model_key]:
            model = models[model_key]["model"]
            predict_fn = model.predict
            break

    if predict_fn is None:
        # Fallback: build a quick model
        from sklearn.ensemble import HistGradientBoostingRegressor
        avail_features = [f for f in feature_names if f in df.columns]
        X = df[avail_features].fillna(0).values
        y = df["log_q"].values if "log_q" in df.columns else np.zeros(len(df))
        model = HistGradientBoostingRegressor(max_iter=200, random_state=ML_CFG.random_seed)
        model.fit(X, y)
        predict_fn = model.predict
        feature_names = avail_features

    return solve_assignment(
        predict_fn,
        captains,
        agents,
        controls={"scarcity": float(df["scarcity"].median()) if "scarcity" in df.columns else 0,
                  "tonnage": float(df["tonnage"].median()) if "tonnage" in df.columns else 0},
        feature_names=feature_names,
        save_outputs=save_outputs,
    )
