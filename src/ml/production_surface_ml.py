"""
ML Layer — Phase ML-5: Flexible Production Surface.

Estimate a flexible response surface in captain skill (θ), organizational
capability (ψ), and scarcity. Compute marginal returns, cross-partials,
and submodularity tests.

Models: linear+interactions baseline, splines, RF, HistGBRegressor.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.ml.config import ML_CFG, ML_TABLES_DIR, ML_FIGURES_DIR, ML_MODELS_DIR
from src.ml.metrics import regression_metrics
from src.ml.splits import split_rolling_time
from src.ml.baselines import fit_linear_baseline

logger = logging.getLogger(__name__)

FEATURES = [
    "theta_hat_holdout", "psi_hat_holdout",
    "scarcity",
    "captain_voyage_num",
    "tonnage",
]


# ═══════════════════════════════════════════════════════════════════════════
# Production Surface Estimation
# ═══════════════════════════════════════════════════════════════════════════

def estimate_production_surface(
    df: pd.DataFrame = None,
    *,
    save_outputs: bool = True,
) -> Dict[str, Any]:
    """
    Estimate flexible production surface: outcome = f(θ, ψ, scarcity, controls).

    Returns
    -------
    Dict with benchmark table, prediction grids, and marginal effects.
    """
    t0 = time.time()
    logger.info("Estimating production surface...")

    if df is None:
        from src.ml.build_outcome_ml_dataset import build_outcome_ml_dataset
        df = build_outcome_ml_dataset()

    target_col = "log_q"
    if target_col not in df.columns:
        return {"error": "no_outcome_column"}

    features = [f for f in FEATURES if f in df.columns]
    df_valid = df.dropna(subset=[target_col] + features).copy()

    train_idx, val_idx, test_idx = split_rolling_time(df_valid)

    X = df_valid[features].fillna(0).values
    y = df_valid[target_col].values

    X_tr, y_tr = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_te, y_te = X[test_idx], y[test_idx]

    # ── Fit models ──────────────────────────────────────────────────
    models = {}

    # Linear baseline with interactions
    try:
        # Create interaction terms manually
        theta_idx = features.index("theta_hat_holdout") if "theta_hat_holdout" in features else None
        psi_idx = features.index("psi_hat_holdout") if "psi_hat_holdout" in features else None
        scar_idx = features.index("scarcity") if "scarcity" in features else None

        X_inter_tr = _add_interaction_features(X_tr, theta_idx, psi_idx, scar_idx)
        X_inter_val = _add_interaction_features(X_val, theta_idx, psi_idx, scar_idx)
        X_inter_te = _add_interaction_features(X_te, theta_idx, psi_idx, scar_idx)

        linear = fit_linear_baseline(X_inter_tr, y_tr)
        models["linear_interactions"] = {
            "model": linear,
            "val": regression_metrics(y_val, linear.predict(X_inter_val)),
            "test": regression_metrics(y_te, linear.predict(X_inter_te)),
        }
    except Exception as e:
        logger.warning("Linear interactions failed: %s", e)

    # Spline-augmented linear
    try:
        spline_cols = [i for i, f in enumerate(features) if f in ["theta_hat_holdout", "psi_hat_holdout", "scarcity"]]
        spline_model = fit_linear_baseline(X_tr, y_tr, with_splines=True, spline_cols=spline_cols)
        models["spline_linear"] = {
            "model": spline_model,
            "val": regression_metrics(y_val, spline_model.predict(X_val)),
            "test": regression_metrics(y_te, spline_model.predict(X_te)),
        }
    except Exception as e:
        logger.warning("Spline linear failed: %s", e)

    # Random forest
    try:
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(
            n_estimators=ML_CFG.rf_n_estimators,
            max_depth=ML_CFG.rf_max_depth,
            min_samples_leaf=ML_CFG.rf_min_samples_leaf,
            random_state=ML_CFG.random_seed,
            n_jobs=-1,
        )
        rf.fit(X_tr, y_tr)
        models["random_forest"] = {
            "model": rf,
            "val": regression_metrics(y_val, rf.predict(X_val)),
            "test": regression_metrics(y_te, rf.predict(X_te)),
        }
    except Exception as e:
        logger.warning("RF failed: %s", e)

    # HistGradientBoosting
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        hgb = HistGradientBoostingRegressor(
            max_iter=ML_CFG.n_estimators,
            max_depth=ML_CFG.max_depth,
            learning_rate=ML_CFG.learning_rate,
            min_samples_leaf=ML_CFG.min_samples_leaf,
            random_state=ML_CFG.random_seed,
        )
        hgb.fit(X_tr, y_tr)
        models["hist_gradient_boosting"] = {
            "model": hgb,
            "val": regression_metrics(y_val, hgb.predict(X_val)),
            "test": regression_metrics(y_te, hgb.predict(X_te)),
        }
    except Exception as e:
        logger.warning("HGB failed: %s", e)

    # ── Benchmark table ─────────────────────────────────────────────
    benchmark = _build_benchmark(models)

    # ── Select best model for surface analysis ──────────────────────
    best_name, best_info = max(
        models.items(),
        key=lambda x: x[1].get("val", {}).get("r_squared", -np.inf),
    )
    best_model = best_info["model"]
    logger.info("Best surface model: %s (val R²=%.4f)",
                best_name, best_info["val"]["r_squared"])

    # ── Prediction grids ────────────────────────────────────────────
    grids = _compute_prediction_grids(
        best_model, df_valid, features,
        theta_idx=theta_idx, psi_idx=psi_idx, scar_idx=scar_idx,
    )

    # ── Marginal returns ────────────────────────────────────────────
    marginals = _compute_marginal_returns(
        best_model, df_valid, features,
        theta_idx=theta_idx, psi_idx=psi_idx, scar_idx=scar_idx,
    )

    # ── Cross-partial (submodularity test) ──────────────────────────
    cross_partial = _compute_cross_partial(
        best_model, df_valid, features,
        theta_idx=theta_idx, psi_idx=psi_idx, scar_idx=scar_idx,
    )

    if save_outputs:
        benchmark.to_csv(ML_TABLES_DIR / "production_surface_benchmark.csv", index=False)
        _plot_surfaces(grids, save=True)
        _plot_marginal_returns(marginals, save=True)

    elapsed = time.time() - t0
    logger.info("Production surface estimation complete in %.1fs", elapsed)

    return {
        "benchmark": benchmark,
        "best_model_name": best_name,
        "grids": grids,
        "marginals": marginals,
        "cross_partial": cross_partial,
    }


def _add_interaction_features(X, theta_idx, psi_idx, scar_idx):
    """Add interaction columns to feature matrix."""
    extra = []
    if theta_idx is not None and psi_idx is not None:
        extra.append(X[:, theta_idx] * X[:, psi_idx])
    if theta_idx is not None and scar_idx is not None:
        extra.append(X[:, theta_idx] * X[:, scar_idx])
    if psi_idx is not None and scar_idx is not None:
        extra.append(X[:, psi_idx] * X[:, scar_idx])
    if theta_idx is not None and psi_idx is not None and scar_idx is not None:
        extra.append(X[:, theta_idx] * X[:, psi_idx] * X[:, scar_idx])
    if extra:
        return np.column_stack([X] + extra)
    return X


def _build_benchmark(models: Dict) -> pd.DataFrame:
    """Build benchmark DataFrame."""
    rows = []
    for name, info in models.items():
        row = {"model": name}
        for split in ["val", "test"]:
            if split in info:
                for k, v in info[split].items():
                    row[f"{split}_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def _compute_prediction_grids(model, df, features, *, theta_idx, psi_idx, scar_idx):
    """Compute prediction grids over θ, ψ, and scarcity percentiles."""
    grids = {}
    n_grid = 20

    X_median = df[features].fillna(0).median().values

    for scarcity_label, scarcity_pctile in [("sparse", 25), ("medium", 50), ("rich", 75)]:
        theta_range = np.linspace(
            df.iloc[:, theta_idx if theta_idx is not None else 0].quantile(0.05) if theta_idx is not None else 0,
            df.iloc[:, theta_idx if theta_idx is not None else 0].quantile(0.95) if theta_idx is not None else 1,
            n_grid,
        ) if theta_idx is not None else np.linspace(0, 1, n_grid)

        psi_range = np.linspace(
            df[features[psi_idx]].quantile(0.05) if psi_idx is not None else 0,
            df[features[psi_idx]].quantile(0.95) if psi_idx is not None else 1,
            n_grid,
        ) if psi_idx is not None else np.linspace(0, 1, n_grid)

        pred_grid = np.zeros((n_grid, n_grid))

        for i, theta_val in enumerate(theta_range):
            for j, psi_val in enumerate(psi_range):
                x = X_median.copy()
                if theta_idx is not None:
                    x[theta_idx] = theta_val
                if psi_idx is not None:
                    x[psi_idx] = psi_val
                if scar_idx is not None:
                    x[scar_idx] = df[features[scar_idx]].quantile(scarcity_pctile / 100)
                pred_grid[i, j] = model.predict(x.reshape(1, -1))[0]

        grids[scarcity_label] = {
            "theta_range": theta_range,
            "psi_range": psi_range,
            "predictions": pred_grid,
        }

    return grids


def _compute_marginal_returns(model, df, features, *, theta_idx, psi_idx, scar_idx):
    """Compute marginal return to ψ at different θ values."""
    if theta_idx is None or psi_idx is None:
        return {}

    n_grid = 20
    epsilon = df[features[psi_idx]].std() * 0.01

    theta_range = np.linspace(
        df[features[theta_idx]].quantile(0.10),
        df[features[theta_idx]].quantile(0.90),
        n_grid,
    )

    X_median = df[features].fillna(0).median().values

    marginal_psi_at_theta = []
    for theta_val in theta_range:
        x_lo = X_median.copy()
        x_hi = X_median.copy()
        x_lo[theta_idx] = theta_val
        x_hi[theta_idx] = theta_val
        x_lo[psi_idx] = df[features[psi_idx]].quantile(0.25)
        x_hi[psi_idx] = df[features[psi_idx]].quantile(0.75)

        pred_lo = model.predict(x_lo.reshape(1, -1))[0]
        pred_hi = model.predict(x_hi.reshape(1, -1))[0]
        marginal = pred_hi - pred_lo
        marginal_psi_at_theta.append(marginal)

    return {
        "theta_range": theta_range,
        "marginal_psi": np.array(marginal_psi_at_theta),
    }


def _compute_cross_partial(model, df, features, *, theta_idx, psi_idx, scar_idx):
    """Compute cross-partial: does marginal value of ψ decline with θ?"""
    if theta_idx is None or psi_idx is None:
        return {}

    X_median = df[features].fillna(0).median().values
    eps_t = df[features[theta_idx]].std() * 0.05
    eps_p = df[features[psi_idx]].std() * 0.05

    results = {}
    for scar_label, scar_pctile in [("sparse", 25), ("medium", 50), ("rich", 75)]:
        x_base = X_median.copy()
        if scar_idx is not None:
            x_base[scar_idx] = df[features[scar_idx]].quantile(scar_pctile / 100)

        # f(θ+ε, ψ+ε) - f(θ+ε, ψ) - f(θ, ψ+ε) + f(θ, ψ)
        def _pred(t_offset, p_offset):
            x = x_base.copy()
            x[theta_idx] += t_offset
            x[psi_idx] += p_offset
            return model.predict(x.reshape(1, -1))[0]

        cross_partial_val = (
            _pred(eps_t, eps_p) - _pred(eps_t, 0) - _pred(0, eps_p) + _pred(0, 0)
        ) / (eps_t * eps_p)

        results[scar_label] = {
            "cross_partial": float(cross_partial_val),
            "submodular": cross_partial_val < 0,
        }

    return results


def _plot_surfaces(grids, *, save=False):
    """Plot production surface heatmaps."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    n_grids = len(grids)
    fig, axes = plt.subplots(1, n_grids, figsize=(6 * n_grids, 5), squeeze=False)

    for idx, (label, grid) in enumerate(grids.items()):
        ax = axes[0, idx]
        im = ax.imshow(
            grid["predictions"], cmap="viridis", origin="lower", aspect="auto",
        )
        ax.set_xlabel("ψ percentile")
        ax.set_ylabel("θ percentile")
        ax.set_title(f"Production Surface ({label})")
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Learned Production Surface by Scarcity", fontsize=13)
    fig.tight_layout()

    if save:
        path = ML_FIGURES_DIR / f"production_surface_heatmaps.{ML_CFG.figure_format}"
        fig.savefig(path, dpi=ML_CFG.figure_dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.close(fig)


def _plot_marginal_returns(marginals, *, save=False):
    """Plot marginal return to ψ by θ."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not marginals:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(marginals["theta_range"], marginals["marginal_psi"], "o-", lw=2, color="#2ecc71")
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_xlabel("Captain Skill (θ)")
    ax.set_ylabel("Marginal Return to ψ (Q75 - Q25)")
    ax.set_title("Marginal Return to Agent Capability by Captain Skill")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save:
        path = ML_FIGURES_DIR / f"marginal_return_psi_by_theta.{ML_CFG.figure_format}"
        fig.savefig(path, dpi=ML_CFG.figure_dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Main Entry
# ═══════════════════════════════════════════════════════════════════════════

def run_production_surface_ml(*, save_outputs: bool = True) -> Dict[str, Any]:
    """Run the full production surface pipeline."""
    logger.info("=" * 60)
    logger.info("Phase ML-5: Production Surface Estimation")
    logger.info("=" * 60)
    return estimate_production_surface(save_outputs=save_outputs)
