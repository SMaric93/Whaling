"""
ML Layer — Phase ML-4: Distributional and Heterogeneous Effects (Floor-Raising).

Shows that high-capability organizations mainly improve downside outcomes
for novices and other high-variance captains.

Tasks:
A. Predict lower-tail outcomes (bottom decile, expected shortfall)
B. Heterogeneity discovery by experience, volatility, scarcity
C. Distributional outputs (conditional quantiles)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.ml.config import ML_CFG, ML_TABLES_DIR, ML_FIGURES_DIR
from src.ml.metrics import classification_metrics, regression_metrics
from src.ml.splits import split_rolling_time
from src.ml.baselines import fit_logistic_baseline, fit_linear_baseline

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Feature Definitions
# ═══════════════════════════════════════════════════════════════════════════

FEATURES = [
    "theta_hat_holdout", "psi_hat_holdout",
    "captain_voyage_num", "scarcity",
    "tonnage",
]


# ═══════════════════════════════════════════════════════════════════════════
# Part A: Lower-Tail Prediction
# ═══════════════════════════════════════════════════════════════════════════

def predict_lower_tail(
    df: pd.DataFrame,
    *,
    save_outputs: bool = True,
) -> Dict[str, Any]:
    """
    Predict lower-tail outcome indicators.

    Targets: bottom_decile, bottom_5pct.
    """
    t0 = time.time()
    logger.info("Predicting lower-tail outcomes...")

    features = [f for f in FEATURES if f in df.columns]
    if not features:
        return {"error": "no_features_available"}

    results = {}

    for target in ["bottom_decile", "bottom_5pct"]:
        if target not in df.columns:
            continue

        df_valid = df.dropna(subset=[target] + features).copy()
        df_valid[target] = df_valid[target].astype(int)

        if df_valid[target].nunique() < 2:
            continue

        train_idx, val_idx, test_idx = split_rolling_time(df_valid)

        X = df_valid[features].fillna(0).values
        y = df_valid[target].values

        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_te, y_te = X[test_idx], y[test_idx]

        target_results = {}

        # Logistic baseline
        try:
            logit = fit_logistic_baseline(X_tr, y_tr)
            target_results["logistic"] = _eval_clf(logit, X_val, y_val, X_te, y_te)
        except Exception as e:
            logger.warning("Logistic failed for %s: %s", target, e)

        # Gradient boosting
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
            hgb = HistGradientBoostingClassifier(
                max_iter=ML_CFG.n_estimators, max_depth=ML_CFG.max_depth,
                learning_rate=ML_CFG.learning_rate,
                min_samples_leaf=ML_CFG.min_samples_leaf,
                random_state=ML_CFG.random_seed,
            )
            hgb.fit(X_tr, y_tr)
            target_results["hist_gbt"] = _eval_clf(hgb, X_val, y_val, X_te, y_te)
        except Exception as e:
            logger.warning("HGB failed for %s: %s", target, e)

        results[target] = target_results

    if save_outputs and results:
        rows = []
        for target, mods in results.items():
            for model_name, metrics in mods.items():
                rows.append({"target": target, "model": model_name, **metrics})
        pd.DataFrame(rows).to_csv(
            ML_TABLES_DIR / "lower_tail_benchmark.csv", index=False
        )

    elapsed = time.time() - t0
    logger.info("Lower-tail prediction complete in %.1fs", elapsed)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Part B: Heterogeneity Discovery
# ═══════════════════════════════════════════════════════════════════════════

def discover_heterogeneity(
    df: pd.DataFrame,
    *,
    save_outputs: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Estimate how floor-raising effects vary by subgroups.

    Subgroups: experience, prior volatility, scarcity.
    """
    logger.info("Discovering heterogeneity in downside-risk effects...")

    from src.ml.interpret import standard_subgroup_labels, subgroup_performance

    features = [f for f in FEATURES if f in df.columns]
    target_col = "log_q"

    if target_col not in df.columns:
        return {"error": "no_outcome_column"}

    df_valid = df.dropna(subset=[target_col] + features).copy()

    # Fit best model on full sample for subgroup analysis
    from sklearn.ensemble import HistGradientBoostingRegressor

    train_idx, val_idx, test_idx = split_rolling_time(df_valid)

    X = df_valid[features].fillna(0).values
    y = df_valid[target_col].values

    hgb = HistGradientBoostingRegressor(
        max_iter=ML_CFG.n_estimators, max_depth=ML_CFG.max_depth,
        learning_rate=ML_CFG.learning_rate,
        min_samples_leaf=ML_CFG.min_samples_leaf,
        random_state=ML_CFG.random_seed,
    )
    hgb.fit(X[train_idx], y[train_idx])

    y_pred = hgb.predict(X)

    # Subgroup performance
    subgroup_labels = standard_subgroup_labels(df_valid)
    subgroup_tables = {}

    for label_name, labels in subgroup_labels.items():
        labels = labels.reindex(df_valid.index)
        valid = labels.notna()
        if valid.sum() < 50:
            continue

        perf = subgroup_performance(
            y[valid], y_pred[valid], labels[valid], task="regression"
        )
        subgroup_tables[label_name] = perf

    if save_outputs:
        for name, table in subgroup_tables.items():
            table.to_csv(ML_TABLES_DIR / f"heterogeneity_{name}.csv")

    return subgroup_tables


# ═══════════════════════════════════════════════════════════════════════════
# Part C: Distributional / Quantile Effects
# ═══════════════════════════════════════════════════════════════════════════

def estimate_quantile_effects(
    df: pd.DataFrame,
    *,
    quantiles: List[float] = None,
    save_outputs: bool = True,
) -> Dict[str, Any]:
    """
    Estimate conditional quantiles under low vs high psi.

    Uses quantile regression via sklearn (HistGBRegressor with quantile loss)
    or gradient boosting with quantile loss.
    """
    quantiles = quantiles or [0.05, 0.10, 0.25, 0.50, 0.75, 0.90]

    features = [f for f in FEATURES if f in df.columns]
    target_col = "log_q"

    if target_col not in df.columns or "psi_hat_holdout" not in df.columns:
        return {"error": "missing_columns"}

    df_valid = df.dropna(subset=[target_col] + features).copy()
    train_idx, val_idx, test_idx = split_rolling_time(df_valid)

    X = df_valid[features].fillna(0).values
    y = df_valid[target_col].values

    # ── Fit quantile models ────────────────────────────────────────
    quantile_preds = {}
    from src.ml.metrics import pinball_loss

    for q in quantiles:
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor
            qmodel = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=q,
                max_iter=ML_CFG.n_estimators,
                max_depth=ML_CFG.max_depth,
                learning_rate=ML_CFG.learning_rate,
                min_samples_leaf=ML_CFG.min_samples_leaf,
                random_state=ML_CFG.random_seed,
            )
            qmodel.fit(X[train_idx], y[train_idx])
            preds = qmodel.predict(X[test_idx])
            pl = pinball_loss(y[test_idx], preds, q)

            quantile_preds[q] = {
                "predictions": preds,
                "pinball_loss": pl,
                "model": qmodel,
            }
            logger.info("Quantile %.2f: pinball_loss=%.4f", q, pl)
        except Exception as e:
            logger.warning("Quantile regression failed for q=%.2f: %s", q, e)

    # ── Compare quantiles by psi group ──────────────────────────────
    psi_col = "psi_hat_holdout"
    psi_test = df_valid.iloc[test_idx][psi_col]
    psi_median = psi_test.median()

    comparison = {}
    for q, qr in quantile_preds.items():
        preds = qr["predictions"]
        low_psi = preds[psi_test.values <= psi_median]
        high_psi = preds[psi_test.values > psi_median]

        comparison[q] = {
            "low_psi_mean": float(np.mean(low_psi)) if len(low_psi) > 0 else np.nan,
            "high_psi_mean": float(np.mean(high_psi)) if len(high_psi) > 0 else np.nan,
            "diff": float(np.mean(high_psi) - np.mean(low_psi)) if len(low_psi) > 0 and len(high_psi) > 0 else np.nan,
        }

    # ── Plot ────────────────────────────────────────────────────────
    if save_outputs and comparison:
        _plot_quantile_comparison(comparison, quantiles)
        _plot_downside_risk_by_experience(df_valid, quantile_preds, test_idx)

    results = {
        "quantile_pinball_losses": {q: qr["pinball_loss"] for q, qr in quantile_preds.items()},
        "psi_comparison": comparison,
    }

    if save_outputs:
        pd.DataFrame(comparison).T.to_csv(
            ML_TABLES_DIR / "quantile_psi_comparison.csv"
        )

    return results


def _plot_quantile_comparison(comparison: Dict, quantiles: List[float]):
    """Plot conditional quantile comparison by psi."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    qs = sorted(comparison.keys())
    low_vals = [comparison[q]["low_psi_mean"] for q in qs]
    high_vals = [comparison[q]["high_psi_mean"] for q in qs]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(qs, low_vals, "o-", lw=2, color="#e74c3c", label="Low ψ (below median)")
    ax.plot(qs, high_vals, "s-", lw=2, color="#2ecc71", label="High ψ (above median)")
    ax.fill_between(qs, low_vals, high_vals, alpha=0.1, color="#3498db")
    ax.set_xlabel("Quantile")
    ax.set_ylabel("Predicted Output (log)")
    ax.set_title("Conditional Quantiles: Low vs High Agent Capability")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = ML_FIGURES_DIR / f"quantile_psi_comparison.{ML_CFG.figure_format}"
    fig.savefig(path, dpi=ML_CFG.figure_dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_downside_risk_by_experience(df, quantile_preds, test_idx):
    """Plot novice vs expert downside risk by psi."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if "novice" not in df.columns or "psi_hat_holdout" not in df.columns:
        return

    q10_model = quantile_preds.get(0.10, quantile_preds.get(0.05))
    if q10_model is None:
        return

    test_df = df.iloc[test_idx].copy()
    features = [f for f in FEATURES if f in df.columns]

    psi_grid = np.linspace(
        test_df["psi_hat_holdout"].quantile(0.05),
        test_df["psi_hat_holdout"].quantile(0.95),
        30,
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    for exp_label, exp_val in [("Novice", 1), ("Expert", 0)]:
        sub = test_df[test_df["novice"] == exp_val]
        if len(sub) < 30:
            continue

        X_median = sub[features].fillna(0).median().values.reshape(1, -1)
        psi_idx = features.index("psi_hat_holdout") if "psi_hat_holdout" in features else None
        if psi_idx is None:
            continue

        downside_vals = []
        for psi_val in psi_grid:
            X_mod = np.tile(X_median, (1, 1))
            X_mod[0, psi_idx] = psi_val
            pred = q10_model["model"].predict(X_mod)[0]
            downside_vals.append(pred)

        ax.plot(psi_grid, downside_vals, lw=2, label=exp_label)

    ax.set_xlabel("Agent Capability (ψ)")
    ax.set_ylabel("Predicted 10th Percentile Output")
    ax.set_title("Downside Risk by Experience and Agent Capability")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = ML_FIGURES_DIR / f"downside_risk_novice_expert.{ML_CFG.figure_format}"
    fig.savefig(path, dpi=ML_CFG.figure_dpi, bbox_inches="tight")
    plt.close(fig)


def _eval_clf(model, X_val, y_val, X_te, y_te) -> Dict:
    """Quick classifier evaluation."""
    result = {}
    for sname, X_e, y_e in [("val", X_val, y_val), ("test", X_te, y_te)]:
        y_pred = model.predict(X_e)
        y_prob = model.predict_proba(X_e) if hasattr(model, "predict_proba") else None
        if y_prob is not None:
            m = classification_metrics(y_e, y_prob, y_pred)
        else:
            m = {"accuracy": float(np.mean(y_pred == y_e))}
        for k, v in m.items():
            result[f"{sname}_{k}"] = v
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Main Entry
# ═══════════════════════════════════════════════════════════════════════════

def run_heterogeneity_ml(*, save_outputs: bool = True) -> Dict[str, Any]:
    """Run the full heterogeneity/floor-raising pipeline."""
    logger.info("=" * 60)
    logger.info("Phase ML-4: Heterogeneous & Distributional Effects")
    logger.info("=" * 60)

    from src.ml.build_outcome_ml_dataset import build_outcome_ml_dataset
    df = build_outcome_ml_dataset()

    lower_tail = predict_lower_tail(df, save_outputs=save_outputs)
    heterogeneity = discover_heterogeneity(df, save_outputs=save_outputs)
    quantiles = estimate_quantile_effects(df, save_outputs=save_outputs)

    return {
        "lower_tail": lower_tail,
        "heterogeneity": heterogeneity,
        "quantile_effects": quantiles,
    }
