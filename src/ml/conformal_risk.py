"""
ML Layer — Appendix ML-11: Conformal Prediction Intervals.

Constructs calibrated prediction intervals using conformal prediction:
- Split conformal (regression)
- Conformalized quantile regression

Used to assess uncertainty around production surface and assignment
counterfactuals.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.ml.config import ML_CFG, ML_TABLES_DIR

logger = logging.getLogger(__name__)


def conformal_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    alpha: float = 0.10,
) -> Dict[str, float]:
    """
    Split conformal prediction intervals.

    Parameters
    ----------
    y_true : calibration set true values
    y_pred : calibration set predictions
    alpha : miscoverage rate (default 0.10 → 90% intervals)

    Returns
    -------
    Dict with q_hat (conformal quantile), coverage, avg_width.
    """
    residuals = np.abs(y_true - y_pred)
    n = len(residuals)

    # Conformal quantile
    q_hat = np.quantile(residuals, np.ceil((1 - alpha) * (n + 1)) / n)

    # Coverage (on calibration set itself, for diagnostics)
    lower = y_pred - q_hat
    upper = y_pred + q_hat
    coverage = np.mean((y_true >= lower) & (y_true <= upper))
    avg_width = 2 * q_hat

    return {
        "q_hat": float(q_hat),
        "coverage": float(coverage),
        "avg_width": float(avg_width),
        "alpha": alpha,
        "n_calibration": n,
    }


def conformalized_quantile_regression(
    y_true_cal: np.ndarray,
    lower_pred_cal: np.ndarray,
    upper_pred_cal: np.ndarray,
    lower_pred_test: np.ndarray,
    upper_pred_test: np.ndarray,
    y_true_test: np.ndarray = None,
    *,
    alpha: float = 0.10,
) -> Dict[str, Any]:
    """
    Conformalized quantile regression (CQR).

    Uses calibration set to adjust quantile predictions for finite-sample coverage.

    Parameters
    ----------
    y_true_cal : true values on calibration set
    lower_pred_cal, upper_pred_cal : quantile predictions on calibration set
    lower_pred_test, upper_pred_test : quantile predictions on test set
    """
    # Conformity scores on calibration set
    scores = np.maximum(lower_pred_cal - y_true_cal, y_true_cal - upper_pred_cal)
    n = len(scores)

    # Conformal adjustment
    q_hat = np.quantile(scores, np.ceil((1 - alpha) * (n + 1)) / n)

    # Adjusted intervals on test set
    adjusted_lower = lower_pred_test - q_hat
    adjusted_upper = upper_pred_test + q_hat

    result = {
        "q_hat": float(q_hat),
        "avg_width": float(np.mean(adjusted_upper - adjusted_lower)),
        "adjusted_lower": adjusted_lower,
        "adjusted_upper": adjusted_upper,
    }

    if y_true_test is not None:
        coverage = np.mean((y_true_test >= adjusted_lower) & (y_true_test <= adjusted_upper))
        result["test_coverage"] = float(coverage)
        result["n_test"] = len(y_true_test)

    return result


def run_conformal_analysis(
    *,
    save_outputs: bool = True,
) -> Dict[str, Any]:
    """
    Run conformal analysis on production surface predictions.
    """
    t0 = time.time()
    logger.info("Running conformal prediction analysis...")

    from src.ml.build_outcome_ml_dataset import build_outcome_ml_dataset
    from src.ml.splits import split_rolling_time

    df = build_outcome_ml_dataset()
    target_col = "log_q"

    if target_col not in df.columns:
        return {"error": "no_outcome_column"}

    features = [f for f in ["theta_hat_holdout", "psi_hat_holdout", "scarcity",
                           "captain_voyage_num", "tonnage"] if f in df.columns]
    df_valid = df.dropna(subset=[target_col] + features).copy()

    train_idx, val_idx, test_idx = split_rolling_time(df_valid)

    X = df_valid[features].fillna(0).values
    y = df_valid[target_col].values

    # Fit model
    from sklearn.ensemble import HistGradientBoostingRegressor
    model = HistGradientBoostingRegressor(
        max_iter=ML_CFG.n_estimators, max_depth=ML_CFG.max_depth,
        learning_rate=ML_CFG.learning_rate,
        random_state=ML_CFG.random_seed,
    )
    model.fit(X[train_idx], y[train_idx])

    # Split conformal on validation set
    y_pred_cal = model.predict(X[val_idx])
    y_pred_test = model.predict(X[test_idx])

    results = {}
    for alpha in [0.05, 0.10, 0.20]:
        ci = conformal_intervals(y[val_idx], y_pred_cal, alpha=alpha)

        # Apply to test set
        lower = y_pred_test - ci["q_hat"]
        upper = y_pred_test + ci["q_hat"]
        test_coverage = np.mean((y[test_idx] >= lower) & (y[test_idx] <= upper))

        ci["test_coverage"] = float(test_coverage)
        ci["n_test"] = len(test_idx)
        results[f"alpha_{alpha}"] = ci

        logger.info(
            "Conformal α=%.2f: cal_coverage=%.3f, test_coverage=%.3f, width=%.3f",
            alpha, ci["coverage"], test_coverage, ci["avg_width"],
        )

    # ── CQR with quantile models ───────────────────────────────────
    cqr_results = {}
    for alpha in [0.10]:
        try:
            q_lo = alpha / 2
            q_hi = 1 - alpha / 2

            model_lo = HistGradientBoostingRegressor(
                loss="quantile", quantile=q_lo,
                max_iter=ML_CFG.n_estimators, max_depth=ML_CFG.max_depth,
                random_state=ML_CFG.random_seed,
            )
            model_hi = HistGradientBoostingRegressor(
                loss="quantile", quantile=q_hi,
                max_iter=ML_CFG.n_estimators, max_depth=ML_CFG.max_depth,
                random_state=ML_CFG.random_seed,
            )

            model_lo.fit(X[train_idx], y[train_idx])
            model_hi.fit(X[train_idx], y[train_idx])

            cqr = conformalized_quantile_regression(
                y[val_idx],
                model_lo.predict(X[val_idx]),
                model_hi.predict(X[val_idx]),
                model_lo.predict(X[test_idx]),
                model_hi.predict(X[test_idx]),
                y[test_idx],
                alpha=alpha,
            )
            cqr_results[f"cqr_alpha_{alpha}"] = {
                k: v for k, v in cqr.items() if not isinstance(v, np.ndarray)
            }
        except Exception as e:
            logger.warning("CQR failed for alpha=%.2f: %s", alpha, e)

    if save_outputs:
        all_results = {**results, **cqr_results}
        rows = []
        for k, v in all_results.items():
            rows.append({"method": k, **v})
        pd.DataFrame(rows).to_csv(ML_TABLES_DIR / "conformal_results.csv", index=False)

    elapsed = time.time() - t0
    logger.info("Conformal analysis complete in %.1fs", elapsed)

    return {"split_conformal": results, "cqr": cqr_results}
