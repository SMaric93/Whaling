"""
ML Layer — Evaluation Metrics.

Unified evaluation functions for all ML tasks:
- Classification: log loss, Brier score, AUC, macro F1, top-k accuracy
- Regression: RMSE, MAE, R², calibration slope
- Quantile: pinball loss
- Survival: concordance index, integrated Brier score
- Ranking / Assignment: welfare gain, downside-risk reduction
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Classification Metrics
# ═══════════════════════════════════════════════════════════════════════════

def classification_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    y_pred_class: np.ndarray = None,
    *,
    top_k: int = 3,
) -> Dict[str, float]:
    """
    Compute standard classification metrics.

    Parameters
    ----------
    y_true : array of true labels (integer-encoded)
    y_pred_proba : array of predicted probabilities, shape (n, n_classes) or (n,)
    y_pred_class : array of predicted classes (optional, inferred from proba)
    top_k : int
        For top-k accuracy.

    Returns
    -------
    Dict with log_loss, brier_score, auc, macro_f1, top_k_accuracy.
    """
    from sklearn.metrics import (
        log_loss, f1_score, roc_auc_score,
    )

    results = {}
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    # Ensure 2D for multiclass
    if y_pred_proba.ndim == 1:
        y_pred_proba = np.column_stack([1 - y_pred_proba, y_pred_proba])

    if y_pred_class is None:
        y_pred_class = y_pred_proba.argmax(axis=1)

    n_classes = y_pred_proba.shape[1]

    # Log loss
    try:
        results["log_loss"] = log_loss(y_true, y_pred_proba)
    except Exception:
        results["log_loss"] = np.nan

    # Brier score (multi-class one-vs-all average)
    try:
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        if y_bin.shape[1] == 1:
            y_bin = np.column_stack([1 - y_bin, y_bin])
        results["brier_score"] = float(np.mean((y_bin - y_pred_proba) ** 2))
    except Exception:
        results["brier_score"] = np.nan

    # AUC (OVR for multiclass)
    try:
        if n_classes == 2:
            results["auc"] = roc_auc_score(y_true, y_pred_proba[:, 1])
        else:
            results["auc"] = roc_auc_score(
                y_true, y_pred_proba, multi_class="ovr", average="macro"
            )
    except Exception:
        results["auc"] = np.nan

    # Macro F1
    try:
        results["macro_f1"] = f1_score(y_true, y_pred_class, average="macro", zero_division=0)
    except Exception:
        results["macro_f1"] = np.nan

    # Top-k accuracy
    try:
        k = min(top_k, n_classes)
        top_k_preds = np.argsort(y_pred_proba, axis=1)[:, -k:]
        results[f"top_{k}_accuracy"] = float(
            np.mean([y_true[i] in top_k_preds[i] for i in range(len(y_true))])
        )
    except Exception:
        results[f"top_{top_k}_accuracy"] = np.nan

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Regression Metrics
# ═══════════════════════════════════════════════════════════════════════════

def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute standard regression metrics.

    Returns
    -------
    Dict with rmse, mae, r_squared, calibration_slope.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]

    residuals = y_true - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    results = {
        "rmse": float(np.sqrt(np.mean(residuals ** 2))),
        "mae": float(np.mean(np.abs(residuals))),
        "r_squared": float(1 - ss_res / max(ss_tot, 1e-12)),
        "n_obs": int(len(y_true)),
    }

    # Calibration slope: regress y_true on y_pred
    try:
        from numpy.polynomial import polynomial as P
        coeffs = P.polyfit(y_pred, y_true, deg=1)
        results["calibration_slope"] = float(coeffs[1])
    except Exception:
        results["calibration_slope"] = np.nan

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Quantile Metrics
# ═══════════════════════════════════════════════════════════════════════════

def pinball_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantile: float,
) -> float:
    """Pinball (quantile) loss."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    diff = y_true - y_pred
    return float(np.mean(np.where(diff >= 0, quantile * diff, (quantile - 1) * diff)))


# ═══════════════════════════════════════════════════════════════════════════
# Survival Metrics
# ═══════════════════════════════════════════════════════════════════════════

def concordance_index(
    event_times: np.ndarray,
    predicted_risk: np.ndarray,
    event_observed: np.ndarray = None,
) -> float:
    """
    Harrell's concordance index.

    Attempts lifelines first, then falls back to manual computation.
    """
    try:
        from lifelines.utils import concordance_index as _ci
        return float(_ci(event_times, -predicted_risk, event_observed))
    except ImportError:
        pass

    # Manual concordance
    event_times = np.asarray(event_times)
    predicted_risk = np.asarray(predicted_risk)
    if event_observed is None:
        event_observed = np.ones_like(event_times)
    event_observed = np.asarray(event_observed)

    concordant = 0
    discordant = 0
    tied = 0
    n = len(event_times)

    for i in range(n):
        if not event_observed[i]:
            continue
        for j in range(n):
            if event_times[i] < event_times[j]:
                if predicted_risk[i] > predicted_risk[j]:
                    concordant += 1
                elif predicted_risk[i] < predicted_risk[j]:
                    discordant += 1
                else:
                    tied += 1

    total = concordant + discordant + tied
    if total == 0:
        return 0.5
    return float((concordant + 0.5 * tied) / total)


# ═══════════════════════════════════════════════════════════════════════════
# Assignment / Welfare Metrics
# ═══════════════════════════════════════════════════════════════════════════

def welfare_metrics(
    outcomes: np.ndarray,
    *,
    label: str = "assignment",
) -> Dict[str, float]:
    """
    Compute welfare metrics for an assignment rule.

    Returns aggregate, bottom-decile, novice-avg if available.
    """
    outcomes = np.asarray(outcomes, dtype=float)
    outcomes = outcomes[np.isfinite(outcomes)]

    results = {
        f"{label}_aggregate": float(np.mean(outcomes)),
        f"{label}_median": float(np.median(outcomes)),
        f"{label}_bottom_10pct": float(np.percentile(outcomes, 10)),
        f"{label}_bottom_5pct": float(np.percentile(outcomes, 5)),
        f"{label}_std": float(np.std(outcomes)),
        f"{label}_n": int(len(outcomes)),
    }
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Unified Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: str = "regression",
    *,
    y_pred_proba: np.ndarray = None,
    quantile: float = None,
    event_times: np.ndarray = None,
    event_observed: np.ndarray = None,
) -> Dict[str, float]:
    """
    Unified evaluation dispatcher.

    Parameters
    ----------
    task : str
        One of 'classification', 'regression', 'quantile', 'survival'.
    """
    if task == "classification":
        if y_pred_proba is None:
            raise ValueError("y_pred_proba required for classification")
        return classification_metrics(y_true, y_pred_proba, y_pred)
    elif task == "regression":
        return regression_metrics(y_true, y_pred)
    elif task == "quantile":
        if quantile is None:
            raise ValueError("quantile required for quantile task")
        return {"pinball_loss": pinball_loss(y_true, y_pred, quantile)}
    elif task == "survival":
        return {"concordance_index": concordance_index(
            event_times or y_true, y_pred, event_observed
        )}
    else:
        raise ValueError(f"Unknown task: {task}")
