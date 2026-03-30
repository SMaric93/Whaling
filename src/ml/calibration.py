"""
ML Layer — Calibration Diagnostics.

Calibration plots, reliability diagrams, and calibration slope computation.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.ml.config import ML_CFG

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Calibration Curve
# ═══════════════════════════════════════════════════════════════════════════

def calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute calibration curve.

    Parameters
    ----------
    y_true : binary labels
    y_prob : predicted probabilities for the positive class
    n_bins : number of bins
    strategy : 'uniform' or 'quantile'

    Returns
    -------
    (fraction_of_positives, mean_predicted, bin_counts)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    if strategy == "quantile":
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(y_prob, quantiles * 100)
        bins = np.unique(bins)
    else:
        bins = np.linspace(0, 1, n_bins + 1)

    bin_ids = np.digitize(y_prob, bins[1:-1])

    fractions = []
    means = []
    counts = []

    for i in range(len(bins) - 1):
        mask = bin_ids == i
        if mask.sum() == 0:
            continue
        fractions.append(y_true[mask].mean())
        means.append(y_prob[mask].mean())
        counts.append(mask.sum())

    return np.array(fractions), np.array(means), np.array(counts)


# ═══════════════════════════════════════════════════════════════════════════
# Calibration Metrics
# ═══════════════════════════════════════════════════════════════════════════

def calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Compute calibration summary metrics.

    Returns
    -------
    Dict with brier_score, calibration_slope, calibration_intercept,
    expected_calibration_error (ECE), max_calibration_error (MCE).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    results = {}

    # Brier score
    results["brier_score"] = float(np.mean((y_true - y_prob) ** 2))

    # Calibration slope & intercept via logistic regression
    try:
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(penalty=None, max_iter=1000, n_jobs=1)
        lr.fit(y_prob.reshape(-1, 1), y_true)
        results["calibration_slope"] = float(lr.coef_[0, 0])
        results["calibration_intercept"] = float(lr.intercept_[0])
    except Exception:
        # Fallback: linear regression on log-odds
        try:
            eps = 1e-8
            logit_p = np.log(np.clip(y_prob, eps, 1 - eps) / (1 - np.clip(y_prob, eps, 1 - eps)))
            from numpy.polynomial import polynomial as P
            coeffs = P.polyfit(logit_p, y_true, deg=1)
            results["calibration_slope"] = float(coeffs[1])
            results["calibration_intercept"] = float(coeffs[0])
        except Exception:
            results["calibration_slope"] = np.nan
            results["calibration_intercept"] = np.nan

    # ECE and MCE
    frac, mean, counts = calibration_curve(y_true, y_prob, n_bins=n_bins)
    if len(frac) > 0:
        total = counts.sum()
        ece = np.sum(counts * np.abs(frac - mean)) / total
        mce = np.max(np.abs(frac - mean))
        results["ece"] = float(ece)
        results["mce"] = float(mce)
    else:
        results["ece"] = np.nan
        results["mce"] = np.nan

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Calibration Plot
# ═══════════════════════════════════════════════════════════════════════════

def calibration_plot(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    model_name: str = "Model",
    n_bins: int = 10,
    save_path: str = None,
    ax=None,
) -> Optional[object]:
    """
    Create a reliability diagram.

    Parameters
    ----------
    y_true, y_prob : arrays
    model_name : str for legend
    save_path : optional path to save figure
    ax : optional matplotlib axes

    Returns
    -------
    Figure or None.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available for calibration plot")
        return None

    frac, mean, counts = calibration_curve(y_true, y_prob, n_bins=n_bins)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.get_figure()

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")

    # Model calibration
    ax.plot(mean, frac, "s-", lw=2, label=model_name)

    # Histogram of predictions
    ax2 = ax.twinx()
    ax2.hist(y_prob, bins=n_bins, range=(0, 1), alpha=0.15, color="gray")
    ax2.set_ylabel("Count")
    ax2.set_ylim(bottom=0)

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Plot (Reliability Diagram)")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=ML_CFG.figure_dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Calibration plot saved to %s", save_path)
        return None

    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Multi-model Calibration Comparison
# ═══════════════════════════════════════════════════════════════════════════

def compare_calibration(
    y_true: np.ndarray,
    model_probs: Dict[str, np.ndarray],
    *,
    save_path: str = None,
) -> pd.DataFrame:
    """
    Compare calibration across multiple models.

    Parameters
    ----------
    model_probs : dict mapping model name → predicted probs

    Returns
    -------
    DataFrame of calibration metrics per model.
    """
    rows = []
    for name, probs in model_probs.items():
        metrics = calibration_metrics(y_true, probs)
        metrics["model"] = name
        rows.append(metrics)

    summary = pd.DataFrame(rows).set_index("model")

    if save_path:
        summary.to_csv(save_path)
        logger.info("Calibration comparison saved to %s", save_path)

    return summary
