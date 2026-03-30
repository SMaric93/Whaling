"""
ML Layer — Appendix ML-12: Off-Policy Evaluation.

Evaluate counterfactual assignment rules from observational data,
without re-running the world.

Methods:
- Inverse propensity weighting (IPW)
- Self-normalized IPW (SNIPW)
- Doubly robust (DR) estimator
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.ml.config import ML_CFG, ML_TABLES_DIR

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Propensity Score Estimation
# ═══════════════════════════════════════════════════════════════════════════

def estimate_propensity(
    df: pd.DataFrame,
    treatment_col: str,
    covariate_cols: List[str],
    *,
    method: str = "logistic",
) -> np.ndarray:
    """
    Estimate propensity scores P(treatment | covariates).

    The caller MUST ensure that `df` has no NaN values in
    `treatment_col` or `covariate_cols` before calling this function.
    An internal check raises ValueError if NaNs are detected, rather
    than silently dropping rows (which would mis-align arrays).

    Returns
    -------
    np.ndarray
        Propensity scores, same length as `df`.
    """
    from sklearn.linear_model import LogisticRegression

    check_cols = [treatment_col] + covariate_cols
    n_missing = df[check_cols].isna().any(axis=1).sum()
    if n_missing > 0:
        raise ValueError(
            f"estimate_propensity received {n_missing} rows with NaN in "
            f"{check_cols}. The caller must drop these rows first to "
            f"keep arrays aligned."
        )

    X = df[covariate_cols].values
    t = df[treatment_col].astype(int).values

    if method == "logistic":
        lr = LogisticRegression(max_iter=2000, random_state=ML_CFG.random_seed, n_jobs=1)
        lr.fit(X, t)
        ps = lr.predict_proba(X)[:, 1]
    elif method == "hist_gbt":
        from sklearn.ensemble import HistGradientBoostingClassifier
        hgb = HistGradientBoostingClassifier(
            max_iter=200, max_depth=4, random_state=ML_CFG.random_seed,
        )
        hgb.fit(X, t)
        ps = hgb.predict_proba(X)[:, 1]
    else:
        raise ValueError(f"Unknown propensity method: {method}")

    # Clip for stability
    ps = np.clip(ps, 0.01, 0.99)
    return ps


# ═══════════════════════════════════════════════════════════════════════════
# IPW Estimator
# ═══════════════════════════════════════════════════════════════════════════

def ipw_estimate(
    y: np.ndarray,
    treatment: np.ndarray,
    propensity: np.ndarray,
) -> Dict[str, float]:
    """
    Inverse propensity weighting estimate of ATE.

    Returns
    -------
    Dict with ate, att, ate_se (approximate).
    """
    t = treatment.astype(float)
    w1 = t / propensity
    w0 = (1 - t) / (1 - propensity)

    ate = np.mean(y * w1) - np.mean(y * w0)

    # Self-normalized
    snipw_1 = np.sum(y * w1) / np.sum(w1)
    snipw_0 = np.sum(y * w0) / np.sum(w0)
    ate_sn = snipw_1 - snipw_0

    # ATT
    att = np.mean(y[t == 1]) - np.sum(y[t == 0] * propensity[t == 0] / (1 - propensity[t == 0])) / np.sum(propensity[t == 0] / (1 - propensity[t == 0]))

    # Bootstrap SE
    n_boot = 200
    rng = np.random.RandomState(ML_CFG.random_seed)
    boot_ates = []
    for _ in range(n_boot):
        idx = rng.choice(len(y), len(y), replace=True)
        w1_b = t[idx] / propensity[idx]
        w0_b = (1 - t[idx]) / (1 - propensity[idx])
        boot_ates.append(np.mean(y[idx] * w1_b) - np.mean(y[idx] * w0_b))

    return {
        "ate_ipw": float(ate),
        "ate_snipw": float(ate_sn),
        "att": float(att),
        "ate_se": float(np.std(boot_ates)),
        "n_treated": int(t.sum()),
        "n_control": int((1 - t).sum()),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Doubly Robust Estimator
# ═══════════════════════════════════════════════════════════════════════════

def doubly_robust_estimate(
    y: np.ndarray,
    treatment: np.ndarray,
    propensity: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
) -> Dict[str, float]:
    """
    Doubly robust (AIPW) estimate of ATE.

    Parameters
    ----------
    mu0 : predicted outcome under control
    mu1 : predicted outcome under treatment
    """
    t = treatment.astype(float)

    dr_1 = mu1 + t * (y - mu1) / propensity
    dr_0 = mu0 + (1 - t) * (y - mu0) / (1 - propensity)

    ate_dr = np.mean(dr_1 - dr_0)

    # Bootstrap SE
    n_boot = 200
    rng = np.random.RandomState(ML_CFG.random_seed)
    boot_ates = []
    for _ in range(n_boot):
        idx = rng.choice(len(y), len(y), replace=True)
        dr1_b = mu1[idx] + t[idx] * (y[idx] - mu1[idx]) / propensity[idx]
        dr0_b = mu0[idx] + (1 - t[idx]) * (y[idx] - mu0[idx]) / (1 - propensity[idx])
        boot_ates.append(np.mean(dr1_b - dr0_b))

    return {
        "ate_dr": float(ate_dr),
        "ate_dr_se": float(np.std(boot_ates)),
        "ate_dr_ci_lo": float(ate_dr - 1.96 * np.std(boot_ates)),
        "ate_dr_ci_hi": float(ate_dr + 1.96 * np.std(boot_ates)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main OPE Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_off_policy_evaluation(
    *,
    save_outputs: bool = True,
) -> Dict[str, Any]:
    """
    Run off-policy evaluation for agent assignment effects.

    Treatment: high-psi agent (above median).
    Outcome: log_q.
    """
    t0 = time.time()
    logger.info("Running off-policy evaluation...")

    from src.ml.build_outcome_ml_dataset import build_outcome_ml_dataset

    df = build_outcome_ml_dataset()

    target_col = "log_q"
    psi_col = "psi_hat_holdout"

    if target_col not in df.columns or psi_col not in df.columns:
        return {"error": "missing_columns"}

    df_valid = df.dropna(subset=[target_col, psi_col]).copy()

    # Treatment: high psi (above median)
    psi_median = df_valid[psi_col].median()
    df_valid["treatment"] = (df_valid[psi_col] > psi_median).astype(int)

    # Covariates
    covariates = [c for c in ["theta_hat_holdout", "captain_voyage_num",
                              "scarcity", "tonnage"] if c in df_valid.columns]

    if not covariates:
        return {"error": "no_covariates"}

    # Drop rows with NAs in any covariate, outcome, or treatment to keep
    # all arrays perfectly aligned (propensity, outcome, treatment).
    df_valid = df_valid.dropna(
        subset=covariates + [target_col, "treatment"]
    ).copy().reset_index(drop=True)

    n = len(df_valid)
    y = df_valid[target_col].values
    treatment = df_valid["treatment"].values

    # ── Propensity scores ───────────────────────────────────────────
    propensity = estimate_propensity(df_valid, "treatment", covariates)

    # Guard: arrays must be the same length
    assert len(propensity) == n, (
        f"Propensity array length ({len(propensity)}) != sample size ({n})"
    )

    # ── IPW ─────────────────────────────────────────────────────────
    ipw_results = ipw_estimate(y, treatment, propensity)
    logger.info("IPW ATE: %.4f (SE: %.4f)", ipw_results["ate_ipw"], ipw_results["ate_se"])

    # ── Outcome models for DR ───────────────────────────────────────
    from sklearn.ensemble import HistGradientBoostingRegressor

    X = df_valid[covariates].fillna(0).values
    treated = treatment == 1
    control = treatment == 0

    model_1 = HistGradientBoostingRegressor(
        max_iter=200, max_depth=4, random_state=ML_CFG.random_seed
    ).fit(X[treated], y[treated])

    model_0 = HistGradientBoostingRegressor(
        max_iter=200, max_depth=4, random_state=ML_CFG.random_seed
    ).fit(X[control], y[control])

    mu1 = model_1.predict(X)
    mu0 = model_0.predict(X)

    # ── Doubly robust ───────────────────────────────────────────────
    dr_results = doubly_robust_estimate(y, treatment, propensity, mu0, mu1)
    logger.info("DR ATE: %.4f (SE: %.4f)", dr_results["ate_dr"], dr_results["ate_dr_se"])

    # ── Naive comparison ────────────────────────────────────────────
    naive_ate = y[treated].mean() - y[control].mean()

    results = {
        "naive": {"ate": float(naive_ate)},
        "ipw": ipw_results,
        "doubly_robust": dr_results,
        "propensity_diagnostics": {
            "mean_ps": float(propensity.mean()),
            "std_ps": float(propensity.std()),
            "min_ps": float(propensity.min()),
            "max_ps": float(propensity.max()),
        },
    }

    if save_outputs:
        rows = [
            {"method": "naive", "ate": naive_ate, "se": np.nan},
            {"method": "ipw", "ate": ipw_results["ate_ipw"], "se": ipw_results["ate_se"]},
            {"method": "snipw", "ate": ipw_results["ate_snipw"], "se": np.nan},
            {"method": "doubly_robust", "ate": dr_results["ate_dr"], "se": dr_results["ate_dr_se"]},
        ]
        pd.DataFrame(rows).to_csv(ML_TABLES_DIR / "off_policy_evaluation.csv", index=False)

    elapsed = time.time() - t0
    logger.info("Off-policy evaluation complete in %.1fs", elapsed)

    return results
