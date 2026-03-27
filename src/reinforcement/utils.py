"""
Reinforcement Test Suite — Shared Utilities.

Regression helpers, clustered standard errors, table and figure formatters,
and memo writers used across all test modules.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

from .config import CFG, COLS, TABLES_DIR, FIGURES_DIR, MEMOS_DIR

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Fixed-Effect Absorption
# ═══════════════════════════════════════════════════════════════════════════

def absorb_fixed_effects(
    y: np.ndarray,
    X: np.ndarray,
    fe_groups: List[np.ndarray],
    *,
    return_residuals: bool = False,
) -> Dict:
    """
    Absorb high-dimensional fixed effects via sparse LSQR.

    Parameters
    ----------
    y : array (n,)
        Dependent variable.
    X : array (n, k)
        Regressors of interest.
    fe_groups : list of arrays
        Each array (n,) contains group labels for one set of FEs.
    return_residuals : bool
        If True, also return residuals.

    Returns
    -------
    dict with keys: coefficients, std_errors, residuals (optional),
    n_obs, n_fe_levels, r_squared
    """
    n = len(y)
    mask = np.isfinite(y)
    for col_idx in range(X.shape[1]):
        mask &= np.isfinite(X[:, col_idx])

    y_clean = y[mask]
    X_clean = X[mask, :]
    n_clean = len(y_clean)
    k = X_clean.shape[1]

    # Build sparse FE dummies
    fe_blocks = []
    n_fe_total = 0
    for groups in fe_groups:
        g = groups[mask]
        codes, uniques = pd.factorize(g)
        n_levels = len(uniques)
        fe_mat = sp.csc_matrix(
            (np.ones(n_clean), (np.arange(n_clean), codes)),
            shape=(n_clean, n_levels),
        )
        fe_blocks.append(fe_mat)
        n_fe_total += n_levels

    # Stack: [X | FE1 | FE2 | ...]
    X_sparse = sp.csc_matrix(X_clean)
    design = sp.hstack([X_sparse] + fe_blocks, format="csc")

    # Solve via LSQR
    result = lsqr(design, y_clean, atol=1e-10, btol=1e-10)
    beta_full = result[0]
    coefficients = beta_full[:k]

    # Fitted and residuals
    fitted = design @ beta_full
    residuals = y_clean - fitted
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_clean - y_clean.mean()) ** 2)
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    out = {
        "coefficients": coefficients,
        "n_obs": n_clean,
        "n_fe_levels": n_fe_total,
        "r_squared": r_sq,
        "dof": n_clean - k - n_fe_total,
        "_mask": mask,
    }

    if return_residuals:
        out["residuals"] = residuals
        out["fitted"] = fitted

    return out


# ═══════════════════════════════════════════════════════════════════════════
# Clustered Standard Errors
# ═══════════════════════════════════════════════════════════════════════════

def cluster_se(
    X: np.ndarray,
    residuals: np.ndarray,
    clusters: np.ndarray,
) -> np.ndarray:
    """
    Compute cluster-robust (HC1-style) standard errors.

    Parameters
    ----------
    X : array (n, k)
        Regressors (after FE absorption if applicable).
    residuals : array (n,)
        OLS residuals.
    clusters : array (n,)
        Cluster assignments.

    Returns
    -------
    array (k,) of standard errors.
    """
    n, k = X.shape
    unique_clusters = np.unique(clusters)
    G = len(unique_clusters)

    # Bread: (X'X)^{-1}
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)

    # Meat: Σ_g (X_g' e_g)(X_g' e_g)'
    meat = np.zeros((k, k))
    for g in unique_clusters:
        idx = clusters == g
        Xg = X[idx]
        eg = residuals[idx]
        score_g = Xg.T @ eg
        meat += np.outer(score_g, score_g)

    # HC1 small-sample correction
    correction = (G / (G - 1)) * ((n - 1) / (n - k))
    V = correction * XtX_inv @ meat @ XtX_inv

    return np.sqrt(np.maximum(np.diag(V), 0.0))


def multi_way_cluster_se(
    X: np.ndarray,
    residuals: np.ndarray,
    cluster_vars: List[np.ndarray],
) -> np.ndarray:
    """
    Two-way (or multi-way) clustered standard errors via Cameron-Gelbach-Miller.

    Parameters
    ----------
    X, residuals : as in cluster_se
    cluster_vars : list of cluster arrays (length 2 for two-way)

    Returns
    -------
    array (k,) of standard errors.
    """
    from itertools import combinations

    k = X.shape[1]
    n_dims = len(cluster_vars)

    # Individual cluster variances
    V_individual = []
    for cv in cluster_vars:
        se = cluster_se(X, residuals, cv)
        V_individual.append(np.diag(se**2))

    # Intersection cluster variances
    V_intersections = []
    for combo in combinations(range(n_dims), 2):
        # Intersection cluster = paste of both
        intersection = np.array([
            f"{cluster_vars[combo[0]][i]}_{cluster_vars[combo[1]][i]}"
            for i in range(len(residuals))
        ])
        se = cluster_se(X, residuals, intersection)
        V_intersections.append(np.diag(se**2))

    # Two-way: V = V1 + V2 - V12
    V = sum(V_individual) - sum(V_intersections)

    return np.sqrt(np.maximum(np.diag(V), 0.0))


# ═══════════════════════════════════════════════════════════════════════════
# Wild Cluster Bootstrap
# ═══════════════════════════════════════════════════════════════════════════

def wild_bootstrap_pvalue(
    X: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    coef_idx: int = 0,
    n_reps: int = 999,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Wild cluster bootstrap p-value (Rademacher weights, WCR).

    Parameters
    ----------
    X : (n, k) regressors
    y : (n,) outcome
    clusters : (n,) cluster IDs
    coef_idx : which coefficient to test
    n_reps : bootstrap replications
    seed : random seed

    Returns
    -------
    (p_value, t_stat_original)
    """
    rng = np.random.RandomState(seed)
    n, k = X.shape

    # Original OLS
    beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta_hat
    se_orig = cluster_se(X, resid, clusters)
    t_orig = beta_hat[coef_idx] / se_orig[coef_idx] if se_orig[coef_idx] > 0 else 0.0

    # Impose null: set coefficient to zero
    X_r = np.delete(X, coef_idx, axis=1)
    beta_r = np.linalg.lstsq(X_r, y, rcond=None)[0]
    y_r = X_r @ beta_r
    resid_r = y - y_r

    unique_clusters = np.unique(clusters)

    t_boots = np.zeros(n_reps)
    for b in range(n_reps):
        # Rademacher weights per cluster
        weights = rng.choice([-1, 1], size=len(unique_clusters))
        w = np.ones(n)
        for i, g in enumerate(unique_clusters):
            w[clusters == g] = weights[i]

        y_boot = y_r + w * resid_r
        beta_boot = np.linalg.lstsq(X, y_boot, rcond=None)[0]
        resid_boot = y_boot - X @ beta_boot
        se_boot = cluster_se(X, resid_boot, clusters)
        t_boots[b] = (
            beta_boot[coef_idx] / se_boot[coef_idx]
            if se_boot[coef_idx] > 0
            else 0.0
        )

    p_value = np.mean(np.abs(t_boots) >= np.abs(t_orig))
    return p_value, t_orig


# ═══════════════════════════════════════════════════════════════════════════
# Table Formatting
# ═══════════════════════════════════════════════════════════════════════════

def _star(p: float) -> str:
    """Return significance stars."""
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.10:
        return "*"
    return ""


def make_table(
    results: List[Dict],
    outcome_label: str,
    test_name: str,
    *,
    save: bool = True,
) -> pd.DataFrame:
    """
    Produce a clean regression results table.

    Parameters
    ----------
    results : list of dicts
        Each dict has: name, coefficients (dict), se (dict),
        pvalues (dict), n_obs, r_squared, n_clusters, fe_structure
    outcome_label : str
        Dependent variable label.
    test_name : str
        Test identifier for filename.
    save : bool
        Whether to save CSV + LaTeX.

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    for spec in results:
        for var_name, coef in spec.get("coefficients", {}).items():
            se = spec.get("se", {}).get(var_name, np.nan)
            pval = spec.get("pvalues", {}).get(var_name, np.nan)
            rows.append({
                "specification": spec.get("name", ""),
                "outcome": outcome_label,
                "variable": var_name,
                "coefficient": coef,
                "std_error": se,
                "p_value": pval,
                "stars": _star(pval) if np.isfinite(pval) else "",
                "n_obs": spec.get("n_obs", np.nan),
                "r_squared": spec.get("r_squared", np.nan),
                "n_clusters": spec.get("n_clusters", np.nan),
                "fe_structure": spec.get("fe_structure", ""),
                "cluster_var": spec.get("cluster_var", CFG.default_cluster_var),
            })

    df = pd.DataFrame(rows)

    if save:
        path = TABLES_DIR / f"{test_name}.csv"
        df.to_csv(path, index=False)
        logger.info("Saved table: %s", path)

    return df


# ═══════════════════════════════════════════════════════════════════════════
# Figure Helpers
# ═══════════════════════════════════════════════════════════════════════════

def make_figure(
    test_name: str,
    fig_name: str,
    *,
    figsize: Tuple[float, float] = (8, 5),
):
    """
    Create a publication-ready matplotlib figure with standard styling.

    Returns (fig, ax) and sets up save-on-close via test_name/fig_name.
    """
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 11,
        "font.family": "serif",
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": CFG.figure_dpi,
    })

    fig, ax = plt.subplots(figsize=figsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Store path for later saving
    fig._save_path = FIGURES_DIR / f"{test_name}_{fig_name}.{CFG.figure_format}"

    return fig, ax


def save_figure(fig, test_name: str = None, fig_name: str = None):
    """Save figure to the standard output directory."""
    import matplotlib.pyplot as plt

    path = getattr(fig, "_save_path", None)
    if path is None and test_name and fig_name:
        path = FIGURES_DIR / f"{test_name}_{fig_name}.{CFG.figure_format}"

    if path:
        fig.savefig(path, dpi=CFG.figure_dpi, bbox_inches="tight")
        logger.info("Saved figure: %s", path)
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Memo Writer
# ═══════════════════════════════════════════════════════════════════════════

def write_memo(
    test_name: str,
    content: str,
    *,
    threats: Optional[List[str]] = None,
) -> Path:
    """
    Write a short interpretation memo for a test.

    Parameters
    ----------
    test_name : str
        Test identifier.
    content : str
        Main memo text.
    threats : list of str, optional
        Remaining identification threats.

    Returns
    -------
    Path to the memo file.
    """
    path = MEMOS_DIR / f"{test_name}_memo.md"

    text = f"# {test_name} — Interpretation Memo\n\n{content}\n"

    if threats:
        text += "\n## Remaining Identification Threats\n\n"
        for t in threats:
            text += f"- {t}\n"

    path.write_text(text)
    logger.info("Wrote memo: %s", path)
    return path


# ═══════════════════════════════════════════════════════════════════════════
# Haversine (vectorized)
# ═══════════════════════════════════════════════════════════════════════════

EARTH_RADIUS_NM = 3440.065


def haversine_nm(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Vectorized great-circle distance in nautical miles."""
    lat1, lon1, lat2, lon2 = (
        np.radians(lat1), np.radians(lon1),
        np.radians(lat2), np.radians(lon2),
    )
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_NM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
