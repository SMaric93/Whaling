"""
Shared regression utilities.

Functions used across multiple analysis modules for building
design matrices and computing robust standard errors.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import scipy.sparse as sp


def cluster_robust_se(
    x: np.ndarray,
    residuals: np.ndarray,
    clusters: np.ndarray,
    coef_idx: int = 0,
) -> float:
    """
    Compute cluster-robust standard error for a single coefficient.
    
    Uses the HC1 small-sample correction.
    
    Parameters
    ----------
    x : np.ndarray
        Regressor values (n,).
    residuals : np.ndarray
        Regression residuals.
    clusters : np.ndarray
        Cluster identifiers.
    coef_idx : int
        Index of coefficient (for documentation, not used in computation).
        
    Returns
    -------
    float
        Cluster-robust standard error.
    """
    cluster_codes, unique_clusters = pd.factorize(clusters, sort=False)
    G = len(unique_clusters)
    n = len(x)

    # Cluster sums of x * residual
    cluster_sums = np.bincount(
        cluster_codes,
        weights=x * residuals,
        minlength=G,
    )

    # Meat of sandwich
    meat = np.dot(cluster_sums, cluster_sums)

    # Bread
    bread = np.dot(x, x)

    # Cluster-robust variance with small-sample HC1 correction
    # k=1 for this single-regressor API; for multi-regressor see
    # src.reinforcement.utils.cluster_se which handles k > 1.
    k = 1
    correction = (G / (G - 1)) * ((n - 1) / (n - k))
    var_beta = correction * meat / (bread**2)

    return np.sqrt(max(0, var_beta))


def build_fe_design_matrix(
    df: pd.DataFrame,
    fe_cols: List[str],
    drop_first: bool = True,
) -> Tuple[sp.csr_matrix, Dict]:
    """
    Build sparse design matrix for fixed effects.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with FE group columns.
    fe_cols : List[str]
        Column names for fixed effect groups.
    drop_first : bool
        If True, drop first category of each FE (except first FE column)
        for identification.
        
    Returns
    -------
    Tuple[sp.csr_matrix, Dict]
        (design_matrix, index_maps) where index_maps contains category
        mappings for each FE column.
    """
    n = len(df)
    matrices = []
    index_maps = {}

    for i, col in enumerate(fe_cols):
        col_idx, ids = pd.factorize(df[col], sort=False)
        first_kept_col = 1 if drop_first and i > 0 else 0
        keep_mask = col_idx >= first_kept_col

        X = sp.csr_matrix(
            (
                np.ones(keep_mask.sum()),
                (np.arange(n)[keep_mask], col_idx[keep_mask] - first_kept_col),
            ),
            shape=(n, len(ids) - first_kept_col),
        )

        matrices.append(X)
        id_map = {v: idx for idx, v in enumerate(ids)}
        index_maps[col] = {"ids": ids, "map": id_map, "n": len(ids)}

    X = sp.hstack(matrices)
    return X, index_maps
