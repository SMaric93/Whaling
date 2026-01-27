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
    unique_clusters = np.unique(clusters)
    G = len(unique_clusters)
    n = len(x)
    
    # Cluster sums of x * residual
    cluster_sums = np.zeros(G)
    for g, c in enumerate(unique_clusters):
        mask = clusters == c
        cluster_sums[g] = np.sum(x[mask] * residuals[mask])
    
    # Meat of sandwich
    meat = np.sum(cluster_sums**2)
    
    # Bread
    bread = np.sum(x**2)
    
    # Cluster-robust variance with small-sample correction
    correction = (G / (G - 1)) * ((n - 1) / (n - 1))
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
        ids = df[col].unique()
        id_map = {v: idx for idx, v in enumerate(ids)}
        col_idx = df[col].map(id_map).values
        
        X_full = sp.csr_matrix(
            (np.ones(n), (np.arange(n), col_idx)),
            shape=(n, len(ids))
        )
        
        # Drop first category for identification (except first FE)
        if drop_first and i > 0:
            X = X_full[:, 1:]
        else:
            X = X_full
        
        matrices.append(X)
        index_maps[col] = {"ids": ids, "map": id_map, "n": len(ids)}
    
    X = sp.hstack(matrices)
    return X, index_maps
