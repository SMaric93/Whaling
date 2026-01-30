"""
Parallel implementations of AKM and KSS correction operations.

This module provides optimized parallel versions of computationally expensive 
operations in the AKM (Abowd-Kramarz-Margolis) fixed effects estimation 
and KSS (Kline-Saggio-Sølvsten) bias correction.

IMPORTANT: NumPy already uses multi-threaded BLAS for matrix operations.
Threading overhead can actually slow down NumPy-heavy code. This module:
1. Uses joblib with 'loky' backend for true multiprocessing
2. Only parallelizes when data size justifies the overhead
3. Falls back to sequential for small operations

Key Functions:
- parallel_kss_leverage: Chunked parallel leverage computation
- parallel_loo_fe: Parallel leave-one-out fixed effect computation
- parallel_eb_shrinkage: Block-wise parallel empirical Bayes shrinkage
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import scipy.sparse as sp

# Try to import joblib for better parallelism
try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


# ============================================================================
# CONFIGURATION
# ============================================================================

# Minimum data size thresholds where parallelism helps
MIN_ROWS_FOR_PARALLEL = 50000  # For vectorized operations
MIN_ENTITIES_FOR_PARALLEL = 1000  # For entity-level parallelism

def get_n_workers(n_workers: Optional[int] = None) -> int:
    """Get number of workers, defaulting to CPU count capped at 8."""
    if n_workers is not None:
        return n_workers
    return min(os.cpu_count() or 4, 8)


def should_parallelize(n: int, threshold: int = MIN_ROWS_FOR_PARALLEL) -> bool:
    """Determine if parallelization is worthwhile for given data size."""
    return n >= threshold and HAS_JOBLIB


# ============================================================================
# PARALLEL KSS LEVERAGE COMPUTATION
# ============================================================================

def _compute_leverage_chunk_job(
    chunk_idx: int,
    start: int,
    end: int,
    X_alpha: np.ndarray,
    S_alpha: np.ndarray,
    X_gamma: np.ndarray,
    S_gamma: np.ndarray,
    S_cross: np.ndarray,
) -> Tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute leverage values for a chunk of observations (joblib version).
    """
    chunk_alpha = X_alpha[start:end]
    chunk_gamma = X_gamma[start:end]
    
    leverage_alpha = np.sum((chunk_alpha @ S_alpha) * chunk_alpha, axis=1)
    leverage_gamma = np.sum((chunk_gamma @ S_gamma) * chunk_gamma, axis=1)
    leverage_cov = np.sum((chunk_alpha @ S_cross) * chunk_gamma, axis=1)
    
    return chunk_idx, start, end, leverage_alpha, leverage_gamma, leverage_cov


def parallel_kss_leverage(
    X_alpha: np.ndarray,
    S_alpha: np.ndarray,
    X_gamma: np.ndarray,
    S_gamma: np.ndarray,
    S_cross: np.ndarray,
    n_workers: Optional[int] = None,
    chunk_size: Optional[int] = None,
    force_parallel: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute KSS leverage values, optionally in parallel across observation chunks.
    
    NOTE: NumPy matrix operations already use multi-threaded BLAS internally.
    This function only adds Python-level parallelism for very large datasets
    where chunking helps with memory access patterns.
    
    Parameters
    ----------
    X_alpha : np.ndarray
        Captain design matrix (n x n_captains)
    S_alpha : np.ndarray
        (X'X)^-1 block for captains
    X_gamma : np.ndarray
        Agent design matrix (n x n_agents-1)
    S_gamma : np.ndarray
        (X'X)^-1 block for agents
    S_cross : np.ndarray
        (X'X)^-1 cross block
    n_workers : int, optional
        Number of worker processes
    chunk_size : int, optional
        Size of each chunk
    force_parallel : bool
        Force parallel execution even for small data
        
    Returns
    -------
    Tuple of (leverage_alpha, leverage_gamma, leverage_cov) arrays
    """
    n = X_alpha.shape[0]
    n_workers = get_n_workers(n_workers)
    
    # Check if parallelization is worthwhile
    use_parallel = force_parallel or (should_parallelize(n) and HAS_JOBLIB)
    
    if not use_parallel:
        # Sequential: NumPy's internal BLAS parallelism is sufficient
        leverage_alpha = np.sum((X_alpha @ S_alpha) * X_alpha, axis=1)
        leverage_gamma = np.sum((X_gamma @ S_gamma) * X_gamma, axis=1)
        leverage_cov = np.sum((X_alpha @ S_cross) * X_gamma, axis=1)
        return leverage_alpha, leverage_gamma, leverage_cov
    
    # Parallel processing with joblib
    if chunk_size is None:
        chunk_size = max(n // n_workers, 5000)
    
    # Create chunks
    chunks = []
    for idx, start in enumerate(range(0, n, chunk_size)):
        end = min(start + chunk_size, n)
        chunks.append((idx, start, end))
    
    # Preallocate output arrays
    leverage_alpha = np.zeros(n)
    leverage_gamma = np.zeros(n)
    leverage_cov = np.zeros(n)
    
    # Process chunks in parallel using joblib multiprocessing
    results = Parallel(n_jobs=n_workers, backend='loky')(
        delayed(_compute_leverage_chunk_job)(
            idx, start, end,
            X_alpha, S_alpha, X_gamma, S_gamma, S_cross
        )
        for idx, start, end in chunks
    )
    
    # Collect results
    for chunk_idx, start, end, chunk_alpha, chunk_gamma, chunk_cov in results:
        leverage_alpha[start:end] = chunk_alpha
        leverage_gamma[start:end] = chunk_gamma
        leverage_cov[start:end] = chunk_cov
    
    return leverage_alpha, leverage_gamma, leverage_cov


# ============================================================================
# PARALLEL LOO FIXED EFFECT COMPUTATION
# ============================================================================

def _compute_loo_fe_for_entity(
    entity_id: str,
    entity_voyages: pd.DataFrame,
    resid_col: str,
    fe_col: str,
) -> Dict:
    """
    Compute LOO fixed effect for a single entity.
    """
    n_obs = len(entity_voyages)
    
    if n_obs <= 1:
        return {
            'entity_id': entity_id,
            'loo_fe_hat': np.nan,
            'loo_se': np.nan,
            'n_obs': n_obs,
            'fe_hat': entity_voyages[fe_col].iloc[0] if len(entity_voyages) > 0 else np.nan,
        }
    
    mean_resid = entity_voyages[resid_col].mean()
    var_resid = entity_voyages[resid_col].var()
    loo_se = np.sqrt(var_resid / (n_obs - 1)) if var_resid > 0 else np.nan
    
    return {
        'entity_id': entity_id,
        'loo_fe_hat': mean_resid,
        'loo_se': loo_se,
        'n_obs': n_obs,
        'fe_hat': entity_voyages[fe_col].iloc[0],
    }


def parallel_loo_fe(
    voyage_df: pd.DataFrame,
    entity_col: str,
    resid_col: str,
    fe_col: str,
    n_workers: Optional[int] = None,
    force_parallel: bool = False,
) -> pd.DataFrame:
    """
    Compute LOO fixed effects, optionally in parallel across entities.
    
    For small entity counts, uses vectorized pandas operations.
    For large entity counts, uses joblib multiprocessing.
    
    Parameters
    ----------
    voyage_df : pd.DataFrame
        Voyage data with residualized outcomes
    entity_col : str
        Column identifying entities (e.g., 'captain_id')
    resid_col : str
        Column with residualized outcome
    fe_col : str
        Column with plug-in FE estimate
    n_workers : int, optional
        Number of worker processes
    force_parallel : bool
        Force parallel execution
        
    Returns
    -------
    pd.DataFrame with entity_id, loo_fe_hat, loo_se, n_obs, fe_hat
    """
    n_entities = voyage_df[entity_col].nunique()
    n_workers = get_n_workers(n_workers)
    
    # For small entity counts, use vectorized groupby (faster)
    use_parallel = force_parallel or (n_entities >= MIN_ENTITIES_FOR_PARALLEL and HAS_JOBLIB)
    
    if not use_parallel:
        # Vectorized implementation using pandas groupby
        entity_stats = voyage_df.groupby(entity_col).agg(
            n_obs=(resid_col, 'count'),
            mean_resid=(resid_col, 'mean'),
            var_resid=(resid_col, 'var'),
            fe_hat=(fe_col, 'first'),
        ).reset_index()
        
        entity_stats = entity_stats.rename(columns={entity_col: 'entity_id'})
        
        entity_stats['loo_fe_hat'] = np.where(
            entity_stats['n_obs'] > 1,
            entity_stats['mean_resid'],
            np.nan
        )
        
        entity_stats['loo_se'] = np.where(
            entity_stats['n_obs'] > 1,
            np.sqrt(entity_stats['var_resid'].fillna(0) / (entity_stats['n_obs'] - 1)),
            np.nan
        )
        
        return entity_stats[['entity_id', 'loo_fe_hat', 'loo_se', 'n_obs', 'fe_hat']]
    
    # Parallel processing for large entity counts
    grouped = voyage_df.groupby(entity_col)
    entity_groups = [(entity_id, group) for entity_id, group in grouped]
    
    results = Parallel(n_jobs=n_workers, backend='loky')(
        delayed(_compute_loo_fe_for_entity)(entity_id, group, resid_col, fe_col)
        for entity_id, group in entity_groups
    )
    
    return pd.DataFrame(results)


# ============================================================================
# PARALLEL EMPIRICAL BAYES SHRINKAGE
# ============================================================================

def parallel_eb_shrinkage(
    fe_hat: np.ndarray,
    n_obs: np.ndarray,
    sigma2_eps: float,
    n_workers: Optional[int] = None,
    chunk_size: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Apply Empirical Bayes shrinkage.
    
    NOTE: This operation is fully vectorized and fast. NumPy's internal
    parallelism handles this efficiently. No Python-level parallelism needed.
    
    Parameters
    ----------
    fe_hat : np.ndarray
        Raw fixed effect estimates
    n_obs : np.ndarray
        Number of observations per entity
    sigma2_eps : float
        Residual variance
    n_workers : int, optional
        Unused (kept for API compatibility)
    chunk_size : int, optional
        Unused (kept for API compatibility)
        
    Returns
    -------
    Tuple of (fe_eb, shrinkage_lambda, var_signal)
    """
    # Fully vectorized - NumPy BLAS handles parallelism
    global_mean = np.mean(fe_hat)
    noise_var = sigma2_eps * np.mean(1 / n_obs)
    var_signal = max(0, np.var(fe_hat) - noise_var)
    
    if var_signal > 0:
        shrinkage_lambda = var_signal / (var_signal + sigma2_eps / n_obs)
    else:
        shrinkage_lambda = np.zeros_like(n_obs, dtype=float)
    
    fe_eb = shrinkage_lambda * fe_hat + (1 - shrinkage_lambda) * global_mean
    
    return fe_eb, shrinkage_lambda, var_signal


# ============================================================================
# TESTING UTILITIES
# ============================================================================

def verify_parallel_correctness(
    sequential_result: np.ndarray,
    parallel_result: np.ndarray,
    rtol: float = 1e-10,
    atol: float = 1e-10,
) -> bool:
    """
    Verify that parallel and sequential results match.
    """
    return np.allclose(sequential_result, parallel_result, rtol=rtol, atol=atol)


def benchmark_parallel_speedup(
    func_sequential,
    func_parallel,
    *args,
    n_runs: int = 3,
    **kwargs,
) -> Dict:
    """
    Benchmark parallel vs sequential execution.
    """
    import time
    
    seq_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        seq_result = func_sequential(*args, **kwargs)
        seq_times.append(time.perf_counter() - start)
    
    par_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        par_result = func_parallel(*args, **kwargs)
        par_times.append(time.perf_counter() - start)
    
    return {
        'sequential_time': np.mean(seq_times),
        'parallel_time': np.mean(par_times),
        'speedup': np.mean(seq_times) / np.mean(par_times),
        'n_runs': n_runs,
    }
