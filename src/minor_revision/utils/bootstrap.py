"""
Block bootstrap engine for confidence intervals.
"""

import numpy as np
import pandas as pd
from typing import Callable, Optional


def voyage_bootstrap_ci(
    df: pd.DataFrame,
    statistic_fn: Callable[[pd.DataFrame], float],
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
    block_col: str = "captain_id",
) -> dict:
    """
    Block bootstrap CI resampling at the block_col level.

    Parameters
    ----------
    df : pd.DataFrame
        Data to resample.
    statistic_fn : callable
        Function that takes a DataFrame and returns a scalar statistic.
    n_boot : int
        Number of bootstrap replications.
    alpha : float
        Significance level (0.05 → 95% CI).
    seed : int
        Random seed.
    block_col : str
        Column to block-resample on (e.g. captain_id for captain-level bootstrap).

    Returns
    -------
    dict with keys: estimate, ci_lower, ci_upper, se, boot_values
    """
    rng = np.random.RandomState(seed)
    point_estimate = statistic_fn(df)

    blocks = df[block_col].unique()
    n_blocks = len(blocks)
    boot_values = np.empty(n_boot)

    for b in range(n_boot):
        sampled_blocks = rng.choice(blocks, size=n_blocks, replace=True)
        # Build resampled DataFrame
        parts = []
        for blk in sampled_blocks:
            parts.append(df[df[block_col] == blk])
        df_boot = pd.concat(parts, ignore_index=True)

        try:
            boot_values[b] = statistic_fn(df_boot)
        except Exception:
            boot_values[b] = np.nan

    boot_values = boot_values[~np.isnan(boot_values)]
    ci_lower = float(np.percentile(boot_values, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_values, 100 * (1 - alpha / 2)))
    se = float(np.std(boot_values))

    return {
        "estimate": point_estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "se": se,
        "boot_values": boot_values,
    }


def simple_bootstrap_ci(
    values: np.ndarray,
    statistic_fn: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Simple iid bootstrap CI on an array."""
    rng = np.random.RandomState(seed)
    point = statistic_fn(values)
    n = len(values)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boots[b] = statistic_fn(values[idx])

    return {
        "estimate": float(point),
        "ci_lower": float(np.percentile(boots, 100 * alpha / 2)),
        "ci_upper": float(np.percentile(boots, 100 * (1 - alpha / 2))),
        "se": float(np.std(boots)),
    }
