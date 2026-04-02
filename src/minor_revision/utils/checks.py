"""
Assertion helpers for scale consistency across tables.
"""

import numpy as np


def assert_var_sd_consistent(
    sd: float, var: float, label: str = "", tol: float = 1e-4
) -> None:
    """Check that SD² ≈ Var for a reported statistic."""
    delta = abs(sd ** 2 - var)
    if delta > tol:
        raise AssertionError(
            f"Scale inconsistency in {label}: "
            f"SD²={sd**2:.6f} ≠ Var={var:.6f} (Δ={delta:.6f})"
        )


def assert_sample_sizes_match(
    n1: int, n2: int, label1: str = "Table A", label2: str = "Table B"
) -> None:
    """Check that two sample sizes are identical."""
    if n1 != n2:
        raise AssertionError(
            f"Sample size mismatch: {label1} has N={n1:,}, "
            f"{label2} has N={n2:,}"
        )


def compute_descriptive_stats(values: np.ndarray, label: str = "") -> dict:
    """Compute full descriptive statistics for an outcome array."""
    clean = values[~np.isnan(values)]
    return {
        "label": label,
        "N": len(clean),
        "mean": float(np.mean(clean)),
        "sd": float(np.std(clean, ddof=1)),
        "variance": float(np.var(clean, ddof=1)),
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
        "p25": float(np.percentile(clean, 25)),
        "p75": float(np.percentile(clean, 75)),
    }
