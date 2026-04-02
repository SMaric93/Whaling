"""
Entropy and mutual-information helpers.

Implements:
  - Raw mutual information
  - Adjusted mutual information (via sklearn)
  - Normalized mutual information
  - Conditional mutual information from smoothed empirical counts
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_mutual_info_score,
    mutual_info_score,
    normalized_mutual_info_score,
)


def raw_mi(x: np.ndarray, y: np.ndarray) -> float:
    """Raw mutual information I(X; Y) in nats, converted to bits."""
    return float(mutual_info_score(x, y)) / np.log(2)


def ami(x: np.ndarray, y: np.ndarray) -> float:
    """Adjusted mutual information (chance-corrected)."""
    return float(adjusted_mutual_info_score(x, y))


def nmi(x: np.ndarray, y: np.ndarray) -> float:
    """Normalized mutual information (arithmetic average normalizer)."""
    return float(normalized_mutual_info_score(x, y, average_method="arithmetic"))


def entropy_bits(x: np.ndarray) -> float:
    """Shannon entropy H(X) in bits."""
    _, counts = np.unique(x, return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def conditional_entropy_bits(x: np.ndarray, cond: np.ndarray) -> float:
    """Conditional entropy H(X | Cond) in bits."""
    return entropy_bits(x) - raw_mi(x, cond)


def conditional_mi_smoothed(
    target: np.ndarray,
    var_a: np.ndarray,
    cond: np.ndarray,
    alpha: float = 1.0,
) -> float:
    """
    I(target; var_a | cond) via the chain rule:
      I(X; Y | Z) = I(X; (Y, Z)) - I(X; Z)

    This is exact for discrete variables and runs in O(n) instead of
    the O(|X| × |Y| × |Z|) triple loop.

    Parameters
    ----------
    target : array — the target variable (e.g. ground)
    var_a : array — the variable whose incremental info we want (e.g. captain)
    cond : array — the conditioning variable (e.g. agent)
    alpha : float — unused (kept for API compatibility)

    Returns
    -------
    float — I(target; var_a | cond) in bits
    """
    # Create joint label (var_a, cond) by string concatenation
    joint_label = np.array(
        [f"{a}__||__{c}" for a, c in zip(var_a, cond)], dtype=object
    )

    # I(target; (var_a, cond)) - I(target; cond), both in nats → convert to bits
    mi_joint = mutual_info_score(target, joint_label)
    mi_cond = mutual_info_score(target, cond)

    cmi_bits = float(mi_joint - mi_cond) / np.log(2)
    return max(cmi_bits, 0.0)  # CMI is non-negative; clamp rounding errors


def compute_all_mi_metrics(
    ground: np.ndarray,
    predictor: np.ndarray,
    predictor_name: str,
) -> dict:
    """Compute raw MI, AMI, NMI for a single predictor."""
    return {
        "Predictor": predictor_name,
        "Raw MI (bits)": f"{raw_mi(ground, predictor):.3f}",
        "AMI": f"{ami(ground, predictor):.3f}",
        "NMI": f"{nmi(ground, predictor):.3f}",
    }
