"""Tests for regression utility helpers."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure src is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_cluster_robust_se_returns_non_negative_value():
    from utils.regression import cluster_robust_se

    x = np.array([1.0, 2.0, 3.0, 4.0])
    residuals = np.array([0.5, -0.5, 0.25, -0.25])
    clusters = np.array(["A", "A", "B", "B"])

    se = cluster_robust_se(x, residuals, clusters)

    assert se >= 0.0


def test_build_fe_design_matrix_drops_first_category_for_later_effects():
    from utils.regression import build_fe_design_matrix

    df = pd.DataFrame({
        "captain_id": ["C1", "C2", "C1", "C3"],
        "agent_id": ["A1", "A1", "A2", "A3"],
    })

    X, index_maps = build_fe_design_matrix(df, ["captain_id", "agent_id"])

    expected_cols = index_maps["captain_id"]["n"] + index_maps["agent_id"]["n"] - 1
    assert X.shape == (len(df), expected_cols)
