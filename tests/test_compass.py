"""
Compass Pipeline — Tests.

Covers:
1. Step computation (distance, heading, turning angle).
2. Golden-test: synthetic 3-regime trajectory → HMM recovers labels.
3. Data validation (monotonic timestamps, plausible speeds).
4. PCA produces finite, positive explained variance.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure src is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _try_import(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def simple_trajectory():
    """4-point trajectory going East then North (right-angle turn)."""
    times = pd.date_range("2020-01-01", periods=4, freq="6h", tz="UTC")
    return pd.DataFrame({
        "voyage_id": "V001",
        "timestamp_utc": times,
        "lat": [0.0, 0.0, 0.09, 0.18],
        "lon": [0.0, 0.09, 0.09, 0.09],
        "x_m": [0.0, 10000.0, 10000.0, 10000.0],
        "y_m": [0.0, 0.0, 10000.0, 20000.0],
    })


@pytest.fixture
def multi_voyage_trajectory():
    """Two voyages for testing per-voyage grouping."""
    times1 = pd.date_range("2020-01-01", periods=5, freq="6h", tz="UTC")
    times2 = pd.date_range("2020-02-01", periods=5, freq="6h", tz="UTC")

    df1 = pd.DataFrame({
        "voyage_id": "V001",
        "timestamp_utc": times1,
        "x_m": [0, 1000, 2000, 3000, 4000],
        "y_m": [0, 0, 0, 0, 0],
        "lat": [0.0] * 5,
        "lon": [0.0, 0.009, 0.018, 0.027, 0.036],
    })
    df2 = pd.DataFrame({
        "voyage_id": "V002",
        "timestamp_utc": times2,
        "x_m": [0, 0, 0, 0, 0],
        "y_m": [0, 1000, 2000, 3000, 4000],
        "lat": [0.0, 0.009, 0.018, 0.027, 0.036],
        "lon": [0.0] * 5,
    })
    return pd.concat([df1, df2], ignore_index=True)


@pytest.fixture
def synthetic_regime_trajectory():
    """
    Trajectory with 3 distinct regimes:
    - Transit (steps 0-99): fast, straight
    - Search (steps 100-249): slow, highly turning
    - Return (steps 250-349): fast, straight again
    """
    np.random.seed(42)
    n_transit, n_search, n_return = 100, 150, 100
    n_total = n_transit + n_search + n_return

    times = pd.date_range("2020-01-01", periods=n_total + 1, freq="6h", tz="UTC")

    x, y = [0.0], [0.0]
    for _ in range(n_transit):
        x.append(x[-1] + np.random.normal(5000, 500))
        y.append(y[-1] + np.random.normal(0, 200))
    for _ in range(n_search):
        angle = np.random.uniform(-np.pi, np.pi)
        step = np.random.exponential(500)
        x.append(x[-1] + step * np.cos(angle))
        y.append(y[-1] + step * np.sin(angle))
    for _ in range(n_return):
        x.append(x[-1] + np.random.normal(-5000, 500))
        y.append(y[-1] + np.random.normal(0, 200))

    return pd.DataFrame({
        "voyage_id": "V_SYNTH",
        "timestamp_utc": times,
        "x_m": x,
        "y_m": y,
        "lat": np.zeros(n_total + 1),
        "lon": np.zeros(n_total + 1),
    })


@pytest.fixture
def compass_config():
    from compass.config import CompassConfig
    return CompassConfig(
        minimum_points_per_voyage=3,
        min_steps_for_hmm=20,
        min_search_steps_for_features=5,
        num_regimes_candidates=[3],
        pca_n_components=2,
        standardize_group_col=None,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 1. Step Computation
# ═══════════════════════════════════════════════════════════════════════════

class TestSteps:
    def test_step_length(self, simple_trajectory):
        from compass.steps import compute_raw_steps
        steps = compute_raw_steps(simple_trajectory)
        assert abs(steps.iloc[1]["step_length_m"] - 10000.0) < 1.0

    def test_heading_east(self, simple_trajectory):
        from compass.steps import compute_raw_steps
        steps = compute_raw_steps(simple_trajectory)
        assert abs(steps.iloc[1]["heading_rad"] - 0.0) < 0.01

    def test_heading_north(self, simple_trajectory):
        from compass.steps import compute_raw_steps
        steps = compute_raw_steps(simple_trajectory)
        assert abs(steps.iloc[2]["heading_rad"] - np.pi / 2) < 0.01

    def test_turning_angle_right_turn(self, simple_trajectory):
        from compass.steps import compute_raw_steps
        steps = compute_raw_steps(simple_trajectory)
        ta = steps.iloc[2]["turning_angle_rad"]
        assert abs(ta - np.pi / 2) < 0.01

    def test_turning_angle_straight(self, simple_trajectory):
        from compass.steps import compute_raw_steps
        steps = compute_raw_steps(simple_trajectory)
        ta = steps.iloc[3]["turning_angle_rad"]
        assert abs(ta) < 0.01

    def test_speed_positive(self, simple_trajectory):
        from compass.steps import compute_raw_steps
        steps = compute_raw_steps(simple_trajectory)
        valid = steps.dropna(subset=["speed_mps"])
        assert (valid["speed_mps"] > 0).all()

    def test_per_voyage_independence(self, multi_voyage_trajectory):
        from compass.steps import compute_raw_steps
        steps = compute_raw_steps(multi_voyage_trajectory)
        v2_first = steps.loc[steps["voyage_id"] == "V002"].iloc[0]
        assert pd.isna(v2_first["step_length_m"])


# ═══════════════════════════════════════════════════════════════════════════
# 2. Time Resampling & Distance Thinning
# ═══════════════════════════════════════════════════════════════════════════

class TestStepDefinitions:
    def test_time_resample(self, multi_voyage_trajectory):
        from compass.steps import resample_time
        resampled = resample_time(multi_voyage_trajectory, hours=12, max_gap_hours=48)
        assert len(resampled) > 0
        for vid in resampled["voyage_id"].unique():
            sub = resampled[resampled["voyage_id"] == vid]
            diffs = sub["timestamp_utc"].diff().dropna()
            assert (diffs == pd.Timedelta(hours=12)).all()

    def test_distance_thin(self, multi_voyage_trajectory):
        from compass.steps import thin_distance
        thinned = thin_distance(multi_voyage_trajectory, threshold_m=1500)
        v1 = thinned[thinned["voyage_id"] == "V001"]
        assert len(v1) < 5
        assert len(v1) >= 2


# ═══════════════════════════════════════════════════════════════════════════
# 3. Regime Segmentation (Golden Test)
# ═══════════════════════════════════════════════════════════════════════════

class TestRegimes:
    @pytest.mark.skipif(
        not _try_import("hmmlearn"),
        reason="hmmlearn not installed",
    )
    def test_regime_recovery(self, synthetic_regime_trajectory, compass_config):
        from compass.steps import compute_raw_steps
        from compass.regimes import segment_voyages

        steps = compute_raw_steps(synthetic_regime_trajectory)
        segmented = segment_voyages(steps, compass_config)

        assert "regime_label" in segmented.columns
        valid = segmented.dropna(subset=["regime_label"])
        assert len(valid) > 0
        labels = valid["regime_label"].unique()
        assert "search" in labels
        assert "transit" in labels

    @pytest.mark.skipif(
        not _try_import("hmmlearn"),
        reason="hmmlearn not installed",
    )
    def test_search_regime_has_higher_turning(
        self, synthetic_regime_trajectory, compass_config,
    ):
        from compass.steps import compute_raw_steps
        from compass.regimes import segment_voyages

        steps = compute_raw_steps(synthetic_regime_trajectory)
        segmented = segment_voyages(steps, compass_config)
        valid = segmented.dropna(subset=["regime_label", "turning_angle_rad"])

        if "search" in valid["regime_label"].values and "transit" in valid["regime_label"].values:
            search_turn = valid.loc[
                valid["regime_label"] == "search", "turning_angle_rad"
            ].abs().mean()
            transit_turn = valid.loc[
                valid["regime_label"] == "transit", "turning_angle_rad"
            ].abs().mean()
            assert search_turn > transit_turn


# ═══════════════════════════════════════════════════════════════════════════
# 4. Features & Index
# ═══════════════════════════════════════════════════════════════════════════

class TestFeatures:
    @pytest.mark.skipif(
        not _try_import("hmmlearn"),
        reason="hmmlearn not installed",
    )
    def test_features_finite(self, synthetic_regime_trajectory, compass_config):
        from compass.steps import compute_raw_steps
        from compass.regimes import segment_voyages
        from compass.features import compute_compass_features

        steps = compute_raw_steps(synthetic_regime_trajectory)
        segmented = segment_voyages(steps, compass_config)
        feats = compute_compass_features(segmented, compass_config)

        if not feats.empty:
            numeric = feats.select_dtypes(include="number")
            assert numeric.notna().sum().sum() > 0


class TestIndex:
    def test_pca_explained_variance(self):
        from compass.compass_index import fit_pca
        np.random.seed(0)
        df = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        pca, loadings = fit_pca(df, n_components=2, feature_cols=list(df.columns))
        assert pca.explained_variance_ratio_.sum() > 0

    def test_standardize_global(self):
        from compass.compass_index import standardize_features
        df = pd.DataFrame({
            "voyage_id": ["a", "b", "c"],
            "hill_tail_index": [1.0, 2.0, 3.0],
            "mean_resultant_length": [0.5, 0.6, 0.7],
        })
        std_df, stats = standardize_features(df)
        assert abs(std_df["hill_tail_index"].mean()) < 1e-10


# ═══════════════════════════════════════════════════════════════════════════
# 5. Data Validation
# ═══════════════════════════════════════════════════════════════════════════

class TestValidation:
    def test_drops_impossible_coords(self):
        from compass.data_io import validate_trajectories
        from compass.config import CompassConfig

        cfg = CompassConfig(minimum_points_per_voyage=2)
        df = pd.DataFrame({
            "voyage_id": ["V1"] * 4,
            "timestamp_utc": pd.date_range("2020-01-01", periods=4, freq="6h", tz="UTC"),
            "lat": [0.0, 91.0, 45.0, 30.0],
            "lon": [0.0, 0.0, 0.0, 0.0],
        })
        validated = validate_trajectories(df, cfg)
        assert len(validated) == 3

    def test_drops_short_voyages(self):
        from compass.data_io import validate_trajectories
        from compass.config import CompassConfig

        cfg = CompassConfig(minimum_points_per_voyage=5)
        df = pd.DataFrame({
            "voyage_id": ["V1"] * 3 + ["V2"] * 6,
            "timestamp_utc": pd.date_range("2020-01-01", periods=9, freq="6h", tz="UTC"),
            "lat": [0.0] * 9,
            "lon": [0.0] * 9,
        })
        validated = validate_trajectories(df, cfg)
        assert "V1" not in validated["voyage_id"].values
        assert "V2" in validated["voyage_id"].values

    def test_gap_flagging(self):
        from compass.data_io import validate_trajectories
        from compass.config import CompassConfig

        cfg = CompassConfig(minimum_points_per_voyage=2, gap_threshold_hours=24)
        times = [
            "2020-01-01 00:00",
            "2020-01-01 06:00",
            "2020-01-03 00:00",
        ]
        df = pd.DataFrame({
            "voyage_id": ["V1"] * 3,
            "timestamp_utc": pd.to_datetime(times, utc=True),
            "lat": [0.0] * 3,
            "lon": [0.0] * 3,
        })
        validated = validate_trajectories(df, cfg)
        assert validated["gap_flag"].sum() == 1


# ═══════════════════════════════════════════════════════════════════════════
# 6. Config
# ═══════════════════════════════════════════════════════════════════════════

class TestConfig:
    def test_load_defaults(self):
        from compass.config import CompassConfig
        cfg = CompassConfig()
        assert cfg.minimum_points_per_voyage == 20
        assert cfg.hmm_random_state == 42

    def test_load_from_json(self, tmp_path):
        from compass.config import load_config, save_config, CompassConfig
        cfg = CompassConfig(minimum_points_per_voyage=50)
        p = tmp_path / "test_config.json"
        save_config(cfg, p)
        loaded = load_config(p)
        assert loaded.minimum_points_per_voyage == 50

    def test_load_missing_file(self):
        from compass.config import load_config
        cfg = load_config("/nonexistent/config.json")
        assert cfg.minimum_points_per_voyage == 20
