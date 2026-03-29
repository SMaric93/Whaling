from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _fake_torch(*, mps_available: bool, cuda_available: bool):
    return types.SimpleNamespace(
        __version__="fake",
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: mps_available),
        ),
        cuda=types.SimpleNamespace(is_available=lambda: cuda_available),
        device=lambda name: types.SimpleNamespace(type=name),
    )


def test_xgboost_kwargs_fall_back_to_cpu_on_mps(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "torch",
        _fake_torch(mps_available=True, cuda_available=False),
    )

    from ml.acceleration import get_xgboost_regressor_kwargs

    kwargs, backend = get_xgboost_regressor_kwargs(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
    )

    assert kwargs["tree_method"] == "hist"
    assert "device" not in kwargs
    assert backend["runtime"] == "cpu"
    assert backend["accelerated"] is False


def test_xgboost_kwargs_use_cuda_when_available(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "torch",
        _fake_torch(mps_available=False, cuda_available=True),
    )

    from ml.acceleration import get_xgboost_regressor_kwargs

    kwargs, backend = get_xgboost_regressor_kwargs(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        preferred_torch_device="cuda",
    )

    assert kwargs["tree_method"] == "hist"
    assert kwargs["device"] == "cuda"
    assert backend["runtime"] == "cuda"
    assert backend["accelerated"] is True


def test_ml_runtime_info_flags_apple_silicon_cpu_bound_sklearn(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "torch",
        _fake_torch(mps_available=True, cuda_available=False),
    )
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64")

    from ml.acceleration import get_ml_runtime_info

    info = get_ml_runtime_info()

    assert info["apple_silicon"] is True
    assert info["torch"]["selected_device"] == "mps"
    assert info["sklearn_runtime"] == "cpu"
    assert any("CPU-bound" in note for note in info["notes"])


def test_fit_logistic_baseline_uses_safe_single_process_path():
    from ml.baselines import fit_logistic_baseline

    rng = np.random.default_rng(0)
    X = rng.normal(size=(90, 4))
    y = np.repeat([0, 1, 2], 30)

    model = fit_logistic_baseline(X, y, max_iter=200)

    assert tuple(model.classes_) == (0, 1, 2)
    assert getattr(model, "n_jobs", None) == 1
    assert hasattr(model, "coef_")
    assert model.predict_proba(X).shape == (90, 3)


def test_estimate_propensity_logistic_runs_without_parallel_workers():
    from ml.off_policy_eval import estimate_propensity

    df = pytest.importorskip("pandas").DataFrame({
        "treated": [0, 1, 0, 1, 0, 1],
        "x1": [0.1, 0.8, 0.2, 0.7, 0.3, 0.9],
        "x2": [1.0, 0.0, 0.5, 0.2, 0.8, 0.1],
    })

    ps = estimate_propensity(df, "treated", ["x1", "x2"])

    assert len(ps) == len(df)
    assert np.all((ps >= 0.01) & (ps <= 0.99))
