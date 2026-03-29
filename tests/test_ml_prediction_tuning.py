from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _DummyRegressor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.feature_importances_ = None

    def fit(self, X, y):
        n_features = X.shape[1] if X.ndim == 2 else 1
        self.feature_importances_ = np.linspace(1.0, 0.1, n_features)
        self._prediction = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self._prediction, dtype=float)


def _sample_prediction_df() -> pd.DataFrame:
    return pd.DataFrame({
        "alpha_hat": np.linspace(0.1, 1.2, 12),
        "gamma_hat": np.linspace(1.1, 2.2, 12),
        "log_tonnage": np.linspace(2.0, 3.1, 12),
        "log_duration": np.linspace(0.5, 1.6, 12),
        "experience": np.arange(12),
        "decade": np.repeat([1830, 1840, 1850], 4),
        "log_q": np.linspace(3.0, 6.3, 12),
    })


def test_train_prediction_model_accepts_xgboost_tuning_overrides(monkeypatch):
    from src.analyses import ml_prediction

    captured = {}

    def fake_get_xgboost_regressor_kwargs(**kwargs):
        captured.update(kwargs)
        return (
            {
                "n_estimators": kwargs["n_estimators"],
                "max_depth": kwargs["max_depth"],
                "learning_rate": kwargs["learning_rate"],
                "random_state": kwargs["random_state"],
                "n_jobs": kwargs["n_jobs"],
                "tree_method": "hist",
            },
            {
                "family": "xgboost",
                "runtime": "cpu",
                "accelerated": False,
                "reason": "test",
            },
        )

    monkeypatch.setattr(
        ml_prediction,
        "get_xgboost_regressor_kwargs",
        fake_get_xgboost_regressor_kwargs,
    )
    monkeypatch.setattr(
        ml_prediction,
        "get_ml_runtime_info",
        lambda preferred_torch_device=None: {
            "torch": {"selected_device": preferred_torch_device or "cpu"}
        },
    )
    monkeypatch.setitem(
        sys.modules,
        "xgboost",
        types.SimpleNamespace(XGBRegressor=_DummyRegressor),
    )

    results = ml_prediction.train_prediction_model(
        _sample_prediction_df(),
        random_state=17,
        model_params={
            "n_estimators": 75,
            "max_depth": 3,
            "learning_rate": 0.2,
            "n_jobs": 2,
            "subsample": 0.65,
            "colsample_bytree": 0.8,
        },
    )

    assert captured["n_estimators"] == 75
    assert captured["max_depth"] == 3
    assert captured["learning_rate"] == 0.2
    assert captured["random_state"] == 17
    assert captured["n_jobs"] == 2
    assert results["model_kwargs"]["subsample"] == 0.65
    assert results["model_kwargs"]["colsample_bytree"] == 0.8
    assert results["model"].kwargs["n_estimators"] == 75


def test_train_prediction_model_preserves_default_booster_tuning(monkeypatch):
    from src.analyses import ml_prediction
    from src.ml.config import ML_CFG

    captured = {}

    def fake_get_xgboost_regressor_kwargs(**kwargs):
        captured.update(kwargs)
        return (
            {
                "n_estimators": kwargs["n_estimators"],
                "max_depth": kwargs["max_depth"],
                "learning_rate": kwargs["learning_rate"],
                "random_state": kwargs["random_state"],
                "n_jobs": kwargs["n_jobs"],
                "tree_method": "hist",
            },
            {
                "family": "xgboost",
                "runtime": "cpu",
                "accelerated": False,
                "reason": "test",
            },
        )

    monkeypatch.setattr(
        ml_prediction,
        "get_xgboost_regressor_kwargs",
        fake_get_xgboost_regressor_kwargs,
    )
    monkeypatch.setattr(
        ml_prediction,
        "get_ml_runtime_info",
        lambda preferred_torch_device=None: {"torch": {"selected_device": "cpu"}},
    )
    monkeypatch.setitem(
        sys.modules,
        "xgboost",
        types.SimpleNamespace(XGBRegressor=_DummyRegressor),
    )

    results = ml_prediction.train_prediction_model(
        _sample_prediction_df(),
        random_state=11,
    )

    assert captured["n_estimators"] == 200
    assert captured["max_depth"] == 6
    assert captured["learning_rate"] == 0.1
    assert captured["random_state"] == 11
    assert captured["n_jobs"] == -1
    assert results["model_kwargs"]["subsample"] == ML_CFG.subsample


def test_run_ml_prediction_analysis_forwards_tuning_inputs(monkeypatch):
    from src.analyses import ml_prediction

    train_calls = {}
    shap_calls = {}

    def fake_train_prediction_model(df, **kwargs):
        train_calls.update(kwargs)
        return {
            "model": object(),
            "model_type": kwargs["model_type"],
            "feature_cols": kwargs["feature_cols"],
            "target_col": kwargs["target_col"],
            "model_kwargs": kwargs["model_params"],
            "r2_train": 0.5,
            "r2_test": 0.4,
            "rmse_train": 1.0,
            "rmse_test": 1.1,
            "runtime_info": {"torch": {"selected_device": "cpu"}},
            "acceleration": {
                "family": kwargs["model_type"],
                "runtime": "cpu",
                "reason": "test",
            },
        }

    def fake_compute_shap_values(model_results, max_samples, save_outputs):
        shap_calls["max_samples"] = max_samples
        shap_calls["save_outputs"] = save_outputs
        return {
            "shap_importance": {"alpha_hat": 0.1},
            "theta_psi_interaction_proxy": None,
        }

    monkeypatch.setattr(
        ml_prediction,
        "train_prediction_model",
        fake_train_prediction_model,
    )
    monkeypatch.setattr(
        ml_prediction,
        "compute_shap_values",
        fake_compute_shap_values,
    )
    monkeypatch.setattr(
        ml_prediction,
        "analyze_residuals",
        lambda model_results, df, save_outputs: {
            "residual_std": 0.2,
            "n_outperformers": 1,
        },
    )

    ml_prediction.run_ml_prediction_analysis(
        _sample_prediction_df().rename(columns={"log_q": "target"}),
        target_col="target",
        feature_cols=["alpha_hat", "gamma_hat"],
        model_type="lightgbm",
        test_size=0.3,
        random_state=8,
        model_params={"n_estimators": 50, "num_leaves": 31},
        preferred_torch_device="cuda",
        shap_max_samples=33,
        save_outputs=False,
    )

    assert train_calls["target_col"] == "target"
    assert train_calls["feature_cols"] == ["alpha_hat", "gamma_hat"]
    assert train_calls["model_type"] == "lightgbm"
    assert train_calls["test_size"] == 0.3
    assert train_calls["random_state"] == 8
    assert train_calls["model_params"] == {
        "n_estimators": 50,
        "num_leaves": 31,
    }
    assert train_calls["preferred_torch_device"] == "cuda"
    assert shap_calls == {"max_samples": 33, "save_outputs": False}
