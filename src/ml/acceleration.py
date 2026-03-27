"""
ML acceleration helpers.

Targets Apple Silicon first: PyTorch models should use MPS when
available, while non-PyTorch estimators are explicit about when they
remain CPU-bound.
"""

from __future__ import annotations

import os
import platform
from typing import Any, Optional

from torch_device import get_torch_runtime_info


def is_apple_silicon() -> bool:
    """Return True on Apple Silicon macOS machines."""
    return platform.system() == "Darwin" and platform.machine() in {"arm64", "aarch64"}


def _resolve_lightgbm_device_type(preferred_torch_device: Optional[str] = None) -> str:
    """Resolve the LightGBM device type with safe fallbacks."""
    requested = (
        os.environ.get("ML_LIGHTGBM_DEVICE_TYPE")
        or os.environ.get("LIGHTGBM_DEVICE_TYPE")
        or ""
    ).strip().lower()
    if requested in {"cpu", "gpu", "cuda"}:
        return requested

    torch_info = get_torch_runtime_info(preferred_torch_device)
    if torch_info["selected_device"] == "cuda" and torch_info["cuda_available"]:
        return "gpu"
    return "cpu"


def get_ml_runtime_info(preferred_torch_device: Optional[str] = None) -> dict[str, Any]:
    """Describe the ML acceleration posture for the current machine."""
    torch_info = get_torch_runtime_info(preferred_torch_device)
    lightgbm_device_type = _resolve_lightgbm_device_type(preferred_torch_device)
    apple_silicon = is_apple_silicon()

    notes: list[str] = []
    if apple_silicon:
        notes.append("PyTorch workloads can use Apple Metal through the MPS backend.")
        notes.append("scikit-learn estimators in this repository remain CPU-bound on Apple Silicon.")
        if torch_info["selected_device"] != "cuda":
            notes.append("XGBoost GPU training is CUDA-only, so Apple Silicon falls back to CPU.")
        if lightgbm_device_type == "cpu":
            notes.append("LightGBM does not natively target Apple Metal/MPS in this configuration.")
    else:
        if torch_info["selected_device"] == "cuda":
            notes.append("CUDA is available for supported PyTorch and XGBoost workloads.")
        else:
            notes.append("No supported GPU backend detected; CPU fallbacks are in use.")

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "apple_silicon": apple_silicon,
        "torch": torch_info,
        "xgboost_runtime": (
            "cuda"
            if torch_info["selected_device"] == "cuda" and torch_info["cuda_available"]
            else "cpu"
        ),
        "lightgbm_device_type": lightgbm_device_type,
        "sklearn_runtime": "cpu",
        "notes": notes,
    }


def get_xgboost_regressor_kwargs(
    *,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    random_state: int,
    n_jobs: int = -1,
    preferred_torch_device: Optional[str] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return XGBoost kwargs plus backend metadata."""
    torch_info = get_torch_runtime_info(preferred_torch_device)
    kwargs: dict[str, Any] = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "random_state": random_state,
        "n_jobs": n_jobs,
        "tree_method": "hist",
    }

    if torch_info["selected_device"] == "cuda" and torch_info["cuda_available"]:
        kwargs["device"] = "cuda"
        backend = {
            "family": "xgboost",
            "runtime": "cuda",
            "accelerated": True,
            "reason": "Using CUDA-enabled histogram tree builder.",
        }
    else:
        backend = {
            "family": "xgboost",
            "runtime": "cpu",
            "accelerated": False,
            "reason": (
                "XGBoost GPU training requires CUDA; Apple MPS is not supported."
                if is_apple_silicon()
                else "No CUDA backend available for XGBoost."
            ),
        }

    return kwargs, backend


def get_lightgbm_regressor_kwargs(
    *,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    random_state: int,
    n_jobs: int = -1,
    verbose: int = -1,
    preferred_torch_device: Optional[str] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return LightGBM kwargs plus backend metadata."""
    device_type = _resolve_lightgbm_device_type(preferred_torch_device)
    kwargs: dict[str, Any] = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "random_state": random_state,
        "n_jobs": n_jobs,
        "verbose": verbose,
        "device_type": device_type,
    }

    if device_type != "cpu":
        backend = {
            "family": "lightgbm",
            "runtime": device_type,
            "accelerated": True,
            "reason": "Using an explicitly requested LightGBM GPU backend.",
        }
    else:
        backend = {
            "family": "lightgbm",
            "runtime": "cpu",
            "accelerated": False,
            "reason": (
                "LightGBM does not natively target Apple Metal/MPS in this configuration."
                if is_apple_silicon()
                else "No supported LightGBM GPU backend requested or available."
            ),
        }

    return kwargs, backend
