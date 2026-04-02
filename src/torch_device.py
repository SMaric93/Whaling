"""
PyTorch device helpers.

Keeps Apple Silicon MPS support in one place so GPU-capable modules can
share the same device-selection and CPU fallback behavior.
"""

from __future__ import annotations

import os
from typing import Any


def _prepare_torch_env() -> None:
    """
    Enable graceful CPU fallback for unsupported MPS ops.

    This keeps Apple Silicon runs usable even when a small number of
    operations are not implemented on MPS yet.
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _import_torch() -> Any | None:
    _prepare_torch_env()
    try:
        import torch
    except ImportError:
        return None
    return torch


def _mps_is_available(torch: Any) -> bool:
    return bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )


def torch_is_available() -> bool:
    """Return True when PyTorch can be imported."""
    return _import_torch() is not None


def resolve_torch_device(preferred: str | None = None) -> str:
    """
    Resolve the best available PyTorch device.

    Preference order for automatic selection:
    1. Apple Silicon MPS
    2. CUDA
    3. CPU
    """
    requested = (preferred or "").strip().lower()
    if requested == "auto":
        requested = ""

    torch = _import_torch()
    if torch is None:
        return "cpu"

    mps_available = _mps_is_available(torch)
    cuda_available = torch.cuda.is_available()

    if requested == "mps":
        return "mps" if mps_available else "cpu"
    if requested == "cuda":
        return "cuda" if cuda_available else "cpu"
    if requested == "cpu":
        return "cpu"

    if mps_available:
        return "mps"
    if cuda_available:
        return "cuda"
    return "cpu"


def get_torch_device(preferred: str | None = None) -> Any:
    """Return ``torch.device`` for the resolved runtime target."""
    torch = _import_torch()
    if torch is None:
        raise ImportError("PyTorch is required to construct a torch.device")

    return torch.device(resolve_torch_device(preferred))


def get_torch_runtime_info(preferred: str | None = None) -> dict[str, Any]:
    """Describe the current PyTorch runtime and selected device."""
    info: dict[str, Any] = {
        "torch_available": False,
        "requested_device": preferred or "auto",
        "selected_device": "cpu",
        "mps_available": False,
        "cuda_available": False,
        "mps_fallback_enabled": False,
        "float32_matmul_precision": None,
    }

    torch = _import_torch()
    if torch is None:
        return info

    info["torch_available"] = True
    info["torch_version"] = getattr(torch, "__version__", "unknown")
    info["mps_available"] = _mps_is_available(torch)
    info["cuda_available"] = torch.cuda.is_available()
    info["selected_device"] = resolve_torch_device(preferred)
    info["mps_fallback_enabled"] = (
        os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"
    )
    if hasattr(torch, "get_float32_matmul_precision"):
        try:
            info["float32_matmul_precision"] = torch.get_float32_matmul_precision()
        except Exception:
            info["float32_matmul_precision"] = None
    return info


def configure_torch_runtime(preferred: str | None = None) -> dict[str, Any]:
    """
    Apply lightweight runtime tuning for the selected PyTorch backend.

    On Apple Silicon this enables MPS fallback and opts into higher
    float32 matmul precision when supported.
    """
    info = get_torch_runtime_info(preferred)
    if not info["torch_available"]:
        return info

    torch = _import_torch()
    if torch is None:
        return info

    precision = os.environ.get("PYTORCH_FLOAT32_MATMUL_PRECISION", "high")
    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision(precision)
            info["float32_matmul_precision"] = precision
        except Exception:
            pass

    return info


def tensor_to_numpy(tensor: Any) -> Any:
    """Detach a tensor and move it back to CPU NumPy."""
    return tensor.detach().to("cpu").numpy()
