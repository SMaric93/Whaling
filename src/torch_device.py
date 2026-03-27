"""
PyTorch device helpers.

Keeps Apple Silicon MPS support in one place so GPU-capable modules can
share the same device-selection and CPU fallback behavior.
"""

from __future__ import annotations

import os
from typing import Any, Optional


def _prepare_torch_env() -> None:
    """
    Enable graceful CPU fallback for unsupported MPS ops.

    This keeps Apple Silicon runs usable even when a small number of
    operations are not implemented on MPS yet.
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def torch_is_available() -> bool:
    """Return True when PyTorch can be imported."""
    try:
        _prepare_torch_env()
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def resolve_torch_device(preferred: Optional[str] = None) -> str:
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

    if not torch_is_available():
        return "cpu"

    _prepare_torch_env()
    import torch

    mps_available = bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
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


def get_torch_device(preferred: Optional[str] = None):
    """Return ``torch.device`` for the resolved runtime target."""
    _prepare_torch_env()
    import torch

    return torch.device(resolve_torch_device(preferred))


def get_torch_runtime_info(preferred: Optional[str] = None) -> dict[str, Any]:
    """Describe the current PyTorch runtime and selected device."""
    info: dict[str, Any] = {
        "torch_available": False,
        "requested_device": preferred or "auto",
        "selected_device": "cpu",
        "mps_available": False,
        "cuda_available": False,
        "mps_fallback_enabled": False,
    }

    try:
        _prepare_torch_env()
        import torch
    except ImportError:
        return info

    info["torch_available"] = True
    info["torch_version"] = getattr(torch, "__version__", "unknown")
    info["mps_available"] = bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    info["cuda_available"] = torch.cuda.is_available()
    info["selected_device"] = resolve_torch_device(preferred)
    info["mps_fallback_enabled"] = (
        os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"
    )
    return info


def tensor_to_numpy(tensor):
    """Detach a tensor and move it back to CPU NumPy."""
    return tensor.detach().to("cpu").numpy()
