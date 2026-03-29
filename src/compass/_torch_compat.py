"""Import shared torch helpers across both supported package layouts."""

from __future__ import annotations

try:
    from ..torch_device import (
        configure_torch_runtime,
        get_torch_device,
        get_torch_runtime_info,
        tensor_to_numpy,
        torch_is_available,
    )
except ImportError:
    from torch_device import (
        configure_torch_runtime,
        get_torch_device,
        get_torch_runtime_info,
        tensor_to_numpy,
        torch_is_available,
    )

__all__ = [
    "configure_torch_runtime",
    "get_torch_device",
    "get_torch_runtime_info",
    "tensor_to_numpy",
    "torch_is_available",
]
