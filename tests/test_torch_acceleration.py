from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _try_import(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def _fake_torch(*, mps_available: bool, cuda_available: bool):
    return types.SimpleNamespace(
        __version__="fake",
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: mps_available),
        ),
        cuda=types.SimpleNamespace(is_available=lambda: cuda_available),
        device=lambda name: types.SimpleNamespace(type=name),
    )


def test_resolve_torch_device_prefers_mps(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "torch",
        _fake_torch(mps_available=True, cuda_available=False),
    )

    from torch_device import resolve_torch_device

    assert resolve_torch_device() == "mps"


def test_resolve_torch_device_falls_back_to_cpu_when_requested_backend_missing(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "torch",
        _fake_torch(mps_available=False, cuda_available=False),
    )

    from torch_device import resolve_torch_device

    assert resolve_torch_device("mps") == "cpu"


def test_get_torch_runtime_info_reports_selected_device(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "torch",
        _fake_torch(mps_available=False, cuda_available=True),
    )

    from torch_device import get_torch_runtime_info

    info = get_torch_runtime_info()
    assert info["torch_available"] is True
    assert info["selected_device"] == "cuda"
    assert info["mps_fallback_enabled"] is True


@pytest.mark.skipif(
    not _try_import("torch"),
    reason="torch not installed",
)
def test_train_embedding_respects_cpu_device():
    from compass.config import CompassConfig
    from compass.embedding_optional import train_embedding

    steps = pd.DataFrame({
        "voyage_id": ["V1"] * 4 + ["V2"] * 4,
        "regime_label": ["search"] * 8,
        "step_length_m": np.linspace(100.0, 800.0, 8),
        "speed_mps": np.linspace(1.0, 8.0, 8),
        "heading_rad": np.linspace(0.1, 0.8, 8),
        "turning_angle_rad": np.linspace(0.05, 0.4, 8),
    })
    cfg = CompassConfig(
        embedding_enabled=True,
        segment_length_steps=4,
        embedding_dim=8,
        embedding_epochs=1,
        embedding_batch_size=2,
        embedding_lr=1e-3,
        torch_device="cpu",
        verbose=False,
    )

    encoder, embeddings, vids = train_embedding(steps, cfg)

    assert encoder is not None
    assert next(encoder.parameters()).device.type == "cpu"
    assert embeddings.shape == (2, 8)
    assert vids == ["V1", "V2"]
