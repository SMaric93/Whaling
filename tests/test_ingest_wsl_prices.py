"""Tests for Stage 2 price-ingest step."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _price_payload() -> list[dict]:
    """A minimal JSONL-record list that passes the price validation gate."""
    out = []
    # 50 sperm + 50 whale + 50 bone quotes across multiple issues → ≥50 rows and
    # no single-commodity dominance.
    for i in range(50):
        d = f"1850_{1 + (i % 12):02d}_15"
        out.append({
            "pdf": f"wsl_{d}.pdf", "year": 1850, "page": 2,
            "page_key": f"wsl_{d}.pdf:p2_{i}",
            "prices": [
                {"c": "sperm_oil", "lo": 1.10, "hi": 1.20, "u": "gal"},
                {"c": "whale_oil", "lo": 0.45, "hi": 0.55, "u": "gal"},
                {"c": "whalebone", "lo": 0.35, "hi": 0.45, "u": "lb"},
            ],
        })
    return out


def test_ingest_wsl_prices_skips_when_no_jsonl(tmp_path: Path, monkeypatch) -> None:
    """With no extraction files present, step returns None (skipped)."""
    from src.pipeline import stage2_clean

    empty_dir = tmp_path / "nothing"
    monkeypatch.setattr(stage2_clean, "_price_extracted_dir",
                        lambda: empty_dir, raising=False)

    assert stage2_clean.ingest_wsl_prices() is None


def test_ingest_wsl_prices_produces_parquets_panel_and_run_log(tmp_path: Path,
                                                                monkeypatch) -> None:
    """Happy path: parquet written, panel written, run recorded."""
    from src.pipeline import stage2_clean
    from src.parsing import wsl_price_postprocess
    from src.utils import run_registry

    extracted = tmp_path / "extracted"
    staging = tmp_path / "staging"
    runs_dir = tmp_path / "runs"
    registry = tmp_path / "runs.jsonl"
    extracted.mkdir()

    _write_jsonl(extracted / "wsl_events_1850.jsonl", _price_payload())

    monkeypatch.setattr(stage2_clean, "STAGING_DIR", staging)
    monkeypatch.setattr(wsl_price_postprocess, "logger",
                        wsl_price_postprocess.logger)  # no-op, keeps access
    monkeypatch.setattr("src.config.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(run_registry, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(run_registry, "RUNS_REGISTRY", registry)

    # Point the ingest step's extracted_dir lookup at our fixture dir.
    monkeypatch.setattr(stage2_clean, "_price_extracted_dir",
                        lambda: extracted, raising=False)

    ok = stage2_clean.ingest_wsl_prices()

    assert ok is True
    assert (staging / "wsl_prices_v4.parquet").exists()
    assert (staging / "wsl_price_panel_monthly.parquet").exists()
    assert registry.exists()

    panel = pd.read_parquet(staging / "wsl_price_panel_monthly.parquet")
    assert set(panel["commodity"].unique()) == {"sperm_oil", "whale_oil", "whalebone"}

    with open(registry) as f:
        logged = [json.loads(line) for line in f if line.strip()]
    assert any(r["stage"] == "wsl_price_ingest" for r in logged)


def test_ingest_wsl_prices_raises_on_validation_failure(tmp_path: Path,
                                                          monkeypatch) -> None:
    """Bad prices (out-of-range) halt the pipeline."""
    from src.pipeline import stage2_clean
    from src.utils import run_registry
    from src.utils.price_validators import PriceValidationError

    extracted = tmp_path / "extracted"
    staging = tmp_path / "staging"
    extracted.mkdir()

    bad = []
    for i in range(60):
        d = f"1850_{1 + (i % 12):02d}_15"
        bad.append({
            "pdf": f"wsl_{d}.pdf", "year": 1850, "page": 2,
            "page_key": f"wsl_{d}.pdf:p2_{i}",
            "prices": [{"c": "sperm_oil", "lo": 999, "hi": 1001, "u": "gal"}],
        })
    _write_jsonl(extracted / "wsl_events_1850.jsonl", bad)

    monkeypatch.setattr(stage2_clean, "STAGING_DIR", staging)
    monkeypatch.setattr(run_registry, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(run_registry, "RUNS_REGISTRY", tmp_path / "runs.jsonl")
    monkeypatch.setattr(stage2_clean, "_price_extracted_dir",
                        lambda: extracted, raising=False)

    with pytest.raises(PriceValidationError):
        stage2_clean.ingest_wsl_prices()
