"""Tests for VLM price post-processor (WSL commodity price time series)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def test_flatten_one_record_produces_row_per_price(tmp_path: Path) -> None:
    """A page record with N prices flattens to N rows."""
    from src.parsing.wsl_price_postprocess import flatten_price_record

    rec = {
        "pdf": "wsl_1850_01_15.pdf",
        "year": 1850,
        "page": 2,
        "page_key": "wsl_1850_01_15.pdf:p2",
        "page_type": "market_prices",
        "prices": [
            {"c": "sperm_oil", "lo": 1.15, "hi": 1.20, "u": "gal"},
            {"c": "whale_oil", "lo": 0.45, "hi": 0.50, "u": "gal"},
            {"c": "whalebone", "lo": 0.38, "hi": 0.42, "u": "lb"},
        ],
    }

    rows = flatten_price_record(rec)

    assert len(rows) == 3
    assert {r["commodity"] for r in rows} == {"sperm_oil", "whale_oil", "whalebone"}


def test_flatten_carries_price_low_high_and_computes_mid() -> None:
    """price_mid = (low + high) / 2 when both present; falls back to whichever is set."""
    from src.parsing.wsl_price_postprocess import flatten_price_record

    rec = {
        "pdf": "wsl_1850_01_15.pdf", "year": 1850, "page": 2,
        "page_key": "wsl_1850_01_15.pdf:p2", "page_type": "market_prices",
        "prices": [
            {"c": "sperm_oil", "lo": 1.15, "hi": 1.25, "u": "gal"},
            {"c": "whale_oil", "lo": 0.45, "hi": None, "u": "gal"},
            {"c": "whalebone", "lo": None, "hi": 0.42, "u": "lb"},
        ],
    }

    rows = flatten_price_record(rec)
    by_c = {r["commodity"]: r for r in rows}

    assert by_c["sperm_oil"]["price_low"] == 1.15
    assert by_c["sperm_oil"]["price_high"] == 1.25
    assert by_c["sperm_oil"]["price_mid"] == pytest.approx(1.20)
    assert by_c["whale_oil"]["price_mid"] == 0.45
    assert by_c["whalebone"]["price_mid"] == 0.42


def test_flatten_normalizes_unit_aliases() -> None:
    """VLM may emit 'gal', 'gallon', 'per_gallon' — all normalize to 'per_gallon'."""
    from src.parsing.wsl_price_postprocess import flatten_price_record

    rec = {
        "pdf": "x.pdf", "year": 1850, "page": 1, "page_key": "x.pdf:p1",
        "prices": [
            {"c": "sperm_oil", "lo": 1.0, "hi": 1.0, "u": "gal"},
            {"c": "whale_oil", "lo": 0.5, "hi": 0.5, "u": "gallon"},
            {"c": "whalebone", "lo": 0.4, "hi": 0.4, "u": "per_lb"},
            {"c": "sperm_oil", "lo": 30, "hi": 30, "u": "bbl"},
        ],
    }

    rows = flatten_price_record(rec)
    units = [r["unit"] for r in rows]
    assert units == ["per_gallon", "per_gallon", "per_lb", "per_barrel"]


def test_flatten_attaches_page_metadata_and_issue_date() -> None:
    """Each row carries pdf, page, page_id, issue_date (parsed from filename), year."""
    from src.parsing.wsl_price_postprocess import flatten_price_record

    rec = {
        "pdf": "wsl_1855_06_30.pdf", "year": 1855, "page": 3,
        "page_key": "wsl_1855_06_30.pdf:p3", "page_type": "market_prices",
        "prices": [{"c": "sperm_oil", "lo": 1.40, "hi": 1.45, "u": "gal"}],
    }

    (row,) = flatten_price_record(rec)

    assert row["pdf"] == "wsl_1855_06_30.pdf"
    assert row["page"] == 3
    assert row["year"] == 1855
    assert row["issue_date"] == "1855-06-30"
    assert row["page_id"].startswith("wsl_pg_")


def test_flatten_generates_stable_price_id() -> None:
    """Same page_key + index + commodity produces the same wsl_price_id across runs."""
    from src.parsing.wsl_price_postprocess import flatten_price_record

    rec = {
        "pdf": "wsl_1850_01_15.pdf", "year": 1850, "page": 2,
        "page_key": "wsl_1850_01_15.pdf:p2",
        "prices": [{"c": "sperm_oil", "lo": 1.15, "hi": 1.20, "u": "gal"}],
    }

    row1 = flatten_price_record(rec)[0]
    row2 = flatten_price_record(rec)[0]
    assert row1["wsl_price_id"] == row2["wsl_price_id"]
    assert row1["wsl_price_id"].startswith("wsl_price_")


def test_empty_prices_yields_no_rows() -> None:
    from src.parsing.wsl_price_postprocess import flatten_price_record

    assert flatten_price_record({"pdf": "x.pdf", "prices": []}) == []
    assert flatten_price_record({"pdf": "x.pdf"}) == []


def test_commodity_aliases_normalize_to_canonical() -> None:
    """'Sperm Oil', 'sperm', 'SP OIL' → 'sperm_oil'. Unknown → None."""
    from src.parsing.wsl_price_postprocess import flatten_price_record

    rec = {
        "pdf": "x.pdf", "year": 1850, "page": 1, "page_key": "x.pdf:p1",
        "prices": [
            {"c": "Sperm Oil", "lo": 1, "hi": 1, "u": "gal"},
            {"c": "sperm", "lo": 1, "hi": 1, "u": "gal"},
            {"c": "SP OIL", "lo": 1, "hi": 1, "u": "gal"},
            {"c": "Whale", "lo": 1, "hi": 1, "u": "gal"},
            {"c": "bone", "lo": 1, "hi": 1, "u": "lb"},
            {"c": "tallow", "lo": 1, "hi": 1, "u": "lb"},
        ],
    }
    commodities = [r["commodity"] for r in flatten_price_record(rec)]
    assert commodities == ["sperm_oil", "sperm_oil", "sperm_oil",
                            "whale_oil", "whalebone", None]


def test_confidence_and_currency_passthrough_with_defaults() -> None:
    """_confidence defaults to 0.5 when missing; currency defaults to USD."""
    from src.parsing.wsl_price_postprocess import flatten_price_record

    rec = {
        "pdf": "x.pdf", "year": 1850, "page": 1, "page_key": "x.pdf:p1",
        "prices": [
            {"c": "sperm_oil", "lo": 1, "hi": 1, "u": "gal",
             "_confidence": 0.92, "cur": "GBP"},
            {"c": "whale_oil", "lo": 1, "hi": 1, "u": "gal"},
        ],
    }
    rows = flatten_price_record(rec)
    assert rows[0]["_confidence"] == 0.92
    assert rows[0]["currency"] == "GBP"
    assert rows[1]["_confidence"] == 0.5
    assert rows[1]["currency"] == "USD"


def test_process_jsonl_file_concatenates_pages(tmp_path: Path) -> None:
    """process_jsonl reads a JSONL file and returns a DataFrame with all rows."""
    from src.parsing.wsl_price_postprocess import process_jsonl_file

    jsonl = tmp_path / "wsl_events_1850.jsonl"
    _write_jsonl(jsonl, [
        {"pdf": "wsl_1850_01_15.pdf", "year": 1850, "page": 2,
         "page_key": "wsl_1850_01_15.pdf:p2",
         "prices": [{"c": "sperm_oil", "lo": 1.15, "hi": 1.20, "u": "gal"},
                    {"c": "whale_oil", "lo": 0.45, "hi": 0.50, "u": "gal"}]},
        {"pdf": "wsl_1850_01_22.pdf", "year": 1850, "page": 2,
         "page_key": "wsl_1850_01_22.pdf:p2",
         "prices": [{"c": "whalebone", "lo": 0.38, "hi": 0.42, "u": "lb"}]},
        {"pdf": "wsl_1850_01_29.pdf", "year": 1850, "page": 1,
         "page_key": "wsl_1850_01_29.pdf:p1",
         "prices": []},
    ])

    df = process_jsonl_file(jsonl)

    assert len(df) == 3
    assert set(df["commodity"]) == {"sperm_oil", "whale_oil", "whalebone"}
    assert "issue_date" in df.columns


def test_run_price_postprocess_writes_parquet(tmp_path: Path) -> None:
    """Full pipeline: JSONL dir → wsl_prices_v4.parquet in staging dir."""
    from src.parsing.wsl_price_postprocess import run_price_postprocess

    extracted = tmp_path / "extracted"
    staging = tmp_path / "staging"
    extracted.mkdir()

    _write_jsonl(extracted / "wsl_events_1850.jsonl", [
        {"pdf": "wsl_1850_01_15.pdf", "year": 1850, "page": 2,
         "page_key": "wsl_1850_01_15.pdf:p2",
         "prices": [{"c": "sperm_oil", "lo": 1.15, "hi": 1.20, "u": "gal"}]},
    ])

    summary = run_price_postprocess(extracted_dir=extracted, staging_dir=staging)

    out = staging / "wsl_prices_v4.parquet"
    assert out.exists()
    df = pd.read_parquet(out)
    assert len(df) == 1
    assert df.iloc[0]["commodity"] == "sperm_oil"
    assert summary["status"] == "success"
    assert summary["n_prices"] == 1
