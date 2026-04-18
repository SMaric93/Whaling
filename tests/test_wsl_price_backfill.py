"""Tests for the HPCC price-backfill candidate enumerator."""

from __future__ import annotations

import json
from pathlib import Path


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def test_enumerate_reads_events_jsonl_and_filters_by_page_type(tmp_path: Path) -> None:
    """Only mixed/sparse pages are candidates; shipping_table and skip are excluded."""
    from src.parsing.wsl_price_backfill import enumerate_candidates

    events_path = tmp_path / "wsl_events_1850.jsonl"
    _write_jsonl(events_path, [
        {"pdf": "wsl_1850_01_01.pdf", "page": 1, "page_key": "wsl_1850_01_01.pdf:p1",
         "page_type": "skip", "status": "skip"},
        {"pdf": "wsl_1850_01_01.pdf", "page": 2, "page_key": "wsl_1850_01_01.pdf:p2",
         "page_type": "shipping_table"},
        {"pdf": "wsl_1850_01_01.pdf", "page": 3, "page_key": "wsl_1850_01_01.pdf:p3",
         "page_type": "mixed"},
        {"pdf": "wsl_1850_01_08.pdf", "page": 4, "page_key": "wsl_1850_01_08.pdf:p4",
         "page_type": "sparse"},
    ])

    cands = list(enumerate_candidates(events_path))
    keys = [c["page_key"] for c in cands]
    assert keys == ["wsl_1850_01_01.pdf:p3", "wsl_1850_01_08.pdf:p4"]


def test_enumerate_dedupes_against_existing_price_sidecar(tmp_path: Path) -> None:
    """Pages already in wsl_prices_YYYY.jsonl are skipped."""
    from src.parsing.wsl_price_backfill import enumerate_candidates

    events = tmp_path / "wsl_events_1850.jsonl"
    _write_jsonl(events, [
        {"pdf": "a.pdf", "page": 3, "page_key": "a.pdf:p3", "page_type": "mixed"},
        {"pdf": "b.pdf", "page": 4, "page_key": "b.pdf:p4", "page_type": "sparse"},
    ])
    already = tmp_path / "wsl_prices_1850.jsonl"
    _write_jsonl(already, [
        {"page_key": "a.pdf:p3", "prices": []},
    ])

    cands = list(enumerate_candidates(events, existing_prices=already))
    assert [c["page_key"] for c in cands] == ["b.pdf:p4"]


def test_enumerate_skips_error_status_rows(tmp_path: Path) -> None:
    """Rows with status='error'/'empty_response' in events JSONL aren't candidates."""
    from src.parsing.wsl_price_backfill import enumerate_candidates

    events = tmp_path / "wsl_events_1850.jsonl"
    _write_jsonl(events, [
        {"pdf": "a.pdf", "page": 3, "page_key": "a.pdf:p3", "page_type": "mixed",
         "status": "ok"},
        {"pdf": "b.pdf", "page": 4, "page_key": "b.pdf:p4", "page_type": "mixed",
         "status": "error"},
        {"pdf": "c.pdf", "page": 5, "page_key": "c.pdf:p5", "page_type": "mixed"},
    ])

    keys = [c["page_key"] for c in enumerate_candidates(events)]
    assert keys == ["a.pdf:p3", "c.pdf:p5"]


def test_enumerate_handles_blank_lines_and_bad_json(tmp_path: Path) -> None:
    from src.parsing.wsl_price_backfill import enumerate_candidates

    events = tmp_path / "wsl_events_1850.jsonl"
    with open(events, "w") as f:
        f.write("\n")
        f.write('{"pdf": "a.pdf", "page": 3, "page_key": "a.pdf:p3", "page_type": "mixed"}\n')
        f.write("not-json\n")
        f.write("\n")

    keys = [c["page_key"] for c in enumerate_candidates(events)]
    assert keys == ["a.pdf:p3"]


def test_enumerate_nonexistent_file_yields_nothing(tmp_path: Path) -> None:
    from src.parsing.wsl_price_backfill import enumerate_candidates
    assert list(enumerate_candidates(tmp_path / "missing.jsonl")) == []


def test_parse_vlm_content_plain_json() -> None:
    from src.parsing.wsl_price_backfill import parse_vlm_content
    out = parse_vlm_content('{"prices": [{"c": "sperm_oil", "lo": 1.1, "hi": 1.2, "u": "gal"}]}')
    assert out["prices"][0]["c"] == "sperm_oil"


def test_parse_vlm_content_strips_json_code_fences() -> None:
    from src.parsing.wsl_price_backfill import parse_vlm_content
    wrapped = '```json\n{"prices": []}\n```'
    assert parse_vlm_content(wrapped) == {"prices": []}


def test_parse_vlm_content_strips_plain_code_fences() -> None:
    from src.parsing.wsl_price_backfill import parse_vlm_content
    wrapped = '```\n{"prices": [{"c": "whale_oil"}]}\n```'
    out = parse_vlm_content(wrapped)
    assert out["prices"][0]["c"] == "whale_oil"


def test_parse_vlm_content_garbage_returns_empty_with_flag() -> None:
    from src.parsing.wsl_price_backfill import parse_vlm_content
    out = parse_vlm_content("this is not json at all")
    assert out["prices"] == []
    assert out["_parse_error"] is True
    assert "not json" in out["_raw"]


def test_parse_vlm_content_ensures_prices_key_always_present() -> None:
    """Even a valid response without 'prices' key normalizes to 'prices': []."""
    from src.parsing.wsl_price_backfill import parse_vlm_content
    out = parse_vlm_content('{"events": []}')
    assert out["prices"] == []


def test_process_one_dry_run_emits_realistic_record_without_network() -> None:
    """dry_run=True returns a record shaped like a real one, no PDF/HTTP needed."""
    from src.parsing.wsl_price_backfill import process_one

    rec = process_one(
        pdf_path=Path("/nonexistent/wsl_1850_01_15.pdf"),
        pg_idx=4, page_key="wsl_1850_01_15.pdf:p4", year=1850,
        vllm_url="http://unused", dry_run=True,
    )

    assert rec["page_key"] == "wsl_1850_01_15.pdf:p4"
    assert rec["pdf"] == "wsl_1850_01_15.pdf"
    assert rec["page"] == 4
    assert rec["year"] == 1850
    assert rec["status"] == "ok"
    assert rec["n_prices"] >= 1
    assert rec["prices"][0]["c"] in {"sperm_oil", "whale_oil", "whalebone"}
    assert rec["prompt_tag"].endswith("_dry_run")


def test_backfill_year_writes_sidecar_from_events(tmp_path: Path) -> None:
    """backfill_year writes wsl_prices_YYYY.jsonl with one row per candidate."""
    from src.parsing.wsl_price_backfill import backfill_year

    extracted = tmp_path / "extracted"
    pdf_dir = tmp_path / "pdfs" / "1850"
    extracted.mkdir(parents=True)
    pdf_dir.mkdir(parents=True)

    events = extracted / "wsl_events_1850.jsonl"
    _write_jsonl(events, [
        {"pdf": "a.pdf", "page": 3, "page_key": "a.pdf:p3", "page_type": "mixed"},
        {"pdf": "b.pdf", "page": 4, "page_key": "b.pdf:p4", "page_type": "sparse"},
        {"pdf": "c.pdf", "page": 5, "page_key": "c.pdf:p5", "page_type": "shipping_table"},
    ])
    # Touch fake PDFs so enumerate_candidates finds them (in dry_run we don't read them).
    for name in ("a.pdf", "b.pdf"):
        (pdf_dir / name).write_bytes(b"")

    summary = backfill_year(
        year=1850,
        extracted_dir=extracted,
        pdf_dir=tmp_path / "pdfs",
        vllm_url="http://unused",
        dry_run=True,
    )

    sidecar = extracted / "wsl_prices_1850.jsonl"
    assert sidecar.exists()
    with open(sidecar) as f:
        lines = [json.loads(l) for l in f if l.strip()]
    assert {r["page_key"] for r in lines} == {"a.pdf:p3", "b.pdf:p4"}
    assert summary["year"] == 1850
    assert summary["processed"] == 2
    assert summary["prices"] >= 2


def test_backfill_year_is_idempotent_on_rerun(tmp_path: Path) -> None:
    """Running twice doesn't re-process already-done page_keys."""
    from src.parsing.wsl_price_backfill import backfill_year

    extracted = tmp_path / "extracted"
    pdf_dir = tmp_path / "pdfs" / "1850"
    extracted.mkdir(parents=True)
    pdf_dir.mkdir(parents=True)

    _write_jsonl(extracted / "wsl_events_1850.jsonl", [
        {"pdf": "a.pdf", "page": 3, "page_key": "a.pdf:p3", "page_type": "mixed"},
        {"pdf": "b.pdf", "page": 4, "page_key": "b.pdf:p4", "page_type": "sparse"},
    ])
    for name in ("a.pdf", "b.pdf"):
        (pdf_dir / name).write_bytes(b"")

    s1 = backfill_year(1850, extracted, tmp_path / "pdfs",
                        "http://u", dry_run=True)
    s2 = backfill_year(1850, extracted, tmp_path / "pdfs",
                        "http://u", dry_run=True)

    assert s1["processed"] == 2
    assert s2["processed"] == 0  # nothing left to do

    sidecar = extracted / "wsl_prices_1850.jsonl"
    with open(sidecar) as f:
        n = sum(1 for _ in f)
    assert n == 2  # no duplicates


def test_backfill_year_missing_events_file_returns_zero(tmp_path: Path) -> None:
    from src.parsing.wsl_price_backfill import backfill_year

    extracted = tmp_path / "extracted"
    extracted.mkdir()

    summary = backfill_year(1850, extracted, tmp_path / "pdfs",
                             "http://u", dry_run=True)
    assert summary == {"year": 1850, "processed": 0, "prices": 0}


def test_process_one_dry_run_varies_prices_by_page_key() -> None:
    """Deterministic per page_key so re-runs produce stable fake outputs."""
    from src.parsing.wsl_price_backfill import process_one

    r1 = process_one(Path("/x/a.pdf"), 2, "a.pdf:p2", 1850,
                      "http://x", dry_run=True)
    r2 = process_one(Path("/x/a.pdf"), 2, "a.pdf:p2", 1850,
                      "http://x", dry_run=True)
    assert r1["prices"] == r2["prices"]

    r3 = process_one(Path("/x/b.pdf"), 3, "b.pdf:p3", 1850,
                      "http://x", dry_run=True)
    # different page → at least one different field
    assert r1["prices"] != r3["prices"] or r1["page_key"] != r3["page_key"]
