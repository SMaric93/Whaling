from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _FakeResponse:
    def __init__(self, text: str = "", status_code: int = 200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"status={self.status_code}")


def test_fetch_wsl_issue_catalog_uses_jsonp_rows(monkeypatch):
    from src.download import wsl_pdf_downloader

    def fake_get(url, params=None, headers=None, timeout=None):
        start = int(params["start"])
        if start == 0:
            rows = [
                [1843, 3, 17, "18430317.pdf"],
                [1843, 3, 21, "18430321.pdf"],
            ]
        else:
            rows = [[1843, 3, 28, "18430328.pdf"]]
        payload = {
            "draw": params["draw"],
            "recordsTotal": 3,
            "recordsFiltered": 3,
            "data": rows,
        }
        return _FakeResponse(f"codexWSL({json.dumps(payload)})")

    monkeypatch.setattr(wsl_pdf_downloader.requests, "get", fake_get)

    issues = wsl_pdf_downloader.fetch_wsl_issue_catalog(page_size=2)

    assert [issue["issue_id"] for issue in issues] == [
        "wsl_1843_03_17",
        "wsl_1843_03_21",
        "wsl_1843_03_28",
    ]
    assert issues[0]["url"] == "https://img.mysticseaport.org/images/wsl/18430317.pdf"


def test_download_starbuck_continues_when_pdf_is_unavailable(monkeypatch, tmp_path):
    from src.download import online_sources_downloader

    downloaded_paths = []

    def fake_download_file(url, target_path, timeout=120):
        downloaded_paths.append(target_path.name)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.suffix == ".pdf":
            raise requests.exceptions.HTTPError("401")
        target_path.write_text("reader html", encoding="utf-8")
        return target_path

    class _Manifest:
        def __init__(self):
            self.entries = []

        def add_entry(self, **kwargs):
            self.entries.append(kwargs)

    monkeypatch.setattr(online_sources_downloader, "RAW_STARBUCK", tmp_path)
    monkeypatch.setattr(online_sources_downloader, "download_file", fake_download_file)

    manifest = _Manifest()
    results = online_sources_downloader.download_starbuck(manifest=manifest, force=True)

    assert "pdf" not in results
    assert results["ocr"] == tmp_path / "starbuck_1878_ocr.txt"
    assert results["ocr"].read_text(encoding="utf-8") == "reader html"
    assert downloaded_paths == ["starbuck_1878.pdf", "starbuck_1878_ocr.txt"]
    assert len(manifest.entries) == 1


def test_extract_all_wsl_events_recurses_into_year_subdirectories(monkeypatch, tmp_path):
    from parsing import wsl_event_extractor
    from parsing import wsl_pdf_parser

    pdf_path = tmp_path / "1843" / "wsl_1843_03_17.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.1")

    seen_paths = {}

    def fake_batch_parse(pdf_paths):
        seen_paths["paths"] = pdf_paths
        return ["issue"]

    monkeypatch.setattr(wsl_pdf_parser, "batch_parse_wsl_issues", fake_batch_parse)
    monkeypatch.setattr(wsl_event_extractor, "extract_events_from_issue", lambda issue: ["event"])
    monkeypatch.setattr(
        wsl_event_extractor,
        "events_to_dataframe",
        lambda events: pd.DataFrame({"events": [len(events)]}),
    )

    df = wsl_event_extractor.extract_all_wsl_events(tmp_path)

    assert seen_paths["paths"] == [pdf_path]
    assert df.loc[0, "events"] == 1
