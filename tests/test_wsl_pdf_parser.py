from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_parse_wsl_issue_uses_ocr_for_low_confidence_pages(monkeypatch, tmp_path):
    from src.parsing import wsl_pdf_parser

    pdf_path = tmp_path / "wsl_1843_03_17.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    low_conf_page = wsl_pdf_parser.WSLPage(
        page_number=1,
        text="",
        extraction_method="text_layer",
        confidence=0.1,
    )
    ocr_page = wsl_pdf_parser.WSLPage(
        page_number=1,
        text="July 20, sailed from Tahiti",
        extraction_method="ocr_rapidocr",
        confidence=0.82,
    )

    monkeypatch.setattr(wsl_pdf_parser, "extract_text_layer", lambda path: [low_conf_page])
    monkeypatch.setattr(wsl_pdf_parser, "HAS_OCR", True)
    monkeypatch.setattr(wsl_pdf_parser, "ocr_page", lambda path, page_number: ocr_page)

    issue = wsl_pdf_parser.parse_wsl_issue(pdf_path, ocr_threshold=0.3)

    assert len(issue.pages) == 1
    assert issue.pages[0].text == "July 20, sailed from Tahiti"
    assert issue.pages[0].extraction_method == "ocr_rapidocr"
    assert issue.avg_confidence == 0.82


def test_parse_wsl_issue_batches_rapidocr_for_multiple_low_confidence_pages(monkeypatch, tmp_path):
    from src.parsing import wsl_pdf_parser

    pdf_path = tmp_path / "wsl_1843_03_17.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    pages = [
        wsl_pdf_parser.WSLPage(1, "", "text_layer", 0.1),
        wsl_pdf_parser.WSLPage(2, "", "text_layer", 0.2),
    ]

    monkeypatch.setattr(wsl_pdf_parser, "extract_text_layer", lambda path: pages.copy())
    monkeypatch.setattr(wsl_pdf_parser, "HAS_OCR", True)
    monkeypatch.setattr(wsl_pdf_parser, "HAS_RAPIDOCR", True)
    monkeypatch.setattr(wsl_pdf_parser, "HAS_TESSERACT_OCR", False)
    monkeypatch.setattr(
        wsl_pdf_parser,
        "_ocr_pages_with_rapidocr",
        lambda path, page_numbers: {
            1: wsl_pdf_parser.WSLPage(1, "line one", "ocr_rapidocr", 0.8),
            2: wsl_pdf_parser.WSLPage(2, "line two", "ocr_rapidocr_retry", 0.85),
        },
    )

    issue = wsl_pdf_parser.parse_wsl_issue(pdf_path, ocr_threshold=0.3)

    assert [page.text for page in issue.pages] == ["line one", "line two"]
    assert [page.extraction_method for page in issue.pages] == ["ocr_rapidocr", "ocr_rapidocr_retry"]


def test_prefer_ocr_page_uses_better_confidence_and_text_coverage():
    from src.parsing.wsl_pdf_parser import WSLPage, _prefer_ocr_page

    primary = WSLPage(1, "short text", "ocr_rapidocr", 0.48)
    retry = WSLPage(1, "much better text output", "ocr_rapidocr_retry", 0.62)

    chosen = _prefer_ocr_page(primary, retry)

    assert chosen is retry


def test_ocr_boxes_to_lines_groups_cells_into_rows():
    from src.parsing.wsl_pdf_parser import _ocr_boxes_to_lines

    ocr_result = [
        ([[10, 10], [50, 10], [50, 20], [10, 20]], "Abigefl", "0.9"),
        ([[60, 12], [110, 12], [110, 22], [60, 22]], "310", "0.9"),
        ([[120, 11], [200, 11], [200, 21], [120, 21]], "July 28", "0.9"),
        ([[10, 40], [60, 40], [60, 50], [10, 50]], "Falcon", "0.9"),
        ([[70, 42], [140, 42], [140, 52], [70, 52]], "In port", "0.9"),
    ]

    lines = _ocr_boxes_to_lines(ocr_result)

    assert lines == ["Abigefl 310 July 28", "Falcon In port"]
