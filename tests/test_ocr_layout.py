from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.parsing.ocr_layout import OCRTextBox, ocr_boxes_to_lines
from src.parsing.wsl_page_routing import classify_wsl_page_text, text_layer_looks_usable


def test_ocr_boxes_to_lines_groups_rows_and_sorts_left_to_right() -> None:
    boxes = [
        OCRTextBox(points=((60, 10), (90, 10), (90, 20), (60, 20)), text="Bedford"),
        OCRTextBox(points=((10, 10), (50, 10), (50, 20), (10, 20)), text="New"),
        OCRTextBox(points=((55, 40), (80, 40), (80, 50), (55, 50)), text="Allen"),
        OCRTextBox(points=((10, 40), (50, 40), (50, 50), (10, 50)), text="William"),
    ]

    assert ocr_boxes_to_lines(boxes, min_row_threshold=5.0) == [
        "New Bedford",
        "William Allen",
    ]


def test_ocr_boxes_to_lines_supports_descending_row_order() -> None:
    boxes = [
        OCRTextBox(points=((0.10, 0.80), (0.30, 0.80), (0.30, 0.85), (0.10, 0.85)), text="Top"),
        OCRTextBox(points=((0.35, 0.80), (0.55, 0.80), (0.55, 0.85), (0.35, 0.85)), text="Row"),
        OCRTextBox(points=((0.10, 0.20), (0.30, 0.20), (0.30, 0.25), (0.10, 0.25)), text="Bottom"),
        OCRTextBox(points=((0.35, 0.20), (0.55, 0.20), (0.55, 0.25), (0.35, 0.25)), text="Row"),
    ]

    assert ocr_boxes_to_lines(
        boxes,
        row_sort_descending=True,
        min_row_threshold=0.01,
    ) == [
        "Top Row",
        "Bottom Row",
    ]


def test_text_layer_looks_usable_accepts_reasonable_embedded_text() -> None:
    text = "\n".join(
        [
            "ARRIVALS.",
            "Awashonks 310 Capt Wood arrived at New Bedford Mar 15 with 1500 sp.",
            "Montezuma 285 Capt Allen reported at Oahu bound home.",
            "Sperm Oil - 1.24 per gallon. Whale Oil - .63 per gallon.",
        ]
    )

    assert text_layer_looks_usable(text) is True


def test_classify_wsl_page_text_routes_price_and_ad_pages() -> None:
    price_text = "\n".join(
        [
            "Marine Market.",
            "Sperm Oil 1.25 per gallon.",
            "Whale Oil .62 per gallon.",
            "Whalebone 1.80 per pound.",
        ]
    )
    ads_text = "\n".join(
        [
            "Advertisements.",
            "For Sale at this store.",
            "Notice to merchants and insurance offices.",
            "Terms cash on delivery.",
        ]
    )

    price_route = classify_wsl_page_text(price_text)
    ads_route = classify_wsl_page_text(ads_text)

    assert price_route.page_type == "market_prices"
    assert price_route.run_events is False
    assert price_route.run_prices is True
    assert ads_route.page_type == "advertisements"
    assert ads_route.run_events is False
    assert ads_route.run_prices is False
