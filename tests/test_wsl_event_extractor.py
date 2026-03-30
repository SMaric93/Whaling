from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_extract_events_from_text_prefers_single_leading_vessel_for_ocr_rows():
    from src.parsing.wsl_event_extractor import EventType, extract_events_from_text

    row = (
        "Abigefl 310/00x C w Morgen July 28, 39 Pacific "
        "Nov 2, at Oahu bound home 1600 sp"
    )

    events = extract_events_from_text(
        text=row,
        issue_id="wsl_1843_03_17",
        issue_year=1843,
        issue_month=3,
        issue_day=17,
    )

    assert len(events) == 1
    assert events[0].vessel_name_raw == "Abigefl"
    assert events[0].event_type in {EventType.REPORTED_AT, EventType.OTHER}
