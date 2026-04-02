from __future__ import annotations

import re
import statistics
from dataclasses import dataclass


PRICE_PATTERNS = [
    r"marine\s+market",
    r"prices?\s+current",
    r"oil\s+market",
    r"new\s+bedford\s+market",
    r"sperm\s+oil",
    r"whale\s+oil",
    r"whale\s*bone",
    r"per\s+gallon",
    r"per\s+pound",
]
EVENT_PATTERNS = [
    r"\bsailed\b",
    r"\bdeparted\b",
    r"\barrived\b",
    r"\bspoken\b",
    r"\bspoke\b",
    r"\breported\b",
    r"\bbound\s+home\b",
    r"\bbound\s+to\b",
    r"\bin\s+port\b",
    r"\bcapt\.?\b",
    r"\bmaster\b",
]
AD_PATTERNS = [
    r"\badvertisements?\b",
    r"\bfor\s+sale\b",
    r"\bwanted\b",
    r"\bcommission\s+merchant",
    r"\bnotice\b",
    r"\binsurance\b",
    r"\bcopartnership\b",
    r"\bterms\b",
    r"\bstore\b",
]
VESSEL_ROW_PATTERN = re.compile(
    r"^[A-Za-z][A-Za-z'&.-]+(?:\s+[A-Za-z][A-Za-z'&.-]+){0,2}\s+\d{2,4}",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class WSLPageRoute:
    page_type: str
    run_events: bool
    run_prices: bool
    skip_reason: str | None = None


def _count_hits(patterns: list[str], text: str) -> int:
    return sum(
        len(re.findall(pattern, text, flags=re.IGNORECASE))
        for pattern in patterns
    )


def _nonempty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def text_layer_looks_usable(text: str) -> bool:
    lines = _nonempty_lines(text)
    if len(lines) < 4:
        return False

    stripped = "\n".join(lines)
    if len(stripped) < 120:
        return False

    printable = sum(1 for char in stripped if char.isprintable() and not char.isspace())
    if printable == 0:
        return False

    alpha = sum(1 for char in stripped if char.isalpha())
    weird = stripped.count("\x0c") + stripped.count("\ufffd")
    avg_line_len = statistics.mean(len(line) for line in lines[:40])

    return (
        alpha / printable >= 0.3
        and weird / max(len(stripped), 1) <= 0.02
        and avg_line_len >= 8
    )


def classify_wsl_page_text(text: str) -> WSLPageRoute:
    lines = _nonempty_lines(text)
    if len(lines) < 3:
        return WSLPageRoute(
            page_type="advertisements",
            run_events=False,
            run_prices=False,
            skip_reason="too_few_text_lines",
        )

    sample = "\n".join(lines[:80])
    price_hits = _count_hits(PRICE_PATTERNS, sample)
    event_hits = _count_hits(EVENT_PATTERNS, sample)
    ad_hits = _count_hits(AD_PATTERNS, sample)
    vessel_rows = sum(1 for line in lines[:80] if VESSEL_ROW_PATTERN.search(line))

    if price_hits >= 3 and event_hits == 0 and vessel_rows == 0:
        return WSLPageRoute("market_prices", run_events=False, run_prices=True)

    if ad_hits >= 3 and price_hits == 0 and event_hits == 0 and vessel_rows == 0:
        return WSLPageRoute(
            page_type="advertisements",
            run_events=False,
            run_prices=False,
            skip_reason="advertisement_keywords",
        )

    if price_hits >= 2 and (event_hits >= 1 or vessel_rows >= 1):
        return WSLPageRoute("mixed", run_events=True, run_prices=True)

    if vessel_rows >= 2 or event_hits >= 2:
        return WSLPageRoute("shipping_list", run_events=True, run_prices=False)

    if price_hits >= 2:
        return WSLPageRoute("market_prices", run_events=False, run_prices=True)

    return WSLPageRoute("mixed", run_events=True, run_prices=price_hits >= 1)
