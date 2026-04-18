"""WSL VLM price post-processor: JSONL → price time-series parquet."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


_PDF_DATE_RE = re.compile(r"wsl_(\d{4})_(\d{2})_(\d{2})\.pdf$")


UNIT_ALIASES = {
    "gal": "per_gallon", "gallon": "per_gallon", "gallons": "per_gallon",
    "per_gallon": "per_gallon", "per gallon": "per_gallon",
    "lb": "per_lb", "lbs": "per_lb", "pound": "per_lb", "pounds": "per_lb",
    "per_lb": "per_lb", "per lb": "per_lb", "per pound": "per_lb",
    "bbl": "per_barrel", "barrel": "per_barrel",
    "per_barrel": "per_barrel", "per barrel": "per_barrel",
}


COMMODITY_ALIASES = {
    "sperm_oil": "sperm_oil",
    "sperm oil": "sperm_oil",
    "sperm": "sperm_oil",
    "sp oil": "sperm_oil",
    "sp": "sperm_oil",
    "whale_oil": "whale_oil",
    "whale oil": "whale_oil",
    "whale": "whale_oil",
    "wh oil": "whale_oil",
    "whalebone": "whalebone",
    "whale bone": "whalebone",
    "bone": "whalebone",
    "bones": "whalebone",
}


def _normalize_unit(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    return UNIT_ALIASES.get(str(raw).strip().lower())


def _normalize_commodity(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    return COMMODITY_ALIASES.get(str(raw).strip().lower())


def _price_mid(low: Optional[float], high: Optional[float]) -> Optional[float]:
    if low is not None and high is not None:
        return (low + high) / 2
    return low if low is not None else high


def _parse_issue_date(pdf_name: str) -> Optional[str]:
    m = _PDF_DATE_RE.search(pdf_name or "")
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


def _stable_page_id(page_key: str) -> str:
    h = hashlib.sha256(page_key.encode()).hexdigest()[:12]
    return f"wsl_pg_{h}"


def _stable_price_id(page_key: str, idx: int, commodity: str) -> str:
    seed = f"{page_key}|{idx}|{commodity or ''}"
    h = hashlib.sha256(seed.encode()).hexdigest()[:16]
    return f"wsl_price_{h}"


def flatten_price_record(rec: dict) -> List[dict]:
    """Flatten one page-level VLM record into one row per price quote."""
    prices = rec.get("prices") or []
    if not prices:
        return []

    pdf = rec.get("pdf", "")
    year = rec.get("year")
    page = rec.get("page")
    page_key = rec.get("page_key") or f"{pdf}:p{page}"
    issue_date = _parse_issue_date(pdf)
    page_id = _stable_page_id(page_key)

    rows = []
    for idx, p in enumerate(prices):
        commodity = _normalize_commodity(p.get("c") or p.get("commodity"))
        low = p.get("lo", p.get("price_low"))
        high = p.get("hi", p.get("price_high"))
        rows.append({
            "wsl_price_id": _stable_price_id(page_key, idx, commodity or ""),
            "page_id": page_id,
            "page_key": page_key,
            "pdf": pdf,
            "year": year,
            "page": page,
            "issue_date": issue_date,
            "commodity": commodity,
            "price_low": low,
            "price_high": high,
            "price_mid": _price_mid(low, high),
            "unit": _normalize_unit(p.get("u") or p.get("unit")),
            "currency": p.get("cur") or p.get("currency") or "USD",
            "_confidence": p.get("_confidence", 0.5),
        })
    return rows


def process_jsonl_file(path: Path) -> pd.DataFrame:
    """Parse one per-year JSONL file, return a DataFrame of price rows."""
    rows: List[dict] = []
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("%s:%s JSON decode error: %s", path.name, line_no, e)
                continue
            rows.extend(flatten_price_record(rec))
    return pd.DataFrame(rows)


def run_price_postprocess(
    extracted_dir: Path,
    staging_dir: Path,
    years: Optional[List[int]] = None,
) -> Dict:
    """Consolidate per-year VLM JSONL files into a single price parquet."""
    extracted_dir = Path(extracted_dir)
    staging_dir = Path(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(extracted_dir.glob("wsl_events_*.jsonl"))
    if not jsonl_files:
        return {"status": "error", "message": "no files found", "n_prices": 0}

    if years:
        year_set = set(years)
        jsonl_files = [
            f for f in jsonl_files
            if int(f.stem.replace("wsl_events_", "")) in year_set
        ]

    frames = []
    for jf in jsonl_files:
        df = process_jsonl_file(jf)
        if not df.empty:
            frames.append(df)

    prices = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if "issue_date" in prices.columns:
        prices["issue_date"] = pd.to_datetime(prices["issue_date"], errors="coerce")

    out = staging_dir / "wsl_prices_v4.parquet"
    prices.to_parquet(out, index=False)

    summary = {
        "status": "success",
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "n_files": len(jsonl_files),
        "n_prices": len(prices),
        "output": str(out),
    }
    if not prices.empty:
        summary["commodity_counts"] = prices["commodity"].value_counts(dropna=False).to_dict()
    return summary
