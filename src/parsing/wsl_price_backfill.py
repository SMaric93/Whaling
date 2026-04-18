"""Candidate enumeration for the HPCC price backfill pass.

Reads per-year VLM events JSONL, yielding page records that could plausibly
contain commodity prices (pages the classifier labeled ``mixed`` or ``sparse``).
Excludes pages the extractor failed on, and skips anything already re-processed
and sitting in the ``wsl_prices_YYYY.jsonl`` sidecar.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set


PROMPT_TAG = "PROMPT_PRICES_v1"

_DRY_RUN_COMMODITIES = ["sperm_oil", "whale_oil", "whalebone"]


def _cli() -> int:
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("wsl_price_backfill")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--extracted-dir", type=Path, required=True,
                        help="Dir with wsl_events_YYYY.jsonl files")
    parser.add_argument("--pdf-dir", type=Path, required=True,
                        help="Root dir with {year}/*.pdf subdirs")
    parser.add_argument("--vllm-url", type=str,
                        default="http://localhost:8100/v1/chat/completions")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip PDF render + HTTP; emit canned records")
    args = parser.parse_args()

    summary = backfill_year(
        year=args.year,
        extracted_dir=args.extracted_dir,
        pdf_dir=args.pdf_dir,
        vllm_url=args.vllm_url,
        dry_run=args.dry_run,
    )
    log.info("DONE %s", summary)
    return 0


def _dry_run_prices(page_key: str) -> List[Dict[str, Any]]:
    """Deterministic fake prices derived from page_key — stable across runs."""
    seed = int(hashlib.sha256(page_key.encode()).hexdigest()[:8], 16)
    commodity = _DRY_RUN_COMMODITIES[seed % 3]
    base = 0.30 + (seed % 200) / 100  # 0.30 – 2.30
    return [{
        "c": commodity,
        "lo": round(base, 2),
        "hi": round(base + 0.10, 2),
        "u": "gal" if commodity != "whalebone" else "lb",
        "cur": "USD",
        "raw": f"[dry-run] {commodity} {base:.2f}–{base + 0.10:.2f}",
    }]


def process_one(
    pdf_path: Path,
    pg_idx: int,
    page_key: str,
    year: int,
    vllm_url: str,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Render one page, call the VLM, return a serializable price record."""
    t0 = time.time()

    if dry_run:
        prices = _dry_run_prices(page_key)
        return {
            "page_key": page_key,
            "pdf": pdf_path.name,
            "page": pg_idx,
            "year": year,
            "status": "ok" if prices else "empty",
            "n_prices": len(prices),
            "prices": prices,
            "llm_time_s": round(time.time() - t0, 4),
            "prompt_tag": f"{PROMPT_TAG}_dry_run",
        }

    raise NotImplementedError(
        "live VLM calls are driven by the SLURM worker; use dry_run=True locally"
    )


def parse_vlm_content(content: str) -> Dict[str, Any]:
    """Strip optional markdown fences and parse the VLM's JSON response.

    Always returns a dict with a ``prices`` key (possibly empty). On parse
    failure, sets ``_parse_error=True`` and preserves the raw text in ``_raw``.
    """
    stripped = (content or "").strip()
    stripped = stripped.removeprefix("```json").removeprefix("```").strip()
    stripped = stripped.removesuffix("```").strip()
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return {"prices": [], "_parse_error": True, "_raw": (content or "")[:500]}
    if "prices" not in data:
        data["prices"] = []
    return data


PRICE_CANDIDATE_PAGE_TYPES = {"mixed", "sparse"}
SKIP_STATUSES = {"error", "empty_response", "truncated"}


def _load_existing_keys(path: Optional[Path]) -> Set[str]:
    if path is None or not Path(path).exists():
        return set()
    keys: Set[str] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = rec.get("page_key")
            if key:
                keys.add(key)
    return keys


def backfill_year(
    year: int,
    extracted_dir: Path,
    pdf_dir: Path,
    vllm_url: str,
    dry_run: bool = False,
) -> Dict[str, int]:
    """Run the price-backfill pass for one year: enumerate → process → sidecar append."""
    extracted_dir = Path(extracted_dir)
    pdf_dir = Path(pdf_dir)
    events = extracted_dir / f"wsl_events_{year}.jsonl"
    sidecar = extracted_dir / f"wsl_prices_{year}.jsonl"

    if not events.exists():
        return {"year": year, "processed": 0, "prices": 0}

    candidates = list(enumerate_candidates(events, existing_prices=sidecar))
    if not candidates:
        return {"year": year, "processed": 0, "prices": 0}

    total_prices = 0
    with open(sidecar, "a") as fout:
        for cand in candidates:
            pdf_path = pdf_dir / str(year) / cand["pdf"]
            rec = process_one(
                pdf_path=pdf_path,
                pg_idx=cand["page"],
                page_key=cand["page_key"],
                year=year,
                vllm_url=vllm_url,
                dry_run=dry_run,
            )
            fout.write(json.dumps(rec) + "\n")
            total_prices += rec.get("n_prices", 0)

    return {"year": year, "processed": len(candidates), "prices": total_prices}


def enumerate_candidates(
    events_jsonl: Path,
    existing_prices: Optional[Path] = None,
) -> Iterator[dict]:
    """Yield {page_key, pdf, page, page_type} for pages worth re-scanning for prices."""
    events_jsonl = Path(events_jsonl)
    if not events_jsonl.exists():
        return

    already = _load_existing_keys(existing_prices)

    with open(events_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("page_type") not in PRICE_CANDIDATE_PAGE_TYPES:
                continue
            status = rec.get("status")
            if status in SKIP_STATUSES:
                continue
            key = rec.get("page_key")
            if not key or key in already:
                continue
            yield {
                "page_key": key,
                "pdf": rec.get("pdf"),
                "page": rec.get("page"),
                "page_type": rec.get("page_type"),
            }


if __name__ == "__main__":
    raise SystemExit(_cli())
