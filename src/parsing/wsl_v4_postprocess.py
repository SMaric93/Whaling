"""
WSL V4 Post-Processor — Consolidate VLM extraction output into analysis-ready tables.

Reads per-year JSONL files from `data/extracted/`, flattens page-level records
into event-level rows, derives `issue_date` from PDF filenames, generates stable
event IDs, and outputs to `data/staging/` for downstream pipeline consumption.

Output schema bridges the HPCC VLM extraction (V4) to the existing pipeline:
    - wsl_events_v4.parquet:   event-level table (one row per event mention)
    - wsl_pages_v4.parquet:    page-level metadata table
    - wsl_postprocess_manifest.json: processing metadata

Usage:
    python -m src.parsing.wsl_v4_postprocess              # process all years
    python -m src.parsing.wsl_v4_postprocess --year 1850  # single year
"""

import hashlib
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent.parent
EXTRACTED_DIR = PROJECT_ROOT / "data" / "extracted"
STAGING_DIR = PROJECT_ROOT / "data" / "staging"

EXTRACTOR_VERSION = "v4.0"

# Event type canonical mapping (V4 short codes → pipeline labels)
# Includes non-standard types found in V3/V4 comparison analysis
EVENT_TYPE_MAP = {
    "dep": "DEPARTURE",
    "arr": "ARRIVAL",
    "spk": "SPOKEN_WITH",
    "rpt": "REPORTED_AT",
    "inp": "IN_PORT",
    "wrk": "WRECK",
    "cnd": "CONDEMNED",
    "cap": "CAPTURED",
    "ret": "RETURNED_HOME",
    "sold": "SOLD",
    "lost": "LOSS",
    # Non-standard aliases (12.7K events / 1.1% of corpus)
    "sail": "DEPARTURE",
    "sailed": "DEPARTURE",
    "to sail": "DEPARTURE",
    "sld": "DEPARTURE",
    "rep": "REPORTED_AT",
    "last report": "REPORTED_AT",
    "inport": "IN_PORT",
    "in port": "IN_PORT",
    "for sale": "SOLD",
    "arrive": "ARRIVAL",
    "arrived": "ARRIVAL",
    "spoke": "SPOKEN_WITH",
    "burned": "WRECK",
    "condemned": "CONDEMNED",
    "wrecked": "WRECK",
    "fit": "OTHER",
    "purchased": "SOLD",
}

VESSEL_TYPE_CANONICAL = {
    "ship": "ship",
    "bark": "bark",
    "brig": "brig",
    "sch": "schooner",
    "schooner": "schooner",
    "steamer": "steamer",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("wsl_v4_postprocess")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PDF_DATE_RE = re.compile(r"wsl_(\d{4})_(\d{2})_(\d{2})\.pdf$")


def parse_issue_date(pdf_name: str) -> Optional[str]:
    """Derive ISO date from PDF filename like wsl_1850_01_01.pdf."""
    m = _PDF_DATE_RE.search(pdf_name)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


def stable_event_id(page_key: str, event_idx: int, vessel: str, event_type: str) -> str:
    """Deterministic event ID from page key + position + content hash."""
    seed = f"{page_key}|{event_idx}|{vessel or ''}|{event_type or ''}"
    h = hashlib.sha256(seed.encode()).hexdigest()[:16]
    return f"wsl_v4_{h}"


def stable_page_id(page_key: str) -> str:
    """Deterministic page ID from page key."""
    h = hashlib.sha256(page_key.encode()).hexdigest()[:12]
    return f"wsl_pg_{h}"


# ---------------------------------------------------------------------------
# Flatten one page record → list of event dicts
# ---------------------------------------------------------------------------

def flatten_page_record(rec: dict) -> Tuple[dict, List[dict]]:
    """
    Flatten a page-level JSONL record into (page_meta, list_of_event_dicts).

    Preserves all raw fields and adds derived columns.
    """
    pdf = rec.get("pdf", "")
    year = rec.get("year")
    page = rec.get("page")
    page_key = rec.get("page_key", f"{pdf}:p{page}")
    issue_date = parse_issue_date(pdf)

    # Page-level metadata
    page_meta = {
        "page_id": stable_page_id(page_key),
        "page_key": page_key,
        "pdf": pdf,
        "year": year,
        "page": page,
        "issue_date": issue_date,
        "page_type": rec.get("page_type"),
        "page_route": rec.get("page_route"),
        "n_events": rec.get("n_events", 0),
        "status": rec.get("status"),
        "llm_time_s": rec.get("llm_time_s"),
        "output_chars": rec.get("output_chars"),
        "extracted_at": rec.get("extracted_at"),
        "extractor_version": EXTRACTOR_VERSION,
    }

    # Add preprocessing metadata if present
    pp = rec.get("preprocessing")
    if isinstance(pp, dict):
        page_meta["pp_stages"] = ",".join(pp.get("stages", []))
        page_meta["pp_banded"] = pp.get("banding", {}).get("banded", False)
        page_meta["pp_n_bands"] = pp.get("banding", {}).get("n_bands")
        page_meta["pp_cleaned"] = any(
            s.get("cleaned", False) for s in pp.get("cleanup", [])
            if isinstance(s, dict)
        )

    events = rec.get("events", [])
    event_rows = []

    for idx, ev in enumerate(events):
        # -- Core fields --
        vessel_name = ev.get("vessel_name", ev.get("v", ""))
        vessel_type_raw = ev.get("vessel_type", ev.get("t"))
        captain = ev.get("captain", ev.get("c"))
        agent = ev.get("agent")
        event_type_raw = ev.get("event_type", ev.get("e"))
        port = ev.get("port", ev.get("p"))
        date_str = ev.get("date", ev.get("d"))
        home_port = ev.get("home_port", ev.get("hp"))
        destination = ev.get("destination", ev.get("dest"))
        remarks = ev.get("remarks")

        # -- Cargo --
        sp = ev.get("oil_sperm_bbls", ev.get("sp"))
        wh = ev.get("oil_whale_bbls", ev.get("wh"))
        bn = ev.get("bone_lbs", ev.get("bn"))
        days_out = ev.get("days_out")

        # -- Coordinates --
        lat = ev.get("latitude", ev.get("lat"))
        lon = ev.get("longitude", ev.get("lon"))

        # -- Reported-by --
        reported_by = ev.get("reported_by")

        # -- Quality fields --
        raw = ev.get("_raw", {})
        flags = ev.get("_flags", [])
        confidence = ev.get("_confidence", 0.5)

        # -- Normalize event type --
        event_type_canon = EVENT_TYPE_MAP.get(
            str(event_type_raw).lower().strip() if event_type_raw else "",
            "OTHER",
        )

        # -- Normalize vessel type --
        vessel_type_canon = VESSEL_TYPE_CANONICAL.get(
            str(vessel_type_raw).lower().strip() if vessel_type_raw else "",
        )

        # -- Build row --
        row = {
            "wsl_event_id": stable_event_id(page_key, idx, vessel_name, event_type_raw),
            "page_id": page_meta["page_id"],
            "page_key": page_key,
            "pdf": pdf,
            "year": year,
            "page": page,
            "event_idx": idx,
            "issue_date": issue_date,
            "page_type": rec.get("page_type"),
            "page_route": rec.get("page_route"),
            # Core event fields
            "vessel_name": vessel_name,
            "vessel_name_raw": ev.get("vessel_name_raw", raw.get("v", vessel_name)),
            "vessel_type": vessel_type_canon,
            "vessel_type_raw": vessel_type_raw,
            "captain": captain,
            "captain_raw": ev.get("captain_raw", raw.get("c", captain)),
            "agent": agent,
            "agent_raw": ev.get("agent_raw", raw.get("agent", agent)),
            "event_type": event_type_canon,
            "event_type_raw": event_type_raw,
            "port": port,
            "port_raw": ev.get("port_raw", raw.get("p", port)),
            "date": date_str,
            "date_raw": ev.get("date_raw", raw.get("d", date_str)),
            "home_port": home_port,
            "home_port_raw": ev.get("home_port_raw", raw.get("hp", home_port)),
            "destination": destination,
            "remarks": remarks,
            "reported_by": reported_by,
            # Cargo
            "oil_sperm_bbls": _safe_int(sp),
            "oil_whale_bbls": _safe_int(wh),
            "bone_lbs": _safe_int(bn),
            "days_out": _safe_int(days_out),
            # Coordinates
            "latitude": _safe_float(lat),
            "longitude": _safe_float(lon),
            # Quality
            "_flags": json.dumps(flags) if flags else None,
            "_confidence": float(confidence) if confidence is not None else None,
            "_raw_json": json.dumps(raw) if raw else None,
            # Contamination flag (vessel_is_port_name: 18K events / 1.6%)
            "_vessel_is_port_name": any(
                str(fl).startswith("vessel_is_port_name:") for fl in (flags or [])
            ),
        }
        event_rows.append(row)

    return page_meta, event_rows


def _safe_int(v) -> Optional[int]:
    """Safely convert to int, returning None on failure."""
    if v is None:
        return None
    try:
        val = int(float(v))
        return val if -1_000_000 < val < 1_000_000 else None
    except (ValueError, TypeError):
        return None


def _safe_float(v) -> Optional[float]:
    """Safely convert to float, returning None on failure."""
    if v is None:
        return None
    try:
        val = float(v)
        return val if np.isfinite(val) else None
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Process one JSONL file
# ---------------------------------------------------------------------------

def process_year_file(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read a per-year JSONL file and return (events_df, pages_df).
    """
    page_rows = []
    event_rows = []

    with open(path, "r") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                log.warning(f"{path.name}:{line_no}: JSON decode error: {e}")
                continue

            page_meta, events = flatten_page_record(rec)
            page_rows.append(page_meta)
            event_rows.extend(events)

    events_df = pd.DataFrame(event_rows) if event_rows else pd.DataFrame()
    pages_df = pd.DataFrame(page_rows) if page_rows else pd.DataFrame()

    return events_df, pages_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_postprocess(
    extracted_dir: Optional[Path] = None,
    staging_dir: Optional[Path] = None,
    years: Optional[List[int]] = None,
) -> Dict:
    """
    Consolidate V4 extraction JSONL files into analysis-ready parquets.

    Args:
        extracted_dir: Directory containing wsl_events_YYYY.jsonl files
        staging_dir: Output directory for staging parquets
        years: Optional list of years to process (default: all)

    Returns:
        dict with processing summary
    """
    extracted_dir = extracted_dir or EXTRACTED_DIR
    staging_dir = staging_dir or STAGING_DIR
    staging_dir.mkdir(parents=True, exist_ok=True)

    # Discover JSONL files
    pattern = "wsl_events_*.jsonl"
    jsonl_files = sorted(extracted_dir.glob(pattern))
    if not jsonl_files:
        log.error(f"No JSONL files found in {extracted_dir}")
        return {"status": "error", "message": "no files found"}

    # Filter by year if specified
    if years:
        year_set = set(years)
        jsonl_files = [
            f for f in jsonl_files
            if int(f.stem.replace("wsl_events_", "")) in year_set
        ]

    log.info(f"Processing {len(jsonl_files)} year files from {extracted_dir}")

    all_events = []
    all_pages = []
    year_stats = []

    for jf in jsonl_files:
        year_str = jf.stem.replace("wsl_events_", "")
        log.info(f"  {year_str}...")

        events_df, pages_df = process_year_file(jf)
        n_events = len(events_df)
        n_pages = len(pages_df)

        if n_events > 0:
            all_events.append(events_df)
        if n_pages > 0:
            all_pages.append(pages_df)

        year_stats.append({
            "year": int(year_str),
            "n_pages": n_pages,
            "n_events": n_events,
            "file_size_mb": round(jf.stat().st_size / 1_048_576, 1),
        })

    # Concatenate
    if not all_events:
        log.error("No events extracted from any file")
        return {"status": "error", "message": "no events"}

    events = pd.concat(all_events, ignore_index=True)
    pages = pd.concat(all_pages, ignore_index=True)

    # Type coercions
    for col in ["oil_sperm_bbls", "oil_whale_bbls", "bone_lbs", "days_out"]:
        if col in events.columns:
            events[col] = events[col].astype("Int64")  # nullable int

    if "year" in events.columns:
        events["year"] = events["year"].astype("Int64")
    if "page" in events.columns:
        events["page"] = events["page"].astype("Int64")
    if "event_idx" in events.columns:
        events["event_idx"] = events["event_idx"].astype("Int64")

    # Parse issue_date to datetime
    if "issue_date" in events.columns:
        events["issue_date"] = pd.to_datetime(events["issue_date"], errors="coerce")
    if "issue_date" in pages.columns:
        pages["issue_date"] = pd.to_datetime(pages["issue_date"], errors="coerce")

    # Sort deterministically
    events = events.sort_values(
        ["year", "pdf", "page", "event_idx"], ignore_index=True
    )
    pages = pages.sort_values(
        ["year", "pdf", "page"], ignore_index=True
    )

    # Save
    events_path = staging_dir / "wsl_events_v4.parquet"
    pages_path = staging_dir / "wsl_pages_v4.parquet"

    events.to_parquet(events_path, index=False)
    pages.to_parquet(pages_path, index=False)

    # Also save CSV for inspection (limited to 10k rows)
    events.head(10_000).to_csv(
        staging_dir / "wsl_events_v4_sample.csv", index=False
    )

    # Summary statistics
    summary = {
        "status": "success",
        "extractor_version": EXTRACTOR_VERSION,
        "processed_at": datetime.now().isoformat(),
        "n_years": len(year_stats),
        "n_pages": len(pages),
        "n_events": len(events),
        "years_range": [int(events["year"].min()), int(events["year"].max())],
        "event_types": events["event_type"].value_counts().to_dict(),
        "page_types": pages["page_type"].value_counts().to_dict() if "page_type" in pages.columns else {},
        "vessel_types": events["vessel_type"].value_counts(dropna=False).head(10).to_dict(),
        "confidence_stats": {
            "mean": round(float(events["_confidence"].mean()), 3),
            "median": round(float(events["_confidence"].median()), 3),
            "p05": round(float(events["_confidence"].quantile(0.05)), 3),
            "p95": round(float(events["_confidence"].quantile(0.95)), 3),
        } if "_confidence" in events.columns else {},
        "home_port_top10": events["home_port"].value_counts().head(10).to_dict(),
        "year_stats": year_stats,
        "output_files": {
            "events": str(events_path),
            "pages": str(pages_path),
        },
    }

    # Missing years check
    all_years = set(range(1843, 1915))
    present_years = set(s["year"] for s in year_stats)
    missing = sorted(all_years - present_years)
    if missing:
        summary["missing_years"] = missing
        log.warning(f"Missing years: {missing}")

    # Save manifest
    manifest_path = staging_dir / "wsl_v4_postprocess_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    log.info(f"\n{'='*60}")
    log.info(f"POST-PROCESSING COMPLETE")
    log.info(f"{'='*60}")
    log.info(f"  Events:  {len(events):,} rows → {events_path}")
    log.info(f"  Pages:   {len(pages):,} rows → {pages_path}")
    log.info(f"  Years:   {summary['years_range'][0]}–{summary['years_range'][1]} ({len(year_stats)} files)")
    if missing:
        log.info(f"  Missing: {missing}")
    log.info(f"  Manifest: {manifest_path}")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Post-process V4 WSL extraction output")
    parser.add_argument("--year", type=int, help="Process a single year")
    parser.add_argument("--extracted-dir", type=str, help="Override extracted dir")
    parser.add_argument("--staging-dir", type=str, help="Override staging dir")
    args = parser.parse_args()

    years = [args.year] if args.year else None
    extracted = Path(args.extracted_dir) if args.extracted_dir else None
    staging = Path(args.staging_dir) if args.staging_dir else None

    summary = run_postprocess(
        extracted_dir=extracted,
        staging_dir=staging,
        years=years,
    )

    if summary.get("status") == "success":
        print(f"\n✓ {summary['n_events']:,} events from {summary['n_years']} years")
    else:
        print(f"\n✗ {summary.get('message', 'unknown error')}")
        sys.exit(1)
