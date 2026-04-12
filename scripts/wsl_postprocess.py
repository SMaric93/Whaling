#!/usr/bin/env python3
"""
WSL Post-Processing & Validation Pipeline
==========================================

4-layer post-processing + 3-level validation for WSL extraction output.

Usage:
    # Standalone — process an existing JSONL file:
    python scripts/wsl_postprocess.py /path/to/wsl_events.jsonl -o /path/to/cleaned.jsonl

    # As module — import for inline use:
    from wsl_postprocess import post_process_event, validate_hard, validate_section
"""

import re
import json
import argparse
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter

# ═══════════════════════════════════════════════════════════════════════════════
# Constants / Lookup Tables
# ═══════════════════════════════════════════════════════════════════════════════

VALID_EVENT_TYPES = {"dep", "arr", "spk", "rpt", "inp", "wrk"}

EVENT_TYPE_MAP = {
    "in port": "inp", "inport": "inp", "in_port": "inp",
    "sailed": "dep", "sail": "dep", "departure": "dep", "departed": "dep",
    "arrived": "arr", "arrival": "arr",
    "spoken": "spk", "spk'd": "spk",
    "reported": "rpt", "report": "rpt",
    "wreck": "wrk", "wrecked": "wrk", "condemned": "wrk", "lost": "wrk",
}

VALID_VESSEL_TYPES = {"ship", "bark", "brig", "sch"}

VESSEL_TYPE_MAP = {
    "bk": "bark",
    "brk": "bark",
    "bg": "brig",
    "schooner": "sch",
    "ox": None,      # OCR noise
    "rig": None,     # OCR noise
    "steamer": "steamer",  # Keep — rare but valid
}

# Suffixes that indicate vessel type bled into vessel name
VESSEL_TYPE_SUFFIXES = [
    ", bark", ", bk", ", brk", ", ship", ", brig", ", sch", ", schooner",
    " bark", " bk", " ship", " brig", " sch",
]

# Known OCR port corrections (extend as needed)
PORT_CORRECTIONS = {
    # OCR errors
    "antucket": "Nantucket",
    "anticent": "Nantucket",
    "nantkt": "Nantucket",
    "n bedford": "New Bedford",
    "n. bedford": "New Bedford",
    "new bedfored": "New Bedford",
    "nb": "New Bedford",
    "n b": "New Bedford",
    "n london": "New London",
    "n. london": "New London",
    # Port abbreviations (from string_normalizer.py)
    "s harbor": "Sag Harbor",
    "s. harbor": "Sag Harbor",
    "greenpt": "Greenport",
    "p town": "Provincetown",
    "prov town": "Provincetown",
    # Standard whaling ports (canonical forms)
    "edgartown": "Edgartown",
    "sag harbor": "Sag Harbor",
    "cold spring": "Cold Spring",
    "stonington": "Stonington",
    "provincetown": "Provincetown",
    "fairhaven": "Fairhaven",
    "dartmouth": "Dartmouth",
    "westport": "Westport",
    "warren": "Warren",
    "bristol": "Bristol",
    "mystic": "Mystic",
    "greenport": "Greenport",
    "new bedford": "New Bedford",
    "nantucket": "Nantucket",
    "new london": "New London",
    "newport": "Newport",
    "mattapoisett": "Mattapoisett",
    "holmes hole": "Holmes Hole",
    "holmes' hole": "Holmes Hole",
    "olmes' hole": "Holmes Hole",
    "london": "New London",
    "dover": "Edgartown",
    "sandwich": "Sandwich",
    "marion": "Marion",
    "fall river": "Fall River",
}

MONTH_PATTERN = re.compile(
    r"^(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?|Mch)\b", re.IGNORECASE
)

COMPANY_INDICATORS = re.compile(
    r"(&| Co\b| Sons\b| Brothers\b| Bros\b| Jr\b.*&)", re.IGNORECASE
)

YEAR_PATTERN = re.compile(r"\b(\d{2,4})\b")


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 1: Raw Preservation
# ═══════════════════════════════════════════════════════════════════════════════

def preserve_raw(ev):
    """Store raw values before any cleaning (idempotent — skips if already set)."""
    for field in ("vessel_name", "captain", "agent", "date", "port", "home_port"):
        raw_key = f"{field}_raw"
        if raw_key not in ev:
            ev[raw_key] = ev.get(field)
    return ev


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 2: Structural Normalization
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_vessel_type(ev):
    """Normalize vessel type abbreviations and strip junk."""
    vt = ev.get("vessel_type")
    if not vt:
        return ev
    vt_lower = vt.strip().lower()
    if vt_lower in VALID_VESSEL_TYPES:
        ev["vessel_type"] = vt_lower
    elif vt_lower in VESSEL_TYPE_MAP:
        ev["vessel_type"] = VESSEL_TYPE_MAP[vt_lower]
    else:
        # Unknown type — null it out rather than propagate garbage
        ev["vessel_type"] = None
    return ev


def strip_type_from_name(ev):
    """Remove vessel type suffixes that bled into vessel name."""
    name = ev.get("vessel_name", "")
    if not name:
        return ev
    name_lower = name.lower()
    for suffix in VESSEL_TYPE_SUFFIXES:
        if name_lower.endswith(suffix):
            ev["vessel_name"] = name[: -len(suffix)].strip()
            # If we didn't have a vessel type, infer it from the suffix
            if not ev.get("vessel_type"):
                type_str = suffix.strip().lstrip(", ").lower()
                ev["vessel_type"] = VESSEL_TYPE_MAP.get(type_str, type_str)
            break
    # Also handle prefix pattern: "Bark United States"
    for prefix in ("Bark ", "Ship ", "Brig ", "Sch "):
        if name.startswith(prefix):
            ev["vessel_name"] = name[len(prefix):].strip()
            if not ev.get("vessel_type"):
                ev["vessel_type"] = prefix.strip().lower()
                ev["vessel_type"] = VESSEL_TYPE_MAP.get(
                    ev["vessel_type"], ev["vessel_type"]
                )
            break
    return ev


def coerce_oil_values(ev):
    """Convert oil/bone string values to integers where possible."""
    for field in ("oil_sperm_bbls", "oil_whale_bbls", "bone_lbs"):
        val = ev.get(field)
        if val is None:
            continue
        if isinstance(val, (int, float)):
            ev[field] = int(val) if val >= 0 else None
            continue
        if isinstance(val, str):
            s = val.strip().lower()
            # Qualitative values → null
            if s in ("clean", "not stated", "none", "no report", ""):
                ev[field] = None
                if s and s != "":
                    # Preserve the qualitative note in remarks
                    remarks = ev.get("remarks") or ""
                    if s not in remarks.lower():
                        ev["remarks"] = f"{remarks}; {field}={val}".lstrip("; ")
                continue
            # Try to parse as integer (strip non-digits)
            digits = re.sub(r"[^\d]", "", s)
            if digits:
                try:
                    ev[field] = int(digits)
                except ValueError:
                    ev[field] = None
            else:
                ev[field] = None
    return ev


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 3: Semantic Cleaning
# ═══════════════════════════════════════════════════════════════════════════════

def detect_captain_company(ev):
    """If captain field contains a company name, move it to agent."""
    captain = ev.get("captain") or ""
    if not captain:
        return ev
    if COMPANY_INDICATORS.search(captain):
        # This is a company, not a captain
        if not ev.get("agent"):
            ev["agent"] = captain
        ev["captain"] = None
    return ev


def extract_surname(ev):
    """If captain is a full name (first + last), keep only the surname."""
    captain = ev.get("captain") or ""
    if not captain or " " not in captain:
        return ev
    parts = captain.split()
    if len(parts) == 2:
        first, last = parts
        # Likely "J. Smith" or "Chas. Carroll" — take the surname
        if len(first) <= 5 or first.endswith("."):
            ev["captain"] = last
    elif len(parts) == 3:
        # "J. B. Smith" pattern
        if all(len(p) <= 3 or p.endswith(".") for p in parts[:-1]):
            ev["captain"] = parts[-1]
    return ev


def fix_port_ocr(ev):
    """Correct known OCR errors in port and home_port fields."""
    for field in ("port", "home_port"):
        val = ev.get(field)
        if not val:
            continue
        val_lower = val.strip().lower()
        if val_lower in PORT_CORRECTIONS:
            ev[field] = PORT_CORRECTIONS[val_lower]
    return ev


def detect_field_swap(ev):
    """Detect and correct when port and date fields are swapped."""
    port = ev.get("port") or ""
    date = ev.get("date") or ""

    # If port looks like a date (starts with month name)
    if port and MONTH_PATTERN.match(port):
        # And date doesn't look like a date (or looks like coordinates/port)
        if not date or not MONTH_PATTERN.match(date):
            ev["port"], ev["date"] = date or None, port

    # If date contains coordinates but port doesn't
    if date and ("lat" in date.lower() or "lon" in date.lower()):
        if port and not ("lat" in port.lower() or "lon" in port.lower()):
            # Move coordinates to remarks, keep whatever port we have
            remarks = ev.get("remarks") or ""
            ev["remarks"] = f"{remarks}; coords={date}".lstrip("; ")
            ev["date"] = None

    return ev


def clean_date_pollution(ev):
    """Remove narrative text from date field, move to remarks."""
    date = ev.get("date") or ""
    if not date:
        return ev
    # If date is unreasonably long, it has narrative pollution
    if len(date) > 30:
        # Try to extract just the date portion
        match = re.match(
            r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
            r"\.?\s+\d{1,2}(?:st|nd|rd|th|d)?,?\s*(?:\d{2,4})?)",
            date,
            re.IGNORECASE,
        )
        if match:
            clean_date = match.group(1).strip().rstrip(",")
            remainder = date[match.end():].strip().lstrip(",").strip()
            ev["date"] = clean_date
            if remainder:
                remarks = ev.get("remarks") or ""
                ev["remarks"] = f"{remarks}; {remainder}".lstrip("; ")
        # else leave as-is — we can't parse it
    # "In port" is not a date
    if date.lower().strip() in ("in port", "at sea", "not stated"):
        remarks = ev.get("remarks") or ""
        ev["remarks"] = f"{remarks}; status={date}".lstrip("; ")
        ev["date"] = None
    return ev


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 4: Validation
# ═══════════════════════════════════════════════════════════════════════════════

def validate_hard(ev):
    """Check hard constraints. Corrects where possible, flags otherwise."""
    flags = []

    # Event type must be valid
    et = ev.get("event_type", "")
    if et and et not in VALID_EVENT_TYPES:
        # Try normalization before flagging
        normalized = EVENT_TYPE_MAP.get(et.lower())
        if normalized:
            ev["event_type"] = normalized
        elif MONTH_PATTERN.match(et):
            # Date leaked into event_type field
            if not ev.get("date"):
                ev["date"] = et
            ev["event_type"] = "dep"
            flags.append("date_in_event_type_corrected")
        else:
            flags.append(f"invalid_event_type:{et}")

    # Vessel type must be valid (after normalization)
    vt = ev.get("vessel_type")
    if vt and vt not in VALID_VESSEL_TYPES and vt != "steamer":
        flags.append(f"invalid_vessel_type:{vt}")
        ev["vessel_type"] = None

    # Oil/bone must be non-negative
    for field in ("oil_sperm_bbls", "oil_whale_bbls", "bone_lbs"):
        val = ev.get(field)
        if isinstance(val, (int, float)) and val < 0:
            ev[field] = None
            flags.append(f"negative_{field}")

    # Coordinates must be in range
    lat = ev.get("latitude")
    if isinstance(lat, (int, float)) and not (-90 <= lat <= 90):
        ev["latitude"] = None
        flags.append("invalid_latitude")
    lon = ev.get("longitude")
    if isinstance(lon, (int, float)) and not (-180 <= lon <= 180):
        ev["longitude"] = None
        flags.append("invalid_longitude")

    # Vessel name should not be empty
    if not (ev.get("vessel_name") or "").strip():
        flags.append("missing_vessel_name")

    return ev, flags


def validate_section(ev, issue_year=None):
    """Cross-check event against section/issue-level context."""
    flags = []

    # Registry detection: departure dates >2 years before issue date
    date_str = ev.get("date") or ""
    if date_str and issue_year:
        match = YEAR_PATTERN.search(date_str)
        if match:
            yr = int(match.group(1))
            if yr < 100:
                yr += 1800
            if issue_year - yr > 2:
                flags.append("likely_registry_not_weekly")

    # "In port" in port field is a status, not a location
    port = ev.get("port") or ""
    if port.lower().strip() in ("in port", "at sea"):
        flags.append("status_in_port_field")

    return ev, flags


def validate_soft(ev):
    """Compute a confidence score based on field completeness and consistency."""
    score = 1.0
    penalties = []

    # Core fields missing
    if not ev.get("vessel_name"):
        score -= 0.3
        penalties.append("no_vessel_name")
    if not ev.get("event_type"):
        score -= 0.2
        penalties.append("no_event_type")
    if not ev.get("captain") and not ev.get("agent"):
        score -= 0.15
        penalties.append("no_captain_or_agent")
    if not ev.get("date"):
        score -= 0.1
        penalties.append("no_date")
    if not ev.get("port"):
        score -= 0.1
        penalties.append("no_port")

    # Suspect patterns
    captain = ev.get("captain") or ""
    if captain and len(captain) < 2:
        score -= 0.1
        penalties.append("very_short_captain")

    return ev, max(0.0, round(score, 2)), penalties


# ═══════════════════════════════════════════════════════════════════════════════
# Deduplication
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize_for_dedup(s):
    """Normalize a string for dedup comparison."""
    if not s:
        return ""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _normalize_date_for_dedup(d):
    """Normalize date string for dedup (strip ordinals, whitespace)."""
    if not d:
        return ""
    d = re.sub(r"(\d)(st|nd|rd|th|d)\b", r"\1", d.lower())
    return re.sub(r"\s+", " ", d).strip()


def dedup_key(ev):
    """Create a rich dedup key for an event."""
    return (
        ev.get("page_key", ""),
        _normalize_for_dedup(ev.get("vessel_name", ""))[:12],
        _normalize_for_dedup(ev.get("captain", "")),
        ev.get("event_type", ""),
        _normalize_date_for_dedup(ev.get("date", "")),
        _normalize_for_dedup(ev.get("port", "")),
    )


def deduplicate_events(events):
    """Deduplicate events using a richer key than the V1 fuzzy key."""
    seen = set()
    result = []
    for ev in events:
        key = dedup_key(ev)
        if key not in seen:
            seen.add(key)
            result.append(ev)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Full Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def post_process_event(ev, issue_year=None):
    """Run all 4 layers on a single event. Returns (event, flags, confidence)."""
    # Layer 1: Raw preservation
    ev = preserve_raw(ev)

    # Layer 2: Structural normalization
    ev = normalize_vessel_type(ev)
    ev = strip_type_from_name(ev)
    ev = coerce_oil_values(ev)

    # Layer 3: Semantic cleaning
    ev = detect_captain_company(ev)
    ev = extract_surname(ev)
    ev = fix_port_ocr(ev)
    ev = detect_field_swap(ev)
    ev = clean_date_pollution(ev)

    # Layer 4: Validation
    ev, hard_flags = validate_hard(ev)
    ev, section_flags = validate_section(ev, issue_year)
    ev, confidence, soft_penalties = validate_soft(ev)

    all_flags = hard_flags + section_flags
    ev["_flags"] = all_flags if all_flags else None
    ev["_confidence"] = confidence

    return ev


def post_process_page(page_record):
    """Post-process all events in a page record."""
    events = page_record.get("events", [])
    year = page_record.get("year")

    processed = []
    for ev in events:
        ev = post_process_event(ev, issue_year=year)
        processed.append(ev)

    # Deduplicate
    processed = deduplicate_events(processed)

    page_record["events"] = processed
    page_record["n_events"] = len(processed)
    return page_record


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def process_jsonl(input_path, output_path, verbose=False):
    """Process a JSONL file through the full pipeline."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    stats = Counter()
    total_events_in = 0
    total_events_out = 0

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats["json_errors"] += 1
                continue

            n_before = len(record.get("events", []))
            total_events_in += n_before

            record = post_process_page(record)

            n_after = len(record.get("events", []))
            total_events_out += n_after
            stats["pages"] += 1
            stats["dedup_removed"] += n_before - n_after

            # Count flags
            for ev in record.get("events", []):
                for flag in (ev.get("_flags") or []):
                    stats[f"flag:{flag}"] += 1

            fout.write(json.dumps(record) + "\n")

            if verbose and line_num % 100 == 0:
                print(f"  Processed {line_num} lines, {total_events_out} events...",
                      file=sys.stderr)

    return stats, total_events_in, total_events_out


def main():
    parser = argparse.ArgumentParser(
        description="Post-process WSL extraction JSONL output"
    )
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("-o", "--output", help="Output JSONL file (default: input with _cleaned suffix)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(
            input_path.stem + "_cleaned" + input_path.suffix
        )

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")

    stats, events_in, events_out = process_jsonl(
        input_path, output_path, verbose=args.verbose
    )

    print(f"\nResults:")
    print(f"  Pages processed:  {stats['pages']:,}")
    print(f"  Events in:        {events_in:,}")
    print(f"  Events out:       {events_out:,}")
    print(f"  Dedup removed:    {stats['dedup_removed']:,}")
    print(f"  JSON errors:      {stats.get('json_errors', 0):,}")
    print(f"\nValidation flags:")
    for key in sorted(stats):
        if key.startswith("flag:"):
            print(f"  {key[5:]}: {stats[key]:,}")


if __name__ == "__main__":
    main()
