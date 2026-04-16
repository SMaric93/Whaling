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
    # In port variants
    "in port": "inp", "inport": "inp", "in_port": "inp",
    "in": "inp", "ir port": "inp",
    "in port, ll'g": "inp", "in port fit'g": "inp", "in port,fit'g": "inp",
    # Departure variants
    "sailed": "dep", "sail": "dep", "departure": "dep", "departed": "dep",
    "to sail": "dep", "cleared": "dep", "despatched": "dep",
    # Arrival variants
    "arrived": "arr", "arrival": "arr",
    "returned": "arr", "landed": "arr",
    # Spoken variants
    "spoken": "spk", "spk'd": "spk", "spoke": "spk", "sp": "spk",
    "heard from": "rpt",
    # Reported variants
    "reported": "rpt", "report": "rpt", "rep": "rpt",
    # Wreck / loss variants
    "wreck": "wrk", "wrecked": "wrk", "condemned": "wrk", "lost": "wrk",
    "cond": "wrk", "sold": "wrk", "sunk": "wrk", "burnt": "wrk",
    "crushed": "wrk", "loss": "wrk",
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
# V3: Sorted by length descending to prevent partial-match issues
VESSEL_TYPE_SUFFIXES = sorted([
    ", bark", ", bk", ", brk", ", ship", ", brig", ", sch", ", schooner",
    " bark", " bk", " ship", " brig", " sch",
], key=len, reverse=True)

# V3: Biophysical limits for cargo quantities
MAX_OIL_BBLS = 5000   # Historical max per voyage: ~4000-5000 barrels
MAX_BONE_LBS = 60000  # Historical max per voyage: ~50,000 lbs

# V3: Unicode fraction map for cargo parsing
FRACTIONS = {"¼": 0.25, "½": 0.5, "¾": 0.75}

# V3: Port stop-list for detecting column-alignment hallucinations
KNOWN_PORTS_LOWER = {
    "nantucket", "new bedford", "new london", "fairhaven", "san francisco",
    "provincetown", "boston", "new york", "honolulu", "edgartown",
    "sag harbor", "cold spring harbor", "greenport", "stonington",
    "westport", "mystic", "warren", "mattapoisett", "newport",
    "dartmouth", "holmes hole", "wareham",
}

# V3.2: Safe OCR corrections — unambiguous character-level corruption
SAFE_PORT_CORRECTIONS = {
    # OCR character errors
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
    # Abbreviations
    "s harbor": "Sag Harbor",
    "s. harbor": "Sag Harbor",
    "greenpt": "Greenport",
    "p town": "Provincetown",
    "prov town": "Provincetown",
    "olmes' hole": "Holmes Hole",
    "holmes' hole": "Holmes Hole",
    # Canonical capitalization (same meaning, just case-fix)
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
    "sandwich": "Sandwich",
    "marion": "Marion",
    "fall river": "Fall River",
}

# V3.2: Ambiguous rewrites — these are interpretive, not OCR corrections.
# Applied only when home_port context confirms a New England vessel.
# "london" could be London, England; "dover" could be Dover, England.
AMBIGUOUS_PORT_CORRECTIONS = {
    "london": "New London",
    "dover": "Edgartown",
}

# Combined for backward compat — but fix_port_ocr only uses SAFE by default
PORT_CORRECTIONS = {**SAFE_PORT_CORRECTIONS, **AMBIGUOUS_PORT_CORRECTIONS}

MONTH_PATTERN = re.compile(
    r"^(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?|Mch)\b", re.IGNORECASE
)

COMPANY_INDICATORS = re.compile(
    r"(&| Co\b| Sons\b| Brothers\b| Bros\b| Jr\b.*&)", re.IGNORECASE
)

YEAR_PATTERN = re.compile(r"\b(\d{2,4})\b")

# V3.2: Coordinate / status detection helpers for detect_field_swap
# Note: [NSEW] must follow a digit to avoid matching month names like "Jan", "June"
COORD_HINT_RE = re.compile(r"\b(lat|lon)\b|\d\s*[°']?\s*[NSEW]\b", re.IGNORECASE)
STATUS_HINTS = {"in port", "at sea", "not stated"}

def _looks_coordish(s):
    """Return True if string looks like coordinates."""
    return bool(s and COORD_HINT_RE.search(s))

def _looks_status(s):
    """Return True if string is a status phrase, not a location."""
    return (s or "").strip().lower() in STATUS_HINTS


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


def _parse_cargo_number(val, max_limit):
    """Parse a cargo quantity with strict validation.

    V3 fixes:
      - 'clean' → 0 (true zero, not None — ship caught 0 whales)
      - Rejects values with alphabetic characters (e.g. '1 whale')
      - Handles unicode fractions (e.g. '7½' → 7.5)
      - Extracts only the first contiguous digit block (prevents
        '1 1/2' → '112' and '18,000' stays '18000')
      - Applies biophysical ceiling (nullifies impossible values)
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        if val < 0:
            return None
        if val > max_limit:
            return None
        return int(val) if val == int(val) else val

    s = str(val).strip()
    if not s:
        return None
    s_lower = s.lower()

    # V3: "clean" means 0 whales caught — true zero, not missing data.
    # Treating this as None would drop valid $0-revenue observations from
    # the econometric panel, biasing estimated returns upward.
    if s_lower in ("clean", "cl", "cln"):
        return 0

    # Genuinely missing data
    if s_lower in ("not stated", "none", "no report", "ns", "n/s", ""):
        return None

    # Reject obvious non-quantity text (has letters → not a number)
    if re.search(r"[a-df-z]", s_lower):  # allow 'e' for scientific notation edge case
        return None

    # Handle unicode fractions (e.g., "7½")
    for ch, frac in FRACTIONS.items():
        if ch in s:
            base = s.replace(ch, "").strip()
            base = re.sub(r"[^\d]", "", base)
            if base and base.isdigit():
                result = int(base) + frac
                return result if result <= max_limit else None
            return frac if frac <= max_limit else None

    # Extract FIRST contiguous sequence of digits
    # (prevents "1 1/2" → "112" and handles "18,000" → "18" correctly
    #  since we take the first block; for comma-separated numbers,
    #  remove commas first)
    cleaned = s.replace(",", "")
    m = re.search(r"\d+", cleaned)
    if m:
        extracted = int(m.group())
        return extracted if extracted <= max_limit else None

    return None


def coerce_oil_values(ev):
    """Convert oil/bone string values to integers where possible.

    V3: Uses strict cargo parser with biophysical limits.
    """
    ev["oil_sperm_bbls"] = _parse_cargo_number(ev.get("oil_sperm_bbls"), MAX_OIL_BBLS)
    ev["oil_whale_bbls"] = _parse_cargo_number(ev.get("oil_whale_bbls"), MAX_OIL_BBLS)
    ev["bone_lbs"] = _parse_cargo_number(ev.get("bone_lbs"), MAX_BONE_LBS)
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
    """Correct known OCR errors in port and home_port fields.

    V3.2: Only applies SAFE corrections by default. Ambiguous rewrites
    (e.g., 'london' → 'New London') are only applied when home_port context
    confirms a New England vessel.
    """
    home = (ev.get("home_port") or "").strip().lower()
    home_is_new_england = home in KNOWN_PORTS_LOWER

    for field in ("port", "home_port"):
        val = ev.get(field)
        if not val:
            continue
        val_lower = val.strip().lower()
        if val_lower in SAFE_PORT_CORRECTIONS:
            ev[field] = SAFE_PORT_CORRECTIONS[val_lower]
        elif val_lower in AMBIGUOUS_PORT_CORRECTIONS and home_is_new_england:
            ev[field] = AMBIGUOUS_PORT_CORRECTIONS[val_lower]
    return ev


def detect_field_swap(ev):
    """Detect and correct when port and date fields are swapped.

    V3.2 FIX: Re-reads from ev after mutation to avoid stale-variable bug.
    Previous version used local `date`/`port` variables after the swap,
    which caused the good date to be destroyed when the original date
    contained coordinates.
    """
    port = (ev.get("port") or "").strip()
    date = (ev.get("date") or "").strip()

    port_is_date = bool(port and MONTH_PATTERN.match(port))
    date_is_date = bool(date and MONTH_PATTERN.match(date))
    date_is_coordish = _looks_coordish(date)

    if port_is_date and (not date_is_date or date_is_coordish):
        ev["date"] = port
        # Don't move coordinates or status text into port — they're not locations
        ev["port"] = None if (date_is_coordish or _looks_status(date)) else (date or None)

    # V3.2 FIX: Re-read from the mutated event, not stale locals
    date = (ev.get("date") or "").strip()
    if _looks_coordish(date):
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
        elif et.lower().startswith("arrived "):
            # "arrived november 27, 71" — full arrival string leaked
            date_part = et[8:].strip()
            if not ev.get("date") and date_part:
                ev["date"] = date_part
            ev["event_type"] = "arr"
            flags.append("date_in_event_type_corrected")
        elif re.match(r"^\d{1,2},?\s*'\d{2}$", et):
            # Bare date fragment like "12, '10" or "3, '10"
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


def validate_alignment(ev):
    """V3: Detect column-alignment hallucinations using port stop-list."""
    flags = []
    vessel = (ev.get("vessel_name") or "").strip().lower()
    captain = (ev.get("captain") or "").strip().lower()

    # Vessel name is actually a port name (column misalignment)
    if vessel in KNOWN_PORTS_LOWER:
        flags.append(f"vessel_is_port_name:{vessel}")

    # Captain name is actually a port name
    if captain in KNOWN_PORTS_LOWER:
        flags.append(f"captain_is_port_name:{captain}")
        ev["captain"] = None  # Nullify — clearly wrong

    # Echo detection: vessel and captain are identical (VLM hallucination)
    if vessel and captain and vessel == captain:
        flags.append("vessel_captain_echo")

    return ev, flags


def validate_soft(ev, all_flags=None):
    """Compute a confidence score based on field completeness, consistency,
    and quality flags.

    V3: Confidence is now penalized by validation flags, making it a usable
    filter for downstream analysis.
    """
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

    # V3: Flag-based penalties (makes confidence reflect actual quality)
    if all_flags:
        for f in all_flags:
            if f.startswith("vessel_is_port_name"):
                score -= 0.4
            elif f.startswith("captain_is_port_name"):
                score -= 0.15
            elif f == "vessel_captain_echo":
                score -= 0.3
            elif f.startswith("invalid_event_type"):
                score -= 0.1
            elif f == "likely_registry_not_weekly":
                score -= 0.05

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

def assign_validation_status(ev):
    """V3.2: Assign a decisive validation status based on flags and confidence.

    Returns one of: 'valid', 'suspicious', 'invalid'
    This is more actionable than a float confidence score for downstream filtering.
    """
    flags = set(ev.get("_flags") or [])
    conf = ev.get("_confidence", 1.0)

    # Hard invalid conditions
    if conf < 0.5:
        return "invalid"
    if any(f.startswith("invalid_event_type") for f in flags):
        return "invalid"
    if "vessel_captain_echo" in flags and not ev.get("vessel_name"):
        return "invalid"

    # Suspicious conditions
    if any(f.startswith("vessel_is_port_name") for f in flags):
        return "suspicious"
    if any(f.startswith("captain_is_port_name") for f in flags):
        return "suspicious"
    if "vessel_captain_echo" in flags:
        return "suspicious"
    if "status_in_port_field" in flags:
        return "suspicious"
    if "likely_registry_not_weekly" in flags:
        return "suspicious"

    return "valid"


def post_process_event(ev, issue_year=None, page_type=None, page_key=None):
    """Run all 6 layers on a single event.

    V3.2: Accepts page_type and page_key for downstream context.
    Adds validation_status and panel_include fields.
    """
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

    # Layer 5: Column-alignment detection (port stop-list)
    ev, alignment_flags = validate_alignment(ev)

    all_flags = hard_flags + section_flags + alignment_flags

    # Layer 6: Confidence + validation status
    ev, confidence, soft_penalties = validate_soft(ev, all_flags=all_flags)

    ev["_flags"] = all_flags if all_flags else None
    ev["_confidence"] = confidence

    # V3.2: Stamp page context for downstream filtering
    if page_type:
        ev["_page_type"] = page_type
    if page_key:
        ev["page_key"] = page_key

    # V3.2: Decisive validation status + panel inclusion
    ev["validation_status"] = assign_validation_status(ev)
    ev["panel_include"] = (
        ev["validation_status"] == "valid"
        and ev.get("_page_type") != "registry"
    )

    return ev


def post_process_page(page_record):
    """Post-process all events in a page record.

    V3.2: Passes page_type and page_key to each event for downstream context.
    """
    events = page_record.get("events", [])
    year = page_record.get("year")
    page_type = page_record.get("page_type")
    page_key = page_record.get("page_key")

    processed = []
    for ev in events:
        ev = post_process_event(
            ev, issue_year=year, page_type=page_type, page_key=page_key
        )
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
