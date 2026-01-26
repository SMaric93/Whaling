"""
WSL Event Extractor - Rule-based extraction of voyage events from WSL text.

Extracts structured event mentions from Whalemen's Shipping List issues:
- Departures, Arrivals, Spoken-with reports
- Wrecks, Losses, Captures, Damage reports
- Port visits and returns

Uses regex patterns and heuristics. Optional LLM enhancement can be
enabled via config flag (not implemented in this base version).
"""

import re
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import uuid
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import WSL_EVENT_TYPES, STAGING_DIR
from parsing.string_normalizer import normalize_name, normalize_port_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EventType(Enum):
    """WSL event type enumeration."""
    DEPARTURE = "DEPARTURE"
    ARRIVAL = "ARRIVAL"
    SPOKEN_WITH = "SPOKEN_WITH"
    REPORTED_AT = "REPORTED_AT"
    WRECK = "WRECK"
    LOSS = "LOSS"
    CAPTURED = "CAPTURED"
    DAMAGED = "DAMAGED"
    RETURNED_HOME = "RETURNED_HOME"
    OTHER = "OTHER"


@dataclass
class WSLEvent:
    """Represents a single extracted event from WSL."""
    wsl_event_id: str
    wsl_issue_id: str
    event_date: Optional[str]  # YYYY-MM-DD or partial
    vessel_name_raw: str
    vessel_name_clean: str
    captain_name_raw: Optional[str]
    captain_name_clean: Optional[str]
    port_name_raw: Optional[str]
    port_name_clean: Optional[str]
    event_type: EventType
    event_text_snippet: str
    confidence: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame."""
        return {
            'wsl_event_id': self.wsl_event_id,
            'wsl_issue_id': self.wsl_issue_id,
            'event_date': self.event_date,
            'vessel_name_raw': self.vessel_name_raw,
            'vessel_name_clean': self.vessel_name_clean,
            'captain_name_raw': self.captain_name_raw,
            'captain_name_clean': self.captain_name_clean,
            'port_name_raw': self.port_name_raw,
            'port_name_clean': self.port_name_clean,
            'event_type': self.event_type.value,
            'event_text_snippet': self.event_text_snippet,
            'confidence': self.confidence,
        }


# =============================================================================
# REGEX PATTERNS FOR EVENT DETECTION
# =============================================================================

# Vessel name patterns (ship names are typically proper nouns, often in italics or quotes)
VESSEL_PATTERN = r"(?:ship|bark|brig|schooner|bk\.?\s+|sch\.?\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"

# Captain patterns
CAPTAIN_PATTERNS = [
    r"(?:Capt(?:ain)?\.?\s+|Master\s+)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
    r",\s+([A-Z][a-z]+),\s+(?:master|commander)",
]

# Port patterns (common whaling ports)
PORT_PATTERNS = [
    r"(?:from|at|for|to|of|arrived? at)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
    r"(?:New Bedford|Nantucket|Sag Harbor|Fairhaven|Edgartown|Provincetown|New London)",
]

# Event type patterns
EVENT_PATTERNS = {
    EventType.DEPARTURE: [
        r"(?:sailed|departed|cleared|left)\s+(?:from\s+)?(\w+)",
        r"(?:has\s+)?(?:sailed|cleared)\s+for\s+(\w+)",
    ],
    EventType.ARRIVAL: [
        r"(?:arrived|arr(?:\.)?|returned|came\s+in)\s+(?:at\s+)?(\w+)",
        r"(?:has\s+)?arrived\s+(?:home|at\s+\w+)",
    ],
    EventType.SPOKEN_WITH: [
        r"(?:spoke|spoken|spk\.?)\s+(?:with\s+)?",
        r"(?:was\s+)?spoken\s+(?:by\s+)?",
    ],
    EventType.REPORTED_AT: [
        r"(?:reported|rep\.?|seen)\s+(?:at\s+)?",
        r"(?:was\s+)?(?:reported|seen)\s+",
    ],
    EventType.WRECK: [
        r"(?:wrecked|wreck|run\s+aground|stranded)",
        r"(?:total\s+loss|went\s+ashore)",
    ],
    EventType.LOSS: [
        r"(?:lost|abandoned|sunk|foundered)",
        r"(?:total\s+)?loss\s+of",
    ],
    EventType.CAPTURED: [
        r"(?:captured|taken|seized)",
        r"(?:by\s+)?(?:pirates?|privateers?)",
    ],
    EventType.DAMAGED: [
        r"(?:damaged|dismasted|stove|leaked)",
        r"(?:put\s+into|put\s+back|in\s+distress)",
    ],
    EventType.RETURNED_HOME: [
        r"(?:returned|arrived)\s+(?:home|to\s+port)",
        r"(?:completed|finished)\s+(?:voyage|cruise)",
    ],
}


def generate_event_id() -> str:
    """Generate a unique event ID."""
    return f"wsl_evt_{uuid.uuid4().hex[:12]}"


def extract_vessel_name(text: str) -> List[Tuple[str, int, int]]:
    """
    Extract vessel names from text.
    
    Returns list of (name, start_pos, end_pos) tuples.
    """
    vessels = []
    
    # Pattern for vessel names (typically italicized or after ship type)
    patterns = [
        r"\b(ship|bark|brig|schooner|bk\.|sch\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        r"\b([A-Z][A-Z][A-Z]+)\b",  # All caps names
        r"\"([^\"]+)\"",  # Quoted names
    ]
    
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            if len(match.groups()) >= 2:
                name = match.group(2)
            else:
                name = match.group(1)
            
            if name and len(name) > 2:
                vessels.append((name, match.start(), match.end()))
    
    return vessels


def extract_captain_name(text: str) -> Optional[str]:
    """Extract captain name from text."""
    for pattern in CAPTAIN_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def extract_port_name(text: str) -> Optional[str]:
    """Extract port name from text."""
    # Common whaling ports (direct match)
    common_ports = [
        "New Bedford", "Nantucket", "Sag Harbor", "Fairhaven",
        "Edgartown", "Provincetown", "New London", "Warren",
        "Stonington", "Cold Spring Harbor", "Greenport",
    ]
    
    for port in common_ports:
        if port.lower() in text.lower():
            return port
    
    # Pattern-based extraction
    for pattern in PORT_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            if match.lastindex and match.lastindex >= 1:
                return match.group(1)
            return match.group(0)
    
    return None


def detect_event_type(text: str) -> Tuple[EventType, float]:
    """
    Detect the event type from text snippet.
    
    Returns (EventType, confidence) tuple.
    """
    text_lower = text.lower()
    
    # Check each event type's patterns
    for event_type, patterns in EVENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                # Higher confidence for more specific matches
                confidence = 0.8 if len(patterns) > 1 else 0.7
                return event_type, confidence
    
    return EventType.OTHER, 0.5


def extract_date_from_context(
    text: str,
    issue_year: int,
    issue_month: Optional[int] = None,
    issue_day: Optional[int] = None,
) -> Optional[str]:
    """
    Extract or infer event date from text and issue context.
    
    Args:
        text: Event text snippet
        issue_year: Year of the WSL issue
        issue_month: Month of the WSL issue (if known)
        issue_day: Day of the WSL issue (if known)
        
    Returns:
        Date string in YYYY-MM-DD format (may be partial)
    """
    # Try to find explicit date in text
    date_patterns = [
        r"(\w+)\s+(\d{1,2})",  # "March 15"
        r"(\d{1,2})\s+(\w+)",  # "15 March"
        r"(\d{1,2})/(\d{1,2})/(\d{2,4})",  # MM/DD/YY
    ]
    
    month_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'june': 6, 'july': 7, 'august': 8, 'september': 9,
        'october': 10, 'november': 11, 'december': 12,
    }
    
    for pattern in date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            groups = match.groups()
            if len(groups) == 2:
                # Month Day or Day Month
                g1, g2 = groups
                if g1.lower() in month_map:
                    month = month_map[g1.lower()]
                    day = int(g2)
                elif g2.lower() in month_map:
                    month = month_map[g2.lower()]
                    day = int(g1)
                else:
                    continue
                return f"{issue_year}-{month:02d}-{day:02d}"
    
    # Fall back to issue date
    if issue_month and issue_day:
        return f"{issue_year}-{issue_month:02d}-{issue_day:02d}"
    elif issue_month:
        return f"{issue_year}-{issue_month:02d}"
    else:
        return f"{issue_year}"


def extract_events_from_text(
    text: str,
    issue_id: str,
    issue_year: int,
    issue_month: Optional[int] = None,
    issue_day: Optional[int] = None,
) -> List[WSLEvent]:
    """
    Extract all events from a WSL text block.
    
    Args:
        text: Full text of WSL issue or page
        issue_id: WSL issue identifier
        issue_year: Year of the WSL issue
        issue_month: Month of the WSL issue
        issue_day: Day of the WSL issue
        
    Returns:
        List of extracted WSLEvent objects
    """
    events = []
    
    # Split into potential event segments (sentences or newline-separated)
    segments = re.split(r'[.\n]+', text)
    
    for segment in segments:
        segment = segment.strip()
        if len(segment) < 10:
            continue
        
        # Look for vessel names as anchors
        vessels = extract_vessel_name(segment)
        
        for vessel_raw, start, end in vessels:
            # Extract context around vessel mention
            context_start = max(0, start - 50)
            context_end = min(len(segment), end + 100)
            context = segment[context_start:context_end]
            
            # Detect event type
            event_type, confidence = detect_event_type(context)
            
            # Extract additional fields
            captain_raw = extract_captain_name(context)
            port_raw = extract_port_name(context)
            event_date = extract_date_from_context(
                context, issue_year, issue_month, issue_day
            )
            
            # Normalize names
            vessel_clean = normalize_name(vessel_raw) if vessel_raw else None
            captain_clean = normalize_name(captain_raw) if captain_raw else None
            port_clean = normalize_port_name(port_raw) if port_raw else None
            
            # Create event
            event = WSLEvent(
                wsl_event_id=generate_event_id(),
                wsl_issue_id=issue_id,
                event_date=event_date,
                vessel_name_raw=vessel_raw,
                vessel_name_clean=vessel_clean,
                captain_name_raw=captain_raw,
                captain_name_clean=captain_clean,
                port_name_raw=port_raw,
                port_name_clean=port_clean,
                event_type=event_type,
                event_text_snippet=context[:200],  # Truncate long snippets
                confidence=confidence,
            )
            
            events.append(event)
    
    return events


def extract_events_from_issue(
    issue,  # WSLIssue from wsl_pdf_parser
) -> List[WSLEvent]:
    """
    Extract events from a parsed WSL issue.
    
    Args:
        issue: WSLIssue object from wsl_pdf_parser
        
    Returns:
        List of extracted WSLEvent objects
    """
    all_events = []
    
    for page in issue.pages:
        page_events = extract_events_from_text(
            text=page.text,
            issue_id=issue.issue_id,
            issue_year=issue.year,
            issue_month=issue.month,
            issue_day=issue.day,
        )
        
        # Adjust confidence based on page extraction quality
        for event in page_events:
            event.confidence *= page.confidence
        
        all_events.extend(page_events)
    
    logger.info(f"Extracted {len(all_events)} events from {issue.issue_id}")
    return all_events


def events_to_dataframe(events: List[WSLEvent]) -> pd.DataFrame:
    """Convert list of WSLEvent objects to DataFrame."""
    if not events:
        return pd.DataFrame()
    
    return pd.DataFrame([e.to_dict() for e in events])


def save_extracted_events(
    events: List[WSLEvent],
    output_path: Optional[Path] = None,
) -> Path:
    """
    Save extracted events to staging tables.
    
    Args:
        events: List of extracted events
        output_path: Custom output path (defaults to staging)
        
    Returns:
        Path to saved parquet file
    """
    if output_path is None:
        output_path = STAGING_DIR / "wsl_extracted_events.parquet"
    
    df = events_to_dataframe(events)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)
    df.to_csv(output_path.with_suffix('.csv'), index=False)
    
    logger.info(f"Saved {len(df)} events to {output_path}")
    return output_path


if __name__ == "__main__":
    # Example usage
    sample_text = """
    The ship AWASHONKS, Capt. Wood, arrived at New Bedford on March 15, 
    with 1500 bbls sperm oil. She reports speaking the bark MONTEZUMA, 
    of Fairhaven, on the Coast of Peru, with 800 bbls.
    
    Sailed from Nantucket, March 10 - Ship COLUMBIA, Swain, Indian Ocean.
    
    LOST - The brig HESPER of Sag Harbor was wrecked on the coast of 
    Patagonia. Crew saved.
    """
    
    events = extract_events_from_text(
        text=sample_text,
        issue_id="wsl_1850_03_18",
        issue_year=1850,
        issue_month=3,
        issue_day=18,
    )
    
    print(f"\nExtracted {len(events)} events:\n")
    for e in events:
        print(f"  {e.event_type.value}: {e.vessel_name_raw}")
        print(f"    Captain: {e.captain_name_raw}")
        print(f"    Port: {e.port_name_raw}")
        print(f"    Date: {e.event_date}")
        print(f"    Confidence: {e.confidence:.2f}")
        print()
