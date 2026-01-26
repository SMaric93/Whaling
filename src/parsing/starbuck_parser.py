"""
Starbuck (1878) Parser - Extract voyage lists from OCR text.

Parses the OCR text of Starbuck's "History of the American Whale Fishery"
to extract structured voyage records for reconciliation with AOWV.
"""

import re
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
import logging
import uuid
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_STARBUCK, STAGING_DIR
from parsing.string_normalizer import normalize_name, normalize_port_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StarbuckVoyage:
    """A voyage record extracted from Starbuck."""
    starbuck_row_id: str
    vessel_name_raw: str
    vessel_name_clean: str
    home_port_raw: Optional[str]
    home_port_clean: Optional[str]
    departure_year: Optional[int]
    return_year: Optional[int]
    notes_text: Optional[str]
    confidence: float
    page_number: Optional[int] = None
    
    def to_dict(self) -> Dict:
        return {
            'starbuck_row_id': self.starbuck_row_id,
            'vessel_name_raw': self.vessel_name_raw,
            'vessel_name_clean': self.vessel_name_clean,
            'home_port_raw': self.home_port_raw,
            'home_port_clean': self.home_port_clean,
            'departure_year': self.departure_year,
            'return_year': self.return_year,
            'notes_text': self.notes_text,
            'confidence': self.confidence,
        }


# Common whaling ports for pattern matching
COMMON_PORTS = [
    "New Bedford", "Nantucket", "Sag Harbor", "Fairhaven", "Edgartown",
    "Provincetown", "New London", "Warren", "Stonington", "Cold Spring Harbor",
    "Greenport", "Bristol", "Newport", "Westport", "Mattapoisett", "Marion",
]


def generate_row_id() -> str:
    """Generate unique Starbuck row ID."""
    return f"starbuck_{uuid.uuid4().hex[:10]}"


def load_starbuck_ocr(path: Optional[Path] = None) -> str:
    """
    Load Starbuck OCR text.
    
    Args:
        path: Path to OCR text file (defaults to raw/starbuck/)
        
    Returns:
        Full OCR text content
    """
    if path is None:
        path = RAW_STARBUCK / "starbuck_1878_ocr.txt"
    
    if not path.exists():
        raise FileNotFoundError(f"Starbuck OCR not found: {path}")
    
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        return f.read()


def find_voyage_table_sections(text: str) -> List[Tuple[int, int, str]]:
    """
    Find sections containing voyage tables in Starbuck.
    
    Starbuck organizes voyages by port, with tables listing:
    - Ship names
    - Years sailed/returned
    - Captain names (sometimes)
    - Outcomes and notes
    
    Returns:
        List of (start_pos, end_pos, section_title) tuples
    """
    sections = []
    
    # Look for port headers
    port_header_patterns = [
        r"(?:VOYAGES|VESSELS|SHIPS)\s+(?:FROM|OF)\s+([A-Z][A-Z\s]+)",
        r"([A-Z][A-Z]+(?:\s+[A-Z]+)?)\s*\.?\s*\n\s*[-=]+",
        r"(?:Vessels?\s+sailing\s+from\s+)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
    ]
    
    for pattern in port_header_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            section_title = match.group(1).strip()
            start_pos = match.start()
            
            # Find end of section (next header or 5000 chars)
            end_pos = start_pos + 5000
            for next_match in re.finditer(pattern, text[start_pos+100:], re.IGNORECASE):
                end_pos = start_pos + 100 + next_match.start()
                break
            
            sections.append((start_pos, end_pos, section_title))
    
    # Deduplicate overlapping sections
    sections = sorted(set(sections), key=lambda x: x[0])
    
    return sections


def parse_voyage_line(line: str, context_port: Optional[str] = None) -> Optional[StarbuckVoyage]:
    """
    Parse a single line/entry that may contain voyage information.
    
    Args:
        line: Text line to parse
        context_port: Port name from section header (if known)
        
    Returns:
        StarbuckVoyage if parseable, else None
    """
    line = line.strip()
    
    if len(line) < 10:
        return None
    
    # Pattern 1: Ship name followed by years
    # Example: "Awashonks, 1845-1848"
    pattern1 = r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*,?\s*(\d{4})\s*[-–]\s*(\d{4})"
    match = re.search(pattern1, line)
    if match:
        vessel_raw = match.group(1)
        dep_year = int(match.group(2))
        ret_year = int(match.group(3))
        
        return StarbuckVoyage(
            starbuck_row_id=generate_row_id(),
            vessel_name_raw=vessel_raw,
            vessel_name_clean=normalize_name(vessel_raw),
            home_port_raw=context_port,
            home_port_clean=normalize_port_name(context_port) if context_port else None,
            departure_year=dep_year,
            return_year=ret_year,
            notes_text=line[:100],
            confidence=0.8,
        )
    
    # Pattern 2: Ship name with single year
    # Example: "Minerva sailed 1852"
    pattern2 = r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:sailed|departed|left)?\s*(\d{4})"
    match = re.search(pattern2, line)
    if match:
        vessel_raw = match.group(1)
        year = int(match.group(2))
        
        return StarbuckVoyage(
            starbuck_row_id=generate_row_id(),
            vessel_name_raw=vessel_raw,
            vessel_name_clean=normalize_name(vessel_raw),
            home_port_raw=context_port,
            home_port_clean=normalize_port_name(context_port) if context_port else None,
            departure_year=year,
            return_year=None,
            notes_text=line[:100],
            confidence=0.6,
        )
    
    # Pattern 3: Tabular format with columns
    # Example: "Levi Starbuck    New Bedford    1847    1851    Pacific"
    pattern3 = r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s{2,}(\w+(?:\s+\w+)?)\s{2,}(\d{4})\s{2,}(\d{4})"
    match = re.search(pattern3, line)
    if match:
        vessel_raw = match.group(1)
        port_raw = match.group(2)
        dep_year = int(match.group(3))
        ret_year = int(match.group(4))
        
        return StarbuckVoyage(
            starbuck_row_id=generate_row_id(),
            vessel_name_raw=vessel_raw,
            vessel_name_clean=normalize_name(vessel_raw),
            home_port_raw=port_raw,
            home_port_clean=normalize_port_name(port_raw),
            departure_year=dep_year,
            return_year=ret_year,
            notes_text=line[:100],
            confidence=0.85,
        )
    
    return None


def extract_voyages_from_text(
    text: str,
    context_port: Optional[str] = None,
) -> List[StarbuckVoyage]:
    """
    Extract all voyage records from a text block.
    
    Args:
        text: Text block to parse
        context_port: Port name from context
        
    Returns:
        List of StarbuckVoyage objects
    """
    voyages = []
    
    # Split into lines
    lines = text.split('\n')
    
    for line in lines:
        voyage = parse_voyage_line(line, context_port)
        if voyage:
            voyages.append(voyage)
    
    return voyages


def parse_starbuck_full(
    ocr_path: Optional[Path] = None,
) -> List[StarbuckVoyage]:
    """
    Parse the complete Starbuck OCR text.
    
    Args:
        ocr_path: Path to OCR text file
        
    Returns:
        List of all extracted voyages
    """
    text = load_starbuck_ocr(ocr_path)
    
    logger.info(f"Loaded Starbuck OCR: {len(text):,} characters")
    
    all_voyages = []
    
    # Find table sections
    sections = find_voyage_table_sections(text)
    logger.info(f"Found {len(sections)} potential voyage table sections")
    
    for start, end, section_title in sections:
        section_text = text[start:end]
        
        # Determine port from section title
        port = None
        for known_port in COMMON_PORTS:
            if known_port.upper() in section_title.upper():
                port = known_port
                break
        
        voyages = extract_voyages_from_text(section_text, port)
        all_voyages.extend(voyages)
        
        if voyages:
            logger.info(f"  {section_title}: {len(voyages)} voyages")
    
    # Also do a full-text scan for any missed patterns
    additional = extract_voyages_from_text(text)
    
    # Deduplicate by vessel+year
    seen = set()
    unique_voyages = []
    for v in all_voyages + additional:
        key = (v.vessel_name_clean, v.departure_year)
        if key not in seen:
            seen.add(key)
            unique_voyages.append(v)
    
    logger.info(f"Extracted {len(unique_voyages)} unique voyages from Starbuck")
    
    return unique_voyages


def voyages_to_dataframe(voyages: List[StarbuckVoyage]) -> pd.DataFrame:
    """Convert voyage list to DataFrame."""
    if not voyages:
        return pd.DataFrame()
    return pd.DataFrame([v.to_dict() for v in voyages])


def save_starbuck_voyages(
    voyages: List[StarbuckVoyage],
    output_path: Optional[Path] = None,
) -> Path:
    """
    Save extracted Starbuck voyages to staging.
    
    Args:
        voyages: List of extracted voyages
        output_path: Output path (defaults to staging)
        
    Returns:
        Path to saved file
    """
    if output_path is None:
        output_path = STAGING_DIR / "starbuck_voyage_list.parquet"
    
    df = voyages_to_dataframe(voyages)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)
    df.to_csv(output_path.with_suffix('.csv'), index=False)
    
    logger.info(f"Saved {len(df)} Starbuck voyages to {output_path}")
    return output_path


def run_starbuck_parser(
    ocr_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Run the full Starbuck parsing pipeline.
    
    Args:
        ocr_path: Path to OCR file
        
    Returns:
        DataFrame of extracted voyages
    """
    voyages = parse_starbuck_full(ocr_path)
    save_starbuck_voyages(voyages)
    return voyages_to_dataframe(voyages)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse Starbuck voyage lists")
    parser.add_argument("--ocr-path", type=Path, default=None,
                        help="Path to Starbuck OCR text file")
    
    args = parser.parse_args()
    
    df = run_starbuck_parser(args.ocr_path)
    
    print(f"\nExtracted {len(df)} voyages")
    if len(df) > 0:
        print(f"\nSample voyages:")
        print(df.head(10).to_string())
        
        print(f"\nBy decade:")
        df['decade'] = (df['departure_year'] // 10 * 10).astype('Int64')
        print(df.groupby('decade').size())
