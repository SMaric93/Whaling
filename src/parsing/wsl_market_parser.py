"""
WSL Market Price Extractor - Extract whale oil and bone prices from WSL.

Parses "Marine Markets" / "Prices Current" sections from Whalemen's Shipping List
to extract historical commodity prices:
- Sperm Oil ($/gallon)
- Whale Oil ($/gallon)  
- Whalebone ($/lb)

These prices enable productivity valuation in nominal dollars.
"""

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class WSLPriceQuote:
    """Single price observation from a WSL issue."""
    wsl_issue_id: str
    quote_date: str  # YYYY-MM-DD or YYYY-MM
    commodity: str  # sperm_oil, whale_oil, whalebone
    price_low: Optional[float]
    price_high: Optional[float]
    price_unit: str  # per_gallon, per_lb, per_bbl
    raw_text: str
    confidence: float
    
    @property
    def price_mid(self) -> Optional[float]:
        """Midpoint price for range quotes."""
        if self.price_low is not None and self.price_high is not None:
            return (self.price_low + self.price_high) / 2
        return self.price_low or self.price_high
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame."""
        return {
            "wsl_issue_id": self.wsl_issue_id,
            "quote_date": self.quote_date,
            "commodity": self.commodity,
            "price_low": self.price_low,
            "price_high": self.price_high,
            "price_mid": self.price_mid,
            "price_unit": self.price_unit,
            "raw_text": self.raw_text,
            "confidence": self.confidence,
        }


# =============================================================================
# Price Pattern Definitions
# =============================================================================

# Commodity keywords for section identification
MARKET_SECTION_PATTERNS = [
    r"marine\s+market",
    r"prices?\s+current",
    r"oil\s+market",
    r"new\s+bedford\s+market",
    r"whale\s+oil\s+and\s+bone",
    r"oil\s+and\s+bone",
]

# Commodity identification patterns (ORDER MATTERS - more specific first)
COMMODITY_PATTERNS = {
    "sperm_oil": [
        r"sperm\s+oil",
        r"sperm(?!\s*whale)(?=\s|—|-|$|[^a-zA-Z])",  # sperm alone but not sperm whale
        r"sp(?:\s|\.)+oil",
        r"spermaceti",
    ],
    "whale_oil": [
        r"whale\s+oil(?!\s*and\s*bone)",
        r"whale(?=\s|—|-|$)(?!\s*bone)",  # whale alone but not whalebone
        r"wh(?:\s|\.)+oil",
        r"right\s+whale",
        r"humpback\s+oil",
        r"bowhead\s+oil",
        r"northern(?:\s+(?:oil|whale))?",  # "Northern sells at..."
    ],
    "whalebone": [
        r"whale\s*bone",
        r"bone(?!\s*oil)",  # bone but not bone oil
        r"baleen",
        r"wh(?:\s|\.)+bone",
        r"arctic\s+(?:bone|at)",  # Arctic bone
    ],
}

# Price extraction patterns (handles ranges, single values, various formats)
PRICE_PATTERNS = [
    # Range with dollar signs: $1.25 to $1.50, $2.25-$2.30
    r"\$(\d+(?:\.\d{1,2})?)\s*(?:to|[-–—])\s*\$(\d+(?:\.\d{1,2})?)",
    # Range without dollar signs: 1.25 to 1.50, 95 to 102
    r"(\d+(?:\.\d{1,2})?)\s*(?:to|[-–—])\s*(\d+(?:\.\d{1,2})?)",
    # Single dollar value: $1.25
    r"\$(\d+(?:\.\d{1,2})?)",
    # Cents notation: 95c, 95 cts, 62 cents
    r"(\d+(?:\.\d{1,2})?)\s*(?:c(?:ts)?|cents?)(?:\b|[^a-z])",
    # Value with explicit per: 1.25 per gallon
    r"(\d+(?:\.\d{1,2})?)\s+per\s+(?:gal|lb|bbl)",
    # Fractions: 1 1/2, 1-1/2 (common in 19th century)
    r"(\d+)\s*[-]?\s*(\d)/(\d)",
]

# Unit identification
UNIT_PATTERNS = {
    "per_gallon": [r"per\s+gal(?:lon)?", r"/\s*gal", r"gal\.?"],
    "per_lb": [r"per\s+(?:lb|pound)", r"/\s*lb", r"lb\.?", r"pound"],
    "per_bbl": [r"per\s+bbl", r"per\s+barrel", r"/\s*bbl"],
}


# =============================================================================
# Extraction Functions  
# =============================================================================

def find_market_sections(text: str) -> List[Tuple[int, int, str]]:
    """
    Find market/price sections in WSL text.
    
    Returns list of (start, end, section_text) tuples.
    """
    sections = []
    text_lower = text.lower()
    
    for pattern in MARKET_SECTION_PATTERNS:
        for match in re.finditer(pattern, text_lower, re.IGNORECASE):
            # Extract section around the match (approx 1000 chars forward)
            start = match.start()
            # Look for next section header or end of text
            end = min(start + 2000, len(text))
            
            # Try to find natural section boundary
            next_section = re.search(
                r"\n\s*(?:ARRIVALS|DEPARTURES|SPOKEN|MARINE LIST|MEMORANDA)",
                text[start:end],
                re.IGNORECASE
            )
            if next_section:
                end = start + next_section.start()
            
            section_text = text[start:end]
            sections.append((start, end, section_text))
    
    # Deduplicate overlapping sections
    if len(sections) > 1:
        sections = sorted(set(sections), key=lambda x: x[0])
    
    return sections


def extract_price_from_text(
    text: str, 
    commodity: str
) -> Tuple[Optional[float], Optional[float], str]:
    """
    Extract price values from a text snippet.
    
    Returns (price_low, price_high, raw_text).
    """
    # Try each pattern in order
    for pattern in PRICE_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            groups = match.groups()
            try:
                if len(groups) == 2:
                    # Range: low to high
                    low = float(groups[0])
                    high = float(groups[1])
                    # Handle cents (values < 10 without $ are likely cents)
                    if "c" in text[match.end():match.end()+5].lower() or "cent" in text[match.end():match.end()+10].lower():
                        low = low / 100
                        high = high / 100
                    return (low, high, match.group(0))
                elif len(groups) == 3:
                    # Fraction: whole, numerator, denominator
                    whole = float(groups[0])
                    frac = float(groups[1]) / float(groups[2])
                    value = whole + frac
                    return (value, value, match.group(0))
                elif len(groups) == 1:
                    # Single value
                    value = float(groups[0])
                    # Check if this is cents (indicated by 'c' after the number)
                    if "c" in pattern.lower() or value > 50:  # cents are typically 50-150
                        value = value / 100
                    return (value, value, match.group(0))
            except (ValueError, ZeroDivisionError):
                continue
    
    return (None, None, "")


def detect_unit(text: str) -> str:
    """Detect price unit from text context."""
    text_lower = text.lower()
    
    for unit, patterns in UNIT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return unit
    
    # Default based on common conventions
    return "per_gallon"


def identify_commodity(text: str) -> Optional[str]:
    """Identify which commodity a text snippet refers to."""
    text_lower = text.lower()
    
    for commodity, patterns in COMMODITY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return commodity
    
    return None


def extract_prices_from_text(
    text: str,
    issue_id: str,
    issue_year: int,
    issue_month: Optional[int] = None,
    issue_day: Optional[int] = None,
) -> List[WSLPriceQuote]:
    """
    Extract all price quotes from a WSL text block.
    
    Args:
        text: Full text of WSL issue or page
        issue_id: WSL issue identifier
        issue_year: Year of the WSL issue
        issue_month: Month of the WSL issue
        issue_day: Day of the WSL issue
        
    Returns:
        List of extracted WSLPriceQuote objects
    """
    quotes = []
    
    # Construct date string
    if issue_month and issue_day:
        quote_date = f"{issue_year}-{issue_month:02d}-{issue_day:02d}"
    elif issue_month:
        quote_date = f"{issue_year}-{issue_month:02d}"
    else:
        quote_date = str(issue_year)
    
    # Find market sections
    sections = find_market_sections(text)
    
    if not sections:
        # Try full text if no sections found
        sections = [(0, len(text), text)]
        base_confidence = 0.3  # Lower confidence without section markers
    else:
        base_confidence = 0.7
    
    for _, _, section_text in sections:
        # Split into lines/phrases for granular extraction
        lines = re.split(r'[;.\n]', section_text)
        
        for line in lines:
            line = line.strip()
            if len(line) < 10:
                continue
            
            # Identify commodity
            commodity = identify_commodity(line)
            if not commodity:
                continue
            
            # Extract prices
            price_low, price_high, raw_text = extract_price_from_text(line, commodity)
            
            if price_low is None and price_high is None:
                continue
            
            # Detect unit
            unit = detect_unit(line)
            
            # Adjust confidence based on extraction quality
            confidence = base_confidence
            if raw_text and "$" in raw_text:
                confidence += 0.1
            if unit != "per_gallon":  # Explicit unit found
                confidence += 0.1
            confidence = min(confidence, 1.0)
            
            quotes.append(WSLPriceQuote(
                wsl_issue_id=issue_id,
                quote_date=quote_date,
                commodity=commodity,
                price_low=price_low,
                price_high=price_high,
                price_unit=unit,
                raw_text=line[:200],  # Truncate for storage
                confidence=confidence,
            ))
    
    # Deduplicate by commodity (keep highest confidence)
    if quotes:
        seen = {}
        for q in quotes:
            key = (q.wsl_issue_id, q.commodity)
            if key not in seen or q.confidence > seen[key].confidence:
                seen[key] = q
        quotes = list(seen.values())
    
    return quotes


def extract_prices_from_issue(issue) -> List[WSLPriceQuote]:
    """
    Extract prices from a parsed WSL issue.
    
    Args:
        issue: WSLIssue object from wsl_pdf_parser
        
    Returns:
        List of extracted WSLPriceQuote objects
    """
    return extract_prices_from_text(
        text=issue.full_text,
        issue_id=issue.issue_id,
        issue_year=issue.year,
        issue_month=issue.month,
        issue_day=issue.day,
    )


# =============================================================================
# Aggregation Functions
# =============================================================================

def prices_to_dataframe(quotes: List[WSLPriceQuote]) -> pd.DataFrame:
    """Convert list of WSLPriceQuote objects to DataFrame."""
    if not quotes:
        return pd.DataFrame()
    return pd.DataFrame([q.to_dict() for q in quotes])


def build_annual_price_index(quotes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build annual price index from individual quotes.
    
    Returns DataFrame with columns:
    - year
    - sperm_oil_price ($/gallon, annual average)
    - whale_oil_price ($/gallon, annual average)
    - whalebone_price ($/lb, annual average)
    - n_quotes (number of quotes per year)
    """
    if quotes_df.empty:
        return pd.DataFrame(columns=[
            "year", "sperm_oil_price", "whale_oil_price", "whalebone_price", "n_quotes"
        ])
    
    # Extract year from quote_date
    quotes_df = quotes_df.copy()
    quotes_df["year"] = quotes_df["quote_date"].str[:4].astype(int)
    
    # Calculate weighted averages by commodity
    annual_data = []
    
    for year, year_df in quotes_df.groupby("year"):
        row = {"year": year}
        
        for commodity in ["sperm_oil", "whale_oil", "whalebone"]:
            comm_df = year_df[year_df["commodity"] == commodity]
            if len(comm_df) > 0:
                # Weight by confidence
                weights = comm_df["confidence"]
                prices = comm_df["price_mid"]
                row[f"{commodity}_price"] = (prices * weights).sum() / weights.sum()
            else:
                row[f"{commodity}_price"] = None
        
        row["n_quotes"] = len(year_df)
        annual_data.append(row)
    
    return pd.DataFrame(annual_data).sort_values("year")


def save_price_data(
    quotes: List[WSLPriceQuote],
    output_dir: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """
    Save extracted prices to staging tables.
    
    Returns paths to (quotes_file, annual_index_file).
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "data" / "staging"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    quotes_df = prices_to_dataframe(quotes)
    annual_df = build_annual_price_index(quotes_df)
    
    quotes_path = output_dir / "wsl_price_quotes.parquet"
    annual_path = output_dir / "wsl_annual_prices.parquet"
    
    if len(quotes_df) > 0:
        quotes_df.to_parquet(quotes_path, index=False)
        quotes_df.to_csv(quotes_path.with_suffix(".csv"), index=False)
        logger.info(f"Saved {len(quotes_df)} price quotes to {quotes_path}")
    
    if len(annual_df) > 0:
        annual_df.to_parquet(annual_path, index=False)
        annual_df.to_csv(annual_path.with_suffix(".csv"), index=False)
        logger.info(f"Saved {len(annual_df)} annual prices to {annual_path}")
    
    return quotes_path, annual_path


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    # Example usage with sample text
    sample_text = """
    MARINE MARKET.
    New Bedford, October 17, 1865.
    
    Sperm Oil—The market has been fairly active during the past week.
    Sales include 800 bbls at $2.25 to $2.30 per gallon.
    
    Whale Oil—This article continues in moderate demand.
    Northern sells at 95c to $1.02 per gallon.
    
    Whalebone—The market is firm with sales of Arctic at $1.75 to $1.85 per lb.
    Northwest Coast bone quoted at $1.50 per lb.
    """
    
    quotes = extract_prices_from_text(
        text=sample_text,
        issue_id="18651017",
        issue_year=1865,
        issue_month=10,
        issue_day=17,
    )
    
    print(f"Extracted {len(quotes)} price quotes:")
    for q in quotes:
        print(f"  {q.commodity}: ${q.price_low:.2f}-${q.price_high:.2f} {q.price_unit}")
        print(f"    Confidence: {q.confidence:.2f}")
        print(f"    Raw: {q.raw_text[:80]}...")
        print()
    
    # Convert to DataFrame
    df = prices_to_dataframe(quotes)
    print("\nDataFrame:")
    print(df.to_string(index=False))
