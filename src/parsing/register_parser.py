"""
Parser for Mutual Marine Insurance Register OCR text.

Extracts vessel registry information for vessel quality proxy (VQI).
Handles poor OCR quality with fuzzy matching.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import re
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_INSURANCE, STAGING_DIR
from parsing.string_normalizer import normalize_vessel_name, jaro_winkler_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegisterParser:
    """
    Parser for Mutual Marine Insurance Register OCR text.
    
    The register contains vessel entries with ratings and values.
    OCR quality varies, so robust parsing with fuzzy matching is needed.
    """
    
    # Patterns for extracting vessel entries
    # These are approximate and may need adjustment based on actual OCR
    VESSEL_PATTERNS = [
        # Pattern: Vessel Name, Rating, Value
        r"([A-Z][A-Za-z\s]+)\s+([A-Z]\d?)\s+\$?(\d[\d,]+)",
        # Pattern: Vessel Name (Rig) Rating
        r"([A-Z][A-Za-z\s]+)\s*\((\w+)\)\s+([A-Z]\d?)",
        # Pattern: More flexible - vessel name followed by any code
        r"^([A-Z][A-Za-z\s]{3,25})\s+([A-Z][\d\s]*)",
    ]
    
    # Rating codes (A1, A2, B, etc.)
    VALID_RATINGS = {"A1", "A2", "A", "B", "B1", "B2", "C", "1", "2", "3"}
    
    def __init__(self, raw_dir: Optional[Path] = None):
        self.raw_dir = raw_dir or RAW_INSURANCE
        self._ocr_text: Optional[str] = None
        self._parsed_df: Optional[pd.DataFrame] = None
    
    def _load_ocr_text(self) -> str:
        """Load OCR text file."""
        ocr_file = self.raw_dir / "mutual_marine_register_1843_1862_ocr.txt"
        
        if not ocr_file.exists():
            raise FileNotFoundError(f"OCR file not found: {ocr_file}")
        
        logger.info(f"Loading OCR text from {ocr_file}")
        
        with open(ocr_file, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        
        logger.info(f"Loaded {len(text)} characters")
        return text
    
    def _extract_vessel_entries(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract vessel entries from OCR text.
        
        This is inherently noisy due to OCR quality.
        """
        entries = []
        
        # Split into lines
        lines = text.split("\n")
        
        # Track current page/year context
        current_year = None
        
        year_pattern = re.compile(r"\b(184[3-9]|185\d|186[0-2])\b")
        
        for line in lines:
            line = line.strip()
            
            if not line or len(line) < 5:
                continue
            
            # Check for year markers
            year_match = year_pattern.search(line)
            if year_match:
                candidate_year = int(year_match.group(1))
                # Only update if it looks like a header/context line
                if len(line) < 50:
                    current_year = candidate_year
            
            # Try to extract vessel entries
            for pattern in self.VESSEL_PATTERNS:
                match = re.search(pattern, line)
                if match:
                    groups = match.groups()
                    
                    vessel_name = groups[0].strip() if groups[0] else None
                    
                    # Validate vessel name looks reasonable
                    if vessel_name and len(vessel_name) >= 3:
                        # Clean up common OCR artifacts
                        vessel_name = re.sub(r"[^\w\s]", "", vessel_name)
                        vessel_name = " ".join(vessel_name.split())
                        
                        if len(vessel_name) >= 3:
                            entry = {
                                "vessel_name_raw": vessel_name,
                                "vessel_name_clean": normalize_vessel_name(vessel_name),
                                "year": current_year,
                                "raw_line": line[:100],
                            }
                            
                            # Extract rating if present
                            for group in groups[1:]:
                                if group and group.upper() in self.VALID_RATINGS:
                                    entry["rating"] = group.upper()
                                    break
                            
                            # Extract value if present
                            for group in groups[1:]:
                                if group and re.match(r"\d[\d,]+", str(group)):
                                    try:
                                        value = int(str(group).replace(",", ""))
                                        if 100 <= value <= 100000:  # Reasonable vessel value range
                                            entry["insured_value"] = value
                                    except:
                                        pass
                            
                            entries.append(entry)
                            break
        
        logger.info(f"Extracted {len(entries)} potential vessel entries")
        return entries
    
    def _deduplicate_entries(self, entries: List[Dict]) -> List[Dict]:
        """
        Deduplicate vessel entries, keeping best version per vessel-year.
        """
        # Group by vessel name and year
        vessel_year_groups = {}
        
        for entry in entries:
            key = (entry.get("vessel_name_clean"), entry.get("year"))
            
            if key not in vessel_year_groups:
                vessel_year_groups[key] = []
            vessel_year_groups[key].append(entry)
        
        # Keep entry with most information
        deduplicated = []
        
        for key, group in vessel_year_groups.items():
            if not key[0]:  # Skip entries without vessel name
                continue
            
            # Score entries by completeness
            best_entry = max(group, key=lambda e: (
                (1 if e.get("rating") else 0) +
                (1 if e.get("insured_value") else 0)
            ))
            
            deduplicated.append(best_entry)
        
        logger.info(f"Deduplicated to {len(deduplicated)} vessel-year entries")
        return deduplicated
    
    def parse(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Parse vessel register data.
        
        Returns:
            DataFrame with vessel_register_year schema
        """
        if self._parsed_df is not None and not force_reload:
            return self._parsed_df
        
        # Load OCR text
        try:
            text = self._load_ocr_text()
            self._ocr_text = text
        except FileNotFoundError as e:
            logger.warning(f"OCR file not found, returning empty DataFrame: {e}")
            return pd.DataFrame(columns=[
                "vessel_name_clean", "year", "rating", "insured_value", "vqi_proxy"
            ])
        
        # Extract entries
        entries = self._extract_vessel_entries(text)
        
        # Deduplicate
        entries = self._deduplicate_entries(entries)
        
        if not entries:
            logger.warning("No vessel entries extracted")
            return pd.DataFrame(columns=[
                "vessel_name_clean", "year", "rating", "insured_value", "vqi_proxy"
            ])
        
        # Create DataFrame
        df = pd.DataFrame(entries)
        
        # Compute VQI proxy
        df = self._compute_vqi_proxy(df)
        
        self._parsed_df = df
        logger.info(f"Parsed {len(df)} vessel register entries")
        
        return df
    
    def _compute_vqi_proxy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute vessel quality index proxy from available data.
        """
        result = df.copy()
        
        # Rating score (higher is better)
        rating_scores = {
            "A1": 1.0,
            "A": 0.9,
            "A2": 0.85,
            "B1": 0.7,
            "B": 0.6,
            "B2": 0.5,
            "C": 0.3,
            "1": 1.0,
            "2": 0.7,
            "3": 0.4,
        }
        
        result["rating_score"] = result["rating"].map(rating_scores)
        
        # Normalize insured value to 0-1 scale
        if "insured_value" in result.columns and result["insured_value"].notna().any():
            min_val = result["insured_value"].min()
            max_val = result["insured_value"].max()
            
            if max_val > min_val:
                result["value_score"] = (result["insured_value"] - min_val) / (max_val - min_val)
            else:
                result["value_score"] = 0.5
        else:
            result["value_score"] = np.nan
        
        # Combine into VQI proxy
        # Weight rating more heavily since value may be missing
        def compute_vqi(row):
            rating = row.get("rating_score")
            value = row.get("value_score")
            
            if pd.notna(rating) and pd.notna(value):
                return rating * 0.6 + value * 0.4
            elif pd.notna(rating):
                return rating
            elif pd.notna(value):
                return value * 0.8  # Discount when only value available
            else:
                return np.nan
        
        result["vqi_proxy"] = result.apply(compute_vqi, axis=1)
        
        return result
    
    def save(self, output_path: Optional[Path] = None) -> Path:
        """Save parsed register data."""
        if self._parsed_df is None:
            self.parse()
        
        if output_path is None:
            output_path = STAGING_DIR / "vessel_register_year.parquet"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._parsed_df.to_parquet(output_path, index=False)
        logger.info(f"Saved vessel register to {output_path}")
        
        csv_path = output_path.with_suffix(".csv")
        self._parsed_df.to_csv(csv_path, index=False)
        
        return output_path
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if self._parsed_df is None:
            self.parse()
        
        df = self._parsed_df
        
        return {
            "total_entries": len(df),
            "unique_vessels": df["vessel_name_clean"].nunique(),
            "year_range": (
                df["year"].min() if df["year"].notna().any() else None,
                df["year"].max() if df["year"].notna().any() else None,
            ),
            "rating_coverage": df["rating"].notna().mean(),
            "value_coverage": df.get("insured_value", pd.Series()).notna().mean(),
            "vqi_coverage": df["vqi_proxy"].notna().mean(),
            "rating_distribution": df["rating"].value_counts().to_dict() if df["rating"].notna().any() else {},
        }


if __name__ == "__main__":
    parser = RegisterParser()
    df = parser.parse()
    
    print("\n=== Vessel Register Summary ===")
    summary = parser.get_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    if len(df) > 0:
        print("\n=== Sample Entries ===")
        print(df.head(10))
        
        parser.save()
