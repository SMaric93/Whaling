"""
Parser for AOWV Logbook position data.

Loads logbook observations with lat/lon positions for route analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging
import hashlib

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_LOGBOOKS, STAGING_DIR
from parsing.string_normalizer import parse_date, normalize_vessel_name, normalize_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LogbookParser:
    """
    Parser for AOWV logbook position data.
    
    Produces day-level position observations for route exposure analysis.
    """
    
    EXPECTED_COLUMNS = {
        "voyage_id": ["VoyageID", "VOYAGE_ID", "voyage_id", "VoyageId"],
        "date": ["Date", "DATE", "date", "ObsDate", "LogDate"],
        "lat": ["Lat", "LAT", "lat", "Latitude", "LATITUDE"],
        "lon": ["Lon", "LON", "lon", "Long", "Longitude", "LONGITUDE"],
        "vessel": ["Vessel", "VESSEL", "vessel", "VesselName", "Ship"],
        "captain": ["Master", "MASTER", "master", "Captain"],
        "year": ["Year", "YEAR", "year"],
        "month": ["Month", "MONTH", "month"],
        "day": ["Day", "DAY", "day"],
    }
    
    def __init__(self, raw_dir: Optional[Path] = None):
        self.raw_dir = raw_dir or RAW_LOGBOOKS
        self._raw_df: Optional[pd.DataFrame] = None
        self._parsed_df: Optional[pd.DataFrame] = None
    
    def _find_column(self, df: pd.DataFrame, field: str) -> Optional[str]:
        """Find the actual column name for a conceptual field."""
        candidates = self.EXPECTED_COLUMNS.get(field, [])
        for col in candidates:
            if col in df.columns:
                return col
        return None
    
    def _load_raw_files(self) -> pd.DataFrame:
        """Load all raw logbook files."""
        dfs = []
        
        for pattern in ["*.csv", "*.txt"]:
            for filepath in self.raw_dir.glob(pattern):
                if "readme" in filepath.name.lower():
                    continue
                
                logger.info(f"Loading {filepath.name}")
                
                for sep in [",", "\t", "|"]:
                    try:
                        df = pd.read_csv(filepath, sep=sep, low_memory=False, encoding="utf-8")
                        if len(df.columns) > 2:
                            df["_source_file"] = filepath.name
                            dfs.append(df)
                            logger.info(f"  Loaded {len(df)} rows")
                            break
                    except:
                        continue
                else:
                    try:
                        df = pd.read_csv(filepath, sep=",", low_memory=False, encoding="latin-1")
                        df["_source_file"] = filepath.name
                        dfs.append(df)
                        logger.info(f"  Loaded {len(df)} rows (latin-1)")
                    except Exception as e:
                        logger.warning(f"  Could not load {filepath.name}: {e}")
        
        if not dfs:
            raise ValueError(f"No logbook files found in {self.raw_dir}")
        
        combined = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined {len(dfs)} files: {len(combined)} total rows")
        
        return combined
    
    def _extract_field(self, df: pd.DataFrame, field: str) -> pd.Series:
        """Extract a field from the dataframe."""
        col = self._find_column(df, field)
        if col:
            return df[col]
        return pd.Series([None] * len(df))
    
    def _parse_coordinate(self, val: Any, is_longitude: bool = False) -> Optional[float]:
        """
        Parse a coordinate value to decimal degrees.
        
        Handles various formats:
        - Decimal degrees (float): 42.5
        - Degrees and minutes string: "42 30 N"
        - Signed decimal: -70.5
        """
        if val is None or pd.isna(val):
            return None
        
        try:
            # Try direct float conversion
            coord = float(val)
            
            # Validate range
            if is_longitude:
                if -180 <= coord <= 180:
                    return coord
            else:
                if -90 <= coord <= 90:
                    return coord
            
            return None
            
        except (ValueError, TypeError):
            # Try parsing string format
            val_str = str(val).strip().upper()
            
            # Extract numbers and direction
            import re
            
            # Pattern: degrees minutes seconds direction
            match = re.match(r"(-?\d+\.?\d*)\s*(\d+\.?\d*)?\s*(\d+\.?\d*)?\s*([NSEW])?", val_str)
            if match:
                deg = float(match.group(1))
                minutes = float(match.group(2)) if match.group(2) else 0
                seconds = float(match.group(3)) if match.group(3) else 0
                direction = match.group(4)
                
                coord = deg + minutes / 60 + seconds / 3600
                
                if direction in ["S", "W"]:
                    coord = -coord
                
                # Validate
                if is_longitude:
                    if -180 <= coord <= 180:
                        return coord
                else:
                    if -90 <= coord <= 90:
                        return coord
            
            return None
    
    def parse(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Parse logbook data into standardized format.
        
        Returns:
            DataFrame with day-level position observations
        """
        if self._parsed_df is not None and not force_reload:
            return self._parsed_df
        
        raw = self._load_raw_files()
        self._raw_df = raw
        
        parsed = pd.DataFrame()
        
        # Voyage ID
        voyage_id_col = self._find_column(raw, "voyage_id")
        if voyage_id_col:
            parsed["voyage_id"] = raw[voyage_id_col].astype(str)
        else:
            parsed["voyage_id"] = None
            parsed["_vessel_for_crosswalk"] = self._extract_field(raw, "vessel").apply(normalize_vessel_name)
            parsed["_captain_for_crosswalk"] = self._extract_field(raw, "captain").apply(normalize_name)
        
        # Parse date
        date_col = self._find_column(raw, "date")
        if date_col:
            parsed["obs_date"] = self._extract_field(raw, "date").apply(
                lambda x: parse_date(x, return_year_only=True)
            )
        else:
            # Try constructing from year/month/day
            year = pd.to_numeric(self._extract_field(raw, "year"), errors="coerce")
            month = pd.to_numeric(self._extract_field(raw, "month"), errors="coerce")
            day = pd.to_numeric(self._extract_field(raw, "day"), errors="coerce")
            
            def make_date(row):
                try:
                    from datetime import date
                    return date(int(row["year"]), int(row["month"]), int(row["day"]))
                except:
                    return None
            
            temp_df = pd.DataFrame({"year": year, "month": month, "day": day})
            parsed["obs_date"] = temp_df.apply(make_date, axis=1)
        
        # Parse coordinates
        lat = self._extract_field(raw, "lat")
        lon = self._extract_field(raw, "lon")
        
        parsed["lat"] = lat.apply(lambda x: self._parse_coordinate(x, is_longitude=False))
        parsed["lon"] = lon.apply(lambda x: self._parse_coordinate(x, is_longitude=True))
        
        # Extract year from date
        parsed["year"] = parsed["obs_date"].apply(lambda x: x.year if x else None)
        
        # Source tracking
        parsed["_source_file"] = raw["_source_file"]
        
        # Filter out rows with no valid coordinates
        valid_coords = parsed["lat"].notna() & parsed["lon"].notna()
        logger.info(f"Valid coordinate rows: {valid_coords.sum()} of {len(parsed)}")
        
        self._parsed_df = parsed
        logger.info(f"Parsed {len(parsed)} logbook observations")
        
        return parsed
    
    def save(self, output_path: Optional[Path] = None) -> Path:
        """Save parsed logbook data to staging directory."""
        if self._parsed_df is None:
            self.parse()
        
        if output_path is None:
            output_path = STAGING_DIR / "logbook_positions.parquet"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._parsed_df.to_parquet(output_path, index=False)
        logger.info(f"Saved logbook_positions to {output_path}")
        
        csv_path = output_path.with_suffix(".csv")
        self._parsed_df.to_csv(csv_path, index=False)
        
        return output_path
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if self._parsed_df is None:
            self.parse()
        
        df = self._parsed_df
        valid = df[df["lat"].notna() & df["lon"].notna()]
        
        return {
            "total_observations": len(df),
            "valid_coordinate_observations": len(valid),
            "unique_voyages": df["voyage_id"].nunique() if df["voyage_id"].notna().any() else 0,
            "year_range": (
                df["year"].min() if df["year"].notna().any() else None,
                df["year"].max() if df["year"].notna().any() else None,
            ),
            "lat_range": (valid["lat"].min(), valid["lat"].max()) if len(valid) > 0 else (None, None),
            "lon_range": (valid["lon"].min(), valid["lon"].max()) if len(valid) > 0 else (None, None),
            "missing_voyage_id_rate": df["voyage_id"].isna().mean(),
            "missing_coords_rate": (~valid.index.isin(df.index)).mean() if len(df) > 0 else 0,
        }


if __name__ == "__main__":
    parser = LogbookParser()
    df = parser.parse()
    
    print("\n=== Logbook Data Summary ===")
    summary = parser.get_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    parser.save()
