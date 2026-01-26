"""
Parser for AOWV Voyage data.

Loads and standardizes the American Offshore Whaling Voyages dataset
into the voyages_master schema.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import date
import logging
import hashlib

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_AOWV, STAGING_DIR
from parsing.string_normalizer import (
    normalize_name,
    normalize_vessel_name,
    parse_date,
    parse_year,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoyageParser:
    """
    Parser for AOWV voyage data.
    
    Loads raw voyage files and produces a standardized voyages_master table.
    """
    
    # Expected columns in raw data (flexible - will map what's available)
    EXPECTED_COLUMNS = {
        "voyage_id": ["voyageID", "VoyageID", "VOYAGE_ID", "voyage_id", "VoyageId"],
        "vessel": ["vessel", "Vessel", "VESSEL", "VesselName", "VESSEL_NAME"],
        "rig": ["rig", "Rig", "RIG", "VesselRig"],
        "captain": ["master", "Master", "MASTER", "Captain", "CAPTAIN", "captain", "MasterName"],
        "agent": ["agent", "Agent", "AGENT", "ManagingOwner", "AgentName"],
        "port_out": ["port", "sailingFrom", "From", "FROM", "from", "PortOut", "PORT_OUT", "DeparturePort"],
        "port_in": ["Return", "RETURN", "return", "PortIn", "PORT_IN", "ArrivalPort"],
        "date_out": ["yearOut", "dayOut", "DateOut", "DATE_OUT", "date_out", "Departure", "SailedDate"],
        "date_in": ["yearIn", "dayIn", "DateIn", "DATE_IN", "date_in", "ReturnDate"],
        "destination": ["ground", "Ground", "GROUND", "Destination", "DESTINATION", "Grounds"],
        "sperm_oil": ["sperm", "SpermOil", "SPERM_OIL", "sperm_oil", "Sperm", "SpermBbls"],
        "whale_oil": ["oil", "WhaleOil", "WHALE_OIL", "whale_oil", "Whale", "WhaleBbls"],
        "bone": ["bone", "Bone", "BONE", "WhaleBone", "BoneLbs"],
        "tonnage": ["tonnage", "Tonnage", "TONNAGE", "Tons"],
        "home_port": ["port", "HomePort", "HOME_PORT", "home_port", "Hailing"],
        "year_out": ["yearOut", "YearOut", "YEAR_OUT"],
        "year_in": ["yearIn", "YearIn", "YEAR_IN"],
    }
    
    def __init__(self, raw_dir: Optional[Path] = None):
        self.raw_dir = raw_dir or RAW_AOWV
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
        """Load all raw voyage files from the raw directory."""
        dfs = []
        
        # Look for CSV and TXT files
        for pattern in ["*.csv", "*.txt"]:
            for filepath in self.raw_dir.glob(pattern):
                # Skip readme files
                if "readme" in filepath.name.lower():
                    continue
                
                logger.info(f"Loading {filepath.name}")
                
                # Try different delimiters
                for sep in [",", "\t", "|"]:
                    try:
                        df = pd.read_csv(filepath, sep=sep, low_memory=False, encoding="utf-8")
                        if len(df.columns) > 3:  # Reasonable number of columns
                            df["_source_file"] = filepath.name
                            dfs.append(df)
                            logger.info(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
                            break
                    except Exception as e:
                        continue
                else:
                    # Try latin-1 encoding
                    try:
                        df = pd.read_csv(filepath, sep=",", low_memory=False, encoding="latin-1")
                        df["_source_file"] = filepath.name
                        dfs.append(df)
                        logger.info(f"  Loaded {len(df)} rows (latin-1)")
                    except Exception as e:
                        logger.warning(f"  Could not load {filepath.name}: {e}")
        
        if not dfs:
            raise ValueError(f"No voyage files found in {self.raw_dir}")
        
        # Concatenate all files
        combined = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined {len(dfs)} files: {len(combined)} total rows")
        
        return combined
    
    def _extract_field(self, df: pd.DataFrame, field: str) -> pd.Series:
        """Extract a field from the dataframe, trying multiple column names."""
        col = self._find_column(df, field)
        if col:
            return df[col]
        return pd.Series([None] * len(df))
    
    def _compute_voyage_id(self, row: pd.Series) -> str:
        """
        Compute a deterministic voyage ID from row data.
        
        Used when source data doesn't have a voyage ID.
        """
        components = [
            str(row.get("vessel_name_clean", "")),
            str(row.get("captain_name_clean", "")),
            str(row.get("date_out", "")),
            str(row.get("port_out", "")),
        ]
        key = "|".join(components)
        return hashlib.md5(key.encode()).hexdigest()[:12]
    
    def parse(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Parse voyage data into standardized format.
        
        Args:
            force_reload: If True, reload and reparse even if already done
            
        Returns:
            DataFrame with voyages_master schema
        """
        if self._parsed_df is not None and not force_reload:
            return self._parsed_df
        
        # Load raw data
        raw = self._load_raw_files()
        self._raw_df = raw
        
        # Create output dataframe
        parsed = pd.DataFrame()
        
        # Extract voyage ID if present
        voyage_id_col = self._find_column(raw, "voyage_id")
        if voyage_id_col:
            parsed["voyage_id"] = raw[voyage_id_col].astype(str)
        else:
            parsed["voyage_id"] = None  # Will compute later
        
        # Extract and normalize vessel name
        vessel_series = self._extract_field(raw, "vessel")
        parsed["vessel_name_raw"] = vessel_series
        parsed["vessel_name_clean"] = vessel_series.apply(normalize_vessel_name)
        
        # Extract rig type
        parsed["rig"] = self._extract_field(raw, "rig")
        
        # Extract and normalize captain name
        captain_series = self._extract_field(raw, "captain")
        parsed["captain_name_raw"] = captain_series
        parsed["captain_name_clean"] = captain_series.apply(normalize_name)
        
        # Extract and normalize agent name
        agent_series = self._extract_field(raw, "agent")
        parsed["agent_name_raw"] = agent_series
        parsed["agent_name_clean"] = agent_series.apply(normalize_name)
        
        # Extract ports
        parsed["port_out"] = self._extract_field(raw, "port_out").apply(
            lambda x: str(x).upper().strip() if pd.notna(x) else None
        )
        parsed["port_in"] = self._extract_field(raw, "port_in").apply(
            lambda x: str(x).upper().strip() if pd.notna(x) else None
        )
        parsed["home_port"] = self._extract_field(raw, "home_port").apply(
            lambda x: str(x).upper().strip() if pd.notna(x) else None
        )
        
        # Parse dates - AOWV uses separate yearOut/yearIn columns and dayOut/dayIn columns
        # The dayOut column contains month abbreviations like "1766" or actual day numbers
        year_out_col = self._find_column(raw, "year_out")
        year_in_col = self._find_column(raw, "year_in")
        
        if year_out_col:
            # AOWV format with separate year columns
            parsed["year_out"] = pd.to_numeric(raw[year_out_col], errors="coerce").astype("Int64")
            parsed["year_in"] = pd.to_numeric(raw[year_in_col], errors="coerce").astype("Int64") if year_in_col else None
            
            # Build approximate date from year (set to Jan 1)
            def year_to_date(year):
                if pd.isna(year):
                    return None
                try:
                    return date(int(year), 1, 1)
                except:
                    return None
            
            parsed["date_out"] = parsed["year_out"].apply(year_to_date)
            parsed["date_in"] = parsed["year_in"].apply(year_to_date)
            parsed["date_out_raw"] = raw[year_out_col].astype(str) if year_out_col else None
            parsed["date_in_raw"] = raw[year_in_col].astype(str) if year_in_col else None
        else:
            # Fall back to older format with date columns
            date_out_raw = self._extract_field(raw, "date_out")
            date_in_raw = self._extract_field(raw, "date_in")
            
            parsed["date_out_raw"] = date_out_raw
            parsed["date_in_raw"] = date_in_raw
            
            parsed["date_out"] = date_out_raw.apply(lambda x: parse_date(x, return_year_only=True))
            parsed["date_in"] = date_in_raw.apply(lambda x: parse_date(x, return_year_only=True))
            
            # Derive years
            parsed["year_out"] = parsed["date_out"].apply(lambda x: x.year if x else None)
            parsed["year_in"] = parsed["date_in"].apply(lambda x: x.year if x else None)
        
        # Calculate duration
        def calc_duration(row):
            if row["date_out"] and row["date_in"]:
                return (row["date_in"] - row["date_out"]).days
            return None
        parsed["duration_days"] = parsed.apply(calc_duration, axis=1)
        
        # Extract destination/ground
        parsed["ground_or_route"] = self._extract_field(raw, "destination").apply(
            lambda x: str(x).upper().strip() if pd.notna(x) else None
        )
        
        # Extract quantities
        sperm_oil = self._extract_field(raw, "sperm_oil")
        whale_oil = self._extract_field(raw, "whale_oil")
        bone = self._extract_field(raw, "bone")
        
        # Clean numeric values
        def clean_numeric(x):
            if pd.isna(x) or x == "" or x == " ":
                return None
            try:
                val = float(str(x).replace(",", ""))
                return val if val >= 0 else None
            except:
                return None
        
        parsed["q_sperm_bbl"] = sperm_oil.apply(clean_numeric)
        parsed["q_whale_bbl"] = whale_oil.apply(clean_numeric)
        parsed["q_oil_bbl"] = parsed["q_sperm_bbl"].fillna(0) + parsed["q_whale_bbl"].fillna(0)
        parsed["q_bone_lbs"] = bone.apply(clean_numeric)
        
        # Tonnage
        parsed["tonnage"] = self._extract_field(raw, "tonnage").apply(clean_numeric)
        
        # Compute voyage IDs where missing
        if parsed["voyage_id"].isna().any():
            missing_mask = parsed["voyage_id"].isna()
            parsed.loc[missing_mask, "voyage_id"] = parsed.loc[missing_mask].apply(
                self._compute_voyage_id, axis=1
            )
        
        # Create route_year_cell
        def make_route_year_cell(row):
            ground = row["ground_or_route"]
            year = row["year_out"]
            if pd.notna(ground) and pd.notna(year):
                return f"{ground}_{int(year)}"
            return None
        
        parsed["route_year_cell"] = parsed.apply(make_route_year_cell, axis=1)
        
        # Add source file tracking
        parsed["_source_file"] = raw["_source_file"]
        
        # Deduplicate by voyage_id
        n_before = len(parsed)
        parsed = parsed.drop_duplicates(subset=["voyage_id"], keep="first")
        n_after = len(parsed)
        if n_before > n_after:
            logger.info(f"Removed {n_before - n_after} duplicate voyage IDs")
        
        self._parsed_df = parsed
        logger.info(f"Parsed {len(parsed)} voyages")
        
        return parsed
    
    def save(self, output_path: Optional[Path] = None) -> Path:
        """
        Save parsed voyages to staging directory.
        
        Args:
            output_path: Optional custom output path
            
        Returns:
            Path to saved file
        """
        if self._parsed_df is None:
            self.parse()
        
        if output_path is None:
            output_path = STAGING_DIR / "voyages_master.parquet"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._parsed_df.to_parquet(output_path, index=False)
        logger.info(f"Saved voyages_master to {output_path}")
        
        # Also save CSV for easy inspection
        csv_path = output_path.with_suffix(".csv")
        self._parsed_df.to_csv(csv_path, index=False)
        
        return output_path
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of parsed data."""
        if self._parsed_df is None:
            self.parse()
        
        df = self._parsed_df
        
        return {
            "total_voyages": len(df),
            "unique_vessels": df["vessel_name_clean"].nunique(),
            "unique_captains": df["captain_name_clean"].nunique(),
            "unique_agents": df["agent_name_clean"].nunique(),
            "year_range": (
                df["year_out"].min(),
                df["year_out"].max(),
            ),
            "ports_out": df["port_out"].nunique(),
            "destinations": df["ground_or_route"].nunique(),
            "total_oil_bbl": df["q_oil_bbl"].sum(),
            "total_bone_lbs": df["q_bone_lbs"].sum(),
            "mean_duration_days": df["duration_days"].mean(),
            "missing_rates": {
                "date_out": df["date_out"].isna().mean(),
                "date_in": df["date_in"].isna().mean(),
                "captain": df["captain_name_clean"].isna().mean(),
                "agent": df["agent_name_clean"].isna().mean(),
                "oil": df["q_oil_bbl"].isna().mean(),
                "bone": df["q_bone_lbs"].isna().mean(),
            }
        }


if __name__ == "__main__":
    parser = VoyageParser()
    df = parser.parse()
    
    print("\n=== Voyage Data Summary ===")
    summary = parser.get_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    parser.save()
