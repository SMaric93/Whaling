"""
Parser for AOWV Crew Lists data.

Loads crew roster observations and identifies desertion events.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging
import hashlib

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_CREWLIST, STAGING_DIR
from parsing.string_normalizer import (
    normalize_name,
    parse_date,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CrewParser:
    """
    Parser for AOWV crew list data.
    
    Produces crew_roster table with desertion identification.
    """
    
    # Expected columns (flexible mapping)
    EXPECTED_COLUMNS = {
        "voyage_id": ["voyageID", "VoyageID", "VOYAGE_ID", "voyage_id", "VoyageId"],
        "crew_name": ["name", "Name", "NAME", "CrewName", "CrewMember"],
        "crew_first_name": ["name_first", "NAMEFRST", "first_name", "FirstName"],
        "crew_last_name": ["name_last", "NAMELAST", "last_name", "LastName"],
        "rank": ["rank", "Rank", "RANK", "Position", "Rating"],
        "status": ["remarks", "Remarks", "Status", "STATUS", "status", "Disposition"],
        "age": ["age", "Age", "AGE"],
        "height": ["height_feet", "height_inches", "Height", "HEIGHT", "height"],
        "birthplace": ["birthplace", "Birthplace", "BIRTHPLACE", "BirthPlace", "Origin"],
        "residence": ["res_city", "res_state", "Residence", "RESIDENCE", "residence"],
        "complexion": ["skin", "Complexion", "COMPLEXION", "complexion"],
        "vessel": ["voyage", "Vessel", "VESSEL", "vessel", "VesselName"],
        "date_out": ["list_date", "DateOut", "DATE_OUT", "date_out", "Departure"],
        "captain": ["Master", "MASTER", "master", "Captain"],
    }
    
    # Status keywords indicating desertion
    DESERTION_KEYWORDS = [
        "DESERT", "DESERTED", "RAN", "RUN AWAY", "RUNAWAY", "ABSCONDED",
        "LEFT", "DISCHARGED ASHORE", "PUT ASHORE", "LEFT SICK", "LEFT AT",
        "WENT ASHORE", "REMAINED", "STAYED",
    ]
    
    def __init__(self, raw_dir: Optional[Path] = None):
        self.raw_dir = raw_dir or RAW_CREWLIST
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
        """Load all raw crew files from the raw directory."""
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
            raise ValueError(f"No crew files found in {self.raw_dir}")
        
        combined = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined {len(dfs)} files: {len(combined)} total rows")
        
        return combined
    
    def _extract_field(self, df: pd.DataFrame, field: str) -> pd.Series:
        """Extract a field from the dataframe."""
        col = self._find_column(df, field)
        if col:
            return df[col]
        return pd.Series([None] * len(df))
    
    def _identify_desertion(self, status: Optional[str]) -> bool:
        """Check if status indicates desertion."""
        if status is None or pd.isna(status):
            return False
        
        status_upper = str(status).upper()
        
        for keyword in self.DESERTION_KEYWORDS:
            if keyword in status_upper:
                return True
        
        return False
    
    def _compute_crew_row_id(self, row: pd.Series, idx: int) -> str:
        """Compute a deterministic row ID for a crew member."""
        components = [
            str(row.get("voyage_id", "")),
            str(row.get("crew_name_clean", "")),
            str(row.get("rank", "")),
            str(idx),
        ]
        key = "|".join(components)
        return hashlib.md5(key.encode()).hexdigest()[:12]
    
    def parse(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Parse crew data into standardized format.
        
        Returns:
            DataFrame with crew_roster schema
        """
        if self._parsed_df is not None and not force_reload:
            return self._parsed_df
        
        raw = self._load_raw_files()
        self._raw_df = raw
        
        parsed = pd.DataFrame()
        
        # Extract voyage ID
        voyage_id_col = self._find_column(raw, "voyage_id")
        if voyage_id_col:
            parsed["voyage_id"] = raw[voyage_id_col].astype(str)
        else:
            # Will need crosswalk
            parsed["voyage_id"] = None
            parsed["_vessel_for_crosswalk"] = self._extract_field(raw, "vessel")
            parsed["_captain_for_crosswalk"] = self._extract_field(raw, "captain")
            parsed["_date_for_crosswalk"] = self._extract_field(raw, "date_out")
        
        # Extract crew name - try first/last name columns first (AOWV format)
        first_name = self._extract_field(raw, "crew_first_name")
        last_name = self._extract_field(raw, "crew_last_name")
        
        if first_name.notna().any() or last_name.notna().any():
            # Combine first and last names
            crew_name = first_name.fillna("").astype(str) + " " + last_name.fillna("").astype(str)
            crew_name = crew_name.str.strip()
        else:
            # Fall back to single name column
            crew_name = self._extract_field(raw, "crew_name")
        
        parsed["crew_name_raw"] = crew_name
        parsed["crew_name_clean"] = crew_name.apply(normalize_name)
        
        # Extract rank
        parsed["rank"] = self._extract_field(raw, "rank").apply(
            lambda x: str(x).upper().strip() if pd.notna(x) else None
        )
        
        # Extract status
        status = self._extract_field(raw, "status")
        parsed["crew_status"] = status.apply(
            lambda x: str(x).upper().strip() if pd.notna(x) else None
        )
        
        # Identify desertion
        parsed["is_deserted"] = status.apply(self._identify_desertion)
        
        # Demographic fields (when available)
        age = self._extract_field(raw, "age")
        parsed["age"] = pd.to_numeric(age, errors="coerce")
        
        parsed["birthplace"] = self._extract_field(raw, "birthplace").apply(
            lambda x: str(x).upper().strip() if pd.notna(x) else None
        )
        
        parsed["residence"] = self._extract_field(raw, "residence").apply(
            lambda x: str(x).upper().strip() if pd.notna(x) else None
        )
        
        height = self._extract_field(raw, "height")
        parsed["height"] = height  # Keep as-is for now
        
        parsed["complexion"] = self._extract_field(raw, "complexion").apply(
            lambda x: str(x).upper().strip() if pd.notna(x) else None
        )
        
        # Generate crew member row IDs
        parsed["crew_member_row_id"] = [
            self._compute_crew_row_id(row, idx) 
            for idx, row in parsed.iterrows()
        ]
        
        # Source tracking
        parsed["_source_file"] = raw["_source_file"]
        
        self._parsed_df = parsed
        logger.info(f"Parsed {len(parsed)} crew records")
        
        return parsed
    
    def save(self, output_path: Optional[Path] = None) -> Path:
        """Save parsed crew roster to staging directory."""
        if self._parsed_df is None:
            self.parse()
        
        if output_path is None:
            output_path = STAGING_DIR / "crew_roster.parquet"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._parsed_df.to_parquet(output_path, index=False)
        logger.info(f"Saved crew_roster to {output_path}")
        
        csv_path = output_path.with_suffix(".csv")
        self._parsed_df.to_csv(csv_path, index=False)
        
        return output_path
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if self._parsed_df is None:
            self.parse()
        
        df = self._parsed_df
        
        return {
            "total_crew_records": len(df),
            "unique_crew_names": df["crew_name_clean"].nunique(),
            "unique_voyages": df["voyage_id"].nunique() if df["voyage_id"].notna().any() else 0,
            "unique_ranks": df["rank"].nunique(),
            "desertion_count": df["is_deserted"].sum(),
            "desertion_rate": df["is_deserted"].mean(),
            "missing_voyage_id_rate": df["voyage_id"].isna().mean(),
            "age_available_rate": df["age"].notna().mean(),
            "birthplace_available_rate": df["birthplace"].notna().mean(),
        }


if __name__ == "__main__":
    parser = CrewParser()
    df = parser.parse()
    
    print("\n=== Crew Data Summary ===")
    summary = parser.get_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    parser.save()
