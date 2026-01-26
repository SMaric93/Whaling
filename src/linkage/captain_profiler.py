"""
Captain profiler for census linkage.

Constructs captain candidate profiles from AOWV voyage data
for matching against IPUMS census records.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Any
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import STAGING_DIR, LINKAGE_CONFIG
from parsing.string_normalizer import normalize_name, parse_name_parts, soundex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CaptainProfiler:
    """
    Constructs captain profiles from voyage data for census linkage.
    
    For each captain, computes:
    - Normalized name and variants
    - Career span (first voyage, last voyage)
    - Expected age range at census years
    - Modal home port and region
    - Typical whaling routes/grounds
    """
    
    def __init__(self, voyages_path: Optional[Path] = None):
        self.voyages_path = voyages_path or (STAGING_DIR / "voyages_master.parquet")
        self._voyages: Optional[pd.DataFrame] = None
        self._profiles: Optional[pd.DataFrame] = None
    
    def load_voyages(self) -> pd.DataFrame:
        """Load voyages data."""
        if not self.voyages_path.exists():
            raise FileNotFoundError(f"Voyages file not found: {self.voyages_path}")
        
        self._voyages = pd.read_parquet(self.voyages_path)
        logger.info(f"Loaded {len(self._voyages)} voyages")
        
        return self._voyages
    
    def build_profiles(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Build captain profiles from voyage data.
        
        Returns:
            DataFrame with one row per captain_id
        """
        if self._profiles is not None and not force_reload:
            return self._profiles
        
        if self._voyages is None:
            self.load_voyages()
        
        df = self._voyages
        
        # Filter to voyages with captain info
        has_captain = df[df["captain_id"].notna() & df["captain_name_clean"].notna()].copy()
        
        if len(has_captain) == 0:
            logger.warning("No voyages with captain information")
            return pd.DataFrame()
        
        logger.info(f"Building profiles from {len(has_captain)} voyages with captains")
        
        # Aggregate by captain
        profiles = has_captain.groupby("captain_id").agg(
            # Name info
            captain_name_clean=("captain_name_clean", "first"),
            name_variants=("captain_name_raw", lambda x: "|".join(set(str(v) for v in x if pd.notna(v)))),
            
            # Career span
            first_voyage_year=("year_out", "min"),
            last_voyage_year=("year_out", "max"),
            
            # Volume
            num_voyages=("voyage_id", "count"),
            total_oil_bbl=("q_oil_bbl", "sum"),
            total_bone_lbs=("q_bone_lbs", "sum"),
            
            # Geography
            ports_list=("home_port", lambda x: "|".join(set(str(v) for v in x if pd.notna(v)))),
            grounds_list=("ground_or_route", lambda x: "|".join(set(str(v) for v in x if pd.notna(v)))),
        ).reset_index()
        
        # Modal port
        def get_modal(series):
            if series.isna().all():
                return None
            return series.mode().iloc[0] if len(series.mode()) > 0 else None
        
        modal_ports = has_captain.groupby("captain_id")["home_port"].apply(get_modal)
        profiles["modal_port"] = profiles["captain_id"].map(modal_ports)
        
        # Modal ground
        modal_grounds = has_captain.groupby("captain_id")["ground_or_route"].apply(get_modal)
        profiles["modal_ground"] = profiles["captain_id"].map(modal_grounds)
        
        # Parse name components
        profiles["name_parts"] = profiles["captain_name_clean"].apply(parse_name_parts)
        profiles["first_name"] = profiles["name_parts"].apply(lambda x: x.get("first"))
        profiles["last_name"] = profiles["name_parts"].apply(lambda x: x.get("last"))
        profiles["name_suffix"] = profiles["name_parts"].apply(lambda x: x.get("suffix"))
        profiles = profiles.drop(columns=["name_parts"])
        
        # Soundex for fuzzy matching
        profiles["last_name_soundex"] = profiles["last_name"].apply(
            lambda x: soundex(x) if x else None
        )
        
        # Estimate birth year range
        # Assume captains were typically 25-50 years old during their career
        profiles["est_birth_year_early"] = profiles["first_voyage_year"] - 50
        profiles["est_birth_year_late"] = profiles["first_voyage_year"] - 25
        
        # Compute expected ages at census years
        for census_year in LINKAGE_CONFIG.target_years:
            profiles[f"expected_age_{census_year}_min"] = census_year - profiles["est_birth_year_late"]
            profiles[f"expected_age_{census_year}_max"] = census_year - profiles["est_birth_year_early"]
        
        # Career duration
        profiles["career_years"] = profiles["last_voyage_year"] - profiles["first_voyage_year"]
        
        # Whaling career era
        def classify_era(row):
            first_year = row["first_voyage_year"]
            if pd.isna(first_year):
                return "unknown"
            if first_year < 1820:
                return "early"
            elif first_year < 1840:
                return "golden_age"
            elif first_year < 1860:
                return "peak"
            elif first_year < 1880:
                return "decline"
            else:
                return "late"
        
        profiles["career_era"] = profiles.apply(classify_era, axis=1)
        
        self._profiles = profiles
        logger.info(f"Built {len(profiles)} captain profiles")
        
        return profiles
    
    def get_linkage_candidates(
        self,
        target_year: int,
        min_voyages: int = 1,
    ) -> pd.DataFrame:
        """
        Get captain candidates for linking to a specific census year.
        
        Args:
            target_year: Census year (e.g., 1860)
            min_voyages: Minimum voyages to be included
            
        Returns:
            DataFrame with candidates and expected attributes
        """
        if self._profiles is None:
            self.build_profiles()
        
        profiles = self._profiles.copy()
        
        # Filter by voyage count
        profiles = profiles[profiles["num_voyages"] >= min_voyages]
        
        # Filter to captains active around census year
        # Include captains whose career spans the census year ±10 years
        profiles = profiles[
            (profiles["first_voyage_year"] <= target_year + 10) &
            (profiles["last_voyage_year"] >= target_year - 10)
        ]
        
        # Add expected age columns for this year
        min_age_col = f"expected_age_{target_year}_min"
        max_age_col = f"expected_age_{target_year}_max"
        
        if min_age_col in profiles.columns:
            profiles["expected_age_min"] = profiles[min_age_col]
            profiles["expected_age_max"] = profiles[max_age_col]
        else:
            profiles["expected_age_min"] = target_year - profiles["est_birth_year_late"]
            profiles["expected_age_max"] = target_year - profiles["est_birth_year_early"]
        
        profiles["target_census_year"] = target_year
        
        logger.info(f"Found {len(profiles)} captain candidates for {target_year} census")
        
        return profiles
    
    def save(self, output_path: Optional[Path] = None) -> Path:
        """Save captain profiles."""
        if self._profiles is None:
            self.build_profiles()
        
        if output_path is None:
            output_path = STAGING_DIR / "captain_profiles.parquet"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._profiles.to_parquet(output_path, index=False)
        logger.info(f"Saved captain profiles to {output_path}")
        
        csv_path = output_path.with_suffix(".csv")
        self._profiles.to_csv(csv_path, index=False)
        
        return output_path
    
    def get_summary(self) -> Dict[str, Any]:
        """Get profile summary statistics."""
        if self._profiles is None:
            self.build_profiles()
        
        df = self._profiles
        
        return {
            "total_captains": len(df),
            "career_year_range": (
                df["first_voyage_year"].min(),
                df["last_voyage_year"].max()
            ),
            "mean_voyages": df["num_voyages"].mean(),
            "max_voyages": df["num_voyages"].max(),
            "mean_career_years": df["career_years"].mean(),
            "era_distribution": df["career_era"].value_counts().to_dict(),
            "with_port_info": df["modal_port"].notna().sum(),
            "with_ground_info": df["modal_ground"].notna().sum(),
        }


if __name__ == "__main__":
    profiler = CaptainProfiler()
    
    try:
        profiles = profiler.build_profiles()
        
        print("\n=== Captain Profiles Summary ===")
        summary = profiler.get_summary()
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        profiler.save()
        
        # Show linkage candidates for 1860
        candidates = profiler.get_linkage_candidates(1860)
        print(f"\n1860 linkage candidates: {len(candidates)}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run voyage parsing first.")
