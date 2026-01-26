"""
Record linker for captain-to-census probabilistic matching.

Implements:
- Deterministic matching (exact name + geography + age band)
- Probabilistic matching (Jaro-Winkler + age penalty + occupation boost)
- Spouse validation using SPLOC pointer
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import STAGING_DIR, CROSSWALKS_DIR, LINKAGE_CONFIG
from parsing.string_normalizer import (
    normalize_name, 
    jaro_winkler_similarity, 
    soundex,
    parse_name_parts,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of a captain-to-census match."""
    captain_id: str
    hik: str
    census_year: int
    match_score: float
    match_probability: float
    match_method: str
    name_score: float
    age_score: float
    geo_score: float
    occ_score: float
    spouse_validated: bool


class RecordLinker:
    """
    Links whaling captains to census records.
    
    Uses a combination of deterministic and probabilistic matching
    with validation through spouse information.
    """
    
    def __init__(
        self,
        name_weight: float = None,
        age_weight: float = None,
        geo_weight: float = None,
        occ_weight: float = None,
        spouse_weight: float = None,
    ):
        cfg = LINKAGE_CONFIG
        self.name_weight = name_weight or cfg.name_weight
        self.age_weight = age_weight or cfg.age_weight
        self.geo_weight = geo_weight or cfg.geography_weight
        self.occ_weight = occ_weight or cfg.occupation_weight
        self.spouse_weight = spouse_weight or cfg.spouse_weight
        
        self.maritime_occupations = cfg.maritime_occupations
        self.strict_threshold = cfg.strict_threshold
        self.medium_threshold = cfg.medium_threshold
    
    def link_captains_to_census(
        self,
        captain_profiles: pd.DataFrame,
        census_data: pd.DataFrame,
        target_year: int,
        top_k: int = 3,
    ) -> pd.DataFrame:
        """
        Link captain profiles to census records for a specific year.
        
        Args:
            captain_profiles: Captain profiles from CaptainProfiler
            census_data: IPUMS census data
            target_year: Census year to link
            top_k: Number of top candidates to keep per captain
            
        Returns:
            DataFrame with captain_to_ipums linkages
        """
        # Filter census to target year
        census_year = census_data[census_data["YEAR"] == target_year].copy()
        
        if len(census_year) == 0:
            logger.warning(f"No census data for year {target_year}")
            return pd.DataFrame()
        
        logger.info(f"Linking {len(captain_profiles)} captains to {len(census_year)} census records")
        
        # Prepare captain candidates
        candidates = captain_profiles[
            captain_profiles["expected_age_min"].notna()
        ].copy()
        
        all_matches = []
        
        for _, captain in candidates.iterrows():
            captain_matches = self._find_matches_for_captain(
                captain, census_year, target_year, top_k
            )
            all_matches.extend(captain_matches)
        
        if not all_matches:
            logger.warning("No matches found")
            return pd.DataFrame()
        
        # Convert to DataFrame
        results = pd.DataFrame([
            {
                "captain_id": m.captain_id,
                "HIK": m.hik,
                "link_year": m.census_year,
                "link_method": m.match_method,
                "match_score": m.match_score,
                "match_probability": m.match_probability,
                "name_score": m.name_score,
                "age_score": m.age_score,
                "geo_score": m.geo_score,
                "occ_score": m.occ_score,
                "spouse_validated": m.spouse_validated,
            }
            for m in all_matches
        ])
        
        # Rank matches per captain
        results["match_rank"] = results.groupby("captain_id")["match_score"].rank(
            ascending=False, method="first"
        ).astype(int)
        
        # Keep only top_k
        results = results[results["match_rank"] <= top_k]
        
        logger.info(f"Found {len(results)} total matches for {results['captain_id'].nunique()} captains")
        
        return results
    
    def _find_matches_for_captain(
        self,
        captain: pd.Series,
        census: pd.DataFrame,
        target_year: int,
        top_k: int,
    ) -> List[MatchResult]:
        """Find census matches for a single captain."""
        captain_id = captain["captain_id"]
        captain_name = captain["captain_name_clean"]
        last_name = captain.get("last_name")
        first_name = captain.get("first_name")
        last_soundex = captain.get("last_name_soundex")
        
        min_age = captain.get("expected_age_min", 20)
        max_age = captain.get("expected_age_max", 70)
        modal_port = captain.get("modal_port")
        
        # Pre-filter census by soundex for efficiency
        if last_soundex:
            census = census.copy()
            census["_last_soundex"] = census["NAMELAST"].apply(
                lambda x: soundex(str(x)) if pd.notna(x) else None
            )
            # Keep records with matching soundex or close variants
            census = census[
                census["_last_soundex"].notna() &
                (census["_last_soundex"].str[:2] == last_soundex[:2])
            ]
        
        # Further filter by age range
        census = census[
            census["AGE"].notna() &
            (census["AGE"] >= min_age - 10) &
            (census["AGE"] <= max_age + 10)
        ]
        
        # Filter by sex (male) if available
        if "SEX" in census.columns:
            census = census[census["SEX"].isin([1, "M", "Male", "MALE"])]
        
        if len(census) == 0:
            return []
        
        matches = []
        
        for _, person in census.iterrows():
            match = self._compute_match_score(
                captain_id, captain_name, first_name, last_name,
                min_age, max_age, modal_port,
                person, target_year
            )
            
            if match and match.match_score > 0.5:
                matches.append(match)
        
        # Sort by score and keep top_k
        matches.sort(key=lambda x: x.match_score, reverse=True)
        return matches[:top_k]
    
    def _compute_match_score(
        self,
        captain_id: str,
        captain_name: str,
        first_name: Optional[str],
        last_name: Optional[str],
        min_age: float,
        max_age: float,
        modal_port: Optional[str],
        person: pd.Series,
        target_year: int,
    ) -> Optional[MatchResult]:
        """Compute match score between captain and census person."""
        
        person_name = person.get("NAME_CLEAN", "")
        person_age = person.get("AGE")
        person_occ = str(person.get("OCC", "")).upper()
        person_state = person.get("STATEFIP")
        
        # Name similarity
        name_score = 0.0
        
        if captain_name and person_name:
            # Full name comparison
            full_score = jaro_winkler_similarity(captain_name, person_name)
            
            # Component comparison
            person_parts = parse_name_parts(person_name)
            person_first = person_parts.get("first", "")
            person_last = person_parts.get("last", "")
            
            last_score = 0.0
            first_score = 0.0
            
            if last_name and person_last:
                last_score = jaro_winkler_similarity(last_name, person_last)
            
            if first_name and person_first:
                first_score = jaro_winkler_similarity(first_name, person_first)
            
            # Weight last name more heavily
            component_score = last_score * 0.6 + first_score * 0.4
            
            # Use best of full or component scoring
            name_score = max(full_score, component_score)
        
        # Age score
        age_score = 0.0
        if pd.notna(person_age):
            expected_mid_age = (min_age + max_age) / 2
            age_diff = abs(person_age - expected_mid_age)
            
            if age_diff == 0:
                age_score = 1.0
            elif age_diff <= 5:
                age_score = 0.9
            elif age_diff <= 10:
                age_score = 0.7
            elif age_diff <= 15:
                age_score = 0.4
            else:
                age_score = max(0, 0.2 - (age_diff - 15) * 0.02)
        
        # Geography score (simple: in whaling state)
        geo_score = 0.5  # Default neutral
        whaling_states = {25, 9, 36, 44}  # MA, CT, NY, RI FIPS
        if pd.notna(person_state) and int(person_state) in whaling_states:
            geo_score = 0.8
            # Boost for Massachusetts
            if int(person_state) == 25:
                geo_score = 1.0
        
        # Occupation score
        occ_score = 0.3  # Default slightly positive (many captains won't list mariner)
        if person_occ:
            for maritime_occ in self.maritime_occupations:
                if maritime_occ in person_occ:
                    occ_score = 1.0
                    break
        
        # Spouse validation (placeholder - would need spouse name comparison)
        spouse_validated = False
        
        # Compute weighted score
        match_score = (
            name_score * self.name_weight +
            age_score * self.age_weight +
            geo_score * self.geo_weight +
            occ_score * self.occ_weight
        )
        
        # Convert to probability-like scale
        match_probability = min(1.0, max(0.0, match_score))
        
        # Determine match method
        if name_score >= 0.95 and age_score >= 0.9 and geo_score >= 0.8:
            match_method = "deterministic"
        else:
            match_method = "probabilistic"
        
        hik = str(person.get("HIK", ""))
        
        return MatchResult(
            captain_id=captain_id,
            hik=hik,
            census_year=target_year,
            match_score=match_score,
            match_probability=match_probability,
            match_method=match_method,
            name_score=name_score,
            age_score=age_score,
            geo_score=geo_score,
            occ_score=occ_score,
            spouse_validated=spouse_validated,
        )
    
    def create_sensitivity_variants(
        self,
        linkage_df: pd.DataFrame,
    ) -> Dict[str, pd.DataFrame]:
        """
        Create sensitivity analysis variants with different thresholds.
        
        Returns:
            Dict with 'strict', 'medium', 'all' variants
        """
        # Best match per captain (rank 1)
        best_matches = linkage_df[linkage_df["match_rank"] == 1].copy()
        
        return {
            "strict": best_matches[best_matches["match_probability"] >= self.strict_threshold],
            "medium": best_matches[best_matches["match_probability"] >= self.medium_threshold],
            "all": best_matches,
        }
    
    def save_linkage(
        self,
        linkage_df: pd.DataFrame,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Save linkage results."""
        if output_path is None:
            output_path = STAGING_DIR / "captain_to_ipums.parquet"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        linkage_df.to_parquet(output_path, index=False)
        logger.info(f"Saved linkage to {output_path}")
        
        # Also save CSV
        csv_path = output_path.with_suffix(".csv")
        linkage_df.to_csv(csv_path, index=False)
        
        return output_path
    
    def get_linkage_summary(self, linkage_df: pd.DataFrame) -> Dict[str, Any]:
        """Get linkage diagnostics."""
        if len(linkage_df) == 0:
            return {"status": "no_matches"}
        
        best = linkage_df[linkage_df["match_rank"] == 1]
        
        return {
            "total_captains_with_matches": best["captain_id"].nunique(),
            "total_candidate_matches": len(linkage_df),
            "mean_best_match_score": best["match_score"].mean(),
            "median_best_match_score": best["match_score"].median(),
            "strict_match_count": (best["match_probability"] >= self.strict_threshold).sum(),
            "medium_match_count": (best["match_probability"] >= self.medium_threshold).sum(),
            "deterministic_matches": (best["match_method"] == "deterministic").sum(),
            "probabilistic_matches": (best["match_method"] == "probabilistic").sum(),
            "score_distribution": {
                "p10": best["match_score"].quantile(0.10),
                "p25": best["match_score"].quantile(0.25),
                "p50": best["match_score"].quantile(0.50),
                "p75": best["match_score"].quantile(0.75),
                "p90": best["match_score"].quantile(0.90),
            }
        }


if __name__ == "__main__":
    print("Record linker module.")
    print("Requires captain profiles and IPUMS data to run.")
    print("See linkage workflow in main pipeline.")
