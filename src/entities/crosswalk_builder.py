"""
Crosswalk builder for linking datasets without common primary keys.

Uses fuzzy matching on normalized names and date windows.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import date, timedelta
from dataclasses import dataclass
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CROSSWALK_CONFIG, CROSSWALKS_DIR
from parsing.string_normalizer import (
    normalize_name,
    normalize_vessel_name,
    jaro_winkler_similarity,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CrosswalkMatch:
    """Represents a potential match in a crosswalk."""
    source_id: str
    target_voyage_id: str
    match_score: float
    match_reason: str
    vessel_score: float
    captain_score: float
    date_diff_days: Optional[int]


class CrosswalkBuilder:
    """
    Builds crosswalks between datasets using fuzzy matching.
    
    Used when crew lists or logbook data don't have VoyageID
    and must be linked via vessel name, captain name, and dates.
    """
    
    def __init__(
        self,
        date_out_tolerance: int = None,
        date_in_tolerance: int = None,
        min_name_similarity: float = None,
    ):
        self.date_out_tolerance = date_out_tolerance or CROSSWALK_CONFIG.out_date_tolerance_days
        self.date_in_tolerance = date_in_tolerance or CROSSWALK_CONFIG.in_date_tolerance_days
        self.min_similarity = min_name_similarity or CROSSWALK_CONFIG.min_name_similarity
    
    def build_crew_to_voyage_crosswalk(
        self,
        crew_df: pd.DataFrame,
        voyage_df: pd.DataFrame,
        output_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Build crosswalk linking crew records to voyages.
        
        Args:
            crew_df: Crew roster DataFrame (must have _vessel_for_crosswalk, 
                     _captain_for_crosswalk, _date_for_crosswalk)
            voyage_df: Voyages DataFrame (must have voyage_id, vessel_name_clean,
                       captain_name_clean, date_out)
            output_path: Optional path to save crosswalk
            
        Returns:
            DataFrame with source row indices and matched voyage_ids
        """
        # Filter to crew records needing crosswalk
        needs_crosswalk = crew_df[crew_df["voyage_id"].isna()].copy()
        
        if len(needs_crosswalk) == 0:
            logger.info("All crew records have voyage_id, no crosswalk needed")
            return pd.DataFrame(columns=["crew_index", "voyage_id", "match_score", "match_reason"])
        
        logger.info(f"Building crosswalk for {len(needs_crosswalk)} crew records")
        
        # Normalize crosswalk fields
        needs_crosswalk["_vessel_clean"] = needs_crosswalk["_vessel_for_crosswalk"].apply(
            normalize_vessel_name
        )
        needs_crosswalk["_captain_clean"] = needs_crosswalk["_captain_for_crosswalk"].apply(
            normalize_name
        )
        
        # Build voyage lookup structures
        voyage_index = voyage_df[["voyage_id", "vessel_name_clean", "captain_name_clean", "date_out"]].copy()
        voyage_index = voyage_index.dropna(subset=["vessel_name_clean"])
        
        # Match records
        matches = []
        
        for idx, row in needs_crosswalk.iterrows():
            best_match = self._find_best_voyage_match(row, voyage_index)
            if best_match:
                matches.append({
                    "crew_index": idx,
                    "voyage_id": best_match.target_voyage_id,
                    "match_score": best_match.match_score,
                    "match_reason": best_match.match_reason,
                    "vessel_score": best_match.vessel_score,
                    "captain_score": best_match.captain_score,
                    "date_diff_days": best_match.date_diff_days,
                })
        
        crosswalk_df = pd.DataFrame(matches)
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            crosswalk_df.to_csv(output_path, index=False)
            logger.info(f"Saved crosswalk to {output_path}")
        
        logger.info(f"Created {len(crosswalk_df)} crosswalk matches")
        
        return crosswalk_df
    
    def _find_best_voyage_match(
        self,
        source_row: pd.Series,
        voyage_index: pd.DataFrame,
    ) -> Optional[CrosswalkMatch]:
        """Find the best matching voyage for a source record."""
        source_vessel = source_row.get("_vessel_clean")
        source_captain = source_row.get("_captain_clean")
        source_date_str = source_row.get("_date_for_crosswalk")
        
        if not source_vessel:
            return None
        
        candidates = []
        
        for _, voyage_row in voyage_index.iterrows():
            voyage_id = voyage_row["voyage_id"]
            voyage_vessel = voyage_row["vessel_name_clean"]
            voyage_captain = voyage_row["captain_name_clean"]
            voyage_date = voyage_row["date_out"]
            
            # Calculate vessel similarity
            vessel_score = jaro_winkler_similarity(
                source_vessel or "", 
                voyage_vessel or ""
            )
            
            if vessel_score < self.min_similarity:
                continue
            
            # Calculate captain similarity
            if source_captain and voyage_captain:
                captain_score = jaro_winkler_similarity(source_captain, voyage_captain)
            else:
                captain_score = 0.5  # Neutral if missing
            
            # Calculate date difference (if available)
            date_diff = None
            date_penalty = 0
            
            if source_date_str and voyage_date:
                try:
                    from parsing.string_normalizer import parse_date
                    source_date = parse_date(source_date_str, return_year_only=True)
                    if source_date:
                        date_diff = abs((source_date - voyage_date).days)
                        if date_diff > self.date_out_tolerance:
                            date_penalty = min((date_diff - self.date_out_tolerance) / 365, 0.5)
                except:
                    pass
            
            # Combined score
            combined_score = (
                vessel_score * 0.5 +
                captain_score * 0.4 +
                (0.1 if date_diff is None else 0.1 * max(0, 1 - date_penalty))
            )
            
            candidates.append(CrosswalkMatch(
                source_id=str(source_row.name),
                target_voyage_id=voyage_id,
                match_score=combined_score,
                match_reason=f"vessel={vessel_score:.2f},captain={captain_score:.2f}",
                vessel_score=vessel_score,
                captain_score=captain_score,
                date_diff_days=date_diff,
            ))
        
        if not candidates:
            return None
        
        # Return best match
        best = max(candidates, key=lambda x: x.match_score)
        
        # Apply minimum threshold
        if best.match_score < self.min_similarity:
            return None
        
        return best
    
    def build_logbook_to_voyage_crosswalk(
        self,
        logbook_df: pd.DataFrame,
        voyage_df: pd.DataFrame,
        output_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Build crosswalk linking logbook records to voyages.
        
        Similar to crew crosswalk but handles position observations.
        """
        needs_crosswalk = logbook_df[logbook_df["voyage_id"].isna()].copy()
        
        if len(needs_crosswalk) == 0:
            logger.info("All logbook records have voyage_id, no crosswalk needed")
            return pd.DataFrame(columns=["logbook_index", "voyage_id", "match_score"])
        
        logger.info(f"Building crosswalk for {len(needs_crosswalk)} logbook records")
        
        # Group by vessel/captain to reduce matching iterations
        grouped = needs_crosswalk.groupby(
            ["_vessel_for_crosswalk", "_captain_for_crosswalk"],
            dropna=False
        )
        
        voyage_index = voyage_df[["voyage_id", "vessel_name_clean", "captain_name_clean", 
                                   "date_out", "date_in"]].copy()
        
        matches = []
        
        for (vessel, captain), group in grouped:
            vessel_clean = normalize_vessel_name(vessel) if vessel else None
            captain_clean = normalize_name(captain) if captain else None
            
            if not vessel_clean:
                continue
            
            # Find matching voyages for this vessel/captain combination
            for _, voyage_row in voyage_index.iterrows():
                voyage_vessel = voyage_row["vessel_name_clean"]
                voyage_captain = voyage_row["captain_name_clean"]
                
                vessel_score = jaro_winkler_similarity(
                    vessel_clean or "",
                    voyage_vessel or ""
                )
                
                if vessel_score < self.min_similarity:
                    continue
                
                captain_score = 0.5
                if captain_clean and voyage_captain:
                    captain_score = jaro_winkler_similarity(captain_clean, voyage_captain)
                
                combined_score = vessel_score * 0.6 + captain_score * 0.4
                
                if combined_score >= self.min_similarity:
                    for idx in group.index:
                        matches.append({
                            "logbook_index": idx,
                            "voyage_id": voyage_row["voyage_id"],
                            "match_score": combined_score,
                            "vessel_score": vessel_score,
                            "captain_score": captain_score,
                        })
        
        crosswalk_df = pd.DataFrame(matches)
        
        # Keep only best match per logbook record
        if len(crosswalk_df) > 0:
            crosswalk_df = crosswalk_df.sort_values("match_score", ascending=False)
            crosswalk_df = crosswalk_df.drop_duplicates(subset=["logbook_index"], keep="first")
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            crosswalk_df.to_csv(output_path, index=False)
            logger.info(f"Saved crosswalk to {output_path}")
        
        logger.info(f"Created {len(crosswalk_df)} crosswalk matches")
        
        return crosswalk_df
    
    def apply_crosswalk(
        self,
        source_df: pd.DataFrame,
        crosswalk_df: pd.DataFrame,
        source_index_col: str = "crew_index",
        min_score: float = 0.0,
    ) -> pd.DataFrame:
        """
        Apply a crosswalk to attach voyage_ids to source records.
        
        Args:
            source_df: Source DataFrame needing voyage_ids
            crosswalk_df: Crosswalk with matching voyage_ids
            source_index_col: Column name in crosswalk containing source indices
            min_score: Minimum match score to accept
            
        Returns:
            Source DataFrame with voyage_id filled in from crosswalk
        """
        result = source_df.copy()
        
        # Filter crosswalk by score
        valid_matches = crosswalk_df[crosswalk_df["match_score"] >= min_score]
        
        # Create lookup
        match_lookup = dict(zip(
            valid_matches[source_index_col],
            valid_matches["voyage_id"]
        ))
        
        # Apply matches
        for idx, voyage_id in match_lookup.items():
            if idx in result.index:
                result.loc[idx, "voyage_id"] = voyage_id
        
        # Track crosswalk source
        result["_voyage_id_source"] = np.where(
            result.index.isin(match_lookup.keys()),
            "crosswalk",
            np.where(result["voyage_id"].notna(), "original", "missing")
        )
        
        logger.info(f"Applied {len(match_lookup)} crosswalk matches")
        
        return result
