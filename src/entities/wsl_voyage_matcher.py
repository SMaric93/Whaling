"""
WSL to Voyage Crosswalk Matcher.

Matches extracted WSL events to voyages in voyages_master using:
- Blocking by vessel_name_clean and date window
- Scoring by vessel/captain similarity and date proximity
- Threshold-based assignment with confidence levels
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    STAGING_DIR, MATCH_CONFIDENCE_THRESHOLDS, CROSSWALK_CONFIG,
)

# Try to import fuzzy matching
try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    try:
        from fuzzywuzzy import fuzz
        HAS_RAPIDFUZZ = True
    except ImportError:
        HAS_RAPIDFUZZ = False
        fuzz = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MatchCandidate:
    """A candidate voyage match for a WSL event."""
    voyage_id: str
    vessel_name_clean: str
    captain_name_clean: Optional[str]
    date_out: Optional[str]
    date_in: Optional[str]
    port_out: Optional[str]
    port_in: Optional[str]
    
    # Scoring components
    vessel_score: float = 0.0
    captain_score: float = 0.0
    date_score: float = 0.0
    port_score: float = 0.0
    
    @property
    def total_score(self) -> float:
        """Weighted total match score."""
        # Weights: vessel most important, then date, then captain, then port
        return (
            0.40 * self.vessel_score +
            0.30 * self.date_score +
            0.20 * self.captain_score +
            0.10 * self.port_score
        )


def compute_name_similarity(name1: Optional[str], name2: Optional[str]) -> float:
    """
    Compute fuzzy similarity between two names.
    
    Returns score between 0 and 1.
    """
    if not name1 or not name2:
        return 0.0
    
    if name1 == name2:
        return 1.0
    
    if HAS_RAPIDFUZZ:
        # Use token_sort_ratio for name matching (handles word order)
        return fuzz.token_sort_ratio(name1, name2) / 100.0
    else:
        # Simple fallback: check if one contains the other
        n1, n2 = name1.upper(), name2.upper()
        if n1 in n2 or n2 in n1:
            return 0.8
        # Check first word match (common for ships)
        if n1.split()[0] == n2.split()[0]:
            return 0.7
        return 0.0


def compute_date_score(
    event_date: Optional[str],
    voyage_date_out: Optional[str],
    voyage_date_in: Optional[str],
    tolerance_days: int = 60,
) -> float:
    """
    Compute date proximity score for an event relative to voyage window.
    
    Args:
        event_date: Event date string (YYYY-MM-DD or partial)
        voyage_date_out: Voyage departure date
        voyage_date_in: Voyage return date
        tolerance_days: Days of tolerance outside voyage window
        
    Returns:
        Score between 0 and 1
    """
    if not event_date:
        return 0.3  # Neutral score if no date
    
    # Parse dates
    def parse_date(d: str) -> Optional[datetime]:
        if not d:
            return None
        try:
            # Handle partial dates
            if len(d) == 4:  # Year only
                return datetime(int(d), 6, 15)  # Mid-year
            elif len(d) == 7:  # YYYY-MM
                return datetime.strptime(d, "%Y-%m")
            else:
                return datetime.strptime(d[:10], "%Y-%m-%d")
        except:
            return None
    
    event_dt = parse_date(event_date)
    out_dt = parse_date(voyage_date_out)
    in_dt = parse_date(voyage_date_in)
    
    if not event_dt:
        return 0.3
    
    # If we have both voyage dates, check if event falls within window
    if out_dt and in_dt:
        window_start = out_dt - timedelta(days=tolerance_days)
        window_end = in_dt + timedelta(days=tolerance_days)
        
        if window_start <= event_dt <= window_end:
            # Full score if within core voyage period
            if out_dt <= event_dt <= in_dt:
                return 1.0
            # Reduced score if in tolerance zone
            return 0.7
        else:
            # Outside window - calculate penalty
            if event_dt < window_start:
                days_out = (window_start - event_dt).days
            else:
                days_out = (event_dt - window_end).days
            # Decay score based on distance
            return max(0.0, 0.5 - days_out / 365)
    
    # If only departure date, check if event is after
    elif out_dt:
        if event_dt >= out_dt - timedelta(days=tolerance_days):
            return 0.6
        return 0.2
    
    # If only return date, check if event is before
    elif in_dt:
        if event_dt <= in_dt + timedelta(days=tolerance_days):
            return 0.6
        return 0.2
    
    return 0.3


def compute_port_score(
    event_port: Optional[str],
    voyage_port_out: Optional[str],
    voyage_port_in: Optional[str],
) -> float:
    """
    Compute port consistency score.
    
    Args:
        event_port: Port mentioned in event
        voyage_port_out: Voyage departure port
        voyage_port_in: Voyage return port
        
    Returns:
        Score between 0 and 1
    """
    if not event_port:
        return 0.5  # Neutral
    
    event_port_upper = event_port.upper()
    
    # Check against both voyage ports
    for vport in [voyage_port_out, voyage_port_in]:
        if vport:
            vport_upper = vport.upper()
            if event_port_upper == vport_upper:
                return 1.0
            if event_port_upper in vport_upper or vport_upper in event_port_upper:
                return 0.8
    
    return 0.3


def generate_candidates(
    event: pd.Series,
    voyages: pd.DataFrame,
    date_tolerance_days: int = 60,
) -> List[MatchCandidate]:
    """
    Generate voyage candidates for a WSL event using blocking.
    
    Args:
        event: Single WSL event row
        voyages: Voyages master DataFrame
        date_tolerance_days: Date window expansion
        
    Returns:
        List of MatchCandidate objects
    """
    candidates = []
    
    vessel_clean = event.get('vessel_name_clean')
    event_date = event.get('event_date')
    
    if not vessel_clean:
        return candidates
    
    # Block 1: Exact vessel name match
    exact_matches = voyages[voyages['vessel_name_clean'] == vessel_clean]
    
    # Block 2: Fuzzy vessel name match (if exact yields few results)
    if len(exact_matches) < 5 and HAS_RAPIDFUZZ:
        fuzzy_matches = voyages[
            voyages['vessel_name_clean'].apply(
                lambda x: compute_name_similarity(vessel_clean, x) >= 0.85 if x else False
            )
        ]
        exact_matches = pd.concat([exact_matches, fuzzy_matches]).drop_duplicates()
    
    # Create candidates from matches
    for _, voyage in exact_matches.iterrows():
        candidate = MatchCandidate(
            voyage_id=voyage.get('voyage_id'),
            vessel_name_clean=voyage.get('vessel_name_clean'),
            captain_name_clean=voyage.get('captain_name_clean'),
            date_out=voyage.get('date_out'),
            date_in=voyage.get('date_in'),
            port_out=voyage.get('port_out'),
            port_in=voyage.get('port_in'),
        )
        
        # Compute scores
        candidate.vessel_score = compute_name_similarity(
            vessel_clean, candidate.vessel_name_clean
        )
        candidate.captain_score = compute_name_similarity(
            event.get('captain_name_clean'), candidate.captain_name_clean
        )
        candidate.date_score = compute_date_score(
            event_date, candidate.date_out, candidate.date_in, date_tolerance_days
        )
        candidate.port_score = compute_port_score(
            event.get('port_name_clean'), candidate.port_out, candidate.port_in
        )
        
        candidates.append(candidate)
    
    # Sort by total score
    candidates.sort(key=lambda c: c.total_score, reverse=True)
    
    return candidates


def match_events_to_voyages(
    events_df: pd.DataFrame,
    voyages_df: pd.DataFrame,
    confidence_threshold: float = 0.7,
    max_candidates: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Match WSL events to voyages and create crosswalk.
    
    Args:
        events_df: WSL extracted events DataFrame
        voyages_df: Voyages master DataFrame
        confidence_threshold: Minimum score to assign a match
        max_candidates: Number of top candidates to store for QA
        
    Returns:
        Tuple of (crosswalk_df, diagnostics_df)
    """
    logger.info(f"Matching {len(events_df)} events to {len(voyages_df)} voyages")
    
    crosswalk_rows = []
    
    for idx, event in events_df.iterrows():
        candidates = generate_candidates(event, voyages_df)
        
        # Build crosswalk row
        row = {
            'wsl_event_id': event['wsl_event_id'],
            'voyage_id': None,
            'match_method': 'none',
            'match_confidence': 0.0,
            'match_reason': '',
            'top_candidates': [],
        }
        
        if candidates:
            best = candidates[0]
            
            if best.total_score >= confidence_threshold:
                row['voyage_id'] = best.voyage_id
                row['match_method'] = 'scored'
                row['match_confidence'] = best.total_score
                row['match_reason'] = (
                    f"vessel={best.vessel_score:.2f}, "
                    f"captain={best.captain_score:.2f}, "
                    f"date={best.date_score:.2f}, "
                    f"port={best.port_score:.2f}"
                )
            else:
                row['match_method'] = 'below_threshold'
                row['match_confidence'] = best.total_score
                row['match_reason'] = f"best_score={best.total_score:.2f} < threshold={confidence_threshold}"
            
            # Store top candidates for QA
            row['top_candidates'] = [
                {'voyage_id': c.voyage_id, 'score': c.total_score}
                for c in candidates[:max_candidates]
            ]
        else:
            row['match_method'] = 'no_candidates'
            row['match_reason'] = 'No vessel name matches found'
        
        crosswalk_rows.append(row)
    
    crosswalk_df = pd.DataFrame(crosswalk_rows)
    
    # Compute diagnostics
    matched = crosswalk_df['voyage_id'].notna().sum()
    total = len(crosswalk_df)
    match_rate = matched / total if total > 0 else 0
    
    logger.info(f"Matched {matched}/{total} events ({match_rate:.1%})")
    
    return crosswalk_df, None


def build_voyage_event_panel(
    crosswalk_df: pd.DataFrame,
    events_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate matched events into voyage-level panel.
    
    Args:
        crosswalk_df: WSL to voyage crosswalk
        events_df: WSL extracted events
        
    Returns:
        Voyage event panel DataFrame
    """
    # Merge events with crosswalk
    merged = events_df.merge(
        crosswalk_df[['wsl_event_id', 'voyage_id', 'match_confidence']],
        on='wsl_event_id',
        how='left'
    )
    
    # Filter to matched events
    matched = merged[merged['voyage_id'].notna()]
    
    if len(matched) == 0:
        return pd.DataFrame()
    
    # Group by voyage, date, event type
    panel = matched.groupby(
        ['voyage_id', 'event_date', 'event_type']
    ).agg({
        'wsl_event_id': 'count',
        'wsl_issue_id': 'first',
        'port_name_clean': 'first',
        'match_confidence': 'mean',
    }).reset_index()
    
    panel.columns = [
        'voyage_id', 'event_date', 'event_type',
        'event_count', 'source_wsl_issue_id', 'port_name_clean', 'mean_confidence'
    ]
    
    return panel


def save_crosswalk_outputs(
    crosswalk_df: pd.DataFrame,
    event_panel_df: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Save crosswalk and panel to staging.
    
    Args:
        crosswalk_df: WSL to voyage crosswalk
        event_panel_df: Voyage event panel
        output_dir: Output directory (defaults to staging)
        
    Returns:
        Dict of output file paths
    """
    if output_dir is None:
        output_dir = STAGING_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    
    # Save crosswalk (convert top_candidates to string for parquet)
    crosswalk_for_save = crosswalk_df.copy()
    crosswalk_for_save['top_candidates'] = crosswalk_for_save['top_candidates'].apply(str)
    
    crosswalk_path = output_dir / "wsl_to_voyage_crosswalk.parquet"
    crosswalk_for_save.to_parquet(crosswalk_path)
    crosswalk_for_save.to_csv(crosswalk_path.with_suffix('.csv'), index=False)
    outputs['crosswalk'] = crosswalk_path
    logger.info(f"Saved crosswalk: {crosswalk_path}")
    
    # Save event panel
    if len(event_panel_df) > 0:
        panel_path = output_dir / "wsl_voyage_event_panel.parquet"
        event_panel_df.to_parquet(panel_path)
        event_panel_df.to_csv(panel_path.with_suffix('.csv'), index=False)
        outputs['panel'] = panel_path
        logger.info(f"Saved event panel: {panel_path}")
    
    return outputs


def run_wsl_crosswalk(
    events_path: Optional[Path] = None,
    voyages_path: Optional[Path] = None,
    confidence_threshold: float = 0.7,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full WSL crosswalk pipeline.
    
    Args:
        events_path: Path to wsl_extracted_events (defaults to staging)
        voyages_path: Path to voyages_master (defaults to staging)
        confidence_threshold: Match confidence threshold
        
    Returns:
        Tuple of (crosswalk_df, event_panel_df)
    """
    if events_path is None:
        events_path = STAGING_DIR / "wsl_extracted_events.parquet"
    if voyages_path is None:
        voyages_path = STAGING_DIR / "voyages_master.parquet"
    
    # Load data
    logger.info(f"Loading events from {events_path}")
    events_df = pd.read_parquet(events_path)
    
    logger.info(f"Loading voyages from {voyages_path}")
    voyages_df = pd.read_parquet(voyages_path)
    
    # Run matching
    crosswalk_df, _ = match_events_to_voyages(
        events_df, voyages_df, confidence_threshold
    )
    
    # Build panel
    event_panel_df = build_voyage_event_panel(crosswalk_df, events_df)
    
    # Save outputs
    save_crosswalk_outputs(crosswalk_df, event_panel_df)
    
    return crosswalk_df, event_panel_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Match WSL events to voyages")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Match confidence threshold")
    
    args = parser.parse_args()
    
    crosswalk, panel = run_wsl_crosswalk(confidence_threshold=args.threshold)
    
    print(f"\nCrosswalk: {len(crosswalk)} events")
    print(f"Matched: {crosswalk['voyage_id'].notna().sum()}")
    print(f"Event Panel: {len(panel)} voyage-event combinations")
