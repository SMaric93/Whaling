"""
Maury to Voyage Matcher.

Maps Maury position observations to AOWV voyages and computes
route validation metrics comparing exposure across sources.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import STAGING_DIR, ARCTIC_LAT_THRESHOLD

# Fuzzy matching
try:
    from rapidfuzz import fuzz
    HAS_FUZZ = True
except ImportError:
    HAS_FUZZ = False
    fuzz = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_name_similarity(name1: Optional[str], name2: Optional[str]) -> float:
    """Compute fuzzy name similarity."""
    if not name1 or not name2:
        return 0.0
    if name1 == name2:
        return 1.0
    if HAS_FUZZ:
        return fuzz.token_sort_ratio(name1, name2) / 100.0
    n1, n2 = name1.upper(), name2.upper()
    return 0.8 if (n1 in n2 or n2 in n1) else 0.0


def parse_year(date_str: Optional[str]) -> Optional[int]:
    """Extract year from date string."""
    if not date_str:
        return None
    try:
        return int(str(date_str)[:4])
    except:
        return None


def match_positions_to_voyages(
    positions_df: pd.DataFrame,
    voyages_df: pd.DataFrame,
    name_threshold: float = 0.8,
) -> pd.DataFrame:
    """
    Match Maury positions to AOWV voyages.
    
    Args:
        positions_df: Maury positions DataFrame
        voyages_df: AOWV voyages master
        name_threshold: Minimum name similarity
        
    Returns:
        Crosswalk DataFrame
    """
    logger.info(f"Matching {len(positions_df)} positions to {len(voyages_df)} voyages")
    
    # Get unique vessel-year combinations from positions
    positions_df['obs_year'] = positions_df['obs_date'].apply(parse_year)
    
    vessel_years = positions_df.groupby(
        ['vessel_name_clean', 'obs_year']
    ).agg({
        'maury_obs_id': 'first',
        'captain_name_clean': 'first',
    }).reset_index()
    
    # Prepare voyages with year ranges
    voyages_df = voyages_df.copy()
    voyages_df['year_out'] = voyages_df['date_out'].apply(parse_year)
    voyages_df['year_in'] = voyages_df['date_in'].apply(parse_year)
    
    crosswalk_rows = []
    
    for _, pos_row in vessel_years.iterrows():
        pos_vessel = pos_row['vessel_name_clean']
        pos_year = pos_row['obs_year']
        pos_captain = pos_row['captain_name_clean']
        
        if not pos_vessel or not pos_year:
            continue
        
        best_match = None
        best_score = 0.0
        
        for _, voy_row in voyages_df.iterrows():
            voy_vessel = voy_row.get('vessel_name_clean')
            voy_captain = voy_row.get('captain_name_clean')
            year_out = voy_row.get('year_out')
            year_in = voy_row.get('year_in')
            
            # Name match
            name_sim = compute_name_similarity(pos_vessel, voy_vessel)
            if name_sim < name_threshold:
                continue
            
            # Year match
            if year_out and year_in:
                if not (year_out <= pos_year <= year_in):
                    continue
            elif year_out:
                if pos_year < year_out or pos_year > year_out + 5:
                    continue
            elif year_in:
                if pos_year > year_in or pos_year < year_in - 5:
                    continue
            else:
                continue
            
            # Captain match bonus
            captain_sim = compute_name_similarity(pos_captain, voy_captain)
            
            score = 0.7 * name_sim + 0.3 * captain_sim
            
            if score > best_score:
                best_score = score
                best_match = voy_row.get('voyage_id')
        
        # Add all positions for this vessel-year
        for obs_id in positions_df[
            (positions_df['vessel_name_clean'] == pos_vessel) &
            (positions_df['obs_year'] == pos_year)
        ]['maury_obs_id']:
            crosswalk_rows.append({
                'maury_obs_id': obs_id,
                'voyage_id': best_match,
                'match_confidence': best_score if best_match else 0.0,
                'match_method': 'vessel_year_captain' if best_match else 'no_match',
            })
    
    crosswalk_df = pd.DataFrame(crosswalk_rows)
    
    matched = crosswalk_df['voyage_id'].notna().sum()
    logger.info(f"Matched {matched}/{len(crosswalk_df)} positions ({matched/len(crosswalk_df):.1%})")
    
    return crosswalk_df


def compute_route_validation_metrics(
    positions_df: pd.DataFrame,
    crosswalk_df: pd.DataFrame,
    aowv_routes_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute route validation metrics per voyage.
    
    Args:
        positions_df: Maury positions
        crosswalk_df: Position to voyage crosswalk
        aowv_routes_df: AOWV voyage routes (if available)
        
    Returns:
        Route validation metrics DataFrame
    """
    # Merge positions with crosswalk
    merged = positions_df.merge(
        crosswalk_df[['maury_obs_id', 'voyage_id', 'match_confidence']],
        on='maury_obs_id',
        how='left'
    )
    
    # Filter matched
    matched = merged[merged['voyage_id'].notna()]
    
    if len(matched) == 0:
        return pd.DataFrame()
    
    # Compute per-voyage metrics
    metrics = matched.groupby('voyage_id').agg({
        'maury_obs_id': 'count',
        'lat': ['mean', 'max'],
        'match_confidence': 'mean',
    })
    metrics.columns = ['maury_days', 'mean_lat', 'max_lat', 'mean_confidence']
    metrics = metrics.reset_index()
    
    # Compute arctic exposure from Maury
    metrics['frac_arctic_maury'] = matched.groupby('voyage_id')['lat'].apply(
        lambda x: (x > ARCTIC_LAT_THRESHOLD).mean()
    ).values
    
    # If AOWV routes available, compare
    if aowv_routes_df is not None and len(aowv_routes_df) > 0:
        aowv_metrics = aowv_routes_df[['voyage_id', 'frac_days_arctic', 'route_days']].copy()
        aowv_metrics = aowv_metrics.rename(columns={
            'frac_days_arctic': 'frac_arctic_aowv',
            'route_days': 'aowv_logbook_days'
        })
        
        metrics = metrics.merge(aowv_metrics, on='voyage_id', how='left')
        
        # Compute discrepancy
        metrics['arctic_exposure_diff'] = (
            metrics['frac_arctic_maury'] - metrics['frac_arctic_aowv'].fillna(0)
        ).abs()
        
        # Flag large discrepancies
        metrics['route_discrepancy_flag'] = metrics['arctic_exposure_diff'] > 0.2
    else:
        metrics['aowv_logbook_days'] = None
        metrics['frac_arctic_aowv'] = None
        metrics['arctic_exposure_diff'] = None
        metrics['route_discrepancy_flag'] = False
    
    # Route overlap score (if both sources have data)
    metrics['route_overlap_score'] = metrics.apply(
        lambda r: 1.0 - r['arctic_exposure_diff'] if pd.notna(r['arctic_exposure_diff']) else None,
        axis=1
    )
    
    # Add flags column
    metrics['flags'] = metrics.apply(
        lambda r: 'HIGH_DISCREPANCY' if r['route_discrepancy_flag'] else '',
        axis=1
    )
    
    return metrics


def save_maury_outputs(
    crosswalk_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """Save Maury crosswalk and validation outputs."""
    if output_dir is None:
        output_dir = STAGING_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    
    # Crosswalk
    crosswalk_path = output_dir / "maury_to_voyage_crosswalk.parquet"
    crosswalk_df.to_parquet(crosswalk_path)
    crosswalk_df.to_csv(crosswalk_path.with_suffix('.csv'), index=False)
    outputs['crosswalk'] = crosswalk_path
    
    # Validation metrics
    if len(validation_df) > 0:
        validation_path = output_dir / "route_validation_metrics.parquet"
        validation_df.to_parquet(validation_path)
        validation_df.to_csv(validation_path.with_suffix('.csv'), index=False)
        outputs['validation'] = validation_path
    
    logger.info(f"Saved Maury outputs to {output_dir}")
    return outputs


def run_maury_voyage_matching(
    positions_path: Optional[Path] = None,
    voyages_path: Optional[Path] = None,
    routes_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run full Maury matching and validation pipeline.
    
    Args:
        positions_path: Path to maury_positions
        voyages_path: Path to voyages_master
        routes_path: Path to voyage_routes (optional)
        
    Returns:
        Tuple of (crosswalk_df, validation_df)
    """
    if positions_path is None:
        positions_path = STAGING_DIR / "maury_positions.parquet"
    if voyages_path is None:
        voyages_path = STAGING_DIR / "voyages_master.parquet"
    if routes_path is None:
        routes_path = STAGING_DIR / "voyage_routes.parquet"
    
    # Load data
    positions_df = pd.read_parquet(positions_path)
    voyages_df = pd.read_parquet(voyages_path)
    
    routes_df = None
    if routes_path.exists():
        routes_df = pd.read_parquet(routes_path)
    
    # Match
    crosswalk_df = match_positions_to_voyages(positions_df, voyages_df)
    
    # Validate
    validation_df = compute_route_validation_metrics(
        positions_df, crosswalk_df, routes_df
    )
    
    # Save
    save_maury_outputs(crosswalk_df, validation_df)
    
    return crosswalk_df, validation_df


if __name__ == "__main__":
    crosswalk, validation = run_maury_voyage_matching()
    
    print(f"\nMaury Crosswalk: {len(crosswalk)} positions")
    print(f"Matched: {crosswalk['voyage_id'].notna().sum()}")
    
    if len(validation) > 0:
        print(f"\nRoute Validation: {len(validation)} voyages")
        print(f"Flagged discrepancies: {validation['route_discrepancy_flag'].sum()}")
