"""
Logbook Feature Engineering Module.

Computes voyage-level features from daily logbook position data:
- Route metrics (efficiency, distance, dwell time)
- Behavioral metrics (ground switching, encounter patterns)
- Arrival timing scores

These features enable skill isolation and agent strategy analyses.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import STAGING_DIR, FINAL_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Earth radius in nautical miles
EARTH_RADIUS_NM = 3440.065

# Whaling ground definitions (approximate bounding boxes)
# Based on historical whaling ground classifications
WHALING_GROUNDS = {
    "atlantic_north": {"lat_min": 30, "lat_max": 60, "lon_min": -80, "lon_max": -10},
    "atlantic_south": {"lat_min": -60, "lat_max": -10, "lon_min": -70, "lon_max": 20},
    "pacific_north": {"lat_min": 30, "lat_max": 65, "lon_min": 120, "lon_max": -120},
    "pacific_south": {"lat_min": -60, "lat_max": -10, "lon_min": -180, "lon_max": -70},
    "indian": {"lat_min": -60, "lat_max": 10, "lon_min": 20, "lon_max": 120},
    "arctic": {"lat_min": 60, "lat_max": 90, "lon_min": -180, "lon_max": 180},
    "antarctic": {"lat_min": -90, "lat_max": -60, "lon_min": -180, "lon_max": 180},
}

# Optimal months for whaling grounds (peak whale presence)
GROUND_SEASONS = {
    "atlantic_north": [5, 6, 7, 8, 9],  # May-Sept
    "atlantic_south": [11, 12, 1, 2, 3],  # Nov-Mar (Southern summer)
    "pacific_north": [4, 5, 6, 7, 8, 9],  # Apr-Sept
    "pacific_south": [11, 12, 1, 2, 3],  # Nov-Mar
    "indian": [10, 11, 12, 1, 2],  # Oct-Feb
    "arctic": [6, 7, 8, 9],  # Jun-Sept (ice-free)
    "antarctic": [12, 1, 2, 3],  # Dec-Mar (Southern summer)
}


# =============================================================================
# Haversine Distance Calculation
# =============================================================================

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great-circle distance between two points in nautical miles.
    
    Parameters
    ----------
    lat1, lon1 : float
        First point coordinates in decimal degrees
    lat2, lon2 : float
        Second point coordinates in decimal degrees
        
    Returns
    -------
    float
        Distance in nautical miles
    """
    # Convert to radians
    lat1_r = np.radians(lat1)
    lat2_r = np.radians(lat2)
    lon1_r = np.radians(lon1)
    lon2_r = np.radians(lon2)
    
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return EARTH_RADIUS_NM * c


def haversine_vectorized(
    lat1: np.ndarray, lon1: np.ndarray,
    lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Vectorized haversine distance calculation."""
    lat1_r = np.radians(lat1)
    lat2_r = np.radians(lat2)
    lon1_r = np.radians(lon1)
    lon2_r = np.radians(lon2)
    
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return EARTH_RADIUS_NM * c


# =============================================================================
# Ground Classification
# =============================================================================

def classify_ground(lat: float, lon: float) -> Optional[str]:
    """
    Classify a position into a whaling ground.
    
    Parameters
    ----------
    lat, lon : float
        Position coordinates
        
    Returns
    -------
    str or None
        Ground name or None if not in any defined ground
    """
    for ground_name, bounds in WHALING_GROUNDS.items():
        lat_in = bounds["lat_min"] <= lat <= bounds["lat_max"]
        
        # Handle longitude wrap-around for Pacific
        if bounds["lon_min"] > bounds["lon_max"]:  # Crosses dateline
            lon_in = lon >= bounds["lon_min"] or lon <= bounds["lon_max"]
        else:
            lon_in = bounds["lon_min"] <= lon <= bounds["lon_max"]
        
        if lat_in and lon_in:
            return ground_name
    
    return None


def is_optimal_season(ground: str, month: int) -> bool:
    """Check if month is optimal for whaling in given ground."""
    if ground not in GROUND_SEASONS:
        return False
    return month in GROUND_SEASONS[ground]


# =============================================================================
# Route Feature Computation
# =============================================================================

@dataclass
class VoyageRouteFeatures:
    """Computed route features for a single voyage."""
    voyage_id: str
    n_positions: int
    total_distance_nm: float
    beeline_distance_nm: float
    route_efficiency: float
    avg_daily_distance_nm: float
    max_extent_lat: float
    max_extent_lon: float
    furthest_from_start_nm: float
    ground_dwell_days: int
    transit_days: int
    n_grounds_visited: int
    ground_switching_count: int
    primary_ground: Optional[str]
    arrival_timing_score: float


def compute_route_features(
    positions: pd.DataFrame,
    voyage_id: str,
) -> VoyageRouteFeatures:
    """
    Compute route features from daily position data.
    
    Parameters
    ----------
    positions : pd.DataFrame
        Position data with columns: obs_date, lat, lon
    voyage_id : str
        Voyage identifier
        
    Returns
    -------
    VoyageRouteFeatures
        Computed route features
    """
    # Sort by date
    pos = positions.sort_values("obs_date").reset_index(drop=True)
    n = len(pos)
    
    if n < 2:
        return VoyageRouteFeatures(
            voyage_id=voyage_id,
            n_positions=n,
            total_distance_nm=0.0,
            beeline_distance_nm=0.0,
            route_efficiency=np.nan,
            avg_daily_distance_nm=0.0,
            max_extent_lat=pos['lat'].iloc[0] if n > 0 else np.nan,
            max_extent_lon=pos['lon'].iloc[0] if n > 0 else np.nan,
            furthest_from_start_nm=0.0,
            ground_dwell_days=0,
            transit_days=0,
            n_grounds_visited=0,
            ground_switching_count=0,
            primary_ground=None,
            arrival_timing_score=np.nan,
        )
    
    # Calculate segment distances
    lats = pos['lat'].values
    lons = pos['lon'].values
    
    segment_distances = haversine_vectorized(
        lats[:-1], lons[:-1],
        lats[1:], lons[1:]
    )
    
    total_distance = np.nansum(segment_distances)
    
    # Beeline distance (first to last position)
    beeline_distance = haversine_distance(
        lats[0], lons[0],
        lats[-1], lons[-1]
    )
    
    # Route efficiency
    route_efficiency = beeline_distance / total_distance if total_distance > 0 else np.nan
    
    # Average daily distance
    avg_daily_distance = total_distance / (n - 1) if n > 1 else 0.0
    
    # Max extent
    max_lat = np.abs(lats).max()
    max_lon_idx = np.argmax(np.abs(lons))
    max_lon = lons[max_lon_idx]
    
    # Furthest from start
    distances_from_start = haversine_vectorized(
        np.full(n, lats[0]), np.full(n, lons[0]),
        lats, lons
    )
    furthest_from_start = np.nanmax(distances_from_start)
    
    # Classify grounds
    pos['ground'] = [classify_ground(lat, lon) for lat, lon in zip(lats, lons)]
    
    # Ground dwell vs transit
    ground_positions = pos['ground'].notna()
    ground_dwell_days = ground_positions.sum()
    transit_days = (~ground_positions).sum()
    
    # Number of unique grounds
    grounds_visited = pos['ground'].dropna().unique()
    n_grounds = len(grounds_visited)
    
    # Ground switching count
    ground_changes = (pos['ground'] != pos['ground'].shift()).sum() - 1
    ground_switching_count = max(0, ground_changes)
    
    # Primary ground (most days spent)
    if n_grounds > 0:
        ground_days = pos[pos['ground'].notna()].groupby('ground').size()
        primary_ground = ground_days.idxmax()
    else:
        primary_ground = None
    
    # Arrival timing score
    # Check if arrived at each ground during optimal season
    if 'obs_date' in pos.columns:
        pos['month'] = pd.to_datetime(pos['obs_date']).dt.month
        pos['optimal'] = [
            is_optimal_season(g, m) if pd.notna(g) else np.nan
            for g, m in zip(pos['ground'], pos['month'])
        ]
        valid_optimal = pos['optimal'].dropna()
        arrival_timing_score = valid_optimal.mean() if len(valid_optimal) > 0 else np.nan
    else:
        arrival_timing_score = np.nan
    
    return VoyageRouteFeatures(
        voyage_id=voyage_id,
        n_positions=n,
        total_distance_nm=total_distance,
        beeline_distance_nm=beeline_distance,
        route_efficiency=route_efficiency,
        avg_daily_distance_nm=avg_daily_distance,
        max_extent_lat=max_lat,
        max_extent_lon=max_lon,
        furthest_from_start_nm=furthest_from_start,
        ground_dwell_days=ground_dwell_days,
        transit_days=transit_days,
        n_grounds_visited=n_grounds,
        ground_switching_count=ground_switching_count,
        primary_ground=primary_ground,
        arrival_timing_score=arrival_timing_score,
    )


# =============================================================================
# Batch Processing
# =============================================================================

def load_logbook_positions() -> pd.DataFrame:
    """Load parsed logbook positions from staging."""
    parquet_path = STAGING_DIR / "logbook_positions.parquet"
    csv_path = STAGING_DIR / "logbook_positions.csv"
    
    if parquet_path.exists():
        logger.info(f"Loading logbook positions from {parquet_path}")
        return pd.read_parquet(parquet_path)
    elif csv_path.exists():
        logger.info(f"Loading logbook positions from {csv_path}")
        return pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(
            f"Logbook positions not found at {parquet_path} or {csv_path}"
        )


def compute_all_route_features(
    positions: Optional[pd.DataFrame] = None,
    progress_interval: int = 500,
) -> pd.DataFrame:
    """
    Compute route features for all voyages in logbook data.
    
    Parameters
    ----------
    positions : pd.DataFrame, optional
        Position data. If None, loads from staging.
    progress_interval : int
        Print progress every N voyages.
        
    Returns
    -------
    pd.DataFrame
        Voyage-level route features.
    """
    if positions is None:
        positions = load_logbook_positions()
    
    logger.info(f"Computing route features for {positions['voyage_id'].nunique()} voyages...")
    
    # Group by voyage
    voyage_groups = positions.groupby("voyage_id")
    
    features_list = []
    for i, (voyage_id, group) in enumerate(voyage_groups):
        if i > 0 and i % progress_interval == 0:
            logger.info(f"  Processed {i:,} voyages...")
        
        features = compute_route_features(group, voyage_id)
        features_list.append(features)
    
    # Convert to DataFrame
    df = pd.DataFrame([vars(f) for f in features_list])
    
    logger.info(f"Computed route features for {len(df):,} voyages")
    logger.info(f"  Mean route efficiency: {df['route_efficiency'].mean():.3f}")
    logger.info(f"  Mean total distance: {df['total_distance_nm'].mean():,.0f} nm")
    logger.info(f"  Mean grounds visited: {df['n_grounds_visited'].mean():.1f}")
    
    return df


def save_route_features(
    features: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> Path:
    """Save computed features to final directory."""
    if output_path is None:
        output_path = FINAL_DIR / "voyage_logbook_features.parquet"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    features.to_parquet(output_path, index=False)
    features.to_csv(output_path.with_suffix('.csv'), index=False)
    
    logger.info(f"Saved {len(features):,} voyage features to {output_path}")
    return output_path


# =============================================================================
# Summary Statistics
# =============================================================================

def get_feature_summary(features: pd.DataFrame) -> Dict[str, Any]:
    """Generate summary statistics for computed features."""
    summary = {
        "n_voyages": len(features),
        "mean_positions_per_voyage": features['n_positions'].mean(),
        "mean_route_efficiency": features['route_efficiency'].mean(),
        "median_route_efficiency": features['route_efficiency'].median(),
        "mean_total_distance_nm": features['total_distance_nm'].mean(),
        "mean_grounds_visited": features['n_grounds_visited'].mean(),
        "mean_ground_dwell_days": features['ground_dwell_days'].mean(),
        "mean_transit_days": features['transit_days'].mean(),
        "mean_arrival_timing_score": features['arrival_timing_score'].mean(),
        "primary_ground_distribution": features['primary_ground'].value_counts().to_dict(),
    }
    return summary


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compute voyage route features from logbook data"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save computed features to final directory"
    )
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Only print summary statistics"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("LOGBOOK FEATURE ENGINEERING")
    print("=" * 60)
    
    # Load and compute
    positions = load_logbook_positions()
    print(f"\nLoaded {len(positions):,} position records")
    print(f"Unique voyages: {positions['voyage_id'].nunique():,}")
    
    if args.summary_only:
        print("\n[Summary mode - computing on sample]")
        sample_voyages = positions['voyage_id'].unique()[:100]
        positions = positions[positions['voyage_id'].isin(sample_voyages)]
    
    features = compute_all_route_features(positions)
    
    # Summary
    print("\n" + "-" * 60)
    print("FEATURE SUMMARY")
    print("-" * 60)
    
    summary = get_feature_summary(features)
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in list(value.items())[:5]:
                print(f"  {k}: {v}")
        elif isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    
    # Save
    if args.save:
        save_route_features(features)
    
    print("\nDone!")
