"""
Storm Exposure Feature Engineering.

Computes voyage-level storm encounter metrics by intersecting
daily logbook positions with HURDAT2 hurricane tracks.

Features:
- storm_encounter_count: Storms where vessel was within proximity
- min_storm_distance_nm: Closest approach to any active storm
- hurricane_corridor_days: Actual days in hurricane corridor
- max_storm_wind_kt: Maximum intensity of nearby storms
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.features.logbook_features import haversine_distance, haversine_vectorized
from src.download.weather_downloader import (
    download_hurdat2, 
    HurricaneTrack,
    ATLANTIC_CORRIDOR,
)
from src.config import STAGING_DIR, FINAL_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Proximity threshold for "storm encounter" (nautical miles)
STORM_ENCOUNTER_THRESHOLD_NM = 500

# Danger zone threshold (closer = higher risk)
DANGER_ZONE_THRESHOLD_NM = 200


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class VoyageStormExposure:
    """Storm exposure features for a single voyage."""
    voyage_id: str
    storm_encounter_count: int
    hurricane_encounter_count: int
    min_storm_distance_nm: float
    max_storm_wind_kt: Optional[float]
    days_in_hurricane_corridor: int
    days_near_active_storm: int
    total_storm_exposure_index: float
    storm_ids_encountered: List[str]


# =============================================================================
# Storm Track Processing
# =============================================================================

def build_storm_daily_index(
    storms: List[HurricaneTrack],
) -> Dict[str, List[Dict]]:
    """
    Build an index of storm positions by date for fast lookup.
    
    Returns
    -------
    Dict[str, List[Dict]]
        Mapping from date string to list of storm positions active that day.
    """
    index = {}
    
    for storm in storms:
        for point in storm.track_points:
            date = point['date']
            if date not in index:
                index[date] = []
            
            index[date].append({
                'storm_id': storm.storm_id,
                'name': storm.name,
                'lat': point['lat'],
                'lon': point['lon'],
                'wind_kt': point['wind_kt'],
                'status': point.get('status', 'UNK'),
            })
    
    logger.info(f"Built storm index with {len(index)} dates, {sum(len(v) for v in index.values())} positions")
    return index


def is_in_corridor(lat: float, lon: float) -> bool:
    """Check if position is in hurricane corridor."""
    return (
        ATLANTIC_CORRIDOR["lat_min"] <= lat <= ATLANTIC_CORRIDOR["lat_max"] and
        ATLANTIC_CORRIDOR["lon_min"] <= lon <= ATLANTIC_CORRIDOR["lon_max"]
    )


# =============================================================================
# Voyage-Storm Intersection
# =============================================================================

def compute_voyage_storm_exposure(
    positions: pd.DataFrame,
    voyage_id: str,
    storm_index: Dict[str, List[Dict]],
) -> VoyageStormExposure:
    """
    Compute storm exposure for a single voyage.
    
    Parameters
    ----------
    positions : pd.DataFrame
        Position data with columns: obs_date, lat, lon
    voyage_id : str
        Voyage identifier
    storm_index : Dict
        Storm positions indexed by date
        
    Returns
    -------
    VoyageStormExposure
        Computed storm exposure features
    """
    pos = positions.sort_values("obs_date").reset_index(drop=True)
    n = len(pos)
    
    if n == 0:
        return VoyageStormExposure(
            voyage_id=voyage_id,
            storm_encounter_count=0,
            hurricane_encounter_count=0,
            min_storm_distance_nm=np.nan,
            max_storm_wind_kt=None,
            days_in_hurricane_corridor=0,
            days_near_active_storm=0,
            total_storm_exposure_index=0.0,
            storm_ids_encountered=[],
        )
    
    # Track metrics
    min_distance = float('inf')
    max_wind = 0
    storms_encountered = set()
    hurricanes_encountered = set()
    corridor_days = 0
    near_storm_days = 0
    exposure_index = 0.0
    
    for _, row in pos.iterrows():
        date_str = str(row['obs_date'])[:10]  # Extract YYYY-MM-DD
        lat = row['lat']
        lon = row['lon']
        
        if pd.isna(lat) or pd.isna(lon):
            continue
        
        # Check corridor
        if is_in_corridor(lat, lon):
            corridor_days += 1
        
        # Check storms active on this date
        if date_str in storm_index:
            day_min_distance = float('inf')
            
            for storm_point in storm_index[date_str]:
                storm_lat = storm_point['lat']
                storm_lon = storm_point['lon']
                wind = storm_point['wind_kt']
                storm_id = storm_point['storm_id']
                
                # Calculate distance to storm
                distance = haversine_distance(lat, lon, storm_lat, storm_lon)
                
                if distance < day_min_distance:
                    day_min_distance = distance
                
                if distance < min_distance:
                    min_distance = distance
                
                # Check encounter thresholds
                if distance <= STORM_ENCOUNTER_THRESHOLD_NM:
                    storms_encountered.add(storm_id)
                    
                    if wind and wind >= 64:  # Hurricane intensity
                        hurricanes_encountered.add(storm_id)
                    
                    if wind and wind > max_wind:
                        max_wind = wind
                    
                    # Exposure index: inverse distance weighted by intensity
                    if wind:
                        exposure_index += (wind / max(distance, 1)) * 0.01
            
            if day_min_distance <= DANGER_ZONE_THRESHOLD_NM:
                near_storm_days += 1
    
    return VoyageStormExposure(
        voyage_id=voyage_id,
        storm_encounter_count=len(storms_encountered),
        hurricane_encounter_count=len(hurricanes_encountered),
        min_storm_distance_nm=min_distance if min_distance < float('inf') else np.nan,
        max_storm_wind_kt=max_wind if max_wind > 0 else None,
        days_in_hurricane_corridor=corridor_days,
        days_near_active_storm=near_storm_days,
        total_storm_exposure_index=exposure_index,
        storm_ids_encountered=list(storms_encountered),
    )


# =============================================================================
# Batch Processing
# =============================================================================

def compute_all_storm_exposure(
    positions: Optional[pd.DataFrame] = None,
    storms: Optional[List[HurricaneTrack]] = None,
    progress_interval: int = 500,
) -> pd.DataFrame:
    """
    Compute storm exposure features for all voyages.
    
    Parameters
    ----------
    positions : pd.DataFrame, optional
        Position data. If None, loads from staging.
    storms : List[HurricaneTrack], optional
        Hurricane tracks. If None, downloads from HURDAT2.
    progress_interval : int
        Print progress every N voyages.
        
    Returns
    -------
    pd.DataFrame
        Voyage-level storm exposure features.
    """
    # Load positions if needed
    if positions is None:
        from src.features.logbook_features import load_logbook_positions
        positions = load_logbook_positions()
    
    # Load/download storms if needed
    if storms is None:
        logger.info("Downloading HURDAT2 data...")
        storms = download_hurdat2()
    
    # Build storm index
    storm_index = build_storm_daily_index(storms)
    
    logger.info(f"Computing storm exposure for {positions['voyage_id'].nunique()} voyages...")
    
    # Group by voyage
    voyage_groups = positions.groupby("voyage_id")
    
    features_list = []
    for i, (voyage_id, group) in enumerate(voyage_groups):
        if i > 0 and i % progress_interval == 0:
            logger.info(f"  Processed {i:,} voyages...")
        
        features = compute_voyage_storm_exposure(group, voyage_id, storm_index)
        features_list.append(features)
    
    # Convert to DataFrame
    df = pd.DataFrame([
        {
            'voyage_id': f.voyage_id,
            'storm_encounter_count': f.storm_encounter_count,
            'hurricane_encounter_count': f.hurricane_encounter_count,
            'min_storm_distance_nm': f.min_storm_distance_nm,
            'max_storm_wind_kt': f.max_storm_wind_kt,
            'days_in_hurricane_corridor': f.days_in_hurricane_corridor,
            'days_near_active_storm': f.days_near_active_storm,
            'total_storm_exposure_index': f.total_storm_exposure_index,
            'n_storms_encountered': len(f.storm_ids_encountered),
        }
        for f in features_list
    ])
    
    # Summary
    n_with_encounters = (df['storm_encounter_count'] > 0).sum()
    n_with_hurricanes = (df['hurricane_encounter_count'] > 0).sum()
    
    logger.info(f"Computed storm exposure for {len(df):,} voyages")
    logger.info(f"  Voyages with storm encounters: {n_with_encounters:,}")
    logger.info(f"  Voyages with hurricane encounters: {n_with_hurricanes:,}")
    logger.info(f"  Mean corridor days: {df['days_in_hurricane_corridor'].mean():.1f}")
    
    return df


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compute voyage storm exposure from logbook positions"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save computed features"
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Only process N voyages (for testing)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("STORM EXPOSURE FEATURE ENGINEERING")
    print("=" * 60)
    
    # Load positions
    from src.features.logbook_features import load_logbook_positions
    positions = load_logbook_positions()
    
    if args.sample:
        sample_voyages = positions['voyage_id'].unique()[:args.sample]
        positions = positions[positions['voyage_id'].isin(sample_voyages)]
        print(f"\n[Sample mode: {args.sample} voyages]")
    
    print(f"\nLoaded {len(positions):,} positions, {positions['voyage_id'].nunique()} voyages")
    
    # Compute features
    features = compute_all_storm_exposure(positions)
    
    # Show sample
    print("\n" + "-" * 60)
    print("SAMPLE OUTPUT")
    print("-" * 60)
    print(features.head(10).to_string())
    
    # Summary stats
    print("\n" + "-" * 60)
    print("SUMMARY STATISTICS")
    print("-" * 60)
    for col in features.columns:
        if col != 'voyage_id' and features[col].dtype in ['float64', 'int64']:
            print(f"{col}: mean={features[col].mean():.2f}, max={features[col].max():.0f}")
    
    if args.save:
        output_path = FINAL_DIR / "voyage_storm_exposure.parquet"
        features.to_parquet(output_path, index=False)
        features.to_csv(output_path.with_suffix('.csv'), index=False)
        print(f"\nSaved to {output_path}")
    
    print("\nDone!")
