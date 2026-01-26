"""
Route exposure metrics from logbook position data.

Computes:
- days_observed: Count of position observations
- frac_days_in_arctic_polygon: Fraction of days in Arctic waters
- frac_days_in_bering: Fraction of days in Bering Sea
- mean_lat, mean_lon: Centroid of voyage track
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import STAGING_DIR, ARCTIC_LAT_THRESHOLD, BERING_SEA_BOUNDS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def point_in_bering_sea(lat: float, lon: float) -> bool:
    """Check if a point is within the Bering Sea bounding box."""
    return (
        BERING_SEA_BOUNDS["lat_min"] <= lat <= BERING_SEA_BOUNDS["lat_max"] and
        BERING_SEA_BOUNDS["lon_min"] <= lon <= BERING_SEA_BOUNDS["lon_max"]
    )


def point_in_arctic(lat: float) -> bool:
    """Check if a point is in Arctic waters (north of Arctic Circle)."""
    return lat >= ARCTIC_LAT_THRESHOLD


def compute_route_exposure(
    logbook_df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Aggregate logbook positions to voyage-level route exposure metrics.
    
    Args:
        logbook_df: Logbook positions DataFrame with voyage_id, lat, lon columns
        output_path: Optional path to save results
        
    Returns:
        DataFrame with one row per voyage_id containing route metrics
    """
    # Filter to records with voyage_id and valid coordinates
    has_data = logbook_df[
        logbook_df["voyage_id"].notna() &
        logbook_df["lat"].notna() &
        logbook_df["lon"].notna()
    ].copy()
    
    if len(has_data) == 0:
        logger.warning("No logbook records with voyage_id and coordinates found")
        return pd.DataFrame(columns=[
            "voyage_id", "days_observed", "frac_days_in_arctic_polygon",
            "frac_days_in_bering", "mean_lat", "mean_lon"
        ])
    
    logger.info(f"Computing route exposure from {len(has_data)} observations")
    
    # Compute geographic flags
    has_data["in_arctic"] = has_data["lat"].apply(point_in_arctic)
    has_data["in_bering"] = has_data.apply(
        lambda r: point_in_bering_sea(r["lat"], r["lon"]),
        axis=1
    )
    
    # Compute latitude bands
    def lat_band(lat):
        if lat >= 66.5:
            return "arctic"
        elif lat >= 45:
            return "north_temperate"
        elif lat >= 23.5:
            return "tropical"
        elif lat >= -23.5:
            return "equatorial"
        elif lat >= -45:
            return "south_tropical"
        else:
            return "antarctic"
    
    has_data["lat_band"] = has_data["lat"].apply(lat_band)
    
    # Aggregate by voyage
    metrics = has_data.groupby("voyage_id").agg(
        days_observed=("lat", "count"),
        arctic_days=("in_arctic", "sum"),
        bering_days=("in_bering", "sum"),
        
        # Position statistics
        mean_lat=("lat", "mean"),
        mean_lon=("lon", "mean"),
        std_lat=("lat", "std"),
        std_lon=("lon", "std"),
        min_lat=("lat", "min"),
        max_lat=("lat", "max"),
        min_lon=("lon", "min"),
        max_lon=("lon", "max"),
        
        # Year coverage
        min_year=("year", "min"),
        max_year=("year", "max"),
    ).reset_index()
    
    # Compute fractions
    metrics["frac_days_in_arctic_polygon"] = metrics["arctic_days"] / metrics["days_observed"]
    metrics["frac_days_in_bering"] = metrics["bering_days"] / metrics["days_observed"]
    
    # Compute latitude range (spread of voyage)
    metrics["lat_range"] = metrics["max_lat"] - metrics["min_lat"]
    metrics["lon_range"] = metrics["max_lon"] - metrics["min_lon"]
    
    # Handle longitude wrapping (for voyages crossing date line)
    metrics["lon_range"] = metrics["lon_range"].apply(
        lambda x: min(x, 360 - x) if x > 180 else x
    )
    
    # Classify voyage type by latitude patterns
    def classify_voyage_region(row):
        mean_lat = row["mean_lat"]
        arctic_frac = row["frac_days_in_arctic_polygon"]
        
        if arctic_frac > 0.3:
            return "arctic_focus"
        elif mean_lat > 30:
            return "north_pacific" if row["mean_lon"] < -100 else "north_atlantic"
        elif mean_lat < -30:
            return "south_pacific" if row["mean_lon"] < -100 else "south_atlantic"
        else:
            return "tropical"
    
    metrics["voyage_region"] = metrics.apply(classify_voyage_region, axis=1)
    
    # Add quality flag
    metrics["route_data_quality"] = np.where(
        metrics["days_observed"] >= 50,
        "good",
        np.where(metrics["days_observed"] >= 20, "partial", "sparse")
    )
    
    logger.info(f"Computed route metrics for {len(metrics)} voyages")
    logger.info(f"  Mean observations per voyage: {metrics['days_observed'].mean():.1f}")
    logger.info(f"  Voyages with Arctic exposure: {(metrics['frac_days_in_arctic_polygon'] > 0).sum()}")
    
    # Save if path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        metrics.to_parquet(output_path, index=False)
        logger.info(f"Saved metrics to {output_path}")
        
        csv_path = output_path.with_suffix(".csv")
        metrics.to_csv(csv_path, index=False)
    
    return metrics


def compute_whaling_ground_exposure(logbook_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute exposure to specific whaling grounds.
    
    Uses known approximate boundaries for major whaling grounds.
    """
    # Define approximate whaling ground boundaries
    WHALING_GROUNDS = {
        "pacific_northwest": {
            "lat_min": 35, "lat_max": 55,
            "lon_min": -150, "lon_max": -115
        },
        "japan_ground": {
            "lat_min": 25, "lat_max": 45,
            "lon_min": 130, "lon_max": 180
        },
        "bering_sea": BERING_SEA_BOUNDS,
        "indian_ocean": {
            "lat_min": -45, "lat_max": 0,
            "lon_min": 40, "lon_max": 100
        },
        "south_atlantic": {
            "lat_min": -60, "lat_max": -20,
            "lon_min": -60, "lon_max": 20
        },
        "atlantic_northeast": {
            "lat_min": 30, "lat_max": 55,
            "lon_min": -50, "lon_max": -20
        },
    }
    
    has_data = logbook_df[
        logbook_df["voyage_id"].notna() &
        logbook_df["lat"].notna() &
        logbook_df["lon"].notna()
    ].copy()
    
    # Check each ground
    for ground_name, bounds in WHALING_GROUNDS.items():
        has_data[f"in_{ground_name}"] = (
            (has_data["lat"] >= bounds["lat_min"]) &
            (has_data["lat"] <= bounds["lat_max"]) &
            (has_data["lon"] >= bounds["lon_min"]) &
            (has_data["lon"] <= bounds["lon_max"])
        )
    
    # Aggregate by voyage
    ground_cols = [f"in_{g}" for g in WHALING_GROUNDS.keys()]
    
    agg_dict = {"lat": "count"}  # Total days
    for col in ground_cols:
        agg_dict[col] = "sum"
    
    ground_metrics = has_data.groupby("voyage_id").agg(agg_dict).reset_index()
    ground_metrics = ground_metrics.rename(columns={"lat": "total_days"})
    
    # Convert to fractions
    for ground_name in WHALING_GROUNDS.keys():
        col = f"in_{ground_name}"
        ground_metrics[f"frac_{ground_name}"] = ground_metrics[col] / ground_metrics["total_days"]
        ground_metrics = ground_metrics.drop(columns=[col])
    
    return ground_metrics


if __name__ == "__main__":
    from parsing.logbook_parser import LogbookParser
    
    parser = LogbookParser()
    logbook_df = parser.parse()
    
    metrics = compute_route_exposure(
        logbook_df,
        output_path=STAGING_DIR / "voyage_routes.parquet"
    )
    
    print("\n=== Route Exposure Sample ===")
    print(metrics.head(10))
