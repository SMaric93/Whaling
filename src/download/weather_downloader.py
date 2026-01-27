"""
Weather Data Downloader and Integrator.

Downloads and parses historical climate indices for linking with whaling voyage data:
- NAO (North Atlantic Oscillation) Index: 1865-present
- HURDAT2 (Atlantic Hurricane Database): 1851-present

These provide environmental controls for the AKM decomposition to separate
maritime skill from "luck of the weather."
"""

import io
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests


# =============================================================================
# Configuration
# =============================================================================

# Compute paths directly to work as standalone script
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "final"
WEATHER_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "weather"
WEATHER_FINAL_DIR = DATA_DIR  # Merge into final data directory

NAO_URL = "https://climatedataguide.ucar.edu/sites/default/files/2022-03/nao_station_annual.txt"
HURDAT_URL = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2023-051124.txt"

# Pacific climate indices for Pacific whaling ground analysis
PDO_URL = "https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat"
# Use HadISST-based Niño 3.4 from NOAA PSL (1870-present, more reliable)
NINO34_URL = "https://psl.noaa.gov/data/correlation/nina34.anom.data"

# Atlantic hurricane corridor relevant to New England whaling routes
# Bounding box for "Cape Horn to New Bedford" corridor exposure
ATLANTIC_CORRIDOR = {
    "lat_min": 0.0,    # Equator
    "lat_max": 50.0,   # North Atlantic
    "lon_min": -80.0,  # Eastern seaboard
    "lon_max": -30.0,  # Mid-Atlantic
}


# =============================================================================
# NAO Index Download and Parse
# =============================================================================

def download_nao_index() -> pd.DataFrame:
    """
    Download and parse the Hurrell Station-Based Annual NAO Index.
    
    The NAO affects Atlantic storm tracks and weather patterns:
    - Positive NAO: Calmer Atlantic, milder New England winters
    - Negative NAO: Stormier Atlantic, harsher conditions
    
    Returns
    -------
    pd.DataFrame
        Columns: [year, nao_index]
    """
    print("Downloading NAO Index...")
    
    try:
        response = requests.get(NAO_URL, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"  Warning: Could not download NAO data: {e}")
        return pd.DataFrame(columns=["year", "nao_index"])
    
    # Parse the text data
    lines = response.text.strip().split("\n")
    records = []
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith("Hurrell"):
            continue
        
        parts = line.split()
        if len(parts) >= 2:
            try:
                year = int(parts[0])
                nao = float(parts[1])
                records.append({"year": year, "nao_index": nao})
            except ValueError:
                continue
    
    df = pd.DataFrame(records)
    
    if len(df) > 0:
        print(f"  Downloaded NAO index: {df['year'].min()}-{df['year'].max()} ({len(df)} years)")
    
    return df


def categorize_nao(nao_value: float) -> str:
    """Categorize NAO into phases for easier interpretation."""
    if nao_value >= 1.5:
        return "strong_positive"
    elif nao_value >= 0.5:
        return "positive"
    elif nao_value <= -1.5:
        return "strong_negative"
    elif nao_value <= -0.5:
        return "negative"
    else:
        return "neutral"


# =============================================================================
# PDO Index Download and Parse
# =============================================================================

def download_pdo_index() -> pd.DataFrame:
    """
    Download and parse the Pacific Decadal Oscillation Index.
    
    The PDO affects multi-decadal Pacific climate patterns:
    - Positive (warm) PDO: Warmer eastern Pacific, affects whale prey distribution
    - Negative (cool) PDO: Cooler eastern Pacific, different migration patterns
    
    Returns
    -------
    pd.DataFrame
        Columns: [year, pdo_annual, pdo_phase]
    """
    print("Downloading PDO Index...")
    
    try:
        response = requests.get(PDO_URL, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"  Warning: Could not download PDO data: {e}")
        return pd.DataFrame(columns=["year", "pdo_annual", "pdo_phase"])
    
    # Parse the fixed-width text data
    # Format: YEAR JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC
    lines = response.text.strip().split("\n")
    records = []
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith("Year") or line.startswith("#"):
            continue
        
        parts = line.split()
        if len(parts) >= 13:
            try:
                year = int(parts[0])
                # Parse monthly values, handling missing data (-99.99)
                monthly_vals = []
                for val in parts[1:13]:
                    v = float(val)
                    if v > -90:  # Valid value
                        monthly_vals.append(v)
                
                if monthly_vals:
                    annual_mean = sum(monthly_vals) / len(monthly_vals)
                    records.append({
                        "year": year,
                        "pdo_annual": round(annual_mean, 3),
                    })
            except ValueError:
                continue
    
    df = pd.DataFrame(records)
    
    if len(df) > 0:
        df["pdo_phase"] = df["pdo_annual"].apply(categorize_pdo)
        print(f"  Downloaded PDO index: {df['year'].min()}-{df['year'].max()} ({len(df)} years)")
    
    return df


def categorize_pdo(pdo_value: float) -> str:
    """Categorize PDO into phases for easier interpretation."""
    if pdo_value >= 0.5:
        return "warm"
    elif pdo_value <= -0.5:
        return "cool"
    else:
        return "neutral"


# =============================================================================
# ENSO / Niño 3.4 Index Download and Parse
# =============================================================================

def download_enso_index() -> pd.DataFrame:
    """
    Download and parse the Niño 3.4 SST anomaly index (proxy for ENSO).
    
    ENSO affects inter-annual Pacific climate:
    - El Niño (positive): Warmer eastern Pacific, altered whale distribution
    - La Niña (negative): Cooler eastern Pacific, different migration patterns
    
    Returns
    -------
    pd.DataFrame
        Columns: [year, enso_annual, enso_phase]
    """
    print("Downloading ENSO (Niño 3.4) Index...")
    
    try:
        response = requests.get(NINO34_URL, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"  Warning: Could not download ENSO data: {e}")
        return pd.DataFrame(columns=["year", "enso_annual", "enso_phase"])
    
    # Parse the data - format similar to PDO
    # YEAR JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC
    lines = response.text.strip().split("\n")
    records = []
    
    for line in lines:
        line = line.strip()
        if not line or not line[0].isdigit():
            continue
        
        parts = line.split()
        if len(parts) >= 13:
            try:
                year = int(parts[0])
                # Skip header/footer years outside range
                if year < 1800 or year > 2100:
                    continue
                    
                # Parse monthly values, handling missing data (-99.99)
                monthly_vals = []
                for val in parts[1:13]:
                    v = float(val)
                    if v > -90:  # Valid value
                        monthly_vals.append(v)
                
                if monthly_vals:
                    annual_mean = sum(monthly_vals) / len(monthly_vals)
                    records.append({
                        "year": year,
                        "enso_annual": round(annual_mean, 3),
                    })
            except ValueError:
                continue
    
    df = pd.DataFrame(records)
    
    if len(df) > 0:
        df["enso_phase"] = df["enso_annual"].apply(categorize_enso)
        print(f"  Downloaded ENSO index: {df['year'].min()}-{df['year'].max()} ({len(df)} years)")
    
    return df


def categorize_enso(enso_value: float) -> str:
    """Categorize ENSO into phases for easier interpretation."""
    if enso_value >= 0.5:
        return "el_nino"
    elif enso_value <= -0.5:
        return "la_nina"
    else:
        return "neutral"


# =============================================================================
# HURDAT2 Hurricane Data Download and Parse
# =============================================================================

@dataclass
class HurricaneTrack:
    """Single hurricane track with metadata."""
    storm_id: str
    name: str
    n_entries: int
    track_points: List[Dict]


def download_hurdat2() -> List[HurricaneTrack]:
    """
    Download and parse HURDAT2 Atlantic hurricane database.
    
    Returns
    -------
    List[HurricaneTrack]
        List of hurricane tracks with positions and intensities.
    """
    print("Downloading HURDAT2 hurricane database...")
    
    try:
        response = requests.get(HURDAT_URL, timeout=60)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"  Warning: Could not download HURDAT2 data: {e}")
        return []
    
    lines = response.text.strip().split("\n")
    storms = []
    current_storm = None
    current_points = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        parts = [p.strip() for p in line.split(",")]
        
        # Header line: AL012020, ARTHUR, 22,
        if len(parts) >= 3 and parts[0].startswith("AL"):
            # Save previous storm
            if current_storm is not None:
                storms.append(HurricaneTrack(
                    storm_id=current_storm["id"],
                    name=current_storm["name"],
                    n_entries=current_storm["n"],
                    track_points=current_points
                ))
            
            current_storm = {
                "id": parts[0],
                "name": parts[1].strip() if len(parts) > 1 else "UNNAMED",
                "n": int(parts[2]) if len(parts) > 2 and parts[2].strip().isdigit() else 0
            }
            current_points = []
        
        # Track point line: 20200516, 1800, , TS, 26.9N, 79.0W, 40, ...
        elif len(parts) >= 7 and len(parts[0]) == 8 and parts[0].isdigit():
            try:
                date_str = parts[0]
                time_str = parts[1].strip().zfill(4)
                status = parts[3].strip()
                
                # Parse latitude
                lat_str = parts[4].strip()
                lat = float(lat_str[:-1])
                if lat_str.endswith("S"):
                    lat = -lat
                
                # Parse longitude
                lon_str = parts[5].strip()
                lon = float(lon_str[:-1])
                if lon_str.endswith("W"):
                    lon = -lon
                
                # Wind speed (knots)
                wind = int(parts[6]) if parts[6].strip().lstrip("-").isdigit() else None
                
                current_points.append({
                    "date": f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}",
                    "time": f"{time_str[:2]}:{time_str[2:]}",
                    "status": status,
                    "lat": lat,
                    "lon": lon,
                    "wind_kt": wind,
                    "year": int(date_str[:4]),
                    "month": int(date_str[4:6]),
                })
            except (ValueError, IndexError):
                continue
    
    # Save last storm
    if current_storm is not None:
        storms.append(HurricaneTrack(
            storm_id=current_storm["id"],
            name=current_storm["name"],
            n_entries=current_storm["n"],
            track_points=current_points
        ))
    
    print(f"  Parsed {len(storms)} hurricanes")
    
    return storms


def compute_annual_hurricane_metrics(storms: List[HurricaneTrack]) -> pd.DataFrame:
    """
    Compute annual hurricane metrics from HURDAT2 tracks.
    
    Metrics:
    - n_storms: Total named storms
    - n_hurricanes: Storms reaching hurricane intensity (>= 64 kt)
    - n_major: Major hurricanes (>= 96 kt, Category 3+)
    - ace_index: Accumulated Cyclone Energy approximation
    - corridor_exposure: Storms passing through Atlantic corridor
    
    Returns
    -------
    pd.DataFrame
        Annual hurricane metrics.
    """
    print("Computing annual hurricane metrics...")
    
    annual_data = {}
    
    for storm in storms:
        if not storm.track_points:
            continue
        
        year = storm.track_points[0]["year"]
        if year not in annual_data:
            annual_data[year] = {
                "year": year,
                "n_storms": 0,
                "n_hurricanes": 0,
                "n_major": 0,
                "ace_approx": 0.0,
                "corridor_storms": 0,
                "corridor_hurricane_days": 0,
            }
        
        annual_data[year]["n_storms"] += 1
        
        max_wind = max((p["wind_kt"] for p in storm.track_points if p["wind_kt"]), default=0)
        if max_wind >= 64:
            annual_data[year]["n_hurricanes"] += 1
        if max_wind >= 96:
            annual_data[year]["n_major"] += 1
        
        # ACE approximation (sum of v^2 for all 6-hourly points)
        for point in storm.track_points:
            if point["wind_kt"] and point["wind_kt"] >= 34:  # Tropical storm+
                annual_data[year]["ace_approx"] += (point["wind_kt"] ** 2) / 10000
        
        # Check corridor exposure
        in_corridor = False
        corridor_days = set()
        for point in storm.track_points:
            if (ATLANTIC_CORRIDOR["lat_min"] <= point["lat"] <= ATLANTIC_CORRIDOR["lat_max"] and
                ATLANTIC_CORRIDOR["lon_min"] <= point["lon"] <= ATLANTIC_CORRIDOR["lon_max"]):
                in_corridor = True
                if point["wind_kt"] and point["wind_kt"] >= 64:
                    corridor_days.add(point["date"])
        
        if in_corridor:
            annual_data[year]["corridor_storms"] += 1
            annual_data[year]["corridor_hurricane_days"] += len(corridor_days)
    
    df = pd.DataFrame(list(annual_data.values()))
    df = df.sort_values("year").reset_index(drop=True)
    
    # Compute standardized metrics
    if len(df) > 10:
        df["ace_zscore"] = (df["ace_approx"] - df["ace_approx"].mean()) / df["ace_approx"].std()
        df["storm_intensity"] = df["n_major"] / df["n_storms"].replace(0, 1)
    else:
        df["ace_zscore"] = 0.0
        df["storm_intensity"] = 0.0
    
    print(f"  Computed metrics for {len(df)} years ({df['year'].min()}-{df['year'].max()})")
    
    return df


# =============================================================================
# Voyage-Level Hurricane Exposure
# =============================================================================

def compute_voyage_hurricane_exposure(
    voyages: pd.DataFrame,
    storms: List[HurricaneTrack],
) -> pd.DataFrame:
    """
    Compute voyage-level hurricane exposure metrics.
    
    For each voyage, counts storms active during the voyage window
    that passed through the Atlantic corridor.
    
    Parameters
    ----------
    voyages : pd.DataFrame
        Voyage data with year_out, month_out (or departure_date).
    storms : List[HurricaneTrack]
        Parsed hurricane tracks.
        
    Returns
    -------
    pd.DataFrame
        Voyage-level exposure metrics.
    """
    print("Computing voyage-level hurricane exposure...")
    
    # Build storm date index for fast lookup
    storm_by_year_month = {}
    for storm in storms:
        for point in storm.track_points:
            key = (point["year"], point["month"])
            if key not in storm_by_year_month:
                storm_by_year_month[key] = []
            
            # Check if in corridor and significant intensity
            in_corridor = (
                ATLANTIC_CORRIDOR["lat_min"] <= point["lat"] <= ATLANTIC_CORRIDOR["lat_max"] and
                ATLANTIC_CORRIDOR["lon_min"] <= point["lon"] <= ATLANTIC_CORRIDOR["lon_max"]
            )
            if in_corridor and point["wind_kt"] and point["wind_kt"] >= 34:
                storm_by_year_month[key].append({
                    "storm_id": storm.storm_id,
                    "wind_kt": point["wind_kt"],
                    "is_hurricane": point["wind_kt"] >= 64,
                })
    
    # For each voyage, compute exposure during likely transit months
    # Assume voyages depart and have highest Atlantic exposure in first 2-3 months
    exposure_records = []
    
    for _, voyage in voyages.iterrows():
        voyage_id = voyage.get("voyage_id")
        year_out = voyage.get("year_out")
        
        if pd.isna(year_out):
            exposure_records.append({
                "voyage_id": voyage_id,
                "hurricane_exposure_count": None,
                "hurricane_corridor_intensity": None,
            })
            continue
        
        year_out = int(year_out)
        
        # Check months typical for Atlantic transit (assume departure in summer/fall)
        # June-November is hurricane season
        exposure_count = 0
        max_intensity = 0
        unique_storms = set()
        
        for month in range(6, 12):  # June-November
            key = (year_out, month)
            if key in storm_by_year_month:
                for storm_point in storm_by_year_month[key]:
                    unique_storms.add(storm_point["storm_id"])
                    if storm_point["wind_kt"] and storm_point["wind_kt"] > max_intensity:
                        max_intensity = storm_point["wind_kt"]
        
        exposure_records.append({
            "voyage_id": voyage_id,
            "hurricane_exposure_count": len(unique_storms),
            "hurricane_corridor_intensity": max_intensity if max_intensity > 0 else None,
        })
    
    df = pd.DataFrame(exposure_records)
    
    n_with_exposure = (df["hurricane_exposure_count"] > 0).sum()
    print(f"  {n_with_exposure:,} voyages with hurricane corridor exposure")
    
    return df


# =============================================================================
# Main Integration Function
# =============================================================================

def download_and_integrate_weather(
    voyages_path: Optional[Path] = None,
    save_raw: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download weather data and integrate with voyage data.
    
    Parameters
    ----------
    voyages_path : Path, optional
        Path to voyage parquet file. Defaults to analysis_voyage.parquet.
    save_raw : bool
        Whether to save raw weather data files.
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (annual_weather, voyage_weather) DataFrames ready for merge.
    """
    print("\n" + "=" * 60)
    print("WEATHER DATA INTEGRATION")
    print("=" * 60)
    
    # Create directories
    WEATHER_RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download NAO
    nao_df = download_nao_index()
    nao_df["nao_phase"] = nao_df["nao_index"].apply(categorize_nao)
    
    # Download Pacific climate indices (PDO and ENSO)
    pdo_df = download_pdo_index()
    enso_df = download_enso_index()
    
    # Download and process HURDAT2
    storms = download_hurdat2()
    hurricane_annual = compute_annual_hurricane_metrics(storms)
    
    # Merge annual data
    annual_weather = nao_df.merge(hurricane_annual, on="year", how="outer")
    
    # Add Pacific indices
    if len(pdo_df) > 0:
        annual_weather = annual_weather.merge(pdo_df, on="year", how="outer")
    if len(enso_df) > 0:
        annual_weather = annual_weather.merge(enso_df, on="year", how="outer")
    
    annual_weather = annual_weather.sort_values("year").reset_index(drop=True)
    
    # Filter to whaling era (1800-1920)
    annual_weather = annual_weather[
        (annual_weather["year"] >= 1800) & (annual_weather["year"] <= 1920)
    ].copy()
    
    print(f"\nAnnual weather data: {len(annual_weather)} years")
    print(f"  NAO coverage: {annual_weather['nao_index'].notna().sum()} years")
    print(f"  Hurricane coverage: {annual_weather['n_storms'].notna().sum()} years")
    print(f"  PDO coverage: {annual_weather['pdo_annual'].notna().sum() if 'pdo_annual' in annual_weather.columns else 0} years")
    print(f"  ENSO coverage: {annual_weather['enso_annual'].notna().sum() if 'enso_annual' in annual_weather.columns else 0} years")
    
    # Voyage-level integration
    if voyages_path is None:
        voyages_path = DATA_DIR / "analysis_voyage.parquet"
    
    voyage_weather = pd.DataFrame()
    if voyages_path.exists():
        print(f"\nLoading voyages from {voyages_path}...")
        voyages = pd.read_parquet(voyages_path)
        
        # Compute voyage-level hurricane exposure
        voyage_exposure = compute_voyage_hurricane_exposure(voyages, storms)
        
        # Build list of annual columns to merge (only those that exist)
        annual_cols = ["year", "nao_index", "nao_phase", "n_storms", 
                       "n_hurricanes", "ace_approx", "corridor_storms"]
        # Add Pacific indices if available
        if "pdo_annual" in annual_weather.columns:
            annual_cols.extend(["pdo_annual", "pdo_phase"])
        if "enso_annual" in annual_weather.columns:
            annual_cols.extend(["enso_annual", "enso_phase"])
        
        # Merge annual weather by year_out
        voyage_weather = voyages[["voyage_id", "year_out"]].copy()
        voyage_weather = voyage_weather.merge(
            annual_weather[[c for c in annual_cols if c in annual_weather.columns]],
            left_on="year_out",
            right_on="year",
            how="left"
        ).drop(columns=["year"], errors="ignore")
        
        # Add voyage-level exposure
        voyage_weather = voyage_weather.merge(voyage_exposure, on="voyage_id", how="left")
        
        # Rename for clarity
        voyage_weather = voyage_weather.rename(columns={
            "n_storms": "annual_storms",
            "n_hurricanes": "annual_hurricanes",
            "ace_approx": "annual_ace",
            "corridor_storms": "annual_corridor_storms",
        })
        
        print(f"\nVoyage weather integration complete:")
        print(f"  Total voyages: {len(voyage_weather):,}")
        print(f"  With NAO data: {voyage_weather['nao_index'].notna().sum():,}")
        print(f"  With hurricane data: {voyage_weather['annual_storms'].notna().sum():,}")
        if "pdo_annual" in voyage_weather.columns:
            print(f"  With PDO data: {voyage_weather['pdo_annual'].notna().sum():,}")
        if "enso_annual" in voyage_weather.columns:
            print(f"  With ENSO data: {voyage_weather['enso_annual'].notna().sum():,}")
    
    # Save raw data
    if save_raw:
        nao_df.to_csv(WEATHER_RAW_DIR / "nao_annual.csv", index=False)
        hurricane_annual.to_csv(WEATHER_RAW_DIR / "hurricane_annual.csv", index=False)
        if len(pdo_df) > 0:
            pdo_df.to_csv(WEATHER_RAW_DIR / "pdo_annual.csv", index=False)
        if len(enso_df) > 0:
            enso_df.to_csv(WEATHER_RAW_DIR / "enso_annual.csv", index=False)
        annual_weather.to_csv(WEATHER_RAW_DIR / "weather_annual_combined.csv", index=False)
        
        if len(voyage_weather) > 0:
            voyage_weather.to_parquet(DATA_DIR / "voyage_weather.parquet", index=False)
        
        print(f"\nSaved raw weather data to {WEATHER_RAW_DIR}")
    
    return annual_weather, voyage_weather


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    annual, voyage = download_and_integrate_weather()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print("\nAnnual Weather Sample (1865-1875):")
    print(annual[(annual["year"] >= 1865) & (annual["year"] <= 1875)].to_string(index=False))
    
    if len(voyage) > 0:
        print("\nVoyage Weather Sample:")
        print(voyage.head(10).to_string(index=False))
