"""
Maury Logbook Data Parser.

Parses the Maury Logbook Data ZIP from WhalingHistory.org into
standardized position records for route validation.

Maury data coverage: primarily 1820-1855
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import logging
import uuid

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_MAURY, STAGING_DIR
from parsing.string_normalizer import normalize_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_obs_id() -> str:
    """Generate unique observation ID."""
    return f"maury_{uuid.uuid4().hex[:10]}"


def load_maury_data(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load Maury logbook data from downloaded files.
    
    The Maury ZIP typically contains tab or comma-separated data files.
    
    Args:
        data_dir: Directory containing Maury data files
        
    Returns:
        Raw DataFrame of Maury records
    """
    if data_dir is None:
        data_dir = RAW_MAURY
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Maury data directory not found: {data_dir}")
    
    # Find data files
    data_files = list(data_dir.glob("*.txt")) + list(data_dir.glob("*.csv"))
    
    if not data_files:
        raise FileNotFoundError(f"No data files found in {data_dir}")
    
    logger.info(f"Found {len(data_files)} Maury data files")
    
    dfs = []
    for f in data_files:
        try:
            # Try different delimiters
            for sep in ['\t', ',', '|']:
                try:
                    df = pd.read_csv(f, sep=sep, encoding='utf-8', low_memory=False)
                    if len(df.columns) > 3:
                        dfs.append(df)
                        logger.info(f"  Loaded {f.name}: {len(df)} rows, {len(df.columns)} cols")
                        break
                except:
                    continue
        except Exception as e:
            logger.warning(f"  Failed to load {f.name}: {e}")
    
    if not dfs:
        return pd.DataFrame()
    
    # Concatenate all files
    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined Maury data: {len(combined)} total rows")
    
    return combined


def standardize_maury_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize Maury column names to expected schema.
    
    Expected output columns:
    - maury_obs_id, obs_date, vessel_name_clean, captain_name_clean
    - lat, lon, whale_species_code, source_record_id
    """
    # Map common column names
    column_mapping = {
        # Date columns
        'date': 'obs_date',
        'logdate': 'obs_date',
        'log_date': 'obs_date',
        'year': 'year',
        'month': 'month',
        'day': 'day',
        # Vessel columns
        'vessel': 'vessel_name_raw',
        'ship': 'vessel_name_raw',
        'vesselname': 'vessel_name_raw',
        'vessel_name': 'vessel_name_raw',
        # Captain columns
        'captain': 'captain_name_raw',
        'master': 'captain_name_raw',
        'capt': 'captain_name_raw',
        # Position columns
        'lat': 'lat',
        'latitude': 'lat',
        'lon': 'lon',
        'long': 'lon',
        'longitude': 'lon',
        # Species
        'species': 'whale_species_code',
        'whalespecies': 'whale_species_code',
        'whale': 'whale_species_code',
        # ID
        'id': 'source_record_id',
        'record_id': 'source_record_id',
        'recno': 'source_record_id',
    }
    
    # Lowercase column names for matching
    df.columns = df.columns.str.lower().str.strip()
    
    # Apply mapping
    new_cols = {}
    for old, new in column_mapping.items():
        if old in df.columns:
            new_cols[old] = new
    
    df = df.rename(columns=new_cols)
    
    return df


def parse_maury_positions(
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Parse raw Maury data into standardized position records.
    
    Args:
        raw_df: Raw Maury DataFrame
        
    Returns:
        Standardized positions DataFrame
    """
    df = standardize_maury_columns(raw_df.copy())
    
    # Generate observation IDs
    df['maury_obs_id'] = [generate_obs_id() for _ in range(len(df))]
    
    # Parse dates
    if 'obs_date' not in df.columns:
        # Try to construct from year/month/day
        if all(c in df.columns for c in ['year', 'month', 'day']):
            df['obs_date'] = pd.to_datetime({
                'year': df['year'],
                'month': df['month'],
                'day': df['day']
            }, errors='coerce').dt.strftime('%Y-%m-%d')
        elif 'year' in df.columns:
            df['obs_date'] = df['year'].astype(str)
    
    # Normalize vessel names
    if 'vessel_name_raw' in df.columns:
        df['vessel_name_clean'] = df['vessel_name_raw'].apply(
            lambda x: normalize_name(x) if pd.notna(x) else None
        )
    else:
        df['vessel_name_clean'] = None
    
    # Normalize captain names
    if 'captain_name_raw' in df.columns:
        df['captain_name_clean'] = df['captain_name_raw'].apply(
            lambda x: normalize_name(x) if pd.notna(x) else None
        )
    else:
        df['captain_name_clean'] = None
    
    # Ensure lat/lon are numeric
    for col in ['lat', 'lon']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = None
    
    # Select output columns
    output_cols = [
        'maury_obs_id', 'obs_date', 'vessel_name_clean', 'captain_name_clean',
        'lat', 'lon', 'whale_species_code', 'source_record_id'
    ]
    
    # Fill missing columns
    for col in output_cols:
        if col not in df.columns:
            df[col] = None
    
    result = df[output_cols].copy()
    
    # Filter valid records (must have vessel and date)
    valid = result['vessel_name_clean'].notna() & result['obs_date'].notna()
    result = result[valid].reset_index(drop=True)
    
    logger.info(f"Parsed {len(result)} valid Maury position records")
    
    return result


def save_maury_positions(
    positions_df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> Path:
    """Save Maury positions to staging."""
    if output_path is None:
        output_path = STAGING_DIR / "maury_positions.parquet"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    positions_df.to_parquet(output_path)
    positions_df.to_csv(output_path.with_suffix('.csv'), index=False)
    
    logger.info(f"Saved {len(positions_df)} Maury positions to {output_path}")
    return output_path


def run_maury_parser(
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Run full Maury parsing pipeline.
    
    Args:
        data_dir: Directory containing Maury data
        
    Returns:
        Parsed positions DataFrame
    """
    raw_df = load_maury_data(data_dir)
    
    if len(raw_df) == 0:
        logger.warning("No Maury data loaded")
        return pd.DataFrame()
    
    positions_df = parse_maury_positions(raw_df)
    save_maury_positions(positions_df)
    
    return positions_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse Maury logbook data")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Path to Maury data directory")
    
    args = parser.parse_args()
    
    df = run_maury_parser(args.data_dir)
    
    print(f"\nParsed {len(df)} Maury positions")
    if len(df) > 0:
        print(f"\nDate range: {df['obs_date'].min()} to {df['obs_date'].max()}")
        print(f"Unique vessels: {df['vessel_name_clean'].nunique()}")
        print(f"\nSample records:")
        print(df.head(5).to_string())
