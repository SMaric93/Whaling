"""
Information Network Feature Engineering.

Parses Maury logbook "SpokenVessels" field to compute network metrics:
- spoke_with_count: Number of vessels spoken with during voyage
- spoke_with_success_rate: Average recent performance of spoken vessels
- info_advantage: Weighted information network score

References:
- Maury data: data/raw/maury/maury_download_20180104.txt
- SpokenVessels field format: "Ship/Bark/Schooner [Name] [Port]"
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
import re
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import RAW_DIR, STAGING_DIR, FINAL_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

MAURY_PATH = RAW_DIR / "maury" / "maury_download_20180104.txt"

# Vessel type prefixes to parse
VESSEL_PREFIXES = ["Ship", "Bark", "Brig", "Schooner", "Sloop"]


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class VoyageNetworkFeatures:
    """Network features for a single voyage."""
    voyage_id: str
    spoke_with_count: int
    unique_vessels_spoken: int
    unique_ports_spoken: int
    spoke_with_vessel_names: List[str]


# =============================================================================
# SpokenVessels Parsing
# =============================================================================

def parse_spoken_vessels(spoken_str: str) -> List[Dict[str, str]]:
    """
    Parse SpokenVessels field into structured vessel records.
    
    Format examples:
    - "Ship Sklock"
    - "Bark Parker Cook"
    - "Ship Margaret Newport; Bark Envoy New Bedford"
    - "Ship John New Bedford; Luminary of Warren"
    
    Returns
    -------
    List[Dict[str, str]]
        List of dicts with keys: vessel_type, vessel_name, port
    """
    if pd.isna(spoken_str) or not spoken_str.strip():
        return []
    
    vessels = []
    
    # Split by semicolon for multiple vessels
    parts = [p.strip() for p in str(spoken_str).split(";")]
    
    for part in parts:
        if not part:
            continue
        
        vessel = {"vessel_type": None, "vessel_name": None, "port": None}
        
        # Try to extract vessel type
        for prefix in VESSEL_PREFIXES:
            if part.lower().startswith(prefix.lower()):
                vessel["vessel_type"] = prefix
                part = part[len(prefix):].strip()
                break
        
        # Try to extract port (common ports at end)
        known_ports = [
            "New Bedford", "Nantucket", "Fairhaven", "New London",
            "Edgartown", "Newport", "Warren", "Sag Harbor",
            "Provincetown", "Mattapoisett", "Westport", "Mystic"
        ]
        
        for port in known_ports:
            if part.lower().endswith(port.lower()):
                vessel["port"] = port
                part = part[:-len(port)].strip()
                # Remove "of" if present
                if part.endswith(" of"):
                    part = part[:-3].strip()
                break
        
        # Remaining text is vessel name
        vessel["vessel_name"] = part.strip() if part else None
        
        if vessel["vessel_name"]:
            vessels.append(vessel)
    
    return vessels


# =============================================================================
# Load Maury Data
# =============================================================================

def load_maury_data() -> pd.DataFrame:
    """Load Maury logbook data with SpokenVessels field."""
    if not MAURY_PATH.exists():
        raise FileNotFoundError(f"Maury data not found at {MAURY_PATH}")
    
    logger.info(f"Loading Maury data from {MAURY_PATH}")
    
    df = pd.read_csv(MAURY_PATH, sep="\t", low_memory=False)
    
    logger.info(f"Loaded {len(df):,} Maury records")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Filter to rows with SpokenVessels data
    has_spoken = df["SpokenVessels"].notna() & (df["SpokenVessels"] != "")
    logger.info(f"Records with SpokenVessels: {has_spoken.sum():,}")
    
    return df


def compute_voyage_network_features(
    maury_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute network features from Maury SpokenVessels data.
    
    Parameters
    ----------
    maury_df : pd.DataFrame
        Maury data with SpokenVessels column.
        
    Returns
    -------
    pd.DataFrame
        Voyage-level network features.
    """
    logger.info("Computing voyage network features...")
    
    # Parse SpokenVessels for all records
    maury_df = maury_df.copy()
    maury_df["parsed_vessels"] = maury_df["SpokenVessels"].apply(parse_spoken_vessels)
    
    # Aggregate by voyage
    voyage_groups = maury_df.groupby("VoyageID")
    
    features_list = []
    
    for voyage_id, group in voyage_groups:
        # Collect all spoken vessels across voyage
        all_spoken = []
        for vessels in group["parsed_vessels"]:
            all_spoken.extend(vessels)
        
        # Count unique vessels
        vessel_names = [v["vessel_name"] for v in all_spoken if v["vessel_name"]]
        unique_vessels = list(set(vessel_names))
        
        # Count unique ports
        ports = [v["port"] for v in all_spoken if v["port"]]
        unique_ports = list(set(ports))
        
        features_list.append(VoyageNetworkFeatures(
            voyage_id=voyage_id,
            spoke_with_count=len(all_spoken),
            unique_vessels_spoken=len(unique_vessels),
            unique_ports_spoken=len(unique_ports),
            spoke_with_vessel_names=unique_vessels,
        ))
    
    # Convert to DataFrame
    df = pd.DataFrame([
        {
            "voyage_id": f.voyage_id,
            "spoke_with_count": f.spoke_with_count,
            "unique_vessels_spoken": f.unique_vessels_spoken,
            "unique_ports_spoken": f.unique_ports_spoken,
        }
        for f in features_list
    ])
    
    # Summary stats
    n_with_network = (df["spoke_with_count"] > 0).sum()
    logger.info(f"Computed network features for {len(df):,} voyages")
    logger.info(f"  Voyages with spoken vessels: {n_with_network:,}")
    logger.info(f"  Mean vessels spoken: {df['spoke_with_count'].mean():.2f}")
    
    return df


# =============================================================================
# Information Advantage Computation
# =============================================================================

def compute_information_advantage(
    network_df: pd.DataFrame,
    voyage_outcomes: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute information advantage by linking spoken vessels to their outcomes.
    
    This is a placeholder for more sophisticated network analysis.
    The full implementation would require:
    1. Linking spoken vessel names to voyage_ids in voyages_master
    2. Computing recent performance of those vessels
    3. Creating weighted information advantage score
    
    For now, uses spoke_with_count as proxy.
    """
    df = network_df.copy()
    
    # Simple proxy: log(spoke_with_count + 1) as info advantage
    df["info_advantage_proxy"] = np.log1p(df["spoke_with_count"])
    
    # Normalize to 0-1 scale
    max_val = df["info_advantage_proxy"].max()
    if max_val > 0:
        df["info_advantage_normalized"] = df["info_advantage_proxy"] / max_val
    else:
        df["info_advantage_normalized"] = 0.0
    
    return df


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compute voyage network features from Maury SpokenVessels"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save computed features"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("INFORMATION NETWORK FEATURE ENGINEERING")
    print("=" * 60)
    
    # Load Maury data
    maury_df = load_maury_data()
    
    # Compute features
    network_df = compute_voyage_network_features(maury_df)
    
    # Add information advantage
    network_df = compute_information_advantage(network_df)
    
    # Show sample
    print("\n" + "-" * 60)
    print("SAMPLE OUTPUT")
    print("-" * 60)
    print(network_df[network_df["spoke_with_count"] > 0].head(20).to_string())
    
    # Summary
    print("\n" + "-" * 60)
    print("SUMMARY")
    print("-" * 60)
    print(f"Total voyages: {len(network_df):,}")
    print(f"With network data: {(network_df['spoke_with_count'] > 0).sum():,}")
    print(f"Mean vessels spoken: {network_df['spoke_with_count'].mean():.2f}")
    print(f"Max vessels spoken: {network_df['spoke_with_count'].max()}")
    
    if args.save:
        output_path = FINAL_DIR / "voyage_network_features.parquet"
        network_df.to_parquet(output_path, index=False)
        network_df.to_csv(output_path.with_suffix('.csv'), index=False)
        print(f"\nSaved to {output_path}")
    
    print("\nDone!")
