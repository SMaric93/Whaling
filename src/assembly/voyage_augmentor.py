"""
Voyage Augmentor - Build analysis_voyage_augmented.

Combines base analysis_voyage with WSL event features, route validation
metrics, and optional ICOADS weather controls.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import STAGING_DIR, FINAL_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_base_voyage(path: Optional[Path] = None) -> pd.DataFrame:
    """Load base analysis_voyage."""
    if path is None:
        path = FINAL_DIR / "analysis_voyage.parquet"
    
    if not path.exists():
        raise FileNotFoundError(f"Base analysis_voyage not found: {path}")
    
    df = pd.read_parquet(path)
    logger.info(f"Loaded base analysis_voyage: {len(df)} voyages")
    return df


def build_wsl_voyage_features(
    event_panel_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Aggregate WSL events into voyage-level features.
    
    Features:
    - n_wsl_events_total: Total events mentioning this voyage
    - n_wsl_departures, n_wsl_arrivals, etc.: Event type counts
    - has_wsl_loss: Binary indicator for loss/wreck events
    - first_wsl_mention: Earliest WSL mention date
    - wsl_coverage_score: Mean match confidence
    """
    if event_panel_path is None:
        event_panel_path = STAGING_DIR / "wsl_voyage_event_panel.parquet"
    
    if not event_panel_path.exists():
        logger.warning(f"WSL event panel not found: {event_panel_path}")
        return pd.DataFrame(columns=['voyage_id'])
    
    panel = pd.read_parquet(event_panel_path)
    
    if len(panel) == 0:
        return pd.DataFrame(columns=['voyage_id'])
    
    # Aggregate by voyage
    features = panel.groupby('voyage_id').agg({
        'event_count': 'sum',
        'mean_confidence': 'mean',
        'event_date': 'min',
    }).reset_index()
    features.columns = ['voyage_id', 'n_wsl_events_total', 'wsl_coverage_score', 'first_wsl_mention']
    
    # Event type counts
    event_type_counts = panel.pivot_table(
        index='voyage_id',
        columns='event_type',
        values='event_count',
        aggfunc='sum',
        fill_value=0
    ).reset_index()
    
    # Rename columns
    event_col_map = {
        'DEPARTURE': 'n_wsl_departures',
        'ARRIVAL': 'n_wsl_arrivals',
        'SPOKEN_WITH': 'n_wsl_spoken',
        'REPORTED_AT': 'n_wsl_reported',
        'WRECK': 'n_wsl_wrecks',
        'LOSS': 'n_wsl_losses',
        'CAPTURED': 'n_wsl_captured',
        'DAMAGED': 'n_wsl_damaged',
        'RETURNED_HOME': 'n_wsl_returned',
        'OTHER': 'n_wsl_other',
    }
    event_type_counts = event_type_counts.rename(columns=event_col_map)
    
    # Merge
    features = features.merge(event_type_counts, on='voyage_id', how='left')
    
    # Binary loss indicator
    loss_cols = ['n_wsl_wrecks', 'n_wsl_losses', 'n_wsl_captured']
    for col in loss_cols:
        if col not in features.columns:
            features[col] = 0
    
    features['has_wsl_loss'] = (
        features['n_wsl_wrecks'] + features['n_wsl_losses'] + features['n_wsl_captured']
    ) > 0
    
    logger.info(f"Built WSL features for {len(features)} voyages")
    return features


def load_route_validation(
    path: Optional[Path] = None,
) -> pd.DataFrame:
    """Load route validation metrics."""
    if path is None:
        path = STAGING_DIR / "route_validation_metrics.parquet"
    
    if not path.exists():
        logger.warning(f"Route validation not found: {path}")
        return pd.DataFrame(columns=['voyage_id'])
    
    df = pd.read_parquet(path)
    
    # Select key columns
    keep_cols = [
        'voyage_id', 'maury_days', 'frac_arctic_maury', 'frac_arctic_aowv',
        'arctic_exposure_diff', 'route_discrepancy_flag', 'route_overlap_score'
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    
    df = df[keep_cols]
    logger.info(f"Loaded route validation for {len(df)} voyages")
    return df


def load_icoads_controls(
    path: Optional[Path] = None,
) -> pd.DataFrame:
    """Load optional ICOADS weather controls."""
    if path is None:
        path = STAGING_DIR / "icoads_route_weather_controls_optional.parquet"
    
    if not path.exists():
        logger.info("ICOADS controls not found (optional)")
        return pd.DataFrame(columns=['voyage_id'])
    
    df = pd.read_parquet(path)
    logger.info(f"Loaded ICOADS controls for {len(df)} voyages")
    return df


def build_augmented_voyage(
    base_voyage: pd.DataFrame,
    wsl_features: pd.DataFrame,
    route_validation: pd.DataFrame,
    icoads_controls: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the augmented voyage table.
    
    Args:
        base_voyage: Base analysis_voyage
        wsl_features: WSL-derived features
        route_validation: Route validation metrics
        icoads_controls: Optional ICOADS controls
        
    Returns:
        Augmented voyage DataFrame
    """
    result = base_voyage.copy()
    
    # Left-join WSL features
    if len(wsl_features) > 0 and 'voyage_id' in wsl_features.columns:
        result = result.merge(wsl_features, on='voyage_id', how='left')
        logger.info(f"Added WSL features: {len(wsl_features.columns) - 1} columns")
    
    # Left-join route validation
    if len(route_validation) > 0 and 'voyage_id' in route_validation.columns:
        result = result.merge(route_validation, on='voyage_id', how='left')
        logger.info(f"Added route validation: {len(route_validation.columns) - 1} columns")
    
    # Left-join ICOADS controls (optional)
    if len(icoads_controls) > 0 and 'voyage_id' in icoads_controls.columns:
        result = result.merge(icoads_controls, on='voyage_id', how='left')
        logger.info(f"Added ICOADS controls: {len(icoads_controls.columns) - 1} columns")
    
    # Fill missing values for event counts
    event_count_cols = [c for c in result.columns if c.startswith('n_wsl_')]
    for col in event_count_cols:
        result[col] = result[col].fillna(0).astype(int)
    
    # Fill boolean flags
    bool_cols = ['has_wsl_loss', 'route_discrepancy_flag']
    for col in bool_cols:
        if col in result.columns:
            result[col] = result[col].fillna(False)
    
    logger.info(f"Built augmented voyage: {len(result)} rows, {len(result.columns)} columns")
    return result


def save_augmented_voyage(
    df: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """Save augmented voyage to final directory."""
    if output_dir is None:
        output_dir = FINAL_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    parquet_path = output_dir / "analysis_voyage_augmented.parquet"
    csv_path = output_dir / "analysis_voyage_augmented.csv"
    
    df.to_parquet(parquet_path)
    df.to_csv(csv_path, index=False)
    
    logger.info(f"Saved augmented voyage: {parquet_path}")
    logger.info(f"Saved augmented voyage: {csv_path}")
    
    return {
        'parquet': parquet_path,
        'csv': csv_path,
    }


def run_voyage_augmentation(
    base_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Run the full voyage augmentation pipeline.
    
    Args:
        base_path: Path to base analysis_voyage
        
    Returns:
        Augmented voyage DataFrame
    """
    logger.info("=" * 60)
    logger.info("Building Analysis Voyage Augmented")
    logger.info("=" * 60)
    
    # Load base
    base_voyage = load_base_voyage(base_path)
    
    # Load augmentation sources
    wsl_features = build_wsl_voyage_features()
    route_validation = load_route_validation()
    icoads_controls = load_icoads_controls()
    
    # Build augmented table
    augmented = build_augmented_voyage(
        base_voyage,
        wsl_features,
        route_validation,
        icoads_controls,
    )
    
    # Save outputs
    save_augmented_voyage(augmented)
    
    # Summary stats
    logger.info("\n" + "=" * 60)
    logger.info("Augmentation Summary")
    logger.info("=" * 60)
    logger.info(f"Total voyages: {len(augmented)}")
    
    if 'n_wsl_events_total' in augmented.columns:
        wsl_covered = (augmented['n_wsl_events_total'] > 0).sum()
        logger.info(f"With WSL events: {wsl_covered} ({wsl_covered/len(augmented):.1%})")
    
    if 'maury_days' in augmented.columns:
        maury_covered = augmented['maury_days'].notna().sum()
        logger.info(f"With Maury data: {maury_covered} ({maury_covered/len(augmented):.1%})")
    
    if 'route_discrepancy_flag' in augmented.columns:
        flagged = augmented['route_discrepancy_flag'].sum()
        logger.info(f"Route discrepancies: {flagged}")
    
    return augmented


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build augmented analysis voyage")
    parser.add_argument("--base-path", type=Path, default=None,
                        help="Path to base analysis_voyage")
    
    args = parser.parse_args()
    
    augmented = run_voyage_augmentation(args.base_path)
    
    print(f"\nAugmented voyage: {len(augmented)} rows, {len(augmented.columns)} columns")
    print(f"\nNew columns added:")
    base_cols = set(load_base_voyage(args.base_path).columns) if args.base_path else set()
    new_cols = [c for c in augmented.columns if c not in base_cols]
    for col in new_cols[:20]:
        print(f"  - {col}")
    if len(new_cols) > 20:
        print(f"  ... and {len(new_cols) - 20} more")
