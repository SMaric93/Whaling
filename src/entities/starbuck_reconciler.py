"""
Starbuck to AOWV Reconciler.

Matches Starbuck voyage entries to AOWV voyages_master and computes
coverage statistics to identify systematic gaps.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import STAGING_DIR, MATCH_CONFIDENCE_THRESHOLDS

# Try fuzzy matching
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
    if n1 in n2 or n2 in n1:
        return 0.8
    if n1.split()[0] == n2.split()[0]:
        return 0.7
    return 0.0


def check_year_overlap(
    sb_dep: Optional[int],
    sb_ret: Optional[int],
    aowv_out: Optional[str],
    aowv_in: Optional[str],
    tolerance: int = 2,
) -> Tuple[bool, float]:
    """
    Check if Starbuck and AOWV voyage years overlap.
    
    Args:
        sb_dep: Starbuck departure year
        sb_ret: Starbuck return year
        aowv_out: AOWV date_out string
        aowv_in: AOWV date_in string
        tolerance: Year tolerance for matching
        
    Returns:
        Tuple of (overlaps, score)
    """
    if sb_dep is None:
        return False, 0.0
    
    # Parse AOWV years
    aowv_year_out = None
    aowv_year_in = None
    
    if aowv_out and len(str(aowv_out)) >= 4:
        try:
            aowv_year_out = int(str(aowv_out)[:4])
        except:
            pass
    
    if aowv_in and len(str(aowv_in)) >= 4:
        try:
            aowv_year_in = int(str(aowv_in)[:4])
        except:
            pass
    
    if aowv_year_out is None and aowv_year_in is None:
        return False, 0.0
    
    # Build year ranges
    sb_start = sb_dep
    sb_end = sb_ret if sb_ret else sb_dep + 3  # Assume 3-year voyage if no return
    
    aowv_start = aowv_year_out if aowv_year_out else aowv_year_in - 3
    aowv_end = aowv_year_in if aowv_year_in else aowv_year_out + 3
    
    # Check overlap with tolerance
    if sb_start - tolerance <= aowv_end and sb_end + tolerance >= aowv_start:
        # Compute overlap score
        overlap_start = max(sb_start, aowv_start)
        overlap_end = min(sb_end, aowv_end)
        overlap = max(0, overlap_end - overlap_start + 1)
        duration = max(1, (sb_end - sb_start + 1 + aowv_end - aowv_start + 1) / 2)
        score = min(1.0, overlap / duration)
        return True, score
    
    return False, 0.0


def match_starbuck_to_aowv(
    starbuck_df: pd.DataFrame,
    aowv_df: pd.DataFrame,
    name_threshold: float = 0.8,
    year_tolerance: int = 2,
) -> pd.DataFrame:
    """
    Match Starbuck voyages to AOWV voyages_master.
    
    Args:
        starbuck_df: Starbuck voyage list
        aowv_df: AOWV voyages master
        name_threshold: Minimum name similarity
        year_tolerance: Year tolerance for matching
        
    Returns:
        Reconciliation DataFrame
    """
    logger.info(f"Matching {len(starbuck_df)} Starbuck to {len(aowv_df)} AOWV voyages")
    
    results = []
    
    for idx, sb_row in starbuck_df.iterrows():
        sb_vessel = sb_row.get('vessel_name_clean')
        sb_dep = sb_row.get('departure_year')
        sb_ret = sb_row.get('return_year')
        
        candidates = []
        
        if not sb_vessel:
            results.append({
                'starbuck_row_id': sb_row['starbuck_row_id'],
                'candidate_voyage_ids': [],
                'best_voyage_id': None,
                'match_confidence': 0.0,
                'match_method': 'no_vessel',
                'unmatched_flag': True,
            })
            continue
        
        # Find vessel matches
        for _, aowv_row in aowv_df.iterrows():
            aowv_vessel = aowv_row.get('vessel_name_clean')
            
            name_sim = compute_name_similarity(sb_vessel, aowv_vessel)
            if name_sim < name_threshold:
                continue
            
            # Check year overlap
            overlaps, year_score = check_year_overlap(
                sb_dep, sb_ret,
                aowv_row.get('date_out'), aowv_row.get('date_in'),
                year_tolerance
            )
            
            if overlaps or year_score > 0:
                total_score = 0.6 * name_sim + 0.4 * year_score
                candidates.append({
                    'voyage_id': aowv_row.get('voyage_id'),
                    'score': total_score,
                    'name_sim': name_sim,
                    'year_score': year_score,
                })
        
        # Sort by score
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        if candidates:
            best = candidates[0]
            results.append({
                'starbuck_row_id': sb_row['starbuck_row_id'],
                'candidate_voyage_ids': [c['voyage_id'] for c in candidates[:3]],
                'best_voyage_id': best['voyage_id'],
                'match_confidence': best['score'],
                'match_method': 'vessel_year',
                'unmatched_flag': best['score'] < 0.7,
            })
        else:
            results.append({
                'starbuck_row_id': sb_row['starbuck_row_id'],
                'candidate_voyage_ids': [],
                'best_voyage_id': None,
                'match_confidence': 0.0,
                'match_method': 'no_match',
                'unmatched_flag': True,
            })
    
    df = pd.DataFrame(results)
    
    # Stats
    matched = (~df['unmatched_flag']).sum()
    total = len(df)
    logger.info(f"Matched {matched}/{total} ({matched/total:.1%}) Starbuck entries")
    
    return df


def compute_coverage_metrics(
    starbuck_df: pd.DataFrame,
    reconciliation_df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    Compute coverage metrics by decade and port.
    
    Args:
        starbuck_df: Original Starbuck voyage list
        reconciliation_df: Reconciliation results
        
    Returns:
        Dict with 'by_decade', 'by_port', 'summary' DataFrames
    """
    # Merge for analysis
    merged = starbuck_df.merge(
        reconciliation_df[['starbuck_row_id', 'match_confidence', 'unmatched_flag']],
        on='starbuck_row_id',
        how='left'
    )
    
    # Add decade
    merged['decade'] = (merged['departure_year'] // 10 * 10).astype('Int64')
    
    # By decade
    by_decade = merged.groupby('decade').agg({
        'starbuck_row_id': 'count',
        'unmatched_flag': 'sum',
        'match_confidence': 'mean',
    }).reset_index()
    by_decade.columns = ['decade', 'total', 'unmatched', 'avg_confidence']
    by_decade['match_rate'] = 1 - by_decade['unmatched'] / by_decade['total']
    
    # By port
    by_port = merged.groupby('home_port_clean').agg({
        'starbuck_row_id': 'count',
        'unmatched_flag': 'sum',
        'match_confidence': 'mean',
    }).reset_index()
    by_port.columns = ['port', 'total', 'unmatched', 'avg_confidence']
    by_port['match_rate'] = 1 - by_port['unmatched'] / by_port['total']
    by_port = by_port.sort_values('total', ascending=False)
    
    # Summary
    total = len(merged)
    unmatched = merged['unmatched_flag'].sum()
    summary = pd.DataFrame([{
        'total_starbuck': total,
        'matched': total - unmatched,
        'unmatched': unmatched,
        'match_rate': (total - unmatched) / total if total > 0 else 0,
        'avg_confidence': merged['match_confidence'].mean(),
    }])
    
    return {
        'by_decade': by_decade,
        'by_port': by_port,
        'summary': summary,
    }


def save_reconciliation_outputs(
    reconciliation_df: pd.DataFrame,
    coverage_metrics: Dict[str, pd.DataFrame],
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """Save reconciliation and coverage outputs."""
    if output_dir is None:
        output_dir = STAGING_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    
    # Save reconciliation
    recon_path = output_dir / "starbuck_to_aowv_reconciliation.parquet"
    recon_for_save = reconciliation_df.copy()
    recon_for_save['candidate_voyage_ids'] = recon_for_save['candidate_voyage_ids'].apply(str)
    recon_for_save.to_parquet(recon_path)
    recon_for_save.to_csv(recon_path.with_suffix('.csv'), index=False)
    outputs['reconciliation'] = recon_path
    
    # Save coverage metrics
    for name, df in coverage_metrics.items():
        metric_path = output_dir / f"starbuck_coverage_{name}.csv"
        df.to_csv(metric_path, index=False)
        outputs[f'coverage_{name}'] = metric_path
    
    logger.info(f"Saved reconciliation outputs to {output_dir}")
    return outputs


def run_starbuck_reconciliation(
    starbuck_path: Optional[Path] = None,
    aowv_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Run full Starbuck reconciliation pipeline.
    
    Args:
        starbuck_path: Path to starbuck_voyage_list
        aowv_path: Path to voyages_master
        
    Returns:
        Tuple of (reconciliation_df, coverage_metrics)
    """
    if starbuck_path is None:
        starbuck_path = STAGING_DIR / "starbuck_voyage_list.parquet"
    if aowv_path is None:
        aowv_path = STAGING_DIR / "voyages_master.parquet"
    
    # Load data
    starbuck_df = pd.read_parquet(starbuck_path)
    aowv_df = pd.read_parquet(aowv_path)
    
    # Match
    reconciliation_df = match_starbuck_to_aowv(starbuck_df, aowv_df)
    
    # Coverage
    coverage_metrics = compute_coverage_metrics(starbuck_df, reconciliation_df)
    
    # Save
    save_reconciliation_outputs(reconciliation_df, coverage_metrics)
    
    return reconciliation_df, coverage_metrics


if __name__ == "__main__":
    recon, metrics = run_starbuck_reconciliation()
    
    print("\n=== Starbuck Reconciliation Summary ===")
    print(metrics['summary'].to_string(index=False))
    
    print("\n=== By Decade ===")
    print(metrics['by_decade'].to_string(index=False))
    
    print("\n=== By Port (top 10) ===")
    print(metrics['by_port'].head(10).to_string(index=False))
