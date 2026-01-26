"""
Labor metrics aggregation for voyage-level crew statistics.

Computes:
- crew_count: Total crew members per voyage
- desertion_count: Crew who deserted
- desertion_rate: desertion_count / crew_count
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import STAGING_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_voyage_labor_metrics(
    crew_df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Aggregate crew roster data to voyage-level labor metrics.
    
    Args:
        crew_df: Crew roster DataFrame with voyage_id, is_deserted columns
        output_path: Optional path to save results
        
    Returns:
        DataFrame with one row per voyage_id containing labor metrics
    """
    # Filter to records with voyage_id
    has_voyage = crew_df[crew_df["voyage_id"].notna()].copy()
    
    if len(has_voyage) == 0:
        logger.warning("No crew records with voyage_id found")
        return pd.DataFrame(columns=["voyage_id", "crew_count", "desertion_count", "desertion_rate"])
    
    logger.info(f"Computing labor metrics from {len(has_voyage)} crew records")
    
    # Aggregate by voyage
    metrics = has_voyage.groupby("voyage_id").agg(
        crew_count=("crew_member_row_id", "count"),
        desertion_count=("is_deserted", "sum"),
        
        # Additional metrics
        mean_age=("age", lambda x: x.dropna().mean() if x.notna().any() else np.nan),
        age_available_count=("age", lambda x: x.notna().sum()),
        
        # Rank distribution
        unique_ranks=("rank", "nunique"),
        
        # Birthplace diversity
        unique_birthplaces=("birthplace", "nunique"),
    ).reset_index()
    
    # Compute desertion rate
    metrics["desertion_rate"] = metrics["desertion_count"] / metrics["crew_count"]
    
    # Handle edge cases
    metrics["desertion_rate"] = metrics["desertion_rate"].fillna(0).clip(0, 1)
    
    # Add quality flags
    metrics["labor_data_quality"] = np.where(
        metrics["crew_count"] >= 10,
        "good",
        np.where(metrics["crew_count"] >= 5, "partial", "sparse")
    )
    
    logger.info(f"Computed metrics for {len(metrics)} voyages")
    logger.info(f"  Mean crew size: {metrics['crew_count'].mean():.1f}")
    logger.info(f"  Mean desertion rate: {metrics['desertion_rate'].mean():.3f}")
    
    # Save if path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        metrics.to_parquet(output_path, index=False)
        logger.info(f"Saved metrics to {output_path}")
        
        csv_path = output_path.with_suffix(".csv")
        metrics.to_csv(csv_path, index=False)
    
    return metrics


def compute_rank_distribution(crew_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rank distribution per voyage.
    
    Returns pivot table with counts by rank category.
    """
    has_voyage = crew_df[crew_df["voyage_id"].notna()].copy()
    
    # Categorize ranks
    def categorize_rank(rank: str) -> str:
        if not rank:
            return "unknown"
        
        rank = str(rank).upper()
        
        if rank in ["MASTER", "CAPTAIN", "CAPT"]:
            return "officer_captain"
        elif "MATE" in rank:
            return "officer_mate"
        elif rank in ["BOATSTEERER", "HARPOONER", "BOATHEADER"]:
            return "skilled_whale"
        elif rank in ["COOPER", "CARPENTER", "STEWARD", "COOK"]:
            return "skilled_other"
        elif rank in ["SEAMAN", "SAILOR", "ORDINARY SEAMAN", "ABLE SEAMAN"]:
            return "seaman"
        elif rank in ["BOY", "GREEN HAND", "GREENHND"]:
            return "green_hand"
        else:
            return "other"
    
    has_voyage["rank_category"] = has_voyage["rank"].apply(categorize_rank)
    
    # Pivot
    rank_dist = has_voyage.groupby(
        ["voyage_id", "rank_category"]
    ).size().unstack(fill_value=0)
    
    rank_dist.columns = [f"rank_{col}_count" for col in rank_dist.columns]
    
    return rank_dist.reset_index()


def compute_desertion_by_rank(crew_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute desertion rates by rank category per voyage.
    """
    has_voyage = crew_df[
        crew_df["voyage_id"].notna() & crew_df["rank"].notna()
    ].copy()
    
    # Categorize ranks (simplified)
    def is_officer(rank):
        rank = str(rank).upper()
        return "MASTER" in rank or "CAPTAIN" in rank or "MATE" in rank
    
    has_voyage["is_officer"] = has_voyage["rank"].apply(is_officer)
    
    # Aggregate
    officer_stats = has_voyage.groupby(["voyage_id", "is_officer"]).agg(
        count=("crew_member_row_id", "count"),
        deserted=("is_deserted", "sum"),
    ).reset_index()
    
    officer_stats["desertion_rate"] = officer_stats["deserted"] / officer_stats["count"]
    
    # Pivot to wide format
    officer_wide = officer_stats.pivot(
        index="voyage_id",
        columns="is_officer",
        values=["count", "desertion_rate"]
    )
    
    officer_wide.columns = [
        f"{'officer' if c[1] else 'crew'}_{c[0]}"
        for c in officer_wide.columns
    ]
    
    return officer_wide.reset_index()


if __name__ == "__main__":
    # Test with sample data
    from parsing.crew_parser import CrewParser
    
    parser = CrewParser()
    crew_df = parser.parse()
    
    metrics = compute_voyage_labor_metrics(
        crew_df,
        output_path=STAGING_DIR / "voyage_labor_metrics.parquet"
    )
    
    print("\n=== Labor Metrics Sample ===")
    print(metrics.head(10))
