"""
Agent Strategy Feature Engineering.

Computes agent-level metrics and voyage assignments:
- Fleet exposure variance (risk portfolio diversification)
- Route assignment quality (captain-route matching)
- Ground timing optimality (seasonal timing patterns)

These metrics test whether agents exhibited portfolio management skill.
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
# Data Structures
# =============================================================================

@dataclass
class AgentStrategyMetrics:
    """Strategy metrics for a single agent."""
    agent_id: str
    n_voyages: int
    n_captains: int
    n_routes: int
    fleet_exposure_variance: float
    avg_voyage_revenue: float
    revenue_variance: float
    route_concentration: float  # HHI of route assignments
    captain_loyalty: float  # Avg captain tenure with agent


# =============================================================================
# Agent Portfolio Analysis
# =============================================================================

def compute_agent_portfolio_metrics(
    voyages: pd.DataFrame,
    route_features: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute agent-level portfolio and strategy metrics.
    
    Parameters
    ----------
    voyages : pd.DataFrame
        Voyage data with agent_id, captain_id, route, and outcome columns.
    route_features : pd.DataFrame, optional
        Route-level features (exposure, efficiency) for weighting.
        
    Returns
    -------
    pd.DataFrame
        Agent-level strategy metrics.
    """
    logger.info("Computing agent portfolio metrics...")
    
    # Identify required columns
    required_cols = ["agent_id"]
    if not all(col in voyages.columns for col in required_cols):
        logger.warning(f"Missing required columns. Have: {list(voyages.columns)}")
        return pd.DataFrame()
    
    agent_groups = voyages.groupby("agent_id")
    
    metrics_list = []
    
    for agent_id, group in agent_groups:
        n_voyages = len(group)
        
        # Captain metrics
        if "captain_id" in group.columns:
            n_captains = group["captain_id"].nunique()
            captain_counts = group["captain_id"].value_counts()
            captain_loyalty = (captain_counts > 1).mean() if len(captain_counts) > 0 else 0
        else:
            n_captains = 0
            captain_loyalty = 0
        
        # Route metrics
        if "route" in group.columns:
            n_routes = group["route"].nunique()
            route_counts = group["route"].value_counts(normalize=True)
            route_hhi = (route_counts ** 2).sum() if len(route_counts) > 0 else 1.0
        else:
            n_routes = 0
            route_hhi = 1.0
        
        # Revenue metrics (use outcome variable if available)
        outcome_col = None
        for col in ["log_output", "total_oil", "revenue", "oil_bbls"]:
            if col in group.columns:
                outcome_col = col
                break
        
        if outcome_col:
            avg_revenue = group[outcome_col].mean()
            revenue_variance = group[outcome_col].var()
        else:
            avg_revenue = np.nan
            revenue_variance = np.nan
        
        # Fleet exposure variance (if storm exposure available)
        if "hurricane_exposure_count" in group.columns:
            exposure_variance = group["hurricane_exposure_count"].var()
        else:
            exposure_variance = np.nan
        
        metrics_list.append(AgentStrategyMetrics(
            agent_id=agent_id,
            n_voyages=n_voyages,
            n_captains=n_captains,
            n_routes=n_routes,
            fleet_exposure_variance=exposure_variance,
            avg_voyage_revenue=avg_revenue,
            revenue_variance=revenue_variance,
            route_concentration=route_hhi,
            captain_loyalty=captain_loyalty,
        ))
    
    # Convert to DataFrame
    df = pd.DataFrame([vars(m) for m in metrics_list])
    
    logger.info(f"Computed metrics for {len(df):,} agents")
    logger.info(f"  Mean voyages per agent: {df['n_voyages'].mean():.1f}")
    logger.info(f"  Mean captains per agent: {df['n_captains'].mean():.1f}")
    
    return df


# =============================================================================
# Route Assignment Quality
# =============================================================================

def compute_route_assignment_quality(
    voyages: pd.DataFrame,
    captain_route_outcomes: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute how well agents matched captains to routes.
    
    Quality metric: Did captains assigned by agent to route X
    have prior success on route X or similar routes?
    
    Parameters
    ----------
    voyages : pd.DataFrame
        Voyage data with agent, captain, route, and outcome columns.
    captain_route_outcomes : pd.DataFrame, optional
        Historical captain-route performance for matching analysis.
        
    Returns
    -------
    pd.DataFrame
        Voyage-level route assignment quality scores.
    """
    if "captain_id" not in voyages.columns or "route" not in voyages.columns:
        logger.warning("Cannot compute route assignment quality without captain_id and route")
        return pd.DataFrame()
    
    logger.info("Computing route assignment quality...")
    
    # Compute captain's prior experience on each route
    voyages = voyages.sort_values(["captain_id", "year_out"]).copy()
    
    # For each voyage, count captain's prior voyages on same route
    prior_route_experience = []
    
    captain_route_history = {}
    
    for _, row in voyages.iterrows():
        captain = row.get("captain_id")
        route = row.get("route")
        voyage_id = row.get("voyage_id")
        
        if pd.isna(captain) or pd.isna(route):
            prior_route_experience.append({
                "voyage_id": voyage_id,
                "prior_route_experience": 0,
            })
            continue
        
        key = (captain, route)
        prior_count = captain_route_history.get(key, 0)
        
        prior_route_experience.append({
            "voyage_id": voyage_id,
            "prior_route_experience": prior_count,
        })
        
        captain_route_history[key] = prior_count + 1
    
    df = pd.DataFrame(prior_route_experience)
    
    # Merge back to voyages
    if "voyage_id" in voyages.columns:
        result = voyages[["voyage_id", "agent_id", "captain_id", "route"]].merge(
            df, on="voyage_id", how="left"
        )
    else:
        result = df
    
    # Create binary indicator for "experienced on route"
    result["experienced_on_route"] = (result["prior_route_experience"] > 0).astype(int)
    
    logger.info(f"Route assignment quality computed for {len(result):,} voyages")
    logger.info(f"  Experienced captain-route matches: {result['experienced_on_route'].mean():.1%}")
    
    return result


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compute agent strategy metrics"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save computed features"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("AGENT STRATEGY FEATURE ENGINEERING")
    print("=" * 60)
    
    # Load voyage data
    voyage_path = FINAL_DIR / "analysis_voyage.parquet"
    if not voyage_path.exists():
        print(f"Voyage data not found at {voyage_path}")
        sys.exit(1)
    
    voyages = pd.read_parquet(voyage_path)
    print(f"Loaded {len(voyages):,} voyages")
    
    # Compute agent metrics
    agent_metrics = compute_agent_portfolio_metrics(voyages)
    
    # Compute route assignment quality
    assignment_quality = compute_route_assignment_quality(voyages)
    
    # Show sample
    print("\n" + "-" * 60)
    print("AGENT METRICS SAMPLE")
    print("-" * 60)
    print(agent_metrics.head(20).to_string())
    
    print("\n" + "-" * 60)
    print("ROUTE ASSIGNMENT SAMPLE")
    print("-" * 60)
    if len(assignment_quality) > 0:
        print(assignment_quality.head(10).to_string())
    
    if args.save:
        output_path = FINAL_DIR / "agent_strategy_metrics.parquet"
        agent_metrics.to_parquet(output_path, index=False)
        agent_metrics.to_csv(output_path.with_suffix('.csv'), index=False)
        
        if len(assignment_quality) > 0:
            assign_path = FINAL_DIR / "voyage_route_assignment.parquet"
            assignment_quality.to_parquet(assign_path, index=False)
        
        print(f"\nSaved to {output_path}")
    
    print("\nDone!")
