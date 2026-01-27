"""
Logbook-Based Analyses Module.

Implements regression tests using logbook-derived features:
- Route-Level Skill Isolation (storm exposure × captain FE)
- Agent Selection Strategy (route assignment, information advantage)
- Behavioral/Decision Tests (ground switching, arrival timing)
- TFP Enhancement (route efficiency component)

Integrates with existing R1-R17 regression suite.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
import logging
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import FINAL_DIR, OUTPUT_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Loading
# =============================================================================

def load_analysis_data() -> pd.DataFrame:
    """Load voyage data with all available features."""
    logger.info("Loading analysis data...")
    
    # Base voyage data
    voyage_path = FINAL_DIR / "analysis_voyage_with_climate.parquet"
    if not voyage_path.exists():
        voyage_path = FINAL_DIR / "analysis_voyage.parquet"
    
    df = pd.read_parquet(voyage_path)
    logger.info(f"Loaded {len(df):,} voyages")
    
    # Merge logbook features if available
    logbook_path = FINAL_DIR / "voyage_logbook_features.parquet"
    if logbook_path.exists():
        logbook = pd.read_parquet(logbook_path)
        df = df.merge(logbook, on="voyage_id", how="left")
        logger.info(f"Merged logbook features: {logbook_path.name}")
    
    # Merge storm exposure if available
    storm_path = FINAL_DIR / "voyage_storm_exposure.parquet"
    if storm_path.exists():
        storm = pd.read_parquet(storm_path)
        df = df.merge(storm, on="voyage_id", how="left")
        logger.info(f"Merged storm exposure: {storm_path.name}")
    
    # Merge network features if available
    network_path = FINAL_DIR / "voyage_network_features.parquet"
    if network_path.exists():
        network = pd.read_parquet(network_path)
        df = df.merge(network, on="voyage_id", how="left")
        logger.info(f"Merged network features: {network_path.name}")
    
    return df


# =============================================================================
# Test Category 1: Route-Level Skill Isolation
# =============================================================================

def test_storm_exposure_captain_fe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Test: Do captain FEs shrink when controlling for actual storm exposure?
    
    If captains earned FE credit for avoiding storms, adding precise
    storm exposure should reduce captain effect variance.
    
    Hypothesis: Captain FE variance decreases with storm controls.
    """
    logger.info("Running: Storm Exposure × Captain FE Test")
    
    result = {
        "test_name": "storm_exposure_captain_fe",
        "hypothesis": "Captain FE variance decreases with storm exposure controls",
        "status": "not_run",
    }
    
    required_cols = ["log_output", "captain_id", "storm_encounter_count"]
    if not all(col in df.columns for col in required_cols):
        result["status"] = "skipped"
        result["reason"] = f"Missing columns: {set(required_cols) - set(df.columns)}"
        return result
    
    try:
        from src.analyses.config import run_fe_regression
        
        # Model 1: Captain FE only
        r1 = run_fe_regression(
            df, "log_output",
            controls=[],
            fe_vars=["captain_id"],
        )
        
        # Model 2: Captain FE + Storm exposure
        r2 = run_fe_regression(
            df, "log_output",
            controls=["storm_encounter_count", "days_in_hurricane_corridor"],
            fe_vars=["captain_id"],
        )
        
        result["captain_fe_variance_no_storm"] = r1.get("fe_variance", {}).get("captain_id")
        result["captain_fe_variance_with_storm"] = r2.get("fe_variance", {}).get("captain_id")
        result["variance_reduction"] = (
            (r1.get("fe_variance", {}).get("captain_id", 0) - 
             r2.get("fe_variance", {}).get("captain_id", 0)) /
            max(r1.get("fe_variance", {}).get("captain_id", 1), 0.001)
        )
        result["status"] = "completed"
        
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    return result


def test_fair_weather_captains(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Test: Identify "fair weather captains" whose high FEs are explained by luck.
    
    Compare captain rankings before/after storm exposure controls.
    Captains who drop significantly had inflated skill estimates.
    """
    logger.info("Running: Fair Weather Captains Test")
    
    result = {
        "test_name": "fair_weather_captains",
        "hypothesis": "Some high-FE captains are 'fair weather' - luck explains their success",
        "status": "not_run",
    }
    
    required_cols = ["log_output", "captain_id", "storm_encounter_count"]
    if not all(col in df.columns for col in required_cols):
        result["status"] = "skipped"
        result["reason"] = f"Missing columns: {set(required_cols) - set(df.columns)}"
        return result
    
    try:
        # Simple proxy: correlate captain avg output with avg storm exposure
        captain_stats = df.groupby("captain_id").agg({
            "log_output": "mean",
            "storm_encounter_count": "mean",
        }).dropna()
        
        correlation = captain_stats["log_output"].corr(captain_stats["storm_encounter_count"])
        
        result["output_storm_correlation"] = correlation
        result["interpretation"] = (
            "Negative correlation suggests storms hurt output" 
            if correlation < 0 else
            "Positive correlation is unexpected - may indicate survivor bias"
        )
        result["n_captains"] = len(captain_stats)
        result["status"] = "completed"
        
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    return result


def test_route_efficiency_skill(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Test: Does route efficiency predict outcomes independently of captain FE?
    
    If route efficiency (beeline/actual distance) predicts output
    after captain FE, it represents portable navigation skill.
    """
    logger.info("Running: Route Efficiency Skill Test")
    
    result = {
        "test_name": "route_efficiency_skill",
        "hypothesis": "Route efficiency predicts output beyond captain FE",
        "status": "not_run",
    }
    
    required_cols = ["log_output", "route_efficiency"]
    if not all(col in df.columns for col in required_cols):
        result["status"] = "skipped"
        result["reason"] = f"Missing columns: {set(required_cols) - set(df.columns)}"
        return result
    
    try:
        # Simple regression: output ~ route_efficiency
        valid = df[["log_output", "route_efficiency"]].dropna()
        
        from scipy import stats
        slope, intercept, r, p, se = stats.linregress(
            valid["route_efficiency"], valid["log_output"]
        )
        
        result["efficiency_coefficient"] = slope
        result["efficiency_pvalue"] = p
        result["efficiency_r_squared"] = r ** 2
        result["interpretation"] = (
            f"Route efficiency has {'significant' if p < 0.05 else 'insignificant'} "
            f"{'positive' if slope > 0 else 'negative'} effect on output"
        )
        result["status"] = "completed"
        
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    return result


# =============================================================================
# Test Category 2: Agent Selection Strategy
# =============================================================================

def test_route_assignment_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Test: Did agents assign captains with route experience?
    
    Higher match rates suggest intentional captain-route matching.
    """
    logger.info("Running: Route Assignment Quality Test")
    
    result = {
        "test_name": "route_assignment_quality",
        "hypothesis": "Agents assigned experienced captains to routes more than random",
        "status": "not_run",
    }
    
    required_cols = ["agent_id", "captain_id", "route", "year_out"]
    if not all(col in df.columns for col in required_cols):
        result["status"] = "skipped"
        result["reason"] = f"Missing columns: {set(required_cols) - set(df.columns)}"
        return result
    
    try:
        # Compute prior route experience for each voyage
        df_sorted = df.sort_values(["captain_id", "year_out"]).copy()
        
        prior_experience = []
        captain_route_history = {}
        
        for _, row in df_sorted.iterrows():
            captain = row["captain_id"]
            route = row["route"]
            
            if pd.isna(captain) or pd.isna(route):
                prior_experience.append(0)
                continue
            
            key = (captain, route)
            prior = captain_route_history.get(key, 0)
            prior_experience.append(prior)
            captain_route_history[key] = prior + 1
        
        df_sorted["prior_route_exp"] = prior_experience
        
        # Rate of experienced assignments by agent
        agent_rates = df_sorted.groupby("agent_id").apply(
            lambda x: (x["prior_route_exp"] > 0).mean()
        )
        
        result["mean_experienced_rate"] = agent_rates.mean()
        result["agent_variance"] = agent_rates.var()
        result["top_5_agents"] = agent_rates.nlargest(5).to_dict()
        result["n_agents"] = len(agent_rates)
        result["interpretation"] = (
            f"On average, {result['mean_experienced_rate']:.1%} of voyages "
            f"assigned to captains with prior route experience"
        )
        result["status"] = "completed"
        
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    return result


def test_information_advantage(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Test: Did vessels that 'spoke with' more ships perform better?
    
    Information sharing at sea may have provided competitive advantage.
    """
    logger.info("Running: Information Advantage Test")
    
    result = {
        "test_name": "information_advantage",
        "hypothesis": "Vessels with more spoke-with contacts performed better",
        "status": "not_run",
    }
    
    required_cols = ["log_output", "spoke_with_count"]
    if not all(col in df.columns for col in required_cols):
        result["status"] = "skipped"
        result["reason"] = f"Missing columns: {set(required_cols) - set(df.columns)}"
        return result
    
    try:
        valid = df[["log_output", "spoke_with_count"]].dropna()
        
        from scipy import stats
        slope, intercept, r, p, se = stats.linregress(
            valid["spoke_with_count"], valid["log_output"]
        )
        
        result["spoke_coefficient"] = slope
        result["spoke_pvalue"] = p
        result["spoke_r_squared"] = r ** 2
        result["mean_spoke_count"] = valid["spoke_with_count"].mean()
        result["interpretation"] = (
            f"Speaking with more vessels has "
            f"{'significant' if p < 0.05 else 'insignificant'} "
            f"{'positive' if slope > 0 else 'negative'} effect"
        )
        result["status"] = "completed"
        
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    return result


def test_agent_risk_portfolio(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Test: Did agents diversify fleet storm exposure?
    
    Lower exposure variance within agent suggests intentional diversification.
    """
    logger.info("Running: Agent Risk Portfolio Test")
    
    result = {
        "test_name": "agent_risk_portfolio",
        "hypothesis": "Agents with lower exposure variance had more stable returns",
        "status": "not_run",
    }
    
    required_cols = ["agent_id", "storm_encounter_count", "log_output"]
    if not all(col in df.columns for col in required_cols):
        result["status"] = "skipped"
        result["reason"] = f"Missing columns: {set(required_cols) - set(df.columns)}"
        return result
    
    try:
        agent_stats = df.groupby("agent_id").agg({
            "storm_encounter_count": "var",
            "log_output": ["mean", "var"],
        }).dropna()
        
        agent_stats.columns = ["exposure_var", "output_mean", "output_var"]
        
        from scipy import stats
        slope, _, r, p, _ = stats.linregress(
            agent_stats["exposure_var"], agent_stats["output_var"]
        )
        
        result["exposure_output_var_correlation"] = r
        result["correlation_pvalue"] = p
        result["n_agents"] = len(agent_stats)
        result["interpretation"] = (
            f"Agent exposure variance {'correlates' if p < 0.05 else 'does not correlate'} "
            f"with output variance (r={r:.3f})"
        )
        result["status"] = "completed"
        
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    return result


# =============================================================================
# Test Category 3: Behavioral/Decision Tests
# =============================================================================

def test_ground_switching(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Test: Does ground switching frequency predict outcomes?
    
    Tests explore/exploit tradeoff in hunting strategy.
    """
    logger.info("Running: Ground Switching Test")
    
    result = {
        "test_name": "ground_switching",
        "hypothesis": "Optimal ground switching frequency exists (inverted U)",
        "status": "not_run",
    }
    
    required_cols = ["log_output", "ground_switching_count"]
    if not all(col in df.columns for col in required_cols):
        result["status"] = "skipped"
        result["reason"] = f"Missing columns: {set(required_cols) - set(df.columns)}"
        return result
    
    try:
        valid = df[["log_output", "ground_switching_count"]].dropna()
        
        # Linear effect
        from scipy import stats
        slope, _, r, p, _ = stats.linregress(
            valid["ground_switching_count"], valid["log_output"]
        )
        
        result["switching_linear_effect"] = slope
        result["switching_pvalue"] = p
        result["mean_switching"] = valid["ground_switching_count"].mean()
        result["interpretation"] = (
            f"More ground switching has "
            f"{'positive' if slope > 0 else 'negative'} effect on output"
        )
        result["status"] = "completed"
        
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    return result


def test_arrival_timing(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Test: Does arriving at grounds in optimal season predict outcomes?
    """
    logger.info("Running: Arrival Timing Test")
    
    result = {
        "test_name": "arrival_timing",
        "hypothesis": "Captains arriving during optimal seasons had higher output",
        "status": "not_run",
    }
    
    required_cols = ["log_output", "arrival_timing_score"]
    if not all(col in df.columns for col in required_cols):
        result["status"] = "skipped"
        result["reason"] = f"Missing columns: {set(required_cols) - set(df.columns)}"
        return result
    
    try:
        valid = df[["log_output", "arrival_timing_score"]].dropna()
        
        from scipy import stats
        slope, _, r, p, _ = stats.linregress(
            valid["arrival_timing_score"], valid["log_output"]
        )
        
        result["timing_coefficient"] = slope
        result["timing_pvalue"] = p
        result["mean_timing_score"] = valid["arrival_timing_score"].mean()
        result["interpretation"] = (
            f"Optimal arrival timing has "
            f"{'significant' if p < 0.05 else 'insignificant'} "
            f"{'positive' if slope > 0 else 'negative'} effect"
        )
        result["status"] = "completed"
        
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    return result


# =============================================================================
# Test Category 4: TFP Enhancement
# =============================================================================

def test_route_efficiency_tfp(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Test: Add route efficiency to variance decomposition.
    
    What fraction of output variance is explained by navigation efficiency?
    """
    logger.info("Running: Route Efficiency TFP Test")
    
    result = {
        "test_name": "route_efficiency_tfp",
        "hypothesis": "Route efficiency explains additional output variance",
        "status": "not_run",
    }
    
    required_cols = ["log_output", "route_efficiency"]
    if not all(col in df.columns for col in required_cols):
        result["status"] = "skipped"
        result["reason"] = f"Missing columns: {set(required_cols) - set(df.columns)}"
        return result
    
    try:
        valid = df[["log_output", "route_efficiency"]].dropna()
        
        total_var = valid["log_output"].var()
        
        # R-squared of simple regression
        from scipy import stats
        _, _, r, p, _ = stats.linregress(
            valid["route_efficiency"], valid["log_output"]
        )
        
        variance_explained = r ** 2
        
        result["route_efficiency_r_squared"] = variance_explained
        result["variance_share"] = variance_explained
        result["total_output_variance"] = total_var
        result["interpretation"] = (
            f"Route efficiency explains {variance_explained:.1%} of output variance"
        )
        result["status"] = "completed"
        
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    return result


# =============================================================================
# Run All Tests
# =============================================================================

def run_all_logbook_tests(
    df: Optional[pd.DataFrame] = None,
    save_outputs: bool = True,
) -> Dict[str, Any]:
    """
    Run all logbook-based analyses.
    
    Parameters
    ----------
    df : pd.DataFrame, optional
        Analysis data. If None, loads from disk.
    save_outputs : bool
        Whether to save results.
        
    Returns
    -------
    Dict
        All test results.
    """
    print("\n" + "=" * 70)
    print("LOGBOOK-BASED ANALYSES")
    print("=" * 70)
    
    if df is None:
        df = load_analysis_data()
    
    print(f"\nAnalysis dataset: {len(df):,} voyages")
    print(f"Available columns: {len(df.columns)}")
    
    results = {}
    
    # Category 1: Route-Level Skill Isolation
    print("\n" + "-" * 70)
    print("CATEGORY 1: ROUTE-LEVEL SKILL ISOLATION")
    print("-" * 70)
    
    results["storm_exposure_captain_fe"] = test_storm_exposure_captain_fe(df)
    results["fair_weather_captains"] = test_fair_weather_captains(df)
    results["route_efficiency_skill"] = test_route_efficiency_skill(df)
    
    # Category 2: Agent Selection Strategy  
    print("\n" + "-" * 70)
    print("CATEGORY 2: AGENT SELECTION STRATEGY")
    print("-" * 70)
    
    results["route_assignment_quality"] = test_route_assignment_quality(df)
    results["information_advantage"] = test_information_advantage(df)
    results["agent_risk_portfolio"] = test_agent_risk_portfolio(df)
    
    # Category 3: Behavioral/Decision Tests
    print("\n" + "-" * 70)
    print("CATEGORY 3: BEHAVIORAL/DECISION TESTS")
    print("-" * 70)
    
    results["ground_switching"] = test_ground_switching(df)
    results["arrival_timing"] = test_arrival_timing(df)
    
    # Category 4: TFP Enhancement
    print("\n" + "-" * 70)
    print("CATEGORY 4: TFP ENHANCEMENT")
    print("-" * 70)
    
    results["route_efficiency_tfp"] = test_route_efficiency_tfp(df)
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    completed = sum(1 for r in results.values() if r["status"] == "completed")
    skipped = sum(1 for r in results.values() if r["status"] == "skipped")
    errors = sum(1 for r in results.values() if r["status"] == "error")
    
    print(f"\nCompleted: {completed}")
    print(f"Skipped: {skipped}")
    print(f"Errors: {errors}")
    
    # Print key findings
    print("\nKEY FINDINGS:")
    for name, r in results.items():
        if r["status"] == "completed" and "interpretation" in r:
            print(f"  {name}: {r['interpretation']}")
    
    # Save outputs
    if save_outputs:
        output_path = OUTPUT_DIR / "logbook_analyses_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=convert)
        
        print(f"\nResults saved to: {output_path}")
    
    return results


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run logbook-based analyses"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run without saving outputs"
    )
    
    args = parser.parse_args()
    
    results = run_all_logbook_tests(save_outputs=not args.dry_run)
