"""
Stage 4: Full Analyses

Runs the complete empirical analysis suite including:
    - R1-R17 regression specifications
    - Variance decomposition (AKM with KSS correction)
    - Complementarity analysis (θ × ψ interactions)
    - Event studies (captain switching)
    - Portability tests
    - Shock analysis
    - Counterfactuals (PAM vs AAM matching)
    - Robustness tests
    - Mechanism tests
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Output directories
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'output'
BASELINE_DIR = OUTPUT_DIR / 'baseline_loo_eb'
PAPER_DIR = OUTPUT_DIR / 'paper'


def run_baseline_akm() -> dict:
    """
    Run baseline AKM variance decomposition (R1-R3).
    
    Specifications:
        R1: Captain + Agent + Vessel×Period + Route×Time
        R2: Agent only
        R3: Captain only
    """
    from src.analyses.run_full_baseline_loo_eb import run_r1_baseline
    from src.analyses.data_loader import load_analysis_data
    from src.analyses.connected_set import get_loo_connected_set
    
    logger.info("Running baseline AKM (R1-R3)...")
    
    df = load_analysis_data()
    df_loo = get_loo_connected_set(df)
    results = run_r1_baseline(df_loo)
    
    logger.info(f"  → LOO connected set: {len(df_loo):,} voyages")
    return results


def run_complementarity_analysis() -> dict:
    """Run complementarity analysis: θ × ψ interactions by ground type."""
    from src.analyses.run_full_baseline_loo_eb import run_complementarity
    from src.analyses.data_loader import load_analysis_data
    from src.analyses.connected_set import get_loo_connected_set
    
    logger.info("Running complementarity analysis...")
    
    df = load_analysis_data()
    df_loo = get_loo_connected_set(df)
    
    # Need AKM results first
    from src.analyses.run_full_baseline_loo_eb import run_akm_with_eb
    akm_results = run_akm_with_eb(df_loo)
    
    results = run_complementarity(df_loo, akm_results)
    return results


def run_event_studies() -> dict:
    """Run event study for captain switching agents."""
    from src.analyses.event_study import run_event_study_analysis
    from src.analyses.data_loader import load_analysis_data
    
    logger.info("Running event studies...")
    
    df = load_analysis_data()
    results = run_event_study_analysis(df)
    return results


def run_portability_tests() -> dict:
    """Run skill portability tests (R4-R5)."""
    from src.analyses.portability import run_portability_analysis
    from src.analyses.data_loader import load_analysis_data
    
    logger.info("Running portability tests...")
    
    df = load_analysis_data()
    results = run_portability_analysis(df)
    return results


def run_shock_analysis() -> dict:
    """Run climate shock pass-through analysis (R13-R15)."""
    from src.analyses.shock_analysis import run_shock_pass_through
    from src.analyses.data_loader import load_analysis_data
    
    logger.info("Running shock analysis...")
    
    df = load_analysis_data()
    results = run_shock_pass_through(df)
    return results


def run_strategy_analysis() -> dict:
    """Run route and ground choice analysis (R16-R17)."""
    from src.analyses.strategy import run_strategy_analysis
    from src.analyses.data_loader import load_analysis_data
    
    logger.info("Running strategy analysis...")
    
    df = load_analysis_data()
    results = run_strategy_analysis(df)
    return results


def run_counterfactuals() -> dict:
    """Run counterfactual simulations (PAM vs AAM matching)."""
    from src.analyses.counterfactual_suite import run_full_counterfactual_suite
    
    logger.info("Running counterfactual simulations...")
    
    results = run_full_counterfactual_suite(save_outputs=True)
    return results


def run_robustness_tests() -> dict:
    """
    Run robustness tests:
        - Vessel mover analysis
        - Insurance variance validation
        - Optimal foraging stopping rule
    """
    logger.info("Running robustness tests...")
    
    results = {}
    
    # Vessel mover design
    try:
        from src.analyses.vessel_mover_analysis import run_vessel_mover_robustness
        results['vessel_mover'] = run_vessel_mover_robustness()
        logger.info("  - Vessel mover analysis complete")
    except Exception as e:
        logger.warning(f"  - Vessel mover analysis skipped: {e}")
    
    # Insurance variance test
    try:
        from src.analyses.insurance_variance_test import run_insurance_variances
        results['insurance_variance'] = run_insurance_variances()
        logger.info("  - Insurance variance test complete")
    except Exception as e:
        logger.warning(f"  - Insurance variance test skipped: {e}")
    
    return results


def run_mechanism_tests() -> dict:
    """
    Run mechanism tests:
        - Weather allocation (high-ψ agents and weather shocks)
        - Crew mechanism (hiring, retention, discipline)
        - Context-dependent matching
    """
    logger.info("Running mechanism tests...")
    
    results = {}
    
    # Crew mechanism
    try:
        from src.analyses.mechanism_crew import run_crew_mechanism_analysis
        results['crew'] = run_crew_mechanism_analysis()
        logger.info("  - Crew mechanism analysis complete")
    except Exception as e:
        logger.warning(f"  - Crew mechanism analysis skipped: {e}")
    
    # Weather allocation
    try:
        from src.analyses.weather_regressions import run_weather_regressions
        results['weather'] = run_weather_regressions()
        logger.info("  - Weather allocation analysis complete")
    except Exception as e:
        logger.warning(f"  - Weather allocation analysis skipped: {e}")
    
    return results


def run_search_theory() -> dict:
    """Run search theory analysis (Lévy flight, patch residence)."""
    logger.info("Running search theory analysis...")
    
    try:
        from src.analyses.search_theory import run_search_theory_analysis
        results = run_search_theory_analysis()
        return results
    except Exception as e:
        logger.warning(f"Search theory analysis skipped: {e}")
        return {}


def run_full_baseline_suite() -> dict:
    """Run the comprehensive baseline LOO+EB suite."""
    from src.analyses.run_full_baseline_loo_eb import run_full_suite
    
    logger.info("Running full baseline LOO+EB suite...")
    results = run_full_suite()
    return results


def run_analyze(quick: bool = False, run_all_tests: bool = True) -> dict:
    """
    Run the complete analysis stage.
    
    Args:
        quick: Run only main text regressions (R1-R5)
        run_all_tests: Include robustness and mechanism tests
    
    Returns:
        dict: Summary of all analysis results
    """
    logger.info("=" * 60)
    logger.info("STAGE 4: FULL ANALYSES")
    logger.info("=" * 60)
    
    results = {}
    
    # Core analyses - run the full baseline suite
    try:
        results['baseline'] = run_full_baseline_suite()
        logger.info("Baseline LOO+EB suite complete")
    except Exception as e:
        logger.error(f"Baseline suite failed: {e}")
        
        # Fall back to individual analyses
        try:
            results['akm'] = run_baseline_akm()
        except Exception as e2:
            logger.error(f"AKM failed: {e2}")
        
        try:
            results['complementarity'] = run_complementarity_analysis()
        except Exception as e2:
            logger.error(f"Complementarity failed: {e2}")
    
    if quick:
        logger.info("Quick mode - skipping extended analyses")
        return results
    
    # Extended analyses
    try:
        results['event_studies'] = run_event_studies()
    except Exception as e:
        logger.warning(f"Event studies skipped: {e}")
    
    try:
        results['portability'] = run_portability_tests()
    except Exception as e:
        logger.warning(f"Portability tests skipped: {e}")
    
    try:
        results['counterfactuals'] = run_counterfactuals()
    except Exception as e:
        logger.warning(f"Counterfactuals skipped: {e}")
    
    try:
        results['search_theory'] = run_search_theory()
    except Exception as e:
        logger.warning(f"Search theory skipped: {e}")
    
    # Robustness and mechanism tests
    if run_all_tests:
        try:
            results['robustness'] = run_robustness_tests()
        except Exception as e:
            logger.warning(f"Robustness tests skipped: {e}")
        
        try:
            results['mechanism'] = run_mechanism_tests()
        except Exception as e:
            logger.warning(f"Mechanism tests skipped: {e}")
    
    # Summary
    success_count = len([k for k, v in results.items() if v])
    logger.info(f"Stage 4 complete: {success_count} analysis categories completed")
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_analyze()
