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

from src.pipeline._runner import StepSpec, run_step, run_steps

logger = logging.getLogger(__name__)


def run_baseline_akm() -> dict:
    """
    Run baseline AKM variance decomposition (R1-R3).
    
    Specifications:
        R1: Captain + Agent + Vessel×Period + Route×Time
        R2: Agent only
        R3: Captain only
    """
    from src.analyses.run_full_baseline_loo_eb import get_loo_connected_set, run_r1_baseline
    from src.analyses.data_loader import load_analysis_data
    
    logger.info("Running baseline AKM (R1-R3)...")
    
    df = load_analysis_data()
    df_loo, _ = get_loo_connected_set(df)
    results = run_r1_baseline(df_loo)
    
    logger.info(f"  → LOO connected set: {len(df_loo):,} voyages")
    return results


def run_complementarity_analysis() -> dict:
    """Run complementarity analysis: θ × ψ interactions by ground type."""
    from src.analyses.run_full_baseline_loo_eb import (
        get_loo_connected_set,
        run_akm_with_eb,
        run_complementarity,
    )
    from src.analyses.data_loader import load_analysis_data
    
    logger.info("Running complementarity analysis...")
    
    df = load_analysis_data()
    df_loo, _ = get_loo_connected_set(df)
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

    from src.analyses.vessel_mover_analysis import run_vessel_mover_robustness
    from src.analyses.insurance_variance_test import run_insurance_variances

    run_steps(
        results,
        [
            StepSpec(
                'vessel_mover',
                run_vessel_mover_robustness,
                "  - Vessel mover analysis skipped",
                success_message="  - Vessel mover analysis complete",
            ),
            StepSpec(
                'insurance_variance',
                run_insurance_variances,
                "  - Insurance variance test skipped",
                success_message="  - Insurance variance test complete",
            ),
        ],
        logger=logger,
    )

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

    from src.analyses.mechanism_crew import run_crew_mechanism_analysis
    from src.analyses.weather_regressions import run_weather_regressions

    run_steps(
        results,
        [
            StepSpec(
                'crew',
                run_crew_mechanism_analysis,
                "  - Crew mechanism analysis skipped",
                success_message="  - Crew mechanism analysis complete",
            ),
            StepSpec(
                'weather',
                run_weather_regressions,
                "  - Weather allocation analysis skipped",
                success_message="  - Weather allocation analysis complete",
            ),
        ],
        logger=logger,
    )

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

    baseline_ok = run_step(
        results,
        StepSpec(
            'baseline',
            run_full_baseline_suite,
            "Baseline suite failed",
            failure_level="error",
            success_message="Baseline LOO+EB suite complete",
        ),
        logger=logger,
    )
    if not baseline_ok:
        run_steps(
            results,
            [
                StepSpec('akm', run_baseline_akm, "AKM failed", failure_level="error"),
                StepSpec('complementarity', run_complementarity_analysis, "Complementarity failed", failure_level="error"),
            ],
            logger=logger,
        )

    if quick:
        logger.info("Quick mode - skipping extended analyses")
        return results

    run_steps(
        results,
        [
            StepSpec('event_studies', run_event_studies, "Event studies skipped"),
            StepSpec('portability', run_portability_tests, "Portability tests skipped"),
            StepSpec('counterfactuals', run_counterfactuals, "Counterfactuals skipped"),
            StepSpec('search_theory', run_search_theory, "Search theory skipped"),
        ],
        logger=logger,
    )

    # Robustness and mechanism tests
    if run_all_tests:
        run_steps(
            results,
            [
                StepSpec('robustness', run_robustness_tests, "Robustness tests skipped"),
                StepSpec('mechanism', run_mechanism_tests, "Mechanism tests skipped"),
            ],
            logger=logger,
        )

    # Summary
    success_count = len([k for k, v in results.items() if v])
    logger.info(f"Stage 4 complete: {success_count} analysis categories completed")
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_analyze()
