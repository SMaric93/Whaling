#!/usr/bin/env python
"""
Master orchestration script for whaling empirical analysis.

Runs all R1-R17 regression specifications and generates output exhibits.

Usage:
    python -m src.analyses.run_all           # Run all analyses
    python -m src.analyses.run_all --dry-run # Validate without saving
    python -m src.analyses.run_all --quick   # Run main text only (R1,R2,R3,R6,R9,R13,R14)
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def run_all_analyses(
    quick: bool = False,
    dry_run: bool = False,
    save_outputs: bool = True,
) -> Dict:
    """
    Run all regression analyses.
    
    Parameters
    ----------
    quick : bool
        If True, only run main text regressions.
    dry_run : bool
        If True, validate but don't save outputs.
    save_outputs : bool
        Whether to save output files.
        
    Returns
    -------
    Dict
        All regression results.
    """
    from .data_loader import prepare_analysis_sample
    from .connected_set import full_connected_set_analysis
    from .baseline_production import run_r1
    from .portability import run_portability_analysis
    from .event_study import run_event_study_analysis
    from .complementarity import run_complementarity_analysis
    from .shock_analysis import run_shock_analysis
    from .strategy import run_strategy_analysis
    from .labor_market import run_labor_market_analysis
    from .extensions import run_extensions
    from .output_generator import generate_all_outputs
    
    print("\n" + "=" * 70)
    print("WHALING EMPIRICAL ANALYSIS: FULL REGRESSION SUITE")
    print("=" * 70)
    
    if dry_run:
        save_outputs = False
        print("\n[DRY RUN MODE - No outputs will be saved]")
    
    if quick:
        print("\n[QUICK MODE - Main text regressions only]")
    
    start_time = time.time()
    all_results = {}
    
    # =========================================================================
    # Step 1: Load and prepare data
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 1: PREPARING ANALYSIS SAMPLE")
    print("-" * 70)
    
    df = prepare_analysis_sample(use_climate_data=True)
    
    # =========================================================================
    # Step 2: Connected set analysis
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 2: CONNECTED SET ANALYSIS")
    print("-" * 70)
    
    df_cc, df_loo, diagnostics = full_connected_set_analysis(df)
    all_results["diagnostics"] = diagnostics
    
    # Use the LOO connected set for all AKM-style regressions
    # This is CRITICAL for proper identification of fixed effects
    print(f"\n*** Using LOO connected set: {len(df_loo):,} voyages ***")
    
    # =========================================================================
    # Step 3: R1 - Baseline production function
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 3: R1 - BASELINE PRODUCTION FUNCTION")
    print("-" * 70)
    
    r1_results = run_r1(df_loo, save_outputs=save_outputs)
    all_results["r1"] = r1_results
    
    # =========================================================================
    # Step 4: R2, R4 - Portability validation
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 4: R2, R4 - PORTABILITY VALIDATION")
    print("-" * 70)
    
    portability_results = run_portability_analysis(df_loo, save_outputs=save_outputs)
    all_results["r2"] = portability_results["r2"]
    all_results["r4"] = portability_results["r4"]
    
    # =========================================================================
    # Step 5: R3 - Event study
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 5: R3 - EVENT STUDY")
    print("-" * 70)
    
    r3_results = run_event_study_analysis(df_loo, save_outputs=save_outputs)
    all_results["r3"] = r3_results
    
    # =========================================================================
    # Step 6: R5, R6 - Complementarity and resilience
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 6: R5, R6 - COMPLEMENTARITY AND RESILIENCE")
    print("-" * 70)
    
    comp_results = run_complementarity_analysis(df_loo, save_outputs=save_outputs)
    all_results["r5"] = comp_results["r5"]
    all_results["r6"] = comp_results["r6"]
    
    if not quick:
        # =====================================================================
        # Step 7: R7-R9 - Climate shock analysis
        # =====================================================================
        print("\n" + "-" * 70)
        print("STEP 7: R7-R9 - CLIMATE SHOCK ANALYSIS")
        print("-" * 70)
        
        shock_results = run_shock_analysis(df_loo, save_outputs=save_outputs)
        all_results["r7"] = shock_results["r7"]
        all_results["r8"] = shock_results["r8"]
        all_results["r9"] = shock_results["r9"]
        
        # =====================================================================
        # Step 8: R10-R12 - Strategy choice
        # =====================================================================
        print("\n" + "-" * 70)
        print("STEP 8: R10-R12 - STRATEGY CHOICE")
        print("-" * 70)
        
        strategy_results = run_strategy_analysis(df_loo, save_outputs=save_outputs)
        all_results["r10"] = strategy_results["r10"]
        all_results["r11"] = strategy_results["r11"]
        all_results["r12"] = strategy_results["r12"]
    
    # =========================================================================
    # Step 9: R13-R15 - Labor market dynamics
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 9: R13-R15 - LABOR MARKET DYNAMICS")
    print("-" * 70)
    
    labor_results = run_labor_market_analysis(df_loo, save_outputs=save_outputs)
    all_results["r13"] = labor_results["r13"]
    all_results["r14"] = labor_results["r14"]
    all_results["r15"] = labor_results["r15"]
    
    # =========================================================================
    # Step 10: Mechanism Analysis (Crew-Level)
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 10: MECHANISM ANALYSIS (CREW-LEVEL)")
    print("-" * 70)
    
    try:
        from .mechanism_crew import run_full_mechanism_analysis
        mechanism_results = run_full_mechanism_analysis(df_loo, save_outputs=save_outputs)
        all_results["mechanism"] = mechanism_results
    except Exception as e:
        print(f"Warning: Mechanism analysis failed: {e}")
        all_results["mechanism"] = {"error": str(e)}
    
    # =========================================================================
    # Step 11: Additional Robustness Tests (Killer Tests)
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 11: ADDITIONAL ROBUSTNESS TESTS")
    print("-" * 70)
    
    # 11a: Vessel Mover Design
    try:
        from .vessel_mover_analysis import run_vessel_mover_analysis
        vessel_mover_results = run_vessel_mover_analysis(df_loo, save_outputs=save_outputs)
        all_results["vessel_mover"] = vessel_mover_results
        print("  ✓ Vessel Mover Design complete")
    except Exception as e:
        print(f"  ⚠ Vessel Mover Design failed: {e}")
        all_results["vessel_mover"] = {"error": str(e)}
    
    # 11b: Insurance Variance Validation
    try:
        from .insurance_variance_test import run_insurance_variance_tests
        insurance_results = run_insurance_variance_tests(df_loo, save_outputs=save_outputs)
        all_results["insurance_variance"] = insurance_results
        print("  ✓ Insurance Variance Validation complete")
    except Exception as e:
        print(f"  ⚠ Insurance Variance Validation failed: {e}")
        all_results["insurance_variance"] = {"error": str(e)}
    
    # 11c: Optimal Foraging Stopping Rule
    try:
        from .search_theory import run_stopping_rule_analysis
        stopping_rule_results = run_stopping_rule_analysis(df_loo, save_outputs=save_outputs)
        all_results["stopping_rule"] = stopping_rule_results
        print("  ✓ Stopping Rule Analysis complete")
    except Exception as e:
        print(f"  ⚠ Stopping Rule Analysis failed: {e}")
        all_results["stopping_rule"] = {"error": str(e)}
    
    # =========================================================================
    # Step 12: Compass Regressions (C1-C6)
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 12: C1-C6 - COMPASS REGRESSIONS")
    print("-" * 70)
    
    try:
        from .compass_regressions import run_compass_regressions
        compass_results = run_compass_regressions(df_loo, save_outputs=save_outputs)
        all_results["compass"] = compass_results
        print("  ✓ Compass regressions complete")
    except Exception as e:
        print(f"  ⚠ Compass regressions failed: {e}")
        all_results["compass"] = {"error": str(e)}
    
    if not quick:
        # =====================================================================
        # Step 13: R16-R17 - Optional extensions
        # =====================================================================
        print("\n" + "-" * 70)
        print("STEP 13: R16-R17 - OPTIONAL EXTENSIONS")
        print("-" * 70)
        
        ext_results = run_extensions(df_loo, save_outputs=save_outputs)
        all_results["r16"] = ext_results["r16"]
        all_results["r17"] = ext_results["r17"]
    
    # =========================================================================
    # Step 13: Generate output exhibits
    # =========================================================================
    if save_outputs:
        print("\n" + "-" * 70)
        print("STEP 14: GENERATING OUTPUT EXHIBITS")
        print("-" * 70)
        
        generate_all_outputs(all_results, diagnostics)
    
    # =========================================================================
    # Summary
    # =========================================================================
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nTotal time: {elapsed:.1f} seconds")
    print(f"Regressions run: {len([k for k in all_results if k.startswith('r')])}")
    
    if save_outputs:
        from .config import OUTPUT_DIR
        print(f"\nOutputs saved to: {OUTPUT_DIR}")
    
    return all_results


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Run whaling empirical analysis suite"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run main text regressions only (R1,R2,R3,R6,R13,R14)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate without saving outputs"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run analyses but don't save outputs"
    )
    
    args = parser.parse_args()
    
    results = run_all_analyses(
        quick=args.quick,
        dry_run=args.dry_run,
        save_outputs=not args.no_save and not args.dry_run,
    )
    
    return results


if __name__ == "__main__":
    main()
