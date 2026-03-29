"""
Master Orchestration Script for Whaling Connectivity Reruns
Executes all validation phases (0 to 4) sequentially.
"""

import sys
import time
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Ensure python path is set correctly
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def execute_script(module_name: str, function_name: str):
    """Dynamically imports a script and executes its main routine."""
    logger.info(f"\n{'='*70}\n[RUNNING]: {module_name}.{function_name}()\n{'='*70}")
    t0 = time.time()
    try:
        module = __import__(f"src.post_connectivity.{module_name}", fromlist=[function_name])
        func = getattr(module, function_name)
        func()
        logger.info(f"SUCCESS: {module_name} finished in {time.time() - t0:.2f} seconds.")
    except Exception as e:
        logger.error(f"FAILURE: {module_name} crashed: {e}")
        raise

def main():
    logger.info("Initializing Full Pipeline Rebuild (Connectivity -> Econometrics)")
    
    # PHASE 0: Connectivity Audit
    execute_script("connectivity_audit", "run_audit")
    
    # PHASE 1: Re-estimate authoritative theta and psi
    execute_script("rerun_types", "run_type_estimation")
    
    # PHASE 2: Reconcile Old vs New Tables
    execute_script("reconcile_old_vs_new", "run_reconciliation")
    
    # PHASE 3: Debug unstable modules
    execute_script("switch_event_study", "run_event_study_debug")
    execute_script("stopping_hazard", "run_stopping_hazard")
    execute_script("ml_ablation_audit", "run_ml_audit")
    execute_script("audit_assignment", "run_assignment_audit")
    
    # PHASE 4: Authoritative Reruns
    execute_script("route_choice_hierarchy", "run_route_choice")
    execute_script("vessel_mover_power", "run_vessel_mover_power")
    execute_script("production_surface_authoritative", "run_production_surface")
    execute_script("floor_raising_authoritative", "run_insurance_tests")
    execute_script("matching_welfare_authoritative", "run_matching_welfare")
    execute_script("offpolicy_diagnostics", "run_offpolicy_diagnostics")
    execute_script("mechanism_crew_network", "run_mechanisms")

    logger.info(f"\n{'='*70}\nALL PHASES COMPLETED SUCCESSFULLY\n{'='*70}")

if __name__ == "__main__":
    main()
