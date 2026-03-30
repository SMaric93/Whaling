"""
Next-Round Strengthening Tests — Master Runner.

Executes repairs first, then main-text tests, then appendix tests.
Same threading-safe environment as src/ml/run_all.py.
"""

from __future__ import annotations

import logging
import os
import sys
import time
import traceback

# ── Threading safety (MUST be before any numpy/sklearn imports) ───────
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("next_round.runner")

# ═══════════════════════════════════════════════════════════════════════
# Phase Registry
# ═══════════════════════════════════════════════════════════════════════

REPAIR_PHASES = {
    "R1": ("Ablation Feature Audit", "src.next_round.repairs.ablation_feature_audit", "run_ablation_feature_audit"),
    "R2": ("Lower-Tail Target Audit", "src.next_round.repairs.lower_tail_target_audit", "run_lower_tail_audit"),
    "R3": ("Destination Ontology", "src.next_round.repairs.destination_ontology", "build_destination_ontology"),
    "R4": ("LOO Ground Quality", "src.next_round.repairs.ground_quality_loo", "build_ground_quality_loo"),
}

MAIN_TEXT_PHASES = {
    "T1": ("Hierarchical Map Choice", "src.next_round.hierarchical_map_choice", "run_hierarchical_map_choice"),
    "T2": ("Switch Policy Change", "src.next_round.switch_policy_change", "run_switch_policy_change"),
    "T3": ("State Transition Governance", "src.next_round.state_transition_governance", "run_state_transition_governance"),
    "T4": ("Exit Value Eval", "src.next_round.exit_value_eval", "run_exit_value_eval"),
    "T5": ("Search vs Execution", "src.next_round.search_vs_execution", "run_search_vs_execution"),
    "T6": ("Lower-Tail Repair", "src.next_round.lower_tail_repair", "run_lower_tail_repair"),
    "T7": ("Tail Submodularity", "src.next_round.tail_submodularity", "run_tail_submodularity"),
    "T8": ("Risk Matching", "src.next_round.risk_matching", "run_risk_matching"),
}

APPENDIX_PHASES = {
    "T9": ("Rational Exit Tests", "src.next_round.rational_exit_tests", "run_rational_exit_tests"),
    "T10": ("Policy Entropy", "src.next_round.policy_entropy", "run_policy_entropy"),
    "T11": ("Info vs Routine", "src.next_round.info_vs_routine", "run_info_vs_routine"),
    "T12": ("Hardware/Staffing Placebos", "src.next_round.hardware_staffing_placebos", "run_hardware_staffing_placebos"),
    "T13": ("Portability Tests", "src.next_round.portability_tests", "run_portability_tests"),
    "T14": ("Mediation Floor-Raising", "src.next_round.mediation_floor_raising", "run_mediation_floor_raising"),
    "T15": ("Archival Mechanisms", "src.next_round.archival_mechanisms", "run_archival_mechanisms"),
}


def run_phase(phase_id: str, name: str, module_path: str, func_name: str) -> dict:
    """Run a single phase and catch errors."""
    logger.info("━" * 60)
    logger.info("Phase %s: %s", phase_id, name)
    logger.info("━" * 60)
    t0 = time.time()

    try:
        import importlib
        mod = importlib.import_module(module_path)
        func = getattr(mod, func_name)
        result = func(save_outputs=True) if "save_outputs" in func.__code__.co_varnames else func(save=True)
        elapsed = time.time() - t0
        logger.info("✓ Phase %s completed in %.1fs", phase_id, elapsed)
        return {"status": "ok", "elapsed": elapsed, "result": result}
    except Exception as e:
        elapsed = time.time() - t0
        logger.error("✗ Phase %s FAILED after %.1fs: %s", phase_id, elapsed, e)
        traceback.print_exc()
        return {"status": "error", "elapsed": elapsed, "error": str(e)}


def run_all(*, phases: str = "all"):
    """
    Run all next-round phases.

    Parameters
    ----------
    phases : str
        'all', 'repair', 'main', 'appendix', or comma-separated IDs like 'R1,T3,T7'
    """
    logger.info("=" * 60)
    logger.info("Next-Round Strengthening Tests — Master Runner")
    logger.info("=" * 60)

    # Build phase list
    if phases == "all":
        phase_list = {**REPAIR_PHASES, **MAIN_TEXT_PHASES, **APPENDIX_PHASES}
    elif phases == "repair":
        phase_list = REPAIR_PHASES
    elif phases == "main":
        phase_list = MAIN_TEXT_PHASES
    elif phases == "appendix":
        phase_list = APPENDIX_PHASES
    else:
        ids = [p.strip() for p in phases.split(",")]
        all_phases = {**REPAIR_PHASES, **MAIN_TEXT_PHASES, **APPENDIX_PHASES}
        phase_list = {k: v for k, v in all_phases.items() if k in ids}

    results = {}
    t_total = time.time()

    for phase_id, (name, module_path, func_name) in phase_list.items():
        results[phase_id] = run_phase(phase_id, name, module_path, func_name)

    elapsed_total = time.time() - t_total

    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY — %.1fs total", elapsed_total)
    logger.info("=" * 60)

    ok = sum(1 for r in results.values() if r["status"] == "ok")
    fail = sum(1 for r in results.values() if r["status"] == "error")
    logger.info("  Passed: %d / %d", ok, len(results))
    logger.info("  Failed: %d / %d", fail, len(results))

    for pid, r in results.items():
        status = "✓" if r["status"] == "ok" else "✗"
        logger.info("  %s %s  (%.1fs)", status, pid, r["elapsed"])

    return results


if __name__ == "__main__":
    phase_arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    run_all(phases=phase_arg)
