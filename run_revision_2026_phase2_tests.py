#!/usr/bin/env python
"""
Revision 2026 Phase 2 — Master Entry Point
============================================

    python run_revision_2026_phase2_tests.py                 # all steps
    python run_revision_2026_phase2_tests.py --steps 1 2 4   # specific steps
    python run_revision_2026_phase2_tests.py --dry-run        # verify imports
    python run_revision_2026_phase2_tests.py --skip-bootstrap # skip slow bootstrap

Priority order: 1 → 2/3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 14/15
"""
from __future__ import annotations
import argparse, json, logging, sys, time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.revision_2026_phase2.config import CFG, OUTPUT_BASE, TABLES_DIR, LOGS_DIR

log_path = LOGS_DIR / "phase2_run.log"
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.FileHandler(log_path, mode="a"), logging.StreamHandler()])
logger = logging.getLogger("phase2")

ALL_STEPS = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15]

def run_step(step, skip_bootstrap=False):
    t0 = time.time()
    status, error = "success", ""
    try:
        if step == 1:
            from src.revision_2026_phase2.step01_sample_lineage import build_sample_lineage
            build_sample_lineage()
        elif step == 2:
            from src.revision_2026_phase2.step02_03_proxies import build_all_proxies
            build_all_proxies()
        elif step == 4:
            from src.revision_2026_phase2.step04_zero_margin import run_zero_margin_tests
            run_zero_margin_tests()
        elif step == 5:
            from src.revision_2026_phase2.step05_06_07_inference import run_bootstrap_quantiles
            if skip_bootstrap:
                logger.info("Skipping bootstrap (--skip-bootstrap)")
            else:
                run_bootstrap_quantiles()
        elif step == 6:
            from src.revision_2026_phase2.step05_06_07_inference import run_location_vs_scale
            run_location_vs_scale()
        elif step == 7:
            from src.revision_2026_phase2.step05_06_07_inference import run_skill_heterogeneity
            run_skill_heterogeneity()
        elif step == 8:
            from src.revision_2026_phase2.step08_movers import run_all_mover_tests
            run_all_mover_tests()
        elif step == 9:
            from src.revision_2026_phase2.step09_10_11_robustness import run_connected_vs_broad_robustness
            run_connected_vs_broad_robustness()
        elif step == 10:
            from src.revision_2026_phase2.step09_10_11_robustness import run_trimming_sensitivity
            run_trimming_sensitivity()
        elif step == 11:
            from src.revision_2026_phase2.step09_10_11_robustness import run_pipeline_diagnostics
            run_pipeline_diagnostics()
        elif step == 14:
            from src.revision_2026_phase2.step14_15_qa_summary import generate_results_summary
            generate_results_summary()
        elif step == 15:
            from src.revision_2026_phase2.step14_15_qa_summary import run_qa_checks
            run_qa_checks()
        else:
            status = "skipped"
    except Exception as e:
        status, error = "failed", str(e)
        logger.exception("Step %d failed: %s", step, e)
    return dict(step=step, status=status, elapsed=round(time.time()-t0,1), error=error)

def main():
    parser = argparse.ArgumentParser(description="Revision 2026 Phase 2 tests")
    parser.add_argument("--steps", nargs="*", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-bootstrap", action="store_true")
    args = parser.parse_args()
    steps = args.steps or ALL_STEPS

    logger.info("="*70)
    logger.info("REVISION 2026 PHASE 2")
    logger.info("="*70)
    logger.info("Steps: %s", steps)

    if args.dry_run:
        try:
            from src.revision_2026_phase2 import (step01_sample_lineage, step02_03_proxies,
                step04_zero_margin, step05_06_07_inference, step08_movers,
                step09_10_11_robustness, step14_15_qa_summary)
            logger.info("✓ All imports OK")
        except ImportError as e:
            logger.error("Import failed: %s", e); sys.exit(1)
        return

    results = []
    for s in steps:
        logger.info("\n"+"—"*60+f"\nRunning Step {s}...")
        r = run_step(s, skip_bootstrap=args.skip_bootstrap)
        results.append(r)
        logger.info("Step %d: %s (%.1fs)", s, r["status"], r["elapsed"])

    # Manifest
    manifest = dict(timestamp=datetime.now().isoformat(), config=vars(CFG),
                    steps=results, output_dir=str(OUTPUT_BASE))
    with open(OUTPUT_BASE/"manifest.json","w") as f:
        json.dump(manifest, f, indent=2, default=str)

    # README
    readme_lines = ["# Phase 2 Outputs","",
        "## Reused from Phase 1 / core package",
        "- `psi_connected_loo` (exact LOO AKM on connected set)",
        "- `theta_sep_main` (EB-shrunk captain skill)",
        "- Data loader, connected-set builder, output schema",
        "",
        "## New wrappers added",
        "- `psi_broad_resid_loo`: residualized LOO agent mean (broad positive sample)",
        "- `psi_broad_preperiod`: pre-period agent mean (broad positive sample)", 
        "- `psi_broad_success_loo`: LOO agent positive-catch rate (broad with-zeros)",
        "- `skill_experience_proxy`: log(prior voyages) + years since first command",
        "- Captain-cluster bootstrap quantile regressions (499 draws)",
        "- B-spline continuous skill heterogeneity tests",
        "- Placebo event study (shifted dates)",
        "",
        "## Implemented from scratch",
        "- Broad-sample extensive-margin zero-catch tests",
        "- Location vs scale decomposition with bootstrap variance ratios",
        "- Trimming/winsor/support sensitivity battery",
        "- Selection and sorting diagnostics for pipeline claim",
        "- Auto-generated claim triage",
    ]
    (OUTPUT_BASE/"README_results.md").write_text("\n".join(readme_lines)+"\n")

    n_fail = sum(1 for r in results if r["status"]=="failed")
    logger.info("\n"+"="*70)
    logger.info("COMPLETE: %d/%d succeeded in %.0fs", len(results)-n_fail, len(results),
                sum(r["elapsed"] for r in results))
    if n_fail: sys.exit(1)

if __name__ == "__main__":
    main()
