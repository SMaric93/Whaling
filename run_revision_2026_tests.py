#!/usr/bin/env python
"""
Revision 2026 — Master Entry Point
===================================

Single command to run all revision tests:

    python run_revision_2026_tests.py              # run all steps
    python run_revision_2026_tests.py --steps 2 6  # run specific steps
    python run_revision_2026_tests.py --dry-run     # verify setup only

Steps:
  2  Sample crosswalk and audit
  3  Connected-set representativeness
  4  Separate-sample effect construction (anti-leakage psi_hat, theta_hat)
  5  Mover and event-study tests (Lévy μ outcome)
  6  Floor-raising tests (zero-catch, positive output, quantile, variance)
  7  Officer pipeline and portability
  8  Route prediction (optional)
 10  QA validation checks

All outputs are saved to outputs/revision_2026/.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.revision_2026.config import CFG, OUTPUT_BASE, TABLES_DIR, LOGS_DIR

# Set up logging
log_path = LOGS_DIR / "revision_2026_run.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(log_path, mode="a"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("revision_2026")


ALL_STEPS = [2, 3, 4, 5, 6, 7, 8, 10]
# Priority order: effects first (4), then floor-raising (6), then rest
DEFAULT_ORDER = [4, 2, 3, 6, 5, 7, 8, 10]


def run_step(step: int) -> dict:
    """Run a single step and return timing + status."""
    t0 = time.time()
    status = "success"
    error_msg = ""

    try:
        if step == 2:
            from src.revision_2026.sample_audit import build_sample_crosswalk
            build_sample_crosswalk()

        elif step == 3:
            from src.revision_2026.connected_set_representativeness import run_representativeness_checks
            run_representativeness_checks()

        elif step == 4:
            from src.revision_2026.separate_sample_effects import build_all_separate_sample_effects
            build_all_separate_sample_effects()

        elif step == 5:
            from src.revision_2026.mover_event_study import run_all_mover_event_study_tests
            run_all_mover_event_study_tests()

        elif step == 6:
            from src.revision_2026.floor_raising import run_all_floor_raising_tests
            run_all_floor_raising_tests()

        elif step == 7:
            from src.revision_2026.pipeline_tests import run_all_pipeline_tests
            run_all_pipeline_tests()

        elif step == 8:
            from src.revision_2026.route_prediction import run_route_prediction
            run_route_prediction()

        elif step == 10:
            from src.revision_2026.qa_checks import run_all_qa_checks
            run_all_qa_checks()

        else:
            logger.warning("Unknown step: %d", step)
            status = "skipped"

    except Exception as e:
        status = "failed"
        error_msg = str(e)
        logger.exception("Step %d failed: %s", step, e)

    elapsed = time.time() - t0
    return {
        "step": step,
        "status": status,
        "elapsed_seconds": round(elapsed, 1),
        "error": error_msg,
    }


def write_manifest(step_results: list) -> None:
    """Write the results manifest."""
    manifest = {
        "run_timestamp": datetime.now().isoformat(),
        "config": {
            "use_exact_loo": CFG.use_exact_loo,
            "n_loo_workers": CFG.n_loo_workers,
            "near_zero_percentile": CFG.near_zero_percentile,
            "quantile_taus": CFG.quantile_taus,
            "random_seed": CFG.random_seed,
        },
        "steps": step_results,
        "output_dir": str(OUTPUT_BASE),
    }

    manifest_path = OUTPUT_BASE / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info("Manifest saved to %s", manifest_path)


def write_results_summary(step_results: list) -> None:
    """Write a human-readable results summary."""
    lines = [
        "# Revision 2026 — Results Summary",
        "",
        f"**Run timestamp:** {datetime.now().isoformat()}",
        "",
        "## Step Results",
        "",
        "| Step | Status | Time (s) | Error |",
        "|------|--------|----------|-------|",
    ]
    for r in step_results:
        lines.append(f"| {r['step']} | {r['status']} | {r['elapsed_seconds']} | {r['error'] or '—'} |")

    lines.extend([
        "",
        "## Output Files",
        "",
    ])

    for subdir in ["tables", "figures", "intermediates"]:
        d = OUTPUT_BASE / subdir
        if d.exists():
            files = sorted(d.glob("*"))
            if files:
                lines.append(f"### {subdir}/")
                for f in files:
                    size = f.stat().st_size
                    lines.append(f"- `{f.name}` ({size:,} bytes)")
                lines.append("")

    lines.extend([
        "## QA Status",
        "",
    ])
    qa_path = TABLES_DIR / "qa_validation_flags.csv"
    if qa_path.exists():
        import pandas as pd
        flags = pd.read_csv(qa_path)
        n_err = len(flags[flags["level"] == "ERROR"]) if len(flags) > 0 else 0
        n_warn = len(flags[flags["level"] == "WARNING"]) if len(flags) > 0 else 0
        lines.append(f"- **Errors:** {n_err}")
        lines.append(f"- **Warnings:** {n_warn}")
        if n_err > 0:
            lines.append("")
            lines.append("> [!CAUTION]")
            lines.append("> There are QA errors. Review `qa_validation_flags.csv`.")
    else:
        lines.append("- QA step not yet run.")

    summary_path = OUTPUT_BASE / "RESULTS_SUMMARY.md"
    summary_path.write_text("\n".join(lines) + "\n")
    logger.info("Summary saved to %s", summary_path)


def main():
    parser = argparse.ArgumentParser(description="Run Revision 2026 empirical tests")
    parser.add_argument(
        "--steps", nargs="*", type=int, default=None,
        help="Specific steps to run (e.g., --steps 4 6). Default: all steps."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Verify setup without running analyses."
    )
    args = parser.parse_args()

    steps_to_run = args.steps if args.steps else DEFAULT_ORDER

    logger.info("=" * 70)
    logger.info("REVISION 2026 — EMPIRICAL TESTS")
    logger.info("=" * 70)
    logger.info("Steps to run: %s", steps_to_run)
    logger.info("Output dir: %s", OUTPUT_BASE)
    logger.info("Config: exact_loo=%s, n_workers=%d, near_zero_pct=%d",
                CFG.use_exact_loo, CFG.n_loo_workers, CFG.near_zero_percentile)

    if args.dry_run:
        logger.info("DRY RUN — verifying imports only")
        try:
            from src.revision_2026 import (
                sample_audit, connected_set_representativeness,
                separate_sample_effects, floor_raising,
                mover_event_study, pipeline_tests,
                route_prediction, qa_checks, output_schema,
            )
            logger.info("✓ All imports successful")
            logger.info("✓ Output directories exist")
            logger.info("✓ Config loaded successfully")
        except ImportError as e:
            logger.error("Import failed: %s", e)
            sys.exit(1)
        return

    step_results = []
    for step in steps_to_run:
        logger.info("\n" + "—" * 60)
        logger.info("Running Step %d...", step)
        result = run_step(step)
        step_results.append(result)
        logger.info("Step %d: %s (%.1fs)", step, result["status"], result["elapsed_seconds"])

    write_manifest(step_results)
    write_results_summary(step_results)

    n_failed = sum(1 for r in step_results if r["status"] == "failed")
    total_time = sum(r["elapsed_seconds"] for r in step_results)
    logger.info("\n" + "=" * 70)
    logger.info("COMPLETE: %d/%d steps succeeded in %.0fs",
                len(step_results) - n_failed, len(step_results), total_time)
    logger.info("=" * 70)

    if n_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
