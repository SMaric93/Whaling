"""
Master orchestrator for the minor-revision empirical fixes.

Usage:
    python -m src.minor_revision.run_all

Runs all six phases sequentially:
  Phase 0: Sample and variable audit
  Phase 1: Fix Table 2 (AMI, conditional MI, OOS prediction)
  Phase 2: Stopping threshold sensitivity curve
  Phase 3: Lay-system coverage audit
  Phase 4: Table 1 vs Table 3 scale audit
  Phase 5: Rebuild impacted paper artifacts
  Phase 6: Draft response-letter support files
"""

import sys
import time
import traceback
from pathlib import Path

# Ensure project root on path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    """Run all minor-revision phases."""
    start = time.time()

    print("=" * 70)
    print("MINOR REVISION EMPIRICAL FIXES — MASTER RUN")
    print("=" * 70)
    print(f"Project root: {PROJECT_ROOT}")
    print()

    results = {}
    phase_status = {}

    # =================================================================
    # Phase 0: Sample and variable audit
    # =================================================================
    try:
        from src.minor_revision.manifests import run_manifests
        phase0 = run_manifests()
        results["phase0"] = phase0
        phase_status["Phase 0: Sample Audit"] = "✓ COMPLETE"
    except Exception as e:
        print(f"\n✗ Phase 0 failed: {e}")
        traceback.print_exc()
        phase_status["Phase 0: Sample Audit"] = f"✗ FAILED: {e}"
        # Need the data for subsequent phases, so build minimal versions
        from src.minor_revision.manifests import _load_raw_voyage, _apply_base_filters, _get_connected_set
        df_raw = _load_raw_voyage()
        df_filtered = _apply_base_filters(df_raw)
        df_connected = _get_connected_set(df_filtered)
        phase0 = {"df_filtered": df_filtered, "df_connected": df_connected}
        results["phase0"] = phase0

    df_filtered = results["phase0"]["df_filtered"]
    df_connected = results["phase0"]["df_connected"]

    # =================================================================
    # Phase 1: Fix Table 2
    # =================================================================
    try:
        from src.minor_revision.table2_adjusted_info import run_table2_all
        table2_results = run_table2_all(df_filtered)
        results["table2"] = table2_results
        phase_status["Phase 1A: Table 2 Info Metrics"] = "✓ COMPLETE"
    except Exception as e:
        print(f"\n✗ Phase 1A failed: {e}")
        traceback.print_exc()
        phase_status["Phase 1A: Table 2 Info Metrics"] = f"✗ FAILED: {e}"
        table2_results = {}
        results["table2"] = table2_results

    try:
        from src.minor_revision.route_prediction_oos import run_route_prediction_oos
        oos_results = run_route_prediction_oos(df_filtered)
        results["oos"] = oos_results
        phase_status["Phase 1B: OOS Route Prediction"] = "✓ COMPLETE"
    except Exception as e:
        print(f"\n✗ Phase 1B failed: {e}")
        traceback.print_exc()
        phase_status["Phase 1B: OOS Route Prediction"] = f"✗ FAILED: {e}"
        results["oos"] = None

    # =================================================================
    # Phase 2: Stopping threshold curve
    # =================================================================
    try:
        from src.minor_revision.stopping_threshold_curve import run_stopping_threshold_curve
        stopping_results = run_stopping_threshold_curve()
        results["stopping"] = stopping_results
        phase_status["Phase 2: Stopping Threshold"] = "✓ COMPLETE"
    except Exception as e:
        print(f"\n✗ Phase 2 failed: {e}")
        traceback.print_exc()
        phase_status["Phase 2: Stopping Threshold"] = f"✗ FAILED: {e}"
        stopping_results = {}
        results["stopping"] = stopping_results

    # =================================================================
    # Phase 3: Lay-system audit
    # =================================================================
    try:
        from src.minor_revision.lay_contracts import run_lay_coverage_audit
        lay_results = run_lay_coverage_audit(df_filtered)
        results["lay"] = lay_results
        phase_status["Phase 3: Lay Contract Audit"] = "✓ COMPLETE"
    except Exception as e:
        print(f"\n✗ Phase 3 failed: {e}")
        traceback.print_exc()
        phase_status["Phase 3: Lay Contract Audit"] = f"✗ FAILED: {e}"
        lay_results = {"has_any_lay": False}
        results["lay"] = lay_results

    # =================================================================
    # Phase 4: Scale audit
    # =================================================================
    try:
        from src.minor_revision.scale_audit import run_scale_audit
        scale_results = run_scale_audit(df_filtered, df_connected)
        results["scale"] = scale_results
        phase_status["Phase 4: Scale Audit"] = "✓ COMPLETE"
    except Exception as e:
        print(f"\n✗ Phase 4 failed: {e}")
        traceback.print_exc()
        phase_status["Phase 4: Scale Audit"] = f"✗ FAILED: {e}"
        scale_results = {"discrepancy_note": "Phase 4 failed", "stats_t1": {}, "stats_t3": {}}
        results["scale"] = scale_results

    # =================================================================
    # Phase 5: Rebuild impacted tables
    # =================================================================
    try:
        from src.minor_revision.rebuild_impacted_tables import run_rebuild_impacted
        run_rebuild_impacted(table2_results, stopping_results, lay_results, scale_results)
        phase_status["Phase 5: Rebuild Artifacts"] = "✓ COMPLETE"
    except Exception as e:
        print(f"\n✗ Phase 5 failed: {e}")
        traceback.print_exc()
        phase_status["Phase 5: Rebuild Artifacts"] = f"✗ FAILED: {e}"

    # =================================================================
    # Phase 6: Response assets
    # =================================================================
    try:
        from src.minor_revision.response_assets import run_response_assets
        run_response_assets(table2_results, stopping_results, lay_results, scale_results)
        phase_status["Phase 6: Response Assets"] = "✓ COMPLETE"
    except Exception as e:
        print(f"\n✗ Phase 6 failed: {e}")
        traceback.print_exc()
        phase_status["Phase 6: Response Assets"] = f"✗ FAILED: {e}"

    # =================================================================
    # Summary
    # =================================================================
    elapsed = time.time() - start
    print("\n")
    print("=" * 70)
    print("MINOR REVISION — EXECUTION SUMMARY")
    print("=" * 70)
    for phase, status in phase_status.items():
        print(f"  {phase}: {status}")
    print(f"\n  Total time: {elapsed:.1f}s")

    # Count outputs
    from .utils.io import TABLES_DIR, FIGURES_DIR, MEMOS_DIR, MANIFESTS_DIR, DOCS_DIR
    for d, label in [
        (TABLES_DIR, "Tables"), (FIGURES_DIR, "Figures"),
        (MEMOS_DIR, "Memos"), (MANIFESTS_DIR, "Manifests"),
        (DOCS_DIR, "Docs"),
    ]:
        if d.exists():
            n = len(list(d.iterdir()))
            print(f"  {label}: {n} files in {d}")

    n_success = sum(1 for s in phase_status.values() if "✓" in s)
    n_total = len(phase_status)
    print(f"\n  Result: {n_success}/{n_total} phases completed successfully")

    if n_success < n_total:
        print("\n  ⚠ Some phases failed. Check output above for details.")
        sys.exit(1)
    else:
        print("\n  ✓ All phases complete. Minor revision outputs ready.")


if __name__ == "__main__":
    main()
