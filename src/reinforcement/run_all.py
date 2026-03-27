"""
Reinforcement Test Suite — Run All Tests.

Orchestrates the full pipeline:
1. Build analysis panel
2. Build ground/patch spells from logbook data
3. Compute search metrics
4. Run type estimation (cross-fitted theta/psi)
5. Run Tests 1-5
"""

from __future__ import annotations
import logging
import sys
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-30s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_all(*, skip_spells=False, skip_type_est=False):
    """Run the complete reinforcement test pipeline."""
    from .data_builder import build_analysis_panel, load_logbook_positions
    from .type_estimation import run_type_estimation
    from .ground_spells import build_ground_spells, enrich_spells_with_voyage_data
    from .search_metrics import compute_spell_search_metrics
    from .test1_map_compass import run_test1
    from .test2_decomposition import run_test2
    from .test5_submodularity import run_test5

    print("\n" + "=" * 70)
    print("  REINFORCEMENT TEST SUITE")
    print("=" * 70)

    # ── Step 1: Build analysis panel ───────────────────────────────────
    print("\n▸ Step 1: Building analysis panel...")
    df = build_analysis_panel(require_akm=True, require_logbook=False)
    print(f"  → {len(df)} voyages, {df['captain_id'].nunique()} captains, "
          f"{df['agent_id'].nunique()} agents")

    # ── Step 2: Type estimation ────────────────────────────────────────
    if not skip_type_est:
        print("\n▸ Step 2: Cross-fitted type estimation...")
        df = run_type_estimation(df, method="time_split")
    else:
        print("\n▸ Step 2: Skipped (using in-sample theta/psi)")

    # ── Step 3: Ground/patch spells ────────────────────────────────────
    if not skip_spells:
        print("\n▸ Step 3: Building ground spells from logbook data...")
        try:
            positions = load_logbook_positions()
            spells = build_ground_spells(positions)
            spells = enrich_spells_with_voyage_data(spells, df)

            # Compute spell-level search metrics
            print("  → Computing spell-level search metrics...")
            spell_metrics = compute_spell_search_metrics(positions, spells)

            if len(spell_metrics) > 0 and len(spells) > 0:
                spells = spells.merge(
                    spell_metrics, on=["spell_id", "voyage_id"], how="left"
                )
            print(f"  → {len(spells)} ground spells")

            # Merge spell-level metrics back to voyage level (mean)
            if len(spell_metrics) > 0:
                voyage_means = spell_metrics.groupby("voyage_id").mean(numeric_only=True)
                for col in voyage_means.columns:
                    if col not in df.columns:
                        df = df.merge(
                            voyage_means[[col]],
                            left_on="voyage_id", right_index=True, how="left",
                        )
        except Exception as e:
            logger.warning("Spell construction failed: %s", e)
            spells = None
    else:
        spells = None

    # ── Step 4: Run Tests ──────────────────────────────────────────────
    print("\n▸ Step 4: Running reinforcement tests...")

    # Test 1: Map vs Compass
    print("\n  ┌─ Test 1: Same-Captain, Same-Ground, Different-Agent")
    try:
        t1 = run_test1(df, save_outputs=True)
        print(f"  └─ Status: {t1['status']}")
    except Exception as e:
        logger.error("Test 1 failed: %s", e)

    # Test 2: Decomposition
    print("\n  ┌─ Test 2: Map vs Compass Decomposition")
    try:
        t2 = run_test2(df, save_outputs=True)
        print(f"  └─ Status: {t2['status']}")
    except Exception as e:
        logger.error("Test 2 failed: %s", e)

    # Test 3: Stopping rule (requires patch-day panel)
    print("\n  ┌─ Test 3: Stopping Rule Hazard")
    if spells is not None:
        try:
            from .patch_spells import build_patch_spells, expand_to_patch_days
            positions = load_logbook_positions()
            patches = build_patch_spells(positions)
            patch_days = expand_to_patch_days(patches, positions)

            from .test3_stopping_rule import run_test3
            t3 = run_test3(patch_days, df, save_outputs=True)
            print(f"  └─ Status: {t3['status']}")
        except Exception as e:
            logger.error("Test 3 failed: %s", e)
    else:
        print("  └─ Skipped (no spell data)")

    # Test 4: Variance compression
    print("\n  ┌─ Test 4: Variance Compression")
    try:
        from .test4_variance_compression import run_test4
        t4 = run_test4(df, save_outputs=True)
        print(f"  └─ Status: {t4['status']}")
    except Exception as e:
        logger.error("Test 4 failed: %s", e)

    # Test 5: Submodularity
    print("\n  ┌─ Test 5: Submodularity")
    try:
        t5 = run_test5(df, save_outputs=True)
        print(f"  └─ Status: {t5['status']}")
    except Exception as e:
        logger.error("Test 5 failed: %s", e)

    print("\n" + "=" * 70)
    print("  COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-spells", action="store_true")
    parser.add_argument("--skip-type-est", action="store_true")
    args = parser.parse_args()
    run_all(skip_spells=args.skip_spells, skip_type_est=args.skip_type_est)
