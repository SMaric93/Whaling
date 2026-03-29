"""
ML Layer — Master Runner.

Executes all ML phases in sequence:
  Phase ML-1: Policy Learning (Map vs Compass)
  Phase ML-2: Latent Search States
  Phase ML-3: Nonlinear Survival Models
  Phase ML-4: Heterogeneity / Floor-Raising
  Phase ML-5: Production Surface
  Phase ML-5b: Assignment Optimizer
  Phase ML-6: Trajectory Embeddings
  Phase ML-7: Change-Point Detection
  Phase ML-8: Network Imprinting
  Phase ML-9: Text NLP
  Phase ML-10: Spatial Quality
  Phase ML-11: Conformal Prediction
  Phase ML-12: Off-Policy Evaluation

Usage:
    python -m src.ml.run_all [--phases 1 2 3] [--skip-appendix]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from typing import Dict, Any, List

from tqdm import tqdm

import os

# Prevent dual-OpenMP segfault (Intel MKL + LLVM libomp on macOS/conda).
# Setting KMP_DUPLICATE_LIB_OK is the minimal fix — do NOT cap thread counts
# to 1, as that cripples BLAS performance for all sklearn/scipy operations.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Let BLAS/OpenMP use a reasonable number of threads.
# sklearn's n_jobs=-1 handles process-level parallelism via joblib;
# within each process, BLAS threads handle matrix ops.
import multiprocessing as _mp
_n_cores = str(min(_mp.cpu_count(), 8))  # Cap at 8 to avoid oversubscription
os.environ.setdefault("OMP_NUM_THREADS", _n_cores)
os.environ.setdefault("MKL_NUM_THREADS", _n_cores)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _n_cores)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", _n_cores)
os.environ.setdefault("NUMEXPR_NUM_THREADS", _n_cores)

try:
    import threadpoolctl
    # Patch the _make_module_from_path to swallow version-detection errors
    _orig_make_module = threadpoolctl._ThreadpoolInfo._make_module_from_path
    def _safe_make_module(self, filepath):
        try:
            _orig_make_module(self, filepath)
        except (AttributeError, TypeError, OSError):
            pass  # Skip modules whose version can't be detected
    threadpoolctl._ThreadpoolInfo._make_module_from_path = _safe_make_module
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)


def _purge_cached_datasets():
    """Delete cached ML dataset .parquet files so they are rebuilt."""
    from pathlib import Path
    cache_dir = Path(__file__).resolve().parents[2] / "outputs" / "datasets" / "ml"
    if not cache_dir.exists():
        return
    for pq in cache_dir.glob("*.parquet"):
        logger.info("Deleting cached dataset: %s", pq.name)
        pq.unlink()
    logger.info("All cached ML datasets purged.")


CORE_PHASES = {
    1: ("Policy Learning", "src.ml.policy_learning", "run_policy_learning"),
    2: ("Latent States", "src.ml.state_models", "run_state_models"),
    3: ("Survival ML", "src.ml.survival_ml", "run_survival_ml"),
    4: ("Heterogeneity", "src.ml.heterogeneity_ml", "run_heterogeneity_ml"),
    5: ("Production Surface", "src.ml.production_surface_ml", "run_production_surface_ml"),
}

APPENDIX_PHASES = {
    6: ("Trajectory Embeddings", "src.ml.trajectory_embeddings", "build_embeddings"),
    7: ("Change-Points", "src.ml.changepoints", "detect_changepoints"),
    8: ("Network Imprinting", "src.ml.network_imprinting", "analyze_network_imprinting"),
    9: ("Text NLP", "src.ml.text_nlp", "analyze_text"),
    10: ("Spatial Quality", "src.ml.spatial_quality", "estimate_spatial_quality"),
    11: ("Conformal Risk", "src.ml.conformal_risk", "run_conformal_analysis"),
    12: ("Off-Policy Evaluation", "src.ml.off_policy_eval", "run_off_policy_evaluation"),
}


def run_all(
    *,
    phases: List[int] = None,
    skip_appendix: bool = False,
    save_outputs: bool = True,
) -> Dict[int, Any]:
    """
    Execute ML phases.

    Parameters
    ----------
    phases : list of int
        Specific phases to run. If None, run all.
    skip_appendix : bool
        If True, only run core phases (1-5).
    """
    t0 = time.time()

    from src.ml.acceleration import get_ml_runtime_info
    from src.ml.config import ML_CFG

    runtime_info = get_ml_runtime_info(ML_CFG.torch_device)
    logger.info("ML acceleration runtime: %s", json.dumps(runtime_info, indent=2))

    all_phases = {**CORE_PHASES}
    if not skip_appendix:
        all_phases.update(APPENDIX_PHASES)

    if phases is not None:
        all_phases = {k: v for k, v in all_phases.items() if k in phases}

    results = {}

    phase_items = sorted(all_phases.items())
    pbar = tqdm(phase_items, desc="ML Phases", unit="phase", ncols=88,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    for phase_num, (name, module_path, func_name) in pbar:
        pbar.set_postfix_str(f"ML-{phase_num}: {name}")
        logger.info("=" * 70)
        logger.info("  PHASE ML-%d: %s", phase_num, name)
        logger.info("=" * 70)

        try:
            from importlib import import_module
            module = import_module(module_path)
            func = getattr(module, func_name)
            result = func(save_outputs=save_outputs)
            results[phase_num] = result
            logger.info("Phase ML-%d (%s) completed successfully", phase_num, name)
        except Exception as e:
            logger.error("Phase ML-%d (%s) FAILED: %s", phase_num, name, e, exc_info=True)
            results[phase_num] = {"error": str(e)}
    pbar.close()

    # ── Assignment optimizer (after production surface) ─────────────
    if 5 in results and "error" not in results.get(5, {}) and not skip_appendix:
        logger.info("=" * 70)
        logger.info("  PHASE ML-5b: Assignment Optimizer")
        logger.info("=" * 70)
        try:
            from src.ml.assignment_optimizer import run_assignment_from_surface
            results["5b"] = run_assignment_from_surface(
                surface_results=results.get(5),
                save_outputs=save_outputs,
            )
        except Exception as e:
            logger.error("Phase ML-5b FAILED: %s", e, exc_info=True)
            results["5b"] = {"error": str(e)}

    elapsed = time.time() - t0
    logger.info("=" * 70)
    logger.info("ALL PHASES COMPLETE in %.1f minutes", elapsed / 60)
    logger.info("=" * 70)

    # Summary
    for k, v in sorted(results.items(), key=lambda x: str(x[0])):
        status = "ERROR" if isinstance(v, dict) and "error" in v else "OK"
        logger.info("  Phase %s: %s", k, status)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run ML layer phases")
    parser.add_argument(
        "--phases", nargs="+", type=int, default=None,
        help="Specific phases to run (1-12). Default: all.",
    )
    parser.add_argument(
        "--skip-appendix", action="store_true",
        help="Skip appendix phases (6-12).",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Don't save outputs to disk.",
    )
    parser.add_argument(
        "--force-rebuild", action="store_true",
        help="Delete cached ML datasets and rebuild from scratch.",
    )
    args = parser.parse_args()

    if args.force_rebuild:
        _purge_cached_datasets()

    run_all(
        phases=args.phases,
        skip_appendix=args.skip_appendix,
        save_outputs=not args.no_save,
    )


if __name__ == "__main__":
    main()
