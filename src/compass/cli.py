"""
Compass Pipeline — CLI Entry Point.

Usage
-----
    python -m compass.cli --config compass_config.json
    python -m compass.cli --config compass_config.json --steps 1,2,3
    python -m compass.cli --config compass_config.json --dry-run

The ``--steps`` flag accepts a comma-separated list of stage numbers
(1–10).  Without it the full pipeline runs.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from pathlib import Path
from typing import Optional, Set

import numpy as np
import pandas as pd

from compass.config import CompassConfig, load_config, save_config

logger = logging.getLogger(__name__)


# ── version snapshot ────────────────────────────────────────────────────────

def _log_env():
    """Log library versions and git hash for reproducibility."""
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    for pkg in ("scipy", "sklearn", "hmmlearn", "pyproj", "torch", "pyarrow"):
        try:
            mod = __import__(pkg)
            info[pkg] = getattr(mod, "__version__", "installed")
        except ImportError:
            info[pkg] = "not installed"

    # git hash
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        info["git_hash"] = result.stdout.strip() or "unknown"
    except Exception:
        info["git_hash"] = "unknown"

    logger.info("Environment: %s", json.dumps(info, indent=2))
    return info


# ── pipeline orchestrator ──────────────────────────────────────────────────

def run_compass_pipeline(
    cfg: CompassConfig,
    steps: Optional[Set[int]] = None,
    dry_run: bool = False,
) -> dict:
    """
    Execute the 10-step compass pipeline.

    Parameters
    ----------
    cfg : CompassConfig
    steps : set[int], optional
        Subset of stages to run (1-indexed).  ``None`` ⇒ all.
    dry_run : bool
        If True, validate config and log plan but do not write outputs.

    Returns
    -------
    dict
        Intermediate/final DataFrames keyed by stage name.
    """
    from compass.data_io import (
        load_trajectory_points,
        load_voyage_metadata,
        validate_trajectories,
    )
    from compass.preprocess import project_all_voyages, smooth_positions
    from compass.steps import build_all_step_variants, compute_raw_steps
    from compass.regimes import segment_voyages
    from compass.features import compute_compass_features
    from compass.compass_index import (
        compute_compass_index,
        compute_early_window,
        save_loadings,
    )
    from compass.export import export_panels

    all_steps = set(range(1, 11))
    run = steps if steps else all_steps

    env = _log_env()
    out_dir = cfg.output_path
    results: dict = {"env": env}

    t0 = time.time()

    # ── Step 1: Data validation ──────────────────────────────────────────
    if 1 in run:
        logger.info("═══ Step 1: Data Validation ═══")
        traj = load_trajectory_points(cfg.trajectory_points_path, cfg)
        meta = load_voyage_metadata(cfg.voyage_metadata_path)
        traj = validate_trajectories(traj, cfg)
        results["traj"] = traj
        results["meta"] = meta
        if not dry_run:
            traj.to_parquet(out_dir / "validated_points.parquet", index=False)
    else:
        # load from previous run
        traj = pd.read_parquet(out_dir / "validated_points.parquet")
        meta = pd.read_parquet(cfg.voyage_metadata_path)
        results["traj"] = traj
        results["meta"] = meta

    # ── Step 2: Projection & Cleaning ────────────────────────────────────
    if 2 in run:
        logger.info("═══ Step 2: Projection & Cleaning ═══")
        traj = project_all_voyages(traj, cfg)
        traj = smooth_positions(traj, cfg)
        results["traj"] = traj
        if not dry_run:
            traj.to_parquet(out_dir / "projected_points.parquet", index=False)

    # ── Step 3: Step Construction ────────────────────────────────────────
    if 3 in run:
        logger.info("═══ Step 3: Step Construction ═══")
        raw_steps = compute_raw_steps(traj)
        variants = build_all_step_variants(traj, cfg)
        results["raw_steps"] = raw_steps
        results["variants"] = variants
        if not dry_run:
            raw_steps.to_parquet(out_dir / "raw_steps.parquet", index=False)
            for name, vdf in variants.items():
                vdf.to_parquet(out_dir / f"steps_{name}.parquet", index=False)

    # ── Step 4: Regime Segmentation ──────────────────────────────────────
    if 4 in run:
        logger.info("═══ Step 4: Regime Segmentation ═══")
        raw_steps = results.get("raw_steps") or pd.read_parquet(out_dir / "raw_steps.parquet")
        steps_with_regimes = segment_voyages(raw_steps, cfg)
        results["steps_with_regimes"] = steps_with_regimes
        if not dry_run:
            steps_with_regimes.to_parquet(
                out_dir / "steps_with_regimes.parquet", index=False,
            )
        # also segment variants
        variants = results.get("variants", {})
        segmented_variants: dict = {}
        for name, vdf in variants.items():
            segmented_variants[name] = segment_voyages(vdf, cfg)
            if not dry_run:
                segmented_variants[name].to_parquet(
                    out_dir / f"steps_{name}_regimes.parquet", index=False,
                )
        results["segmented_variants"] = segmented_variants

    # ── Step 5: Compass Features ─────────────────────────────────────────
    if 5 in run:
        logger.info("═══ Step 5: Compass Features ═══")
        swr = results.get("steps_with_regimes") or pd.read_parquet(
            out_dir / "steps_with_regimes.parquet",
        )
        features = compute_compass_features(swr, cfg)
        results["features"] = features
        if not dry_run:
            features.to_parquet(
                out_dir / "voyage_compass_features.parquet", index=False,
            )

    # ── Step 6: Compass Index ────────────────────────────────────────────
    if 6 in run:
        logger.info("═══ Step 6: Compass Index ═══")
        features = results.get("features") or pd.read_parquet(
            out_dir / "voyage_compass_features.parquet",
        )
        index_df, loadings = compute_compass_index(features, cfg)
        results["index_df"] = index_df
        results["loadings"] = loadings
        if not dry_run:
            index_df.to_parquet(
                out_dir / "voyage_compass_index.parquet", index=False,
            )
            save_loadings(loadings, out_dir / "pca_loadings.json")

    # ── Step 7: Early-Window Compass ─────────────────────────────────────
    if 7 in run:
        logger.info("═══ Step 7: Early-Window Compass ═══")
        swr = results.get("steps_with_regimes") or pd.read_parquet(
            out_dir / "steps_with_regimes.parquet",
        )
        meta = results.get("meta") or pd.read_parquet(cfg.voyage_metadata_path)
        ew = compute_early_window(swr, meta, cfg)
        results["early_window"] = ew
        if not dry_run:
            ew.to_parquet(
                out_dir / "voyage_compass_early_window.parquet", index=False,
            )

    # ── Step 8: Self-Supervised Embedding (optional) ─────────────────────
    if 8 in run and cfg.embedding_enabled:
        logger.info("═══ Step 8: Self-Supervised Embedding ═══")
        try:
            from compass.embedding_optional import train_embedding, probe_dl_score

            swr = results.get("steps_with_regimes") or pd.read_parquet(
                out_dir / "steps_with_regimes.parquet",
            )
            encoder, embeddings, vids = train_embedding(swr, cfg)

            index_df = results.get("index_df") or pd.read_parquet(
                out_dir / "voyage_compass_index.parquet",
            )
            emb_df = probe_dl_score(embeddings, vids, index_df)
            results["embeddings"] = emb_df
            if not dry_run:
                emb_df.to_parquet(
                    out_dir / "voyage_compass_embeddings.parquet", index=False,
                )
        except ImportError:
            logger.warning("torch not available — skipping embedding step.")

    # ── Step 9: Robustness ───────────────────────────────────────────────
    if 9 in run and cfg.robustness_enabled:
        logger.info("═══ Step 9: Robustness Checks ═══")
        from compass.robustness import run_all_robustness

        swr = results.get("steps_with_regimes") or pd.read_parquet(
            out_dir / "steps_with_regimes.parquet",
        )
        variants_seg = results.get("segmented_variants")
        raw_steps = results.get("raw_steps")

        report = run_all_robustness(
            swr, variants_seg, raw_steps, cfg,
            output_dir=out_dir if not dry_run else None,
        )
        results["robustness"] = report

    # ── Step 10: Econometric Exports ─────────────────────────────────────
    if 10 in run:
        logger.info("═══ Step 10: Econometric Exports ═══")
        index_df = results.get("index_df") or pd.read_parquet(
            out_dir / "voyage_compass_index.parquet",
        )
        swr = results.get("steps_with_regimes") or pd.read_parquet(
            out_dir / "steps_with_regimes.parquet",
        )
        meta = results.get("meta") or pd.read_parquet(cfg.voyage_metadata_path)

        if not dry_run:
            panels = export_panels(index_df, swr, meta, cfg, output_dir=out_dir)
            results["panels"] = panels

    elapsed = time.time() - t0
    logger.info("Pipeline complete in %.1f s.", elapsed)

    # save config snapshot
    if not dry_run:
        save_config(cfg, out_dir / "config_snapshot.json")

    return results


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Compass Pipeline — Micro-Routing / Search Policy Measurement",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to JSON config file (defaults used if absent).",
    )
    parser.add_argument(
        "--steps", "-s",
        type=str,
        default=None,
        help="Comma-separated list of steps to run (1-10). Default: all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and log plan without writing outputs.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    steps = None
    if args.steps:
        steps = {int(s.strip()) for s in args.steps.split(",")}

    run_compass_pipeline(cfg, steps=steps, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
