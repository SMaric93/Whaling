"""CLI entry point for the latent voyage-state model pipeline.

Runs the full HSMM pipeline:
1. Load events (existing cached flatten)
2. Build voyage linkage (existing)
3. Score remarks (existing) — cached to disk
4. Build weekly observation panel (vectorized)
5. Generate anchor posteriors (vectorized)
6. Train observation encoder
7. Fit HSMM
8. Decode paths (posterior + Viterbi)
9. Summarize state features
10. Validate

Usage::

    python -m src.analyses.wsl_reliability_ml.run_state_model [--max-events N]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Drop columns that can't be serialized to parquet (e.g. mixed list/scalar)
    safe_cols = []
    for col in df.columns:
        try:
            import pyarrow as pa
            pa.array(df[col].values, from_pandas=True)
            safe_cols.append(col)
        except Exception:
            pass
    if len(safe_cols) < len(df.columns):
        dropped = set(df.columns) - set(safe_cols)
        logger.debug("Dropping non-serializable columns for parquet: %s", dropped)
    df[safe_cols].to_parquet(path, index=False)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace | None = None) -> None:
    """Run the full latent voyage-state model pipeline."""
    if args is None:
        parser = argparse.ArgumentParser(
            description="Latent voyage-state model pipeline"
        )
        parser.add_argument(
            "--max-events",
            type=int,
            default=None,
            help="Limit the number of events to process (for testing)",
        )
        parser.add_argument(
            "--max-duration",
            type=int,
            default=52,
            help="Maximum state duration in weeks for the HSMM",
        )
        parser.add_argument(
            "--em-iters",
            type=int,
            default=8,
            help="Number of EM iterations for HSMM fitting",
        )
        parser.add_argument(
            "--skip-fit",
            action="store_true",
            help="Skip HSMM fitting (decode with initialized model only)",
        )
        parser.add_argument(
            "--force-remarks",
            action="store_true",
            help="Re-score remarks even if a cached version exists",
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            default=None,
            help="Override output directory",
        )
        args = parser.parse_args()

    # --- Import pipeline modules ---
    from .utils import (
        PerfTracer,
        WSLReliabilityConfig,
        build_voyage_linkage,
        ensure_output_dirs,
        load_voyage_reference,
        load_wsl_cleaned_events,
    )
    from .remarks_taxonomy import (
        predict_remarks_annotations,
        train_remarks_models,
    )
    from .build_weekly_panel import build_weekly_observation_panel
    from .anchor_posteriors import build_anchor_posteriors
    from .emission_model import (
        predict_state_emissions,
        train_observation_encoder,
    )
    from .fit_hsmm import fit_hsmm, save_hsmm_artifacts
    from .decode_paths import (
        decode_all_voyages,
        export_posterior_paths,
        export_viterbi_paths,
    )
    from .summarize_paths import summarize_state_features
    from .validate_states import run_all_validation

    tracer = PerfTracer()
    config = WSLReliabilityConfig()
    if args.output_dir:
        config.output_root = Path(args.output_dir)

    output_dirs = ensure_output_dirs(config)
    states_dir = output_dirs["root"] / "states_hsmm"
    states_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dirs["root"] / "hsmm_pipeline_run.log"
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s  %(name)s  %(levelname)s  %(message)s"))
    logging.getLogger().addHandler(file_handler)

    # Cache paths
    _remarks_cache = states_dir / "_remarks_scored_events.parquet"
    _linkage_cache = states_dir / "_linkage.parquet"

    # ================================================================
    # Stage 1: Load events
    # ================================================================
    with tracer.span("load_events"):
        logger.info("=" * 60)
        logger.info("Stage 1: Loading WSL events")
        logger.info("=" * 60)
        events_df = load_wsl_cleaned_events(config)
        if args.max_events and len(events_df) > args.max_events:
            logger.info("Limiting to %d events (from %d)", args.max_events, len(events_df))
            events_df = events_df.head(args.max_events).copy()
        tracer.set_metadata(n_events=len(events_df))
        logger.info("Loaded %d events", len(events_df))

    # ================================================================
    # Stage 2: Voyage linkage
    # ================================================================
    with tracer.span("voyage_linkage"):
        logger.info("=" * 60)
        logger.info("Stage 2: Voyage linkage")
        logger.info("=" * 60)
        voyages_df = load_voyage_reference(config)

        if _linkage_cache.exists() and not args.force_remarks:
            logger.info("Loading cached linkage from %s", _linkage_cache)
            linkage_df = pd.read_parquet(_linkage_cache)
        else:
            linkage_df = build_voyage_linkage(events_df, voyages_df, config)
            _save_parquet(linkage_df, _linkage_cache)

        n_linked = linkage_df["voyage_id"].notna().sum()
        tracer.set_metadata(
            n_voyages=int(voyages_df["voyage_id"].nunique()),
            n_linked=int(n_linked),
        )
        logger.info("Linked %d/%d events to voyages", n_linked, len(linkage_df))

    # ================================================================
    # Stage 3: Remarks scoring — cached
    # ================================================================
    with tracer.span("remarks_scoring"):
        logger.info("=" * 60)
        logger.info("Stage 3: Remarks taxonomy scoring")
        logger.info("=" * 60)

        if _remarks_cache.exists() and not args.force_remarks:
            logger.info("Loading cached remarks-scored events from %s", _remarks_cache)
            events_scored = pd.read_parquet(_remarks_cache)
            # Need minimal metrics for tracer
            remarks_macro_f1 = float("nan")
        else:
            remarks_models = train_remarks_models(events_df, config)
            events_scored = predict_remarks_annotations(events_df, remarks_models, config)
            remarks_macro_f1 = remarks_models["metrics"].get("primary_class_macro_f1", 0)
            _save_parquet(events_scored, _remarks_cache)
            logger.info("Cached remarks-scored events to %s", _remarks_cache)

        tracer.set_metadata(
            n_scored=len(events_scored),
            primary_macro_f1=remarks_macro_f1,
        )
        logger.info(
            "Remarks scored: %d events, macro-F1=%s",
            len(events_scored),
            f"{remarks_macro_f1:.3f}" if not np.isnan(remarks_macro_f1) else "cached",
        )

    # ================================================================
    # Stage 4: Weekly observation panel
    # ================================================================
    with tracer.span("weekly_panel"):
        logger.info("=" * 60)
        logger.info("Stage 4: Building weekly observation panel")
        logger.info("=" * 60)
        weekly_df = build_weekly_observation_panel(
            events_scored, linkage_df, voyages_df, config, tracer=tracer
        )
        n_voyages = int(weekly_df["voyage_id"].nunique()) if not weekly_df.empty else 0
        tracer.set_metadata(n_weekly_rows=len(weekly_df), n_voyages=n_voyages)
        logger.info("Weekly panel: %d rows, %d voyages", len(weekly_df), n_voyages)
        _save_parquet(weekly_df, states_dir / "weekly_panel.parquet")

    # ================================================================
    # Stage 5: Anchor posteriors
    # ================================================================
    with tracer.span("anchor_posteriors"):
        logger.info("=" * 60)
        logger.info("Stage 5: Generating anchor posteriors")
        logger.info("=" * 60)
        anchor_df = build_anchor_posteriors(weekly_df)
        n_anchored = int((anchor_df["n_anchor_rules_fired"] > 0).sum())
        tracer.set_metadata(n_anchored=n_anchored, n_total=len(anchor_df))
        logger.info("Anchors: %d/%d weeks have anchor evidence", n_anchored, len(anchor_df))
        _save_parquet(anchor_df, states_dir / "anchor_posteriors.parquet")

    # ================================================================
    # Stage 6: Observation encoder
    # ================================================================
    with tracer.span("observation_encoder"):
        logger.info("=" * 60)
        logger.info("Stage 6: Training observation encoder")
        logger.info("=" * 60)
        encoder_bundle = train_observation_encoder(weekly_df, anchor_df)
        if encoder_bundle.get("model") is not None:
            weekly_scored = predict_state_emissions(weekly_df, encoder_bundle)
        else:
            weekly_scored = weekly_df.copy()
            from .state_space import STATE_NAMES as _sn
            for s in _sn:
                weekly_scored[f"state_prob_{s}"] = 1.0 / len(_sn)
        tracer.set_metadata(**encoder_bundle.get("metrics", {}))
        logger.info(
            "Encoder: n_train=%d, classes=%d",
            encoder_bundle.get("metrics", {}).get("n_train", 0),
            encoder_bundle.get("metrics", {}).get("n_states_observed", 0),
        )
        _save_parquet(weekly_scored, states_dir / "weekly_panel_scored.parquet")

    # ================================================================
    # Stage 7: HSMM fitting
    # ================================================================
    with tracer.span("hsmm_fitting"):
        logger.info("=" * 60)
        logger.info("Stage 7: Fitting HSMM")
        logger.info("=" * 60)
        fit_config = {
            "max_duration": args.max_duration,
            "n_iters": args.em_iters if not args.skip_fit else 0,
            "convergence_tol": 1e-3,
            "anchor_kl_weight": 0.5,
        }
        fit_result = fit_hsmm(weekly_scored, anchor_df, encoder_bundle, config=fit_config)
        model = fit_result["model"]
        sequences = fit_result["sequences"]
        save_hsmm_artifacts(fit_result, states_dir)
        tracer.set_metadata(
            n_sequences=len(sequences),
            n_em_iterations=len(fit_result.get("diagnostics", [])),
        )
        logger.info(
            "HSMM fit: %d sequences, %d iterations",
            len(sequences),
            len(fit_result.get("diagnostics", [])),
        )

    # ================================================================
    # Stage 8: Decoding
    # ================================================================
    with tracer.span("decoding"):
        logger.info("=" * 60)
        logger.info("Stage 8: Decoding state paths")
        logger.info("=" * 60)
        posterior_df, viterbi_df = decode_all_voyages(
            fit_result,
            sequences,
            decode_mode="both",
        )
        tracer.set_metadata(
            n_posterior_rows=len(posterior_df),
            n_viterbi_rows=len(viterbi_df),
        )
        export_posterior_paths(posterior_df, states_dir / "posterior_paths.parquet")
        export_viterbi_paths(viterbi_df, states_dir / "viterbi_paths.parquet")
        logger.info("Decoded: %d posterior rows, %d viterbi rows", len(posterior_df), len(viterbi_df))

    # ================================================================
    # Stage 9: Summarize
    # ================================================================
    with tracer.span("summarize"):
        logger.info("=" * 60)
        logger.info("Stage 9: Summarizing state features")
        logger.info("=" * 60)
        summary_df = summarize_state_features(viterbi_df)
        _save_parquet(summary_df, states_dir / "state_summary.parquet")
        tracer.set_metadata(n_summaries=len(summary_df))
        logger.info("Summaries: %d voyages", len(summary_df))

        if not viterbi_df.empty:
            state_counts = viterbi_df["viterbi_state"].value_counts()
            logger.info("State distribution:\n%s", state_counts.to_string())

    # ================================================================
    # Stage 10: Validation
    # ================================================================
    with tracer.span("validation"):
        logger.info("=" * 60)
        logger.info("Stage 10: Scientific validation")
        logger.info("=" * 60)
        validation = run_all_validation(
            viterbi_df,
            summary_df,
            voyage_ref_df=voyages_df,
        )
        # Serialize transition_counts DataFrame
        validation_serializable = {}
        for k, v in validation.items():
            if isinstance(v, dict):
                v_clean = {}
                for k2, v2 in v.items():
                    if isinstance(v2, pd.DataFrame):
                        v_clean[k2] = v2.to_dict()
                    else:
                        v_clean[k2] = v2
                validation_serializable[k] = v_clean
            else:
                validation_serializable[k] = v
        _write_json(states_dir / "validation_report.json", validation_serializable)

        if validation.get("all_checks_pass"):
            logger.info("✓ All validation checks passed")
        else:
            logger.warning("✗ Some validation checks failed — see validation_report.json")

    # ================================================================
    # Summary
    # ================================================================
    tracer.export_json(states_dir / "hsmm_perf_trace.json")

    summary_text = f"""# HSMM State Model Results

- Events loaded: {len(events_df):,}
- Voyage linkage: {n_linked:,}/{len(linkage_df):,} events linked
- Weekly panel: {len(weekly_df):,} observation-weeks, {n_voyages:,} voyages
- Anchor posteriors: {n_anchored:,}/{len(anchor_df):,} with evidence
- Observation encoder: {encoder_bundle.get('metrics', {}).get('n_train', 0):,} training examples
- HSMM sequences: {len(sequences):,}
- Decoded: {len(viterbi_df):,} viterbi rows, {len(posterior_df):,} posterior rows
- Voyage summaries: {len(summary_df):,}

## Validation
- Duration plausibility: {'✓' if validation.get('duration_plausibility', {}).get('__all_pass__', False) else '✗'}
- Impossible transitions: {validation.get('impossible_transitions', {}).get('n_violations', '?')} violations
- Absorbing states: {'✓' if validation.get('absorbing_states', {}).get('pass', False) else '✗'}
- Predictive power: {'✓' if validation.get('predictive_power', {}).get('pass', False) else '?'}

## Performance
- Total time: {tracer.total_elapsed_seconds:.1f}s
"""
    (states_dir / "HSMM_RESULTS.md").write_text(summary_text)
    logger.info("Pipeline complete. Results at %s", states_dir)
    logger.info("Total time: %.1fs", tracer.total_elapsed_seconds)


if __name__ == "__main__":
    main()
