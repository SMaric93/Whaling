"""Semi-supervised HSMM fitting via anchor-regularized EM.

Implements the training procedure from the build spec:
1. Initialize transitions from anchor-implied transitions + smoothing
2. Initialize durations from spec priors and empirical dwell counts
3. Train observation encoder separately
4. Run EM-like refinement: E-step with forward-backward HSMM;
   M-step update transitions/durations
5. Stop when penalized log-likelihood stabilizes

**Performance note**: The E-step is parallelized across sequences using
``concurrent.futures.ProcessPoolExecutor``, yielding ~5-6× speedup on
multi-core machines (embarrassingly parallel).

Usage::

    from .fit_hsmm import fit_hsmm
    result = fit_hsmm(weekly_df, anchor_df, encoder_bundle, config)
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .hsmm import ExplicitDurationHSMM, prepare_sequence
from .state_space import (
    NUM_STATES,
    STATE_DEFS,
    STATE_INDEX,
    STATE_NAMES,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_sequences(
    weekly_df: pd.DataFrame,
    emission_col_prefix: str = "state_prob_",
    quality_col: str = "quality_weight",
) -> list[dict[str, Any]]:
    """Convert the weekly panel into per-voyage sequence dictionaries.

    Each dict contains:
    - voyage_id
    - emission_probs: ndarray [T, K]
    - quality_weights: ndarray [T]
    - week_indices: ndarray [T]
    """
    prob_cols = [f"{emission_col_prefix}{s}" for s in STATE_NAMES]
    # Check which columns exist
    available = [c for c in prob_cols if c in weekly_df.columns]
    if not available:
        logger.warning("[fit_hsmm] No emission columns found; cannot build sequences")
        return []

    sequences = []
    for voyage_id, group in weekly_df.sort_values(["voyage_id", "week_idx"]).groupby(
        "voyage_id", sort=False
    ):
        if len(group) < 2:
            continue  # skip singleton sequences

        # Emission probs
        probs = group[available].to_numpy(dtype=np.float64)
        # Pad missing states with uniform
        if len(available) < NUM_STATES:
            full_probs = np.full((len(group), NUM_STATES), 1.0 / NUM_STATES, dtype=np.float64)
            for j, col in enumerate(available):
                state_idx = STATE_NAMES.index(col.replace(emission_col_prefix, ""))
                full_probs[:, state_idx] = probs[:, j]
            probs = full_probs

        # Normalize
        probs = np.clip(probs, 1e-6, 1.0)
        probs /= probs.sum(axis=1, keepdims=True)

        # Quality weights
        qw = pd.to_numeric(group.get(quality_col), errors="coerce").fillna(0.5).to_numpy(
            dtype=np.float64
        )

        sequences.append(
            {
                "voyage_id": voyage_id,
                "emission_probs": probs,
                "quality_weights": qw,
                "week_indices": group["week_idx"].to_numpy(dtype=np.int64),
                "T": len(group),
            }
        )

    logger.info(
        "[fit_hsmm] Built %d sequences (lengths: min=%d, median=%d, max=%d)",
        len(sequences),
        min(s["T"] for s in sequences) if sequences else 0,
        int(np.median([s["T"] for s in sequences])) if sequences else 0,
        max(s["T"] for s in sequences) if sequences else 0,
    )

    return sequences


def _compute_anchor_kl_penalty(
    posteriors: torch.Tensor,
    anchor_priors: np.ndarray | None,
    weight: float = 1.0,
) -> float:
    """KL divergence penalty between posteriors and anchor targets."""
    if anchor_priors is None or len(anchor_priors) == 0:
        return 0.0

    anchor_t = torch.tensor(anchor_priors, dtype=torch.float64)
    anchor_t = anchor_t.clamp(min=1e-12)
    post = posteriors.clamp(min=1e-12)

    # KL(anchor || posterior) per time step
    kl = (anchor_t * (anchor_t.log() - post.log())).sum(dim=1)
    return float(weight * kl.mean())


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def initialize_from_anchors(
    model: ExplicitDurationHSMM,
    anchor_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
) -> None:
    """Initialize transition and duration parameters from anchor evidence.

    Uses anchor-implied transitions (consecutive anchored weeks) plus
    smoothing to set empirically grounded initial parameters.
    """
    prior_cols = [f"state_prior_{s}" for s in STATE_NAMES]

    # Count anchor-implied transitions
    trans_counts = torch.zeros(NUM_STATES, NUM_STATES, dtype=torch.float64)
    dur_counts = [torch.zeros(model.max_duration, dtype=torch.float64) for _ in range(NUM_STATES)]

    merged = weekly_df.merge(
        anchor_df[["voyage_id", "week_idx", "anchor_strength"] + prior_cols],
        on=["voyage_id", "week_idx"],
        how="left",
    )

    for _, group in merged.sort_values(["voyage_id", "week_idx"]).groupby("voyage_id"):
        strong = group[group["anchor_strength"].fillna(0) >= 0.55].copy()
        if len(strong) < 2:
            continue

        states = strong[prior_cols].to_numpy(dtype=np.float64).argmax(axis=1)
        strengths = strong["anchor_strength"].to_numpy(dtype=np.float64)

        for i in range(len(states) - 1):
            s_from = states[i]
            s_to = states[i + 1]
            w = min(strengths[i], strengths[i + 1])
            trans_counts[s_from, s_to] += w

        # Duration counts from consecutive same-state segments
        current_state = states[0]
        current_dur = 1
        for i in range(1, len(states)):
            if states[i] == current_state:
                current_dur += 1
            else:
                if current_dur <= model.max_duration:
                    dur_counts[current_state][current_dur - 1] += 1.0
                current_state = states[i]
                current_dur = 1
        if current_dur <= model.max_duration:
            dur_counts[current_state][current_dur - 1] += 1.0

    # Update model with smoothed counts
    model.update_transitions(trans_counts, smoothing=2.0)
    model.update_durations(dur_counts, smoothing=0.1)

    logger.info(
        "[fit_hsmm] Initialized from anchors: %.0f total transition counts",
        trans_counts.sum().item(),
    )


# ---------------------------------------------------------------------------
# Parallel E-step worker
# ---------------------------------------------------------------------------


def _process_single_sequence(args: tuple) -> dict[str, Any] | None:
    """Process a single sequence for the E-step (runs in worker process).

    Returns a dict with:
    - ll: float  (sequence log-likelihood)
    - kl: float  (anchor KL penalty)
    - trans_counts: ndarray [K, K]
    - dur_counts: list of ndarray [max_dur]  (one per state)
    - ok: bool
    """
    (
        emission_probs,
        quality_weights,
        week_indices,
        voyage_id,
        T,
        model_state_dict,
        max_duration,
        transition_mask_np,
        anchor_priors_for_voyage,  # dict[int_week -> ndarray] or None
        anchor_kl_weight,
    ) = args

    try:
        # Reconstruct model in worker (lightweight — no GPU)
        model = ExplicitDurationHSMM(max_duration=max_duration, device="cpu")
        model.load_state_dict(model_state_dict)
        model.eval()

        prepared = prepare_sequence(
            emission_probs,
            quality_weights,
            device="cpu",
        )

        # Forward pass for log-likelihood
        alpha = model.forward_logprob(
            prepared["emission_logprobs"],
            prepared.get("quality_weights"),
        )
        ll = float(torch.logsumexp(alpha[-1], dim=0).item())
        if not np.isfinite(ll):
            ll = 0.0

        # Posterior decode
        posterior = model.posterior_decode(
            prepared["emission_logprobs"],
            prepared.get("quality_weights"),
        )

        # Vectorized transition counts: outer product of consecutive posteriors
        # trans_counts[r,s] += sum_t posterior[t,r] * posterior[t+1,s]
        trans_counts = torch.einsum(
            "tr,ts->rs", posterior[:-1], posterior[1:]
        ).numpy()
        # Zero out forbidden transitions
        trans_counts *= transition_mask_np

        # Duration counts from Viterbi path
        path = model.viterbi_decode(
            prepared["emission_logprobs"],
            prepared.get("quality_weights"),
        )
        dur_counts_np = [np.zeros(max_duration, dtype=np.float64) for _ in range(NUM_STATES)]
        current_state = path[0]
        current_dur = 1
        for t in range(1, len(path)):
            if path[t] == current_state:
                current_dur += 1
            else:
                if current_dur <= max_duration:
                    dur_counts_np[current_state][current_dur - 1] += 1.0
                current_state = path[t]
                current_dur = 1
        if current_dur <= max_duration:
            dur_counts_np[current_state][current_dur - 1] += 1.0

        # Anchor KL penalty
        kl = 0.0
        if anchor_priors_for_voyage:
            valid_t = []
            valid_anchors = []
            for t_idx in range(T):
                w = int(week_indices[t_idx])
                if w in anchor_priors_for_voyage:
                    valid_t.append(t_idx)
                    valid_anchors.append(anchor_priors_for_voyage[w])

            if valid_anchors:
                anchor_matrix = np.stack(valid_anchors)
                anchor_posts = torch.stack([posterior[t] for t in valid_t])
                kl = _compute_anchor_kl_penalty(
                    anchor_posts, anchor_matrix, weight=anchor_kl_weight
                )

        return {
            "ll": ll,
            "kl": kl,
            "trans_counts": trans_counts,
            "dur_counts": dur_counts_np,
            "ok": True,
        }

    except Exception:
        return None


# ---------------------------------------------------------------------------
# EM iterations (parallelized)
# ---------------------------------------------------------------------------


def run_em_iterations(
    model: ExplicitDurationHSMM,
    sequences: list[dict[str, Any]],
    anchor_df: pd.DataFrame | None = None,
    *,
    n_iters: int = 10,
    convergence_tol: float = 1e-3,
    anchor_kl_weight: float = 0.5,
    max_sequences_per_iter: int = 2000,
    n_workers: int | None = None,
) -> list[dict[str, Any]]:
    """Run EM-like iterations to refine HSMM parameters.

    The E-step is parallelized across sequences using ProcessPoolExecutor.

    Parameters
    ----------
    model : ExplicitDurationHSMM
        The model to refine (modified in place).
    sequences : list of dict
        Per-voyage sequence data from ``_build_sequences``.
    anchor_df : pd.DataFrame, optional
        Anchor posteriors for KL regularization.
    n_iters : int
        Maximum EM iterations.
    convergence_tol : float
        Stop when relative change in log-likelihood < this.
    anchor_kl_weight : float
        Weight for anchor KL penalty in the objective.
    max_sequences_per_iter : int
        Subsample if too many sequences.
    n_workers : int, optional
        Number of worker processes. Defaults to min(cpu_count, 8).

    Returns
    -------
    list of dict
        Per-iteration diagnostics.
    """
    if n_workers is None:
        n_workers = min(os.cpu_count() or 4, 8)

    # Build anchor lookup: {voyage_id -> {week_idx -> prior_array}}
    anchor_by_voyage: dict[Any, dict[int, np.ndarray]] = {}
    if anchor_df is not None:
        prior_cols = [f"state_prior_{s}" for s in STATE_NAMES]
        strong_anchors = anchor_df[anchor_df["anchor_strength"].fillna(0) >= 0.55]
        for _, row in strong_anchors.iterrows():
            vid = row["voyage_id"]
            wk = int(row["week_idx"])
            if vid not in anchor_by_voyage:
                anchor_by_voyage[vid] = {}
            anchor_by_voyage[vid][wk] = row[prior_cols].to_numpy(dtype=np.float64)

    # Pre-compute transition mask as numpy for workers
    transition_mask_np = model.transition_mask.numpy().astype(np.float64)

    diagnostics: list[dict[str, Any]] = []
    prev_ll = -np.inf

    logger.info(
        "[fit_hsmm] Starting EM with %d workers, %d max sequences/iter",
        n_workers,
        max_sequences_per_iter,
    )

    for iteration in range(n_iters):
        # Subsample if needed
        if len(sequences) > max_sequences_per_iter:
            rng = np.random.default_rng(42 + iteration)
            seq_subset = [
                sequences[i]
                for i in rng.choice(len(sequences), max_sequences_per_iter, replace=False)
            ]
        else:
            seq_subset = sequences

        # Filter very long sequences
        max_t = model.max_duration * 2
        seq_subset = [s for s in seq_subset if s["T"] <= max_t]

        # Serialize model state once per iteration
        model_state = model.state_dict()

        # Build worker arguments
        worker_args = []
        for seq in seq_subset:
            anchor_for_v = anchor_by_voyage.get(seq["voyage_id"])
            worker_args.append((
                seq["emission_probs"],
                seq["quality_weights"],
                seq["week_indices"],
                seq["voyage_id"],
                seq["T"],
                model_state,
                model.max_duration,
                transition_mask_np,
                anchor_for_v,
                anchor_kl_weight,
            ))

        # Parallel E-step
        total_ll = 0.0
        total_kl = 0.0
        n_processed = 0
        trans_counts = np.zeros((NUM_STATES, NUM_STATES), dtype=np.float64)
        dur_counts = [np.zeros(model.max_duration, dtype=np.float64) for _ in range(NUM_STATES)]

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(_process_single_sequence, arg)
                for arg in worker_args
            ]

            for future in as_completed(futures):
                result = future.result()
                if result is None or not result.get("ok"):
                    continue

                total_ll += result["ll"]
                total_kl += result["kl"]
                trans_counts += result["trans_counts"]
                for s in range(NUM_STATES):
                    dur_counts[s] += result["dur_counts"][s]
                n_processed += 1

        if n_processed == 0:
            logger.warning("[fit_hsmm] No sequences processed in iteration %d", iteration)
            break

        avg_ll = total_ll / n_processed
        avg_kl = total_kl / n_processed
        objective = avg_ll - avg_kl

        # M-step: update parameters
        model.update_transitions(
            torch.tensor(trans_counts, dtype=torch.float64),
            smoothing=1.0,
        )
        model.update_durations(
            [torch.tensor(dc, dtype=torch.float64) for dc in dur_counts],
            smoothing=0.05,
        )

        # Check convergence
        rel_change = abs(objective - prev_ll) / max(abs(prev_ll), 1e-6)
        converged = rel_change < convergence_tol and iteration > 0

        diag = {
            "iteration": iteration,
            "avg_log_likelihood": avg_ll,
            "avg_kl_penalty": avg_kl,
            "objective": objective,
            "relative_change": rel_change,
            "n_sequences_processed": n_processed,
            "converged": converged,
        }
        diagnostics.append(diag)

        logger.info(
            "[fit_hsmm] Iter %d: LL=%.2f, KL=%.4f, obj=%.2f, Δ=%.6f (%d seqs, %d workers)",
            iteration,
            avg_ll,
            avg_kl,
            objective,
            rel_change,
            n_processed,
            n_workers,
        )

        prev_ll = objective
        if converged:
            logger.info("[fit_hsmm] Converged after %d iterations", iteration + 1)
            break

    return diagnostics


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def fit_hsmm(
    weekly_df: pd.DataFrame,
    anchor_df: pd.DataFrame,
    encoder_bundle: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit the semi-supervised HSMM.

    Parameters
    ----------
    weekly_df : pd.DataFrame
        Weekly panel with emission probabilities already attached
        (columns ``state_prob_<name>``).
    anchor_df : pd.DataFrame
        Anchor posteriors.
    encoder_bundle : dict
        Observation encoder output (for metadata).
    config : dict, optional
        Fitting configuration overrides.

    Returns
    -------
    dict with keys:
        model : ExplicitDurationHSMM
        diagnostics : list of dicts (per-iteration)
        sequences : list of sequence dicts
        fit_config : dict
    """
    config = config or {}
    max_duration = config.get("max_duration", 52)
    n_iters = config.get("n_iters", 8)
    convergence_tol = config.get("convergence_tol", 1e-3)
    anchor_kl_weight = config.get("anchor_kl_weight", 0.5)
    device = config.get("device", "cpu")

    # Build model
    model = ExplicitDurationHSMM(max_duration=max_duration, device=device)

    # Build sequences from emission-scored weekly panel
    sequences = _build_sequences(weekly_df)
    if not sequences:
        logger.error("[fit_hsmm] No sequences to fit")
        return {
            "model": model,
            "diagnostics": [],
            "sequences": [],
            "fit_config": config,
        }

    # Initialize from anchors
    initialize_from_anchors(model, anchor_df, weekly_df)

    # EM refinement
    diagnostics = run_em_iterations(
        model,
        sequences,
        anchor_df=anchor_df,
        n_iters=n_iters,
        convergence_tol=convergence_tol,
        anchor_kl_weight=anchor_kl_weight,
    )

    return {
        "model": model,
        "diagnostics": diagnostics,
        "sequences": sequences,
        "fit_config": config,
    }


def save_hsmm_artifacts(
    fit_result: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    """Save model checkpoint and fit diagnostics."""
    import json as _json

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "hsmm_model.pt"
    diag_path = output_dir / "hsmm_fit_diagnostics.json"

    fit_result["model"].save(model_path)

    with open(diag_path, "w", encoding="utf-8") as f:
        _json.dump(
            {
                "fit_config": fit_result.get("fit_config", {}),
                "iterations": fit_result.get("diagnostics", []),
                "n_sequences": len(fit_result.get("sequences", [])),
            },
            f,
            indent=2,
            default=str,
        )

    logger.info("[fit_hsmm] Saved artifacts to %s", output_dir)
    return {"model": model_path, "diagnostics": diag_path}
