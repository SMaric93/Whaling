"""Decode weekly state paths and export per-voyage sequences with uncertainty.

**Performance note**: Decoding is parallelized across voyages using
``concurrent.futures.ProcessPoolExecutor``.

Usage::

    from .decode_paths import decode_all_voyages
    posterior_df, viterbi_df = decode_all_voyages(model_bundle, sequences)
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd
import torch

from .hsmm import ExplicitDurationHSMM, prepare_sequence
from .state_space import NUM_STATES, STATE_NAMES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker function
# ---------------------------------------------------------------------------


def _decode_single_sequence(args: tuple) -> dict[str, Any] | None:
    """Decode a single sequence in a worker process."""
    (
        emission_probs,
        quality_weights,
        week_indices,
        voyage_id,
        T,
        model_state_dict,
        max_duration,
        decode_mode,
    ) = args

    try:
        model = ExplicitDurationHSMM(max_duration=max_duration, device="cpu")
        model.load_state_dict(model_state_dict)
        model.eval()

        prepared = prepare_sequence(
            emission_probs,
            quality_weights,
            device="cpu",
        )

        posterior_rows = []
        viterbi_rows = []

        # Posterior decode
        if decode_mode in ("both", "posterior"):
            with torch.no_grad():
                posterior = model.posterior_decode(
                    prepared["emission_logprobs"],
                    prepared.get("quality_weights"),
                )
            post_np = posterior.cpu().numpy()

            for t in range(T):
                row: dict[str, Any] = {
                    "voyage_id": voyage_id,
                    "week_idx": int(week_indices[t]),
                }
                for si, sname in enumerate(STATE_NAMES):
                    row[f"posterior_{sname}"] = float(post_np[t, si])
                row["posterior_map_state"] = STATE_NAMES[int(post_np[t].argmax())]
                entropy = -np.sum(
                    post_np[t] * np.log(np.clip(post_np[t], 1e-12, 1.0))
                )
                row["posterior_entropy"] = float(entropy)
                row["decoder_version"] = "hsmm_v1"
                posterior_rows.append(row)

        # Viterbi decode
        if decode_mode in ("both", "viterbi"):
            with torch.no_grad():
                path = model.viterbi_decode(
                    prepared["emission_logprobs"],
                    prepared.get("quality_weights"),
                )

            for t in range(T):
                row = {
                    "voyage_id": voyage_id,
                    "week_idx": int(week_indices[t]),
                    "viterbi_state": STATE_NAMES[int(path[t])],
                    "viterbi_state_idx": int(path[t]),
                    "decoder_version": "hsmm_v1",
                }
                viterbi_rows.append(row)

        return {
            "posterior_rows": posterior_rows,
            "viterbi_rows": viterbi_rows,
            "ok": True,
        }

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main decode function
# ---------------------------------------------------------------------------


def decode_all_voyages(
    model_bundle: dict[str, Any],
    sequences: list[dict[str, Any]],
    decode_mode: str = "both",
    n_workers: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decode weekly paths for all voyages (parallelized).

    Parameters
    ----------
    model_bundle : dict
        Output of ``fit_hsmm`` containing the fitted HSMM model.
    sequences : list of dict
        Per-voyage sequence data from ``_build_sequences``.
    decode_mode : str
        ``"both"`` (default), ``"posterior"``, or ``"viterbi"``.
    n_workers : int, optional
        Number of worker processes. Defaults to min(cpu_count, 8).

    Returns
    -------
    posterior_df : pd.DataFrame
        One row per voyage-week with posterior probabilities per state.
    viterbi_df : pd.DataFrame
        One row per voyage-week with MAP state assignment.
    """
    if n_workers is None:
        n_workers = min(os.cpu_count() or 4, 8)

    model: ExplicitDurationHSMM = model_bundle["model"]
    model.eval()

    # Serialize model state once
    model_state = model.state_dict()
    max_duration = model.max_duration

    # Build worker arguments
    worker_args = [
        (
            seq["emission_probs"],
            seq["quality_weights"],
            seq["week_indices"],
            seq["voyage_id"],
            seq["T"],
            model_state,
            max_duration,
            decode_mode,
        )
        for seq in sequences
    ]

    posterior_rows: list[dict[str, Any]] = []
    viterbi_rows: list[dict[str, Any]] = []
    n_decoded = 0
    n_failed = 0
    n_sequences = len(sequences)

    logger.info("[decode] Starting parallel decode of %d voyages with %d workers", n_sequences, n_workers)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_decode_single_sequence, arg): i
            for i, arg in enumerate(worker_args)
        }

        for future in as_completed(futures):
            result = future.result()
            if result is None or not result.get("ok"):
                n_failed += 1
                continue

            posterior_rows.extend(result["posterior_rows"])
            viterbi_rows.extend(result["viterbi_rows"])
            n_decoded += 1

            if n_decoded % 500 == 0 or n_decoded == n_sequences:
                logger.info(
                    "[decode] %d/%d voyages decoded (%d failed)",
                    n_decoded,
                    n_sequences,
                    n_failed,
                )

    posterior_df = pd.DataFrame(posterior_rows)
    viterbi_df = pd.DataFrame(viterbi_rows)

    logger.info(
        "[decode] Complete: %d voyages decoded, %d failed, %d posterior rows, %d viterbi rows",
        n_decoded,
        n_failed,
        len(posterior_df),
        len(viterbi_df),
    )

    return posterior_df, viterbi_df


def export_posterior_paths(
    posterior_df: pd.DataFrame,
    output_path: Any,
) -> None:
    """Save posterior paths to parquet."""
    from pathlib import Path

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    posterior_df.to_parquet(path, index=False)
    logger.info("[decode] Exported posterior paths to %s (%d rows)", path, len(posterior_df))


def export_viterbi_paths(
    viterbi_df: pd.DataFrame,
    output_path: Any,
) -> None:
    """Save Viterbi paths to parquet."""
    from pathlib import Path

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    viterbi_df.to_parquet(path, index=False)
    logger.info("[decode] Exported Viterbi paths to %s (%d rows)", path, len(viterbi_df))
