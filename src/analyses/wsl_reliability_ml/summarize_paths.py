"""Convert decoded state paths into econometric features.

Produces the spec's ``state_summary_table``: per-voyage features for
bad-state entry, duration, recovery, and terminal outcomes.

Usage::

    from .summarize_paths import summarize_state_features
    summary_df = summarize_state_features(decoded_df, config)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .state_space import BAD_STATES, STATE_INDEX, STATE_NAMES

logger = logging.getLogger(__name__)

# Recovery states: any search state is considered "recovered"
_RECOVERY_STATES = {
    "outbound_initial_transit",
    "active_search_neutral",
    "productive_search",
    "low_yield_or_stalled_search",
}


def _summarize_one_voyage(
    group: pd.DataFrame,
    state_col: str = "viterbi_state",
) -> dict[str, Any]:
    """Compute state summary features for one voyage's decoded path."""
    voyage_id = group["voyage_id"].iloc[0]
    states = group[state_col].tolist()
    weeks = group["week_idx"].to_numpy(dtype=int)
    T = len(states)

    # Basic trajectory properties
    first_week = int(weeks[0]) if T > 0 else np.nan
    last_week = int(weeks[-1]) if T > 0 else np.nan
    total_weeks = int(last_week - first_week + 1) if T > 0 else 0

    # Bad state detection
    bad_mask = [s in BAD_STATES for s in states]
    ever_distress = any(s == "distress_at_sea" for s in states)
    ever_interrupted = any(s == "in_port_interruption_or_repair" for s in states)
    ever_bad = any(bad_mask)

    # First occurrence of each concerning state
    first_distress_week = np.nan
    first_interruption_week = np.nan
    first_low_yield_week = np.nan
    first_bad_week = np.nan

    for i, s in enumerate(states):
        w = int(weeks[i])
        if s == "distress_at_sea" and np.isnan(first_distress_week):
            first_distress_week = w
        if s == "in_port_interruption_or_repair" and np.isnan(first_interruption_week):
            first_interruption_week = w
        if s == "low_yield_or_stalled_search" and np.isnan(first_low_yield_week):
            first_low_yield_week = w
        if bad_mask[i] and np.isnan(first_bad_week):
            first_bad_week = w

    # State durations (in weeks)
    distress_duration = sum(1 for s in states if s == "distress_at_sea")
    interruption_duration = sum(1 for s in states if s == "in_port_interruption_or_repair")
    bad_state_duration = sum(bad_mask)

    # Recovery detection: did the voyage return to a search state after being in a bad state?
    recovered_from_distress = False
    recovery_week = np.nan
    if ever_bad:
        first_bad_idx = next(i for i, b in enumerate(bad_mask) if b)
        for i in range(first_bad_idx + 1, T):
            if states[i] in _RECOVERY_STATES:
                recovered_from_distress = True
                recovery_week = int(weeks[i])
                break

    # Homebound early: entered homebound before typical voyage completion
    entered_homebound_early = False
    for i, s in enumerate(states):
        if s == "homebound_or_terminated" and int(weeks[i]) < total_weeks * 0.7:
            entered_homebound_early = True
            break

    # Terminal outcomes
    terminal_failure = states[-1] == "terminal_loss" if T > 0 else False
    completed_successfully = states[-1] == "completed_arrival" if T > 0 else False

    # Productive share
    productive_weeks = sum(1 for s in states if s == "productive_search")
    productive_share = productive_weeks / max(T, 1)

    # State entropy (diversity of states visited)
    state_counts = {}
    for s in states:
        state_counts[s] = state_counts.get(s, 0) + 1
    probs_arr = np.array(list(state_counts.values()), dtype=np.float64)
    probs_arr = probs_arr / probs_arr.sum()
    state_entropy = float(-np.sum(probs_arr * np.log(np.clip(probs_arr, 1e-12, 1.0))))

    return {
        "voyage_id": voyage_id,
        "n_observed_weeks": T,
        "first_week": first_week,
        "last_week": last_week,
        "total_span_weeks": total_weeks,
        # Low yield
        "first_low_yield_week": first_low_yield_week,
        # Distress
        "first_distress_week": first_distress_week,
        "ever_distress": ever_distress,
        "distress_duration_weeks": distress_duration,
        # Interruption
        "first_interruption_week": first_interruption_week,
        "ever_interrupted": ever_interrupted,
        "interruption_duration_weeks": interruption_duration,
        # Bad state (composite)
        "first_bad_state_week": first_bad_week,
        "ever_bad_state": ever_bad,
        "bad_state_duration_weeks": bad_state_duration,
        # Recovery
        "recovered_from_distress": recovered_from_distress,
        "recovery_week": recovery_week,
        "recovery_time_weeks": float(recovery_week - first_bad_week)
        if not np.isnan(recovery_week) and not np.isnan(first_bad_week)
        else np.nan,
        # Homebound
        "entered_homebound_early": entered_homebound_early,
        # Terminal
        "terminal_failure": terminal_failure,
        "completed_successfully": completed_successfully,
        # Productivity
        "productive_share": productive_share,
        # Entropy
        "state_entropy_mean": state_entropy,
        # Terminal state
        "terminal_state": states[-1] if T > 0 else None,
    }


def summarize_state_features(
    decoded_df: pd.DataFrame,
    config: dict[str, Any] | None = None,
    *,
    state_col: str = "viterbi_state",
) -> pd.DataFrame:
    """Convert decoded paths into econometric features keyed by ``voyage_id``.

    Parameters
    ----------
    decoded_df : pd.DataFrame
        Viterbi-decoded paths (from ``decode_all_voyages``).
    config : dict, optional
        Configuration overrides.
    state_col : str
        Name of the decoded state column.

    Returns
    -------
    pd.DataFrame
        One row per voyage with summary features.
    """
    if decoded_df.empty:
        logger.warning("[summarize] Empty decoded paths; returning empty summary")
        return pd.DataFrame()

    # Ensure sorted
    decoded_df = decoded_df.sort_values(["voyage_id", "week_idx"])

    summaries: list[dict[str, Any]] = []
    for voyage_id, group in decoded_df.groupby("voyage_id"):
        summary = _summarize_one_voyage(group, state_col=state_col)
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)

    logger.info(
        "[summarize] %d voyage summaries: ever_bad=%.1f%%, terminal_failure=%.1f%%, completed=%.1f%%",
        len(summary_df),
        100.0 * summary_df["ever_bad_state"].mean() if len(summary_df) > 0 else 0,
        100.0 * summary_df["terminal_failure"].mean() if len(summary_df) > 0 else 0,
        100.0 * summary_df["completed_successfully"].mean() if len(summary_df) > 0 else 0,
    )

    return summary_df


def compute_bad_state_entry(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Extract bad-state entry timing for survival analysis."""
    return summary_df[["voyage_id", "first_bad_state_week", "ever_bad_state", "total_span_weeks"]].copy()


def compute_recovery_metrics(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Extract recovery metrics for econometric analysis."""
    cols = [
        "voyage_id",
        "ever_bad_state",
        "recovered_from_distress",
        "recovery_time_weeks",
        "first_bad_state_week",
        "recovery_week",
    ]
    return summary_df[[c for c in cols if c in summary_df.columns]].copy()


def compute_terminal_outcomes(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Extract terminal outcome indicators."""
    cols = [
        "voyage_id",
        "terminal_failure",
        "completed_successfully",
        "terminal_state",
        "total_span_weeks",
    ]
    return summary_df[[c for c in cols if c in summary_df.columns]].copy()
