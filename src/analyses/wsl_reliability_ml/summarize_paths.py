"""Convert decoded state paths into econometric features (5-state model).

Terminal outcome classification uses a hybrid approach:
- Mid-voyage states come from the HSMM Viterbi path
- Terminal classification (completed vs. lost) is determined by anchor
  evidence at the voyage boundary, since the HSMM absorbing states
  are difficult to enter with only 1-2 weeks of terminal evidence.

Usage::

    from .summarize_paths import summarize_state_features
    summary_df = summarize_state_features(decoded_df, anchor_df=anchor_df)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .state_space import BAD_STATES, STATE_INDEX, STATE_NAMES

logger = logging.getLogger(__name__)

# Recovery states: returning to active or outbound is recovery
_RECOVERY_STATES = {
    "outbound_transit",
    "active_voyage",
}


def _classify_terminal_from_anchors(
    voyage_id: Any,
    anchor_group: pd.DataFrame | None,
) -> str:
    """Determine terminal outcome from anchor evidence at voyage boundary.

    Looks at the last 3 weeks of anchor posteriors to classify:
    - 'completed_arrival' if strong arrival evidence
    - 'terminal_loss' if strong loss evidence
    - 'active_voyage' if no terminal evidence (censored/unknown)
    """
    if anchor_group is None or anchor_group.empty:
        return "active_voyage"

    # Last 3 weeks of the voyage
    tail = anchor_group.sort_values("week_idx").tail(3)

    # Check for terminal_loss evidence
    loss_col = "state_prior_terminal_loss"
    arrival_col = "state_prior_completed_arrival"

    if loss_col in tail.columns:
        max_loss = tail[loss_col].max()
        if max_loss > 0.5:
            return "terminal_loss"

    if arrival_col in tail.columns:
        max_arrival = tail[arrival_col].max()
        if max_arrival > 0.5:
            return "completed_arrival"

    # Check if terminal anchor was flagged
    if "is_terminal_anchor" in tail.columns and tail["is_terminal_anchor"].any():
        # Which terminal state is stronger?
        mean_loss = tail[loss_col].mean() if loss_col in tail.columns else 0
        mean_arr = tail[arrival_col].mean() if arrival_col in tail.columns else 0
        if mean_loss > mean_arr:
            return "terminal_loss"
        elif mean_arr > 0.3:
            return "completed_arrival"

    return "active_voyage"  # censored — no terminal evidence


def _summarize_one_voyage(
    group: pd.DataFrame,
    anchor_group: pd.DataFrame | None,
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
    ever_trouble = any(s == "in_trouble" for s in states)
    ever_bad = any(bad_mask)

    # First occurrence of trouble state
    first_trouble_week = np.nan
    first_bad_week = np.nan

    for i, s in enumerate(states):
        w = int(weeks[i])
        if s == "in_trouble" and np.isnan(first_trouble_week):
            first_trouble_week = w
        if bad_mask[i] and np.isnan(first_bad_week):
            first_bad_week = w

    # State durations (in decoded weeks)
    trouble_duration = sum(1 for s in states if s == "in_trouble")
    bad_state_duration = sum(bad_mask)
    outbound_duration = sum(1 for s in states if s == "outbound_transit")
    active_duration = sum(1 for s in states if s == "active_voyage")

    # Recovery detection: did the voyage return to active after trouble?
    recovered_from_trouble = False
    recovery_week = np.nan
    if ever_trouble:
        first_trouble_idx = next(i for i, s in enumerate(states) if s == "in_trouble")
        for i in range(first_trouble_idx + 1, T):
            if states[i] in _RECOVERY_STATES:
                recovered_from_trouble = True
                recovery_week = int(weeks[i])
                break

    # Terminal outcomes — hybrid: Viterbi path + anchor evidence
    viterbi_terminal = states[-1] if T > 0 else None
    if viterbi_terminal in ("terminal_loss", "completed_arrival"):
        # Viterbi already picked a terminal state — trust it
        terminal_state = viterbi_terminal
    else:
        # Viterbi didn't pick a terminal state — use anchor evidence
        terminal_state = _classify_terminal_from_anchors(voyage_id, anchor_group)

    terminal_failure = terminal_state == "terminal_loss"
    completed_successfully = terminal_state == "completed_arrival"

    # Shares
    active_share = active_duration / max(T, 1)
    outbound_share = outbound_duration / max(T, 1)
    trouble_share = trouble_duration / max(T, 1)

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
        # Outbound
        "outbound_duration_weeks": outbound_duration,
        "outbound_share": outbound_share,
        # Active voyage
        "active_duration_weeks": active_duration,
        "active_share": active_share,
        # Trouble
        "first_trouble_week": first_trouble_week,
        "ever_trouble": ever_trouble,
        "trouble_duration_weeks": trouble_duration,
        "trouble_share": trouble_share,
        # Bad state (composite: in_trouble + terminal_loss)
        "first_bad_state_week": first_bad_week,
        "ever_bad_state": ever_bad,
        "bad_state_duration_weeks": bad_state_duration,
        # Recovery
        "recovered_from_trouble": recovered_from_trouble,
        "recovery_week": recovery_week,
        "recovery_time_weeks": float(recovery_week - first_trouble_week)
        if not np.isnan(recovery_week) and not np.isnan(first_trouble_week)
        else np.nan,
        # Terminal (hybrid Viterbi + anchor)
        "terminal_failure": terminal_failure,
        "completed_successfully": completed_successfully,
        "terminal_state": terminal_state,
        # Viterbi terminal (raw, before anchor override)
        "viterbi_terminal_state": viterbi_terminal,
        # Entropy
        "state_entropy_mean": state_entropy,
    }


def summarize_state_features(
    decoded_df: pd.DataFrame,
    config: dict[str, Any] | None = None,
    *,
    state_col: str = "viterbi_state",
    anchor_df: pd.DataFrame | None = None,
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
    anchor_df : pd.DataFrame, optional
        Anchor posteriors for hybrid terminal classification.
    """
    if decoded_df.empty:
        logger.warning("[summarize] Empty decoded paths; returning empty summary")
        return pd.DataFrame()

    decoded_df = decoded_df.sort_values(["voyage_id", "week_idx"])

    # Pre-group anchors by voyage for O(1) lookup
    anchor_groups: dict[Any, pd.DataFrame] = {}
    if anchor_df is not None:
        for vid, grp in anchor_df.groupby("voyage_id"):
            anchor_groups[vid] = grp

    summaries: list[dict[str, Any]] = []
    for voyage_id, group in decoded_df.groupby("voyage_id"):
        ag = anchor_groups.get(voyage_id)
        summary = _summarize_one_voyage(group, ag, state_col=state_col)
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)

    # Log stats
    n = len(summary_df)
    if n > 0:
        n_trouble = summary_df["ever_trouble"].sum()
        n_fail = summary_df["terminal_failure"].sum()
        n_complete = summary_df["completed_successfully"].sum()
        n_censored = n - n_fail - n_complete
        n_overridden = (summary_df["terminal_state"] != summary_df["viterbi_terminal_state"]).sum()

        logger.info(
            "[summarize] %d voyages: completed=%d (%.1f%%), terminal_loss=%d (%.1f%%), "
            "censored=%d (%.1f%%), ever_trouble=%d (%.1f%%), anchor_overrides=%d",
            n,
            n_complete, 100.0 * n_complete / n,
            n_fail, 100.0 * n_fail / n,
            n_censored, 100.0 * n_censored / n,
            n_trouble, 100.0 * n_trouble / n,
            n_overridden,
        )

    return summary_df


def compute_bad_state_entry(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Extract bad-state entry timing for survival analysis."""
    return summary_df[["voyage_id", "first_bad_state_week", "ever_bad_state", "total_span_weeks"]].copy()


def compute_recovery_metrics(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Extract recovery metrics for econometric analysis."""
    cols = [
        "voyage_id",
        "ever_trouble",
        "recovered_from_trouble",
        "recovery_time_weeks",
        "first_trouble_week",
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
