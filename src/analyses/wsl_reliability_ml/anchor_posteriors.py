"""Data-driven anchor posteriors for the 5-state HSMM.

Uses oil quantities, remarks taxonomy probabilities, event type counts,
and temporal position to generate strong soft anchors over the 5 latent
states.  Every feature in the weekly panel that carries state-discriminative
signal is directly incorporated.

Design: accumulate logit scores per state from multiple evidence sources,
then softmax-normalize with certainty caps.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd

from .state_space import (
    BAD_STATE_INDICES,
    NUM_STATES,
    STATE_INDEX,
    STATE_NAMES,
)

logger = logging.getLogger(__name__)

# Certainty caps
_MAX_CERTAINTY_NONTERMINAL = 0.92
_MAX_CERTAINTY_TERMINAL = 0.99


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_event_counts(series: pd.Series) -> pd.DataFrame:
    """Parse event_type_counts_json column into per-event-type count columns."""
    all_types = ["dep", "arr", "spk", "rpt", "inp", "wrk"]
    cols = {et: np.zeros(len(series), dtype=float) for et in all_types}
    for i, val in enumerate(series):
        if isinstance(val, str):
            try:
                d = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                d = {}
        elif isinstance(val, dict):
            d = val
        else:
            d = {}
        for et in all_types:
            cols[et][i] = d.get(et, 0)
    return pd.DataFrame(cols, index=series.index)


def _safe_col(df: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    """Extract a column as numpy array, filling missing values."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default).values
    return np.full(len(df), default, dtype=np.float64)


def build_anchor_posteriors(
    weekly_df: pd.DataFrame,
    config: Any = None,
) -> pd.DataFrame:
    """Generate soft anchor posteriors for each voyage-week.

    Uses ALL available panel features: oil quantities, remarks taxonomy
    probabilities, event types, temporal position, reporting gaps, and
    distress severity.

    Parameters
    ----------
    weekly_df : pd.DataFrame
        Weekly observation panel (output of ``build_weekly_observation_panel``).

    Returns
    -------
    pd.DataFrame
        One row per voyage-week with columns ``state_prior_<name>``,
        ``anchor_strength``, ``anchor_sources``, etc.
    """
    N = len(weekly_df)
    index = weekly_df.index

    # ---- Extract all features ----
    weeks = _safe_col(weekly_df, "weeks_since_departure")
    weeks_since_obs = _safe_col(weekly_df, "weeks_since_last_observation")
    flow_frac = _safe_col(weekly_df, "source_mode_flow_fraction", 0.5)
    distress_sev = _safe_col(weekly_df, "distress_severity_max")
    productivity = _safe_col(weekly_df, "productivity_polarity_mean")
    oil_delta = _safe_col(weekly_df, "oil_delta_per_elapsed_week")
    oil_total = _safe_col(weekly_df, "oil_total")
    oil_sperm = _safe_col(weekly_df, "oil_sperm_bbls")
    oil_whale = _safe_col(weekly_df, "oil_whale_bbls")
    n_events = _safe_col(weekly_df, "n_events")
    confidence = _safe_col(weekly_df, "mean_confidence", 0.5)

    # Remarks taxonomy probabilities
    p_distress = _safe_col(weekly_df, "p_primary__distress_hazard")
    p_terminal = _safe_col(weekly_df, "p_primary__terminal_loss")
    p_productive = _safe_col(weekly_df, "p_primary__positive_productivity")
    p_weak = _safe_col(weekly_df, "p_primary__weak_or_empty_productivity")
    p_repair = _safe_col(weekly_df, "p_primary__interruption_repair")
    p_homebound = _safe_col(weekly_df, "p_primary__homebound_or_termination")
    p_routine = _safe_col(weekly_df, "p_primary__routine_info")
    p_commercial = _safe_col(weekly_df, "p_primary__commercial_admin_status")
    p_assistance = _safe_col(weekly_df, "p_primary__assistance_transfer_coordination")

    # Tag probabilities
    p_bound_home = _safe_col(weekly_df, "p_tag__bound_home")
    p_in_port = _safe_col(weekly_df, "p_tag__in_port")
    p_no_whales = _safe_col(weekly_df, "p_tag__no_whales_or_clean")
    p_spoken = _safe_col(weekly_df, "p_tag__spoken_or_seen")
    p_whales = _safe_col(weekly_df, "p_tag__whales_sighted")

    primary_class = weekly_df.get(
        "primary_class_mode", pd.Series("", index=index)
    ).fillna("").astype(str).values

    evt_df = _parse_event_counts(
        weekly_df.get("event_type_counts_json", pd.Series({}, index=index))
    )
    dep_count = evt_df["dep"].values
    arr_count = evt_df["arr"].values
    inp_count = evt_df["inp"].values
    wrk_count = evt_df["wrk"].values
    spk_count = evt_df.get("spk", pd.Series(0.0, index=index)).values
    rpt_count = evt_df.get("rpt", pd.Series(0.0, index=index)).values

    # Per-voyage last observed week (for terminal anchor)
    voyage_ids = weekly_df["voyage_id"].values
    week_idx = weekly_df["week_idx"].values.astype(int)
    # Compute max week per voyage
    vmax = weekly_df.groupby("voyage_id")["week_idx"].transform("max").values.astype(int)

    # ---- Build logit matrix [N, K] ----
    logits = np.zeros((N, NUM_STATES), dtype=np.float64)
    sources = [[] for _ in range(N)]
    is_terminal = np.zeros(N, dtype=bool)

    IDX_O = STATE_INDEX["outbound_transit"]
    IDX_A = STATE_INDEX["active_voyage"]
    IDX_T = STATE_INDEX["in_trouble"]
    IDX_F = STATE_INDEX["terminal_loss"]
    IDX_C = STATE_INDEX["completed_arrival"]

    # ================================================================
    # RULE 1: OUTBOUND TRANSIT — early weeks, departure events
    # ================================================================
    is_dep = (dep_count > 0) & (flow_frac > 0.4) & (weeks <= 6)
    logits[is_dep, IDX_O] += 3.5
    for i in np.where(is_dep)[0]:
        sources[i].append("departure")

    # Early weeks without other strong signals → outbound
    is_early = (weeks <= 4) & (arr_count == 0) & (wrk_count == 0) & (oil_total <= 0)
    logits[is_early, IDX_O] += 1.5
    for i in np.where(is_early & ~is_dep)[0]:
        sources[i].append("early_week")

    # ================================================================
    # RULE 2: TERMINAL LOSS — wreck events, loss remarks
    # ================================================================
    is_wrk = (wrk_count > 0) | (primary_class == "terminal_loss")
    logits[is_wrk, IDX_F] += 6.0
    is_terminal[is_wrk] = True
    for i in np.where(is_wrk)[0]:
        sources[i].append("wreck_event")

    # High terminal-loss taxonomy probability
    is_loss_remarks = (p_terminal > 0.3) & ~is_wrk
    logits[is_loss_remarks, IDX_F] += 3.0 * p_terminal[is_loss_remarks]
    for i in np.where(is_loss_remarks)[0]:
        sources[i].append("loss_remarks")

    # ================================================================
    # RULE 3: COMPLETED ARRIVAL — arrival at end of record
    # ================================================================
    # Arrival events near the end of the voyage's observed record
    is_final_arr = (arr_count > 0) & (week_idx >= vmax - 2) & (distress_sev < 2)
    logits[is_final_arr, IDX_C] += 5.0
    is_terminal[is_final_arr] = True
    for i in np.where(is_final_arr)[0]:
        sources[i].append("final_arrival")

    # Mid-voyage arrivals (at whaling ports) — weaker signal, still active
    is_mid_arr = (arr_count > 0) & (week_idx < vmax - 2) & ~is_wrk
    logits[is_mid_arr, IDX_A] += 1.0
    for i in np.where(is_mid_arr)[0]:
        sources[i].append("mid_voyage_arr")

    # Homebound remarks near end → completed
    is_homebound_end = (p_bound_home > 0.3) & (week_idx >= vmax - 4)
    logits[is_homebound_end, IDX_C] += 2.5 * p_bound_home[is_homebound_end]
    for i in np.where(is_homebound_end)[0]:
        sources[i].append("homebound_end")

    # ================================================================
    # RULE 4: IN TROUBLE — distress, repair, extended silence
    # ================================================================
    # High distress severity
    is_distress = (distress_sev >= 2) | (primary_class == "distress_hazard")
    logits[is_distress, IDX_T] += 3.5
    for i in np.where(is_distress)[0]:
        sources[i].append("distress")

    # Distress taxonomy probability
    has_distress_prob = (p_distress > 0.25) & ~is_distress
    logits[has_distress_prob, IDX_T] += 2.5 * p_distress[has_distress_prob]
    for i in np.where(has_distress_prob)[0]:
        sources[i].append("distress_prob")

    # In-port repair events with repair taxonomy
    is_repair = (inp_count > 0) & ((p_repair > 0.2) | (primary_class == "interruption_repair"))
    logits[is_repair, IDX_T] += 2.5
    for i in np.where(is_repair)[0]:
        sources[i].append("repair")

    # Assistance/transfer (often signals trouble)
    is_assist = p_assistance > 0.3
    logits[is_assist, IDX_T] += 1.5 * p_assistance[is_assist]
    for i in np.where(is_assist)[0]:
        sources[i].append("assistance")

    # Extended reporting gap (>= 8 weeks since last observation) — silence as signal
    is_gap = weeks_since_obs >= 8
    logits[is_gap, IDX_T] += np.minimum(weeks_since_obs[is_gap] / 8.0, 3.0)
    for i in np.where(is_gap)[0]:
        sources[i].append("silence_gap")

    # ================================================================
    # RULE 5: ACTIVE VOYAGE — the core mid-voyage state
    # ================================================================
    # Spoken/reported events with oil data → strong active signal
    has_oil_report = (
        ((spk_count > 0) | (rpt_count > 0)) &
        (oil_total > 0) &
        (weeks > 4)
    )
    logits[has_oil_report, IDX_A] += 3.0
    for i in np.where(has_oil_report)[0]:
        sources[i].append("oil_report")

    # Positive oil delta → actively accumulating catch
    is_gaining = (oil_delta > 0) & (weeks > 4)
    logits[is_gaining, IDX_A] += 2.0 * np.minimum(oil_delta[is_gaining] / 50.0, 1.5)
    for i in np.where(is_gaining)[0]:
        sources[i].append("oil_gain")

    # Productive remarks
    is_prod = (p_productive > 0.2) & (weeks > 3)
    logits[is_prod, IDX_A] += 2.0 * p_productive[is_prod]
    for i in np.where(is_prod)[0]:
        sources[i].append("productive_remarks")

    # Whales sighted tag
    is_whales = (p_whales > 0.2) & (weeks > 3)
    logits[is_whales, IDX_A] += 1.5 * p_whales[is_whales]
    for i in np.where(is_whales)[0]:
        sources[i].append("whales_sighted")

    # Routine info mid-voyage (coordinates, "heard from") → active
    is_routine = (p_routine > 0.3) & (weeks > 4) & (distress_sev < 1.5)
    logits[is_routine, IDX_A] += 1.5
    for i in np.where(is_routine)[0]:
        sources[i].append("routine_info")

    # In-port without repair → resupply, still active
    is_port_active = (inp_count > 0) & (p_repair < 0.15) & ~is_repair
    logits[is_port_active, IDX_A] += 1.5
    for i in np.where(is_port_active)[0]:
        sources[i].append("port_resupply")

    # Spoken/reported mid-voyage without other signals
    is_spoken = (
        ((spk_count > 0) | (rpt_count > 0)) &
        (distress_sev < 1.5) &
        (weeks > 4) &
        ~has_oil_report &
        ~is_wrk
    )
    logits[is_spoken, IDX_A] += 1.5
    for i in np.where(is_spoken)[0]:
        sources[i].append("spoken")

    # Commercial/admin status → active
    is_commercial = (p_commercial > 0.3) & (weeks > 3) & ~is_wrk
    logits[is_commercial, IDX_A] += 1.0
    for i in np.where(is_commercial)[0]:
        sources[i].append("commercial")

    # ---- Softmax and certainty cap ----
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True) + 1e-30

    # Cap certainty for non-terminal rows
    nonterminal = ~is_terminal
    if nonterminal.any():
        max_probs = probs[nonterminal].max(axis=1)
        excess = np.maximum(max_probs - _MAX_CERTAINTY_NONTERMINAL, 0.0)
        idx_max = probs[nonterminal].argmax(axis=1)
        for j, (i_global, ex) in enumerate(zip(np.where(nonterminal)[0], excess)):
            if ex > 0:
                probs[i_global, idx_max[j]] = _MAX_CERTAINTY_NONTERMINAL
                probs[i_global, :] += ex / (NUM_STATES - 1)
                probs[i_global, idx_max[j]] -= ex / (NUM_STATES - 1)
        probs[nonterminal] = np.clip(probs[nonterminal], 1e-6, 1.0)
        probs[nonterminal] /= probs[nonterminal].sum(axis=1, keepdims=True)

    # Rows with no anchors → uniform
    no_anchor = np.array([len(s) == 0 for s in sources])
    probs[no_anchor] = 1.0 / NUM_STATES

    # ---- Assemble output ----
    anchor_strength = probs.max(axis=1)
    n_rules_fired = np.array([len(s) for s in sources])
    source_strs = [";".join(s) if s else "none" for s in sources]

    result = pd.DataFrame(index=index)
    result["voyage_id"] = weekly_df["voyage_id"].values
    result["week_idx"] = weekly_df["week_idx"].values
    result["anchor_strength"] = anchor_strength
    result["anchor_sources"] = source_strs
    result["is_terminal_anchor"] = is_terminal
    result["n_anchor_rules_fired"] = n_rules_fired

    for si, sname in enumerate(STATE_NAMES):
        result[f"state_prior_{sname}"] = probs[:, si]

    n_anchored = int((n_rules_fired > 0).sum())
    logger.info(
        "[anchors] Generated posteriors for %d voyage-weeks; %d with anchor evidence (%.1f%%)",
        N,
        n_anchored,
        100.0 * n_anchored / max(N, 1),
    )

    return result


def combine_anchor_sources(anchor_df: pd.DataFrame) -> pd.DataFrame:
    """Return the anchor posterior matrix alongside voyage/week identifiers."""
    prior_cols = [f"state_prior_{s}" for s in STATE_NAMES]
    return anchor_df[["voyage_id", "week_idx"] + prior_cols].copy()


def posterior_regularization_targets(
    anchor_df: pd.DataFrame,
    *,
    min_strength: float = 0.5,
) -> pd.DataFrame:
    """Filter anchors to those strong enough for posterior regularization."""
    return anchor_df[anchor_df["anchor_strength"] >= min_strength].copy()
