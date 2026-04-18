"""Anchor posterior generation for the latent state model.

Converts event types, remarks scores, and obvious end states into soft
anchor posteriors over the 9 latent states.  This replaces the discrete
``create_state_anchor_labels()`` in ``voyage_state_model.py`` with the
spec's logit-space combination + posterior regularization targets.

Design principle: generate *soft*, not hard, labels.  Combine multiple
anchor sources in logit space, renormalize, and cap certainty unless a
truly terminal anchor (C/F) is present.

**Performance note**: Uses vectorized numpy operations on the whole DataFrame
rather than row-by-row Python loops.
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

# Uniform logit baseline
_UNIFORM_LOGITS = np.zeros(NUM_STATES, dtype=np.float64)

# Maximum certainty cap for non-terminal anchors
_MAX_CERTAINTY_NONTERMINAL = 0.92
_MAX_CERTAINTY_TERMINAL = 0.99


# ---------------------------------------------------------------------------
# Vectorized anchor computation
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


def build_anchor_posteriors(
    weekly_df: pd.DataFrame,
    config: Any = None,
) -> pd.DataFrame:
    """Generate soft anchor posteriors for each voyage-week (vectorized).

    Parameters
    ----------
    weekly_df : pd.DataFrame
        Weekly observation panel.
    config : WSLReliabilityConfig, optional
        Reserved for future tuning.

    Returns
    -------
    pd.DataFrame
        One row per voyage-week with columns ``state_prior_<name>``,
        ``anchor_strength``, ``anchor_sources``, ``is_terminal_anchor``,
        ``n_anchor_rules_fired``.
    """
    N = len(weekly_df)
    index = weekly_df.index

    # ---- Parse inputs ----
    weeks = weekly_df.get("weeks_since_departure", pd.Series(0, index=index)).fillna(0).astype(int)
    flow_frac = weekly_df.get("source_mode_flow_fraction", pd.Series(0.5, index=index)).fillna(0.5)
    distress = weekly_df.get("distress_severity_max", pd.Series(0.0, index=index)).fillna(0.0)
    productivity = weekly_df.get("productivity_polarity_mean", pd.Series(0.0, index=index)).fillna(0.0)
    oil_delta = weekly_df.get("oil_delta_per_elapsed_week", pd.Series(0.0, index=index)).fillna(0.0)
    primary_class = weekly_df.get("primary_class_mode", pd.Series("", index=index)).fillna("").astype(str)

    evt_df = _parse_event_counts(weekly_df.get("event_type_counts_json", pd.Series({}, index=index)))

    dep_count = evt_df.get("dep", pd.Series(0.0, index=index)).values
    arr_count = evt_df.get("arr", pd.Series(0.0, index=index)).values
    wrk_count = evt_df.get("wrk", pd.Series(0.0, index=index)).values
    inp_count = evt_df.get("inp", pd.Series(0.0, index=index)).values
    spk_count = evt_df.get("spk", pd.Series(0.0, index=index)).values
    rpt_count = evt_df.get("rpt", pd.Series(0.0, index=index)).values

    weeks_arr = weeks.values
    flow_arr = flow_frac.values
    distress_arr = distress.values
    productivity_arr = productivity.values
    oil_delta_arr = oil_delta.values
    pc_arr = primary_class.values

    # ---- Build logit matrix [N, K] ----
    logits = np.zeros((N, NUM_STATES), dtype=np.float64)
    sources = [[] for _ in range(N)]
    is_terminal = np.zeros(N, dtype=bool)

    # Rule 1: departure → outbound (early weeks only)
    is_dep = (dep_count > 0) & (flow_arr > 0.5) & (weeks_arr <= 4)
    logits[is_dep, STATE_INDEX["outbound_initial_transit"]] += 3.0
    for i in np.where(is_dep)[0]:
        sources[i].append("departure")

    # Rule 2: wreck or terminal_loss class → terminal failure
    is_wrk = (wrk_count > 0) | (pc_arr == "terminal_loss")
    logits[is_wrk, STATE_INDEX["terminal_loss"]] += 5.0
    is_terminal[is_wrk] = True
    for i in np.where(is_wrk)[0]:
        sources[i].append("terminal_wreck")

    # Rule 3: arrival without severe distress → completed
    is_arr = (arr_count > 0) & (distress_arr < 2) & (pc_arr != "terminal_loss")
    logits[is_arr, STATE_INDEX["completed_arrival"]] += 3.5
    is_terminal[is_arr] = True
    for i in np.where(is_arr)[0]:
        sources[i].append("arrival")

    # Rule 4: in-port or repair class → interrupted
    is_inp = (inp_count > 0) | (pc_arr == "interruption_repair")
    logits[is_inp, STATE_INDEX["in_port_interruption_or_repair"]] += 2.0
    for i in np.where(is_inp)[0]:
        sources[i].append("inport_repair")

    # Rule 5: homebound class
    is_home = pc_arr == "homebound_or_termination"
    logits[is_home, STATE_INDEX["homebound_or_terminated"]] += 2.2
    for i in np.where(is_home)[0]:
        sources[i].append("homebound")

    # Rule 6: high distress severity
    is_distress = (distress_arr >= 2) | (pc_arr == "distress_hazard")
    logits[is_distress, STATE_INDEX["distress_at_sea"]] += 2.2
    for i in np.where(is_distress)[0]:
        sources[i].append("distress")

    # Rule 7: low yield (mid-voyage, weak/empty productivity class)
    is_low_yield = (pc_arr == "weak_or_empty_productivity") & (weeks_arr >= 8)
    logits[is_low_yield, STATE_INDEX["low_yield_or_stalled_search"]] += 1.5
    for i in np.where(is_low_yield)[0]:
        sources[i].append("low_yield")

    # Rule 8: strong positive productivity (mid-voyage)
    is_productive = (
        (pc_arr == "positive_productivity") |
        ((productivity_arr > 0.5) & (oil_delta_arr > 0))
    ) & (weeks_arr >= 4)
    logits[is_productive, STATE_INDEX["productive_search"]] += 1.5
    for i in np.where(is_productive)[0]:
        sources[i].append("productive")

    # Rule 9: neutral search (spoken/reported mid-voyage without other signals)
    is_neutral = (
        ((spk_count > 0) | (rpt_count > 0)) &
        (distress_arr < 1.5) &
        (weeks_arr >= 4) &
        ~np.isin(pc_arr, [
            "terminal_loss", "distress_hazard", "interruption_repair",
            "homebound_or_termination", "positive_productivity",
        ])
    )
    logits[is_neutral, STATE_INDEX["active_search_neutral"]] += 1.2
    for i in np.where(is_neutral)[0]:
        sources[i].append("neutral_search")

    # ---- Softmax and certainty cap ----
    # log-sum-exp normalize
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
