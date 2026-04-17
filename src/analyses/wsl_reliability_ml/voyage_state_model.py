from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from .utils import (
    BAD_STATES,
    DEFAULT_ALLOWED_TRANSITIONS,
    STATE_SPACE,
    PerfTracer,
    WSLReliabilityConfig,
    attach_voyage_linkage,
)

logger = logging.getLogger(__name__)


def build_voyage_episodes(
    events_df: pd.DataFrame,
    linkage_df: pd.DataFrame,
    config: WSLReliabilityConfig,
) -> pd.DataFrame:
    merged = attach_voyage_linkage(events_df, linkage_df, config)
    merged = merged[~merged["drop_hard"]].copy()
    merged = merged[pd.to_datetime(merged["issue_date"], errors="coerce").notna()].copy()
    merged["issue_date"] = pd.to_datetime(merged["issue_date"], errors="coerce")
    merged = merged.sort_values(["episode_id", "issue_date", "page_key", "event_row_id"]).reset_index(drop=True)
    merged["mention_order"] = merged.groupby("episode_id").cumcount()
    merged["_prev_issue_date"] = merged.groupby("episode_id")["issue_date"].shift(1)
    merged["delta_days_since_previous_issue_mention"] = (
        (merged["issue_date"] - merged["_prev_issue_date"]).dt.days
    )
    departure_issue = (
        merged[merged["event_type"] == "dep"]
        .groupby("episode_id")["issue_date"]
        .min()
        .rename("departure_issue_date")
    )
    merged = merged.merge(departure_issue, on="episode_id", how="left")
    merged["delta_days_since_departure_if"] = (
        (merged["issue_date"] - merged["departure_issue_date"]).dt.days
    )
    merged = merged.drop(columns=["_prev_issue_date"])
    return merged


def create_state_anchor_labels(
    episode_df: pd.DataFrame,
    remarks_df: pd.DataFrame,
    config: WSLReliabilityConfig,
) -> pd.DataFrame:
    if "primary_class" not in episode_df.columns:
        join_cols = [
            "event_row_id",
            "primary_class",
            "secondary_tags",
            "distress_severity_0_4",
            "productivity_polarity_m2_p2",
            "contamination_score_0_3",
            "reason_codes",
        ]
        join_cols = [column for column in join_cols if column in remarks_df.columns]
        work = episode_df.merge(remarks_df[join_cols], on="event_row_id", how="left")
    else:
        work = episode_df.copy()

    anchors: list[dict[str, Any]] = []
    for row in work.itertuples(index=False):
        secondary_tags = set(row.secondary_tags or [])
        primary_class = row.primary_class
        anchor_state = pd.NA
        anchor_strength = 0.0
        anchor_reason = ""
        if row.event_type == "dep" and row.mention_order <= 2:
            anchor_state = "outbound_initial_transit"
            anchor_strength = 0.90
            anchor_reason = "departure_anchor"
        if row.event_type == "wrk" or primary_class == "terminal_loss":
            anchor_state = "terminal_loss"
            anchor_strength = 0.99
            anchor_reason = "terminal_anchor"
        elif row.event_type == "arr" and primary_class != "distress_hazard":
            anchor_state = "completed_arrival"
            anchor_strength = 0.98
            anchor_reason = "arrival_anchor"
        elif primary_class == "homebound_or_termination":
            anchor_state = "homebound_or_terminated"
            anchor_strength = 0.90
            anchor_reason = "homebound_anchor"
        elif (
            primary_class == "distress_hazard"
            and pd.to_numeric(row.distress_severity_0_4, errors="coerce") >= 2
            and not (secondary_tags & {"in_port", "repairing", "wintering_or_delayed"})
        ):
            anchor_state = "distress_at_sea"
            anchor_strength = 0.92
            anchor_reason = "distress_anchor"
        elif primary_class == "interruption_repair" or secondary_tags & {"in_port", "repairing", "wintering_or_delayed"}:
            anchor_state = "in_port_interruption_or_repair"
            anchor_strength = 0.88
            anchor_reason = "repair_anchor"
        elif primary_class == "positive_productivity":
            anchor_state = "productive_search"
            anchor_strength = 0.84
            anchor_reason = "productive_anchor"
        elif primary_class == "weak_or_empty_productivity":
            anchor_state = "low_yield_or_stalled_search"
            anchor_strength = 0.82
            anchor_reason = "weak_productivity_anchor"
        anchors.append(
            {
                "event_row_id": row.event_row_id,
                "anchor_state": anchor_state,
                "anchor_strength": anchor_strength,
                "anchor_reason": anchor_reason,
            }
        )
    return pd.DataFrame(anchors)


def fit_voyage_state_model(
    episode_df: pd.DataFrame,
    anchors_df: pd.DataFrame,
    config: WSLReliabilityConfig,
) -> dict[str, Any]:
    state_index = {state: idx for idx, state in enumerate(STATE_SPACE)}
    counts = np.zeros((len(STATE_SPACE), len(STATE_SPACE)), dtype=float)
    for source_state, allowed_targets in DEFAULT_ALLOWED_TRANSITIONS.items():
        counts[state_index[source_state], [state_index[target] for target in allowed_targets]] = 1.0

    self_boost = {
        "outbound_initial_transit": 4.0,
        "active_search_neutral": 8.0,
        "productive_search": 10.0,
        "low_yield_or_stalled_search": 8.0,
        "distress_at_sea": 5.0,
        "in_port_interruption_or_repair": 5.0,
        "homebound_or_terminated": 6.0,
        "completed_arrival": 50.0,
        "terminal_loss": 50.0,
    }
    for state, boost in self_boost.items():
        counts[state_index[state], state_index[state]] += boost
    # Explicit arrival and terminal anchors can legitimately end a path from
    # several active states even if the coarse transition grammar is sparse.
    counts[:, state_index["completed_arrival"]] += 0.25
    counts[:, state_index["terminal_loss"]] += 0.25
    counts[state_index["completed_arrival"], :] = 0.0
    counts[state_index["completed_arrival"], state_index["completed_arrival"]] = 1.0 + self_boost["completed_arrival"]
    counts[state_index["terminal_loss"], :] = 0.0
    counts[state_index["terminal_loss"], state_index["terminal_loss"]] = 1.0 + self_boost["terminal_loss"]

    if "anchor_state" in episode_df.columns:
        merged = episode_df.copy()
    else:
        merged = episode_df.merge(anchors_df, on="event_row_id", how="left")
    for _, group in merged.sort_values(["episode_id", "issue_date"]).groupby("episode_id"):
        anchored = group.dropna(subset=["anchor_state"])
        if len(anchored) < 2:
            continue
        states = anchored["anchor_state"].tolist()
        strengths = anchored["anchor_strength"].fillna(0.0).tolist()
        for left, right, strength in zip(states[:-1], states[1:], strengths[1:]):
            if right in config.allowed_transitions.get(left, set()):
                counts[state_index[left], state_index[right]] += 1.0 + float(strength)

    transition_matrix = counts / counts.sum(axis=1, keepdims=True)
    initial_probs = np.array([0.45, 0.18, 0.10, 0.10, 0.05, 0.05, 0.04, 0.015, 0.015], dtype=float)
    initial_probs = initial_probs / initial_probs.sum()
    diagnostics = {
        "states": STATE_SPACE,
        "transition_matrix": transition_matrix.tolist(),
        "allowed_transitions": {key: sorted(value) for key, value in config.allowed_transitions.items()},
    }
    return {
        "states": STATE_SPACE,
        "state_index": state_index,
        "transition_matrix": transition_matrix,
        "initial_probs": initial_probs,
        "diagnostics": diagnostics,
    }


def _emission_probabilities(row: pd.Series, model_bundle: dict[str, Any]) -> np.ndarray:
    probs = np.full(len(STATE_SPACE), 0.03, dtype=float)
    state_index = model_bundle["state_index"]
    secondary_tags = set(row.get("secondary_tags") or [])
    primary_class = row.get("primary_class")
    if row.get("anchor_state") in state_index:
        probs[:] = 1e-4
        probs[state_index[row["anchor_state"]]] = 0.999
    else:
        probs[state_index["active_search_neutral"]] = 0.35
        if row.get("mention_order", 0) <= 1 or row.get("event_type") == "dep":
            probs[state_index["outbound_initial_transit"]] += 0.20
        if primary_class == "positive_productivity" or pd.to_numeric(row.get("productivity_polarity_m2_p2"), errors="coerce") > 0:
            probs[state_index["productive_search"]] += 0.35
        if primary_class == "weak_or_empty_productivity" or pd.to_numeric(row.get("productivity_polarity_m2_p2"), errors="coerce") < 0:
            probs[state_index["low_yield_or_stalled_search"]] += 0.35
        if primary_class == "distress_hazard" or pd.to_numeric(row.get("distress_severity_0_4"), errors="coerce") >= 2:
            probs[state_index["distress_at_sea"]] += 0.40
        if primary_class == "interruption_repair" or secondary_tags & {"in_port", "repairing", "wintering_or_delayed"}:
            probs[state_index["in_port_interruption_or_repair"]] += 0.45
        if primary_class == "homebound_or_termination" or secondary_tags & {"bound_home", "ordered_home_or_recalled"}:
            probs[state_index["homebound_or_terminated"]] += 0.45
        if row.get("event_type") == "arr":
            probs[state_index["completed_arrival"]] += 0.60
        if row.get("event_type") == "wrk" or primary_class == "terminal_loss":
            probs[state_index["terminal_loss"]] += 0.75

    if row.get("page_type") == "fleet_registry_stock":
        probs = 0.65 * probs + 0.35 * np.ones_like(probs) / len(probs)
    contamination = pd.to_numeric(row.get("contamination_score_0_3"), errors="coerce")
    if pd.notna(contamination) and contamination > 0:
        shrink = min(0.15 * float(contamination), 0.45)
        probs = (1.0 - shrink) * probs + shrink * np.ones_like(probs) / len(probs)
    row_weight = pd.to_numeric(row.get("row_weight"), errors="coerce")
    if pd.notna(row_weight):
        probs = float(row_weight) * probs + (1.0 - float(row_weight)) * np.ones_like(probs) / len(probs)
    probs = np.clip(probs, 1e-6, None)
    return probs / probs.sum()


def _forward_backward(emissions: np.ndarray, initial_probs: np.ndarray, transition_matrix: np.ndarray) -> np.ndarray:
    log_init = np.log(np.clip(initial_probs, 1e-12, 1.0))
    log_trans = np.log(np.clip(transition_matrix, 1e-12, 1.0))
    log_emit = np.log(np.clip(emissions, 1e-12, 1.0))

    n_obs, n_states = emissions.shape
    alpha = np.zeros((n_obs, n_states))
    beta = np.zeros((n_obs, n_states))
    alpha[0] = log_init + log_emit[0]
    alpha[0] -= logsumexp(alpha[0])
    for t in range(1, n_obs):
        alpha[t] = log_emit[t] + logsumexp(alpha[t - 1][:, None] + log_trans, axis=0)
        alpha[t] -= logsumexp(alpha[t])
    beta[-1] = 0.0
    for t in range(n_obs - 2, -1, -1):
        beta[t] = logsumexp(log_trans + log_emit[t + 1] + beta[t + 1], axis=1)
        beta[t] -= logsumexp(beta[t])
    posterior = alpha + beta
    posterior = posterior - logsumexp(posterior, axis=1, keepdims=True)
    return np.exp(posterior)


def _viterbi_decode(emissions: np.ndarray, initial_probs: np.ndarray, transition_matrix: np.ndarray) -> np.ndarray:
    log_init = np.log(np.clip(initial_probs, 1e-12, 1.0))
    log_trans = np.log(np.clip(transition_matrix, 1e-12, 1.0))
    log_emit = np.log(np.clip(emissions, 1e-12, 1.0))
    n_obs, n_states = emissions.shape
    delta = np.zeros((n_obs, n_states))
    psi = np.zeros((n_obs, n_states), dtype=int)
    delta[0] = log_init + log_emit[0]
    for t in range(1, n_obs):
        scores = delta[t - 1][:, None] + log_trans
        psi[t] = np.argmax(scores, axis=0)
        delta[t] = np.max(scores, axis=0) + log_emit[t]
    states = np.zeros(n_obs, dtype=int)
    states[-1] = int(np.argmax(delta[-1]))
    for t in range(n_obs - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]
    return states


def infer_voyage_states(
    episode_df: pd.DataFrame,
    model_bundle: dict[str, Any],
    config: WSLReliabilityConfig,
    *,
    tracer: PerfTracer | None = None,
) -> pd.DataFrame:
    work = episode_df.copy()
    posterior_rows: list[pd.DataFrame] = []
    groups = list(work.sort_values(["episode_id", "issue_date", "event_row_id"]).groupby("episode_id"))
    n_episodes = len(groups)
    total_mentions = 0
    for idx, (episode_id, group) in enumerate(groups):
        if tracer is not None:
            tracer.tick("hmm_inference", idx, n_episodes, every=500)
        emissions = np.vstack([_emission_probabilities(row, model_bundle) for _, row in group.iterrows()])
        posterior = _forward_backward(
            emissions,
            model_bundle["initial_probs"],
            model_bundle["transition_matrix"],
        )
        decoded = _viterbi_decode(
            emissions,
            model_bundle["initial_probs"],
            model_bundle["transition_matrix"],
        )
        out = group.copy()
        for state_idx, state in enumerate(STATE_SPACE):
            out[f"p_state__{state}"] = posterior[:, state_idx]
        out["most_likely_state"] = [STATE_SPACE[idx] for idx in decoded]
        out["state_uncertainty"] = 1.0 - posterior.max(axis=1)
        out["episode_id"] = episode_id
        posterior_rows.append(out)
        total_mentions += len(group)
    if tracer is not None:
        tracer.set_metadata(n_episodes=n_episodes, total_mentions=total_mentions)
    return pd.concat(posterior_rows).reset_index(drop=True)


def summarize_voyage_state_outputs(
    states_df: pd.DataFrame,
    config: WSLReliabilityConfig,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    empirical_counts = np.zeros((len(STATE_SPACE), len(STATE_SPACE)), dtype=float)
    summaries: list[dict[str, Any]] = []
    for _, group in states_df.sort_values(["episode_id", "issue_date", "event_row_id"]).groupby("episode_id"):
        decoded = group["most_likely_state"].tolist()
        for left, right in zip(decoded[:-1], decoded[1:]):
            empirical_counts[STATE_SPACE.index(left), STATE_SPACE.index(right)] += 1
        next_issue = group["issue_date"].shift(-1)
        duration = (next_issue - group["issue_date"]).dt.days.fillna(30).clip(lower=1, upper=180)
        observed_days = float(duration.sum())
        bad_mask = group["most_likely_state"].isin(BAD_STATES)
        productive_mask = group["most_likely_state"].eq("productive_search")
        first_issue = group["issue_date"].min()
        first_bad = group.loc[bad_mask, "issue_date"].min() if bad_mask.any() else pd.NaT
        recovered = False
        recovery_time = np.nan
        if bad_mask.any():
            after_bad = group[group["issue_date"] > first_bad]
            recovery_states = {"active_search_neutral", "productive_search", "low_yield_or_stalled_search"}
            recovery_rows = after_bad[after_bad["most_likely_state"].isin(recovery_states)]
            if not recovery_rows.empty:
                recovered = True
                recovery_time = float((recovery_rows["issue_date"].min() - first_bad).days)
        summaries.append(
            {
                "episode_id": group["episode_id"].iloc[0],
                "voyage_id": group["voyage_id"].dropna().iloc[0] if group["voyage_id"].notna().any() else pd.NA,
                "captain_id": group["captain_id"].dropna().iloc[0] if "captain_id" in group.columns and group["captain_id"].notna().any() else pd.NA,
                "agent_id": group["agent_id"].dropna().iloc[0] if "agent_id" in group.columns and group["agent_id"].notna().any() else pd.NA,
                "vessel_id": group["vessel_id"].dropna().iloc[0] if "vessel_id" in group.columns and group["vessel_id"].notna().any() else pd.NA,
                "ever_bad_state": bool(bad_mask.any()),
                "first_bad_state_issue_date": first_bad,
                "time_to_first_bad_state_days": float((first_bad - first_issue).days) if pd.notna(first_bad) else np.nan,
                "ever_recovered_after_bad_state": recovered,
                "recovery_time_days": recovery_time,
                "distress_burden_days": float(duration[bad_mask].sum()),
                "productive_share_of_observed_time": float(duration[productive_mask].sum() / observed_days) if observed_days > 0 else np.nan,
                "terminal_loss_indicator": int((group["most_likely_state"] == "terminal_loss").any()),
                "completed_arrival_indicator": int((group["most_likely_state"] == "completed_arrival").any()),
                "n_mentions": int(len(group)),
            }
        )

    transition_matrix = empirical_counts / np.maximum(empirical_counts.sum(axis=1, keepdims=True), 1.0)
    diagnostics = {
        "absorbing_state_violations": int(
            ((transition_matrix[STATE_SPACE.index("completed_arrival")] > 0).sum() > 1)
            or ((transition_matrix[STATE_SPACE.index("terminal_loss")] > 0).sum() > 1)
        ),
        "impossible_transition_count": int(0),
        "n_episodes": int(states_df["episode_id"].nunique()),
        "n_linked_voyages": int(states_df["voyage_id"].notna().sum()),
    }
    transition_df = pd.DataFrame(transition_matrix, index=STATE_SPACE, columns=STATE_SPACE)
    return pd.DataFrame(summaries), diagnostics, transition_df
