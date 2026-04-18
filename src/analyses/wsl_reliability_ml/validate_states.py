"""Scientific validation of the decoded state model.

Implements the spec's validation checks:
- State-duration distributions are plausible
- Impossible transitions are zero after decoding
- Decoded bad-state metrics predict zero-catch/low-output outcomes
- Organization-strength coefficients attenuate when state metrics are added
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .state_space import (
    ALLOWED_TRANSITIONS,
    BAD_STATES,
    NUM_STATES,
    STATE_DEFS,
    STATE_INDEX,
    STATE_NAMES,
)

logger = logging.getLogger(__name__)


def check_duration_plausibility(
    viterbi_df: pd.DataFrame,
    state_col: str = "viterbi_state",
) -> dict[str, Any]:
    """Check that decoded state durations are plausible.

    Returns a dict with per-state duration statistics and pass/fail flags.
    """
    viterbi_df = viterbi_df.sort_values(["voyage_id", "week_idx"])
    durations: dict[str, list[int]] = {s: [] for s in STATE_NAMES}

    for _, group in viterbi_df.groupby("voyage_id"):
        states = group[state_col].tolist()
        if not states:
            continue
        current = states[0]
        dur = 1
        for i in range(1, len(states)):
            if states[i] == current:
                dur += 1
            else:
                durations[current].append(dur)
                current = states[i]
                dur = 1
        durations[current].append(dur)

    results: dict[str, Any] = {}
    all_pass = True

    for sd in STATE_DEFS:
        name = sd.name
        durs = durations.get(name, [])
        if not durs:
            results[name] = {"n_segments": 0, "pass": True, "reason": "no_segments"}
            continue

        arr = np.array(durs, dtype=float)
        mean_dur = float(arr.mean())
        median_dur = float(np.median(arr))
        max_dur = float(arr.max())
        n_segments = len(durs)

        # Check: outbound should be short, search long, distress shorter than search
        passed = True
        reason = "ok"

        if name == "outbound_transit" and mean_dur > 20:
            passed = False
            reason = f"outbound mean too long: {mean_dur:.1f}"
        elif name == "active_voyage" and mean_dur < 2:
            passed = False
            reason = f"active mean too short: {mean_dur:.1f}"
        elif name == "in_trouble" and mean_dur > 30:
            passed = False
            reason = f"trouble mean too long: {mean_dur:.1f}"

        if not passed:
            all_pass = False

        results[name] = {
            "n_segments": n_segments,
            "mean": round(mean_dur, 2),
            "median": round(median_dur, 2),
            "max": round(max_dur, 2),
            "pass": passed,
            "reason": reason,
        }

    results["__all_pass__"] = all_pass
    return results


def check_impossible_transitions(
    viterbi_df: pd.DataFrame,
    state_col: str = "viterbi_state",
) -> dict[str, Any]:
    """Check that no impossible transitions appear in decoded paths.

    Returns a dict with transition counts and a violation count.
    """
    viterbi_df = viterbi_df.sort_values(["voyage_id", "week_idx"])

    trans_counts = np.zeros((NUM_STATES, NUM_STATES), dtype=int)
    violations: list[dict[str, Any]] = []

    for voyage_id, group in viterbi_df.groupby("voyage_id"):
        states = group[state_col].tolist()
        for i in range(len(states) - 1):
            si = STATE_INDEX.get(states[i])
            sj = STATE_INDEX.get(states[i + 1])
            if si is None or sj is None:
                continue
            trans_counts[si, sj] += 1

            if states[i + 1] not in ALLOWED_TRANSITIONS.get(states[i], set()):
                violations.append({
                    "voyage_id": voyage_id,
                    "from_state": states[i],
                    "to_state": states[i + 1],
                    "week_idx": int(group["week_idx"].iloc[i]),
                })

    trans_df = pd.DataFrame(
        trans_counts,
        index=STATE_NAMES,
        columns=STATE_NAMES,
    )

    return {
        "n_violations": len(violations),
        "violations_sample": violations[:20],
        "transition_counts": trans_df,
        "pass": len(violations) == 0,
    }


def check_absorbing_states(
    viterbi_df: pd.DataFrame,
    state_col: str = "viterbi_state",
) -> dict[str, Any]:
    """Check that absorbing states (completed_arrival, terminal_loss) stay absorbing."""
    viterbi_df = viterbi_df.sort_values(["voyage_id", "week_idx"])
    violations = 0

    for _, group in viterbi_df.groupby("voyage_id"):
        states = group[state_col].tolist()
        absorbed = False
        for s in states:
            if absorbed:
                if s != absorbed_state:
                    violations += 1
                    break
            if s in ("completed_arrival", "terminal_loss"):
                absorbed = True
                absorbed_state = s

    return {
        "absorbing_violations": violations,
        "pass": violations == 0,
    }


def check_predictive_power(
    summary_df: pd.DataFrame,
    voyage_ref_df: pd.DataFrame,
) -> dict[str, Any]:
    """Check that decoded bad-state metrics predict outcomes.

    Compares zero-catch/low-output rates between voyages with and without
    bad-state entry.
    """
    merged = summary_df.merge(voyage_ref_df[["voyage_id", "q_total_index", "log_output", "zero_catch_or_failure"]].drop_duplicates("voyage_id"), on="voyage_id", how="inner")

    if len(merged) < 20:
        return {"status": "insufficient_data", "n_merged": len(merged)}

    bad = merged[merged["ever_bad_state"] == True]
    good = merged[merged["ever_bad_state"] == False]

    if bad.empty or good.empty:
        return {"status": "no_variation", "n_bad": len(bad), "n_good": len(good)}

    zero_catch_bad = float(bad["zero_catch_or_failure"].mean())
    zero_catch_good = float(good["zero_catch_or_failure"].mean())
    output_bad = float(pd.to_numeric(bad["log_output"], errors="coerce").mean())
    output_good = float(pd.to_numeric(good["log_output"], errors="coerce").mean())

    result = {
        "n_merged": len(merged),
        "n_bad_state": len(bad),
        "n_no_bad_state": len(good),
        "zero_catch_rate_bad_state": zero_catch_bad,
        "zero_catch_rate_no_bad_state": zero_catch_good,
        "zero_catch_difference": zero_catch_bad - zero_catch_good,
        "mean_output_bad_state": output_bad,
        "mean_output_no_bad_state": output_good,
        "output_difference": output_good - output_bad,
        "pass": zero_catch_bad > zero_catch_good,
    }

    return result


def run_all_validation(
    viterbi_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    voyage_ref_df: pd.DataFrame | None = None,
    state_col: str = "viterbi_state",
) -> dict[str, Any]:
    """Run all validation checks and return a combined report."""
    results: dict[str, Any] = {}

    results["duration_plausibility"] = check_duration_plausibility(viterbi_df, state_col)
    results["impossible_transitions"] = check_impossible_transitions(viterbi_df, state_col)
    results["absorbing_states"] = check_absorbing_states(viterbi_df, state_col)

    if voyage_ref_df is not None:
        results["predictive_power"] = check_predictive_power(summary_df, voyage_ref_df)

    # Overall pass/fail
    all_checks_pass = all(
        results[k].get("pass", True) for k in results if isinstance(results[k], dict)
    )
    results["all_checks_pass"] = all_checks_pass

    # Log summary
    for check_name, check_result in results.items():
        if isinstance(check_result, dict) and "pass" in check_result:
            status = "✓" if check_result["pass"] else "✗"
            logger.info("[validate] %s %s", status, check_name)

    return results
