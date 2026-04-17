from __future__ import annotations

import logging
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import (
    BAD_STATES,
    PerfTracer,
    WSLReliabilityConfig,
    attach_voyage_linkage,
    expected_basin,
    infer_basin_probabilities,
    save_dataframe,
    stable_hash,
    write_json,
)

logger = logging.getLogger(__name__)


def _weighted_entropy(values: pd.Series, weights: np.ndarray) -> float:
    if values.empty:
        return math.nan
    work = pd.DataFrame({"value": values.astype(str), "weight": weights})
    collapsed = work.groupby("value", observed=True)["weight"].sum()
    probs = collapsed / max(collapsed.sum(), 1e-12)
    return float(-(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum())


def _window_slice(frame: pd.DataFrame, t0: pd.Timestamp, tau_days: int) -> tuple[pd.DataFrame, np.ndarray]:
    if frame.empty or pd.isna(t0):
        return frame.iloc[0:0].copy(), np.array([], dtype=float)
    delta = (t0 - frame["issue_date"]).dt.days
    mask = (delta >= 0) & (delta <= tau_days)
    window = frame.loc[mask].reset_index(drop=True).copy()
    if window.empty:
        return window, np.array([], dtype=float)
    window_delta = (t0 - window["issue_date"]).dt.days.to_numpy(dtype=float)
    weights = window["row_weight"].to_numpy(dtype=float) * np.exp(-window_delta / float(tau_days))
    return window, weights


def _summarize_signal_window(window: pd.DataFrame, weights: np.ndarray) -> dict[str, float]:
    if window.empty:
        return {
            "reports": 0.0,
            "distress_mass": 0.0,
            "positive_mass": 0.0,
            "weak_mass": 0.0,
            "unique_vessels": 0.0,
            "signal_entropy": math.nan,
            "port_diversity": 0.0,
            "mean_confidence": math.nan,
        }
    vessel_col = "vessel_id" if "vessel_id" in window.columns and window["vessel_id"].notna().any() else "vessel_name_norm"
    unique_vessels = float(window[vessel_col].fillna("UNK").nunique())
    port_diversity = float(window["home_port_norm"].fillna("UNK").nunique()) if "home_port_norm" in window.columns else 0.0
    return {
        "reports": float(weights.sum()),
        "distress_mass": float(weights[window["signal_distress"].to_numpy(dtype=bool)].sum()),
        "positive_mass": float(weights[window["signal_positive"].to_numpy(dtype=bool)].sum()),
        "weak_mass": float(weights[window["signal_weak"].to_numpy(dtype=bool)].sum()),
        "unique_vessels": unique_vessels,
        "signal_entropy": _weighted_entropy(window["primary_class"].fillna("missing"), weights),
        "port_diversity": port_diversity,
        "mean_confidence": float(pd.to_numeric(window["_confidence"], errors="coerce").fillna(0).mean()),
    }


def _prepare_signal_frame(prior_events_df: pd.DataFrame) -> pd.DataFrame:
    signal_df = prior_events_df.copy()
    signal_df["issue_date"] = pd.to_datetime(signal_df["issue_date"], errors="coerce")
    signal_df = signal_df[signal_df["issue_date"].notna()].copy()
    signal_df["destination_basin"] = signal_df["destination_basin"].fillna(
        signal_df["destination_basin_probs"].map(expected_basin) if "destination_basin_probs" in signal_df.columns else "Unknown"
    )
    signal_df["signal_distress"] = signal_df["primary_class"].isin(["distress_hazard", "terminal_loss"])
    if "most_likely_state" in signal_df.columns:
        signal_df["signal_distress"] = signal_df["signal_distress"] | signal_df["most_likely_state"].isin(BAD_STATES)
    signal_df["signal_positive"] = signal_df["primary_class"].eq("positive_productivity")
    signal_df["signal_weak"] = signal_df["primary_class"].eq("weak_or_empty_productivity")
    signal_df["source_is_registry"] = signal_df["page_type"].eq("fleet_registry_stock")
    signal_df["issue_ord"] = signal_df["issue_date"].astype("int64") // 10**9
    return signal_df


def build_departure_panel(
    events_df: pd.DataFrame,
    linkage_df: pd.DataFrame,
    config: WSLReliabilityConfig,
) -> pd.DataFrame:
    merged = attach_voyage_linkage(events_df, linkage_df, config)
    departures = merged[(merged["event_type"] == "dep") & merged["issue_date"].notna()].copy()
    departures["departure_issue_date"] = pd.to_datetime(departures["issue_date"], errors="coerce")
    departures["departure_destination_basin_probs"] = departures["destination"].map(infer_basin_probabilities)
    departures["departure_destination_basin"] = departures["departure_destination_basin_probs"].map(expected_basin)
    departures["departure_key"] = np.where(
        departures["voyage_id"].notna(),
        departures["voyage_id"].astype(str),
        departures["episode_id"].astype(str),
    )
    departures = departures.sort_values(["departure_key", "departure_issue_date", "event_row_id"]).drop_duplicates("departure_key")
    departures["departure_id"] = [
        stable_hash([row.departure_key, row.event_row_id], prefix="departure_")
        for row in departures.itertuples(index=False)
    ]
    keep = [
        "departure_id",
        "event_row_id",
        "voyage_id",
        "episode_id",
        "captain_id",
        "agent_id",
        "vessel_id",
        "departure_issue_date",
        "issue_date",
        "home_port",
        "home_port_norm",
        "port",
        "port_norm",
        "agent",
        "agent_norm",
        "captain",
        "captain_norm",
        "vessel_name",
        "vessel_name_norm",
        "destination",
        "departure_destination_basin",
        "departure_destination_basin_probs",
        "row_weight",
        "_confidence",
        "page_type",
        "page_key",
    ]
    keep = [column for column in keep if column in departures.columns]
    departures = departures[keep].copy()
    departures["departure_year"] = departures["departure_issue_date"].dt.year
    departures["departure_decade"] = (departures["departure_year"] // 10) * 10
    departures["matched_to_voyage"] = departures["voyage_id"].notna()
    return departures.reset_index(drop=True)


def _prior_voyage_metrics(signal_df: pd.DataFrame) -> pd.DataFrame:
    if "voyage_id" not in signal_df.columns:
        return pd.DataFrame(columns=["voyage_id"])
    rows: list[dict[str, Any]] = []
    for voyage_id, group in signal_df.dropna(subset=["voyage_id"]).sort_values("issue_date").groupby("voyage_id"):
        bad_mask = group["most_likely_state"].isin(BAD_STATES) if "most_likely_state" in group.columns else group["signal_distress"]
        first_bad = group.loc[bad_mask, "issue_date"].min() if bad_mask.any() else pd.NaT
        after_bad = group[group["issue_date"] > first_bad] if pd.notna(first_bad) else group.iloc[0:0]
        recovered = (
            after_bad["most_likely_state"].isin({"active_search_neutral", "productive_search", "low_yield_or_stalled_search"}).any()
            if "most_likely_state" in after_bad.columns
            else False
        )
        rows.append(
            {
                "voyage_id": voyage_id,
                "agent_id": group["agent_id"].dropna().iloc[0] if "agent_id" in group.columns and group["agent_id"].notna().any() else pd.NA,
                "first_issue_date": group["issue_date"].min(),
                "first_bad_state_issue_date": first_bad,
                "ever_bad_state": bool(bad_mask.any()),
                "ever_recovered_after_bad_state": bool(recovered),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Numpy-accelerated inner loop
# ---------------------------------------------------------------------------

_EPOCH = pd.Timestamp("1800-01-01")

_EMPTY_STATS: dict[str, float] = {
    "reports": 0.0,
    "distress_mass": 0.0,
    "positive_mass": 0.0,
    "weak_mass": 0.0,
    "unique_vessels": 0.0,
    "signal_entropy": math.nan,
    "port_diversity": 0.0,
    "mean_confidence": math.nan,
}


def _group_to_arrays(df: pd.DataFrame) -> dict[str, Any] | None:
    """Convert a pre-sorted (by issue_date) group DataFrame to numpy arrays.

    Returns ``None`` for empty DataFrames (sentinel for missing groups).
    """
    if df.empty:
        return None
    vessel_col = (
        "vessel_id"
        if "vessel_id" in df.columns and df["vessel_id"].notna().any()
        else "vessel_name_norm"
    )
    return {
        "n": len(df),
        "issue_days": (df["issue_date"] - _EPOCH).dt.days.to_numpy(dtype=np.int64),
        "row_weight": df["row_weight"].to_numpy(dtype=np.float64),
        "distress": df["signal_distress"].to_numpy(dtype=bool),
        "positive": df["signal_positive"].to_numpy(dtype=bool),
        "weak": df["signal_weak"].to_numpy(dtype=bool),
        "confidence": pd.to_numeric(df["_confidence"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float64),
        "source_is_registry": df["source_is_registry"].to_numpy(dtype=bool),
        "vessel_codes": df[vessel_col]
        .fillna("UNK")
        .astype("category")
        .cat.codes.to_numpy(dtype=np.int32),
        "vessel_name_norm": df["vessel_name_norm"].fillna("UNK").to_numpy()
        if "vessel_name_norm" in df.columns
        else np.empty(len(df), dtype=object),
        "home_port_codes": df["home_port_norm"]
        .fillna("UNK")
        .astype("category")
        .cat.codes.to_numpy(dtype=np.int32)
        if "home_port_norm" in df.columns
        else np.zeros(len(df), dtype=np.int32),
        "agent_norm": df["agent_norm"].to_numpy()
        if "agent_norm" in df.columns
        else np.empty(len(df), dtype=object),
        "primary_class_codes": df["primary_class"]
        .fillna("missing")
        .astype("category")
        .cat.codes.to_numpy(dtype=np.int16),
        "dest_basin": df["destination_basin"].fillna("Unknown").to_numpy(),
        "event_row_id": df["event_row_id"].to_numpy()
        if "event_row_id" in df.columns
        else np.arange(len(df)),
    }


def _window_np(
    ga: dict[str, Any] | None, t0_days: int | None, tau_days: int
) -> tuple[slice, np.ndarray]:
    """Return ``(slice, weights)`` for a time-decay window using searchsorted."""
    if ga is None or t0_days is None:
        return slice(0, 0), np.array([], dtype=np.float64)
    issue_days = ga["issue_days"]
    left = int(np.searchsorted(issue_days, t0_days - tau_days, side="left"))
    right = int(np.searchsorted(issue_days, t0_days, side="right"))
    if left >= right:
        return slice(0, 0), np.array([], dtype=np.float64)
    sl = slice(left, right)
    delta = (t0_days - issue_days[sl]).astype(np.float64)
    weights = ga["row_weight"][sl] * np.exp(-delta / float(tau_days))
    return sl, weights


def _entropy_np(codes: np.ndarray, weights: np.ndarray) -> float:
    """Weighted entropy from integer category codes."""
    if len(codes) == 0:
        return math.nan
    n_bins = int(codes.max()) + 1
    hist = np.bincount(codes.astype(np.intp), weights=weights, minlength=n_bins)
    total = hist.sum()
    if total < 1e-12:
        return math.nan
    probs = hist / total
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum())


def _stats_np(
    ga: dict[str, Any] | None, sl: slice, weights: np.ndarray
) -> dict[str, float]:
    """Compute all 8 summary statistics from numpy arrays."""
    if ga is None or sl.start >= sl.stop:
        return _EMPTY_STATS.copy()
    return {
        "reports": float(weights.sum()),
        "distress_mass": float(weights[ga["distress"][sl]].sum()),
        "positive_mass": float(weights[ga["positive"][sl]].sum()),
        "weak_mass": float(weights[ga["weak"][sl]].sum()),
        "unique_vessels": float(np.unique(ga["vessel_codes"][sl]).size),
        "signal_entropy": _entropy_np(ga["primary_class_codes"][sl], weights),
        "port_diversity": float(np.unique(ga["home_port_codes"][sl]).size),
        "mean_confidence": float(ga["confidence"][sl].mean()),
    }


def _masked_stats_np(
    ga: dict[str, Any], sl: slice, base_weights: np.ndarray, mask: np.ndarray
) -> dict[str, float]:
    """Stats for a boolean-masked subset within a window slice."""
    if not mask.any():
        return _EMPTY_STATS.copy()
    w = base_weights[mask]
    return {
        "reports": float(w.sum()),
        "distress_mass": float(w[ga["distress"][sl][mask]].sum()),
        "positive_mass": float(w[ga["positive"][sl][mask]].sum()),
        "weak_mass": float(w[ga["weak"][sl][mask]].sum()),
        "unique_vessels": float(np.unique(ga["vessel_codes"][sl][mask]).size),
        "signal_entropy": _entropy_np(ga["primary_class_codes"][sl][mask], w),
        "port_diversity": float(np.unique(ga["home_port_codes"][sl][mask]).size),
        "mean_confidence": float(ga["confidence"][sl][mask].mean()),
    }


# ---------------------------------------------------------------------------
# Multiprocessing support
# ---------------------------------------------------------------------------

# Module-level shared data — fork-inherited by worker processes (copy-on-write).
_INFO_WORKER_DATA: dict[str, Any] = {}


def _compute_single_departure_features(dep_tuple: tuple) -> dict[str, Any]:
    """Compute all information stock features for one departure (pure numpy)."""
    (departure_id, voyage_id, t0, basin, home_port, agent, agent_id, _) = dep_tuple

    d = _INFO_WORKER_DATA
    t0_days = int((t0 - _EPOCH).days) if pd.notna(t0) else None

    ga_pub = d["np_basin"].get(basin)
    ga_peer = d["np_hp_basin"].get((home_port, basin))
    ga_ab = d["np_agent_basin"].get((agent, basin))
    ga_agent = d["np_agent"].get(agent)
    ga_reg = d["np_agent_registry"].get(agent)
    voyage_by_agent = d["voyage_by_agent"]
    voyage_metrics = d["voyage_metrics"]

    # ---- window slicing (searchsorted, no DataFrame) ----
    pub30_sl, pub30_w = _window_np(ga_pub, t0_days, 30)
    pub90_sl, pub90_w = _window_np(ga_pub, t0_days, 90)
    peer90_sl, peer90_w = _window_np(ga_peer, t0_days, 90)
    peer180_sl, peer180_w = _window_np(ga_peer, t0_days, 180)
    ab30_sl, ab30_w = _window_np(ga_ab, t0_days, 30)
    ab90_sl, ab90_w = _window_np(ga_ab, t0_days, 90)
    agent180_sl, agent180_w = _window_np(ga_agent, t0_days, 180)
    reg365_sl, reg365_w = _window_np(ga_reg, t0_days, 365)

    # ---- stats (pure numpy) ----
    pub30_s = _stats_np(ga_pub, pub30_sl, pub30_w)
    pub90_s = _stats_np(ga_pub, pub90_sl, pub90_w)
    peer90_s = _stats_np(ga_peer, peer90_sl, peer90_w)
    peer180_s = _stats_np(ga_peer, peer180_sl, peer180_w)
    ab30_s = _stats_np(ga_ab, ab30_sl, ab30_w)
    ab90_s = _stats_np(ga_ab, ab90_sl, ab90_w)
    agent180_s = _stats_np(ga_agent, agent180_sl, agent180_w)
    reg365_s = _stats_np(ga_reg, reg365_sl, reg365_w)

    # ---- peer exclusion (numpy boolean mask) ----
    if ga_peer is not None and peer90_sl.start < peer90_sl.stop:
        excl = ga_peer["agent_norm"][peer90_sl] != agent
        peer_ex_s = _masked_stats_np(ga_peer, peer90_sl, peer90_w, excl)
    else:
        peer_ex_s = _EMPTY_STATS.copy()

    # ---- basin last-signal days (searchsorted) ----
    if ga_pub is not None and t0_days is not None:
        right = int(np.searchsorted(ga_pub["issue_days"], t0_days, side="right"))
        basin_last_signal_days = (
            float(t0_days - ga_pub["issue_days"][right - 1]) if right > 0 else np.nan
        )
    else:
        basin_last_signal_days = np.nan

    # ---- agent portfolio breadth (numpy) ----
    basin_share = np.nan
    breadth_entropy = np.nan
    if ga_agent is not None and agent180_sl.start < agent180_sl.stop:
        basins = ga_agent["dest_basin"][agent180_sl]
        unique_basins, basin_counts = np.unique(basins, return_counts=True)
        probs = basin_counts / basin_counts.sum()
        breadth_entropy = float(-(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum())
        basin_share = float((basins == basin).mean())

    # ---- registry stats ----
    registry_entropy = np.nan
    registry_basin_share = np.nan
    if ga_reg is not None and reg365_sl.start < reg365_sl.stop:
        basins_r = ga_reg["dest_basin"][reg365_sl]
        unique_r, counts_r = np.unique(basins_r, return_counts=True)
        probs_r = counts_r / counts_r.sum()
        registry_entropy = float(-(probs_r * np.log(np.clip(probs_r, 1e-12, 1.0))).sum())
        registry_basin_share = float((basins_r == basin).mean())

    # ---- agent basin active vessels (from agent180 window, filtered to basin) ----
    if ga_agent is not None and agent180_sl.start < agent180_sl.stop:
        basin_mask = ga_agent["dest_basin"][agent180_sl] == basin
        agent_basin_vessels = float(np.unique(ga_agent["vessel_codes"][agent180_sl][basin_mask]).size) if basin_mask.any() else 0.0
    else:
        agent_basin_vessels = 0.0

    # ---- registry basin active vessels ----
    if ga_reg is not None and reg365_sl.start < reg365_sl.stop:
        reg_basin_mask = ga_reg["dest_basin"][reg365_sl] == basin
        reg_basin_vessels = float(np.unique(ga_reg["vessel_name_norm"][reg365_sl][reg_basin_mask]).size) if reg_basin_mask.any() else 0.0
    else:
        reg_basin_vessels = 0.0

    # ---- prior agent voyage metrics (small DataFrame, kept as-is) ----
    agent_history_key = agent_id
    if agent_history_key is None or (isinstance(agent_history_key, float) and np.isnan(agent_history_key)):
        agent_history_key = agent
    prior_agent_voyages = voyage_by_agent.get(agent_history_key, voyage_metrics.iloc[0:0])
    if not prior_agent_voyages.empty:
        prior_agent_voyages = prior_agent_voyages[prior_agent_voyages["first_issue_date"] <= t0]
        within_180 = prior_agent_voyages[
            (t0 - prior_agent_voyages["first_issue_date"]).dt.days.between(0, 180)
        ]
        prior_bad = within_180[within_180["ever_bad_state"]]
        bad_rate = float(within_180["ever_bad_state"].mean()) if not within_180.empty else np.nan
        recovery_rate = float(prior_bad["ever_recovered_after_bad_state"].mean()) if not prior_bad.empty else np.nan
    else:
        bad_rate = np.nan
        recovery_rate = np.nan

    # ---- shocks ----
    basin_opportunity_shock = pub30_s["positive_mass"] - max(pub90_s["positive_mass"] - pub30_s["positive_mass"], 0.0)
    basin_risk_shock = pub30_s["distress_mass"] - max(pub90_s["distress_mass"] - pub30_s["distress_mass"], 0.0)

    # ---- contributing signals (numpy dedup) ----
    id_parts, reg_parts, basin_parts, conf_parts = [], [], [], []
    for ga, sl in [(ga_pub, pub90_sl), (ga_peer, peer90_sl), (ga_ab, ab90_sl)]:
        if ga is not None and sl.start < sl.stop:
            id_parts.append(ga["event_row_id"][sl])
            reg_parts.append(ga["source_is_registry"][sl])
            basin_parts.append(ga["dest_basin"][sl])
            conf_parts.append(ga["confidence"][sl])
    if id_parts:
        all_ids = np.concatenate(id_parts)
        _, unique_idx = np.unique(all_ids, return_index=True)
        n_contributing = len(unique_idx)
        all_reg = np.concatenate(reg_parts)
        all_basins = np.concatenate(basin_parts)
        all_conf = np.concatenate(conf_parts)
        mean_confidence = float(all_conf[unique_idx].mean())
        share_registry = float(all_reg[unique_idx].mean())
        share_unresolved = float((all_basins[unique_idx] == "Unknown").mean())
    else:
        n_contributing = 0
        mean_confidence = np.nan
        share_registry = np.nan
        share_unresolved = np.nan

    return {
        "departure_id": departure_id,
        "voyage_id": voyage_id,
        "departure_issue_date": t0,
        "home_port_norm": home_port,
        "agent_norm": agent,
        "departure_destination_basin": basin,
        "pub_basin_reports_tau30": pub30_s["reports"],
        "pub_basin_reports_tau90": pub90_s["reports"],
        "pub_basin_distress_mass_tau30": pub30_s["distress_mass"],
        "pub_basin_distress_mass_tau90": pub90_s["distress_mass"],
        "pub_basin_positive_mass_tau30": pub30_s["positive_mass"],
        "pub_basin_positive_mass_tau90": pub90_s["positive_mass"],
        "pub_basin_weak_mass_tau30": pub30_s["weak_mass"],
        "pub_basin_last_signal_days": basin_last_signal_days,
        "pub_basin_unique_vessels_tau90": pub90_s["unique_vessels"],
        "pub_basin_signal_entropy_tau90": pub90_s["signal_entropy"],
        "pub_basin_port_diversity_tau90": pub90_s["port_diversity"],
        "peer_homeport_basin_reports_tau90": peer90_s["reports"],
        "peer_homeport_basin_distress_mass_tau90": peer90_s["distress_mass"],
        "peer_homeport_basin_positive_mass_tau90": peer90_s["positive_mass"],
        "peer_homeport_active_vessels_tau180": peer180_s["unique_vessels"],
        "peer_homeport_basin_reports_tau90_excluding_agent": peer_ex_s["reports"],
        "peer_homeport_basin_distress_mass_tau90_excluding_agent": peer_ex_s["distress_mass"],
        "agent_basin_reports_tau30": ab30_s["reports"],
        "agent_basin_reports_tau90": ab90_s["reports"],
        "agent_basin_positive_mass_tau90": ab90_s["positive_mass"],
        "agent_basin_distress_mass_tau90": ab90_s["distress_mass"],
        "agent_basin_active_vessels_tau180": agent_basin_vessels,
        "agent_portfolio_total_active_vessels_tau180": agent180_s["unique_vessels"],
        "agent_portfolio_basin_share_tau180": basin_share,
        "agent_portfolio_breadth_entropy_tau180": breadth_entropy,
        "agent_recent_bad_state_rate_tau180": bad_rate,
        "agent_recent_recovery_rate_tau180": recovery_rate,
        "agent_registry_basin_active_vessels": reg_basin_vessels,
        "agent_registry_total_active_vessels": reg365_s["unique_vessels"],
        "agent_registry_basin_share": registry_basin_share,
        "agent_registry_portfolio_breadth_entropy": registry_entropy,
        "basin_opportunity_shock": basin_opportunity_shock,
        "basin_risk_shock": basin_risk_shock,
        "agent_relative_information_advantage": ab90_s["reports"] - peer_ex_s["reports"],
        "agent_relative_risk_advantage": peer_ex_s["distress_mass"] - ab90_s["distress_mass"],
        "signal_count_total": float(n_contributing),
        "mean_row_confidence": mean_confidence,
        "share_registry_based": share_registry,
        "share_unresolved_basin": share_unresolved,
    }


def _process_departure_chunk(dep_tuples: list[tuple]) -> list[dict[str, Any]]:
    """Process a batch of departure tuples.  Called by worker processes."""
    return [_compute_single_departure_features(dt) for dt in dep_tuples]


def compute_information_stock_features(
    departure_df: pd.DataFrame,
    prior_events_df: pd.DataFrame,
    config: WSLReliabilityConfig,
    *,
    tracer: PerfTracer | None = None,
    n_workers: int | None = None,
) -> pd.DataFrame:
    global _INFO_WORKER_DATA

    signal_df = _prepare_signal_frame(prior_events_df)
    voyage_metrics = _prior_voyage_metrics(signal_df)

    # ---- Pre-group and convert to numpy arrays (one-time cost) ----
    logger.info("[info_stock] Pre-converting %d events to numpy arrays...", len(signal_df))

    by_basin = {key: group.sort_values("issue_date") for key, group in signal_df.groupby("destination_basin", observed=True)}
    by_homeport_basin = {
        key: group.sort_values("issue_date")
        for key, group in signal_df.groupby(["home_port_norm", "destination_basin"], observed=True)
    }
    by_agent_basin = {
        key: group.sort_values("issue_date")
        for key, group in signal_df.groupby(["agent_norm", "destination_basin"], observed=True)
    }
    by_agent = {key: group.sort_values("issue_date") for key, group in signal_df.groupby("agent_norm", observed=True)}

    np_basin = {k: _group_to_arrays(v) for k, v in by_basin.items()}
    np_hp_basin = {k: _group_to_arrays(v) for k, v in by_homeport_basin.items()}
    np_agent_basin = {k: _group_to_arrays(v) for k, v in by_agent_basin.items()}
    np_agent = {k: _group_to_arrays(v) for k, v in by_agent.items()}
    np_agent_registry = {}
    for agent_key, agent_df in by_agent.items():
        reg_df = agent_df[agent_df["source_is_registry"]].copy()
        if not reg_df.empty:
            np_agent_registry[agent_key] = _group_to_arrays(reg_df)

    if not voyage_metrics.empty and "agent_id" in voyage_metrics.columns:
        voyage_by_agent = {
            key: group.sort_values("first_issue_date")
            for key, group in voyage_metrics.groupby("agent_id", observed=True)
        }
    else:
        voyage_by_agent = {}

    logger.info("[info_stock] Numpy conversion done (%d groups total)",
                len(np_basin) + len(np_hp_basin) + len(np_agent_basin) + len(np_agent))

    # Store at module level for fork-inherited workers.
    _INFO_WORKER_DATA = {
        "np_basin": np_basin,
        "np_hp_basin": np_hp_basin,
        "np_agent_basin": np_agent_basin,
        "np_agent": np_agent,
        "np_agent_registry": np_agent_registry,
        "voyage_by_agent": voyage_by_agent,
        "voyage_metrics": voyage_metrics,
    }

    # Prepare lightweight departure tuples.
    dep_tuples: list[tuple] = []
    for row in departure_df.itertuples(index=False):
        dep_tuples.append((
            row.departure_id,
            row.voyage_id,
            row.departure_issue_date,
            row.departure_destination_basin,
            row.home_port_norm,
            row.agent_norm,
            getattr(row, "agent_id", None),
            getattr(row, "departure_decade", None),
        ))

    n_departures = len(dep_tuples)
    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, 10)
    use_parallel = n_workers >= 2 and n_departures >= 200

    if use_parallel:
        import multiprocessing as mp

        logger.info(
            "[info_stock] Parallel mode: %d departures across %d workers",
            n_departures, n_workers,
        )
        n_chunks = n_workers * 4
        chunk_size = max(1, n_departures // n_chunks)
        chunks = [dep_tuples[i : i + chunk_size] for i in range(0, n_departures, chunk_size)]

        try:
            ctx = mp.get_context("fork")
            with ctx.Pool(n_workers) as pool:
                batch_results = pool.map(_process_departure_chunk, chunks)
            rows: list[dict[str, Any]] = [row for batch in batch_results for row in batch]
        except Exception:
            logger.warning(
                "[info_stock] Parallel processing failed, falling back to sequential",
                exc_info=True,
            )
            use_parallel = False

    if not use_parallel:
        logger.info("[info_stock] Sequential mode: %d departures", n_departures)
        rows = []
        for dep_idx, dt in enumerate(dep_tuples):
            if tracer is not None:
                tracer.tick("info_stock_features", dep_idx, n_departures, every=200)
            rows.append(_compute_single_departure_features(dt))

    feature_df = pd.DataFrame(rows)
    feature_df["feature_missingness_count"] = feature_df.isna().sum(axis=1)
    if tracer is not None:
        tracer.set_metadata(n_departures=n_departures, n_prior_events=len(prior_events_df), parallel=use_parallel, n_workers=n_workers if use_parallel else 1)
    return feature_df


def _zscore_within_decade(frame: pd.DataFrame, columns: list[str], decade_col: str) -> pd.DataFrame:
    scored = frame.copy()
    for column in columns:
        if column not in scored.columns:
            continue
        scored[f"z__{column}"] = scored.groupby(decade_col)[column].transform(
            lambda series: (
                (series - series.mean()) / series.std(ddof=0)
                if pd.notna(series.std(ddof=0)) and series.std(ddof=0) > 0
                else 0.0
            )
        )
        scored[f"z__{column}"] = scored[f"z__{column}"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return scored


def build_information_indexes(feature_df: pd.DataFrame, config: WSLReliabilityConfig) -> pd.DataFrame:
    work = feature_df.copy()
    work["departure_decade"] = (pd.to_datetime(work["departure_issue_date"]).dt.year // 10) * 10
    groups = {
        "public_information_index": [
            "pub_basin_reports_tau30",
            "pub_basin_reports_tau90",
            "pub_basin_positive_mass_tau30",
            "pub_basin_positive_mass_tau90",
            "pub_basin_unique_vessels_tau90",
        ],
        "portfolio_information_index": [
            "agent_basin_reports_tau90",
            "agent_portfolio_total_active_vessels_tau180",
            "agent_portfolio_basin_share_tau180",
            "agent_registry_total_active_vessels",
            "agent_registry_basin_share",
        ],
        "risk_index": [
            "pub_basin_distress_mass_tau30",
            "pub_basin_distress_mass_tau90",
            "agent_basin_distress_mass_tau90",
            "agent_recent_bad_state_rate_tau180",
            "basin_risk_shock",
        ],
        "opportunity_index": [
            "pub_basin_positive_mass_tau30",
            "pub_basin_positive_mass_tau90",
            "agent_basin_positive_mass_tau90",
            "basin_opportunity_shock",
        ],
        "information_advantage_index": [
            "agent_relative_information_advantage",
            "agent_relative_risk_advantage",
        ],
    }
    all_columns = sorted({column for cols in groups.values() for column in cols if column in work.columns})
    work = _zscore_within_decade(work, all_columns, "departure_decade")
    for index_name, columns in groups.items():
        z_columns = [f"z__{column}" for column in columns if f"z__{column}" in work.columns]
        work[index_name] = work[z_columns].mean(axis=1) if z_columns else np.nan
    return work


def export_information_stock_outputs(
    output_dir: Path,
    departure_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    index_df: pd.DataFrame,
    metadata: dict[str, Any],
    audit_df: pd.DataFrame,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "departure_panel": output_dir / "departure_panel.parquet",
        "raw_features": output_dir / "departure_information_stock_raw.parquet",
        "indexes": output_dir / "departure_information_indexes.parquet",
        "metadata": output_dir / "information_stock_metadata.json",
        "audit": output_dir / "information_stock_audit_sample.csv",
    }
    save_dataframe(departure_df, paths["departure_panel"])
    save_dataframe(feature_df, paths["raw_features"])
    save_dataframe(index_df, paths["indexes"])
    audit_df.to_csv(paths["audit"], index=False)
    write_json(paths["metadata"], metadata)
    return paths
