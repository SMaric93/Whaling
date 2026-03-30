from __future__ import annotations

import math
import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np
import pandas as pd

from src.ml.state_models import fit_gmm_states, label_states

from .._table_common import save_table_outputs
from ..config import BuildContext
from ..data import load_action_dataset, load_connected_sample, load_state_dataset
from ..utils.footnotes import standard_footnote
from ..utils.inference import clustered_ols, normal_pvalue

BARREN_THRESHOLD = 7


def _numeric(series: pd.Series | None, index: pd.Index | None = None) -> pd.Series:
    if series is None:
        return pd.Series(np.nan, index=index, dtype=float)
    return pd.to_numeric(series, errors="coerce")


def _merge_voyage_info(df: pd.DataFrame, connected: pd.DataFrame) -> pd.DataFrame:
    info_cols = [
        "voyage_id",
        "captain_id",
        "agent_id",
        "theta",
        "psi",
        "scarcity",
        "captain_experience",
        "novice",
        "year_out",
        "date_out",
    ]
    info = connected[[c for c in info_cols if c in connected.columns]].drop_duplicates("voyage_id")
    merged = df.merge(info, on="voyage_id", how="left", suffixes=("", "_connected"))
    for col in ["captain_id", "agent_id", "theta", "psi", "scarcity", "captain_experience", "novice"]:
        connected_col = f"{col}_connected"
        if connected_col in merged.columns:
            merged[col] = merged[col].where(merged[col].notna(), merged[connected_col])
    return merged


def _fallback_state_labels(df: pd.DataFrame) -> pd.Series:
    transit = _numeric(df.get("transit_flag"), df.index).fillna(0) > 0
    homebound = _numeric(df.get("homebound_flag"), df.index).fillna(0) > 0
    active = _numeric(df.get("active_search_flag"), df.index).fillna(0) > 0
    time_since_success = _numeric(df.get("time_since_success"), df.index)
    empty_streak = _numeric(df.get("max_empty_streak_window"), df.index)
    patch_residence = _numeric(df.get("patch_residence"), df.index)

    labels = np.full(len(df), "inactive", dtype=object)
    labels[transit] = "transit"
    labels[homebound] = "homebound"

    barren = active & ((time_since_success >= 30) | (empty_streak >= BARREN_THRESHOLD))
    exploitation = active & ~barren & ((time_since_success <= 1) | (patch_residence >= 3))
    local_search = active & ~barren & ~exploitation

    labels[barren] = "barren_search"
    labels[exploitation] = "exploitation"
    labels[local_search] = "local_search"
    return pd.Series(labels, index=df.index, dtype="object")


def _derive_state_labels(state_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    for col in ["hmm_state_label", "state_label", "gmm_state_label"]:
        if col in state_df.columns and state_df[col].notna().any():
            labeled = state_df.copy()
            labeled["state_label"] = labeled[col].astype(str)
            return labeled, f"Using shipped `{col}` from the state dataset."

    sample = state_df.sample(n=min(len(state_df), 50_000), random_state=0).copy()
    try:
        gmm = fit_gmm_states(sample, n_components_range=[4])
        model = gmm.get("model")
        scaler = gmm.get("scaler")
        feature_names = gmm.get("feature_names", [])
        if model is None or scaler is None or not feature_names:
            raise ValueError("GMM state model did not return a fitted model, scaler, and feature names.")

        full = state_df.copy()
        X_full = full[feature_names].fillna(0).to_numpy()
        X_scaled = scaler.transform(X_full)
        labels = model.predict(X_scaled)
        labeled, _ = label_states(full, labels, feature_names)
        return labeled, "State labels rebuilt with the repository's GMM latent-state model and rule-based labeler."
    except Exception:
        labeled = state_df.copy()
        labeled["state_label"] = _fallback_state_labels(labeled)
        return labeled, "State-model fitting failed in the current environment, so the paper builder fell back to deterministic movement-based state labels."


def _prepare_state_transitions(state_df: pd.DataFrame, connected: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    labeled_state_df, label_note = _derive_state_labels(state_df)
    df = _merge_voyage_info(labeled_state_df.copy(), connected)
    sort_cols = ["voyage_id"] + [col for col in ["date", "obs_date", "voyage_day"] if col in df.columns]
    df = df.sort_values(sort_cols).copy()
    df["next_state"] = df.groupby("voyage_id")["state_label"].shift(-1)
    df["scarcity"] = _numeric(df.get("scarcity"), df.index)
    df["psi"] = _numeric(df.get("psi"), df.index)
    df["theta"] = _numeric(df.get("theta"), df.index)
    df["novice"] = _numeric(df.get("novice"), df.index)
    df["psi_x_scarcity"] = df["psi"] * df["scarcity"]
    return df.dropna(subset=["next_state", "captain_id", "psi", "theta"]).copy(), label_note


def _panel_a(state_df: pd.DataFrame, connected: pd.DataFrame) -> tuple[list[dict], pd.DataFrame, str]:
    transitions, label_note = _prepare_state_transitions(state_df, connected)
    specs = [
        ("barren_search", "exit/relocation", lambda s: s.isin(["transit", "homebound", "local_exploration", "local_search"])),
        ("barren_search", "stay", lambda s: s.eq("barren_search")),
        ("exploitation", "stay", lambda s: s.eq("exploitation")),
        ("exploitation", "exit", lambda s: s.isin(["transit", "homebound"])),
        ("transit", "local_search", lambda s: s.isin(["local_exploration", "local_search"])),
        ("transit", "exploitation", lambda s: s.eq("exploitation")),
    ]
    rows: list[dict] = []
    for current_state, target_label, rule in specs:
        sample = transitions[transitions["state_label"] == current_state].copy()
        if sample.empty:
            continue
        sample["transition_outcome"] = rule(sample["next_state"]).astype(float)
        model = clustered_ols(
            sample,
            outcome="transition_outcome",
            regressors=["psi", "psi_x_scarcity", "theta", "scarcity", "novice"],
            cluster_col="captain_id",
        )
        rows.append(
            {
                "panel": "Panel A",
                "row_label": f"{current_state} -> {target_label}",
                "coefficient_on_psi_hat": model["coef"].get("psi", np.nan),
                "coefficient_on_psi_hat_x_scarcity": model["coef"].get("psi_x_scarcity", np.nan),
                "std_error": model["se"].get("psi", np.nan),
                "p_value": model["p"].get("psi", np.nan),
                "n_obs": int(model["n_obs"]),
                "note": label_note,
            }
        )
    return rows, transitions, label_note


def _prepare_switch_voyages(connected: pd.DataFrame) -> pd.DataFrame:
    voyages = connected[
        [
            "voyage_id",
            "captain_id",
            "agent_id",
            "psi",
            "theta",
            "year_out",
            "date_out",
            "captain_experience",
            "novice",
            "scarcity",
        ]
    ].dropna(subset=["voyage_id", "captain_id", "agent_id"]).copy()
    voyages["year_out"] = _numeric(voyages["year_out"], voyages.index)
    voyages["date_out"] = pd.to_datetime(voyages["date_out"], errors="coerce")
    fallback_date = pd.to_datetime(voyages["year_out"].fillna(0).astype(int).astype(str) + "-01-01", errors="coerce")
    voyages["sort_date"] = voyages["date_out"].where(voyages["date_out"].notna(), fallback_date)
    voyages = voyages.sort_values(["captain_id", "sort_date", "voyage_id"]).copy()
    voyages["voyage_order"] = voyages.groupby("captain_id").cumcount()
    voyages["prev_agent_id"] = voyages.groupby("captain_id")["agent_id"].shift(1)
    voyages["prev_psi"] = voyages.groupby("captain_id")["psi"].shift(1)
    voyages["switched"] = voyages["prev_agent_id"].notna() & voyages["agent_id"].ne(voyages["prev_agent_id"])
    switched = voyages[voyages["switched"]].copy()
    if switched.empty:
        voyages["first_switch_order"] = np.nan
        voyages["switch_direction"] = 0.0
        voyages["post_switch"] = 0.0
        voyages["higher_psi_post"] = 0.0
        voyages["lower_psi_post"] = 0.0
        voyages["event_time"] = np.nan
        return voyages

    first_switch = (
        switched.sort_values(["captain_id", "voyage_order"])
        .groupby("captain_id")
        .first()
        .reset_index()
    )
    first_switch["first_switch_order"] = first_switch["voyage_order"]
    first_switch["switch_direction"] = np.sign(first_switch["psi"] - first_switch["prev_psi"])
    voyages = voyages.merge(
        first_switch[["captain_id", "first_switch_order", "switch_direction"]],
        on="captain_id",
        how="left",
    )
    voyages["post_switch"] = (
        voyages["first_switch_order"].notna()
        & (voyages["voyage_order"] >= voyages["first_switch_order"])
    ).astype(float)
    voyages["higher_psi_post"] = voyages["post_switch"] * (voyages["switch_direction"] > 0).astype(float)
    voyages["lower_psi_post"] = voyages["post_switch"] * (voyages["switch_direction"] < 0).astype(float)
    voyages["event_time"] = voyages["voyage_order"] - voyages["first_switch_order"]
    return voyages


def _merge_action_states(action: pd.DataFrame, labeled_states: pd.DataFrame) -> pd.DataFrame:
    state_lookup_cols = [c for c in ["voyage_id", "obs_date", "date", "state_label"] if c in labeled_states.columns]
    lookup = labeled_states[state_lookup_cols].copy()
    if "obs_date" in lookup.columns and "obs_date" in action.columns:
        return action.merge(lookup.drop_duplicates(["voyage_id", "obs_date"]), on=["voyage_id", "obs_date"], how="left")
    if "date" in lookup.columns and "date" in action.columns:
        return action.merge(lookup.drop_duplicates(["voyage_id", "date"]), on=["voyage_id", "date"], how="left")
    return action.copy()


def _panel_b(action: pd.DataFrame, connected: pd.DataFrame, labeled_states: pd.DataFrame, label_note: str) -> list[dict]:
    voyages = _prepare_switch_voyages(connected)
    switcher_voyages = voyages[voyages["first_switch_order"].notna()].copy()
    if switcher_voyages.empty:
        return []

    df = _merge_voyage_info(action.copy(), connected)
    df = _merge_action_states(df, labeled_states)
    df = df.merge(
        switcher_voyages[["voyage_id", "post_switch", "higher_psi_post", "lower_psi_post", "captain_id"]].drop_duplicates("voyage_id"),
        on=["voyage_id", "captain_id"],
        how="inner",
    )
    df["days_in_ground"] = _numeric(df.get("days_in_ground"), df.index)
    df["scarcity"] = _numeric(df.get("scarcity"), df.index)
    df["theta"] = _numeric(df.get("theta"), df.index)
    df["psi"] = _numeric(df.get("psi"), df.index)
    df["captain_experience"] = _numeric(df.get("captain_experience"), df.index)
    if "state_label" in df.columns and df["state_label"].notna().any():
        df["barren_state"] = df["state_label"].eq("barren_search").astype(float)
        df["exploitation_state"] = df["state_label"].eq("exploitation").astype(float)
    else:
        active = _numeric(df.get("active_search_flag"), df.index).fillna(0) > 0
        encounter = df.get("encounter", pd.Series("NoEnc", index=df.index)).fillna("NoEnc").astype(str)
        df["barren_state"] = (active & _numeric(df.get("consecutive_empty_days"), df.index).ge(BARREN_THRESHOLD)).astype(float)
        df["exploitation_state"] = (active & encounter.ne("NoEnc")).astype(float)
    df["post_x_barren_state"] = df["post_switch"] * df["barren_state"]
    df["post_x_exploitation_state"] = df["post_switch"] * df["exploitation_state"]

    model = clustered_ols(
        df,
        outcome="exit_patch_next",
        regressors=[
            "post_switch",
            "higher_psi_post",
            "lower_psi_post",
            "post_x_barren_state",
            "post_x_exploitation_state",
            "theta",
            "scarcity",
            "captain_experience",
            "days_in_ground",
        ],
        cluster_col="captain_id",
        fe_cols=["captain_id"],
    )
    row_map = {
        "post-switch": "post_switch",
        "switch to higher-psi": "higher_psi_post",
        "switch to lower-psi": "lower_psi_post",
        "post-switch × barren state": "post_x_barren_state",
        "post-switch × exploitation state": "post_x_exploitation_state",
    }
    rows = []
    for row_label, variable in row_map.items():
        rows.append(
            {
                "panel": "Panel B",
                "row_label": row_label,
                "coefficient": model["coef"].get(variable, np.nan),
                "std_error": model["se"].get(variable, np.nan),
                "p_value": model["p"].get(variable, np.nan),
                "n_obs": int(model["n_obs"]),
                "note": f"Captain fixed-effects estimate on action-level exit behavior among captains who ever switch agents. {label_note}",
            }
        )
    return rows


def _event_diff(sample: pd.DataFrame, event_time: int) -> tuple[float, float, float, int]:
    baseline = sample[sample["event_time"] == -1][["captain_id", "policy_rate"]].rename(columns={"policy_rate": "baseline_rate"})
    target = sample[sample["event_time"] == event_time][["captain_id", "policy_rate"]].rename(columns={"policy_rate": "target_rate"})
    paired = baseline.merge(target, on="captain_id", how="inner")
    if paired.empty:
        return np.nan, np.nan, np.nan, 0
    diff = paired["target_rate"] - paired["baseline_rate"]
    coef = float(diff.mean())
    se = float(diff.std(ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 else np.nan
    p_value = normal_pvalue(coef, se)
    return coef, se, p_value, int(len(diff))


def _panel_c(action: pd.DataFrame, connected: pd.DataFrame, labeled_states: pd.DataFrame, label_note: str) -> list[dict]:
    voyages = _prepare_switch_voyages(connected)
    switcher_voyages = voyages[voyages["event_time"].between(-2, 2, inclusive="both")].copy()
    if switcher_voyages.empty:
        return []

    df = _merge_action_states(action.copy(), labeled_states)
    if "state_label" in df.columns and df["state_label"].notna().any():
        df["barren_state"] = df["state_label"].eq("barren_search")
    else:
        active = _numeric(df.get("active_search_flag"), df.index).fillna(0) > 0
        df["barren_state"] = active & _numeric(df.get("consecutive_empty_days"), df.index).ge(BARREN_THRESHOLD)

    barren_rates = df[df["barren_state"]].groupby("voyage_id")["exit_patch_next"].mean().rename("policy_rate")
    overall_rates = df.groupby("voyage_id")["exit_patch_next"].mean().rename("overall_rate")
    voyage_rates = overall_rates.to_frame().join(barren_rates, how="left").reset_index()
    voyage_rates["policy_rate"] = voyage_rates["policy_rate"].fillna(voyage_rates["overall_rate"])

    sample = switcher_voyages.merge(voyage_rates[["voyage_id", "policy_rate"]], on="voyage_id", how="left").dropna(subset=["policy_rate"])
    rows = []
    for event_time, label in [(-2, "t-2"), (-1, "t-1"), (0, "switch"), (1, "t+1"), (2, "t+2")]:
        if event_time == -1:
            n_obs = int(sample[sample["event_time"] == -1]["captain_id"].nunique())
            coef, se, p_value = 0.0, 0.0, 1.0
        else:
            coef, se, p_value, n_obs = _event_diff(sample, event_time)
        rows.append(
            {
                "panel": "Panel C",
                "row_label": label,
                "coefficient": coef,
                "std_error": se,
                "p_value": p_value,
                "n_obs": n_obs,
                "note": f"Event-study coefficient is the within-captain change in voyage-level barren-state exit propensity relative to t-1. {label_note}",
            }
        )
    return rows


def build(context: BuildContext):
    connected = load_connected_sample(context)
    state_df = load_state_dataset(context)
    action = load_action_dataset(context)

    panel_a_rows, labeled_states, label_note = _panel_a(state_df, connected)
    frame = pd.DataFrame(
        panel_a_rows
        + _panel_b(action, connected, labeled_states, label_note)
        + _panel_c(action, connected, labeled_states, label_note)
    )

    memo = standard_footnote(
        sample="Panel A uses the state dataset merged to connected-set voyage types; Panels B and C use action-level switcher observations merged to connected-set voyage histories.",
        unit="State transition in Panel A, action-day in Panel B, and voyage-relative event time in Panel C.",
        types_note="theta_hat and psi_hat are connected-set AKM estimates merged at the voyage level; switch direction is defined using the change in voyage-level psi across agents.",
        fe="Captain fixed effects in Panel B; Panels A and C are transition and event-time summaries without additional fixed effects.",
        cluster="Captain clustering throughout.",
        controls="Scarcity, theta_hat, novice status, captain experience, and within-ground duration where available.",
        interpretation="The switching evidence in this repository is about policy changes under uncertainty: higher-psi organizations alter how captains leave barren states and how state transitions evolve after an agent change.",
        caution="If the latent-state model cannot be fit in the current environment, the paper layer falls back to deterministic movement-based states and records that fact in the table notes.",
    )
    memo = (
        "# table05_state_switching\n\n"
        + memo
        + "\n\nImplementation notes:\n"
        + f"- {label_note}\n"
        + "- Panel B uses a first-switch design: captains are tagged as post-switch from the first observed agent change onward, with higher- and lower-psi switch directions defined relative to the previous agent.\n"
        + "- Panel C is a within-captain voyage-event summary around the first switch and uses the voyage-level barren-state exit rate when available.\n"
    )

    return save_table_outputs(
        name="table05_state_switching",
        frame=frame,
        out_dir=context.outputs / "tables",
        context=context,
        memo=memo,
        title="Table 5. State Transitions and Policy Change After Switching Agents",
    )
