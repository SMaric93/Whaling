from __future__ import annotations

import numpy as np
import pandas as pd

from .._table_common import save_table_outputs
from ..config import BuildContext
from ..data import load_action_dataset, load_connected_sample
from ..utils.footnotes import standard_footnote
from ..utils.inference import clustered_ols
from ..utils.risk import expected_shortfall_proxy, lower_tail_reference

BARREN_THRESHOLD = 7


def _numeric(series: pd.Series | None, index: pd.Index | None = None) -> pd.Series:
    if series is None:
        return pd.Series(np.nan, index=index, dtype=float)
    return pd.to_numeric(series, errors="coerce")


def _prepare_sample(connected: pd.DataFrame, action: pd.DataFrame) -> pd.DataFrame:
    action = action.copy()
    action["active_search_flag"] = _numeric(action.get("active_search_flag"), action.index).fillna(0)
    action["consecutive_empty_days"] = _numeric(action.get("consecutive_empty_days"), action.index)
    action["exit_patch_next"] = _numeric(action.get("exit_patch_next"), action.index)
    action["encounter_any"] = action.get("encounter", pd.Series("NoEnc", index=action.index)).fillna("NoEnc").astype(str).ne("NoEnc").astype(float)
    action["barren_search"] = (action["active_search_flag"].gt(0) & action["consecutive_empty_days"].ge(BARREN_THRESHOLD)).astype(float)
    action["exploitation_state"] = action["encounter_any"]
    action["negative_signal_exit"] = np.where(action["barren_search"] > 0, action["exit_patch_next"], np.nan)

    action_agg = (
        action.groupby("voyage_id")
        .agg(
            barren_search_share=("barren_search", "mean"),
            exploitation_share=("exploitation_state", "mean"),
            exit_after_negative_signal=("negative_signal_exit", "mean"),
        )
        .reset_index()
        if not action.empty
        else pd.DataFrame(columns=["voyage_id", "barren_search_share", "exploitation_share", "exit_after_negative_signal"])
    )

    df = connected.merge(action_agg, on="voyage_id", how="left")
    for col in [
        "q_total_index",
        "theta",
        "psi",
        "scarcity",
        "captain_experience",
        "duration_days",
        "n_grounds_visited",
        "ground_switching_count",
    ]:
        if col in df.columns:
            df[col] = _numeric(df[col], df.index)

    p10 = lower_tail_reference(df["q_total_index"], quantile=0.10)
    p05 = float(df["q_total_index"].quantile(0.05))
    df["bottom_decile"] = (df["q_total_index"] <= p10).astype(float)
    df["catastrophic_voyage"] = ((df["q_total_index"] <= p05) & (df["duration_days"] >= df["duration_days"].median())).astype(float)
    df["expected_shortfall_proxy"] = expected_shortfall_proxy(df["q_total_index"], quantile=0.10)
    return df


def _total_effect(df: pd.DataFrame, outcome: str) -> dict[str, float]:
    model = clustered_ols(
        df,
        outcome=outcome,
        regressors=["psi", "theta", "captain_experience"],
        cluster_col="captain_id",
    )
    return {
        "coef": model["coef"].get("psi", np.nan),
        "se": model["se"].get("psi", np.nan),
        "p": model["p"].get("psi", np.nan),
        "n_obs": int(model["n_obs"]),
    }


def _mediation_share(df: pd.DataFrame, outcome: str, mediator: str) -> dict[str, float]:
    total = _total_effect(df, outcome)
    mediator_model = clustered_ols(
        df,
        outcome=mediator,
        regressors=["psi", "theta", "captain_experience"],
        cluster_col="captain_id",
    )
    outcome_model = clustered_ols(
        df,
        outcome=outcome,
        regressors=["psi", mediator, "theta", "captain_experience"],
        cluster_col="captain_id",
    )
    a_path = mediator_model["coef"].get("psi", np.nan)
    b_path = outcome_model["coef"].get(mediator, np.nan)
    indirect = a_path * b_path if np.isfinite(a_path) and np.isfinite(b_path) else np.nan
    direct = outcome_model["coef"].get("psi", np.nan)
    denom = total["coef"]
    share = indirect / denom if np.isfinite(indirect) and np.isfinite(denom) and abs(denom) > 1e-12 else np.nan
    direct_share = direct / denom if np.isfinite(direct) and np.isfinite(denom) and abs(denom) > 1e-12 else np.nan
    return {
        "mediated_share": share,
        "remaining_direct_share": direct_share,
        "n_obs": int(outcome_model["n_obs"]),
    }


def _panel_a(df: pd.DataFrame) -> list[dict]:
    outcomes = [
        ("psi -> bottom decile", "bottom_decile"),
        ("psi -> catastrophic voyage", "catastrophic_voyage"),
        ("psi -> expected shortfall", "expected_shortfall_proxy"),
    ]
    rows = []
    for row_label, outcome in outcomes:
        effect = _total_effect(df, outcome)
        rows.append(
            {
                "panel": "Panel A",
                "row_label": row_label,
                "total_effect": effect["coef"],
                "std_error": effect["se"],
                "p_value": effect["p"],
                "n_obs": effect["n_obs"],
                "note": "Total-effect row from a captain-clustered descriptive regression on psi.",
            }
        )
    return rows


def _panel_b(df: pd.DataFrame) -> list[dict]:
    mediator_map = [
        ("voyage duration", "duration_days"),
        ("time in barren-search state", "barren_search_share"),
        ("exit hazard after negative signal", "exit_after_negative_signal"),
        ("time in exploitation state", "exploitation_share"),
        ("number of grounds visited", "n_grounds_visited"),
        ("destination diversification", "ground_switching_count"),
    ]
    rows = []
    for row_label, mediator in mediator_map:
        share = _mediation_share(df, "bottom_decile", mediator)
        rows.append(
            {
                "panel": "Panel B",
                "row_label": row_label,
                "mediated_share": share["mediated_share"],
                "remaining_direct_share": share["remaining_direct_share"],
                "n_obs": share["n_obs"],
                "note": "Descriptive mediation share for the bottom-decile outcome, not a causal mediation estimate.",
            }
        )
    return rows


def _panel_c(df: pd.DataFrame) -> list[dict]:
    theta_median = float(df["theta"].median())
    groups = [
        ("novice", df["novice"].fillna(0).astype(float) > 0),
        ("expert", df["novice"].fillna(0).astype(float) == 0),
        ("low theta", df["theta"] <= theta_median),
        ("high theta", df["theta"] > theta_median),
    ]
    mediators = [
        ("mediated_share_via_duration", "duration_days"),
        ("mediated_share_via_state_occupancy", "barren_search_share"),
        ("mediated_share_via_diversification", "ground_switching_count"),
    ]
    rows = []
    for row_label, mask in groups:
        sample = df.loc[mask].copy()
        row = {
            "panel": "Panel C",
            "row_label": row_label,
            "n_obs": int(len(sample)),
            "note": "Group-specific descriptive mediation shares for the bottom-decile outcome.",
        }
        for col_label, mediator in mediators:
            row[col_label] = _mediation_share(sample, "bottom_decile", mediator)["mediated_share"]
        rows.append(row)
    return rows


def build(context: BuildContext):
    connected = load_connected_sample(context)
    action = load_action_dataset(context)
    df = _prepare_sample(connected, action)

    frame = pd.DataFrame(_panel_a(df) + _panel_b(df) + _panel_c(df))

    memo = standard_footnote(
        sample="Connected-set voyages merged to voyage-level mediator summaries constructed from the action panel.",
        unit="Voyage throughout.",
        types_note="theta_hat and psi_hat are connected-set AKM types; all mediation rows are descriptive decompositions rather than causal mediation estimates.",
        fe="No fixed effects in the main mediation table.",
        cluster="Captain clustering throughout.",
        controls="theta_hat and captain experience in total-effect, mediator, and conditional-outcome regressions; scarcity is left out of the baseline decomposition so the environment-risk channel is not partialled away.",
        interpretation="The downside-risk effect of organizational capability operates primarily through duration control, with smaller descriptive contributions from barren-search time and destination diversification.",
        caution="Shares can exceed one in magnitude when the estimated direct and indirect components have opposite signs; these are descriptive accounting objects, not causal path effects.",
    )
    memo = (
        "# table09_mediation\n\n"
        + memo
        + "\n\nImplementation notes:\n"
        + "- `time in barren-search state`, `time in exploitation state`, and `exit hazard after negative signal` are aggregated from the daily action panel to the voyage level.\n"
        + "- Panel B uses the bottom-decile outcome as the focal downside-risk endpoint for mediation shares.\n"
        + "- Panel C compresses the decomposition to the three mediators most central to the governance mechanism: duration, state occupancy, and diversification.\n"
    )

    return save_table_outputs(
        name="table09_mediation",
        frame=frame,
        out_dir=context.outputs / "tables",
        context=context,
        memo=memo,
        title="Table 9. How Organizations Raise the Floor",
    )
