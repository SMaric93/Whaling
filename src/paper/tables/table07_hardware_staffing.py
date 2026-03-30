from __future__ import annotations

import numpy as np
import pandas as pd

from .._table_common import save_table_outputs
from ..config import BuildContext
from ..data import load_action_dataset, load_connected_sample, load_survival_dataset
from ..utils.footnotes import standard_footnote
from ..utils.inference import clustered_ols

NEG_SIGNAL_THRESHOLD = 7


def _numeric(series: pd.Series | None, index: pd.Index | None = None) -> pd.Series:
    if series is None:
        return pd.Series(np.nan, index=index, dtype=float)
    return pd.to_numeric(series, errors="coerce")


def _spec_controls() -> list[tuple[str, list[str]]]:
    return [
        ("baseline", []),
        ("+ vessel controls", ["tonnage"]),
        ("+ crew/officer controls", ["tonnage", "crew_count", "mean_age", "desertion_rate", "unique_ranks"]),
        (
            "+ disruption controls",
            ["tonnage", "crew_count", "mean_age", "desertion_rate", "unique_ranks", "route_efficiency", "total_distance_nm", "frac_days_in_arctic_polygon", "arctic_route"],
        ),
        (
            "+ incentive/lay controls",
            ["tonnage", "crew_count", "mean_age", "desertion_rate", "unique_ranks", "route_efficiency", "total_distance_nm", "frac_days_in_arctic_polygon", "arctic_route", "vqi_proxy", "has_vqi_data"],
        ),
        (
            "full model",
            ["tonnage", "crew_count", "mean_age", "desertion_rate", "unique_ranks", "route_efficiency", "total_distance_nm", "frac_days_in_arctic_polygon", "arctic_route", "vqi_proxy", "has_vqi_data", "ground_switching_count", "n_grounds_visited"],
        ),
    ]


def _prepare_connected(connected: pd.DataFrame) -> pd.DataFrame:
    df = connected.copy()
    numeric_cols = [
        "theta",
        "psi",
        "q_total_index",
        "tonnage",
        "crew_count",
        "mean_age",
        "desertion_rate",
        "unique_ranks",
        "route_efficiency",
        "total_distance_nm",
        "frac_days_in_arctic_polygon",
        "arctic_route",
        "vqi_proxy",
        "has_vqi_data",
        "ground_switching_count",
        "n_grounds_visited",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = _numeric(df[col], df.index)
    df["theta_x_psi"] = df["theta"] * df["psi"]
    return df


def _panel_a(connected: pd.DataFrame) -> list[dict]:
    df = _prepare_connected(connected)
    rows = []
    base_regressors = ["psi", "theta", "theta_x_psi"]
    for spec_label, controls in _spec_controls():
        available = [c for c in controls if c in df.columns]
        model = clustered_ols(
            df,
            outcome="q_total_index",
            regressors=base_regressors + available,
            cluster_col="captain_id",
        )
        rows.extend(
            [
                {
                    "panel": "Panel A",
                    "specification": spec_label,
                    "row_label": "psi_hat",
                    "estimate": model["coef"].get("psi", np.nan),
                    "note": "Sequential voyage-output horse race with captain-clustered inference.",
                },
                {
                    "panel": "Panel A",
                    "specification": spec_label,
                    "row_label": "theta_hat",
                    "estimate": model["coef"].get("theta", np.nan),
                    "note": "Sequential voyage-output horse race with captain-clustered inference.",
                },
                {
                    "panel": "Panel A",
                    "specification": spec_label,
                    "row_label": "psi_hat × theta_hat",
                    "estimate": model["coef"].get("theta_x_psi", np.nan),
                    "note": "Sequential voyage-output horse race with captain-clustered inference.",
                },
                {
                    "panel": "Panel A",
                    "specification": spec_label,
                    "row_label": "R²",
                    "estimate": model["r_squared"],
                    "note": "Sequential voyage-output horse race with captain-clustered inference.",
                },
                {
                    "panel": "Panel A",
                    "specification": spec_label,
                    "row_label": "N",
                    "estimate": float(model["n_obs"]),
                    "note": "Sequential voyage-output horse race with captain-clustered inference.",
                },
            ]
        )
    return rows


def _panel_b(survival: pd.DataFrame, connected: pd.DataFrame) -> list[dict]:
    info_cols = [
        "voyage_id",
        "captain_id",
        "theta",
        "psi",
        "scarcity",
        "captain_experience",
        "tonnage",
        "crew_count",
        "mean_age",
        "desertion_rate",
        "unique_ranks",
        "route_efficiency",
        "total_distance_nm",
        "frac_days_in_arctic_polygon",
        "arctic_route",
        "vqi_proxy",
        "has_vqi_data",
        "ground_switching_count",
        "n_grounds_visited",
    ]
    info = connected[[c for c in info_cols if c in connected.columns]].drop_duplicates("voyage_id")
    df = survival.merge(info, on="voyage_id", how="left", suffixes=("", "_connected"))
    if "captain_id_connected" in df.columns:
        df["captain_id"] = df["captain_id"].where(df["captain_id"].notna(), df["captain_id_connected"])
    for col in [c for c in info_cols if c != "voyage_id"]:
        if col in df.columns:
            df[col] = _numeric(df[col], df.index)
    df["neg_signal"] = _numeric(df.get("consecutive_empty_days"), df.index).ge(NEG_SIGNAL_THRESHOLD).astype(float)
    df["psi_x_neg_signal"] = df["psi"] * df["neg_signal"]
    df["day_in_patch"] = _numeric(df.get("day_in_patch"), df.index)

    base_regressors = ["psi", "neg_signal", "psi_x_neg_signal", "theta", "scarcity", "captain_experience", "day_in_patch"]
    rows = []
    for spec_label, controls in _spec_controls():
        available = [c for c in controls if c in df.columns]
        model = clustered_ols(
            df,
            outcome="exit_tomorrow",
            regressors=base_regressors + available,
            cluster_col="captain_id",
        )
        rows.extend(
            [
                {
                    "panel": "Panel B",
                    "specification": spec_label,
                    "row_label": "psi_hat × negative signal",
                    "estimate": model["coef"].get("psi_x_neg_signal", np.nan),
                    "note": "Stopping-rule interaction with sequential hardware/staffing controls.",
                },
                {
                    "panel": "Panel B",
                    "specification": spec_label,
                    "row_label": "SE",
                    "estimate": model["se"].get("psi_x_neg_signal", np.nan),
                    "note": "Stopping-rule interaction with sequential hardware/staffing controls.",
                },
                {
                    "panel": "Panel B",
                    "specification": spec_label,
                    "row_label": "p-value",
                    "estimate": model["p"].get("psi_x_neg_signal", np.nan),
                    "note": "Stopping-rule interaction with sequential hardware/staffing controls.",
                },
            ]
        )
    return rows


def _panel_c(action: pd.DataFrame, connected: pd.DataFrame) -> list[dict]:
    info = connected[["voyage_id", "captain_id", "theta", "psi", "scarcity", "captain_experience"]].drop_duplicates("voyage_id")
    df = action.merge(info, on="voyage_id", how="left", suffixes=("", "_connected"))
    if "captain_id_connected" in df.columns:
        df["captain_id"] = df["captain_id"].where(df["captain_id"].notna(), df["captain_id_connected"])
    for col in ["theta", "psi", "scarcity", "captain_experience", "speed", "move_length", "turn_angle"]:
        if col in df.columns:
            df[col] = _numeric(df[col], df.index)
    transit = _numeric(df.get("transit_flag"), df.index).fillna(0) > 0
    homebound = _numeric(df.get("homebound_flag"), df.index).fillna(0) > 0
    straight_open_ocean = transit & df["move_length"].gt(df["move_length"].median()) & df["turn_angle"].abs().lt(20)

    rows = []
    for row_label, mask in [
        ("transit speed", transit),
        ("homebound speed", homebound),
        ("straight open-ocean leg speed", straight_open_ocean),
    ]:
        sample = df.loc[mask].copy()
        model = clustered_ols(
            sample,
            outcome="speed",
            regressors=["psi", "theta", "scarcity", "captain_experience"],
            cluster_col="captain_id",
        )
        rows.append(
            {
                "panel": "Panel C",
                "row_label": row_label,
                "psi_hat_coefficient": model["coef"].get("psi", np.nan),
                "std_error": model["se"].get("psi", np.nan),
                "p_value": model["p"].get("psi", np.nan),
                "n_obs": int(model["n_obs"]),
                "note": "Negative-control speed regression in routine or transit states where governance effects should attenuate.",
            }
        )
    return rows


def build(context: BuildContext):
    connected = load_connected_sample(context)
    survival = load_survival_dataset(context)
    action = load_action_dataset(context)

    frame = pd.DataFrame(_panel_a(connected) + _panel_b(survival, connected) + _panel_c(action, connected))

    memo = standard_footnote(
        sample="Panel A uses connected-set voyages; Panel B uses the patch-day survival sample merged to connected-set voyage controls; Panel C uses action-level transit and routine observations.",
        unit="Voyage in Panel A, patch-day in Panel B, and action-day in Panel C.",
        types_note="theta_hat and psi_hat are connected-set AKM estimates; all rows use the paper-facing notation in labels and memos.",
        fe="No fixed effects in the sequential horse-race rows; Appendix same-vessel FE remains a separate robustness layer.",
        cluster="Captain clustering throughout.",
        controls="Sequentially added vessel, crew/officer, disruption, and incentive/archive proxies from the shipped voyage panel.",
        interpretation="The governance effect survives substantial attenuation tests: adding vessel, crew, route, and archival incentive proxies reduces but does not eliminate the organizational signal.",
        caution="The repository does not ship a direct lay-share variable, so the incentive block uses `vqi_proxy` and related archival coverage flags as the closest available incentive proxy.",
    )
    memo = (
        "# table07_hardware_staffing\n\n"
        + memo
        + "\n\nImplementation notes:\n"
        + "- Panel A is rebuilt directly from the connected outcome sample rather than the thin `hardware_staffing_placebos.csv` export.\n"
        + "- Panel B carries the same control blocks into the stopping-rule interaction on the patch-day survival sample.\n"
        + "- Panel C uses transit, homebound, and straight open-ocean speed as negative controls.\n"
    )

    return save_table_outputs(
        name="table07_hardware_staffing",
        frame=frame,
        out_dir=context.outputs / "tables",
        context=context,
        memo=memo,
        title="Table 7. Governance Versus Hardware, Staffing, and Incentives",
    )
