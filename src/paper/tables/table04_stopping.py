from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.reinforcement.utils import cluster_se

from .._table_common import save_table_outputs
from ..config import BuildContext
from ..data import (
    load_action_dataset,
    load_connected_sample,
    load_ground_quality,
    load_rational_exit_output,
    load_survival_dataset,
    load_test3_stopping_output,
)
from ..utils.footnotes import standard_footnote
from ..utils.inference import clustered_ols


NEG_SIGNAL_THRESHOLD = 3


def _numeric(series: pd.Series | None, index: pd.Index | None = None) -> pd.Series:
    if series is None:
        return pd.Series(np.nan, index=index, dtype=float)
    return pd.to_numeric(series, errors="coerce")


def _normal_pvalue(coef: float, se: float) -> float:
    if not np.isfinite(coef) or not np.isfinite(se) or se <= 0:
        return np.nan
    return float(2 * stats.norm.sf(abs(coef / se)))


def _fit_clustered_lpm(df: pd.DataFrame, outcome: str, regressors: list[str], cluster_col: str) -> dict[str, dict[str, float]]:
    clean = df.dropna(subset=[outcome, cluster_col] + regressors).copy()
    if clean.empty:
        return {"coef": {}, "se": {}, "p": {}, "n_obs": 0}

    X = clean[regressors].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(clean)), X])
    y = clean[outcome].to_numpy(dtype=float)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    residuals = y - X @ beta
    se = cluster_se(X, residuals, clean[cluster_col].to_numpy())
    names = ["const"] + regressors
    coef_map = dict(zip(names, beta))
    se_map = dict(zip(names, se))
    p_map = {name: _normal_pvalue(coef_map[name], se_map[name]) for name in names}
    return {"coef": coef_map, "se": se_map, "p": p_map, "n_obs": len(clean)}


def _panel_a_from_logit(exported: pd.DataFrame, survival: pd.DataFrame) -> list[dict]:
    if exported.empty:
        return []
    exit_rate = float(pd.to_numeric(survival["exit_tomorrow"], errors="coerce").mean()) if "exit_tomorrow" in survival.columns and not survival.empty else np.nan
    slope = exit_rate * (1 - exit_rate) if np.isfinite(exit_rate) else np.nan
    spec_map = {
        "psi_hat": ("Logit Hazard (interaction)", "psi_heldout"),
        "negative signal": ("Logit Hazard (interaction)", "neg_signal"),
        "psi_hat × negative signal": ("Logit Hazard (interaction)", "psi_heldout_x_neg_signal"),
        "positive signal": ("Logit Hazard (placebo_positive)", "pos_signal"),
        "psi_hat × positive signal": ("Logit Hazard (placebo_positive)", "psi_heldout_x_pos_signal"),
    }
    rows = []
    for row_label, (specification, variable) in spec_map.items():
        match = exported[(exported["specification"] == specification) & (exported["variable"] == variable)]
        if match.empty:
            continue
        item = match.iloc[0]
        coef = float(item["coefficient"])
        rows.append(
            {
                "panel": "Panel A",
                "row_label": row_label,
                "model": "Exported logit hazard",
                "coefficient": coef,
                "std_error": float(item["std_error"]),
                "p_value": float(item["p_value"]),
                "marginal_effect": coef * slope if np.isfinite(slope) else np.nan,
                "n_obs": int(item["n_obs"]),
                "note": "Average-slope marginal effect proxy uses p(1-p) from the exported survival sample.",
            }
        )
    return rows


def _panel_a_lpm(survival: pd.DataFrame, connected: pd.DataFrame) -> list[dict]:
    merge = connected[["voyage_id", "theta", "psi", "captain_experience", "captain_id", "scarcity"]].drop_duplicates("voyage_id")
    df = survival.merge(merge, on="voyage_id", how="left", suffixes=("", "_connected"))
    if "captain_id_connected" in df.columns:
        df["captain_id"] = df["captain_id"].fillna(df["captain_id_connected"])
    if "scarcity_connected" in df.columns:
        df["scarcity"] = df["scarcity"].fillna(df["scarcity_connected"])

    df["neg_signal"] = (df["consecutive_empty_days"] >= NEG_SIGNAL_THRESHOLD).astype(float)
    df["pos_signal"] = df["encounter_today"].astype(float)
    df["psi_x_neg_signal"] = df["psi"] * df["neg_signal"]
    df["psi_x_pos_signal"] = df["psi"] * df["pos_signal"]

    model = _fit_clustered_lpm(
        df=df,
        outcome="exit_tomorrow",
        regressors=[
            "psi",
            "neg_signal",
            "psi_x_neg_signal",
            "pos_signal",
            "psi_x_pos_signal",
            "scarcity",
            "theta",
            "captain_experience",
            "day_in_patch",
        ],
        cluster_col="captain_id",
    )
    row_map = {
        "psi_hat": "psi",
        "negative signal": "neg_signal",
        "psi_hat × negative signal": "psi_x_neg_signal",
        "positive signal": "pos_signal",
        "psi_hat × positive signal": "psi_x_pos_signal",
        "scarcity": "scarcity",
        "theta_hat": "theta",
        "experience": "captain_experience",
    }
    rows = []
    for row_label, variable in row_map.items():
        rows.append(
            {
                "panel": "Panel A",
                "row_label": row_label,
                "model": "Clustered LPM supplement",
                "coefficient": model["coef"].get(variable, np.nan),
                "std_error": model["se"].get(variable, np.nan),
                "p_value": model["p"].get(variable, np.nan),
                "marginal_effect": model["coef"].get(variable, np.nan),
                "n_obs": int(model["n_obs"]),
                "note": "Linear-probability fallback used because statsmodels logit is not installed in the repository venv.",
            }
        )
    return rows


def _panel_b(action: pd.DataFrame, connected: pd.DataFrame, ground_quality: pd.DataFrame) -> list[dict]:
    if action.empty:
        return []
    voyage_merge = connected[["voyage_id", "theta", "psi", "captain_experience", "captain_id", "scarcity"]].drop_duplicates("voyage_id")
    quality_merge = (
        ground_quality[["voyage_id", "quality_loo_ground_year"]].drop_duplicates("voyage_id")
        if not ground_quality.empty
        else pd.DataFrame(columns=["voyage_id", "quality_loo_ground_year"])
    )
    df = action.merge(voyage_merge, on="voyage_id", how="left", suffixes=("", "_connected")).merge(quality_merge, on="voyage_id", how="left")
    if "captain_id_connected" in df.columns:
        df["captain_id"] = df["captain_id"].fillna(df["captain_id_connected"])
    sample = df[df["active_search_flag"] == 1].copy()
    signal_map = {
        "consecutive empty days": sample["consecutive_empty_days"],
        "days since last success": sample["days_since_last_success"],
        "barren-search-state indicator": (sample["consecutive_empty_days"] >= NEG_SIGNAL_THRESHOLD).astype(float),
        "leave-one-out local quality": sample.get("quality_loo_ground_year", pd.Series(index=sample.index, dtype=float)),
        "recent failure streak": (sample["consecutive_empty_days"] >= 5).astype(float),
    }

    rows = []
    for label, series in signal_map.items():
        sample = sample.copy()
        sample["signal"] = pd.to_numeric(series, errors="coerce")
        sample["psi_x_signal"] = sample["psi"] * sample["signal"]
        model = _fit_clustered_lpm(
            df=sample,
            outcome="exit_patch_next",
            regressors=["psi", "signal", "psi_x_signal", "theta", "captain_experience", "scarcity", "days_in_ground"],
            cluster_col="captain_id",
        )
        rows.append(
            {
                "panel": "Panel B",
                "row_label": label,
                "model": "Clustered LPM robustness",
                "coefficient": model["coef"].get("psi_x_signal", np.nan),
                "std_error": model["se"].get("psi_x_signal", np.nan),
                "p_value": model["p"].get("psi_x_signal", np.nan),
                "marginal_effect": model["coef"].get("psi_x_signal", np.nan),
                "n_obs": int(model["n_obs"]),
                "note": "Day-level robustness from the action dataset.",
            }
        )
    return rows


def _panel_c(action: pd.DataFrame, connected: pd.DataFrame, rational_exit: pd.DataFrame) -> list[dict]:
    if action.empty:
        return []
    voyage_merge = connected[["voyage_id", "theta", "psi", "captain_experience", "captain_id", "scarcity"]].drop_duplicates("voyage_id")
    df = action.merge(voyage_merge, on="voyage_id", how="left", suffixes=("", "_connected"))
    for column in ["theta", "psi", "captain_experience", "captain_id", "scarcity"]:
        connected_column = f"{column}_connected"
        if connected_column in df.columns:
            df[column] = _numeric(df.get(column), df.index).where(_numeric(df.get(column), df.index).notna(), _numeric(df.get(connected_column), df.index)) if column != "captain_id" else df[column].fillna(df[connected_column])
    df["duration_control"] = _numeric(df.get("days_in_ground"), df.index)
    if "days_in_patch" in df.columns:
        df["duration_control"] = df["duration_control"].fillna(_numeric(df.get("days_in_patch"), df.index))
    if "voyage_day" in df.columns:
        df["duration_control"] = df["duration_control"].fillna(_numeric(df.get("voyage_day"), df.index))
    df["duration_control"] = df["duration_control"].fillna(0.0)
    if "scarcity" in df.columns:
        df["scarcity"] = _numeric(df.get("scarcity"), df.index)
        if "scarcity_connected" in df.columns:
            df["scarcity"] = df["scarcity"].fillna(_numeric(df.get("scarcity_connected"), df.index))
        df["scarcity"] = df["scarcity"].fillna(float(df["scarcity"].median()) if df["scarcity"].notna().any() else 0.0)

    sample_map = {
        "active search": df[df["active_search_flag"] == 1].copy(),
        "transit": df[df["transit_flag"] == 1].copy(),
        "homebound": df[df["homebound_flag"] == 1].copy(),
        "productive/exploitation state": df[df["encounter"].fillna("NoEnc") != "NoEnc"].copy(),
    }

    rows = []
    for label, sample in sample_map.items():
        sample["neg_signal"] = (sample["consecutive_empty_days"] >= NEG_SIGNAL_THRESHOLD).astype(float)
        sample["psi_x_neg_signal"] = sample["psi"] * sample["neg_signal"]
        model = _fit_clustered_lpm(
            df=sample,
            outcome="exit_patch_next",
            regressors=["psi", "neg_signal", "psi_x_neg_signal", "theta", "captain_experience", "scarcity", "duration_control"],
            cluster_col="captain_id",
        )
        note = "Day-level placebo sample."
        if label == "transit" and not rational_exit.empty:
            transit_row = rational_exit.loc[rational_exit["test"] == "placebo_transit"]
            if not transit_row.empty:
                note = f"Existing next-round transit exit rate = {float(transit_row.iloc[0]['exit_rate']):.3f}."
        rows.append(
            {
                "panel": "Panel C",
                "row_label": label,
                "model": "Clustered LPM placebo",
                "coefficient": model["coef"].get("psi_x_neg_signal", np.nan),
                "std_error": model["se"].get("psi_x_neg_signal", np.nan),
                "p_value": model["p"].get("psi_x_neg_signal", np.nan),
                "marginal_effect": model["coef"].get("psi_x_neg_signal", np.nan),
                "n_obs": int(model["n_obs"]),
                "note": note,
            }
        )
    return rows


def _panel_d(survival: pd.DataFrame, connected: pd.DataFrame) -> list[dict]:
    merge = connected[["voyage_id", "theta", "psi", "captain_experience", "captain_id", "agent_id", "scarcity"]].drop_duplicates("voyage_id")
    df = survival.merge(merge, on="voyage_id", how="left", suffixes=("", "_connected"))
    for key in ["captain_id", "agent_id", "scarcity"]:
        other = f"{key}_connected"
        if other in df.columns:
            df[key] = df[key].where(df[key].notna(), df[other])
    df["neg_signal"] = (df["consecutive_empty_days"] >= NEG_SIGNAL_THRESHOLD).astype(float)
    df["pos_signal"] = _numeric(df.get("encounter_today"), df.index).fillna(0)
    df["psi_x_neg_signal"] = df["psi"] * df["neg_signal"]
    df["psi_x_pos_signal"] = df["psi"] * df["pos_signal"]
    regressors = ["psi", "neg_signal", "psi_x_neg_signal", "pos_signal", "psi_x_pos_signal", "scarcity", "theta", "captain_experience", "day_in_patch"]

    rows = []
    captain_fe = clustered_ols(df, outcome="exit_tomorrow", regressors=regressors, cluster_col="captain_id", fe_cols=["captain_id"])
    rows.append(
        {
            "panel": "Panel D",
            "row_label": "captain FE stopping rule",
            "model": "Captain FE LPM robustness",
            "coefficient": captain_fe["coef"].get("psi_x_neg_signal", np.nan),
            "std_error": captain_fe["se"].get("psi_x_neg_signal", np.nan),
            "p_value": captain_fe["p"].get("psi_x_neg_signal", np.nan),
            "marginal_effect": captain_fe["coef"].get("psi_x_neg_signal", np.nan),
            "n_obs": int(captain_fe["n_obs"]),
            "note": "Captain fixed-effects robustness for the psi × negative-signal stopping slope.",
        }
    )

    agent_cluster = _fit_clustered_lpm(df, outcome="exit_tomorrow", regressors=regressors, cluster_col="agent_id")
    rows.append(
        {
            "panel": "Panel D",
            "row_label": "agent-clustered inference",
            "model": "Agent-clustered LPM robustness",
            "coefficient": agent_cluster["coef"].get("psi_x_neg_signal", np.nan),
            "std_error": agent_cluster["se"].get("psi_x_neg_signal", np.nan),
            "p_value": agent_cluster["p"].get("psi_x_neg_signal", np.nan),
            "marginal_effect": agent_cluster["coef"].get("psi_x_neg_signal", np.nan),
            "n_obs": int(agent_cluster["n_obs"]),
            "note": "Same stopping slope, but clustering on agent rather than captain.",
        }
    )

    cox_proxy = _fit_clustered_lpm(df, outcome="event_exit", regressors=regressors, cluster_col="captain_id")
    rows.append(
        {
            "panel": "Panel D",
            "row_label": "Cox robustness",
            "model": "Spell-exit hazard proxy",
            "coefficient": cox_proxy["coef"].get("psi_x_neg_signal", np.nan),
            "std_error": cox_proxy["se"].get("psi_x_neg_signal", np.nan),
            "p_value": cox_proxy["p"].get("psi_x_neg_signal", np.nan),
            "marginal_effect": cox_proxy["coef"].get("psi_x_neg_signal", np.nan),
            "n_obs": int(cox_proxy["n_obs"]),
            "note": "Closest shipped proxy to a Cox robustness check, built on spell-end exit events because survival-model packages are not installed in the current venv.",
        }
    )

    spell = (
        df.sort_values(["patch_spell_id", "duration_day"])
        .groupby("patch_spell_id")
        .agg(
            log_duration=("duration_day", lambda s: np.log(float(np.nanmax(s)) + 1.0)),
            psi=("psi", "first"),
            theta=("theta", "first"),
            scarcity=("scarcity", "first"),
            captain_experience=("captain_experience", "first"),
            captain_id=("captain_id", "first"),
            neg_signal=("neg_signal", "max"),
        )
        .reset_index()
    )
    spell["psi_x_neg_signal"] = spell["psi"] * spell["neg_signal"]
    aft_proxy = _fit_clustered_lpm(
        spell,
        outcome="log_duration",
        regressors=["psi", "neg_signal", "psi_x_neg_signal", "theta", "scarcity", "captain_experience"],
        cluster_col="captain_id",
    )
    rows.append(
        {
            "panel": "Panel D",
            "row_label": "AFT robustness",
            "model": "Log-duration proxy",
            "coefficient": aft_proxy["coef"].get("psi_x_neg_signal", np.nan),
            "std_error": aft_proxy["se"].get("psi_x_neg_signal", np.nan),
            "p_value": aft_proxy["p"].get("psi_x_neg_signal", np.nan),
            "marginal_effect": aft_proxy["coef"].get("psi_x_neg_signal", np.nan),
            "n_obs": int(aft_proxy["n_obs"]),
            "note": "Log-duration proxy for an AFT-style check using spell lengths because dedicated survival-model packages are absent from the current venv.",
        }
    )
    return rows


def build(context: BuildContext):
    survival = load_survival_dataset(context)
    connected = load_connected_sample(context)
    action = load_action_dataset(context)
    ground_quality = load_ground_quality(context)
    exported = load_test3_stopping_output(context)
    rational_exit = load_rational_exit_output(context)

    frame = pd.DataFrame(
        _panel_a_from_logit(exported, survival)
        + _panel_a_lpm(survival, connected)
        + _panel_b(action, connected, ground_quality)
        + _panel_c(action, connected, rational_exit)
        + _panel_d(survival, connected)
    )

    memo = standard_footnote(
        sample="Panel A uses the exported patch-day stopping-rule sample and the connected voyage sample; Panels B and C use the action dataset's day-level search observations.",
        unit="Patch-day for exported logit rows; day-level search observation for LPM robustness rows.",
        types_note="The upstream exported hazard uses held-out psi_heldout; supplemental rows use connected-sample theta and psi from the outcome ML dataset.",
        fe="No fixed effects in the exported hazard file; captain clustering used in the supplemental paper builder.",
        cluster="Exported logit rows inherit voyage clustering from the saved reinforcement output; supplemental rows cluster by captain.",
        controls="Negative/positive signal indicators, scarcity, theta_hat, captain experience, and within-patch or within-ground duration controls.",
        interpretation="The strongest stopping evidence in the repository remains state contingent: exit behavior shifts with negative signals, and the sign/magnitude can be traced across active-search and placebo subsamples.",
        caution="Because `statsmodels` is not available in the current venv, rows marked as clustered LPM are linear-probability supplements rather than re-estimated logit hazards.",
    )
    memo = (
        "# table04_stopping\n\n"
        + memo
        + "\n\nImplementation notes:\n"
        + "- `Exported logit hazard` rows come directly from `output/reinforcement/tables/test3_stopping_rule.csv`.\n"
        + "- `Clustered LPM` rows are paper-layer supplements used to recover controls and placebo splits not preserved in the exported logit table.\n"
        + "- `Leave-one-out local quality` uses `quality_loo_ground_year` from `data/derived/ground_quality_loo.parquet` merged at the voyage level.\n"
        + "- Panel D records the closest feasible captain-FE, agent-clustered, Cox-style, and AFT-style robustness checks available in the current environment without `statsmodels` or `lifelines`.\n"
    )

    return save_table_outputs(
        name="table04_stopping",
        frame=frame,
        out_dir=context.outputs / "tables",
        context=context,
        memo=memo,
        title="Table 4. State-Contingent Stopping Rules",
    )
