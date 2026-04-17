from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from .utils import WSLReliabilityConfig, summarize_overlap

logger = logging.getLogger(__name__)


def _series_or_default(
    df: pd.DataFrame,
    column: str,
    default: float | int | str | None = np.nan,
) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series(default, index=df.index)


def _ensure_support_scores(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for column in [
        "portfolio_information_index",
        "information_advantage_index",
        "public_information_index",
        "risk_index",
        "psi_hat_holdout",
    ]:
        if column not in work.columns:
            work[column] = np.nan
    z_cols = {}
    for column in [
        "portfolio_information_index",
        "information_advantage_index",
        "public_information_index",
        "risk_index",
        "psi_hat_holdout",
    ]:
        series = pd.to_numeric(work[column], errors="coerce")
        std = series.std(ddof=0)
        z_cols[column] = ((series - series.mean()) / std) if pd.notna(std) and std > 0 else pd.Series(0.0, index=work.index)
    work["organizational_support_score"] = (
        z_cols["portfolio_information_index"]
        + z_cols["information_advantage_index"]
        + z_cols["public_information_index"]
        + z_cols["psi_hat_holdout"]
        - z_cols["risk_index"]
    ) / 5.0
    if "high_support_context_top_quartile" not in work.columns:
        cutoff = work["organizational_support_score"].quantile(0.75)
        work["high_support_context_top_quartile"] = (work["organizational_support_score"] >= cutoff).astype(int)
    if "high_portfolio_information_top_decile" not in work.columns:
        cutoff = work["portfolio_information_index"].quantile(0.90)
        work["high_portfolio_information_top_decile"] = (work["portfolio_information_index"] >= cutoff).astype(int)
    if "high_public_information_top_decile" not in work.columns:
        cutoff = work["public_information_index"].quantile(0.90)
        work["high_public_information_top_decile"] = (work["public_information_index"] >= cutoff).astype(int)
    return work


def _factor_encode(df: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
    work = df[columns].copy()
    metadata: dict[str, Any] = {"columns": columns, "categorical_maps": {}, "medians": {}}
    for column in columns:
        series = work[column]
        if series.dtype == "O" or str(series.dtype).startswith("category") or series.dtype == bool:
            series = series.fillna("UNK").astype(str)
            categories = {value: idx for idx, value in enumerate(sorted(series.unique()))}
            work[column] = series.map(categories).astype(float)
            metadata["categorical_maps"][column] = categories
        else:
            numeric = pd.to_numeric(series, errors="coerce")
            median = float(numeric.median()) if numeric.notna().any() else 0.0
            work[column] = numeric.fillna(median).astype(float)
            metadata["medians"][column] = median
    return work.to_numpy(dtype=float), metadata


def _time_folds(df: pd.DataFrame, time_col: str, n_folds: int = 3) -> np.ndarray:
    if time_col not in df.columns or pd.to_datetime(df[time_col], errors="coerce").notna().sum() < len(df) // 2:
        order = np.arange(len(df))
    else:
        time_values = pd.to_datetime(df[time_col], errors="coerce").astype("int64").to_numpy()
        order = np.argsort(time_values)
    folds = np.zeros(len(df), dtype=int)
    boundaries = np.array_split(order, n_folds)
    for fold_id, indices in enumerate(boundaries):
        folds[indices] = fold_id
    return folds


def _fit_propensity(X_train: np.ndarray, treatment_train: np.ndarray) -> Any:
    model = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, random_state=42)
    model.fit(X_train, treatment_train)
    return model


def _fit_outcome_model(X_train: np.ndarray, y_train: np.ndarray) -> Any:
    unique = np.unique(y_train[~pd.isna(y_train)])
    if set(unique.tolist()) <= {0, 1}:
        model = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, random_state=42)
    else:
        model = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    return model


def _predict_model(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def _crossfit_binary_policy(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_cols: list[str],
    covariates: list[str],
    time_col: str,
    config: WSLReliabilityConfig,
) -> dict[str, Any]:
    clean = df.dropna(subset=[treatment_col] + [column for column in outcome_cols if column in df.columns]).copy().reset_index(drop=True)
    X, encoding_meta = _factor_encode(clean, covariates)
    treatment = clean[treatment_col].astype(int).to_numpy()
    folds = _time_folds(clean, time_col)
    clean["propensity_score"] = np.nan
    for outcome in outcome_cols:
        clean[f"mu0__{outcome}"] = np.nan
        clean[f"mu1__{outcome}"] = np.nan

    for fold_id in np.unique(folds):
        train_idx = folds != fold_id
        test_idx = folds == fold_id
        propensity_model = _fit_propensity(X[train_idx], treatment[train_idx])
        clean.loc[test_idx, "propensity_score"] = np.clip(
            _predict_model(propensity_model, X[test_idx]),
            config.policy_propensity_clip_low,
            config.policy_propensity_clip_high,
        )
        for outcome in outcome_cols:
            outcome_values = clean[outcome].to_numpy()
            for treat_value, column_name in [(0, f"mu0__{outcome}"), (1, f"mu1__{outcome}")]:
                group_train = train_idx & (treatment == treat_value)
                if group_train.sum() < 25:
                    fallback = float(np.nanmean(outcome_values[group_train])) if group_train.sum() > 0 else float(np.nanmean(outcome_values[train_idx]))
                    clean.loc[test_idx, column_name] = fallback
                    continue
                model = _fit_outcome_model(X[group_train], outcome_values[group_train])
                clean.loc[test_idx, column_name] = _predict_model(model, X[test_idx])

    diagnostics = {
        "treatment_col": treatment_col,
        "time_col": time_col,
        "covariates": covariates,
        "overlap": summarize_overlap(
            clean["propensity_score"],
            config.policy_propensity_clip_low,
            config.policy_propensity_clip_high,
        ),
        "encoding_meta": encoding_meta,
        "n_rows": int(len(clean)),
        "treated_share": float(clean[treatment_col].mean()),
    }
    return {"scored_df": clean, "diagnostics": diagnostics}


def _utility_uplift(df: pd.DataFrame, config: WSLReliabilityConfig) -> pd.Series:
    output_gain = df["mu1__log_output"] - df["mu0__log_output"]
    failure_delta = df["mu1__zero_catch_or_failure"] - df["mu0__zero_catch_or_failure"]
    distress_delta = df["mu1__distress_burden_days"] - df["mu0__distress_burden_days"]
    return (
        config.utility_alpha * output_gain
        - config.utility_beta * failure_delta
        - config.utility_gamma * distress_delta
    )


def _reason_codes(row: pd.Series) -> tuple[str, str]:
    reasons: list[str] = []
    if pd.to_numeric(row.get("novice"), errors="coerce") == 1:
        reasons.append("novice_captain")
    if pd.to_numeric(row.get("risk_index"), errors="coerce") > 0.5:
        reasons.append("high_environment_risk")
    if pd.to_numeric(row.get("portfolio_information_index"), errors="coerce") < 0:
        reasons.append("thin_portfolio_information")
    if pd.to_numeric(row.get("information_advantage_index"), errors="coerce") > 0.25:
        reasons.append("agent_information_advantage")
    if pd.to_numeric(row.get("mu0__zero_catch_or_failure"), errors="coerce") > 0.5:
        reasons.append("high_failure_risk")
    if pd.to_numeric(row.get("utility_uplift"), errors="coerce") > 0:
        reasons.append("positive_local_uplift")
    if not reasons:
        reasons.append("baseline_case")
    summary = ", ".join(reasons[:4]).replace("_", " ")
    return "; ".join(reasons), summary


def fit_predeparture_policy_models(panel_df: pd.DataFrame, config: WSLReliabilityConfig) -> dict[str, Any]:
    work = _ensure_support_scores(panel_df)
    if "zero_catch_or_failure" not in work.columns:
        work["zero_catch_or_failure"] = (
            pd.to_numeric(_series_or_default(work, "q_total_index"), errors="coerce").fillna(0) <= 0
        ).astype(int)
    if "log_output" not in work.columns:
        work["log_output"] = np.log1p(
            pd.to_numeric(_series_or_default(work, "q_total_index"), errors="coerce").fillna(0).clip(lower=0)
        )
    if "ever_bad_state" not in work.columns:
        work["ever_bad_state"] = 0
    if "distress_burden_days" not in work.columns:
        work["distress_burden_days"] = 0.0
    covariates = [
        column
        for column in [
            "theta_hat_holdout",
            "novice",
            "tonnage",
            "home_port",
            "departure_destination_basin",
            "departure_month",
            "departure_decade",
            "public_information_index",
            "portfolio_information_index",
            "risk_index",
            "information_advantage_index",
            "agent_recent_bad_state_rate_tau180",
            "agent_recent_recovery_rate_tau180",
        ]
        if column in work.columns
    ]
    return _crossfit_binary_policy(
        work,
        treatment_col="high_support_context_top_quartile",
        outcome_cols=["zero_catch_or_failure", "log_output", "ever_bad_state", "distress_burden_days"],
        covariates=covariates,
        time_col="departure_issue_date" if "departure_issue_date" in work.columns else "year_out",
        config=config,
    )


def score_predeparture_policies(
    panel_df: pd.DataFrame,
    model_bundle: dict[str, Any],
    config: WSLReliabilityConfig,
) -> pd.DataFrame:
    scored = model_bundle["scored_df"].copy()
    if "mu1__log_output" not in scored.columns:
        return scored
    scored["utility_uplift"] = _utility_uplift(scored, config)
    scored["on_support"] = scored["propensity_score"].between(
        config.policy_propensity_clip_low,
        config.policy_propensity_clip_high,
        inclusive="both",
    )
    theta = pd.to_numeric(_series_or_default(scored, "theta_hat_holdout"), errors="coerce").fillna(0)
    novice = pd.to_numeric(_series_or_default(scored, "novice"), errors="coerce").fillna(0)
    scored["score_top_skill_first"] = theta
    scored["score_weak_first"] = (
        novice - theta
    )
    scored["score_failure_risk_first"] = scored["mu0__zero_catch_or_failure"]
    scored["score_highest_uplift_first"] = scored["utility_uplift"]
    scored["score_balanced_lambda_05"] = (
        0.5 * (scored["mu1__log_output"] - scored["mu0__log_output"])
        - 0.5 * (scored["mu1__zero_catch_or_failure"] - scored["mu0__zero_catch_or_failure"])
    )
    scored["score_balanced_lambda_20"] = (
        2.0 * (scored["mu1__log_output"] - scored["mu0__log_output"])
        - 1.0 * (scored["mu1__zero_catch_or_failure"] - scored["mu0__zero_catch_or_failure"])
    )
    reason_payload = scored.apply(_reason_codes, axis=1)
    scored["policy_reason_codes"] = reason_payload.map(lambda pair: pair[0])
    scored["policy_reason_summary"] = reason_payload.map(lambda pair: pair[1])
    return scored


def fit_triage_policy_models(triage_df: pd.DataFrame, config: WSLReliabilityConfig) -> dict[str, Any]:
    work = triage_df.copy()
    if "high_historical_recovery_context" not in work.columns:
        recovery_rate = pd.to_numeric(_series_or_default(work, "agent_recent_recovery_rate_tau180"), errors="coerce")
        cutoff = recovery_rate.quantile(0.75)
        work["high_historical_recovery_context"] = (
            recovery_rate.fillna(0) >= cutoff
        ).astype(int)
    if "recovery_within_90_days" not in work.columns:
        work["recovery_within_90_days"] = (
            pd.to_numeric(_series_or_default(work, "recovery_time_days"), errors="coerce").fillna(np.inf) <= 90
        ).astype(int)
    if "terminal_loss" not in work.columns:
        work["terminal_loss"] = pd.to_numeric(
            _series_or_default(work, "terminal_loss_indicator"), errors="coerce"
        ).fillna(0).astype(int)
    if "completed_arrival" not in work.columns:
        work["completed_arrival"] = pd.to_numeric(
            _series_or_default(work, "completed_arrival_indicator"), errors="coerce"
        ).fillna(0).astype(int)
    if "remaining_distress_burden_days" not in work.columns:
        work["remaining_distress_burden_days"] = pd.to_numeric(
            _series_or_default(work, "distress_burden_days"), errors="coerce"
        ).fillna(0.0)
    covariates = [
        column
        for column in [
            "theta_hat_holdout",
            "novice",
            "tonnage",
            "home_port",
            "departure_destination_basin",
            "triage_state",
            "agent_recent_bad_state_rate_tau180",
            "agent_recent_recovery_rate_tau180",
            "p_state__distress_at_sea",
            "p_state__in_port_interruption_or_repair",
            "p_state__terminal_loss",
            "risk_index",
        ]
        if column in work.columns
    ]
    return _crossfit_binary_policy(
        work,
        treatment_col="high_historical_recovery_context",
        outcome_cols=["recovery_within_90_days", "terminal_loss", "completed_arrival", "remaining_distress_burden_days"],
        covariates=covariates,
        time_col="triage_issue_date" if "triage_issue_date" in work.columns else "first_bad_state_issue_date",
        config=config,
    )


def score_triage_policies(
    triage_df: pd.DataFrame,
    model_bundle: dict[str, Any],
    config: WSLReliabilityConfig,
) -> pd.DataFrame:
    scored = model_bundle["scored_df"].copy()
    scored["utility_uplift"] = (
        config.utility_alpha * (scored["mu1__recovery_within_90_days"] - scored["mu0__recovery_within_90_days"])
        - config.utility_beta * (scored["mu1__terminal_loss"] - scored["mu0__terminal_loss"])
        - config.utility_gamma * (scored["mu1__remaining_distress_burden_days"] - scored["mu0__remaining_distress_burden_days"])
    )
    scored["on_support"] = scored["propensity_score"].between(
        config.policy_propensity_clip_low,
        config.policy_propensity_clip_high,
        inclusive="both",
    )
    theta = pd.to_numeric(_series_or_default(scored, "theta_hat_holdout"), errors="coerce").fillna(0)
    novice = pd.to_numeric(_series_or_default(scored, "novice"), errors="coerce").fillna(0)
    scored["score_top_skill_first"] = theta
    scored["score_weak_first"] = (
        novice - theta
    )
    scored["score_failure_risk_first"] = scored["mu0__terminal_loss"]
    scored["score_highest_uplift_first"] = scored["utility_uplift"]
    reason_payload = scored.apply(_reason_codes, axis=1)
    scored["policy_reason_codes"] = reason_payload.map(lambda pair: pair[0])
    scored["policy_reason_summary"] = reason_payload.map(lambda pair: pair[1])
    return scored


def _policy_assignment(score_df: pd.DataFrame, score_col: str, budget: float, seed: int) -> np.ndarray:
    on_support = score_df["on_support"].to_numpy(dtype=bool)
    assignment = np.zeros(len(score_df), dtype=int)
    eligible = np.where(on_support)[0]
    if score_col == "__observed__":
        assignment = score_df.iloc[:, score_df.columns.get_loc("treatment_observed")].to_numpy(dtype=int)
        return assignment
    if len(eligible) == 0:
        return assignment
    k = max(1, int(round(budget * len(eligible))))
    if score_col == "__random__":
        rng = np.random.default_rng(seed)
        chosen = rng.choice(eligible, size=min(k, len(eligible)), replace=False)
        assignment[chosen] = 1
        return assignment
    order = eligible[np.argsort(score_df.iloc[eligible][score_col].to_numpy())[::-1]]
    assignment[order[:k]] = 1
    return assignment


def _policy_value(score_df: pd.DataFrame, assignment: np.ndarray, config: WSLReliabilityConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    treat = score_df["treatment_observed"].to_numpy(dtype=int)
    e = score_df["propensity_score"].to_numpy(dtype=float)
    e = np.clip(e, config.policy_propensity_clip_low, config.policy_propensity_clip_high)
    y_output = score_df["outcome_output"].to_numpy(dtype=float)
    y_failure = score_df["outcome_failure"].to_numpy(dtype=float)
    y_distress = score_df["outcome_distress"].to_numpy(dtype=float)
    dr_output = assignment * (score_df["mu1_output"].to_numpy(dtype=float) + treat * (y_output - score_df["mu1_output"].to_numpy(dtype=float)) / e) + (1 - assignment) * (score_df["mu0_output"].to_numpy(dtype=float) + (1 - treat) * (y_output - score_df["mu0_output"].to_numpy(dtype=float)) / (1 - e))
    dr_failure = assignment * (score_df["mu1_failure"].to_numpy(dtype=float) + treat * (y_failure - score_df["mu1_failure"].to_numpy(dtype=float)) / e) + (1 - assignment) * (score_df["mu0_failure"].to_numpy(dtype=float) + (1 - treat) * (y_failure - score_df["mu0_failure"].to_numpy(dtype=float)) / (1 - e))
    dr_distress = assignment * (score_df["mu1_distress"].to_numpy(dtype=float) + treat * (y_distress - score_df["mu1_distress"].to_numpy(dtype=float)) / e) + (1 - assignment) * (score_df["mu0_distress"].to_numpy(dtype=float) + (1 - treat) * (y_distress - score_df["mu0_distress"].to_numpy(dtype=float)) / (1 - e))
    utility = config.utility_alpha * dr_output - config.utility_beta * dr_failure - config.utility_gamma * dr_distress
    return utility, dr_output, dr_failure


def evaluate_policy_frontiers(score_df: pd.DataFrame, config: WSLReliabilityConfig) -> pd.DataFrame:
    work = score_df.copy()
    if "high_support_context_top_quartile" in work.columns:
        work["treatment_observed"] = work["high_support_context_top_quartile"].astype(int)
        work["outcome_failure"] = work["zero_catch_or_failure"].astype(float)
        work["outcome_output"] = work["log_output"].astype(float)
        work["outcome_distress"] = work["distress_burden_days"].astype(float)
        work["mu0_failure"] = work["mu0__zero_catch_or_failure"]
        work["mu1_failure"] = work["mu1__zero_catch_or_failure"]
        work["mu0_output"] = work["mu0__log_output"]
        work["mu1_output"] = work["mu1__log_output"]
        work["mu0_distress"] = work["mu0__distress_burden_days"]
        work["mu1_distress"] = work["mu1__distress_burden_days"]
    else:
        work["treatment_observed"] = work["high_historical_recovery_context"].astype(int)
        work["outcome_failure"] = work["terminal_loss"].astype(float)
        work["outcome_output"] = work["recovery_within_90_days"].astype(float)
        work["outcome_distress"] = work["remaining_distress_burden_days"].astype(float)
        work["mu0_failure"] = work["mu0__terminal_loss"]
        work["mu1_failure"] = work["mu1__terminal_loss"]
        work["mu0_output"] = work["mu0__recovery_within_90_days"]
        work["mu1_output"] = work["mu1__recovery_within_90_days"]
        work["mu0_distress"] = work["mu0__remaining_distress_burden_days"]
        work["mu1_distress"] = work["mu1__remaining_distress_burden_days"]

    frontier_rows: list[dict[str, Any]] = []
    score_columns = {
        "observed_historical_assignment": "__observed__",
        "random_assignment": "__random__",
        "top_skill_first": "score_top_skill_first",
        "weak_first": "score_weak_first",
        "highest_failure_risk_first": "score_failure_risk_first",
        "highest_uplift_first": "score_highest_uplift_first",
    }
    for column in [col for col in work.columns if col.startswith("score_balanced_lambda_")]:
        score_columns[column.replace("score_", "")] = column

    for budget in config.policy_budget_grid:
        for policy_name, score_col in score_columns.items():
            assignment = _policy_assignment(work, score_col, budget, seed=config.random_seed)
            utility, dr_output, dr_failure = _policy_value(work, assignment, config)
            boot_values = []
            rng = np.random.default_rng(config.random_seed)
            for _ in range(config.policy_bootstrap_reps):
                sample_idx = rng.integers(0, len(work), len(work))
                boot_values.append(float(np.mean(utility[sample_idx])))
            frontier_rows.append(
                {
                    "policy_name": policy_name,
                    "budget": budget,
                    "treated_share": float(assignment.mean()),
                    "n_on_support": int(work["on_support"].sum()),
                    "policy_value": float(np.mean(utility)),
                    "policy_value_ci_low": float(np.quantile(boot_values, 0.025)),
                    "policy_value_ci_high": float(np.quantile(boot_values, 0.975)),
                    "expected_output_component": float(np.mean(dr_output)),
                    "expected_failure_component": float(np.mean(dr_failure)),
                }
            )
    return pd.DataFrame(frontier_rows)


def run_information_equalization_counterfactual(panel_df: pd.DataFrame, config: WSLReliabilityConfig) -> pd.DataFrame:
    work = _ensure_support_scores(panel_df)
    info_columns = [
        column
        for column in [
            "public_information_index",
            "portfolio_information_index",
            "information_advantage_index",
            "risk_index",
        ]
        if column in work.columns
    ]
    base_covariates = [
        column
        for column in [
            "theta_hat_holdout",
            "novice",
            "tonnage",
            "departure_destination_basin",
            "departure_decade",
        ]
        if column in work.columns
    ]
    if not info_columns:
        return pd.DataFrame(columns=["group", "n_rows", "mean_output_gain", "mean_failure_reduction"])

    work["support_group"] = pd.qcut(work["organizational_support_score"], q=4, labels=False, duplicates="drop")
    strong = work[work["support_group"] == work["support_group"].max()].copy()
    weak = work[work["support_group"] == work["support_group"].min()].copy()
    if strong.empty or weak.empty:
        return pd.DataFrame(columns=["group", "n_rows", "mean_output_gain", "mean_failure_reduction"])

    full_covariates = list(dict.fromkeys(base_covariates + info_columns))
    model_frame = work.dropna(subset=["zero_catch_or_failure", "log_output"] + full_covariates).copy()
    X, meta = _factor_encode(model_frame, full_covariates)
    output_model = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, random_state=config.random_seed)
    failure_model = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, random_state=config.random_seed)
    output_model.fit(X, model_frame["log_output"].astype(float))
    failure_model.fit(X, model_frame["zero_catch_or_failure"].astype(int))

    weak = weak.dropna(subset=base_covariates + info_columns).copy().reset_index(drop=True)
    strong = strong.dropna(subset=base_covariates + info_columns).copy().reset_index(drop=True)
    if weak.empty or strong.empty:
        return pd.DataFrame(columns=["group", "n_rows", "mean_output_gain", "mean_failure_reduction"])

    strong_base, _ = _factor_encode(strong, base_covariates)
    weak_base, _ = _factor_encode(weak, base_covariates)
    donor_rows = []
    for i in range(len(weak)):
        distances = np.linalg.norm(strong_base - weak_base[i], axis=1)
        k = min(5, len(strong))
        donor_idx = np.argsort(distances)[:k]
        donor_rows.append(strong.iloc[donor_idx][info_columns].mean())
    donor_info = pd.DataFrame(donor_rows)
    counterfactual = weak.copy()
    for column in info_columns:
        lower = strong[column].quantile(0.05)
        upper = strong[column].quantile(0.95)
        counterfactual[column] = donor_info[column].clip(lower=lower, upper=upper)

    observed_X, _ = _factor_encode(weak[full_covariates], full_covariates)
    counterfactual_X, _ = _factor_encode(counterfactual[full_covariates], full_covariates)
    observed_output = output_model.predict(observed_X)
    equalized_output = output_model.predict(counterfactual_X)
    observed_failure = failure_model.predict_proba(observed_X)[:, 1]
    equalized_failure = failure_model.predict_proba(counterfactual_X)[:, 1]
    result = pd.DataFrame(
        {
            "group": "bottom_quartile_support",
            "n_rows": [int(len(weak))],
            "mean_output_gain": [float(np.mean(equalized_output - observed_output))],
            "mean_failure_reduction": [float(np.mean(observed_failure - equalized_failure))],
            "share_on_support": [float((weak["organizational_support_score"] <= strong["organizational_support_score"].max()).mean())],
        }
    )
    novice_breakout = weak.groupby(weak.get("novice", pd.Series(0, index=weak.index))).size().to_dict()
    if novice_breakout:
        result["novice_breakout"] = [str(novice_breakout)]
    return result
