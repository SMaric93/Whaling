from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .._table_common import save_table_outputs
from ..config import BuildContext
from ..data import load_action_dataset, load_connected_sample
from ..utils.footnotes import standard_footnote
from ..utils.inference import clustered_ols, normal_pvalue
from ..utils.risk import expected_shortfall_proxy, lower_tail_reference


def _numeric(series: pd.Series | None, index: pd.Index | None = None) -> pd.Series:
    if series is None:
        return pd.Series(np.nan, index=index, dtype=float)
    return pd.to_numeric(series, errors="coerce")


def _prepare_sample(connected: pd.DataFrame, action: pd.DataFrame) -> pd.DataFrame:
    dry_spells = (
        action.groupby("voyage_id")
        .agg(
            max_consecutive_empty_days=("consecutive_empty_days", "max"),
            mean_days_since_last_success=("days_since_last_success", "mean"),
        )
        .reset_index()
        if not action.empty
        else pd.DataFrame(columns=["voyage_id", "max_consecutive_empty_days", "mean_days_since_last_success"])
    )
    df = connected.merge(dry_spells, on="voyage_id", how="left")
    for col in [
        "q_total_index",
        "theta",
        "psi",
        "scarcity",
        "captain_experience",
        "duration_days",
        "max_consecutive_empty_days",
        "year_out",
    ]:
        if col in df.columns:
            df[col] = _numeric(df[col], df.index)

    outcome = df["q_total_index"]
    p10 = lower_tail_reference(outcome, quantile=0.10)
    p05 = float(outcome.quantile(0.05))
    df["bottom_decile"] = (outcome <= p10).astype(float)
    df["bottom_5pct"] = (outcome <= p05).astype(float)
    df["catastrophic_voyage"] = ((df["bottom_5pct"] > 0) & (df["duration_days"] >= df["duration_days"].median())).astype(float)
    dry_cutoff = float(df["max_consecutive_empty_days"].quantile(0.90)) if df["max_consecutive_empty_days"].notna().any() else np.nan
    df["long_dry_spell"] = (df["max_consecutive_empty_days"] >= dry_cutoff).astype(float) if np.isfinite(dry_cutoff) else np.nan
    df["expected_shortfall_proxy"] = expected_shortfall_proxy(outcome, quantile=0.10)
    df["high_psi"] = (df["psi"] >= df["psi"].median()).astype(float)

    df = df.sort_values(["captain_id", "year_out", "voyage_id"]).copy()
    df["prior_output_volatility"] = (
        df.groupby("captain_id")["q_total_index"]
        .transform(lambda s: s.expanding().std().shift(1))
    )
    df["within_captain_sq_dev"] = df["q_total_index"] - df.groupby("captain_id")["q_total_index"].transform("mean")
    df["within_captain_sq_dev"] = df["within_captain_sq_dev"] ** 2
    return df


def _panel_a(df: pd.DataFrame) -> list[dict]:
    outcomes = [
        ("bottom decile output", "bottom_decile"),
        ("bottom 5% output", "bottom_5pct"),
        ("catastrophic voyage", "catastrophic_voyage"),
        ("long dry spell", "long_dry_spell"),
        ("expected shortfall proxy", "expected_shortfall_proxy"),
    ]
    rows = []
    for row_label, outcome in outcomes:
        model = clustered_ols(
            df,
            outcome=outcome,
            regressors=["high_psi", "theta", "captain_experience"],
            cluster_col="captain_id",
        )
        prevalence = float(df[outcome].gt(0).mean()) if outcome == "expected_shortfall_proxy" else float(df[outcome].mean())
        rows.append(
            {
                "panel": "Panel A",
                "row_label": row_label,
                "marginal_effect_of_high_psi": model["coef"].get("high_psi", np.nan),
                "std_error": model["se"].get("high_psi", np.nan),
                "p_value": model["p"].get("high_psi", np.nan),
                "baseline_prevalence": prevalence,
                "n_obs": int(model["n_obs"]),
                "note": "High-psi is defined as above-median agent capability in the connected voyage sample.",
            }
        )
    return rows


def _panel_b(df: pd.DataFrame) -> list[dict]:
    theta_median = float(df["theta"].median())
    scarcity_median = float(df["scarcity"].median())
    prior_vol_median = float(df["prior_output_volatility"].median())
    groups = [
        ("novice", df["novice"].fillna(0).astype(float) > 0),
        ("expert", df["novice"].fillna(0).astype(float) == 0),
        ("low theta", df["theta"] <= theta_median),
        ("high theta", df["theta"] > theta_median),
        ("high prior volatility", df["prior_output_volatility"] >= prior_vol_median),
        ("low prior volatility", df["prior_output_volatility"] < prior_vol_median),
        ("sparse grounds", df["scarcity"] >= scarcity_median),
        ("rich grounds", df["scarcity"] < scarcity_median),
    ]
    rows = []
    for row_label, mask in groups:
        sample = df.loc[mask].copy()
        model = clustered_ols(
            sample,
            outcome="bottom_decile",
            regressors=["psi", "theta", "captain_experience"],
            cluster_col="captain_id",
        )
        rows.append(
            {
                "panel": "Panel B",
                "row_label": row_label,
                "marginal_effect_of_psi": model["coef"].get("psi", np.nan),
                "std_error": model["se"].get("psi", np.nan),
                "p_value": model["p"].get("psi", np.nan),
                "n_obs": int(model["n_obs"]),
                "note": "Subgroup effect estimated on bottom-decile risk, the focal downside-risk outcome.",
            }
        )
    return rows


def _bootstrap_quantile_effect(df: pd.DataFrame, quantile: float, n_boot: int = 80) -> tuple[float, float, float]:
    clean = df.dropna(subset=["q_total_index", "high_psi"]).copy()
    high = clean.loc[clean["high_psi"] == 1, "q_total_index"]
    low = clean.loc[clean["high_psi"] == 0, "q_total_index"]
    if high.empty or low.empty:
        return np.nan, np.nan, np.nan
    effect = float(high.quantile(quantile) - low.quantile(quantile))
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(n_boot):
        sample = clean.sample(n=len(clean), replace=True, random_state=int(rng.integers(0, 1_000_000)))
        high_b = sample.loc[sample["high_psi"] == 1, "q_total_index"]
        low_b = sample.loc[sample["high_psi"] == 0, "q_total_index"]
        if high_b.empty or low_b.empty:
            continue
        boot.append(float(high_b.quantile(quantile) - low_b.quantile(quantile)))
    se = float(np.std(boot, ddof=1)) if len(boot) > 1 else np.nan
    p_value = normal_pvalue(effect, se)
    return effect, se, p_value


def _panel_c(df: pd.DataFrame) -> list[dict]:
    rows = []
    for row_label, quantile in [("P10", 0.10), ("P25", 0.25), ("P50", 0.50), ("P75", 0.75), ("P90", 0.90)]:
        effect, se, p_value = _bootstrap_quantile_effect(df, quantile)
        rows.append(
            {
                "panel": "Panel C",
                "row_label": row_label,
                "effect_of_psi": effect,
                "std_error": se,
                "p_value": p_value,
                "n_obs": int(df["q_total_index"].notna().sum()),
                "note": "Effect is the high-psi minus low-psi quantile gap with nonparametric bootstrap uncertainty.",
            }
        )
    variance_model = clustered_ols(
        df,
        outcome="within_captain_sq_dev",
        regressors=["psi", "theta", "captain_experience"],
        cluster_col="captain_id",
    )
    rows.append(
        {
            "panel": "Panel C",
            "row_label": "within-captain variance",
            "effect_of_psi": variance_model["coef"].get("psi", np.nan),
            "std_error": variance_model["se"].get("psi", np.nan),
            "p_value": variance_model["p"].get("psi", np.nan),
            "n_obs": int(variance_model["n_obs"]),
            "note": "Estimated as the psi slope in a regression for squared deviation from the captain-specific mean output.",
        }
    )
    return rows


def build(context: BuildContext):
    connected = load_connected_sample(context)
    action = load_action_dataset(context)
    df = _prepare_sample(connected, action)

    frame = pd.DataFrame(_panel_a(df) + _panel_b(df) + _panel_c(df))

    memo = standard_footnote(
        sample="Connected-set voyages with action-derived dry-spell measures merged from the daily action panel.",
        unit="Voyage throughout.",
        types_note="theta_hat and psi_hat are connected-set AKM types; Panel A uses a high-psi indicator, while Panels B and C use continuous psi or high-vs-low psi gaps as noted.",
        fe="No fixed effects in the main table; within-captain dispersion in Panel C uses captain-centered squared deviations.",
        cluster="Captain clustering for regression-based rows; Panel C quantile rows use nonparametric bootstrap uncertainty.",
        controls="theta_hat and captain experience in the main rows, with scarcity and prior output volatility used to define heterogeneity cells rather than soaked up as baseline controls.",
        interpretation="High-psi organizations materially improve output-tail outcomes and expected shortfall; the longest dry-spell proxy is more mixed, which is consistent with capable organizations sometimes sustaining longer search before abandonment.",
        caution="Expected shortfall and long-dry-spell rows use paper-layer proxies built from shipped voyage and action panels rather than a dedicated archival downside-risk file, so the dry-spell row should be read as suggestive rather than decisive.",
    )
    memo = (
        "# table08_floor_raising\n\n"
        + memo
        + "\n\nImplementation notes:\n"
        + "- `long dry spell` is defined from the voyage-level maximum consecutive empty-day streak aggregated from the action panel.\n"
        + "- `expected shortfall proxy` is the shortfall below the first nondegenerate lower-tail cutoff: the voyage-output 10th percentile when positive, or the positive-output 10th percentile when the overall cutoff is zero.\n"
        + "- Panel C quantile rows are unconditional distributional gaps between high- and low-psi voyages with bootstrap uncertainty.\n"
    )

    return save_table_outputs(
        name="table08_floor_raising",
        frame=frame,
        out_dir=context.outputs / "tables",
        context=context,
        memo=memo,
        title="Table 8. Floor-Raising and Downside-Risk Reduction",
    )
