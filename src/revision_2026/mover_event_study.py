"""
Mover and Event-Study Extensions (Step 5).

  5A  Preferred mover reruns with Lévy exponent (μ) as outcome
  5B  Event study around organization switches with μ as outcome

Reuses search_theory.py for Lévy estimation and event_study.py for
switch identification.  All psi_hat objects use separate-sample
construction.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .config import CFG, DATA_FINAL, VOYAGE_PARQUET, INTERMEDIATES_DIR, TABLES_DIR, FIGURES_DIR
from .output_schema import build_tidy_row, rows_to_tidy_df, save_result_table, save_markdown_table

logger = logging.getLogger(__name__)


def _load_mover_panel() -> pd.DataFrame:
    """Load voyage panel with switch indicators and Lévy data."""
    df = pd.read_parquet(VOYAGE_PARQUET)
    df = df.dropna(subset=["captain_id", "agent_id", "year_out"]).copy()

    q_col = "q_total_index" if "q_total_index" in df.columns else "q_oil_bbl"
    if q_col in df.columns:
        df = df[df[q_col] > 0].copy()
        df["log_q"] = np.log(df[q_col].clip(lower=1e-6))

    # Controls
    if "tonnage" in df.columns:
        df["log_tonnage"] = np.log(df["tonnage"].clip(lower=1))
    if "duration_days" in df.columns:
        df["log_duration"] = np.log(df["duration_days"].clip(lower=1))
    for col in ["log_tonnage", "log_duration"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    if "year_out" in df.columns:
        df["decade"] = (df["year_out"] // 10) * 10

    # Sort and compute switches
    df = df.sort_values(["captain_id", "year_out", "date_out"]).copy()
    df["prev_agent"] = df.groupby("captain_id")["agent_id"].shift(1)
    df["switched"] = (
        (df["agent_id"] != df["prev_agent"]) & df["prev_agent"].notna()
    ).astype(int)
    df["voyage_num"] = df.groupby("captain_id").cumcount()

    # Movers: captains with 2+ agents
    cap_agents = df.groupby("captain_id")["agent_id"].nunique()
    movers = cap_agents[cap_agents >= 2].index
    df["is_mover"] = df["captain_id"].isin(movers).astype(int)

    return df


def _merge_effects(df: pd.DataFrame) -> pd.DataFrame:
    """Merge separate-sample effects."""
    psi_path = INTERMEDIATES_DIR / "psi_hat_leave_one_captain_out.parquet"
    if psi_path.exists():
        psi_loo = pd.read_parquet(psi_path)
        df = df.merge(psi_loo[["voyage_id", "psi_hat_loo"]], on="voyage_id", how="left")
        df["psi"] = df["psi_hat_loo"]
        df["effect_source"] = "loo"
    else:
        logger.warning("  LOO psi_hat not found")
        df["psi"] = np.nan
        df["effect_source"] = "none"

    theta_path = INTERMEDIATES_DIR / "theta_hat_separate_sample.parquet"
    if theta_path.exists():
        theta = pd.read_parquet(theta_path)
        df = df.merge(theta[["captain_id", "theta_hat_sep"]], on="captain_id", how="left")
        df["theta"] = df["theta_hat_sep"]

    for col in ["psi", "theta"]:
        if col in df.columns and df[col].notna().sum() > 10:
            mu, sigma = df[col].mean(), df[col].std()
            df[f"{col}_std"] = (df[col] - mu) / sigma if sigma > 0 else 0

    return df


# =====================================================================
# 5A: Mover regressions (within-captain Δμ)
# =====================================================================

def run_mover_regressions(df: pd.DataFrame) -> pd.DataFrame:
    """Within-captain mover design: Δμ ~ Δψ."""
    logger.info("  --- 5A: Mover regressions (Lévy exponent) ---")

    sample = df[df["is_mover"] == 1].copy()

    # Try to get Lévy exponent
    mu_col = None
    for candidate in ["levy_exponent", "mu", "levy_mu"]:
        if candidate in sample.columns:
            mu_col = candidate
            break

    outcomes = [("log_q", "log_q")]
    if mu_col:
        outcomes.append((mu_col, "levy_mu"))
        logger.info("    Found Lévy exponent column: %s", mu_col)
    else:
        logger.warning("    No Lévy exponent column found; running log_q only")

    tidy_rows = []

    for outcome_col, outcome_label in outcomes:
        valid = sample.dropna(subset=[outcome_col, "psi_std"]).copy()
        if len(valid) < 50:
            logger.warning("    Insufficient observations for %s: %d", outcome_label, len(valid))
            continue

        # Spec 1: Captain FE only
        try:
            fe_cap = pd.get_dummies(valid["captain_id"], prefix="cap", drop_first=True, dtype=int)
            X1 = pd.concat([valid[["psi_std"]], fe_cap], axis=1)
            X1 = sm.add_constant(X1, has_constant="add")
            y1 = valid[outcome_col]
            m1 = sm.OLS(y1, X1).fit(
                cov_type="cluster", cov_kwds={"groups": valid["captain_id"]}
            )
            tidy_rows.append(build_tidy_row(
                outcome=outcome_label,
                sample_name="movers",
                spec_name="captain_FE",
                term="psi_std",
                estimate=float(m1.params["psi_std"]),
                std_error=float(m1.bse["psi_std"]),
                n_obs=len(y1),
                n_captains=int(valid["captain_id"].nunique()),
                n_agents=int(valid["agent_id"].nunique()),
                fixed_effects="captain",
                cluster_scheme="captain_id",
                effect_object_used=valid["effect_source"].iloc[0] if "effect_source" in valid else "unknown",
            ))
        except Exception as e:
            logger.warning("    Captain FE spec failed for %s: %s", outcome_label, e)

        # Spec 2: Captain FE + route×decade FE
        route_col = "ground_or_route" if "ground_or_route" in valid.columns else None
        if route_col and "decade" in valid.columns:
            try:
                valid["route_decade"] = valid[route_col].astype(str) + "_" + valid["decade"].astype(str)
                fe_rt = pd.get_dummies(valid["route_decade"], prefix="rt", drop_first=True, dtype=int)
                X2 = pd.concat([valid[["psi_std"]], fe_cap, fe_rt], axis=1)
                X2 = sm.add_constant(X2, has_constant="add")
                y2 = valid[outcome_col]
                m2 = sm.OLS(y2, X2).fit(
                    cov_type="cluster", cov_kwds={"groups": valid["captain_id"]}
                )
                tidy_rows.append(build_tidy_row(
                    outcome=outcome_label,
                    sample_name="movers",
                    spec_name="captain_FE_route_decade_FE",
                    term="psi_std",
                    estimate=float(m2.params["psi_std"]),
                    std_error=float(m2.bse["psi_std"]),
                    n_obs=len(y2),
                    n_captains=int(valid["captain_id"].nunique()),
                    n_agents=int(valid["agent_id"].nunique()),
                    fixed_effects="captain, route×decade",
                    cluster_scheme="captain_id",
                    effect_object_used=valid["effect_source"].iloc[0] if "effect_source" in valid else "unknown",
                ))
            except Exception as e:
                logger.warning("    Route×Decade FE spec failed for %s: %s", outcome_label, e)

        # Spec 3: + hardware controls (log_tonnage, rig)
        try:
            hw_cols = [c for c in ["log_tonnage"] if c in valid.columns]
            if "rig" in valid.columns:
                valid["is_ship"] = (valid["rig"] == "Ship").astype(float)
                hw_cols.append("is_ship")
            if hw_cols:
                X3_parts = [valid[["psi_std"] + hw_cols], fe_cap]
                if route_col and "fe_rt" in dir():
                    X3_parts.append(fe_rt)
                X3 = pd.concat(X3_parts, axis=1)
                X3 = sm.add_constant(X3, has_constant="add")
                y3 = valid[outcome_col]
                m3 = sm.OLS(y3, X3).fit(
                    cov_type="cluster", cov_kwds={"groups": valid["captain_id"]}
                )
                tidy_rows.append(build_tidy_row(
                    outcome=outcome_label,
                    sample_name="movers",
                    spec_name="captain_FE_route_decade_FE_hardware",
                    term="psi_std",
                    estimate=float(m3.params["psi_std"]),
                    std_error=float(m3.bse["psi_std"]),
                    n_obs=len(y3),
                    n_captains=int(valid["captain_id"].nunique()),
                    n_agents=int(valid["agent_id"].nunique()),
                    fixed_effects="captain, route×decade, hardware",
                    cluster_scheme="captain_id",
                    effect_object_used=valid["effect_source"].iloc[0] if "effect_source" in valid else "unknown",
                ))
        except Exception as e:
            logger.warning("    Hardware controls spec failed for %s: %s", outcome_label, e)

    result = rows_to_tidy_df(tidy_rows)
    save_result_table(result, "table_mover_preferred", metadata={
        "description": "Within-captain mover regressions with multiple outcomes (log_q, μ)",
    })
    logger.info("    Saved %d coefficient rows", len(result))
    return result


# =====================================================================
# 5B: Event study around organization switches
# =====================================================================

def run_switch_event_study(df: pd.DataFrame) -> pd.DataFrame:
    """Event study: balanced-panel design around agent switches."""
    logger.info("  --- 5B: Switch event study ---")

    # Identify first switch per captain
    switch_df = df[df["switched"] == 1].drop_duplicates("captain_id", keep="first")[
        ["captain_id", "voyage_num"]
    ].rename(columns={"voyage_num": "switch_voyage"})

    df = df.merge(switch_df, on="captain_id", how="left")
    df["event_time"] = df["voyage_num"] - df["switch_voyage"]

    # Balanced window
    window = list(range(-CFG.event_min_pre, CFG.event_min_post + 1))
    df_window = df[df["event_time"].isin(window)].copy()

    # Keep only captains with complete windows
    counts = df_window.groupby("captain_id")["event_time"].nunique()
    balanced_caps = counts[counts == len(window)].index
    df_bal = df_window[df_window["captain_id"].isin(balanced_caps)].copy()

    logger.info("    Balanced captains: %d / %d (window %s)",
                len(balanced_caps), len(switch_df), window)
    logger.info("    Balanced observations: %d", len(df_bal))

    if len(df_bal) < 30:
        logger.warning("    Insufficient balanced observations; skipping event study")
        return pd.DataFrame()

    outcomes = ["log_q"]
    for mu_candidate in ["levy_exponent", "mu", "levy_mu"]:
        if mu_candidate in df_bal.columns and df_bal[mu_candidate].notna().sum() > 20:
            outcomes.append(mu_candidate)
            break

    tidy_rows = []
    for outcome_col in outcomes:
        outcome_label = "levy_mu" if outcome_col != "log_q" else "log_q"
        df_sub = df_bal.dropna(subset=[outcome_col]).copy()
        if len(df_sub) < 20:
            continue

        # Time dummies (omit t = -1 as reference)
        time_dummies = pd.get_dummies(df_sub["event_time"].astype(int), prefix="t", dtype=int)
        if "t_-1" in time_dummies.columns:
            time_dummies = time_dummies.drop(columns=["t_-1"])

        # Captain FE
        fe_cap = pd.get_dummies(df_sub["captain_id"], prefix="cap", drop_first=True, dtype=int)
        X = pd.concat([time_dummies, fe_cap], axis=1)
        X = sm.add_constant(X, has_constant="add")
        y = df_sub[outcome_col]

        try:
            model = sm.OLS(y, X).fit(
                cov_type="cluster", cov_kwds={"groups": df_sub["captain_id"]}
            )

            for t in window:
                if t == -1:
                    continue  # reference period
                col_name = f"t_{t}"
                if col_name in model.params.index:
                    tidy_rows.append(build_tidy_row(
                        outcome=outcome_label,
                        sample_name="event_study_balanced",
                        spec_name="captain_FE",
                        term=f"event_time_{t}",
                        estimate=float(model.params[col_name]),
                        std_error=float(model.bse[col_name]),
                        n_obs=len(y),
                        n_captains=len(balanced_caps),
                        fixed_effects="captain",
                        cluster_scheme="captain_id",
                        notes=f"Reference: t=-1",
                    ))

            # Joint pre-trend F-test
            pre_terms = [f"t_{t}" for t in window if t < -1 and f"t_{t}" in model.params.index]
            if pre_terms:
                try:
                    constraints = " = ".join([f"{t} = 0" for t in pre_terms])
                    # Simple approach: test each pre-trend coefficient = 0
                    for pt in pre_terms:
                        f_test = model.f_test(f"{pt} = 0")
                        tidy_rows.append(build_tidy_row(
                            outcome=outcome_label,
                            sample_name="event_study_balanced",
                            spec_name="pretrend_test",
                            term=pt,
                            estimate=float(model.params[pt]),
                            std_error=float(model.bse[pt]),
                            n_obs=len(y),
                            notes=f"Pre-trend F-test p={float(f_test.pvalue):.4f}",
                        ))
                except Exception:
                    pass

        except Exception as e:
            logger.warning("    Event study model failed for %s: %s", outcome_label, e)

    result = rows_to_tidy_df(tidy_rows)
    save_result_table(result, "table_event_study_pretrend_tests", metadata={
        "description": "Event study coefficients and pre-trend tests around agent switches",
        "window": window,
        "reference_period": -1,
    })

    # Save event-study data for figure
    es_fig_data = result[
        (result["spec_name"] == "captain_FE") & result["term"].str.startswith("event_time_")
    ][["outcome", "term", "estimate", "std_error", "ci_low", "ci_high"]].copy()
    es_fig_data.to_csv(FIGURES_DIR / "figure_event_study_mu.csv", index=False)

    logger.info("    Saved %d coefficient rows", len(result))
    return result


# =====================================================================
# Master entry point
# =====================================================================

def run_all_mover_event_study_tests() -> Dict[str, pd.DataFrame]:
    """Execute all mover and event-study tests (Step 5)."""
    logger.info("=" * 60)
    logger.info("STEP 5: MOVER AND EVENT-STUDY TESTS")
    logger.info("=" * 60)

    df = _load_mover_panel()
    df = _merge_effects(df)

    n_movers = df[df["is_mover"] == 1]["captain_id"].nunique()
    n_switches = df["switched"].sum()
    logger.info("Panel: %d voyages, %d movers, %d switches", len(df), n_movers, n_switches)

    results = {}
    results["mover_regressions"] = run_mover_regressions(df)
    results["event_study"] = run_switch_event_study(df)

    logger.info("Mover and event-study tests complete.")
    return results
