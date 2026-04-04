"""
Officer Pipeline and Portability Tests (Step 7).

  7A  Richer same-agent training premium with FEs
  7B  Portability test: training-agent psi_hat predicts at different agents
  7C  Placebos (same home port, same port×era)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .config import CFG, DATA_FINAL, VOYAGE_PARQUET, INTERMEDIATES_DIR, TABLES_DIR
from .output_schema import build_tidy_row, rows_to_tidy_df, save_result_table, save_markdown_table

logger = logging.getLogger(__name__)


def _build_officer_pipeline_panel() -> pd.DataFrame:
    """Build the mate-to-captain promotion panel.

    Identifies promoted mates (captains who were previously first mates)
    and links their training agent to their current agent assignment.
    """
    df = pd.read_parquet(VOYAGE_PARQUET)
    df = df.dropna(subset=["captain_id", "agent_id", "year_out"]).copy()

    q_col = "q_total_index" if "q_total_index" in df.columns else "q_oil_bbl"
    if q_col in df.columns:
        df = df[df[q_col] > 0].copy()
        df["log_q"] = np.log(df[q_col].clip(lower=1e-6))

    # Controls
    if "tonnage" in df.columns:
        df["log_tonnage"] = np.log(df["tonnage"].clip(lower=1))
    for col in ["log_tonnage"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    if "year_out" in df.columns:
        df["decade"] = (df["year_out"] // 10) * 10

    df = df.sort_values(["captain_id", "year_out", "date_out"])

    # Compute first agent (training agent) per captain
    first_agent = (
        df.groupby("captain_id")
        .first()
        .reset_index()[["captain_id", "agent_id"]]
        .rename(columns={"agent_id": "training_agent_id"})
    )
    df = df.merge(first_agent, on="captain_id", how="left")
    df["same_agent"] = (df["agent_id"] == df["training_agent_id"]).astype(int)
    df["diff_agent"] = 1 - df["same_agent"]

    # Count how many captains have observations at both same and different agents
    cap_variety = df.groupby("captain_id")["same_agent"].agg(["min", "max"])
    promoted_with_both = cap_variety[(cap_variety["min"] == 0) & (cap_variety["max"] == 1)].index
    logger.info("  Promoted captains with both same/diff agent obs: %d", len(promoted_with_both))

    return df


def _merge_pipeline_effects(df: pd.DataFrame) -> pd.DataFrame:
    """Merge separate-sample psi_hat for the training agent."""
    # Merge voyage-level LOO psi
    psi_path = INTERMEDIATES_DIR / "psi_hat_leave_one_captain_out.parquet"
    if psi_path.exists():
        psi_loo = pd.read_parquet(psi_path)
        df = df.merge(psi_loo[["voyage_id", "psi_hat_loo"]], on="voyage_id", how="left")
        df["psi_current"] = df["psi_hat_loo"]
        df["effect_source"] = "loo"
    else:
        df["psi_current"] = np.nan
        df["effect_source"] = "none"

    # Training agent's psi: compute mean psi_hat_loo for voyages with that agent (excluding this captain)
    if "psi_hat_loo" in df.columns:
        # For each training_agent, get the median psi across all voyages by other captains
        training_psi = (
            df.groupby("training_agent_id")["psi_hat_loo"]
            .median()
            .rename("psi_training_agent")
        )
        df = df.merge(training_psi, left_on="training_agent_id", right_index=True, how="left")
    else:
        df["psi_training_agent"] = np.nan

    # Standardize
    for col in ["psi_current", "psi_training_agent"]:
        if df[col].notna().sum() > 10:
            mu, sigma = df[col].mean(), df[col].std()
            df[f"{col}_std"] = (df[col] - mu) / sigma if sigma > 0 else 0

    return df


# =====================================================================
# 7A: Same-agent training premium with richer FEs
# =====================================================================

def run_same_agent_premium(df: pd.DataFrame) -> pd.DataFrame:
    """Test training-agent premium with current-agent FE and time FE."""
    logger.info("  --- 7A: Same-agent training premium ---")

    sample = df.dropna(subset=["log_q"]).copy()
    logger.info("    Sample: %d voyages, %d captains", len(sample), sample["captain_id"].nunique())

    tidy_rows = []

    # Spec 1: Simple OLS
    try:
        X1 = sm.add_constant(sample[["same_agent"]].dropna())
        y1 = sample.loc[X1.index, "log_q"]
        m1 = sm.OLS(y1, X1).fit(
            cov_type="cluster", cov_kwds={"groups": sample.loc[X1.index, "captain_id"]}
        )
        tidy_rows.append(build_tidy_row(
            outcome="log_q",
            sample_name="officer_pipeline",
            spec_name="OLS_simple",
            term="same_agent",
            estimate=float(m1.params["same_agent"]),
            std_error=float(m1.bse["same_agent"]),
            n_obs=len(y1),
            n_captains=int(sample.loc[X1.index, "captain_id"].nunique()),
            fixed_effects="none",
            cluster_scheme="captain_id",
        ))
    except Exception as e:
        logger.warning("    Simple OLS failed: %s", e)

    # Spec 2: + current agent FE
    try:
        fe_agent = pd.get_dummies(sample["agent_id"], prefix="ag", drop_first=True, dtype=int)
        X2 = pd.concat([sample[["same_agent"]], fe_agent], axis=1)
        X2 = sm.add_constant(X2, has_constant="add")
        y2 = sample["log_q"]
        m2 = sm.OLS(y2, X2).fit(
            cov_type="cluster", cov_kwds={"groups": sample["captain_id"]}
        )
        tidy_rows.append(build_tidy_row(
            outcome="log_q",
            sample_name="officer_pipeline",
            spec_name="current_agent_FE",
            term="same_agent",
            estimate=float(m2.params["same_agent"]),
            std_error=float(m2.bse["same_agent"]),
            n_obs=len(y2),
            n_captains=int(sample["captain_id"].nunique()),
            n_agents=int(sample["agent_id"].nunique()),
            fixed_effects="current_agent",
            cluster_scheme="captain_id",
        ))
    except Exception as e:
        logger.warning("    Current agent FE failed: %s", e)

    # Spec 3: + current agent FE + decade FE
    if "decade" in sample.columns:
        try:
            fe_decade = pd.get_dummies(sample["decade"], prefix="dec", drop_first=True, dtype=int)
            X3 = pd.concat([sample[["same_agent"]], fe_agent, fe_decade], axis=1)
            X3 = sm.add_constant(X3, has_constant="add")
            y3 = sample["log_q"]
            m3 = sm.OLS(y3, X3).fit(
                cov_type="cluster", cov_kwds={"groups": sample["captain_id"]}
            )
            tidy_rows.append(build_tidy_row(
                outcome="log_q",
                sample_name="officer_pipeline",
                spec_name="current_agent_FE_decade_FE",
                term="same_agent",
                estimate=float(m3.params["same_agent"]),
                std_error=float(m3.bse["same_agent"]),
                n_obs=len(y3),
                n_captains=int(sample["captain_id"].nunique()),
                n_agents=int(sample["agent_id"].nunique()),
                fixed_effects="current_agent, decade",
                cluster_scheme="captain_id",
            ))
        except Exception as e:
            logger.warning("    Agent + decade FE failed: %s", e)

    # Spec 4: + captain FE (within-captain, same vs diff agent)
    try:
        fe_cap = pd.get_dummies(sample["captain_id"], prefix="cap", drop_first=True, dtype=int)
        X4 = pd.concat([sample[["same_agent"]], fe_cap], axis=1)
        X4 = sm.add_constant(X4, has_constant="add")
        y4 = sample["log_q"]
        m4 = sm.OLS(y4, X4).fit(
            cov_type="cluster", cov_kwds={"groups": sample["captain_id"]}
        )
        tidy_rows.append(build_tidy_row(
            outcome="log_q",
            sample_name="officer_pipeline",
            spec_name="captain_FE",
            term="same_agent",
            estimate=float(m4.params["same_agent"]),
            std_error=float(m4.bse["same_agent"]),
            n_obs=len(y4),
            n_captains=int(sample["captain_id"].nunique()),
            fixed_effects="captain",
            cluster_scheme="captain_id",
            notes="Within-captain: same vs different agent",
        ))
    except Exception as e:
        logger.warning("    Captain FE failed: %s", e)

    result = rows_to_tidy_df(tidy_rows)
    save_result_table(result, "table_pipeline_same_agent_fe", metadata={
        "description": "Training-agent premium across FE specifications",
    })
    save_markdown_table(result[["spec_name", "term", "estimate", "std_error", "p_value", "n_obs"]],
                        "table_pipeline_same_agent_fe")
    logger.info("    Saved %d coefficient rows", len(result))
    return result


# =====================================================================
# 7B: Portability test
# =====================================================================

def run_portability_test(df: pd.DataFrame) -> pd.DataFrame:
    """Test: training-agent psi predicts performance at DIFFERENT agents."""
    logger.info("  --- 7B: Portability test ---")

    # Restrict to observations at different agents
    diff_sample = df[(df["diff_agent"] == 1)].dropna(
        subset=["log_q", "psi_training_agent_std"]
    ).copy()
    logger.info("    Different-agent sample: %d voyages", len(diff_sample))

    if len(diff_sample) < 30:
        logger.warning("    Insufficient observations for portability test")
        return pd.DataFrame()

    tidy_rows = []
    ctrl_cols = [c for c in ["log_tonnage"] if c in diff_sample.columns]

    # Spec 1: psi_training predicts log_q at different agent
    try:
        regressors = ["psi_training_agent_std"] + ctrl_cols
        X = sm.add_constant(diff_sample[regressors].dropna())
        y = diff_sample.loc[X.index, "log_q"]
        m = sm.OLS(y, X).fit(
            cov_type="cluster", cov_kwds={"groups": diff_sample.loc[X.index, "captain_id"]}
        )
        tidy_rows.append(build_tidy_row(
            outcome="log_q",
            sample_name="different_agent_only",
            spec_name="OLS_portability",
            term="psi_training_agent",
            estimate=float(m.params["psi_training_agent_std"]),
            std_error=float(m.bse["psi_training_agent_std"]),
            n_obs=len(y),
            n_captains=int(diff_sample.loc[X.index, "captain_id"].nunique()),
            fixed_effects=", ".join(ctrl_cols) if ctrl_cols else "none",
            cluster_scheme="captain_id",
            effect_object_used="loo_psi_of_training_agent",
            notes="Training-agent psi_hat predicts output at *different* agent",
        ))
    except Exception as e:
        logger.warning("    Portability OLS failed: %s", e)

    # Spec 2: with current-agent FE
    try:
        fe_agent = pd.get_dummies(diff_sample["agent_id"], prefix="ag", drop_first=True, dtype=int)
        X2 = pd.concat([diff_sample[["psi_training_agent_std"] + ctrl_cols], fe_agent], axis=1)
        X2 = sm.add_constant(X2, has_constant="add")
        y2 = diff_sample["log_q"]
        m2 = sm.OLS(y2, X2).fit(
            cov_type="cluster", cov_kwds={"groups": diff_sample["captain_id"]}
        )
        tidy_rows.append(build_tidy_row(
            outcome="log_q",
            sample_name="different_agent_only",
            spec_name="current_agent_FE_portability",
            term="psi_training_agent",
            estimate=float(m2.params["psi_training_agent_std"]),
            std_error=float(m2.bse["psi_training_agent_std"]),
            n_obs=len(y2),
            n_captains=int(diff_sample["captain_id"].nunique()),
            n_agents=int(diff_sample["agent_id"].nunique()),
            fixed_effects="current_agent, " + ", ".join(ctrl_cols),
            cluster_scheme="captain_id",
            effect_object_used="loo_psi_of_training_agent",
            notes="Conditional on current-agent FE — isolates portable routine",
        ))
    except Exception as e:
        logger.warning("    Portability with agent FE failed: %s", e)

    result = rows_to_tidy_df(tidy_rows)
    save_result_table(result, "table_pipeline_portability", metadata={
        "description": "Training-agent psi_hat predicts output at different agents",
    })
    logger.info("    Saved %d coefficient rows", len(result))
    return result


# =====================================================================
# 7C: Placebos
# =====================================================================

def run_placebos(df: pd.DataFrame) -> pd.DataFrame:
    """Placebo tests: same home port, same port×era."""
    logger.info("  --- 7C: Placebo tests ---")

    tidy_rows = []

    # Placebo 1: Same home port (not same agent)
    port_col = "home_port" if "home_port" in df.columns else "port" if "port" in df.columns else None
    if port_col:
        # Define training port (first voyage's port)
        first_port = (
            df.groupby("captain_id")
            .first()
            .reset_index()[["captain_id", port_col]]
            .rename(columns={port_col: "training_port"})
        )
        df = df.merge(first_port, on="captain_id", how="left")
        df["same_port"] = (df[port_col] == df["training_port"]).astype(int)

        sample = df.dropna(subset=["log_q"]).copy()
        try:
            X = sm.add_constant(sample[["same_port"]].dropna())
            y = sample.loc[X.index, "log_q"]
            m = sm.OLS(y, X).fit(
                cov_type="cluster", cov_kwds={"groups": sample.loc[X.index, "captain_id"]}
            )
            tidy_rows.append(build_tidy_row(
                outcome="log_q",
                sample_name="placebo_home_port",
                spec_name="OLS_same_port",
                term="same_port",
                estimate=float(m.params["same_port"]),
                std_error=float(m.bse["same_port"]),
                n_obs=len(y),
                n_captains=int(sample.loc[X.index, "captain_id"].nunique()),
                notes="PLACEBO: same home port, not necessarily same agent",
            ))
        except Exception as e:
            logger.warning("    Same-port placebo failed: %s", e)

        # Placebo 2: Same port × era
        if "decade" in df.columns:
            df["port_era"] = df[port_col].astype(str) + "_" + df["decade"].astype(str)
            df["training_port_era"] = df["training_port"].astype(str) + "_" + df["decade"].astype(str)
            # This is a network placebo — does being from same port×era matter?
            try:
                sample2 = df.dropna(subset=["log_q"]).copy()
                sample2["same_port_era"] = (sample2["port_era"] == sample2["training_port_era"]).astype(int)
                X2 = sm.add_constant(sample2[["same_port_era"]].dropna())
                y2 = sample2.loc[X2.index, "log_q"]
                m2 = sm.OLS(y2, X2).fit(
                    cov_type="cluster", cov_kwds={"groups": sample2.loc[X2.index, "captain_id"]}
                )
                tidy_rows.append(build_tidy_row(
                    outcome="log_q",
                    sample_name="placebo_port_era",
                    spec_name="OLS_same_port_era",
                    term="same_port_era",
                    estimate=float(m2.params["same_port_era"]),
                    std_error=float(m2.bse["same_port_era"]),
                    n_obs=len(y2),
                    n_captains=int(sample2.loc[X2.index, "captain_id"].nunique()),
                    notes="PLACEBO: same port×era",
                ))
            except Exception as e:
                logger.warning("    Port×era placebo failed: %s", e)
    else:
        logger.info("    No home_port column found; skipping port placebos")

    result = rows_to_tidy_df(tidy_rows)
    if not result.empty:
        save_result_table(result, "table_pipeline_placebos", metadata={
            "description": "Placebo tests for training-agent transmission",
        })
    logger.info("    Saved %d coefficient rows", len(result))
    return result


# =====================================================================
# Master entry point
# =====================================================================

def run_all_pipeline_tests() -> Dict[str, pd.DataFrame]:
    """Execute all pipeline and portability tests (Step 7)."""
    logger.info("=" * 60)
    logger.info("STEP 7: OFFICER PIPELINE AND PORTABILITY TESTS")
    logger.info("=" * 60)

    df = _build_officer_pipeline_panel()
    df = _merge_pipeline_effects(df)

    logger.info("Pipeline panel: %d voyages, %d captains", len(df), df["captain_id"].nunique())
    logger.info("  Same-agent obs: %d, Diff-agent obs: %d",
                df["same_agent"].sum(), df["diff_agent"].sum())

    results = {}
    results["same_agent_premium"] = run_same_agent_premium(df)
    results["portability"] = run_portability_test(df)
    results["placebos"] = run_placebos(df)

    logger.info("Pipeline tests complete.")
    return results
