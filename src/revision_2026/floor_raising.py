"""
Floor-Raising Tests (Step 6 — highest priority).

Tests:
  6A  Zero-catch extensive margin (LPM, Logit with psi × skill interactions)
  6B  Conditional positive output (log output on positive-catch subsample)
  6C  Quantile regressions (tau = 0.10, 0.25, 0.50, 0.75, 0.90)
  6D  Variance compression / dispersion checks

All psi_hat and theta_hat objects use separate-sample construction
from Step 4 to prevent leakage.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from .config import (
    CFG, DATA_FINAL, VOYAGE_PARQUET, INTERMEDIATES_DIR,
    TABLES_DIR, FIGURES_DIR,
)
from .output_schema import build_tidy_row, rows_to_tidy_df, save_result_table, save_markdown_table

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────

def _load_floor_raising_panel() -> pd.DataFrame:
    """Load a panel that INCLUDES zero and near-zero catch voyages.

    The standard data_loader filters to q > 0. For floor-raising, we
    need the full extensive margin.
    """
    df = pd.read_parquet(VOYAGE_PARQUET)
    df = df.dropna(subset=["captain_id", "agent_id", "year_out"]).copy()

    # Output variables
    q_col = "q_total_index" if "q_total_index" in df.columns else "q_oil_bbl"
    df["q_raw"] = df[q_col].fillna(0)
    df["positive_catch"] = (df["q_raw"] > 0).astype(int)
    df["log_q_positive"] = np.where(df["q_raw"] > 0, np.log(df["q_raw"]), np.nan)

    # Near-zero indicator (bottom percentile including zeros)
    threshold = df["q_raw"].quantile(CFG.near_zero_percentile / 100)
    df["near_zero"] = (df["q_raw"] <= threshold).astype(int)

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

    return df


def _merge_separate_sample_effects(df: pd.DataFrame) -> pd.DataFrame:
    """Merge LOO psi_hat and separate-sample theta_hat onto the panel."""
    # psi_hat LOO
    psi_path = INTERMEDIATES_DIR / "psi_hat_leave_one_captain_out.parquet"
    if psi_path.exists():
        psi_loo = pd.read_parquet(psi_path)
        df = df.merge(psi_loo[["voyage_id", "psi_hat_loo"]], on="voyage_id", how="left")
        logger.info("  Merged LOO psi_hat: %.1f%% coverage",
                    100 * df["psi_hat_loo"].notna().mean())
    else:
        logger.warning("  LOO psi_hat not found; skipping merge")

    # psi_hat pre-period
    psi_pre_path = INTERMEDIATES_DIR / "psi_hat_preperiod.parquet"
    if psi_pre_path.exists():
        psi_pre = pd.read_parquet(psi_pre_path)
        df = df.merge(psi_pre[["voyage_id", "psi_hat_pre"]], on="voyage_id", how="left")

    # theta_hat separate sample
    theta_path = INTERMEDIATES_DIR / "theta_hat_separate_sample.parquet"
    if theta_path.exists():
        theta_sep = pd.read_parquet(theta_path)
        df = df.merge(
            theta_sep[["captain_id", "theta_hat_sep", "theta_hat_quartile", "low_skill"]],
            on="captain_id", how="left",
        )
        logger.info("  Merged separate theta_hat: %.1f%% coverage",
                    100 * df["theta_hat_sep"].notna().mean())

    # Determine which psi_hat to use (prefer LOO, fall back to pre-period)
    if "psi_hat_loo" in df.columns:
        df["psi"] = df["psi_hat_loo"]
        df["effect_source"] = "loo"
    elif "psi_hat_pre" in df.columns:
        df["psi"] = df["psi_hat_pre"]
        df["effect_source"] = "pre_period"
    else:
        logger.error("  No separate-sample psi_hat available!")
        df["psi"] = np.nan
        df["effect_source"] = "none"

    if "theta_hat_sep" in df.columns:
        df["theta"] = df["theta_hat_sep"]
    else:
        df["theta"] = np.nan

    # Standardize
    for col in ["psi", "theta"]:
        if df[col].notna().sum() > 10:
            mu, sigma = df[col].mean(), df[col].std()
            df[f"{col}_std"] = (df[col] - mu) / sigma if sigma > 0 else 0

    return df


# =====================================================================
# 6A: Zero-catch extensive margin
# =====================================================================

def run_zero_catch_tests(df: pd.DataFrame) -> pd.DataFrame:
    """LPM and Logit: Pr(positive_catch) = f(psi, theta, interactions)."""
    logger.info("  --- 6A: Zero-catch extensive margin ---")

    sample = df.dropna(subset=["psi_std", "theta"]).copy()
    if "theta_hat_quartile" not in sample.columns:
        sample["theta_hat_quartile"] = pd.qcut(
            sample["theta"].rank(method="first"),
            CFG.n_skill_bins,
            labels=[f"Q{i+1}" for i in range(CFG.n_skill_bins)],
        )
    if "low_skill" not in sample.columns:
        sample["low_skill"] = (
            sample["theta"] <= sample["theta"].quantile(CFG.low_skill_fraction)
        ).astype(int)

    logger.info("    N = %d, positive_catch mean = %.3f, near_zero mean = %.3f",
                len(sample), sample["positive_catch"].mean(), sample["near_zero"].mean())

    tidy_rows = []

    # --- LPM specifications ---
    for outcome_var, outcome_label in [
        ("positive_catch", "Pr(positive_catch)"),
        ("near_zero", "Pr(near_zero)"),
    ]:
        # Spec 1: baseline
        spec1_cols = ["psi_std"]
        if "theta_std" in sample.columns:
            spec1_cols.append("theta_std")
        X1 = sm.add_constant(sample[spec1_cols].dropna())
        y1 = sample.loc[X1.index, outcome_var]
        try:
            m1 = sm.OLS(y1, X1).fit(
                cov_type="cluster", cov_kwds={"groups": sample.loc[X1.index, "captain_id"]}
            )
            for term in ["psi_std", "theta_std"]:
                if term in m1.params.index:
                    tidy_rows.append(build_tidy_row(
                        outcome=outcome_label,
                        sample_name="with_zeros",
                        spec_name="LPM_baseline",
                        term=term,
                        estimate=float(m1.params[term]),
                        std_error=float(m1.bse[term]),
                        n_obs=len(y1),
                        n_captains=int(sample.loc[X1.index, "captain_id"].nunique()),
                        n_agents=int(sample.loc[X1.index, "agent_id"].nunique()),
                        fixed_effects="none",
                        cluster_scheme="captain_id",
                        effect_object_used=sample["effect_source"].iloc[0] if "effect_source" in sample else "unknown",
                    ))
        except Exception as e:
            logger.warning("    LPM baseline failed for %s: %s", outcome_var, e)

        # Spec 2: with psi × low_skill interaction
        try:
            sample["psi_x_lowskill"] = sample["psi_std"] * sample["low_skill"]
            spec2_cols = ["psi_std", "low_skill", "psi_x_lowskill"]
            if "theta_std" in sample.columns:
                spec2_cols.append("theta_std")
            X2 = sm.add_constant(sample[spec2_cols].dropna())
            y2 = sample.loc[X2.index, outcome_var]
            m2 = sm.OLS(y2, X2).fit(
                cov_type="cluster", cov_kwds={"groups": sample.loc[X2.index, "captain_id"]}
            )
            for term in ["psi_std", "low_skill", "psi_x_lowskill"]:
                if term in m2.params.index:
                    tidy_rows.append(build_tidy_row(
                        outcome=outcome_label,
                        sample_name="with_zeros",
                        spec_name="LPM_interaction",
                        term=term,
                        estimate=float(m2.params[term]),
                        std_error=float(m2.bse[term]),
                        n_obs=len(y2),
                        n_captains=int(sample.loc[X2.index, "captain_id"].nunique()),
                        n_agents=int(sample.loc[X2.index, "agent_id"].nunique()),
                        fixed_effects="none",
                        cluster_scheme="captain_id",
                        effect_object_used=sample["effect_source"].iloc[0] if "effect_source" in sample else "unknown",
                    ))
        except Exception as e:
            logger.warning("    LPM interaction failed for %s: %s", outcome_var, e)

        # Spec 3: Logit (for positive_catch only)
        if outcome_var == "positive_catch":
            try:
                f_logit = f"{outcome_var} ~ psi_std + theta_std + psi_std:low_skill"
                if "theta_std" not in sample.columns:
                    f_logit = f"{outcome_var} ~ psi_std + psi_std:low_skill"
                m_logit = smf.logit(f_logit, data=sample.dropna(subset=["psi_std"])).fit(disp=0)
                # Marginal effects at means
                mfx = m_logit.get_margeff(at="mean")
                for i, term in enumerate(mfx.summary_frame().index):
                    tidy_rows.append(build_tidy_row(
                        outcome=outcome_label,
                        sample_name="with_zeros",
                        spec_name="Logit_marginal_effects",
                        term=str(term),
                        estimate=float(mfx.margeff[i]),
                        std_error=float(mfx.margeff_se[i]),
                        n_obs=int(m_logit.nobs),
                        effect_object_used=sample["effect_source"].iloc[0] if "effect_source" in sample else "unknown",
                    ))
            except Exception as e:
                logger.warning("    Logit failed: %s", e)

    result = rows_to_tidy_df(tidy_rows)
    save_result_table(result, "table_floor_zero_catch", metadata={
        "description": "Zero-catch and near-zero extensive margin tests",
        "near_zero_percentile": CFG.near_zero_percentile,
    })
    logger.info("    Saved %d coefficient rows", len(result))
    return result


# =====================================================================
# 6B: Conditional positive output
# =====================================================================

def run_positive_output_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Regressions on log(output) conditional on positive catch."""
    logger.info("  --- 6B: Conditional positive output ---")

    sample = df[df["positive_catch"] == 1].dropna(
        subset=["log_q_positive", "psi_std"]
    ).copy()
    logger.info("    Positive-catch sample: N = %d", len(sample))

    tidy_rows = []
    ctrl_cols = [c for c in ["log_tonnage", "log_duration"] if c in sample.columns]

    # By skill quartile
    if "theta_hat_quartile" in sample.columns:
        for q in sample["theta_hat_quartile"].dropna().unique():
            q_sub = sample[sample["theta_hat_quartile"] == q].copy()
            if len(q_sub) < 30:
                continue
            regressors = ["psi_std"] + ctrl_cols
            valid_regressors = [r for r in regressors if r in q_sub.columns]
            X = sm.add_constant(q_sub[valid_regressors].dropna())
            y = q_sub.loc[X.index, "log_q_positive"]
            try:
                m = sm.OLS(y, X).fit(
                    cov_type="cluster",
                    cov_kwds={"groups": q_sub.loc[X.index, "captain_id"]},
                )
                tidy_rows.append(build_tidy_row(
                    outcome="log_q_positive",
                    sample_name=f"positive_catch_{q}",
                    spec_name="OLS_with_controls",
                    term="psi_std",
                    estimate=float(m.params["psi_std"]),
                    std_error=float(m.bse["psi_std"]),
                    n_obs=len(y),
                    n_captains=int(q_sub.loc[X.index, "captain_id"].nunique()),
                    n_agents=int(q_sub.loc[X.index, "agent_id"].nunique()),
                    fixed_effects=", ".join(ctrl_cols),
                    cluster_scheme="captain_id",
                    effect_object_used=q_sub["effect_source"].iloc[0] if "effect_source" in q_sub else "unknown",
                    notes=f"Skill quartile {q}",
                ))
            except Exception as e:
                logger.warning("    OLS by quartile failed for %s: %s", q, e)

    # Pooled with interaction
    try:
        if "low_skill" in sample.columns:
            sample["psi_x_lowskill"] = sample["psi_std"] * sample["low_skill"]
            regressors = ["psi_std", "low_skill", "psi_x_lowskill"] + ctrl_cols
            valid = [r for r in regressors if r in sample.columns]
            X = sm.add_constant(sample[valid].dropna())
            y = sample.loc[X.index, "log_q_positive"]
            m = sm.OLS(y, X).fit(
                cov_type="cluster",
                cov_kwds={"groups": sample.loc[X.index, "captain_id"]},
            )
            for term in ["psi_std", "low_skill", "psi_x_lowskill"]:
                if term in m.params.index:
                    tidy_rows.append(build_tidy_row(
                        outcome="log_q_positive",
                        sample_name="positive_catch_pooled",
                        spec_name="OLS_interaction",
                        term=term,
                        estimate=float(m.params[term]),
                        std_error=float(m.bse[term]),
                        n_obs=len(y),
                        n_captains=int(sample.loc[X.index, "captain_id"].nunique()),
                        n_agents=int(sample.loc[X.index, "agent_id"].nunique()),
                        fixed_effects=", ".join(ctrl_cols),
                        cluster_scheme="captain_id",
                        effect_object_used=sample["effect_source"].iloc[0] if "effect_source" in sample else "unknown",
                    ))
    except Exception as e:
        logger.warning("    Pooled interaction failed: %s", e)

    result = rows_to_tidy_df(tidy_rows)
    save_result_table(result, "table_floor_positive_output", metadata={
        "description": "Conditional positive output regressions by skill quartile",
    })
    logger.info("    Saved %d coefficient rows", len(result))
    return result


# =====================================================================
# 6C: Quantile regressions
# =====================================================================

def run_quantile_regressions(df: pd.DataFrame) -> pd.DataFrame:
    """Quantile regression: log_q ~ psi + theta + controls at multiple taus."""
    logger.info("  --- 6C: Quantile regressions ---")

    sample = df[df["positive_catch"] == 1].dropna(subset=["log_q_positive", "psi_std"]).copy()
    logger.info("    Sample: N = %d", len(sample))

    tidy_rows = []
    ctrl_cols = [c for c in ["log_tonnage", "log_duration"] if c in sample.columns]
    theta_col = "theta_std" if "theta_std" in sample.columns else None

    # Build formula
    rhs_terms = ["psi_std"]
    if theta_col:
        rhs_terms.append(theta_col)
    rhs_terms.extend(ctrl_cols)
    formula = "log_q_positive ~ " + " + ".join(rhs_terms)

    for tau in CFG.quantile_taus:
        try:
            model = smf.quantreg(formula, data=sample).fit(q=tau, max_iter=5000)
            for term in rhs_terms:
                if term in model.params.index:
                    tidy_rows.append(build_tidy_row(
                        outcome=f"log_q_positive_Q{int(tau*100)}",
                        sample_name="positive_catch",
                        spec_name=f"QR_tau{int(tau*100)}",
                        term=term,
                        estimate=float(model.params[term]),
                        std_error=float(model.bse[term]),
                        n_obs=int(model.nobs),
                        fixed_effects=", ".join(ctrl_cols),
                        cluster_scheme="none (QR)",
                        effect_object_used=sample["effect_source"].iloc[0] if "effect_source" in sample else "unknown",
                        notes=f"tau={tau}",
                    ))
            logger.info("    tau=%.2f: β(psi) = %.4f (SE=%.4f)",
                        tau, model.params["psi_std"], model.bse["psi_std"])
        except Exception as e:
            logger.warning("    Quantile regression failed at tau=%.2f: %s", tau, e)

    result = rows_to_tidy_df(tidy_rows)
    save_result_table(result, "table_floor_quantiles", metadata={
        "description": "Quantile regression coefficients for floor-raising evidence",
        "taus": CFG.quantile_taus,
    })

    # Also save a compact summary for the quantile-path figure
    qr_summary = result[result["term"] == "psi_std"][
        ["spec_name", "estimate", "std_error", "ci_low", "ci_high"]
    ].copy()
    qr_summary["tau"] = [tau for tau in CFG.quantile_taus if f"QR_tau{int(tau*100)}" in qr_summary["spec_name"].values][:len(qr_summary)]
    qr_summary.to_csv(FIGURES_DIR / "figure_floor_quantile_path.csv", index=False)

    logger.info("    Saved %d coefficient rows", len(result))
    return result


# =====================================================================
# 6D: Variance compression / dispersion
# =====================================================================

def run_variance_compression(df: pd.DataFrame) -> pd.DataFrame:
    """Test whether high-psi agents compress output variance for weak captains."""
    logger.info("  --- 6D: Variance compression ---")

    sample = df[df["positive_catch"] == 1].dropna(subset=["log_q_positive", "psi"]).copy()

    # Compute residual variance
    ctrl_cols = [c for c in ["log_tonnage", "log_duration"] if c in sample.columns]
    rhs = ["psi_std"] + ctrl_cols
    valid_rhs = [r for r in rhs if r in sample.columns]

    try:
        X = sm.add_constant(sample[valid_rhs].dropna())
        y = sample.loc[X.index, "log_q_positive"]
        m = sm.OLS(y, X).fit()
        sample.loc[X.index, "resid_sq"] = m.resid ** 2
    except Exception as e:
        logger.error("    First-stage OLS failed: %s", e)
        return pd.DataFrame()

    # Bin captains by psi and theta for cell-level variance comparison
    rows = []
    if "theta_hat_quartile" in sample.columns:
        psi_med = sample["psi"].median()
        sample["high_psi"] = (sample["psi"] >= psi_med).astype(int)

        for q in sample["theta_hat_quartile"].dropna().unique():
            for hp, hp_label in [(0, "Low_psi"), (1, "High_psi")]:
                cell = sample[
                    (sample["theta_hat_quartile"] == q) & (sample["high_psi"] == hp)
                ].dropna(subset=["resid_sq"])
                if len(cell) < 10:
                    continue
                rows.append({
                    "theta_quartile": str(q),
                    "psi_group": hp_label,
                    "n": len(cell),
                    "mean_log_q": float(cell["log_q_positive"].mean()),
                    "sd_log_q": float(cell["log_q_positive"].std()),
                    "mean_resid_sq": float(cell["resid_sq"].mean()),
                    "p10_log_q": float(cell["log_q_positive"].quantile(0.10)),
                    "var_ratio": None,  # filled below
                })

    cell_df = pd.DataFrame(rows)

    # Compute variance ratios (high_psi / low_psi within each quartile)
    if not cell_df.empty:
        for q in cell_df["theta_quartile"].unique():
            low = cell_df[(cell_df["theta_quartile"] == q) & (cell_df["psi_group"] == "Low_psi")]
            high = cell_df[(cell_df["theta_quartile"] == q) & (cell_df["psi_group"] == "High_psi")]
            if len(low) > 0 and len(high) > 0:
                var_low = low.iloc[0]["sd_log_q"] ** 2
                var_high = high.iloc[0]["sd_log_q"] ** 2
                ratio = var_high / var_low if var_low > 0 else np.nan
                cell_df.loc[
                    (cell_df["theta_quartile"] == q) & (cell_df["psi_group"] == "High_psi"),
                    "var_ratio",
                ] = ratio

    save_result_table(cell_df, "table_variance_compression", metadata={
        "description": "Variance compression by psi × theta cell",
        "variance_metric": "Var(log_q | positive catch)",
    })

    # Heteroskedasticity regression: resid² ~ psi + low_skill × psi
    tidy_rows = []
    if "low_skill" in sample.columns:
        try:
            sample["psi_x_lowskill"] = sample["psi_std"] * sample["low_skill"]
            het_cols = ["psi_std", "low_skill", "psi_x_lowskill"]
            valid_het = [c for c in het_cols if c in sample.columns]
            sample_het = sample.dropna(subset=["resid_sq"] + valid_het)
            X_het = sm.add_constant(sample_het[valid_het])
            y_het = sample_het["resid_sq"]
            m_het = sm.OLS(y_het, X_het).fit(cov_type="HC3")

            for term in valid_het:
                tidy_rows.append(build_tidy_row(
                    outcome="resid_sq",
                    sample_name="positive_catch",
                    spec_name="heteroskedasticity_test",
                    term=term,
                    estimate=float(m_het.params[term]),
                    std_error=float(m_het.bse[term]),
                    n_obs=len(y_het),
                    fixed_effects="none",
                    cluster_scheme="HC3",
                    effect_object_used=sample["effect_source"].iloc[0] if "effect_source" in sample else "unknown",
                    notes="Squared residuals as outcome",
                ))
        except Exception as e:
            logger.warning("    Heteroskedasticity regression failed: %s", e)

    if tidy_rows:
        het_df = rows_to_tidy_df(tidy_rows)
        save_result_table(het_df, "table_variance_compression_regression", metadata={
            "description": "Breusch-Pagan style heteroskedasticity test with interactions",
        })

    logger.info("    Variance compression analysis complete")
    return cell_df


# =====================================================================
# Master entry point
# =====================================================================

def run_all_floor_raising_tests() -> Dict[str, pd.DataFrame]:
    """Execute all floor-raising tests (Step 6)."""
    logger.info("=" * 60)
    logger.info("STEP 6: FLOOR-RAISING TESTS")
    logger.info("=" * 60)

    df = _load_floor_raising_panel()
    df = _merge_separate_sample_effects(df)

    logger.info("Floor-raising panel: %d voyages, %d positive, %d zero/near-zero",
                len(df), df["positive_catch"].sum(),
                len(df) - df["positive_catch"].sum())

    results = {}
    results["zero_catch"] = run_zero_catch_tests(df)
    results["positive_output"] = run_positive_output_tests(df)
    results["quantile_reg"] = run_quantile_regressions(df)
    results["variance_compression"] = run_variance_compression(df)

    logger.info("Floor-raising tests complete.")
    return results
