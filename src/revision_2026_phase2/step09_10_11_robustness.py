"""
Steps 9-11 — Connected-set robustness, trimming sensitivity, pipeline diagnostics.
"""
from __future__ import annotations
import logging
import numpy as np, pandas as pd
import statsmodels.api as sm
from .config import CFG, INTERMEDIATES_DIR, TABLES_DIR
from .helpers import (load_positive_analysis_panel, load_broad_analysis_panel,
                      build_tidy_row, rows_to_tidy, save_table)

logger = logging.getLogger(__name__)

# =====================================================================
# Step 9: Connected-set vs broad-sample floor-raising
# =====================================================================
def run_connected_vs_broad_robustness():
    logger.info("="*60+"\nSTEP 9: CONNECTED-SET VS BROAD ROBUSTNESS\n"+"="*60)
    df_pos = load_positive_analysis_panel()
    ctrl = [c for c in ["log_tonnage","log_duration"] if c in df_pos.columns]
    tidy_rows = []

    # Merge all proxies
    for fname, col in [("psi_connected_loo.parquet","psi_hat_loo"),
                        ("psi_broad_resid_loo.parquet","psi_broad_resid")]:
        p = INTERMEDIATES_DIR / fname
        if p.exists():
            tmp = pd.read_parquet(p)
            if col in tmp.columns:
                df_pos = df_pos.merge(tmp[["voyage_id",col]], on="voyage_id", how="left")
    p_theta = INTERMEDIATES_DIR / "theta_sep_main.parquet"
    if p_theta.exists():
        theta = pd.read_parquet(p_theta)
        merge_cols = ["captain_id","theta_hat_sep"]
        for c in ["low_skill"]:
            if c in theta.columns: merge_cols.append(c)
        df_pos = df_pos.merge(theta[merge_cols].drop_duplicates("captain_id"), on="captain_id", how="left")
    # Standardize
    for col in ["psi_hat_loo","psi_broad_resid","theta_hat_sep"]:
        if col in df_pos.columns:
            mu,sig = df_pos[col].mean(), df_pos[col].std()
            df_pos[f"{col}_std"] = (df_pos[col]-mu)/sig if sig>0 else 0

    samples = [
        ("connected_psi_loo", df_pos.dropna(subset=["psi_hat_loo_std","log_q"]), "psi_hat_loo_std", "psi_connected_loo"),
        ("broad_psi_resid", df_pos.dropna(subset=["psi_broad_resid_std","log_q"]), "psi_broad_resid_std", "psi_broad_resid_loo"),
    ]
    for sample_name, sample, psi_col, obj_label in samples:
        if len(sample) < 50: continue
        # Main effect
        rhs = [psi_col] + ctrl
        valid = [r for r in rhs if r in sample.columns]
        try:
            X = sm.add_constant(sample[valid])
            y = sample["log_q"]
            m = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups":sample["captain_id"]})
            tidy_rows.append(build_tidy_row(outcome="log_q", sample_name=sample_name,
                spec_name="OLS_main", term="psi", estimate=float(m.params[psi_col]),
                std_error=float(m.bse[psi_col]), n_obs=len(y),
                n_captains=int(sample["captain_id"].nunique()),
                effect_object_used=obj_label, cluster_scheme="captain_id"))
        except Exception as e:
            logger.warning("  %s main OLS failed: %s", sample_name, e)
        # Low-skill interaction
        if "low_skill" in sample.columns:
            try:
                s2 = sample.dropna(subset=["low_skill"]).copy()
                s2["psi_x_low"] = s2[psi_col]*s2["low_skill"]
                rhs2 = [psi_col,"low_skill","psi_x_low"]+ctrl
                valid2 = [r for r in rhs2 if r in s2.columns]
                X2 = sm.add_constant(s2[valid2])
                y2 = s2["log_q"]
                m2 = sm.OLS(y2, X2).fit(cov_type="cluster", cov_kwds={"groups":s2["captain_id"]})
                for term in [psi_col,"low_skill","psi_x_low"]:
                    if term in m2.params.index:
                        tidy_rows.append(build_tidy_row(outcome="log_q", sample_name=sample_name,
                            spec_name="OLS_interaction", term=term,
                            estimate=float(m2.params[term]), std_error=float(m2.bse[term]),
                            n_obs=len(y2), n_captains=int(s2["captain_id"].nunique()),
                            effect_object_used=obj_label))
            except Exception as e:
                logger.warning("  %s interaction failed: %s", sample_name, e)

    result = rows_to_tidy(tidy_rows)
    save_table(result, "table_connected_vs_broad_floorraising")
    logger.info("  Connected vs broad: %d rows", len(result))
    return result

# =====================================================================
# Step 10: Trimming sensitivity
# =====================================================================
def run_trimming_sensitivity():
    logger.info("="*60+"\nSTEP 10: TRIMMING SENSITIVITY\n"+"="*60)
    df = load_positive_analysis_panel()
    for fname, col in [("psi_connected_loo.parquet","psi_hat_loo")]:
        p = INTERMEDIATES_DIR / fname
        if p.exists():
            tmp = pd.read_parquet(p)
            if col in tmp.columns:
                df = df.merge(tmp[["voyage_id",col]], on="voyage_id", how="left")
    if "psi_hat_loo" not in df.columns:
        logger.warning("  No psi_hat_loo; skipping trimming sensitivity")
        return pd.DataFrame()
    df = df.dropna(subset=["log_q","psi_hat_loo"]).copy()
    mu,sig = df["psi_hat_loo"].mean(), df["psi_hat_loo"].std()
    df["psi_std"] = (df["psi_hat_loo"]-mu)/sig if sig>0 else 0
    ctrl = [c for c in ["log_tonnage","log_duration"] if c in df.columns]

    tidy_rows = []
    for trim_pct in CFG.trim_pcts:
        if trim_pct > 0:
            lo = df["log_q"].quantile(trim_pct/100)
            hi = df["log_q"].quantile(1-trim_pct/100)
            s = df[(df["log_q"]>=lo)&(df["log_q"]<=hi)].copy()
            label = f"trim_{trim_pct}pct"
        else:
            s = df.copy()
            label = "no_trim"
        if len(s) < 50: continue
        try:
            rhs = ["psi_std"]+ctrl
            X = sm.add_constant(s[rhs])
            y = s["log_q"]
            m = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups":s["captain_id"]})
            tidy_rows.append(build_tidy_row(outcome="log_q", sample_name="positive_connected",
                spec_name=f"OLS_{label}", term="psi", estimate=float(m.params["psi_std"]),
                std_error=float(m.bse["psi_std"]), n_obs=len(y),
                n_captains=int(s["captain_id"].nunique()),
                effect_object_used="psi_connected_loo", trim_rule=label))
        except Exception as e:
            logger.warning("  Trim %s failed: %s", label, e)

    # Winsorization variants
    for win_pct in [0.5, 1.0]:
        s = df.copy()
        lo = s["log_q"].quantile(win_pct/100)
        hi = s["log_q"].quantile(1-win_pct/100)
        s["log_q"] = s["log_q"].clip(lo, hi)
        label = f"winsor_{win_pct}pct"
        try:
            rhs = ["psi_std"]+ctrl
            X = sm.add_constant(s[rhs])
            y = s["log_q"]
            m = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups":s["captain_id"]})
            tidy_rows.append(build_tidy_row(outcome="log_q", sample_name="positive_connected",
                spec_name=f"OLS_{label}", term="psi", estimate=float(m.params["psi_std"]),
                std_error=float(m.bse["psi_std"]), n_obs=len(y),
                n_captains=int(s["captain_id"].nunique()),
                effect_object_used="psi_connected_loo", trim_rule=label))
        except Exception as e:
            logger.warning("  Winsor %s failed: %s", label, e)

    # Extreme psi support restriction
    for cutoff in [3.0, 2.5]:
        s = df[df["psi_std"].abs()<=cutoff].copy()
        label = f"psi_support_{cutoff}sd"
        try:
            rhs = ["psi_std"]+ctrl
            X = sm.add_constant(s[rhs])
            y = s["log_q"]
            m = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups":s["captain_id"]})
            tidy_rows.append(build_tidy_row(outcome="log_q", sample_name="positive_connected",
                spec_name=f"OLS_{label}", term="psi", estimate=float(m.params["psi_std"]),
                std_error=float(m.bse["psi_std"]), n_obs=len(y),
                n_captains=int(s["captain_id"].nunique()),
                effect_object_used="psi_connected_loo", trim_rule=label))
        except Exception as e:
            logger.warning("  Support %s failed: %s", label, e)

    result = rows_to_tidy(tidy_rows)
    save_table(result, "table_trimming_sensitivity")
    save_table(result, "table_support_sensitivity")
    logger.info("  Trimming sensitivity: %d rows", len(result))
    return result

# =====================================================================
# Step 11: Pipeline diagnostics (appendix only)
# =====================================================================
def run_pipeline_diagnostics():
    logger.info("="*60+"\nSTEP 11: PIPELINE DIAGNOSTICS (APPENDIX)\n"+"="*60)
    df = load_positive_analysis_panel()
    df = df.sort_values(["captain_id","year_out","date_out"]).copy()
    first_agent = df.groupby("captain_id").first().reset_index()[["captain_id","agent_id"]].rename(
        columns={"agent_id":"training_agent_id"})
    df = df.merge(first_agent, on="captain_id", how="left")
    df["same_agent"] = (df["agent_id"]==df["training_agent_id"]).astype(int)
    ctrl = [c for c in ["log_tonnage","log_duration"] if c in df.columns]

    tidy_rows = []

    # Merge psi for training agent
    psi_path = INTERMEDIATES_DIR / "psi_broad_resid_loo.parquet"
    if psi_path.exists():
        psi = pd.read_parquet(psi_path)
        if "psi_broad_resid" in psi.columns:
            training_psi = psi.groupby("agent_id")["psi_broad_resid"].median().rename("psi_training")
            df = df.merge(training_psi, left_on="training_agent_id", right_index=True, how="left")
            mu,sig = df["psi_training"].mean(), df["psi_training"].std()
            df["psi_training_std"] = (df["psi_training"]-mu)/sig if sig>0 else 0

    # Same-agent regressions
    specs_sa = [
        ("no_FE", [], df),
    ]
    # Current-agent FE
    fe_ag = pd.get_dummies(df["agent_id"], prefix="a", drop_first=True, dtype=int)
    specs_sa.append(("current_agent_FE", [fe_ag], df))
    if "decade" in df.columns:
        fe_dec = pd.get_dummies(df["decade"], prefix="d", drop_first=True, dtype=int)
        specs_sa.append(("agent_FE+decade_FE", [fe_ag, fe_dec], df))
    # First post-promotion voyage
    df["voyage_num"] = df.groupby("captain_id").cumcount()
    first_voy = df[df["voyage_num"]==0].copy()
    if len(first_voy)>100:
        specs_sa.append(("first_voyage_only", [], first_voy))

    for spec_name, fe_list, sample in specs_sa:
        try:
            rhs = ["same_agent"]+ctrl
            parts = [sample[rhs]]
            parts.extend(fe_list)
            X = pd.concat(parts, axis=1)
            X = sm.add_constant(X, has_constant="add")
            y = sample["log_q"]
            m = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups":sample["captain_id"]})
            if "same_agent" in m.params.index:
                tidy_rows.append(build_tidy_row(outcome="log_q", sample_name="pipeline",
                    spec_name=spec_name, term="same_agent",
                    estimate=float(m.params["same_agent"]), std_error=float(m.bse["same_agent"]),
                    n_obs=int(m.nobs), n_captains=int(sample["captain_id"].nunique()),
                    cluster_scheme="captain_id"))
        except Exception as e:
            logger.warning("  Pipeline spec %s failed: %s", spec_name, e)

    # Selection diagnostics: Pr(stay with training agent)
    sel_rows = []
    theta_path = INTERMEDIATES_DIR / "theta_sep_main.parquet"
    if theta_path.exists() and "psi_training_std" in df.columns:
        theta = pd.read_parquet(theta_path)
        df = df.merge(theta[["captain_id","theta_hat_sep"]].drop_duplicates("captain_id"),
                       on="captain_id", how="left")
        mu_t, sig_t = df["theta_hat_sep"].mean(), df["theta_hat_sep"].std()
        df["theta_std"] = (df["theta_hat_sep"]-mu_t)/sig_t if sig_t>0 else 0
        # On first voyage: does captain skill predict staying?
        fv = df[df["voyage_num"]==0].dropna(subset=["same_agent","theta_std","psi_training_std"]).copy()
        if len(fv)>100:
            try:
                X_sel = sm.add_constant(fv[["theta_std","psi_training_std"]])
                y_sel = fv["same_agent"]
                m_sel = sm.OLS(y_sel, X_sel).fit(cov_type="cluster", cov_kwds={"groups":fv["captain_id"]})
                for t in ["theta_std","psi_training_std"]:
                    sel_rows.append(build_tidy_row(outcome="Pr(stay_with_training_agent)",
                        sample_name="first_voyage", spec_name="selection_LPM", term=t,
                        estimate=float(m_sel.params[t]), std_error=float(m_sel.bse[t]),
                        n_obs=int(m_sel.nobs), n_captains=int(fv["captain_id"].nunique())))
            except Exception as e:
                logger.warning("  Selection model failed: %s", e)

    # Sorting: does training-agent quality predict current-agent quality?
    if "psi_training_std" in df.columns:
        cur_agent_psi = psi.groupby("agent_id")["psi_broad_resid"].median().rename("psi_current_agent") \
            if psi_path.exists() else pd.Series(dtype=float)
        if len(cur_agent_psi)>0:
            df = df.merge(cur_agent_psi, left_on="agent_id", right_index=True, how="left")
            diff = df[df["same_agent"]==0].dropna(subset=["psi_training_std","psi_current_agent"]).copy()
            if len(diff)>100:
                try:
                    X_s = sm.add_constant(diff[["psi_training_std"]])
                    y_s = diff["psi_current_agent"]
                    m_s = sm.OLS(y_s, X_s).fit(cov_type="cluster", cov_kwds={"groups":diff["captain_id"]})
                    sel_rows.append(build_tidy_row(outcome="psi_current_agent",
                        sample_name="different_agent_voyages", spec_name="sorting_regression",
                        term="psi_training", estimate=float(m_s.params["psi_training_std"]),
                        std_error=float(m_s.bse["psi_training_std"]),
                        n_obs=int(m_s.nobs), n_captains=int(diff["captain_id"].nunique()),
                        notes="Does training-agent quality predict subsequent agent quality?"))
                except Exception as e:
                    logger.warning("  Sorting regression failed: %s", e)

    result = rows_to_tidy(tidy_rows)
    save_table(result, "table_pipeline_null_recheck")
    if sel_rows:
        save_table(rows_to_tidy(sel_rows), "table_pipeline_selection_diagnostics")

    logger.info("  Pipeline diagnostics: %d spec rows, %d selection rows", len(tidy_rows), len(sel_rows))
    return result
