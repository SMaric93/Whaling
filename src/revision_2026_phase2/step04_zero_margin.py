"""
Step 4 — Zero-catch and near-zero extensive-margin tests on the broad sample.

Runs on the with-zeros panel using broad-coverage proxies, NOT the connected set.
"""
from __future__ import annotations
import logging
import numpy as np, pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from .config import CFG, INTERMEDIATES_DIR, TABLES_DIR, FIGURES_DIR
from .helpers import load_broad_analysis_panel, build_tidy_row, rows_to_tidy, save_table, save_md
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def _merge_broad_proxies(df):
    for fname, col in [("psi_broad_resid_loo.parquet","psi_broad_resid"),
                        ("psi_broad_preperiod.parquet","psi_broad_pre"),
                        ("psi_broad_success_loo.parquet","psi_broad_success"),
                        ("psi_connected_loo.parquet","psi_hat_loo")]:
        p = INTERMEDIATES_DIR / fname
        if p.exists():
            tmp = pd.read_parquet(p)
            if col in tmp.columns:
                df = df.merge(tmp[["voyage_id",col]], on="voyage_id", how="left")
    for fname, col in [("skill_experience_proxy.parquet","log_experience"),
                        ("theta_sep_main.parquet","theta_hat_sep")]:
        p = INTERMEDIATES_DIR / fname
        if p.exists():
            tmp = pd.read_parquet(p)
            if col in tmp.columns:
                if "voyage_id" in tmp.columns:
                    merge_cols = ["voyage_id", col]
                    extra = [c for c in ["exp_quartile","low_experience","voyage_seq"] if c in tmp.columns]
                    merge_cols.extend(extra)
                    df = df.merge(tmp[merge_cols], on="voyage_id", how="left")
                elif "captain_id" in tmp.columns:
                    merge_cols = ["captain_id", col]
                    extra = [c for c in ["theta_hat_quartile","low_skill"] if c in tmp.columns]
                    merge_cols.extend(extra)
                    df = df.merge(tmp[merge_cols].drop_duplicates("captain_id"),
                                  on="captain_id", how="left")
    # Standardize all proxies
    for col in ["psi_broad_resid","psi_broad_pre","psi_broad_success","psi_hat_loo"]:
        if col in df.columns:
            mu, sig = df[col].mean(), df[col].std()
            df[f"{col}_std"] = (df[col]-mu)/sig if sig>0 else 0
    return df

def run_zero_margin_tests():
    logger.info("="*60+"\nSTEP 4: ZERO-CATCH EXTENSIVE-MARGIN (BROAD SAMPLE)\n"+"="*60)
    df = load_broad_analysis_panel()
    df = _merge_broad_proxies(df)
    ctrl = [c for c in ["log_tonnage","log_duration"] if c in df.columns]
    logger.info("  Broad panel: %d voyages, zero_catch=%.1f%%, near_zero=%.1f%%",
                len(df), 100*df["zero_catch"].mean(), 100*df["near_zero"].mean())

    tidy_rows = []
    proxy_list = [
        ("psi_broad_resid","psi_broad_resid_std","psi_broad_resid_loo"),
        ("psi_broad_pre","psi_broad_pre_std","psi_broad_preperiod"),
        ("psi_broad_success","psi_broad_success_std","psi_broad_success_loo"),
        ("psi_hat_loo","psi_hat_loo_std","psi_connected_loo"),
    ]
    # Skill definitions
    skill_defs = []
    if "theta_hat_sep" in df.columns:
        skill_defs.append(("theta_sep", "theta_hat_sep", "low_skill", "theta_hat_quartile"))
    if "log_experience" in df.columns:
        skill_defs.append(("experience", "log_experience", "low_experience", "exp_quartile"))

    for outcome_var, outcome_label in [("zero_catch","Pr(zero_catch)"),("near_zero","Pr(near_zero)")]:
        for psi_raw, psi_std, psi_label in proxy_list:
            if psi_std not in df.columns: continue
            sample = df.dropna(subset=[psi_std]).copy()
            if sample[outcome_var].nunique() < 2:
                logger.info("    %s on %s: degenerate (only %d unique values)", outcome_label, psi_label,
                            sample[outcome_var].nunique())
                tidy_rows.append(build_tidy_row(outcome=outcome_label, sample_name="broad_with_zeros",
                    spec_name="DEGENERATE", term=psi_label, estimate=0, std_error=0, n_obs=len(sample),
                    n_captains=int(sample["captain_id"].nunique()),
                    effect_object_used=psi_label, notes="outcome is constant in this sample"))
                continue
            # Spec 1: LPM baseline
            rhs = [psi_std] + ctrl
            valid_rhs = [r for r in rhs if r in sample.columns]
            X = sm.add_constant(sample[valid_rhs].dropna())
            y = sample.loc[X.index, outcome_var]
            try:
                m = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": sample.loc[X.index,"captain_id"]})
                tidy_rows.append(build_tidy_row(outcome=outcome_label, sample_name="broad_with_zeros",
                    spec_name="LPM_baseline", term=psi_label, estimate=float(m.params[psi_std]),
                    std_error=float(m.bse[psi_std]), n_obs=len(y),
                    n_captains=int(sample.loc[X.index,"captain_id"].nunique()),
                    n_agents=int(sample.loc[X.index,"agent_id"].nunique()),
                    fixed_effects=", ".join(ctrl), cluster_scheme="captain_id",
                    effect_object_used=psi_label))
            except Exception as e:
                logger.warning("    LPM baseline failed (%s, %s): %s", psi_label, outcome_label, e)

            # Spec 2-3: interactions with each skill def
            for skill_name, skill_col, low_col, q_col in skill_defs:
                if low_col not in sample.columns: continue
                try:
                    s2 = sample.dropna(subset=[psi_std, low_col]).copy()
                    s2["psi_x_low"] = s2[psi_std] * s2[low_col]
                    rhs2 = [psi_std, low_col, "psi_x_low"] + ctrl
                    valid2 = [r for r in rhs2 if r in s2.columns]
                    X2 = sm.add_constant(s2[valid2].dropna())
                    y2 = s2.loc[X2.index, outcome_var]
                    m2 = sm.OLS(y2, X2).fit(cov_type="cluster", cov_kwds={"groups":s2.loc[X2.index,"captain_id"]})
                    for term in [psi_std, low_col, "psi_x_low"]:
                        if term in m2.params.index:
                            tidy_rows.append(build_tidy_row(
                                outcome=outcome_label, sample_name="broad_with_zeros",
                                spec_name=f"LPM_interact_{skill_name}", term=f"{psi_label}:{term}",
                                estimate=float(m2.params[term]), std_error=float(m2.bse[term]),
                                n_obs=len(y2), n_captains=int(s2.loc[X2.index,"captain_id"].nunique()),
                                effect_object_used=psi_label, cluster_scheme="captain_id"))
                except Exception as e:
                    logger.warning("    LPM interact failed (%s, %s, %s): %s", psi_label, skill_name, outcome_label, e)

            # Spec 5: captain-FE within-captain switching (only for movers)
            try:
                movers = sample.groupby("captain_id")["agent_id"].nunique()
                mover_ids = movers[movers>=2].index
                s_mov = sample[sample["captain_id"].isin(mover_ids)].copy()
                if len(s_mov)>100 and s_mov[outcome_var].nunique()>=2:
                    fe_cap = pd.get_dummies(s_mov["captain_id"], prefix="c", drop_first=True, dtype=int)
                    X_mov = pd.concat([s_mov[[psi_std]], fe_cap], axis=1)
                    X_mov = sm.add_constant(X_mov, has_constant="add")
                    y_mov = s_mov[outcome_var]
                    m_mov = sm.OLS(y_mov, X_mov).fit(cov_type="cluster",
                                                      cov_kwds={"groups":s_mov["captain_id"]})
                    tidy_rows.append(build_tidy_row(
                        outcome=outcome_label, sample_name="broad_movers_only",
                        spec_name="LPM_captain_FE", term=psi_label,
                        estimate=float(m_mov.params[psi_std]), std_error=float(m_mov.bse[psi_std]),
                        n_obs=len(y_mov), n_captains=len(mover_ids),
                        fixed_effects="captain", cluster_scheme="captain_id",
                        effect_object_used=psi_label))
            except Exception as e:
                logger.warning("    Captain-FE LPM failed (%s, %s): %s", psi_label, outcome_label, e)

    result = rows_to_tidy(tidy_rows)
    save_table(result, "table_zero_margin_broad")
    save_md(result[["outcome","spec_name","term","estimate","std_error","p_value","n_obs","effect_object_used"]],
            "table_zero_margin_broad")

    # Proxy comparison table
    proxy_comp = result[result["spec_name"]=="LPM_baseline"][
        ["outcome","term","estimate","std_error","p_value","n_obs","effect_object_used"]
    ].copy()
    save_table(proxy_comp, "table_zero_margin_proxy_comparison")

    # Figure: zero margin by skill (using best proxy)
    _plot_zero_margin_by_skill(df, tidy_rows)

    logger.info("  Zero-margin tests: %d rows saved", len(result))
    return result

def _plot_zero_margin_by_skill(df, tidy_rows):
    try:
        best_psi = "psi_broad_resid_std" if "psi_broad_resid_std" in df.columns else "psi_broad_success_std"
        if best_psi not in df.columns: return
        skill_col = "exp_quartile" if "exp_quartile" in df.columns else "theta_hat_quartile"
        if skill_col not in df.columns: return
        s = df.dropna(subset=[best_psi, skill_col, "zero_catch"]).copy()
        med = s[best_psi].median()
        s["high_psi"] = (s[best_psi] >= med).astype(int)
        tab = s.groupby([skill_col,"high_psi"])["zero_catch"].mean().reset_index()
        tab_pivot = tab.pivot(index=skill_col, columns="high_psi", values="zero_catch")
        fig, ax = plt.subplots(figsize=(7,5))
        x = range(len(tab_pivot))
        ax.bar([i-0.15 for i in x], tab_pivot[0], 0.3, label="Low ψ", color="#e74c3c", alpha=0.8)
        ax.bar([i+0.15 for i in x], tab_pivot[1], 0.3, label="High ψ", color="#2ecc71", alpha=0.8)
        ax.set_xticks(list(x)); ax.set_xticklabels(tab_pivot.index)
        ax.set_ylabel("Pr(zero catch)"); ax.set_xlabel("Skill quartile")
        ax.set_title("Zero-Catch Rate by Organization Capability × Skill")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR/"figure_zero_margin_by_skill.png", dpi=CFG.figure_dpi)
        plt.close(fig)
        tab.to_csv(FIGURES_DIR/"figure_zero_margin_by_skill.csv", index=False)
    except Exception as e:
        logger.warning("  Figure failed: %s", e)
