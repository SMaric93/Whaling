"""
Steps 5-7 — Clustered bootstrap quantile inference, location vs scale
decomposition, and continuous skill heterogeneity.
"""
from __future__ import annotations
import logging
import numpy as np, pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats as sp_stats
from .config import CFG, INTERMEDIATES_DIR, TABLES_DIR, FIGURES_DIR
from .helpers import (load_positive_analysis_panel, build_tidy_row, rows_to_tidy,
                      save_table, save_md)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def _load_analysis_with_effects():
    df = load_positive_analysis_panel()
    # Merge connected-set psi
    p = INTERMEDIATES_DIR / "psi_connected_loo.parquet"
    if p.exists():
        psi = pd.read_parquet(p)
        df = df.merge(psi[["voyage_id","psi_hat_loo"]], on="voyage_id", how="left")
    # Merge theta
    p2 = INTERMEDIATES_DIR / "theta_sep_main.parquet"
    if p2.exists():
        theta = pd.read_parquet(p2)
        merge_cols = ["captain_id","theta_hat_sep"]
        for c in ["theta_hat_quartile","low_skill"]:
            if c in theta.columns: merge_cols.append(c)
        df = df.merge(theta[merge_cols].drop_duplicates("captain_id"), on="captain_id", how="left")
    # Standardize
    for col in ["psi_hat_loo","theta_hat_sep"]:
        if col in df.columns:
            mu, sig = df[col].mean(), df[col].std()
            df[f"{col}_std"] = (df[col]-mu)/sig if sig>0 else 0
    return df

# =====================================================================
# Step 5: Captain-cluster bootstrap quantile regressions
# =====================================================================
def run_bootstrap_quantiles():
    logger.info("="*60+"\nSTEP 5: BOOTSTRAP QUANTILE INFERENCE\n"+"="*60)
    df = _load_analysis_with_effects()
    sample = df.dropna(subset=["log_q","psi_hat_loo_std"]).copy()
    ctrl = [c for c in ["log_tonnage","log_duration"] if c in sample.columns]
    theta_col = "theta_hat_sep_std" if "theta_hat_sep_std" in sample.columns else None
    rhs = ["psi_hat_loo_std"]
    if theta_col: rhs.append(theta_col)
    rhs.extend(ctrl)
    formula = "log_q ~ " + " + ".join(rhs)
    logger.info("  Sample: %d, formula: %s", len(sample), formula)

    # Point estimates
    tidy_rows = []
    point_estimates = {}
    for tau in CFG.quantile_taus:
        try:
            m = smf.quantreg(formula, data=sample).fit(q=tau, max_iter=5000)
            for term in rhs:
                if term in m.params.index:
                    point_estimates[(tau, term)] = float(m.params[term])
                    tidy_rows.append(build_tidy_row(
                        outcome=f"log_q_Q{int(tau*100)}", sample_name="positive_connected",
                        spec_name=f"QR_point_tau{int(tau*100)}", term=term,
                        estimate=float(m.params[term]), std_error=float(m.bse[term]),
                        n_obs=int(m.nobs), effect_object_used="psi_connected_loo",
                        notes=f"point estimate, analytical SE"))
        except Exception as e:
            logger.warning("  QR tau=%.2f failed: %s", tau, e)

    # Captain-cluster bootstrap
    captains = sample["captain_id"].unique()
    n_cap = len(captains)
    np.random.seed(CFG.random_seed)
    boot_results = {(tau, term): [] for tau in CFG.quantile_taus for term in rhs}
    logger.info("  Running %d bootstrap replications (resampling %d captains)...", CFG.n_bootstrap, n_cap)

    for b in range(CFG.n_bootstrap):
        boot_caps = np.random.choice(captains, size=n_cap, replace=True)
        boot_df = pd.concat([sample[sample["captain_id"]==c] for c in boot_caps], ignore_index=True)
        for tau in CFG.quantile_taus:
            try:
                m_b = smf.quantreg(formula, data=boot_df).fit(q=tau, max_iter=3000)
                for term in rhs:
                    if term in m_b.params.index:
                        boot_results[(tau, term)].append(float(m_b.params[term]))
            except:
                pass
        if (b+1) % 100 == 0:
            logger.info("    Bootstrap: %d / %d", b+1, CFG.n_bootstrap)

    # Bootstrap SEs and CIs
    boot_rows = []
    for tau in CFG.quantile_taus:
        for term in rhs:
            draws = np.array(boot_results[(tau, term)])
            if len(draws) < 50: continue
            pe = point_estimates.get((tau, term), np.nan)
            se_boot = float(np.std(draws, ddof=1))
            ci_lo = float(np.percentile(draws, 2.5))
            ci_hi = float(np.percentile(draws, 97.5))
            p_boot = float(2 * min(np.mean(draws > 0), np.mean(draws < 0)))
            tidy_rows.append(build_tidy_row(
                outcome=f"log_q_Q{int(tau*100)}", sample_name="positive_connected",
                spec_name=f"QR_bootstrap_tau{int(tau*100)}", term=term,
                estimate=pe, std_error=se_boot, n_obs=len(sample),
                effect_object_used="psi_connected_loo",
                notes=f"captain-cluster bootstrap B={len(draws)}, percentile CI"))
            # Override CI
            tidy_rows[-1]["ci_low"] = ci_lo
            tidy_rows[-1]["ci_high"] = ci_hi
            tidy_rows[-1]["p_value"] = p_boot
            boot_rows.append(dict(tau=tau, term=term, estimate=pe, se_boot=se_boot,
                                   ci_low=ci_lo, ci_high=ci_hi, p_boot=p_boot, n_draws=len(draws)))

    # Bootstrap contrasts
    contrast_rows = []
    for t1, t2, label in [(0.10,0.50,"P10_vs_P50"),(0.10,0.90,"P10_vs_P90")]:
        d1 = np.array(boot_results.get((t1,"psi_hat_loo_std"),[]))
        d2 = np.array(boot_results.get((t2,"psi_hat_loo_std"),[]))
        n = min(len(d1), len(d2))
        if n >= 50:
            diff = d1[:n] - d2[:n]
            contrast_rows.append(dict(contrast=label, mean_diff=float(np.mean(diff)),
                                       se_diff=float(np.std(diff,ddof=1)),
                                       ci_low=float(np.percentile(diff,2.5)),
                                       ci_high=float(np.percentile(diff,97.5)),
                                       p_value=float(2*min(np.mean(diff>0),np.mean(diff<0))),
                                       n_draws=n))

    result = rows_to_tidy(tidy_rows)
    save_table(result, "table_quantile_clustered_primary")
    save_table(pd.DataFrame(contrast_rows) if contrast_rows else pd.DataFrame(), "table_quantile_bootstrap_contrasts")

    # Save bootstrap draws
    boot_draw_df = pd.DataFrame({f"tau{int(t*100)}_{term}": boot_results[(t,term)]
                                  for t in CFG.quantile_taus for term in rhs
                                  if len(boot_results[(t,term)])>0})
    boot_draw_df.to_parquet(INTERMEDIATES_DIR / "quantile_bootstrap_draws.parquet", index=False)

    # Figure: quantile path with bootstrap CIs
    _plot_quantile_path(boot_rows)
    logger.info("  Bootstrap quantiles complete: %d rows", len(result))
    return result

def _plot_quantile_path(boot_rows):
    try:
        psi_rows = [r for r in boot_rows if r["term"]=="psi_hat_loo_std"]
        if not psi_rows: return
        taus = [r["tau"] for r in psi_rows]
        ests = [r["estimate"] for r in psi_rows]
        ci_lo = [r["ci_low"] for r in psi_rows]
        ci_hi = [r["ci_high"] for r in psi_rows]
        fig, ax = plt.subplots(figsize=(7,5))
        ax.fill_between(taus, ci_lo, ci_hi, alpha=0.2, color="#3498db")
        ax.plot(taus, ests, "o-", color="#2c3e50", linewidth=2, markersize=8)
        ax.axhline(0, color="grey", linestyle="--", alpha=0.5)
        ax.set_xlabel("Quantile (τ)"); ax.set_ylabel("β̂(ψ̂)")
        ax.set_title("Organization Effect Across Output Distribution\n(Captain-Cluster Bootstrap 95% CI)")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR/"figure_quantile_path_clustered.png", dpi=CFG.figure_dpi)
        plt.close(fig)
        pd.DataFrame(dict(tau=taus, estimate=ests, ci_low=ci_lo, ci_high=ci_hi)).to_csv(
            FIGURES_DIR/"figure_quantile_path_clustered.csv", index=False)
    except Exception as e:
        logger.warning("  Quantile path figure failed: %s", e)

# =====================================================================
# Step 6: Location vs scale decomposition
# =====================================================================
def run_location_vs_scale():
    logger.info("="*60+"\nSTEP 6: LOCATION VS SCALE DECOMPOSITION\n"+"="*60)
    df = _load_analysis_with_effects()
    sample = df.dropna(subset=["log_q","psi_hat_loo","theta_hat_sep"]).copy()
    if "theta_hat_quartile" not in sample.columns:
        sample["theta_hat_quartile"] = pd.qcut(sample["theta_hat_sep"].rank(method="first"),
                                                 CFG.n_skill_bins, labels=[f"Q{i+1}" for i in range(CFG.n_skill_bins)])
    if "low_skill" not in sample.columns:
        sample["low_skill"] = (sample["theta_hat_sep"]<=sample["theta_hat_sep"].quantile(CFG.low_skill_fraction)).astype(int)
    psi_med = sample["psi_hat_loo"].median()
    sample["high_psi"] = (sample["psi_hat_loo"]>=psi_med).astype(int)
    ctrl = [c for c in ["log_tonnage","log_duration"] if c in sample.columns]

    # Residualize for dispersion
    X = sm.add_constant(sample[["psi_hat_loo_std"]+ctrl].dropna())
    y = sample.loc[X.index, "log_q"]
    m = sm.OLS(y, X).fit()
    sample.loc[X.index, "resid"] = m.resid
    sample["resid_sq"] = sample["resid"]**2

    # Cell-level statistics
    cell_rows = []
    quantiles_to_report = [0.10, 0.25, 0.50, 0.75, 0.90]
    for q_label in sample["theta_hat_quartile"].dropna().unique():
        for hp in [0, 1]:
            cell = sample[(sample["theta_hat_quartile"]==q_label)&(sample["high_psi"]==hp)]
            if len(cell) < 10: continue
            lq = cell["log_q"].dropna()
            row = dict(theta_quartile=str(q_label), psi_group="High" if hp else "Low",
                       n=len(cell), mean_log_q=float(lq.mean()), sd_log_q=float(lq.std()),
                       var_log_q=float(lq.var()), iqr_log_q=float(lq.quantile(0.75)-lq.quantile(0.25)))
            for p in quantiles_to_report:
                row[f"p{int(p*100)}_log_q"] = float(lq.quantile(p))
            # Below fixed threshold (p25 of full sample)
            thresh = sample["log_q"].quantile(0.25)
            row["pr_below_p25"] = float((lq < thresh).mean())
            cell_rows.append(row)
    cell_df = pd.DataFrame(cell_rows)

    # Compute variance ratios per quartile
    for q_label in cell_df["theta_quartile"].unique():
        low_var = cell_df.loc[(cell_df["theta_quartile"]==q_label)&(cell_df["psi_group"]=="Low"), "var_log_q"]
        high_var = cell_df.loc[(cell_df["theta_quartile"]==q_label)&(cell_df["psi_group"]=="High"), "var_log_q"]
        if len(low_var)>0 and len(high_var)>0 and float(low_var.iloc[0])>0:
            cell_df.loc[(cell_df["theta_quartile"]==q_label)&(cell_df["psi_group"]=="High"),
                        "var_ratio"] = float(high_var.iloc[0]) / float(low_var.iloc[0])
    save_table(cell_df, "table_location_vs_scale_by_skill")

    # Bootstrap variance ratios
    np.random.seed(CFG.random_seed)
    captains = sample["captain_id"].unique()
    vr_boot = {q: [] for q in sample["theta_hat_quartile"].dropna().unique()}
    for b in range(min(CFG.n_bootstrap, 499)):
        boot_caps = np.random.choice(captains, size=len(captains), replace=True)
        boot_df = pd.concat([sample[sample["captain_id"]==c] for c in boot_caps], ignore_index=True)
        for q_label in vr_boot:
            lo = boot_df[(boot_df["theta_hat_quartile"]==q_label)&(boot_df["high_psi"]==0)]["log_q"]
            hi = boot_df[(boot_df["theta_hat_quartile"]==q_label)&(boot_df["high_psi"]==1)]["log_q"]
            if len(lo)>5 and len(hi)>5 and lo.var()>0:
                vr_boot[q_label].append(hi.var()/lo.var())

    vr_rows = []
    for q_label in sorted(vr_boot.keys()):
        draws = np.array(vr_boot[q_label])
        if len(draws)<50: continue
        vr_rows.append(dict(theta_quartile=str(q_label), var_ratio_mean=float(np.mean(draws)),
                            ci_low=float(np.percentile(draws,2.5)), ci_high=float(np.percentile(draws,97.5)),
                            p_less_than_1=float(np.mean(draws<1)), n_draws=len(draws)))
    save_table(pd.DataFrame(vr_rows), "table_variance_ratio_bootstrap")

    # Scale regression
    tidy_rows = []
    het_sample = sample.dropna(subset=["resid_sq","psi_hat_loo_std","low_skill"]).copy()
    het_sample["psi_x_low"] = het_sample["psi_hat_loo_std"] * het_sample["low_skill"]
    for outcome_col, outcome_label in [("resid_sq","resid_sq"),
                                         ("log_q","log_resid_sq_proxy")]:
        if outcome_col == "log_q": continue  # Only use resid_sq
        try:
            rhs_het = ["psi_hat_loo_std","low_skill","psi_x_low"]
            X_het = sm.add_constant(het_sample[rhs_het])
            y_het = het_sample[outcome_col]
            m_het = sm.OLS(y_het, X_het).fit(cov_type="HC3")
            for term in rhs_het:
                tidy_rows.append(build_tidy_row(outcome=outcome_label, sample_name="positive_connected",
                    spec_name="scale_regression", term=term, estimate=float(m_het.params[term]),
                    std_error=float(m_het.bse[term]), n_obs=len(y_het),
                    fixed_effects="none", cluster_scheme="HC3",
                    effect_object_used="psi_connected_loo"))
        except Exception as e:
            logger.warning("  Scale regression failed: %s", e)
    if tidy_rows:
        save_table(rows_to_tidy(tidy_rows), "table_scale_regression")

    # Figure
    _plot_location_vs_scale(cell_df)
    logger.info("  Location vs scale complete")
    return cell_df

def _plot_location_vs_scale(cell_df):
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        qs = sorted(cell_df["theta_quartile"].unique())
        x = range(len(qs))
        for ax, metric, title in zip(axes,
                                      ["mean_log_q","sd_log_q","pr_below_p25"],
                                      ["Mean log(output)","SD log(output)","Pr(below P25)"]):
            for hp, color, label in [(0,"#e74c3c","Low ψ"),(1,"#2ecc71","High ψ")]:
                vals = [float(cell_df.loc[(cell_df["theta_quartile"]==q)&
                                          (cell_df["psi_group"]==("High" if hp else "Low")), metric].iloc[0])
                        for q in qs if len(cell_df.loc[(cell_df["theta_quartile"]==q)&
                                                        (cell_df["psi_group"]==("High" if hp else "Low"))])>0]
                offset = -0.15 if hp==0 else 0.15
                ax.bar([i+offset for i in range(len(vals))], vals, 0.3, label=label, color=color, alpha=0.8)
            ax.set_xticks(list(x)); ax.set_xticklabels(qs)
            ax.set_title(title); ax.set_xlabel("Captain Skill Quartile")
            ax.legend()
        fig.suptitle("Location vs Scale Decomposition by Skill × Organization", fontsize=13)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR/"figure_location_vs_scale_by_skill.png", dpi=CFG.figure_dpi)
        plt.close(fig)
        cell_df.to_csv(FIGURES_DIR/"figure_location_vs_scale_by_skill.csv", index=False)
    except Exception as e:
        logger.warning("  Location/scale figure failed: %s", e)

# =====================================================================
# Step 7: Continuous skill heterogeneity
# =====================================================================
def run_skill_heterogeneity():
    logger.info("="*60+"\nSTEP 7: CONTINUOUS SKILL HETEROGENEITY\n"+"="*60)
    df = _load_analysis_with_effects()
    sample = df.dropna(subset=["log_q","psi_hat_loo_std","theta_hat_sep_std"]).copy()
    ctrl = [c for c in ["log_tonnage","log_duration"] if c in sample.columns]
    logger.info("  Sample: %d", len(sample))

    # Spline interaction: psi * B-spline(theta)
    from patsy import dmatrix
    tidy_rows = []
    try:
        # Create B-spline basis for theta
        theta_bs = dmatrix("bs(theta_hat_sep_std, df=4, degree=3) - 1", data=sample, return_type="dataframe")
        theta_bs.columns = [f"theta_bs{i}" for i in range(theta_bs.shape[1])]
        # Interactions
        for col in theta_bs.columns:
            sample[f"psi_x_{col}"] = sample["psi_hat_loo_std"] * theta_bs[col]
        interaction_cols = [f"psi_x_{col}" for col in theta_bs.columns]
        rhs = ["psi_hat_loo_std","theta_hat_sep_std"] + list(theta_bs.columns) + interaction_cols + ctrl
        X = sm.add_constant(sample[rhs])
        y = sample["log_q"]
        m = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": sample["captain_id"]})
        # Joint test that interaction terms = 0
        try:
            f_test = m.f_test(" = ".join([f"{c} = 0" for c in interaction_cols]))
            tidy_rows.append(build_tidy_row(outcome="log_q", sample_name="positive_connected",
                spec_name="spline_interaction_joint_test", term="psi_x_theta_spline_joint",
                estimate=float(f_test.fvalue[0][0]), std_error=0, n_obs=int(m.nobs),
                notes=f"F={float(f_test.fvalue[0][0]):.2f}, p={float(f_test.pvalue):.4f}"))
        except: pass
    except Exception as e:
        logger.warning("  Spline interaction failed: %s", e)

    # Binned percentile plot (20 bins)
    n_bins = 20
    sample["theta_bin"] = pd.qcut(sample["theta_hat_sep"].rank(method="first"), n_bins, labels=False)
    bin_rows = []
    for b in range(n_bins):
        sub = sample[sample["theta_bin"]==b].copy()
        if len(sub) < 30: continue
        try:
            rhs_b = ["psi_hat_loo_std"] + ctrl
            X_b = sm.add_constant(sub[rhs_b])
            y_b = sub["log_q"]
            m_b = sm.OLS(y_b, X_b).fit(cov_type="cluster", cov_kwds={"groups": sub["captain_id"]})
            bin_rows.append(dict(theta_percentile=int((b+0.5)*100/n_bins),
                                  theta_bin=b, beta_psi=float(m_b.params["psi_hat_loo_std"]),
                                  se_psi=float(m_b.bse["psi_hat_loo_std"]),
                                  n=len(sub), mean_theta=float(sub["theta_hat_sep_std"].mean())))
        except: pass
    bin_df = pd.DataFrame(bin_rows) if bin_rows else pd.DataFrame()
    if not bin_df.empty:
        save_table(bin_df, "table_skill_spline_interactions")
        # Figure
        try:
            fig, ax = plt.subplots(figsize=(8,5))
            ax.errorbar(bin_df["theta_percentile"], bin_df["beta_psi"],
                        yerr=1.96*bin_df["se_psi"], fmt="o-", color="#2c3e50",
                        capsize=3, linewidth=1.5, markersize=6)
            ax.axhline(0, color="grey", linestyle="--", alpha=0.5)
            ax.set_xlabel("Captain Skill Percentile (θ̂)")
            ax.set_ylabel("β̂(ψ̂) on log output")
            ax.set_title("Marginal Return to Organization by Captain Skill\n(20-bin local regression)")
            fig.tight_layout()
            fig.savefig(FIGURES_DIR/"figure_marginal_return_by_skill_percentile.png", dpi=CFG.figure_dpi)
            plt.close(fig)
            bin_df.to_csv(FIGURES_DIR/"figure_marginal_return_by_skill_percentile.csv", index=False)
        except Exception as e:
            logger.warning("  Skill heterogeneity figure failed: %s", e)
    logger.info("  Skill heterogeneity: %d bins", len(bin_df))
    return bin_df
