"""
Steps 2 & 3 — Broad-sample capability and skill proxies.

Constructs organization proxies that cover the full with-zeros panel,
not just the connected set.
"""
from __future__ import annotations
import logging, json
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np, pandas as pd
import statsmodels.api as sm
from .config import CFG, INTERMEDIATES_DIR, TABLES_DIR, P1_INTERMEDIATES
from .helpers import (load_broad_analysis_panel, load_positive_analysis_panel,
                      save_table, save_md)

logger = logging.getLogger(__name__)

# =====================================================================
# Step 2: organization-capability proxies
# =====================================================================

def _build_psi_connected_loo():
    """Reuse Phase 1 exact LOO psi_hat on the connected set."""
    path = P1_INTERMEDIATES / "psi_hat_leave_one_captain_out.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        df.to_parquet(INTERMEDIATES_DIR / "psi_connected_loo.parquet", index=False)
        return df
    logger.warning("Phase 1 psi_hat_loo not found; skipping psi_connected_loo")
    return pd.DataFrame()

def _build_psi_broad_resid_loo():
    """Broad-sample LOO agent capability via residualized mean output.

    For each voyage v of captain c at agent a:
      psi_broad = mean(resid_j) for all voyages j at agent a by captains != c
    where resid = log_q − controls (decade FE + log_tonnage + log_duration).
    """
    logger.info("  Building psi_broad_resid_loo...")
    df = load_positive_analysis_panel()
    # Residualize
    ctrl = [c for c in ["log_tonnage","log_duration"] if c in df.columns]
    if "decade" in df.columns:
        dec_dummies = pd.get_dummies(df["decade"], prefix="d", drop_first=True, dtype=float)
        X = pd.concat([df[ctrl], dec_dummies], axis=1) if ctrl else dec_dummies
    else:
        X = df[ctrl] if ctrl else None
    if X is not None and len(X) > 100:
        Xc = sm.add_constant(X)
        m = sm.OLS(df["log_q"], Xc).fit()
        df["resid"] = m.resid
    else:
        df["resid"] = df["log_q"] - df["log_q"].mean()

    # LOO agent mean residual
    rows = []
    agent_groups = df.groupby("agent_id")
    for agent_id, grp in agent_groups:
        captains_at_agent = grp["captain_id"].unique()
        for cap in captains_at_agent:
            others = grp[grp["captain_id"] != cap]["resid"]
            if len(others) >= 1:
                val = float(others.mean())
            else:
                val = np.nan
            # Assign to all voyages of this captain at this agent
            cap_voys = grp[grp["captain_id"] == cap]["voyage_id"].values
            for vid in cap_voys:
                rows.append(dict(voyage_id=vid, captain_id=cap, agent_id=agent_id, psi_broad_resid=val))
    result = pd.DataFrame(rows)
    # Standardize
    mu, sigma = result["psi_broad_resid"].mean(), result["psi_broad_resid"].std()
    result["psi_broad_resid_std"] = (result["psi_broad_resid"] - mu) / sigma if sigma > 0 else 0
    result.to_parquet(INTERMEDIATES_DIR / "psi_broad_resid_loo.parquet", index=False)
    logger.info("    Coverage: %d voyages (%.1f%% of positive panel)",
                result["psi_broad_resid"].notna().sum(),
                100 * result["psi_broad_resid"].notna().mean())
    return result

def _build_psi_broad_preperiod():
    """Broad-sample pre-period agent mean residual output."""
    logger.info("  Building psi_broad_preperiod...")
    df = load_positive_analysis_panel()
    ctrl = [c for c in ["log_tonnage","log_duration"] if c in df.columns]
    if "decade" in df.columns:
        dec_dummies = pd.get_dummies(df["decade"], prefix="d", drop_first=True, dtype=float)
        X = pd.concat([df[ctrl], dec_dummies], axis=1) if ctrl else dec_dummies
    else:
        X = df[ctrl] if ctrl else None
    if X is not None:
        m = sm.OLS(df["log_q"], sm.add_constant(X)).fit()
        df["resid"] = m.resid
    else:
        df["resid"] = df["log_q"] - df["log_q"].mean()
    # Captain's first year
    first_yr = df.groupby("captain_id")["year_out"].min().rename("first_year")
    df = df.merge(first_yr, on="captain_id", how="left")
    # For each voyage, pre-period agent mean
    rows = []
    for vid, cap, agent, yr, fy, resid_val in zip(
        df["voyage_id"], df["captain_id"], df["agent_id"], df["year_out"], df["first_year"], df["resid"]
    ):
        pre = df[(df["agent_id"]==agent) & (df["year_out"]<fy) & (df["captain_id"]!=cap)]
        val = float(pre["resid"].mean()) if len(pre) >= 1 else np.nan
        rows.append(dict(voyage_id=vid, captain_id=cap, agent_id=agent, psi_broad_pre=val))
    result = pd.DataFrame(rows)
    mu, sigma = result["psi_broad_pre"].mean(), result["psi_broad_pre"].std()
    result["psi_broad_pre_std"] = (result["psi_broad_pre"] - mu) / sigma if sigma > 0 else 0
    result.to_parquet(INTERMEDIATES_DIR / "psi_broad_preperiod.parquet", index=False)
    logger.info("    Coverage: %.1f%%", 100*result["psi_broad_pre"].notna().mean())
    return result

def _build_psi_broad_success_loo():
    """LOO agent positive-catch rate from other captains (broad panel)."""
    logger.info("  Building psi_broad_success_loo...")
    df = load_broad_analysis_panel()
    rows = []
    for agent_id, grp in df.groupby("agent_id"):
        for cap in grp["captain_id"].unique():
            others = grp[grp["captain_id"] != cap]["positive_catch"]
            val = float(others.mean()) if len(others) >= 1 else np.nan
            for vid in grp[grp["captain_id"]==cap]["voyage_id"].values:
                rows.append(dict(voyage_id=vid, captain_id=cap, agent_id=agent_id,
                                 psi_broad_success=val))
    result = pd.DataFrame(rows)
    mu, sigma = result["psi_broad_success"].mean(), result["psi_broad_success"].std()
    result["psi_broad_success_std"] = (result["psi_broad_success"] - mu) / sigma if sigma > 0 else 0
    result.to_parquet(INTERMEDIATES_DIR / "psi_broad_success_loo.parquet", index=False)
    logger.info("    Coverage: %.1f%%", 100*result["psi_broad_success"].notna().mean())
    return result

# =====================================================================
# Step 3: captain-skill proxies
# =====================================================================

def _build_theta_sep_main():
    """Reuse Phase 1 separate-sample theta_hat."""
    path = P1_INTERMEDIATES / "theta_hat_separate_sample.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        df.to_parquet(INTERMEDIATES_DIR / "theta_sep_main.parquet", index=False)
        return df
    return pd.DataFrame()

def _build_skill_experience_proxy():
    """Broad-coverage experience proxy: log(1 + prior_voyages) at time of voyage."""
    logger.info("  Building skill_experience_proxy...")
    df = load_broad_analysis_panel().sort_values(["captain_id","year_out","date_out"])
    df["voyage_seq"] = df.groupby("captain_id").cumcount()
    df["log_experience"] = np.log1p(df["voyage_seq"])
    # Also: years since first command
    first_yr = df.groupby("captain_id")["year_out"].transform("min")
    df["years_since_first"] = df["year_out"] - first_yr
    # Skill quartiles based on experience
    df["exp_quartile"] = pd.qcut(df["log_experience"].rank(method="first"),
                                  CFG.n_skill_bins,
                                  labels=[f"Q{i+1}" for i in range(CFG.n_skill_bins)])
    df["low_experience"] = (df["voyage_seq"] == 0).astype(int)  # first voyage = novice
    out = df[["voyage_id","captain_id","voyage_seq","log_experience",
              "years_since_first","exp_quartile","low_experience"]].copy()
    out.to_parquet(INTERMEDIATES_DIR / "skill_experience_proxy.parquet", index=False)
    logger.info("    Coverage: %d voyages (100%%)", len(out))
    return out

# =====================================================================
# Proxy comparison table
# =====================================================================

def build_all_proxies():
    logger.info("="*60+"\nSTEPS 2-3: BUILDING BROAD-SAMPLE PROXIES\n"+"="*60)

    psi_conn = _build_psi_connected_loo()
    psi_resid = _build_psi_broad_resid_loo()
    psi_pre = _build_psi_broad_preperiod()
    psi_success = _build_psi_broad_success_loo()
    theta_main = _build_theta_sep_main()
    skill_exp = _build_skill_experience_proxy()

    # Coverage and correlation table
    broad = load_broad_analysis_panel()
    n_broad = len(broad)
    rows = []
    proxy_dfs = [
        ("psi_connected_loo", psi_conn, "psi_hat_loo", "connected_set"),
        ("psi_broad_resid_loo", psi_resid, "psi_broad_resid", "positive_output"),
        ("psi_broad_preperiod", psi_pre, "psi_broad_pre", "positive_output"),
        ("psi_broad_success_loo", psi_success, "psi_broad_success", "broad_with_zeros"),
    ]
    # Merge all onto broad panel for correlations
    merged = broad[["voyage_id"]].copy()
    for label, pdf, col, sample in proxy_dfs:
        if col in pdf.columns:
            merged = merged.merge(pdf[["voyage_id",col]], on="voyage_id", how="left")
            n_valid = merged[col].notna().sum()
            rows.append(dict(proxy=label, value_col=col, sample_basis=sample,
                             n_valid=int(n_valid),
                             coverage_of_broad_pct=round(100*n_valid/n_broad,1),
                             mean=float(merged[col].dropna().mean()),
                             sd=float(merged[col].dropna().std())))
    # Pairwise correlations with psi_connected_loo
    ref_col = "psi_hat_loo"
    for label, pdf, col, sample in proxy_dfs:
        if col == ref_col or col not in merged.columns: continue
        valid = merged.dropna(subset=[ref_col, col])
        if len(valid) > 30:
            r = float(valid[ref_col].corr(valid[col]))
            for row in rows:
                if row["proxy"] == label:
                    row["corr_with_psi_connected_loo"] = round(r, 3)

    # Skill proxies
    if "theta_hat_sep" in theta_main.columns:
        merged = merged.merge(theta_main[["captain_id","theta_hat_sep"]].drop_duplicates("captain_id"),
                              left_on=broad["captain_id"].values, right_on="captain_id", how="left")
        n_valid_theta = merged["theta_hat_sep"].notna().sum()
        rows.append(dict(proxy="theta_sep_main", value_col="theta_hat_sep",
                         sample_basis="connected_set",
                         n_valid=int(n_valid_theta),
                         coverage_of_broad_pct=round(100*n_valid_theta/n_broad,1),
                         mean=float(merged["theta_hat_sep"].dropna().mean()),
                         sd=float(merged["theta_hat_sep"].dropna().std())))

    rows.append(dict(proxy="skill_experience_proxy", value_col="log_experience",
                     sample_basis="broad_with_zeros", n_valid=n_broad,
                     coverage_of_broad_pct=100.0,
                     mean=float(skill_exp["log_experience"].mean()),
                     sd=float(skill_exp["log_experience"].std())))

    coverage_df = pd.DataFrame(rows)
    save_table(coverage_df, "table_proxy_coverage_and_correlations")
    save_md(coverage_df, "table_proxy_coverage_and_correlations")
    logger.info("  Proxy table saved (%d proxies)", len(rows))
    return coverage_df
