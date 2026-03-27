"""
Test 4: Behavioral Standardization / Variance Compression.

Tests whether high-ψ organizations reduce search behavior dispersion.
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from .config import CFG, COLS, TABLES_DIR, FIGURES_DIR
from .utils import absorb_fixed_effects, make_table, make_figure, save_figure, write_memo

logger = logging.getLogger(__name__)


def run_test4(df, *, outcomes=None, save_outputs=True):
    """Run Test 4: variance compression."""
    psi_col = "psi" if "psi" in df.columns else "psi_heldout"
    if outcomes is None:
        outcomes = [c for c in ["straightness_index", "turning_concentration",
                    "levy_mu", "revisit_rate", "route_efficiency"]
                    if c in df.columns and df[c].notna().sum() > 50]

    results = {}
    results["within_agent"] = _within_agent_dispersion(df, outcomes, psi_col)
    results["switch_comparison"] = _switch_variance(df, outcomes, psi_col)
    results["residual_variance"] = _residual_variance(df, outcomes, psi_col)

    if save_outputs:
        _save_outputs(results, outcomes)
    return {"results": results, "status": "complete"}


def _within_agent_dispersion(df, outcomes, psi_col):
    clean = df.dropna(subset=[psi_col])
    median_psi = clean[psi_col].median()
    clean = clean.copy()
    clean["high_psi"] = (clean[psi_col] >= median_psi).astype(int)
    records = []
    for outcome in outcomes:
        if outcome not in clean.columns:
            continue
        agent_sd = clean.groupby([COLS.agent_id, "high_psi"])[outcome].std().reset_index()
        agent_sd.columns = [COLS.agent_id, "high_psi", "within_sd"]
        agent_sd = agent_sd.dropna()
        high_sd = agent_sd[agent_sd["high_psi"] == 1]["within_sd"]
        low_sd = agent_sd[agent_sd["high_psi"] == 0]["within_sd"]
        if len(high_sd) < 5 or len(low_sd) < 5:
            continue
        from scipy.stats import levene
        stat, p_val = levene(high_sd.values, low_sd.values, center="median")
        records.append({
            "outcome": outcome,
            "high_psi_mean_sd": high_sd.mean(), "low_psi_mean_sd": low_sd.mean(),
            "compression_pct": 100*(1 - high_sd.mean()/low_sd.mean()) if low_sd.mean() > 0 else np.nan,
            "levene_stat": stat, "levene_pval": p_val,
            "n_high": len(high_sd), "n_low": len(low_sd),
        })
    return records


def _switch_variance(df, outcomes, psi_col):
    if COLS.switch_agent not in df.columns:
        return []
    df = df.sort_values([COLS.captain_id, COLS.year_out])
    records = []
    for outcome in outcomes:
        if outcome not in df.columns:
            continue
        pre_sds, post_sds = [], []
        for captain, cdf in df.groupby(COLS.captain_id):
            sw = cdf[cdf[COLS.switch_agent] == 1]
            if len(sw) == 0:
                continue
            yr = sw[COLS.year_out].min()
            pre = cdf[cdf[COLS.year_out] < yr][outcome].dropna()
            post = cdf[cdf[COLS.year_out] >= yr][outcome].dropna()
            if len(pre) >= 2 and len(post) >= 2:
                pre_sds.append(pre.std()); post_sds.append(post.std())
        if len(pre_sds) < 10:
            continue
        pre_arr, post_arr = np.array(pre_sds), np.array(post_sds)
        from scipy.stats import wilcoxon
        try:
            stat, p = wilcoxon(pre_arr, post_arr)
        except Exception:
            stat, p = np.nan, np.nan
        records.append({
            "outcome": outcome, "pre_sd": pre_arr.mean(), "post_sd": post_arr.mean(),
            "change_pct": 100*(post_arr.mean()-pre_arr.mean())/pre_arr.mean() if pre_arr.mean()>0 else np.nan,
            "wilcoxon_stat": stat, "wilcoxon_pval": p, "n_captains": len(pre_sds),
        })
    return records


def _residual_variance(df, outcomes, psi_col):
    records = []
    for outcome in outcomes:
        clean = df.dropna(subset=[outcome, psi_col, COLS.captain_id, COLS.year_out])
        if len(clean) < 100:
            continue
        y = clean[outcome].values.astype(float)
        X = clean[[psi_col]].values.astype(float)
        fe = [clean[COLS.captain_id].values, clean[COLS.year_out].values]
        res = absorb_fixed_effects(y, X, fe, return_residuals=True)
        if res["dof"] < 10:
            continue
        abs_resid = np.abs(res["residuals"])
        mask = res["_mask"]
        from scipy.stats import pearsonr
        corr, p = pearsonr(clean[psi_col].values[mask], abs_resid)
        records.append({"outcome": outcome, "corr": corr, "p_value": p, "n": len(abs_resid)})
    return records


def _save_outputs(results, outcomes):
    for key in ["within_agent", "switch_comparison", "residual_variance"]:
        if results.get(key):
            pd.DataFrame(results[key]).to_csv(TABLES_DIR / f"test4_{key}.csv", index=False)
    write_memo("test4_variance_compression",
        "## Behavioral Standardization\n\n"
        "Tests whether high-ψ orgs reduce cross-captain search behavior variance.\n"
        "Three approaches: within-agent SD, pre/post switch, residual variance.\n",
        threats=["Selection into high-ψ agents", "Pre/post confounded by experience"])
    logger.info("Test 4 outputs saved")
