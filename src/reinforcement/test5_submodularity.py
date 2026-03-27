"""
Test 5: Direct Submodularity Test & Matching Counterfactual.

Tests whether captain skill and organizational capability are
substitutes (submodular) under scarcity:
    ∂²y / ∂θ∂ψ < 0 when whales are scarce

Key specification:
    y = f(θ) + g(ψ) + λ·S + γ·θ·ψ + κ·(θ·ψ·S) + Γ·X + ε

γ < 0 → submodularity (substitutes)
κ < 0 → scarcity amplifies substitutability
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from .config import CFG, COLS, TABLES_DIR, FIGURES_DIR
from .type_measures import compute_scarcity_index
from .utils import absorb_fixed_effects, cluster_se, make_table, make_figure, save_figure, write_memo

logger = logging.getLogger(__name__)


def run_test5(df, *, use_heldout=True, save_outputs=True):
    """Run Test 5: submodularity + matching counterfactual."""
    theta_col = "theta_heldout" if (use_heldout and "theta_heldout" in df.columns) else "theta"
    psi_col = "psi_heldout" if (use_heldout and "psi_heldout" in df.columns) else "psi"

    df = df.dropna(subset=[theta_col, psi_col, COLS.log_q]).copy()
    if "scarcity_index" not in df.columns:
        df = compute_scarcity_index(df)

    logger.info("Test 5: n=%d, theta=%s, psi=%s", len(df), theta_col, psi_col)

    results = []
    # Spec 1: Linear interaction
    res = _interaction_regression(df, theta_col, psi_col, spec="linear")
    if res: results.append(res)

    # Spec 2: Triple interaction θ × ψ × Scarcity
    res = _interaction_regression(df, theta_col, psi_col, spec="triple")
    if res: results.append(res)

    # Spec 3: By scarcity bin
    bin_results = _by_scarcity_bin(df, theta_col, psi_col)
    results.extend(bin_results)

    # Matching counterfactual
    cf = _matching_counterfactual(df, theta_col, psi_col)

    if save_outputs:
        _save_outputs(results, cf, df, theta_col, psi_col)

    return {"results": results, "counterfactual": cf, "status": "complete"}


def _interaction_regression(df, theta_col, psi_col, spec="linear"):
    """Estimate interaction specification."""
    df = df.copy()
    df["theta_x_psi"] = df[theta_col] * df[psi_col]

    if spec == "triple" and "scarcity_index" in df.columns:
        df["theta_x_psi_x_S"] = df["theta_x_psi"] * df["scarcity_index"]
        X_cols = [theta_col, psi_col, "scarcity_index", "theta_x_psi", "theta_x_psi_x_S"]
    else:
        X_cols = [theta_col, psi_col, "theta_x_psi"]

    clean = df.dropna(subset=X_cols + [COLS.log_q])
    if len(clean) < 100:
        return None

    y = clean[COLS.log_q].values
    X = clean[X_cols].values.astype(float)
    fe = [clean[COLS.year_out].values]
    if COLS.ground_or_route in clean.columns:
        fe.append(clean[COLS.ground_or_route].fillna("UNK").values)

    res = absorb_fixed_effects(y, X, fe, return_residuals=True)
    if res["dof"] < 10:
        return None

    mask = res["_mask"]
    se = cluster_se(X[mask], res["residuals"], clean[COLS.captain_id].values[mask])

    coefs = dict(zip(X_cols, res["coefficients"]))
    ses = dict(zip(X_cols, se))
    from scipy.stats import t as t_dist
    pvals = {}
    for i, col in enumerate(X_cols):
        t = res["coefficients"][i] / se[i] if se[i] > 0 else 0
        pvals[col] = 2 * (1 - t_dist.cdf(abs(t), df=res["dof"]))

    return {
        "name": f"Submodularity ({spec})",
        "coefficients": coefs, "se": ses, "pvalues": pvals,
        "n_obs": res["n_obs"], "r_squared": res["r_squared"],
        "n_clusters": len(np.unique(clean[COLS.captain_id].values[mask])),
        "fe_structure": "year + ground", "cluster_var": COLS.captain_id,
    }


def _by_scarcity_bin(df, theta_col, psi_col):
    """Estimate θ×ψ interaction separately by scarcity bin."""
    if "scarcity_bin" not in df.columns:
        return []
    results = []
    for sbin, sdf in df.groupby("scarcity_bin"):
        sdf = sdf.copy()
        sdf["theta_x_psi"] = sdf[theta_col] * sdf[psi_col]
        X_cols = [theta_col, psi_col, "theta_x_psi"]
        clean = sdf.dropna(subset=X_cols + [COLS.log_q])
        if len(clean) < 50:
            continue

        y = clean[COLS.log_q].values
        X = clean[X_cols].values.astype(float)
        fe = [clean[COLS.year_out].values]
        res = absorb_fixed_effects(y, X, fe, return_residuals=True)
        if res["dof"] < 5:
            continue

        mask = res["_mask"]
        se = cluster_se(X[mask], res["residuals"], clean[COLS.captain_id].values[mask])
        coefs = dict(zip(X_cols, res["coefficients"]))
        ses = dict(zip(X_cols, se))

        results.append({
            "name": f"Scarcity bin={sbin}",
            "coefficients": coefs, "se": ses,
            "pvalues": {k: np.nan for k in X_cols},
            "n_obs": res["n_obs"], "r_squared": res["r_squared"],
            "n_clusters": len(np.unique(clean[COLS.captain_id].values[mask])),
            "fe_structure": "year", "scarcity_bin": str(sbin),
        })
    return results


def _matching_counterfactual(df, theta_col, psi_col):
    """Simple matching counterfactual: PAM vs NAM vs observed."""
    clean = df.dropna(subset=[theta_col, psi_col, COLS.log_q])
    if len(clean) < 100:
        return {}

    # Status quo output
    observed_q = clean[COLS.log_q].sum()

    # Estimate production function coefficients
    clean = clean.copy()
    clean["theta_x_psi"] = clean[theta_col] * clean[psi_col]
    X = clean[[theta_col, psi_col, "theta_x_psi"]].values
    y = clean[COLS.log_q].values
    from numpy.linalg import lstsq
    beta, _, _, _ = lstsq(np.column_stack([X, np.ones(len(X))]), y, rcond=None)

    # Counterfactual assignments
    thetas = clean[theta_col].values.copy()
    psis = clean[psi_col].values.copy()
    n = len(thetas)

    # PAM: sort both ascending, match by rank
    theta_sorted = np.sort(thetas)
    psi_sorted = np.sort(psis)
    pam_q = sum(beta[0]*t + beta[1]*p + beta[2]*t*p + beta[3]
                for t, p in zip(theta_sorted, psi_sorted))

    # NAM: sort theta ascending, psi descending
    nam_q = sum(beta[0]*t + beta[1]*p + beta[2]*t*p + beta[3]
                for t, p in zip(theta_sorted, psi_sorted[::-1]))

    return {
        "observed_total_q": float(observed_q),
        "pam_total_q": float(pam_q),
        "nam_total_q": float(nam_q),
        "pam_gain_pct": 100 * (pam_q - observed_q) / abs(observed_q),
        "nam_gain_pct": 100 * (nam_q - observed_q) / abs(observed_q),
        "efficient_matching": "NAM" if nam_q > pam_q else "PAM",
        "gamma_hat": float(beta[2]),
        "n_obs": n,
    }


def _save_outputs(results, cf, df, theta_col, psi_col):
    if results:
        make_table(results, "log(Q)", "test5_submodularity")

    if cf:
        pd.DataFrame([cf]).to_csv(TABLES_DIR / "test5_counterfactual.csv", index=False)

    # Marginal return plot
    try:
        fig, ax = make_figure("test5", "marginal_returns", figsize=(8, 6))
        clean = df.dropna(subset=[theta_col, psi_col, COLS.log_q])
        clean = clean.copy()
        theta_bins = pd.qcut(clean[theta_col], 4, labels=["Q1","Q2","Q3","Q4"], duplicates="drop")
        for tbin in theta_bins.unique():
            mask = theta_bins == tbin
            sub = clean[mask].sort_values(psi_col)
            from scipy.stats import binned_statistic
            psi_vals = sub[psi_col].values
            q_vals = sub[COLS.log_q].values
            try:
                means, edges, _ = binned_statistic(psi_vals, q_vals, bins=8)
                centers = 0.5*(edges[:-1]+edges[1:])
                ax.plot(centers, means, marker="o", label=f"θ {tbin}", alpha=0.8)
            except Exception:
                pass
        ax.set_xlabel("Agent Capability (ψ)")
        ax.set_ylabel("Mean log(Q)")
        ax.set_title("Marginal Return to ψ by Captain Skill Quartile")
        ax.legend()
        save_figure(fig, "test5", "marginal_returns")
    except Exception as e:
        logger.warning("Failed marginal return plot: %s", e)

    write_memo("test5_submodularity",
        "## Submodularity & Matching Counterfactual\n\n"
        f"- γ̂ (θ×ψ interaction): {cf.get('gamma_hat', 'N/A'):.4f}\n"
        f"- Efficient matching: {cf.get('efficient_matching', 'N/A')}\n"
        f"- NAM gain: {cf.get('nam_gain_pct', 0):.2f}%\n"
        f"- PAM gain: {cf.get('pam_gain_pct', 0):.2f}%\n",
        threats=["AKM bias may inflate interaction", "Scarcity proxy is coarse",
                 "Counterfactual ignores GE / capacity constraints"])
    logger.info("Test 5 outputs saved")
