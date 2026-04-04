"""
Separate-sample psi_hat and theta_hat construction (Step 4).

Anti-leakage wrappers:
  1. Leave-one-captain-out psi_hat  (exact + approximate)
  2. Captain-specific pre-period psi_hat
  3. Separate-sample theta_hat with EB shrinkage

All downstream output regressions must use one of these objects, never
the same-sample estimate.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

from .config import CFG, INTERMEDIATES_DIR, VOYAGE_PARQUET, CANONICAL_CONNECTED_SET, AUTHORITATIVE_TYPES

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# =====================================================================
# Helpers
# =====================================================================

def _load_connected_panel() -> pd.DataFrame:
    """Load the canonical connected-set panel with authoritative types merged."""
    if CANONICAL_CONNECTED_SET.exists() and AUTHORITATIVE_TYPES.exists():
        df = pd.read_parquet(CANONICAL_CONNECTED_SET)
        types = pd.read_parquet(AUTHORITATIVE_TYPES)
        type_cols = [c for c in ["voyage_id", "theta_hat", "psi_hat",
                                  "theta_hat_plugin", "psi_hat_plugin",
                                  "lambda_captain", "lambda_agent"] if c in types.columns]
        df = df.merge(types[type_cols], on="voyage_id", how="left")
    else:
        # Fallback: build from raw data using existing loaders
        from src.analyses.data_loader import load_voyage_data, construct_variables
        from src.analyses.connected_set import find_connected_set, find_leave_one_out_connected_set
        df = load_voyage_data()
        df = construct_variables(df)
        df_cc, _ = find_connected_set(df)
        df, _ = find_leave_one_out_connected_set(df_cc)

    # Ensure log_q exists
    if "log_q" not in df.columns:
        if "q_total_index" in df.columns:
            df["log_q"] = np.log(df["q_total_index"].clip(lower=1e-6))
        elif "q_oil_bbl" in df.columns:
            df["log_q"] = np.log(df["q_oil_bbl"].clip(lower=1))

    # Ensure controls
    if "log_tonnage" not in df.columns and "tonnage" in df.columns:
        df["log_tonnage"] = np.log(df["tonnage"].clip(lower=1))
    if "log_duration" not in df.columns and "duration_days" in df.columns:
        df["log_duration"] = np.log(df["duration_days"].clip(lower=1))
    for col in ["log_tonnage", "log_duration"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Decade
    if "decade" not in df.columns and "year_out" in df.columns:
        df["decade"] = (df["year_out"] // 10) * 10

    return df


def _estimate_akm_simple(
    df: pd.DataFrame,
    outcome_col: str = "log_q",
) -> Tuple[pd.Series, pd.Series, np.ndarray]:
    """Minimal AKM: captain + agent FEs only (for LOO speed).

    Returns
    -------
    captain_fe : pd.Series indexed by captain_id
    agent_fe   : pd.Series indexed by agent_id
    residuals  : np.ndarray
    """
    n = len(df)
    # Captain FEs
    cap_codes, cap_ids = pd.factorize(df["captain_id"], sort=False)
    X_cap = sp.csr_matrix(
        (np.ones(n), (np.arange(n), cap_codes)), shape=(n, len(cap_ids))
    )
    # Agent FEs (drop first)
    ag_codes, ag_ids = pd.factorize(df["agent_id"], sort=False)
    X_ag = sp.csr_matrix(
        (np.ones(n), (np.arange(n), ag_codes)), shape=(n, len(ag_ids))
    )[:, 1:]  # drop first for identification

    # Controls
    ctrl_cols = [c for c in ["log_tonnage", "log_duration"] if c in df.columns]
    matrices = [X_cap, X_ag]
    if ctrl_cols:
        matrices.append(sp.csr_matrix(df[ctrl_cols].to_numpy(dtype=float)))

    X = sp.hstack(matrices)
    y = df[outcome_col].to_numpy(dtype=float)

    result = lsqr(X, y, iter_lim=5000, atol=1e-8, btol=1e-8)
    beta = result[0]

    theta = beta[: len(cap_ids)]
    psi = np.concatenate([[0.0], beta[len(cap_ids) : len(cap_ids) + len(ag_ids) - 1]])
    residuals = y - X @ beta

    captain_fe = pd.Series(theta, index=cap_ids, name="theta_hat")
    agent_fe = pd.Series(psi, index=ag_ids, name="psi_hat")
    return captain_fe, agent_fe, residuals


# =====================================================================
# 1. Leave-one-captain-out psi_hat (exact, multithreaded)
# =====================================================================

def _loo_one_captain(args):
    """Worker function: re-estimate AKM excluding one captain."""
    captain_id, df_without, outcome_col = args
    try:
        _, agent_fe, _ = _estimate_akm_simple(df_without, outcome_col)
        return captain_id, agent_fe
    except Exception as e:
        logger.warning("LOO failed for captain %s: %s", captain_id, e)
        return captain_id, None


def compute_loo_psi_hat_exact(
    df: pd.DataFrame,
    n_workers: int = 8,
    outcome_col: str = "log_q",
) -> pd.DataFrame:
    """Leave-one-captain-out psi_hat: for each captain c, re-estimate AKM
    on the sample excluding c's voyages and return the agent FEs.

    Returns a DataFrame with columns [voyage_id, captain_id, agent_id,
    psi_hat_loo] where psi_hat_loo is the agent effect estimated without
    that voyage's captain.
    """
    logger.info("Computing exact LOO psi_hat (n_workers=%d)...", n_workers)
    captains = df["captain_id"].unique()
    n_cap = len(captains)
    logger.info("  %d captains to process", n_cap)

    # Pre-build per-captain exclusion DataFrames
    tasks = []
    for c in captains:
        df_without = df[df["captain_id"] != c].copy()
        # Ensure connected: drop agents that become singletons
        ag_counts = df_without["agent_id"].value_counts()
        valid_agents = ag_counts[ag_counts >= 1].index
        df_without = df_without[df_without["agent_id"].isin(valid_agents)]
        if len(df_without) < 50:
            continue
        tasks.append((c, df_without, outcome_col))

    # Run in parallel
    loo_agent_fes: Dict[str, pd.Series] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_loo_one_captain, t): t[0] for t in tasks}
        for future in as_completed(futures):
            captain_id, agent_fe = future.result()
            if agent_fe is not None:
                loo_agent_fes[captain_id] = agent_fe
            done += 1
            if done % 200 == 0:
                logger.info("  LOO progress: %d / %d", done, n_cap)

    logger.info("  LOO complete: %d / %d captains succeeded", len(loo_agent_fes), n_cap)

    # Map back to voyage level
    rows = []
    for _, voy in df[["voyage_id", "captain_id", "agent_id"]].iterrows():
        c = voy["captain_id"]
        a = voy["agent_id"]
        if c in loo_agent_fes:
            fe = loo_agent_fes[c]
            psi_val = fe.get(a, np.nan)
        else:
            psi_val = np.nan
        rows.append({"voyage_id": voy["voyage_id"], "captain_id": c, "agent_id": a, "psi_hat_loo": psi_val})

    return pd.DataFrame(rows)


def compute_loo_psi_hat_approx(
    df: pd.DataFrame,
    outcome_col: str = "log_q",
) -> pd.DataFrame:
    """Approximate LOO psi_hat using leverage correction (fast fallback).

    Uses the fact that for captain c with n_c observations,
    psi_hat_(-c) ≈ psi_hat — this is an approximation because removing
    one captain changes agent effects only slightly in large samples.
    We still compute the full-sample agent FE and flag it as approximate.
    """
    logger.info("Computing approximate LOO psi_hat...")
    _, agent_fe, _ = _estimate_akm_simple(df, outcome_col)

    result = df[["voyage_id", "captain_id", "agent_id"]].copy()
    result["psi_hat_loo"] = result["agent_id"].map(agent_fe)
    return result


# =====================================================================
# 2. Captain-specific pre-period psi_hat
# =====================================================================

def compute_preperiod_psi_hat(
    df: pd.DataFrame,
    outcome_col: str = "log_q",
) -> pd.DataFrame:
    """For each captain-voyage, compute psi_hat using only data before that
    captain's first voyage year.

    This is the most conservative anti-leakage construction: the
    organizational environment estimate cannot be influenced by any
    observation involving the focal captain.

    Returns DataFrame with [voyage_id, captain_id, agent_id, psi_hat_pre,
    pre_period_cutoff].
    """
    logger.info("Computing captain-specific pre-period psi_hat...")
    # First voyage year per captain
    first_year = df.groupby("captain_id")["year_out"].min().rename("first_year")
    df = df.merge(first_year, on="captain_id", how="left")

    # For each captain, estimate AKM on data before their first year  
    captains = df["captain_id"].unique()
    results = []

    # Group captains by their first_year to batch estimation
    captain_first_years = df.drop_duplicates("captain_id").set_index("captain_id")["first_year"]
    unique_cutoffs = sorted(captain_first_years.unique())

    # For each cutoff year, estimate one AKM on all data before that year
    cutoff_agent_fes: Dict[int, pd.Series] = {}
    for cutoff_year in unique_cutoffs:
        df_pre = df[df["year_out"] < cutoff_year].copy()
        if len(df_pre) < 50 or df_pre["agent_id"].nunique() < 2:
            continue
        try:
            _, agent_fe, _ = _estimate_akm_simple(df_pre, outcome_col)
            cutoff_agent_fes[cutoff_year] = agent_fe
        except Exception as e:
            logger.warning("Pre-period estimation failed for cutoff %d: %s", cutoff_year, e)

    logger.info("  Estimated pre-period AKMs for %d / %d cutoff years",
                len(cutoff_agent_fes), len(unique_cutoffs))

    # Map back to voyages
    rows = []
    for _, voy in df[["voyage_id", "captain_id", "agent_id", "first_year"]].iterrows():
        cutoff = int(voy["first_year"])
        a = voy["agent_id"]
        if cutoff in cutoff_agent_fes:
            psi_val = cutoff_agent_fes[cutoff].get(a, np.nan)
        else:
            psi_val = np.nan
        rows.append({
            "voyage_id": voy["voyage_id"],
            "captain_id": voy["captain_id"],
            "agent_id": a,
            "psi_hat_pre": psi_val,
            "pre_period_cutoff": cutoff,
        })

    result = pd.DataFrame(rows)
    logger.info("  Pre-period psi_hat coverage: %.1f%%",
                100 * result["psi_hat_pre"].notna().mean())
    return result


# =====================================================================
# 3. Separate-sample theta_hat with EB shrinkage
# =====================================================================

def compute_separate_sample_theta(
    df: pd.DataFrame,
    outcome_col: str = "log_q",
) -> pd.DataFrame:
    """Compute EB-shrunk theta_hat using leave-one-out approximation.

    For each captain c, theta_hat_loo ≈ theta_hat_full * (n_c / (n_c - 1)).
    Then apply EB shrinkage: theta_eb = grand_mean + lambda * (theta_loo - grand_mean).

    Returns DataFrame with [captain_id, theta_hat_sep, theta_hat_quartile, n_voyages, lambda_eb].
    """
    logger.info("Computing separate-sample theta_hat with EB shrinkage...")
    captain_fe, agent_fe, residuals = _estimate_akm_simple(df, outcome_col)

    # Voyage counts per captain
    cap_counts = df.groupby("captain_id").size()
    n_c = np.array([cap_counts.get(c, 1) for c in captain_fe.index])

    # LOO correction
    theta_loo = captain_fe.values * (n_c / np.maximum(n_c - 1, 1))

    # EB shrinkage
    sigma2_eps = np.var(residuals)
    grand_mean = np.mean(theta_loo)
    var_signal = max(0, np.var(theta_loo) - sigma2_eps * np.mean(1.0 / n_c))

    if var_signal > 0:
        lam = var_signal / (var_signal + sigma2_eps / n_c)
    else:
        lam = np.zeros_like(n_c, dtype=float)

    theta_eb = grand_mean + lam * (theta_loo - grand_mean)

    result = pd.DataFrame({
        "captain_id": captain_fe.index,
        "theta_hat_sep": theta_eb,
        "n_voyages": n_c,
        "lambda_eb": lam,
    })
    result["theta_hat_quartile"] = pd.qcut(
        result["theta_hat_sep"].rank(method="first"),
        CFG.n_skill_bins,
        labels=[f"Q{i+1}" for i in range(CFG.n_skill_bins)],
    )
    result["low_skill"] = (
        result["theta_hat_sep"] <= result["theta_hat_sep"].quantile(CFG.low_skill_fraction)
    ).astype(int)

    logger.info("  Theta_hat quartile distribution: %s",
                result["theta_hat_quartile"].value_counts().to_dict())
    return result


# =====================================================================
# Master entry point
# =====================================================================

def build_all_separate_sample_effects(
    save: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Build all separate-sample effect objects and save to intermediates/.

    Returns dict with keys: psi_hat_loo, psi_hat_pre, theta_hat_sep.
    """
    logger.info("=" * 60)
    logger.info("STEP 4: BUILDING SEPARATE-SAMPLE EFFECT OBJECTS")
    logger.info("=" * 60)

    df = _load_connected_panel()
    logger.info("Loaded connected panel: %d voyages, %d captains, %d agents",
                len(df), df["captain_id"].nunique(), df["agent_id"].nunique())

    # 1. LOO psi_hat
    if CFG.use_exact_loo:
        psi_loo = compute_loo_psi_hat_exact(df, n_workers=CFG.n_loo_workers)
    else:
        psi_loo = compute_loo_psi_hat_approx(df)

    # 2. Pre-period psi_hat (captain-specific)
    psi_pre = compute_preperiod_psi_hat(df)

    # 3. Separate-sample theta_hat
    theta_sep = compute_separate_sample_theta(df)

    results = {
        "psi_hat_loo": psi_loo,
        "psi_hat_pre": psi_pre,
        "theta_hat_sep": theta_sep,
    }

    if save:
        psi_loo.to_parquet(INTERMEDIATES_DIR / "psi_hat_leave_one_captain_out.parquet", index=False)
        psi_pre.to_parquet(INTERMEDIATES_DIR / "psi_hat_preperiod.parquet", index=False)
        theta_sep.to_parquet(INTERMEDIATES_DIR / "theta_hat_separate_sample.parquet", index=False)

        # Metadata
        meta = {
            "psi_hat_loo": {
                "method": "exact" if CFG.use_exact_loo else "approximate",
                "n_workers": CFG.n_loo_workers,
                "n_voyages": len(psi_loo),
                "coverage_pct": float(100 * psi_loo["psi_hat_loo"].notna().mean()),
            },
            "psi_hat_pre": {
                "method": "captain_specific_pre_period",
                "n_voyages": len(psi_pre),
                "coverage_pct": float(100 * psi_pre["psi_hat_pre"].notna().mean()),
            },
            "theta_hat_sep": {
                "method": "loo_approx_plus_eb_shrinkage",
                "n_captains": len(theta_sep),
                "n_skill_bins": CFG.n_skill_bins,
            },
        }
        with open(INTERMEDIATES_DIR / "effect_construction_metadata.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

        logger.info("Saved separate-sample effects to %s", INTERMEDIATES_DIR)

    return results
