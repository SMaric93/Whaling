"""
Reinforcement Test Suite — Cross-Fitted Type Estimation.

Implements held-out estimates of captain skill (theta) and
organizational capability (psi) to avoid circularity:
- Time-split cross-fitting (pre/post median year)
- K-fold cross-fitting (by decade)
- Leave-one-voyage-out (LOVO)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

from .config import CFG, COLS, TABLES_DIR
from .utils import make_table

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# AKM Estimation Core
# ═══════════════════════════════════════════════════════════════════════════

def estimate_akm(
    df: pd.DataFrame,
    outcome_col: str = None,
    *,
    captain_col: str = None,
    agent_col: str = None,
    controls: Optional[List[str]] = None,
) -> Dict:
    """
    Estimate additive two-way fixed effects (AKM) model:

        y_{v} = α_{captain} + ψ_{agent} + X'β + ε_{v}

    Parameters
    ----------
    df : pd.DataFrame
        Voyage panel.
    outcome_col : str
        Dependent variable column.
    captain_col, agent_col : str
        Identifier columns.
    controls : list of str
        Additional control variable columns.

    Returns
    -------
    dict with:
        captain_effects : pd.Series (captain_id → alpha)
        agent_effects : pd.Series (agent_id → psi)
        control_coefs : dict (control → coefficient)
        residuals : np.ndarray
        r_squared : float
        n_obs : int
    """
    outcome_col = outcome_col or COLS.log_q
    captain_col = captain_col or COLS.captain_id
    agent_col = agent_col or COLS.agent_id
    controls = controls or []

    # Clean data
    required = [outcome_col, captain_col, agent_col] + controls
    df_clean = df.dropna(subset=required).copy()
    n = len(df_clean)

    if n < 50:
        logger.warning("AKM: too few observations (%d), skipping", n)
        return {"captain_effects": pd.Series(dtype=float),
                "agent_effects": pd.Series(dtype=float),
                "n_obs": n}

    y = df_clean[outcome_col].values.astype(float)

    # Encode FEs
    captain_codes, captain_ids = pd.factorize(df_clean[captain_col])
    agent_codes, agent_ids = pd.factorize(df_clean[agent_col])
    n_captains = len(captain_ids)
    n_agents = len(agent_ids)

    # Captain dummies
    D_captain = sp.csc_matrix(
        (np.ones(n), (np.arange(n), captain_codes)),
        shape=(n, n_captains),
    )
    # Agent dummies (drop last for identification)
    D_agent = sp.csc_matrix(
        (np.ones(n), (np.arange(n), agent_codes)),
        shape=(n, n_agents),
    )

    # Controls
    if controls:
        X_ctrl = df_clean[controls].values.astype(float)
        design = sp.hstack([D_captain, D_agent, sp.csc_matrix(X_ctrl)], format="csc")
    else:
        design = sp.hstack([D_captain, D_agent], format="csc")

    # Solve
    result = lsqr(design, y, atol=1e-12, btol=1e-12)
    beta = result[0]

    # Partition effects
    alpha_hat = beta[:n_captains]
    psi_hat = beta[n_captains:n_captains + n_agents]
    ctrl_coefs = {}
    if controls:
        for i, ctrl in enumerate(controls):
            ctrl_coefs[ctrl] = beta[n_captains + n_agents + i]

    # Normalize: mean(alpha) = 0
    grand_mean = alpha_hat.mean()
    alpha_hat = alpha_hat - grand_mean

    # Residuals
    fitted = design @ beta
    residuals = y - fitted
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - y.mean())**2)
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    captain_effects = pd.Series(alpha_hat, index=captain_ids, name="theta")
    agent_effects = pd.Series(psi_hat, index=agent_ids, name="psi")

    logger.info(
        "AKM: n=%d, captains=%d, agents=%d, R²=%.4f, "
        "Var(α)=%.4f, Var(ψ)=%.4f",
        n, n_captains, n_agents, r_sq,
        alpha_hat.var(), psi_hat.var(),
    )

    return {
        "captain_effects": captain_effects,
        "agent_effects": agent_effects,
        "control_coefs": ctrl_coefs,
        "residuals": residuals,
        "r_squared": r_sq,
        "n_obs": n,
        "n_captains": n_captains,
        "n_agents": n_agents,
        "var_alpha": alpha_hat.var(),
        "var_psi": psi_hat.var(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Cross-Fitting
# ═══════════════════════════════════════════════════════════════════════════

def cross_fit_time_split(
    df: pd.DataFrame,
    *,
    outcome_col: str = None,
    controls: Optional[List[str]] = None,
    year_col: str = None,
) -> pd.DataFrame:
    """
    Time-split cross-fitting: estimate on one half, predict on other.

    Split at median year. Two folds:
    - Train on early, predict theta/psi for late voyages
    - Train on late, predict theta/psi for early voyages

    Returns
    -------
    pd.DataFrame with theta_heldout and psi_heldout columns.
    """
    outcome_col = outcome_col or COLS.log_q
    year_col = year_col or COLS.year_out
    controls = controls or []

    df = df.copy()
    median_year = df[year_col].median()
    logger.info("Time-split cross-fitting at median year = %.0f", median_year)

    df["_fold"] = (df[year_col] >= median_year).astype(int)

    df["theta_heldout"] = np.nan
    df["psi_heldout"] = np.nan

    for fold in [0, 1]:
        train_mask = df["_fold"] != fold
        test_mask = df["_fold"] == fold
        train = df[train_mask]
        test = df[test_mask]

        logger.info(
            "Fold %d: train=%d, test=%d",
            fold, len(train), len(test),
        )

        # Estimate AKM on training set
        akm = estimate_akm(train, outcome_col, controls=controls)
        captain_fx = akm["captain_effects"]
        agent_fx = akm["agent_effects"]

        # Predict on test set
        df.loc[test_mask, "theta_heldout"] = (
            df.loc[test_mask, COLS.captain_id].map(captain_fx)
        )
        df.loc[test_mask, "psi_heldout"] = (
            df.loc[test_mask, COLS.agent_id].map(agent_fx)
        )

    # Coverage stats
    n_theta = df["theta_heldout"].notna().sum()
    n_psi = df["psi_heldout"].notna().sum()
    logger.info(
        "Cross-fitted coverage: theta=%d (%.1f%%), psi=%d (%.1f%%)",
        n_theta, 100 * n_theta / len(df),
        n_psi, 100 * n_psi / len(df),
    )

    df = df.drop(columns=["_fold"])
    return df


def cross_fit_kfold(
    df: pd.DataFrame,
    *,
    n_folds: int = None,
    outcome_col: str = None,
    controls: Optional[List[str]] = None,
    seed: int = None,
) -> pd.DataFrame:
    """
    K-fold cross-fitting by decade (or random folds).

    Returns
    -------
    pd.DataFrame with theta_heldout and psi_heldout columns.
    """
    n_folds = n_folds or CFG.cross_fit_n_folds
    outcome_col = outcome_col or COLS.log_q
    controls = controls or []
    seed = seed or CFG.cross_fit_seed

    df = df.copy()

    # Assign folds by decade for temporal stability
    if "decade" in df.columns:
        decades = sorted(df["decade"].unique())
        decade_to_fold = {d: i % n_folds for i, d in enumerate(decades)}
        df["_fold"] = df["decade"].map(decade_to_fold)
    else:
        rng = np.random.RandomState(seed)
        df["_fold"] = rng.randint(0, n_folds, size=len(df))

    df["theta_heldout"] = np.nan
    df["psi_heldout"] = np.nan

    for fold in range(n_folds):
        train_mask = df["_fold"] != fold
        test_mask = df["_fold"] == fold
        train = df[train_mask]

        akm = estimate_akm(train, outcome_col, controls=controls)

        df.loc[test_mask, "theta_heldout"] = (
            df.loc[test_mask, COLS.captain_id].map(akm["captain_effects"])
        )
        df.loc[test_mask, "psi_heldout"] = (
            df.loc[test_mask, COLS.agent_id].map(akm["agent_effects"])
        )

    n_theta = df["theta_heldout"].notna().sum()
    n_psi = df["psi_heldout"].notna().sum()
    logger.info(
        "K-fold cross-fit (k=%d): theta=%d (%.1f%%), psi=%d (%.1f%%)",
        n_folds, n_theta, 100 * n_theta / len(df),
        n_psi, 100 * n_psi / len(df),
    )

    df = df.drop(columns=["_fold"])
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Summary Table
# ═══════════════════════════════════════════════════════════════════════════

def generate_type_estimation_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a summary table comparing in-sample and held-out estimates.

    Returns a DataFrame saved to outputs/tables/type_estimation_summary.csv.
    """
    rows = []

    for label, theta_col, psi_col in [
        ("In-sample", "theta", "psi"),
        ("Held-out", "theta_heldout", "psi_heldout"),
    ]:
        theta = df[theta_col].dropna() if theta_col in df.columns else pd.Series(dtype=float)
        psi = df[psi_col].dropna() if psi_col in df.columns else pd.Series(dtype=float)

        rows.append({
            "estimate_type": label,
            "n_theta": len(theta),
            "theta_mean": theta.mean() if len(theta) else np.nan,
            "theta_std": theta.std() if len(theta) else np.nan,
            "theta_var": theta.var() if len(theta) else np.nan,
            "theta_iqr": theta.quantile(0.75) - theta.quantile(0.25) if len(theta) else np.nan,
            "n_psi": len(psi),
            "psi_mean": psi.mean() if len(psi) else np.nan,
            "psi_std": psi.std() if len(psi) else np.nan,
            "psi_var": psi.var() if len(psi) else np.nan,
            "psi_iqr": psi.quantile(0.75) - psi.quantile(0.25) if len(psi) else np.nan,
        })

        # Correlation between in-sample and held-out
        if label == "Held-out" and "theta" in df.columns and theta_col in df.columns:
            both = df[[theta_col, "theta"]].dropna()
            if len(both) > 10:
                rows[-1]["theta_corr_insample"] = both.corr().iloc[0, 1]
            both_psi = df[[psi_col, "psi"]].dropna()
            if len(both_psi) > 10:
                rows[-1]["psi_corr_insample"] = both_psi.corr().iloc[0, 1]

    summary = pd.DataFrame(rows)
    path = TABLES_DIR / "type_estimation_summary.csv"
    summary.to_csv(path, index=False)
    logger.info("Saved type estimation summary: %s", path)
    return summary


# ═══════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def run_type_estimation(
    df: pd.DataFrame,
    *,
    method: str = None,
    outcome_col: str = None,
    controls: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Run the full type estimation pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis panel.
    method : str
        "time_split" or "kfold". Default from config.
    outcome_col : str
        Dependent variable.
    controls : list of str
        Additional controls.

    Returns
    -------
    pd.DataFrame with theta_heldout and psi_heldout columns added.
    """
    method = method or CFG.cross_fit_method
    outcome_col = outcome_col or COLS.log_q

    logger.info("Running type estimation (method=%s)", method)

    if method == "time_split":
        df = cross_fit_time_split(df, outcome_col=outcome_col, controls=controls)
    elif method == "kfold":
        df = cross_fit_kfold(df, outcome_col=outcome_col, controls=controls)
    else:
        raise ValueError(f"Unknown cross-fit method: {method}")

    summary = generate_type_estimation_summary(df)
    print("\n" + "=" * 60)
    print("TYPE ESTIMATION SUMMARY")
    print("=" * 60)
    print(summary.to_string(index=False))

    return df
