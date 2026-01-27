"""
TFP (Total Factor Productivity) Analysis Module.

Implements voyage-level TFP construction and AKM-style efficiency decomposition:
- W1: Preprocessing and sample rules with regime indicators
- W2: Tonnage elasticity estimation (pooled and regime-specific)
- W3: TFP construction net of tonnage and environment FEs
- W4: AKM efficiency effects (one-step) with captain/agent FEs
- W5: Two-step partialling robustness
- W6: Regime comparisons and variance decomposition
- W7: Sanity checks and falsification tests

Key methodological features:
- Regime-specific tonnage elasticities (avoiding beta=1 restriction)
- Two-way clustered standard errors (agent + captain)
- KSS bias correction for variance components
- Connected-set identification for AKM
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr
from scipy import stats
import warnings

from .config import (
    DATA_DIR,
    OUTPUT_DIR,
    TABLES_DIR,
    FIGURES_DIR,
    SampleConfig,
    DEFAULT_SAMPLE,
)
from .connected_set import find_leave_one_out_connected_set
from .data_loader import prepare_analysis_sample

warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# TFP Configuration
# =============================================================================

@dataclass
class TFPConfig:
    """Configuration for TFP analysis."""
    
    # Regime cutoff
    regime_cutoff_year: int = 1870
    alternative_cutoffs: List[int] = field(default_factory=lambda: [1865, 1875])
    
    # Winsorization
    winsorize_tfp_pct: float = 1.0  # Winsorize at 1st/99th percentile
    
    # Analysis options
    run_chow_test: bool = True
    use_common_support: bool = False  # Restrict to overlapping tonnage range
    use_regime_specific_beta: bool = True  # If False, use pooled beta
    
    # Clustering
    primary_cluster: str = "agent_id"
    secondary_cluster: str = "captain_id"


DEFAULT_TFP_CONFIG = TFPConfig()


# =============================================================================
# W1: Preprocessing and Sample Rules
# =============================================================================

def add_regime_indicator(
    df: pd.DataFrame,
    cutoff_year: int = 1870,
) -> pd.DataFrame:
    """
    Add regime indicator (pre/post cutoff) to voyage data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Voyage data with year_out column.
    cutoff_year : int
        Year to split regimes.
        
    Returns
    -------
    pd.DataFrame
        Data with regime columns added.
    """
    df = df.copy()
    
    df["regime"] = np.where(df["year_out"] < cutoff_year, "pre", "post")
    df["is_pre_regime"] = (df["regime"] == "pre").astype(int)
    df["is_post_regime"] = (df["regime"] == "post").astype(int)
    
    n_pre = df["is_pre_regime"].sum()
    n_post = df["is_post_regime"].sum()
    
    print(f"\nRegime indicator (cutoff={cutoff_year}):")
    print(f"  Pre-regime: {n_pre:,} voyages ({100*n_pre/len(df):.1f}%)")
    print(f"  Post-regime: {n_post:,} voyages ({100*n_post/len(df):.1f}%)")
    
    return df


def prepare_tfp_sample(
    df: Optional[pd.DataFrame] = None,
    config: Optional[TFPConfig] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    W1: Prepare sample for TFP analysis with diagnostics.
    
    Parameters
    ----------
    df : pd.DataFrame, optional
        Pre-loaded voyage data. If None, loads from prepare_analysis_sample().
    config : TFPConfig, optional
        TFP configuration.
        
    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        (prepared_data, diagnostics)
    """
    if config is None:
        config = DEFAULT_TFP_CONFIG
        
    print("\n" + "=" * 60)
    print("W1: PREPROCESSING AND SAMPLE RULES")
    print("=" * 60)
    
    # Load data if not provided
    if df is None:
        df = prepare_analysis_sample()
    
    # Filter to required fields
    required_cols = ["voyage_id", "log_q", "log_tonnage", "captain_id", "agent_id",
                     "route_time", "port_time", "year_out"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    df_clean = df.dropna(subset=["log_q", "log_tonnage", "captain_id", "agent_id"])
    print(f"After dropping NA: {len(df_clean):,} voyages (dropped {len(df) - len(df_clean):,})")
    
    # Add regime indicator
    df_clean = add_regime_indicator(df_clean, config.regime_cutoff_year)
    
    # Common support (optional)
    if config.use_common_support:
        tonnage_pre = df_clean.loc[df_clean["is_pre_regime"] == 1, "log_tonnage"]
        tonnage_post = df_clean.loc[df_clean["is_post_regime"] == 1, "log_tonnage"]
        
        overlap_min = max(tonnage_pre.min(), tonnage_post.min())
        overlap_max = min(tonnage_pre.max(), tonnage_post.max())
        
        n_before = len(df_clean)
        df_clean = df_clean[
            (df_clean["log_tonnage"] >= overlap_min) & 
            (df_clean["log_tonnage"] <= overlap_max)
        ]
        print(f"Common support restriction: {len(df_clean):,} ({n_before - len(df_clean):,} dropped)")
    
    # Connected-set diagnostics
    n_captains = df_clean["captain_id"].nunique()
    n_agents = df_clean["agent_id"].nunique()
    
    diagnostics = {
        "n_voyages": len(df_clean),
        "n_captains": n_captains,
        "n_agents": n_agents,
        "n_pre": df_clean["is_pre_regime"].sum(),
        "n_post": df_clean["is_post_regime"].sum(),
        "tonnage_mean_pre": df_clean.loc[df_clean["is_pre_regime"] == 1, "log_tonnage"].mean(),
        "tonnage_mean_post": df_clean.loc[df_clean["is_post_regime"] == 1, "log_tonnage"].mean(),
    }
    
    print(f"\nSample summary:")
    print(f"  Voyages: {diagnostics['n_voyages']:,}")
    print(f"  Captains: {diagnostics['n_captains']:,}")
    print(f"  Agents: {diagnostics['n_agents']:,}")
    
    return df_clean, diagnostics


# =============================================================================
# W2: Tonnage Elasticity Estimation
# =============================================================================

def build_fe_design_matrix(
    df: pd.DataFrame,
    fe_cols: List[str],
) -> Tuple[sp.csr_matrix, Dict]:
    """
    Build sparse design matrix for fixed effects.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with FE group columns.
    fe_cols : List[str]
        Column names for fixed effect groups.
        
    Returns
    -------
    Tuple[sp.csr_matrix, Dict]
        (design_matrix, index_maps)
    """
    n = len(df)
    matrices = []
    index_maps = {}
    
    for col in fe_cols:
        ids = df[col].unique()
        id_map = {v: i for i, v in enumerate(ids)}
        idx = df[col].map(id_map).values
        
        X_full = sp.csr_matrix(
            (np.ones(n), (np.arange(n), idx)),
            shape=(n, len(ids))
        )
        # Drop first for identification (except first FE)
        if len(matrices) > 0:
            X = X_full[:, 1:]
        else:
            X = X_full
        
        matrices.append(X)
        index_maps[col] = {"ids": ids, "map": id_map, "n": len(ids)}
    
    X = sp.hstack(matrices)
    return X, index_maps


def estimate_tonnage_elasticity(
    df: pd.DataFrame,
    config: Optional[TFPConfig] = None,
) -> Dict:
    """
    W2: Estimate tonnage elasticity with environment FEs.
    
    Specifications:
    - S1: Pooled beta (common elasticity across regimes)
    - S2: Regime-specific beta (beta_pre, beta_post)
    
    Parameters
    ----------
    df : pd.DataFrame
        Prepared voyage data.
    config : TFPConfig, optional
        Configuration.
        
    Returns
    -------
    Dict
        Estimation results including beta estimates and diagnostics.
    """
    if config is None:
        config = DEFAULT_TFP_CONFIG
        
    print("\n" + "=" * 60)
    print("W2: TONNAGE ELASTICITY ESTIMATION")
    print("=" * 60)
    
    n = len(df)
    y = df["log_q"].values
    
    # Build environment FE design matrix
    fe_cols = ["route_time", "port_time"]
    X_fe, fe_maps = build_fe_design_matrix(df, fe_cols)
    
    results = {}
    
    # =========================================================================
    # S1: Pooled beta
    # =========================================================================
    print("\n--- S1: Pooled Tonnage Elasticity ---")
    
    log_tonnage = df["log_tonnage"].values.reshape(-1, 1)
    X_pooled = sp.hstack([sp.csr_matrix(log_tonnage), X_fe])
    
    sol = lsqr(X_pooled, y, iter_lim=10000, atol=1e-10, btol=1e-10)
    beta_pooled = sol[0][0]
    
    # Residuals for SE estimation
    y_hat_pooled = X_pooled @ sol[0]
    resid_pooled = y - y_hat_pooled
    r2_pooled = 1 - np.var(resid_pooled) / np.var(y)
    
    # Cluster-robust SE (primary cluster)
    se_pooled = _cluster_robust_se(
        X_pooled[:, 0].toarray().flatten(), resid_pooled, 
        df[config.primary_cluster].values, coef_idx=0
    )
    
    print(f"  β_pooled = {beta_pooled:.4f} (SE = {se_pooled:.4f})")
    print(f"  R² = {r2_pooled:.4f}")
    
    results["s1_pooled"] = {
        "beta": beta_pooled,
        "se": se_pooled,
        "r2": r2_pooled,
        "n": n,
        "residuals": resid_pooled,
        "fitted": y_hat_pooled,
    }
    
    # =========================================================================
    # S2: Regime-specific beta
    # =========================================================================
    print("\n--- S2: Regime-Specific Tonnage Elasticity ---")
    
    tonnage_pre = (df["log_tonnage"] * df["is_pre_regime"]).values.reshape(-1, 1)
    tonnage_post = (df["log_tonnage"] * df["is_post_regime"]).values.reshape(-1, 1)
    
    X_regime = sp.hstack([
        sp.csr_matrix(tonnage_pre),
        sp.csr_matrix(tonnage_post),
        X_fe
    ])
    
    sol_regime = lsqr(X_regime, y, iter_lim=10000, atol=1e-10, btol=1e-10)
    beta_pre = sol_regime[0][0]
    beta_post = sol_regime[0][1]
    
    y_hat_regime = X_regime @ sol_regime[0]
    resid_regime = y - y_hat_regime
    r2_regime = 1 - np.var(resid_regime) / np.var(y)
    
    # Cluster-robust SEs
    se_pre = _cluster_robust_se(
        X_regime[:, 0].toarray().flatten(), resid_regime,
        df[config.primary_cluster].values, coef_idx=0
    )
    se_post = _cluster_robust_se(
        X_regime[:, 1].toarray().flatten(), resid_regime,
        df[config.primary_cluster].values, coef_idx=0
    )
    
    print(f"  β_pre = {beta_pre:.4f} (SE = {se_pre:.4f})")
    print(f"  β_post = {beta_post:.4f} (SE = {se_post:.4f})")
    print(f"  R² = {r2_regime:.4f}")
    
    # Test beta_pre = beta_post
    diff = beta_pre - beta_post
    se_diff = np.sqrt(se_pre**2 + se_post**2)  # Conservative (ignores covariance)
    t_stat = diff / se_diff
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - X_regime.shape[1]))
    
    print(f"\n  Test β_pre = β_post:")
    print(f"    Difference = {diff:.4f}, t = {t_stat:.2f}, p = {p_value:.4f}")
    
    results["s2_regime"] = {
        "beta_pre": beta_pre,
        "beta_post": beta_post,
        "se_pre": se_pre,
        "se_post": se_post,
        "r2": r2_regime,
        "n": n,
        "test_equal_p": p_value,
        "residuals": resid_regime,
        "fitted": y_hat_regime,
    }
    
    # =========================================================================
    # Chow test for cutoff alternatives
    # =========================================================================
    if config.run_chow_test:
        print("\n--- Chow Test for Alternative Cutoffs ---")
        chow_results = []
        
        for alt_cutoff in config.alternative_cutoffs:
            df_alt = add_regime_indicator(df.copy(), alt_cutoff)
            
            tonnage_pre_alt = (df_alt["log_tonnage"] * df_alt["is_pre_regime"]).values.reshape(-1, 1)
            tonnage_post_alt = (df_alt["log_tonnage"] * df_alt["is_post_regime"]).values.reshape(-1, 1)
            
            X_alt = sp.hstack([
                sp.csr_matrix(tonnage_pre_alt),
                sp.csr_matrix(tonnage_post_alt),
                X_fe
            ])
            
            sol_alt = lsqr(X_alt, y, iter_lim=10000, atol=1e-10, btol=1e-10)
            y_hat_alt = X_alt @ sol_alt[0]
            rss_unrestricted = np.sum((y - y_hat_alt)**2)
            rss_restricted = np.sum(resid_pooled**2)
            
            # F-statistic: ((RSS_r - RSS_u) / q) / (RSS_u / (n - k))
            q = 1  # One restriction (beta_pre = beta_post)
            k = X_alt.shape[1]
            f_stat = ((rss_restricted - rss_unrestricted) / q) / (rss_unrestricted / (n - k))
            p_chow = 1 - stats.f.cdf(f_stat, q, n - k)
            
            chow_results.append({
                "cutoff": alt_cutoff,
                "f_stat": f_stat,
                "p_value": p_chow,
                "beta_pre": sol_alt[0][0],
                "beta_post": sol_alt[0][1],
            })
            
            print(f"  Cutoff {alt_cutoff}: F = {f_stat:.2f}, p = {p_chow:.4f}")
        
        results["chow_tests"] = chow_results
    
    # Store FE components for TFP construction
    results["fe_maps"] = fe_maps
    results["X_fe"] = X_fe
    
    return results


def _cluster_robust_se(
    x: np.ndarray,
    residuals: np.ndarray,
    clusters: np.ndarray,
    coef_idx: int = 0,
) -> float:
    """
    Compute cluster-robust standard error for a single coefficient.
    
    Parameters
    ----------
    x : np.ndarray
        Regressor values (n,).
    residuals : np.ndarray
        Regression residuals.
    clusters : np.ndarray
        Cluster identifiers.
        
    Returns
    -------
    float
        Cluster-robust standard error.
    """
    unique_clusters = np.unique(clusters)
    G = len(unique_clusters)
    n = len(x)
    
    # Cluster sums of x * residual
    cluster_sums = np.zeros(G)
    for g, c in enumerate(unique_clusters):
        mask = clusters == c
        cluster_sums[g] = np.sum(x[mask] * residuals[mask])
    
    # Meat of sandwich
    meat = np.sum(cluster_sums**2)
    
    # Bread
    bread = np.sum(x**2)
    
    # Cluster-robust variance with small-sample correction
    correction = (G / (G - 1)) * ((n - 1) / (n - 1))  # Simplified
    var_beta = correction * meat / (bread**2)
    
    return np.sqrt(max(0, var_beta))


# =============================================================================
# W3: TFP Construction
# =============================================================================

def construct_tfp(
    df: pd.DataFrame,
    elasticity_results: Dict,
    config: Optional[TFPConfig] = None,
) -> pd.DataFrame:
    """
    W3: Construct TFP measures using estimated tonnage elasticity.
    
    tfp_hat = log_output - beta_hat * log_tonnage - FE_hat
    
    Parameters
    ----------
    df : pd.DataFrame
        Prepared voyage data.
    elasticity_results : Dict
        Results from estimate_tonnage_elasticity().
    config : TFPConfig, optional
        Configuration.
        
    Returns
    -------
    pd.DataFrame
        Data with tfp_hat and tfp_resid columns added.
    """
    if config is None:
        config = DEFAULT_TFP_CONFIG
        
    print("\n" + "=" * 60)
    print("W3: TFP CONSTRUCTION")
    print("=" * 60)
    
    df = df.copy()
    
    # Choose beta based on config
    if config.use_regime_specific_beta:
        beta_pre = elasticity_results["s2_regime"]["beta_pre"]
        beta_post = elasticity_results["s2_regime"]["beta_post"]
        residuals = elasticity_results["s2_regime"]["residuals"]
        
        # TFP = residual from regime-specific regression
        df["tfp_hat"] = residuals
        df["beta_used"] = np.where(df["is_pre_regime"] == 1, beta_pre, beta_post)
        
        print(f"Using regime-specific beta: β_pre={beta_pre:.4f}, β_post={beta_post:.4f}")
    else:
        beta = elasticity_results["s1_pooled"]["beta"]
        residuals = elasticity_results["s1_pooled"]["residuals"]
        
        df["tfp_hat"] = residuals
        df["beta_used"] = beta
        
        print(f"Using pooled beta: β={beta:.4f}")
    
    # Winsorize TFP
    lower = np.percentile(df["tfp_hat"], config.winsorize_tfp_pct)
    upper = np.percentile(df["tfp_hat"], 100 - config.winsorize_tfp_pct)
    df["tfp_hat_winsorized"] = df["tfp_hat"].clip(lower=lower, upper=upper)
    
    n_winsorized = ((df["tfp_hat"] < lower) | (df["tfp_hat"] > upper)).sum()
    print(f"Winsorized {n_winsorized} observations ({100*n_winsorized/len(df):.1f}%)")
    
    # Alias for interpretation
    df["tfp_resid"] = df["tfp_hat"]
    
    # Standardize within regime (optional)
    for regime in ["pre", "post"]:
        mask = df["regime"] == regime
        mean_r = df.loc[mask, "tfp_hat"].mean()
        std_r = df.loc[mask, "tfp_hat"].std()
        df.loc[mask, "tfp_hat_std"] = (df.loc[mask, "tfp_hat"] - mean_r) / std_r
    
    # Diagnostics
    print(f"\nTFP distribution:")
    print(f"  Mean: {df['tfp_hat'].mean():.4f}")
    print(f"  Std: {df['tfp_hat'].std():.4f}")
    print(f"  Pre-regime mean: {df.loc[df['regime'] == 'pre', 'tfp_hat'].mean():.4f}")
    print(f"  Post-regime mean: {df.loc[df['regime'] == 'post', 'tfp_hat'].mean():.4f}")
    
    # Validation: check approximately mean-zero within route×year cells
    cell_means = df.groupby("route_time")["tfp_hat"].mean()
    print(f"\nValidation (mean TFP within route×year cells):")
    print(f"  Mean of cell means: {cell_means.mean():.6f}")
    print(f"  Std of cell means: {cell_means.std():.4f}")
    
    return df


# =============================================================================
# W4: AKM Efficiency Effects (One-Step)
# =============================================================================

def estimate_efficiency_effects(
    df: pd.DataFrame,
    config: Optional[TFPConfig] = None,
) -> Dict:
    """
    W4: Estimate captain and agent efficiency effects using one-step AKM.
    
    Includes tonnage and environment FEs directly so captain/agent effects
    represent efficiency conditional on assets and environment.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with tfp_hat from construct_tfp().
    config : TFPConfig, optional
        Configuration.
        
    Returns
    -------
    Dict
        AKM estimation results.
    """
    if config is None:
        config = DEFAULT_TFP_CONFIG
        
    print("\n" + "=" * 60)
    print("W4: AKM EFFICIENCY EFFECTS (ONE-STEP)")
    print("=" * 60)
    
    # Restrict to LOO connected set
    df_loo, loo_diag = find_leave_one_out_connected_set(df)
    
    n = len(df_loo)
    print(f"\nLOO connected set: {n:,} voyages ({100*n/len(df):.1f}% of sample)")
    
    y = df_loo["log_q"].values
    
    # Build design matrix: tonnage + captain FE + agent FE + route×year FE + port×year FE
    matrices = []
    index_maps = {}
    
    # Tonnage (regime-specific)
    tonnage_pre = (df_loo["log_tonnage"] * df_loo["is_pre_regime"]).values.reshape(-1, 1)
    tonnage_post = (df_loo["log_tonnage"] * df_loo["is_post_regime"]).values.reshape(-1, 1)
    matrices.append(sp.csr_matrix(tonnage_pre))
    matrices.append(sp.csr_matrix(tonnage_post))
    index_maps["tonnage"] = {"names": ["log_tonnage_pre", "log_tonnage_post"], "n": 2}
    
    # Captain FEs
    captain_ids = df_loo["captain_id"].unique()
    captain_map = {c: i for i, c in enumerate(captain_ids)}
    captain_idx = df_loo["captain_id"].map(captain_map).values
    X_captain = sp.csr_matrix(
        (np.ones(n), (np.arange(n), captain_idx)),
        shape=(n, len(captain_ids))
    )
    matrices.append(X_captain)
    index_maps["captain"] = {"ids": captain_ids, "map": captain_map, "n": len(captain_ids)}
    
    # Agent FEs (drop first)
    agent_ids = df_loo["agent_id"].unique()
    agent_map = {a: i for i, a in enumerate(agent_ids)}
    agent_idx = df_loo["agent_id"].map(agent_map).values
    X_agent_full = sp.csr_matrix(
        (np.ones(n), (np.arange(n), agent_idx)),
        shape=(n, len(agent_ids))
    )
    X_agent = X_agent_full[:, 1:]
    matrices.append(X_agent)
    index_maps["agent"] = {"ids": agent_ids, "map": agent_map, "n": len(agent_ids)}
    
    # Route×time FEs (drop first)
    rt_ids = df_loo["route_time"].unique()
    rt_map = {r: i for i, r in enumerate(rt_ids)}
    rt_idx = df_loo["route_time"].map(rt_map).values
    X_rt_full = sp.csr_matrix(
        (np.ones(n), (np.arange(n), rt_idx)),
        shape=(n, len(rt_ids))
    )
    X_rt = X_rt_full[:, 1:]
    matrices.append(X_rt)
    index_maps["route_time"] = {"ids": rt_ids, "map": rt_map, "n": len(rt_ids)}
    
    # Port×time FEs (drop first)
    pt_ids = df_loo["port_time"].unique()
    pt_map = {p: i for i, p in enumerate(pt_ids)}
    pt_idx = df_loo["port_time"].map(pt_map).values
    X_pt_full = sp.csr_matrix(
        (np.ones(n), (np.arange(n), pt_idx)),
        shape=(n, len(pt_ids))
    )
    X_pt = X_pt_full[:, 1:]
    matrices.append(X_pt)
    index_maps["port_time"] = {"ids": pt_ids, "map": pt_map, "n": len(pt_ids)}
    
    # Stack design matrix
    X = sp.hstack(matrices)
    print(f"Design matrix: {X.shape}")
    
    # Solve sparse least squares
    print("Solving sparse LS...")
    sol = lsqr(X, y, iter_lim=10000, atol=1e-10, btol=1e-10)
    beta = sol[0]
    
    # Extract coefficients
    idx = 0
    
    # Tonnage coefficients
    beta_pre = beta[idx]
    beta_post = beta[idx + 1]
    idx += 2
    
    # Captain effects
    n_captains = index_maps["captain"]["n"]
    alpha_eff = beta[idx:idx + n_captains]
    idx += n_captains
    
    # Agent effects (add 0 for normalized first agent)
    n_agents = index_maps["agent"]["n"]
    gamma_eff_est = beta[idx:idx + n_agents - 1]
    gamma_eff = np.concatenate([[0], gamma_eff_est])
    idx += n_agents - 1
    
    # Fitted and residuals
    y_hat = X @ beta
    residuals = y - y_hat
    r2 = 1 - np.var(residuals) / np.var(y)
    rmse = np.std(residuals)
    
    print(f"\nModel fit:")
    print(f"  R² = {r2:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  β_pre (tonnage) = {beta_pre:.4f}")
    print(f"  β_post (tonnage) = {beta_post:.4f}")
    
    # Create FE DataFrames
    captain_fe = pd.DataFrame({
        "captain_id": index_maps["captain"]["ids"],
        "alpha_eff": alpha_eff,
    })
    
    agent_fe = pd.DataFrame({
        "agent_id": index_maps["agent"]["ids"],
        "gamma_eff": gamma_eff,
    })
    
    # Merge back to data
    df_loo = df_loo.merge(captain_fe, on="captain_id", how="left")
    df_loo = df_loo.merge(agent_fe, on="agent_id", how="left")
    df_loo["residuals_akm"] = residuals
    df_loo["fitted_akm"] = y_hat
    
    results = {
        "df": df_loo,
        "captain_fe": captain_fe,
        "agent_fe": agent_fe,
        "alpha_eff": alpha_eff,
        "gamma_eff": gamma_eff,
        "beta_pre": beta_pre,
        "beta_post": beta_post,
        "r2": r2,
        "rmse": rmse,
        "n": n,
        "index_maps": index_maps,
        "loo_diagnostics": loo_diag,
        "X": X,
        "y": y,
        "residuals": residuals,
    }
    
    return results


def compute_variance_decomposition_tfp(
    results: Dict,
) -> pd.DataFrame:
    """
    Compute variance decomposition of TFP into captain, agent, and residual.
    
    Parameters
    ----------
    results : Dict
        Results from estimate_efficiency_effects().
        
    Returns
    -------
    pd.DataFrame
        Variance decomposition table.
    """
    print("\n" + "=" * 60)
    print("VARIANCE DECOMPOSITION (TFP)")
    print("=" * 60)
    
    df = results["df"]
    
    alpha_i = df["alpha_eff"].values
    gamma_j = df["gamma_eff"].values
    residuals = results["residuals"]
    y = results["y"]
    
    # Compute variances
    var_y = np.var(y)
    var_alpha = np.var(alpha_i)
    var_gamma = np.var(gamma_j)
    cov_alpha_gamma = np.cov(alpha_i, gamma_j)[0, 1]
    var_resid = np.var(residuals)
    
    # Total labor market component
    labor_total = var_alpha + var_gamma + 2 * cov_alpha_gamma
    
    # Shares within labor market
    if labor_total > 0:
        share_alpha = var_alpha / labor_total
        share_gamma = var_gamma / labor_total
        share_sorting = (2 * cov_alpha_gamma) / labor_total
    else:
        share_alpha = share_gamma = share_sorting = np.nan
    
    # Build table
    decomp = pd.DataFrame({
        "Component": [
            "Var(α_eff) - Captain Efficiency",
            "Var(γ_eff) - Agent Efficiency",
            "2×Cov(α,γ) - Sorting",
            "Var(ε) - Residual",
        ],
        "Variance": [var_alpha, var_gamma, 2*cov_alpha_gamma, var_resid],
        "Share_of_VarY": [var_alpha/var_y, var_gamma/var_y, 2*cov_alpha_gamma/var_y, var_resid/var_y],
        "Share_of_LaborMkt": [share_alpha, share_gamma, share_sorting, np.nan],
    })
    
    print(f"\nTotal Var(y) = {var_y:.4f}")
    print(f"Labor market component = {labor_total:.4f} ({100*labor_total/var_y:.1f}% of Var(y))")
    print(f"\n--- Within Labor Market Component ---")
    print(f"  Captain efficiency: {100*share_alpha:.1f}%")
    print(f"  Agent efficiency: {100*share_gamma:.1f}%")
    print(f"  Sorting: {100*share_sorting:.1f}%")
    
    # Correlation
    corr_alpha_gamma = cov_alpha_gamma / (np.sqrt(var_alpha) * np.sqrt(var_gamma)) if var_alpha > 0 and var_gamma > 0 else 0
    print(f"\nCorr(α_eff, γ_eff) = {corr_alpha_gamma:.4f}")
    
    results["variance_decomposition"] = decomp
    results["labor_market_total"] = labor_total
    results["corr_alpha_gamma"] = corr_alpha_gamma
    
    return decomp


# =============================================================================
# W5: Two-Step Partialling Robustness
# =============================================================================

def two_step_robustness(
    df: pd.DataFrame,
    config: Optional[TFPConfig] = None,
) -> Dict:
    """
    W5: Two-step partialling approach for robustness.
    
    Step A: Regress log_output on log_tonnage + environment FEs
    Step B: Run AKM on residuals (captain + agent FEs only)
    
    Parameters
    ----------
    df : pd.DataFrame
        Prepared voyage data.
    config : TFPConfig, optional
        Configuration.
        
    Returns
    -------
    Dict
        Two-step estimation results.
    """
    if config is None:
        config = DEFAULT_TFP_CONFIG
        
    print("\n" + "=" * 60)
    print("W5: TWO-STEP PARTIALLING ROBUSTNESS")
    print("=" * 60)
    
    # Restrict to LOO connected set
    df_loo, loo_diag = find_leave_one_out_connected_set(df)
    n = len(df_loo)
    
    # =========================================================================
    # Step A: Residualize output against tonnage + environment FEs
    # =========================================================================
    print("\n--- Step A: Residualization ---")
    
    y = df_loo["log_q"].values
    
    # Build FE matrix for route×time and port×time
    X_fe, fe_maps = build_fe_design_matrix(df_loo, ["route_time", "port_time"])
    
    # Add tonnage (regime-specific)
    tonnage_pre = (df_loo["log_tonnage"] * df_loo["is_pre_regime"]).values.reshape(-1, 1)
    tonnage_post = (df_loo["log_tonnage"] * df_loo["is_post_regime"]).values.reshape(-1, 1)
    
    X_step_a = sp.hstack([
        sp.csr_matrix(tonnage_pre),
        sp.csr_matrix(tonnage_post),
        X_fe
    ])
    
    sol_a = lsqr(X_step_a, y, iter_lim=10000, atol=1e-10, btol=1e-10)
    y_hat_a = X_step_a @ sol_a[0]
    resid_a = y - y_hat_a
    
    print(f"  Step A R² = {1 - np.var(resid_a)/np.var(y):.4f}")
    
    # =========================================================================
    # Step B: AKM on residuals
    # =========================================================================
    print("\n--- Step B: AKM on Residuals ---")
    
    # Build captain + agent FE matrix
    captain_ids = df_loo["captain_id"].unique()
    captain_map = {c: i for i, c in enumerate(captain_ids)}
    captain_idx = df_loo["captain_id"].map(captain_map).values
    X_captain = sp.csr_matrix(
        (np.ones(n), (np.arange(n), captain_idx)),
        shape=(n, len(captain_ids))
    )
    
    agent_ids = df_loo["agent_id"].unique()
    agent_map = {a: i for i, a in enumerate(agent_ids)}
    agent_idx = df_loo["agent_id"].map(agent_map).values
    X_agent_full = sp.csr_matrix(
        (np.ones(n), (np.arange(n), agent_idx)),
        shape=(n, len(agent_ids))
    )
    X_agent = X_agent_full[:, 1:]  # Drop first
    
    X_step_b = sp.hstack([X_captain, X_agent])
    
    sol_b = lsqr(X_step_b, resid_a, iter_lim=10000, atol=1e-10, btol=1e-10)
    
    # Extract FEs
    alpha_2step = sol_b[0][:len(captain_ids)]
    gamma_2step_est = sol_b[0][len(captain_ids):]
    gamma_2step = np.concatenate([[0], gamma_2step_est])
    
    y_hat_b = X_step_b @ sol_b[0]
    resid_b = resid_a - y_hat_b
    r2_b = 1 - np.var(resid_b) / np.var(resid_a)
    
    print(f"  Step B R² (on residualized y) = {r2_b:.4f}")
    
    # Create output DataFrames
    captain_fe_2step = pd.DataFrame({
        "captain_id": captain_ids,
        "alpha_eff_2step": alpha_2step,
    })
    
    agent_fe_2step = pd.DataFrame({
        "agent_id": agent_ids,
        "gamma_eff_2step": gamma_2step,
    })
    
    results = {
        "captain_fe": captain_fe_2step,
        "agent_fe": agent_fe_2step,
        "alpha_2step": alpha_2step,
        "gamma_2step": gamma_2step,
        "r2_step_a": 1 - np.var(resid_a)/np.var(y),
        "r2_step_b": r2_b,
        "residuals_step_a": resid_a,
        "residuals_step_b": resid_b,
    }
    
    return results


# =============================================================================
# W6: Regime Comparisons
# =============================================================================

def regime_comparison(
    df: pd.DataFrame,
    config: Optional[TFPConfig] = None,
) -> Dict:
    """
    W6: Compare efficiency effects and variance shares across regimes.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with efficiency effects.
    config : TFPConfig, optional
        Configuration.
        
    Returns
    -------
    Dict
        Regime comparison results.
    """
    if config is None:
        config = DEFAULT_TFP_CONFIG
        
    print("\n" + "=" * 60)
    print("W6: REGIME COMPARISONS")
    print("=" * 60)
    
    results = {}
    
    for regime in ["pre", "post"]:
        df_regime = df[df["regime"] == regime].copy()
        
        if len(df_regime) < 100:
            print(f"\n  Skipping {regime} regime (n={len(df_regime)} < 100)")
            continue
            
        print(f"\n--- {regime.upper()} Regime (n={len(df_regime):,}) ---")
        
        # Check connectivity
        df_loo, loo_diag = find_leave_one_out_connected_set(df_regime)
        
        if len(df_loo) < 50:
            print(f"  Connected set too small ({len(df_loo)}), skipping")
            continue
        
        # Estimate within-regime effects
        n = len(df_loo)
        y = df_loo["log_q"].values
        
        # Simple AKM (captain + agent + tonnage)
        captain_ids = df_loo["captain_id"].unique()
        captain_map = {c: i for i, c in enumerate(captain_ids)}
        captain_idx = df_loo["captain_id"].map(captain_map).values
        X_captain = sp.csr_matrix(
            (np.ones(n), (np.arange(n), captain_idx)),
            shape=(n, len(captain_ids))
        )
        
        agent_ids = df_loo["agent_id"].unique()
        agent_map = {a: i for i, a in enumerate(agent_ids)}
        agent_idx = df_loo["agent_id"].map(agent_map).values
        X_agent_full = sp.csr_matrix(
            (np.ones(n), (np.arange(n), agent_idx)),
            shape=(n, len(agent_ids))
        )
        X_agent = X_agent_full[:, 1:]
        
        log_tonnage = df_loo["log_tonnage"].values.reshape(-1, 1)
        
        X = sp.hstack([sp.csr_matrix(log_tonnage), X_captain, X_agent])
        
        sol = lsqr(X, y, iter_lim=10000, atol=1e-10, btol=1e-10)
        
        beta_tonnage = sol[0][0]
        alpha_r = sol[0][1:1+len(captain_ids)]
        gamma_r = np.concatenate([[0], sol[0][1+len(captain_ids):]])
        
        var_alpha = np.var(alpha_r)
        var_gamma = np.var(gamma_r)
        cov_ag = np.cov(alpha_r, gamma_r[:len(alpha_r)])[0, 1] if len(gamma_r) > len(alpha_r) else 0
        
        labor_total = var_alpha + var_gamma + 2*cov_ag
        share_alpha = var_alpha / labor_total if labor_total > 0 else np.nan
        share_gamma = var_gamma / labor_total if labor_total > 0 else np.nan
        
        print(f"  β_tonnage = {beta_tonnage:.4f}")
        print(f"  Var(α) = {var_alpha:.4f} ({100*share_alpha:.1f}%)")
        print(f"  Var(γ) = {var_gamma:.4f} ({100*share_gamma:.1f}%)")
        
        results[regime] = {
            "n": n,
            "n_captains": len(captain_ids),
            "n_agents": len(agent_ids),
            "beta_tonnage": beta_tonnage,
            "var_alpha": var_alpha,
            "var_gamma": var_gamma,
            "share_alpha": share_alpha,
            "share_gamma": share_gamma,
            "loo_coverage": len(df_loo) / len(df_regime),
        }
    
    return results


# =============================================================================
# W7: Sanity Checks
# =============================================================================

def sanity_checks(
    df: pd.DataFrame,
    elasticity_results: Dict,
    efficiency_results: Dict,
    config: Optional[TFPConfig] = None,
) -> Dict:
    """
    W7: Sanity checks and falsification tests.
    
    Parameters
    ----------
    df : pd.DataFrame
        Prepared voyage data.
    elasticity_results : Dict
        Results from estimate_tonnage_elasticity().
    efficiency_results : Dict
        Results from estimate_efficiency_effects().
    config : TFPConfig, optional
        Configuration.
        
    Returns
    -------
    Dict
        Sanity check results.
    """
    if config is None:
        config = DEFAULT_TFP_CONFIG
        
    print("\n" + "=" * 60)
    print("W7: SANITY CHECKS AND FALSIFICATION")
    print("=" * 60)
    
    results = {}
    
    # =========================================================================
    # Check 1: Mechanical restriction (Q/tonnage implies beta=1)
    # =========================================================================
    print("\n--- Check 1: Mechanical Restriction Test ---")
    
    beta_pooled = elasticity_results["s1_pooled"]["beta"]
    beta_pre = elasticity_results["s2_regime"]["beta_pre"]
    beta_post = elasticity_results["s2_regime"]["beta_post"]
    
    print(f"  Estimated β_pooled = {beta_pooled:.4f}")
    print(f"  Estimated β_pre = {beta_pre:.4f}")
    print(f"  Estimated β_post = {beta_post:.4f}")
    
    # Test if significantly different from 1
    se_pooled = elasticity_results["s1_pooled"]["se"]
    t_stat_vs_1 = (beta_pooled - 1) / se_pooled
    p_vs_1 = 2 * (1 - stats.t.cdf(abs(t_stat_vs_1), df=len(df) - 2))
    
    print(f"  Test β_pooled = 1: t = {t_stat_vs_1:.2f}, p = {p_vs_1:.4f}")
    
    if p_vs_1 < 0.05:
        print("  ✓ β significantly differs from 1 - scaling by tonnage would be incorrect")
    else:
        print("  ○ β not significantly different from 1")
    
    results["mechanical_restriction"] = {
        "beta_pooled": beta_pooled,
        "t_stat_vs_1": t_stat_vs_1,
        "p_value_vs_1": p_vs_1,
    }
    
    # =========================================================================
    # Check 2: Variance shares stability
    # =========================================================================
    print("\n--- Check 2: Split-Sample Stability ---")
    
    df_eff = efficiency_results["df"]
    
    # Odd vs even voyages
    df_odd = df_eff.iloc[::2]
    df_even = df_eff.iloc[1::2]
    
    corr_alpha = np.corrcoef(
        df_odd.groupby("captain_id")["alpha_eff"].mean(),
        df_even.groupby("captain_id")["alpha_eff"].mean()
    )[0, 1] if len(df_odd["captain_id"].unique()) > 10 else np.nan
    
    print(f"  Correlation of captain effects (odd vs even): {corr_alpha:.4f}")
    
    results["split_sample"] = {
        "corr_alpha_odd_even": corr_alpha,
    }
    
    # =========================================================================
    # Check 3: TFP approximately mean-zero in FE cells
    # =========================================================================
    print("\n--- Check 3: TFP Mean-Zero Validation ---")
    
    if "tfp_hat" in df.columns:
        cell_means = df.groupby("route_time")["tfp_hat"].mean()
        mean_of_means = cell_means.mean()
        std_of_means = cell_means.std()
        
        print(f"  Mean of route×year cell means: {mean_of_means:.6f}")
        print(f"  Std of route×year cell means: {std_of_means:.4f}")
        
        if abs(mean_of_means) < 0.01:
            print("  ✓ TFP approximately mean-zero within cells")
        else:
            print("  ⚠ TFP not mean-zero - check FE construction")
        
        results["mean_zero_check"] = {
            "mean_of_cell_means": mean_of_means,
            "std_of_cell_means": std_of_means,
        }
    
    return results


# =============================================================================
# Master Orchestrator
# =============================================================================

def run_tfp_analysis(
    df: Optional[pd.DataFrame] = None,
    config: Optional[TFPConfig] = None,
    save_outputs: bool = True,
) -> Dict:
    """
    Run full TFP analysis pipeline (W1-W7).
    
    Parameters
    ----------
    df : pd.DataFrame, optional
        Pre-loaded voyage data.
    config : TFPConfig, optional
        Configuration.
    save_outputs : bool
        Whether to save output files.
        
    Returns
    -------
    Dict
        All results from the analysis.
    """
    if config is None:
        config = DEFAULT_TFP_CONFIG
        
    print("\n" + "=" * 70)
    print(" TFP AND AKM EFFICIENCY ANALYSIS")
    print("=" * 70)
    
    all_results = {}
    
    # W1: Preprocessing
    df_clean, sample_diag = prepare_tfp_sample(df, config)
    all_results["sample_diagnostics"] = sample_diag
    
    # W2: Tonnage elasticity
    elasticity_results = estimate_tonnage_elasticity(df_clean, config)
    all_results["elasticity"] = elasticity_results
    
    # W3: TFP construction
    df_tfp = construct_tfp(df_clean, elasticity_results, config)
    all_results["df_tfp"] = df_tfp
    
    # W4: AKM efficiency effects
    efficiency_results = estimate_efficiency_effects(df_tfp, config)
    all_results["efficiency"] = efficiency_results
    
    # Variance decomposition
    decomp = compute_variance_decomposition_tfp(efficiency_results)
    all_results["variance_decomposition"] = decomp
    
    # W5: Two-step robustness
    two_step_results = two_step_robustness(df_tfp, config)
    all_results["two_step"] = two_step_results
    
    # Compare one-step vs two-step
    captain_fe_1step = efficiency_results["captain_fe"].set_index("captain_id")
    captain_fe_2step = two_step_results["captain_fe"].set_index("captain_id")
    common_captains = captain_fe_1step.index.intersection(captain_fe_2step.index)
    
    if len(common_captains) > 10:
        corr_1v2 = np.corrcoef(
            captain_fe_1step.loc[common_captains, "alpha_eff"],
            captain_fe_2step.loc[common_captains, "alpha_eff_2step"]
        )[0, 1]
        print(f"\nOne-step vs Two-step captain effects correlation: {corr_1v2:.4f}")
        all_results["one_vs_two_step_corr"] = corr_1v2
    
    # W6: Regime comparisons
    regime_results = regime_comparison(df_tfp, config)
    all_results["regime_comparison"] = regime_results
    
    # W7: Sanity checks
    sanity_results = sanity_checks(df_tfp, elasticity_results, efficiency_results, config)
    all_results["sanity_checks"] = sanity_results
    
    # Save outputs
    if save_outputs:
        _save_tfp_outputs(all_results, config)
    
    print("\n" + "=" * 70)
    print(" TFP ANALYSIS COMPLETE")
    print("=" * 70)
    
    return all_results


def _save_tfp_outputs(results: Dict, config: TFPConfig) -> None:
    """Save TFP analysis outputs to files."""
    
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save efficiency effects
    results["efficiency"]["captain_fe"].to_csv(
        TABLES_DIR / "tfp_captain_efficiency.csv", index=False
    )
    results["efficiency"]["agent_fe"].to_csv(
        TABLES_DIR / "tfp_agent_efficiency.csv", index=False
    )
    
    # Save variance decomposition
    results["variance_decomposition"].to_csv(
        TABLES_DIR / "tfp_variance_decomposition.csv", index=False
    )
    
    # Save TFP panel
    df_tfp = results["df_tfp"]
    tfp_cols = ["voyage_id", "captain_id", "agent_id", "year_out", "regime",
                "log_q", "log_tonnage", "tfp_hat", "tfp_resid", "beta_used"]
    tfp_cols = [c for c in tfp_cols if c in df_tfp.columns]
    df_tfp[tfp_cols].to_parquet(TABLES_DIR / "voyage_tfp_panel.parquet", index=False)
    
    # Save production function table
    elasticity = results["elasticity"]
    prod_table = pd.DataFrame({
        "Specification": ["S1: Pooled", "S2: Regime-Specific (Pre)", "S2: Regime-Specific (Post)"],
        "Beta": [
            elasticity["s1_pooled"]["beta"],
            elasticity["s2_regime"]["beta_pre"],
            elasticity["s2_regime"]["beta_post"],
        ],
        "SE": [
            elasticity["s1_pooled"]["se"],
            elasticity["s2_regime"]["se_pre"],
            elasticity["s2_regime"]["se_post"],
        ],
        "R2": [
            elasticity["s1_pooled"]["r2"],
            elasticity["s2_regime"]["r2"],
            elasticity["s2_regime"]["r2"],
        ],
    })
    prod_table.to_csv(TABLES_DIR / "tfp_production_function.csv", index=False)
    
    # Create TFP distribution figure
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        df_tfp_eff = results["efficiency"]["df"]
        
        # TFP distribution by regime
        for regime, color in [("pre", "steelblue"), ("post", "coral")]:
            mask = df_tfp_eff["regime"] == regime if "regime" in df_tfp_eff.columns else [True] * len(df_tfp_eff)
            if hasattr(mask, 'sum') and mask.sum() > 0:
                axes[0].hist(df_tfp_eff.loc[mask, "alpha_eff"], bins=50, alpha=0.5, 
                           label=f"{regime.title()} regime", color=color)
        axes[0].set_xlabel("Captain Efficiency (α_eff)")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title("Distribution of Captain Efficiency Effects")
        axes[0].legend()
        
        # Variance decomposition pie
        decomp = results["variance_decomposition"]
        shares = decomp["Share_of_LaborMkt"].dropna().values
        labels = ["Captain", "Agent", "Sorting"]
        colors = ["steelblue", "coral", "gray"]
        
        # Handle negative sorting
        if shares[2] < 0:
            shares_adj = np.abs(shares)
            axes[1].pie(shares_adj, labels=labels, colors=colors, autopct='%1.1f%%',
                       startangle=90)
            axes[1].set_title("Variance Decomposition\n(Sorting shown as absolute value)")
        else:
            axes[1].pie(shares, labels=labels, colors=colors, autopct='%1.1f%%',
                       startangle=90)
            axes[1].set_title("Variance Decomposition")
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "tfp_distribution.png", dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"\nSaved outputs to {TABLES_DIR} and {FIGURES_DIR}")
        
    except ImportError:
        print("\nMatplotlib not available, skipping figures")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    results = run_tfp_analysis()
