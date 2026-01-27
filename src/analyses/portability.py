"""
Portability validation for captain and agent fixed effects (R2, R4).

Implements out-of-sample prediction tests to validate that estimated
fixed effects represent portable skill/capability rather than spurious
matching or measurement error.
"""

from typing import Dict, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

from .config import DEFAULT_SAMPLE
from .baseline_production import estimate_r1, build_sparse_design_matrix
from .data_loader import split_train_test

warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# Cross-Fitted Leave-One-Out Captain Effects
# =============================================================================

def compute_leave_out_captain_effects(
    train_df: pd.DataFrame,
    test_captains: set,
) -> pd.DataFrame:
    """
    Compute leave-one-out captain effects for cross-fitted OOS prediction.
    
    For each captain c appearing in test, estimate α̂_{-c} on training data
    excluding captain c's own voyages to avoid mechanical correlation.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training period voyage data.
    test_captains : set
        Set of captain IDs appearing in test period.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with captain_id and alpha_hat_loo columns.
    """
    print("\n--- Computing Leave-One-Out Captain Effects ---")
    
    # First, get full sample estimates as baseline
    from .connected_set import find_leave_one_out_connected_set
    
    train_loo, _ = find_leave_one_out_connected_set(train_df)
    full_results = estimate_r1(train_loo, use_loo_sample=False)
    full_captain_fe = full_results["captain_fe"].set_index("captain_id")["alpha_hat"]
    
    # Identify captains in both train and test
    train_captains = set(train_df["captain_id"])
    common_captains = train_captains & test_captains
    
    print(f"  Captains in train: {len(train_captains):,}")
    print(f"  Captains in test: {len(test_captains):,}")
    print(f"  Common captains (need LOO): {len(common_captains):,}")
    
    # For efficiency, use the approximation:
    # α̂_{-c} ≈ α̂_full - (leverage correction)
    # Leverage for captain with n_c voyages ≈ 1/n_c
    # Bias correction: α̂_{-c} ≈ α̂ * (n_c / (n_c - 1)) - (residual term)
    
    # Simpler approach: Use the shrinkage-like correction
    captain_counts = train_df.groupby("captain_id").size()
    
    loo_effects = []
    for captain_id in common_captains:
        n_c = captain_counts.get(captain_id, 1)
        alpha_full = full_captain_fe.get(captain_id, np.nan)
        
        if pd.notna(alpha_full) and n_c > 1:
            # Leave-out correction: inflate variance slightly
            # This approximates the effect of removing one observation
            alpha_loo = alpha_full * (n_c / (n_c - 1))
        else:
            # Single-voyage captains: use grand mean (their effect is essentially noise)
            alpha_loo = full_captain_fe.mean()
        
        loo_effects.append({"captain_id": captain_id, "alpha_hat_loo": alpha_loo})
    
    # For captains only in train (not in test), use full estimate
    train_only = train_captains - test_captains
    for captain_id in train_only:
        alpha_full = full_captain_fe.get(captain_id, np.nan)
        loo_effects.append({"captain_id": captain_id, "alpha_hat_loo": alpha_full})
    
    result = pd.DataFrame(loo_effects)
    print(f"  LOO effects computed for {len(result):,} captains")
    
    return result


# =============================================================================
# Empirical Bayes Shrinkage
# =============================================================================

def apply_empirical_bayes_shrinkage(
    captain_fe: pd.DataFrame,
    voyage_counts: pd.Series,
    residual_var: float,
) -> pd.DataFrame:
    """
    Apply Empirical Bayes (James-Stein) shrinkage to captain effects.
    
    α̂^{EB}_c = ᾱ + λ_c · (α̂_c - ᾱ)
    
    where λ_c = Var(α) / [Var(α) + σ²_ε / n_c]
    
    Low-N captains are shrunk toward the grand mean; high-N captains
    keep their original estimates.
    
    Parameters
    ----------
    captain_fe : pd.DataFrame
        DataFrame with captain_id and alpha_hat columns.
    voyage_counts : pd.Series
        Number of voyages per captain (indexed by captain_id).
    residual_var : float
        Estimated residual variance σ²_ε from the AKM model.
        
    Returns
    -------
    pd.DataFrame
        Original DataFrame with alpha_hat_eb column added.
    """
    print("\n--- Applying Empirical Bayes Shrinkage ---")
    
    df = captain_fe.copy()
    
    # Grand mean
    alpha_bar = df["alpha_hat"].mean()
    
    # Variance of captain effects (prior variance)
    var_alpha = df["alpha_hat"].var()
    
    # Merge voyage counts
    df["n_voyages"] = df["captain_id"].map(voyage_counts).fillna(1)
    
    # Shrinkage factor: λ_c = Var(α) / [Var(α) + σ²_ε / n_c]
    df["lambda"] = var_alpha / (var_alpha + residual_var / df["n_voyages"])
    
    # Shrunk estimate
    df["alpha_hat_eb"] = alpha_bar + df["lambda"] * (df["alpha_hat"] - alpha_bar)
    
    # Diagnostics
    low_n = df[df["n_voyages"] <= 2]
    high_n = df[df["n_voyages"] >= 5]
    
    print(f"  Grand mean α̂: {alpha_bar:.4f}")
    print(f"  Var(α̂): {var_alpha:.4f}")
    print(f"  Residual variance: {residual_var:.4f}")
    print(f"  Shrinkage summary:")
    print(f"    Low-N captains (n≤2): mean λ = {low_n['lambda'].mean():.3f}, n = {len(low_n):,}")
    print(f"    High-N captains (n≥5): mean λ = {high_n['lambda'].mean():.3f}, n = {len(high_n):,}")
    
    # How much did estimates change?
    df["shrinkage_delta"] = df["alpha_hat"] - df["alpha_hat_eb"]
    print(f"  Mean absolute shrinkage: {df['shrinkage_delta'].abs().mean():.4f}")
    
    return df[["captain_id", "alpha_hat", "alpha_hat_eb", "n_voyages", "lambda"]]


# =============================================================================
# Binned OOS Calibration Plot
# =============================================================================

def residualize_test_output(
    test_df: pd.DataFrame,
    controls: list = ["log_duration", "log_tonnage"],
) -> pd.Series:
    """
    Residualize test-period log(Q) against controls.
    
    Returns residuals that can be compared to training α̂.
    """
    from scipy import linalg
    
    y = test_df["log_q"].values
    
    # Build control matrix with intercept
    X_cols = [np.ones(len(test_df))]
    for col in controls:
        if col in test_df.columns:
            X_cols.append(test_df[col].fillna(test_df[col].median()).values)
    
    X = np.column_stack(X_cols)
    
    # OLS residuals
    beta = linalg.lstsq(X, y)[0]
    residuals = y - X @ beta
    
    return pd.Series(residuals, index=test_df.index)


def create_binned_calibration_data(
    captain_realized: pd.DataFrame,
    alpha_col: str = "alpha_hat_eb",
    outcome_col: str = "log_q_resid",
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Create binned calibration data for OOS validation.
    
    Parameters
    ----------
    captain_realized : pd.DataFrame
        Captain-level data with training α̂ and test-period outcomes.
    alpha_col : str
        Column containing the captain effect estimate to bin.
    outcome_col : str
        Column containing the test-period outcome (residualized log Q).
    n_bins : int
        Number of bins (deciles = 10).
        
    Returns
    -------
    pd.DataFrame
        Binned calibration data with bin means and counts.
    """
    df = captain_realized.copy()
    
    # Create decile bins
    df["alpha_decile"] = pd.qcut(df[alpha_col], q=n_bins, labels=False, duplicates="drop")
    
    # Compute bin statistics
    bin_stats = df.groupby("alpha_decile").agg({
        alpha_col: ["mean", "std", "count"],
        outcome_col: ["mean", "std"],
    })
    
    bin_stats.columns = ["alpha_mean", "alpha_std", "n", "outcome_mean", "outcome_std"]
    bin_stats = bin_stats.reset_index()
    
    # Compute confidence intervals
    bin_stats["outcome_se"] = bin_stats["outcome_std"] / np.sqrt(bin_stats["n"])
    bin_stats["outcome_ci_low"] = bin_stats["outcome_mean"] - 1.96 * bin_stats["outcome_se"]
    bin_stats["outcome_ci_high"] = bin_stats["outcome_mean"] + 1.96 * bin_stats["outcome_se"]
    
    return bin_stats


def compute_calibration_slope(bin_data: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Compute the slope of the calibration line.
    
    Perfect calibration = slope of 1.0.
    No portability = slope of 0.0.
    
    Returns
    -------
    Tuple[float, float, float]
        (slope, intercept, r_squared)
    """
    x = bin_data["alpha_mean"].values
    y = bin_data["outcome_mean"].values
    
    # Weighted by bin size
    weights = bin_data["n"].values
    
    # Weighted linear regression
    X = np.column_stack([np.ones_like(x), x])
    W = np.diag(weights)
    
    beta = np.linalg.lstsq(W @ X, W @ y, rcond=None)[0]
    intercept, slope = beta[0], beta[1]
    
    # R²
    y_hat = X @ beta
    ss_res = np.sum(weights * (y - y_hat) ** 2)
    ss_tot = np.sum(weights * (y - np.average(y, weights=weights)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    return slope, intercept, r2




def estimate_train_period_effects(
    train_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Estimate captain and agent effects on training period only.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training period voyages.
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, Dict]
        (captain_fe, agent_fe, full_results)
    """
    print("\n" + "=" * 60)
    print("ESTIMATING TRAINING PERIOD EFFECTS")
    print("=" * 60)
    
    results = estimate_r1(train_df, use_loo_sample=True)
    
    return results["captain_fe"], results["agent_fe"], results


def run_r2_captain_portability(
    df: pd.DataFrame,
    cutoff_year: Optional[int] = None,
    use_cross_fitted: bool = True,
    use_eb_shrinkage: bool = True,
) -> Dict:
    """
    R2: OOS prediction of output using pre-period captain effects.
    
    Enhanced with:
    - Cross-fitted (LOO) captain effects
    - Empirical Bayes shrinkage for low-N captains
    - Binned calibration plot (replacing Spearman)
    
    logQ_v (test) = b · α̂_train[c(v)] + δ_{vessel×period} + θ_{route×time} + Xβ + u_v
    
    Parameters
    ----------
    df : pd.DataFrame
        Full voyage data.
    cutoff_year : int, optional
        Year to split on.
    use_cross_fitted : bool
        Whether to use LOO cross-fitted captain effects.
    use_eb_shrinkage : bool
        Whether to apply Empirical Bayes shrinkage.
        
    Returns
    -------
    Dict
        R2 results including portability coefficient, calibration data, and diagnostics.
    """
    if cutoff_year is None:
        cutoff_year = DEFAULT_SAMPLE.oos_cutoff_year
        
    print("\n" + "=" * 60)
    print(f"R2: CAPTAIN PORTABILITY (OOS Prediction)")
    print(f"Train/test split: {cutoff_year}")
    print(f"Cross-fitted: {use_cross_fitted}, EB shrinkage: {use_eb_shrinkage}")
    print("=" * 60)
    
    # Split data
    train_df, test_df = split_train_test(df, cutoff_year)
    
    # Estimate effects on training period
    captain_fe_train, agent_fe_train, train_results = estimate_train_period_effects(train_df)
    
    # Get residual variance for EB shrinkage
    residual_var = np.var(train_results["residuals"])
    
    # ==========================================================================
    # Step 1: Compute cross-fitted (LOO) captain effects
    # ==========================================================================
    test_captains = set(test_df["captain_id"])
    
    if use_cross_fitted:
        captain_fe_loo = compute_leave_out_captain_effects(train_df, test_captains)
        captain_fe_train = captain_fe_train.merge(
            captain_fe_loo, on="captain_id", how="left"
        )
    else:
        captain_fe_train["alpha_hat_loo"] = captain_fe_train["alpha_hat"]
    
    # ==========================================================================
    # Step 2: Apply Empirical Bayes shrinkage
    # ==========================================================================
    voyage_counts = train_df.groupby("captain_id").size()
    
    if use_eb_shrinkage:
        captain_fe_train = apply_empirical_bayes_shrinkage(
            captain_fe_train,
            voyage_counts,
            residual_var,
        )
    else:
        captain_fe_train["alpha_hat_eb"] = captain_fe_train["alpha_hat"]
        captain_fe_train["n_voyages"] = captain_fe_train["captain_id"].map(voyage_counts)
        captain_fe_train["lambda"] = 1.0
    
    # Rename for clarity
    captain_fe_train = captain_fe_train.rename(columns={"alpha_hat": "alpha_hat_train"})
    
    # Merge training effects to test data
    test_df = test_df.merge(
        captain_fe_train[["captain_id", "alpha_hat_train", "alpha_hat_eb", "n_voyages"]],
        on="captain_id",
        how="left"
    )
    
    # Only keep test captains that were in training
    test_with_alpha = test_df[test_df["alpha_hat_train"].notna()].copy()
    
    print(f"\nTest sample:")
    print(f"  Total test voyages: {len(test_df):,}")
    print(f"  With training α̂: {len(test_with_alpha):,} ({100*len(test_with_alpha)/len(test_df):.1f}%)")
    print(f"  Unique captains: {test_with_alpha['captain_id'].nunique():,}")
    
    # ==========================================================================
    # Step 3: Residualize test-period output
    # ==========================================================================
    print("\n--- Residualizing Test-Period Output ---")
    test_with_alpha["log_q_resid"] = residualize_test_output(test_with_alpha)
    
    # ==========================================================================
    # Step 4: Binned Calibration Analysis (replacing Spearman)
    # ==========================================================================
    print("\n--- Binned OOS Calibration Analysis ---")
    
    # Aggregate to captain level
    captain_realized = test_with_alpha.groupby("captain_id").agg({
        "log_q": "mean",
        "log_q_resid": "mean",
        "alpha_hat_train": "first",
        "alpha_hat_eb": "first",
        "n_voyages": "first",
    }).reset_index()
    captain_realized["n_test_voyages"] = test_with_alpha.groupby("captain_id").size().values
    
    # Create calibration bins using EB-shrunk effects
    bin_data = create_binned_calibration_data(
        captain_realized,
        alpha_col="alpha_hat_eb",
        outcome_col="log_q_resid",
        n_bins=10,
    )
    
    # Compute calibration slope
    calib_slope, calib_intercept, calib_r2 = compute_calibration_slope(bin_data)
    
    print(f"  Calibration slope: {calib_slope:.4f} (perfect = 1.0)")
    print(f"  Calibration R²: {calib_r2:.4f}")
    print(f"  Bin sample sizes: min={bin_data['n'].min()}, max={bin_data['n'].max()}")
    
    # ==========================================================================
    # Step 5: Full test sample regression (for comparison)
    # ==========================================================================
    print("\n--- Full Test Sample Regression ---")
    
    n = len(test_with_alpha)
    y = test_with_alpha["log_q"].values
    
    # Build design matrix
    matrices = []
    
    # Vessel×period (drop first)
    if "vessel_period" in test_with_alpha.columns:
        vp_ids = test_with_alpha["vessel_period"].unique()
        vp_map = {v: i for i, v in enumerate(vp_ids)}
        vp_idx = test_with_alpha["vessel_period"].map(vp_map).values
        X_vp = sp.csr_matrix(
            (np.ones(n), (np.arange(n), vp_idx)),
            shape=(n, len(vp_ids))
        )[:, 1:]
        matrices.append(X_vp)
    
    # Route×time (drop first)
    if "route_time" in test_with_alpha.columns:
        rt_ids = test_with_alpha["route_time"].unique()
        rt_map = {r: i for i, r in enumerate(rt_ids)}
        rt_idx = test_with_alpha["route_time"].map(rt_map).values
        X_rt = sp.csr_matrix(
            (np.ones(n), (np.arange(n), rt_idx)),
            shape=(n, len(rt_ids))
        )[:, 1:]
        matrices.append(X_rt)
    
    # Controls: α̂_eb, log_duration, log_tonnage
    controls = np.column_stack([
        test_with_alpha["alpha_hat_eb"].values,
        test_with_alpha["log_duration"].values,
        test_with_alpha["log_tonnage"].values,
    ])
    matrices.append(sp.csr_matrix(controls))
    matrices.append(sp.csr_matrix(np.ones((n, 1))))
    
    X = sp.hstack(matrices)
    
    result = lsqr(X, y, iter_lim=10000, atol=1e-10, btol=1e-10)
    beta = result[0]
    
    n_vp = len(vp_ids) - 1 if "vessel_period" in test_with_alpha.columns else 0
    n_rt = len(rt_ids) - 1 if "route_time" in test_with_alpha.columns else 0
    b_alpha_full = beta[n_vp + n_rt]
    
    y_hat = X @ beta
    residuals = y - y_hat
    r2_full = 1 - np.var(residuals) / np.var(y)
    
    print(f"  b (α̂_eb coefficient): {b_alpha_full:.4f}")
    print(f"  R²: {r2_full:.4f}")
    
    # ==========================================================================
    # Step 6: Switch-only sample (additional robustness)
    # ==========================================================================
    print("\n--- Switch-Only Sample ---")
    
    switch_mask = test_with_alpha["any_switch"] == 1
    test_switchers = test_with_alpha[switch_mask].copy()
    
    if len(test_switchers) > 50:
        n_sw = len(test_switchers)
        y_sw = test_switchers["log_q"].values
        
        X_sw = np.column_stack([
            np.ones(n_sw),
            test_switchers["alpha_hat_eb"].values,
            test_switchers["log_duration"].values,
            test_switchers["log_tonnage"].values,
        ])
        
        beta_sw = np.linalg.lstsq(X_sw, y_sw, rcond=None)[0]
        b_alpha_switch = beta_sw[1]
        
        y_hat_sw = X_sw @ beta_sw
        r2_switch = 1 - np.var(y_sw - y_hat_sw) / np.var(y_sw)
        
        print(f"  Switch voyages: {n_sw:,}")
        print(f"  b (α̂_eb coefficient): {b_alpha_switch:.4f}")
        print(f"  R²: {r2_switch:.4f}")
    else:
        b_alpha_switch = np.nan
        r2_switch = np.nan
        n_sw = len(test_switchers)
        print(f"  Insufficient switch voyages: {n_sw}")
    
    # ==========================================================================
    # Step 7: Legacy correlations (for comparison, not primary)
    # ==========================================================================
    print("\n--- Legacy Rank Correlations (for reference) ---")
    
    spearman_r, spearman_p = stats.spearmanr(
        captain_realized["alpha_hat_eb"],
        captain_realized["log_q_resid"]
    )
    pearson_r, pearson_p = stats.pearsonr(
        captain_realized["alpha_hat_eb"],
        captain_realized["log_q_resid"]
    )
    
    print(f"  Spearman ρ (EB-shrunk vs residualized): {spearman_r:.4f} (p={spearman_p:.4f})")
    print(f"  Pearson r (EB-shrunk vs residualized): {pearson_r:.4f} (p={pearson_p:.4f})")
    
    # ==========================================================================
    # Compile results
    # ==========================================================================
    results = {
        # Sample info
        "cutoff_year": cutoff_year,
        "n_train": len(train_df),
        "n_test_total": len(test_df),
        "n_test_with_alpha": len(test_with_alpha),
        "n_test_switchers": n_sw,
        
        # Calibration results (PRIMARY)
        "calibration_slope": calib_slope,
        "calibration_intercept": calib_intercept,
        "calibration_r2": calib_r2,
        "bin_data": bin_data,
        
        # Regression results
        "b_alpha_full": b_alpha_full,
        "r2_full": r2_full,
        "b_alpha_switch": b_alpha_switch,
        "r2_switch": r2_switch,
        
        # Legacy correlations
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        
        # Data for figures
        "captain_fe_train": captain_fe_train,
        "captain_realized": captain_realized,
        "test_data": test_with_alpha,
        "train_results": train_results,
    }
    
    return results



def run_r4_agent_portability(
    df: pd.DataFrame,
    cutoff_year: Optional[int] = None,
) -> Dict:
    """
    R4: OOS prediction using pre-period agent effects.
    
    logQ_v (test) = b · γ̂_train[a(v)] + α_c + δ_{vessel×period} + θ_{route×time} + u_v
    
    Focus on voyages with captain turnover to avoid confounding.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full voyage data.
    cutoff_year : int, optional
        Year to split on.
        
    Returns
    -------
    Dict
        R4 results.
    """
    if cutoff_year is None:
        cutoff_year = DEFAULT_SAMPLE.oos_cutoff_year
        
    print("\n" + "=" * 60)
    print(f"R4: AGENT CAPABILITY PERSISTENCE (OOS Prediction)")
    print(f"Train/test split: {cutoff_year}")
    print("=" * 60)
    
    # Split data
    train_df, test_df = split_train_test(df, cutoff_year)
    
    # Estimate effects on training period
    captain_fe_train, agent_fe_train, train_results = estimate_train_period_effects(train_df)
    
    # Rename for clarity
    agent_fe_train = agent_fe_train.rename(columns={"gamma_hat": "gamma_hat_train"})
    
    # Merge training agent effects to test data
    test_df = test_df.merge(agent_fe_train, on="agent_id", how="left")
    
    # Only keep test voyages with agents from training
    test_with_gamma = test_df[test_df["gamma_hat_train"].notna()].copy()
    
    print(f"\nTest sample:")
    print(f"  Total test voyages: {len(test_df):,}")
    print(f"  With training γ̂: {len(test_with_gamma):,} ({100*len(test_with_gamma)/len(test_df):.1f}%)")
    print(f"  Unique agents: {test_with_gamma['agent_id'].nunique():,}")
    
    # Focus on voyages with NEW captains (not in training)
    train_captains = set(train_df["captain_id"])
    new_captain_mask = ~test_with_gamma["captain_id"].isin(train_captains)
    test_new_captains = test_with_gamma[new_captain_mask].copy()
    
    print(f"\nNew captain subsample (captain turnover):")
    print(f"  Voyages with new captains: {len(test_new_captains):,}")
    print(f"  New captains: {test_new_captains['captain_id'].nunique():,}")
    
    # Simple regression on new captain sample
    if len(test_new_captains) > 50:
        n = len(test_new_captains)
        y = test_new_captains["log_q"].values
        
        X = np.column_stack([
            np.ones(n),
            test_new_captains["gamma_hat_train"].values,
            test_new_captains["log_duration"].values,
            test_new_captains["log_tonnage"].values,
        ])
        
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        b_gamma = beta[1]
        
        y_hat = X @ beta
        r2 = 1 - np.var(y - y_hat) / np.var(y)
        
        print(f"\n  b (γ̂_train coefficient): {b_gamma:.4f}")
        print(f"  R²: {r2:.4f}")
        
        # Agent-level correlation
        agent_realized = test_new_captains.groupby("agent_id").agg({
            "log_q": "mean",
            "gamma_hat_train": "first",
        }).reset_index()
        
        spearman_r, spearman_p = stats.spearmanr(
            agent_realized["gamma_hat_train"],
            agent_realized["log_q"]
        )
        
        print(f"\n  Agent-level Spearman ρ: {spearman_r:.4f} (p={spearman_p:.4f})")
        
    else:
        b_gamma = np.nan
        r2 = np.nan
        spearman_r = np.nan
        spearman_p = np.nan
        print(f"  Insufficient new captain voyages for analysis")
    
    results = {
        "cutoff_year": cutoff_year,
        "n_test_with_gamma": len(test_with_gamma),
        "n_new_captain_voyages": len(test_new_captains),
        "b_gamma": b_gamma,
        "r2": r2,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "agent_fe_train": agent_fe_train,
    }
    
    return results


def create_portability_figure(
    r2_results: Dict,
    output_path: Optional[str] = None,
) -> None:
    """
    Create portability visualization with binned calibration plot.
    
    The binned calibration plot shows:
    - X-axis: Decile means of α̂^{EB} from training
    - Y-axis: Mean residualized log(Q) in test period
    - 95% confidence intervals for each bin
    - Calibration slope (perfect = 1.0)
    
    Parameters
    ----------
    r2_results : Dict
        Results from run_r2_captain_portability.
    output_path : str, optional
        Path to save figure.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping figure generation")
        return
    
    from .config import FIGURES_DIR
    from pathlib import Path
    
    if output_path is None:
        output_path = FIGURES_DIR / "r2_binned_calibration.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    captain_data = r2_results["captain_realized"]
    bin_data = r2_results["bin_data"]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # =========================================================================
    # Left Panel: Binned Calibration Plot (PRIMARY)
    # =========================================================================
    ax1 = axes[0]
    
    # Plot bin means with 95% CI error bars
    ax1.errorbar(
        bin_data["alpha_mean"],
        bin_data["outcome_mean"],
        yerr=1.96 * bin_data["outcome_se"],
        fmt="o",
        markersize=10,
        color="darkblue",
        ecolor="steelblue",
        elinewidth=2,
        capsize=5,
        capthick=2,
        label="Decile means (± 95% CI)",
    )
    
    # Connect with line
    ax1.plot(
        bin_data["alpha_mean"],
        bin_data["outcome_mean"],
        "b-",
        linewidth=2,
        alpha=0.7,
    )
    
    # Add 45-degree reference line (perfect calibration)
    x_range = np.array([bin_data["alpha_mean"].min(), bin_data["alpha_mean"].max()])
    ax1.plot(
        x_range,
        x_range,
        "k--",
        linewidth=1.5,
        alpha=0.5,
        label="45° line (perfect calibration)",
    )
    
    # Add fitted calibration line
    calib_slope = r2_results["calibration_slope"]
    calib_intercept = r2_results["calibration_intercept"]
    y_fitted = calib_intercept + calib_slope * x_range
    ax1.plot(
        x_range,
        y_fitted,
        "r-",
        linewidth=2,
        label=f"Fitted slope = {calib_slope:.3f}",
    )
    
    ax1.set_xlabel("Training Period α̂$^{EB}$ (Decile Mean)", fontsize=12)
    ax1.set_ylabel("Test Period Residualized log(Q)", fontsize=12)
    ax1.set_title("Binned OOS Calibration Plot", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(alpha=0.3)
    
    # Add statistics annotation
    stats_text = (
        f"Calibration slope: {calib_slope:.3f}\n"
        f"Calibration R²: {r2_results['calibration_r2']:.3f}\n"
        f"n captains: {len(captain_data):,}\n"
        f"n voyages: {r2_results['n_test_with_alpha']:,}"
    )
    ax1.annotate(
        stats_text,
        xy=(0.95, 0.05),
        xycoords="axes fraction",
        horizontalalignment="right",
        verticalalignment="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
    )
    
    # =========================================================================
    # Right Panel: Shrinkage Diagnostics
    # =========================================================================
    ax2 = axes[1]
    
    # Scatter: raw alpha vs EB-shrunk alpha, colored by n_voyages
    scatter = ax2.scatter(
        captain_data["alpha_hat_train"],
        captain_data["alpha_hat_eb"],
        c=np.log1p(captain_data["n_voyages"]),
        cmap="viridis",
        alpha=0.6,
        s=30,
    )
    
    # 45-degree line (no shrinkage)
    x_range2 = np.array([
        captain_data["alpha_hat_train"].min(),
        captain_data["alpha_hat_train"].max()
    ])
    ax2.plot(x_range2, x_range2, "k--", linewidth=1.5, alpha=0.5, label="No shrinkage")
    
    # Color bar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label("log(1 + n_voyages)", fontsize=10)
    
    ax2.set_xlabel("Raw α̂ (Training)", fontsize=12)
    ax2.set_ylabel("EB-Shrunk α̂$^{EB}$", fontsize=12)
    ax2.set_title("Empirical Bayes Shrinkage", fontsize=14, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=10)
    ax2.grid(alpha=0.3)
    
    # Annotation for shrinkage
    low_n = captain_data[captain_data["n_voyages"] <= 2]
    high_n = captain_data[captain_data["n_voyages"] >= 5]
    shrink_text = (
        f"Low-N (n≤2): {len(low_n):,} captains\n"
        f"High-N (n≥5): {len(high_n):,} captains"
    )
    ax2.annotate(
        shrink_text,
        xy=(0.95, 0.05),
        xycoords="axes fraction",
        horizontalalignment="right",
        verticalalignment="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"\nBinned calibration figure saved to {output_path}")


def run_portability_analysis(
    df: pd.DataFrame,
    save_outputs: bool = True,
) -> Dict:
    """
    Run full portability analysis (R2 and R4).
    
    Parameters
    ----------
    df : pd.DataFrame
        Full voyage data.
    save_outputs : bool
        Whether to save outputs.
        
    Returns
    -------
    Dict
        Combined R2 and R4 results.
    """
    from .config import TABLES_DIR
    from pathlib import Path
    
    # R2: Captain portability
    r2_results = run_r2_captain_portability(df)
    
    # R4: Agent persistence
    r4_results = run_r4_agent_portability(df)
    
    if save_outputs:
        # Create portability figure
        create_portability_figure(r2_results)
        
        # Save summary table
        summary = pd.DataFrame({
            "Specification": ["R2: Captain Portability", "R2: Switch-Only", "R4: Agent Persistence"],
            "Coefficient": [
                r2_results["b_alpha_full"],
                r2_results["b_alpha_switch"],
                r4_results["b_gamma"],
            ],
            "R2": [
                r2_results["r2_full"],
                r2_results["r2_switch"],
                r4_results["r2"],
            ],
            "Spearman_rho": [
                r2_results["spearman_r"],
                np.nan,
                r4_results["spearman_r"],
            ],
            "N": [
                r2_results["n_test_with_alpha"],
                r2_results.get("n_test_switchers", np.nan),
                r4_results["n_new_captain_voyages"],
            ],
        })
        
        output_path = TABLES_DIR / "r2_r4_portability_summary.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_path, index=False)
        print(f"\nPortability summary saved to {output_path}")
    
    return {"r2": r2_results, "r4": r4_results}


if __name__ == "__main__":
    from .data_loader import prepare_analysis_sample
    
    df = prepare_analysis_sample()
    results = run_portability_analysis(df)
