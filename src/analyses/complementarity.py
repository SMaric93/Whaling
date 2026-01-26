"""
Complementarity and resilience analysis (R5, R6).

Implements:
- R5: Skill-capability complementarity (α × γ interactions)
- R6: Resilience under adversity (adverse conditions × agent capability)
"""

from typing import Dict, Optional
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

from .config import DEFAULT_SAMPLE
from .baseline_production import estimate_r1

warnings.filterwarnings("ignore", category=FutureWarning)


def compute_leave_one_out_effects(
    df: pd.DataFrame,
    group_col: str,
    effect_col: str,
) -> pd.DataFrame:
    """
    Compute leave-one-out estimates of fixed effects to avoid mechanical bias.
    
    For each observation, compute the mean effect excluding that observation.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with effect estimates.
    group_col : str
        Grouping variable (e.g., captain_id).
    effect_col : str  
        Effect column (e.g., alpha_hat).
        
    Returns
    -------
    pd.DataFrame
        Data with LOO effect estimates.
    """
    df = df.copy()
    
    # Group sums and counts
    group_sums = df.groupby(group_col)[effect_col].transform("sum")
    group_counts = df.groupby(group_col)[effect_col].transform("count")
    
    # LOO mean = (sum - current) / (count - 1)
    df[f"{effect_col}_loo"] = (group_sums - df[effect_col]) / (group_counts - 1)
    
    # For singletons, use the original estimate
    singleton_mask = group_counts == 1
    df.loc[singleton_mask, f"{effect_col}_loo"] = df.loc[singleton_mask, effect_col]
    
    return df


def run_r5_complementarity(
    df: pd.DataFrame,
    use_loo: bool = True,
) -> Dict:
    """
    R5: Skill-capability complementarity.
    
    logQ_v = α_c + γ_a + η·(α̂_c × γ̂_a) + δ_{vessel×period} + θ_{route×time} + ε
    
    Parameters
    ----------
    df : pd.DataFrame
        Voyage data.
    use_loo : bool
        Whether to use leave-one-out effects for interaction.
        
    Returns
    -------
    Dict
        R5 results including complementarity coefficient.
    """
    print("\n" + "=" * 60)
    print("R5: SKILL-CAPABILITY COMPLEMENTARITY")
    print("=" * 60)
    
    # First estimate baseline to get FE estimates
    r1_results = estimate_r1(df, use_loo_sample=True)
    df_est = r1_results["df"]
    
    print(f"\nSample: {len(df_est):,} voyages")
    
    # Use LOO effects if requested
    if use_loo:
        df_est = compute_leave_one_out_effects(df_est, "captain_id", "alpha_hat")
        df_est = compute_leave_one_out_effects(df_est, "agent_id", "gamma_hat")
        alpha_col = "alpha_hat_loo"
        gamma_col = "gamma_hat_loo"
        print("Using leave-one-out effect estimates")
    else:
        alpha_col = "alpha_hat"
        gamma_col = "gamma_hat"
    
    # Create interaction term
    df_est["alpha_x_gamma"] = df_est[alpha_col] * df_est[gamma_col]
    
    # Also create quartile-based interaction
    df_est["alpha_q"] = pd.qcut(df_est[alpha_col], q=4, labels=[1, 2, 3, 4])
    df_est["gamma_q"] = pd.qcut(df_est[gamma_col], q=4, labels=[1, 2, 3, 4])
    df_est["top_alpha"] = (df_est["alpha_q"] == 4).astype(int)
    df_est["top_gamma"] = (df_est["gamma_q"] == 4).astype(int)
    df_est["top_both"] = df_est["top_alpha"] * df_est["top_gamma"]
    
    # Build design matrix
    n = len(df_est)
    y = df_est["log_q"].values
    
    matrices = []
    
    # Captain FEs
    captain_ids = df_est["captain_id"].unique()
    captain_map = {c: i for i, c in enumerate(captain_ids)}
    captain_idx = df_est["captain_id"].map(captain_map).values
    matrices.append(sp.csr_matrix(
        (np.ones(n), (np.arange(n), captain_idx)),
        shape=(n, len(captain_ids))
    ))
    
    # Agent FEs (drop first)
    agent_ids = df_est["agent_id"].unique()
    agent_map = {a: i for i, a in enumerate(agent_ids)}
    agent_idx = df_est["agent_id"].map(agent_map).values
    X_agent = sp.csr_matrix(
        (np.ones(n), (np.arange(n), agent_idx)),
        shape=(n, len(agent_ids))
    )[:, 1:]
    matrices.append(X_agent)
    
    # Vessel×period FEs (drop first)
    if "vessel_period" in df_est.columns:
        vp_ids = df_est["vessel_period"].unique()
        vp_map = {v: i for i, v in enumerate(vp_ids)}
        vp_idx = df_est["vessel_period"].map(vp_map).values
        matrices.append(sp.csr_matrix(
            (np.ones(n), (np.arange(n), vp_idx)),
            shape=(n, len(vp_ids))
        )[:, 1:])
    
    # Controls + interaction
    controls = np.column_stack([
        df_est["log_duration"].values,
        df_est["log_tonnage"].values,
        df_est["alpha_x_gamma"].values,  # The complementarity term
    ])
    matrices.append(sp.csr_matrix(controls))
    
    X = sp.hstack(matrices)
    
    # Solve
    result = lsqr(X, y, iter_lim=10000, atol=1e-10, btol=1e-10)
    beta = result[0]
    
    # The interaction coefficient is the last control
    eta = beta[-1]  # alpha_x_gamma coefficient
    
    # Compute residuals and R²
    y_hat = X @ beta
    residuals = y - y_hat
    r2 = 1 - np.var(residuals) / np.var(y)
    
    # Approximate SE for eta
    n_params = X.shape[1]
    sigma2 = np.sum(residuals**2) / (n - n_params)
    # Use simpler SE approximation
    var_interaction = np.var(df_est["alpha_x_gamma"])
    se_eta = np.sqrt(sigma2 / (n * var_interaction))
    t_stat = eta / se_eta
    
    print(f"\n--- Complementarity Results ---")
    print(f"η (α × γ interaction): {eta:.4f}")
    print(f"  SE: {se_eta:.4f}")
    print(f"  t-stat: {t_stat:.2f}")
    print(f"R²: {r2:.4f}")
    
    if eta > 0:
        print("\nInterpretation: Positive η suggests complementarity between captain skill and agent capability")
    else:
        print("\nInterpretation: Non-positive η suggests no complementarity (or substitutability)")
    
    # Quartile analysis
    print("\n--- Quartile Analysis ---")
    quartile_means = df_est.groupby(["alpha_q", "gamma_q"])["log_q"].mean().unstack()
    print("Mean log(Q) by skill × capability quartiles:")
    print(quartile_means.round(3).to_string())
    
    results = {
        "eta": eta,
        "se_eta": se_eta,
        "t_stat": t_stat,
        "r2": r2,
        "n": n,
        "quartile_means": quartile_means,
        "use_loo": use_loo,
    }
    
    return results


def run_r6_resilience(
    df: pd.DataFrame,
    adversity_col: str = "arctic_route",
) -> Dict:
    """
    R6: Resilience under adverse conditions.
    
    logQ_v = α_c + γ_a + b·Z_v + φ·(Z_v × HighCapAgent) + δ_{vessel×period} + θ_{route×time} + ε
    
    Parameters
    ----------
    df : pd.DataFrame
        Voyage data.
    adversity_col : str
        Column representing adverse conditions (e.g., arctic_route, ice_anomaly).
        
    Returns
    -------
    Dict
        R6 results including resilience coefficient.
    """
    print("\n" + "=" * 60)
    print("R6: RESILIENCE UNDER ADVERSITY")
    print(f"Adversity measure: {adversity_col}")
    print("=" * 60)
    
    # First estimate baseline to get agent capability estimates
    r1_results = estimate_r1(df, use_loo_sample=True)
    df_est = r1_results["df"]
    
    # Check if adversity column exists
    if adversity_col not in df_est.columns:
        print(f"Warning: {adversity_col} not in data, creating proxy")
        if "route_or_ground" in df_est.columns:
            arctic_keywords = ["arctic", "bering", "hudson", "bowhead", "polar", "ice"]
            df_est[adversity_col] = df_est["route_or_ground"].str.lower().str.contains(
                "|".join(arctic_keywords), na=False
            ).astype(int)
        else:
            df_est[adversity_col] = 0
    
    print(f"\nSample: {len(df_est):,} voyages")
    print(f"Adverse conditions: {df_est[adversity_col].sum():,} ({100*df_est[adversity_col].mean():.1f}%)")
    
    # Define high-capability agents (top quartile)
    gamma_75 = df_est["gamma_hat"].quantile(0.75)
    df_est["high_cap_agent"] = (df_est["gamma_hat"] >= gamma_75).astype(int)
    print(f"High-capability agents (top 25%): {df_est['high_cap_agent'].sum():,} voyages")
    
    # Interaction term
    df_est["Z_x_high_cap"] = df_est[adversity_col] * df_est["high_cap_agent"]
    
    # Build design matrix
    n = len(df_est)
    y = df_est["log_q"].values
    
    matrices = []
    
    # Captain FEs
    captain_ids = df_est["captain_id"].unique()
    captain_map = {c: i for i, c in enumerate(captain_ids)}
    captain_idx = df_est["captain_id"].map(captain_map).values
    matrices.append(sp.csr_matrix(
        (np.ones(n), (np.arange(n), captain_idx)),
        shape=(n, len(captain_ids))
    ))
    
    # Agent FEs (drop first)
    agent_ids = df_est["agent_id"].unique()
    agent_map = {a: i for i, a in enumerate(agent_ids)}
    agent_idx = df_est["agent_id"].map(agent_map).values
    matrices.append(sp.csr_matrix(
        (np.ones(n), (np.arange(n), agent_idx)),
        shape=(n, len(agent_ids))
    )[:, 1:])
    
    # Vessel×period FEs (drop first)
    if "vessel_period" in df_est.columns:
        vp_ids = df_est["vessel_period"].unique()
        vp_map = {v: i for i, v in enumerate(vp_ids)}
        vp_idx = df_est["vessel_period"].map(vp_map).values
        matrices.append(sp.csr_matrix(
            (np.ones(n), (np.arange(n), vp_idx)),
            shape=(n, len(vp_ids))
        )[:, 1:])
    
    # Controls: Z, Z × high_cap, log_duration, log_tonnage
    controls = np.column_stack([
        df_est["log_duration"].values,
        df_est["log_tonnage"].values,
        df_est[adversity_col].values,      # Z (adversity)
        df_est["Z_x_high_cap"].values,     # Z × HighCapAgent
    ])
    matrices.append(sp.csr_matrix(controls))
    
    X = sp.hstack(matrices)
    
    # Solve
    result = lsqr(X, y, iter_lim=10000, atol=1e-10, btol=1e-10)
    beta = result[0]
    
    # Coefficients (last 2 controls are Z and Z×high_cap)
    b_Z = beta[-2]      # Adversity main effect
    phi = beta[-1]      # Resilience interaction
    
    # Compute residuals and R²
    y_hat = X @ beta
    residuals = y - y_hat
    r2 = 1 - np.var(residuals) / np.var(y)
    
    # Approximate SEs
    n_params = X.shape[1]
    sigma2 = np.sum(residuals**2) / (n - n_params)
    
    print(f"\n--- Resilience Results ---")
    print(f"b (adversity main effect): {b_Z:.4f}")
    print(f"φ (Z × HighCapAgent): {phi:.4f}")
    print(f"R²: {r2:.4f}")
    
    # Marginal effects
    effect_low_cap = b_Z
    effect_high_cap = b_Z + phi
    
    print(f"\nMarginal effect of adversity:")
    print(f"  Low-capability agents: {effect_low_cap:.4f}")
    print(f"  High-capability agents: {effect_high_cap:.4f}")
    
    if phi > 0:
        print("\nInterpretation: φ > 0 means high-capability agents ATTENUATE output losses under adversity")
    elif phi < 0:
        print("\nInterpretation: φ < 0 means high-capability agents face LARGER losses under adversity")
    
    # Subgroup means
    print("\n--- Subgroup Analysis ---")
    subgroup_means = df_est.groupby([adversity_col, "high_cap_agent"])["log_q"].agg(["mean", "std", "count"])
    print(subgroup_means.round(3).to_string())
    
    results = {
        "b_Z": b_Z,
        "phi": phi,
        "r2": r2,
        "n": n,
        "adversity_col": adversity_col,
        "effect_low_cap": effect_low_cap,
        "effect_high_cap": effect_high_cap,
        "subgroup_means": subgroup_means,
    }
    
    return results


def run_complementarity_analysis(
    df: pd.DataFrame,
    save_outputs: bool = True,
) -> Dict:
    """
    Run full complementarity and resilience analysis (R5, R6).
    
    Parameters
    ----------
    df : pd.DataFrame
        Voyage data.
    save_outputs : bool
        Whether to save outputs.
        
    Returns
    -------
    Dict
        Combined R5 and R6 results.
    """
    from .config import TABLES_DIR
    from pathlib import Path
    
    # R5: Complementarity
    r5_results = run_r5_complementarity(df)
    
    # R6: Resilience
    r6_results = run_r6_resilience(df)
    
    if save_outputs:
        output_path = TABLES_DIR / "r5_r6_complementarity_resilience.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        summary = pd.DataFrame({
            "Specification": ["R5: Complementarity (η)", "R6: Resilience (φ)"],
            "Coefficient": [r5_results["eta"], r6_results["phi"]],
            "R2": [r5_results["r2"], r6_results["r2"]],
            "N": [r5_results["n"], r6_results["n"]],
        })
        summary.to_csv(output_path, index=False)
        print(f"\nComplementarity/resilience summary saved to {output_path}")
    
    return {"r5": r5_results, "r6": r6_results}


if __name__ == "__main__":
    from .data_loader import prepare_analysis_sample
    
    df = prepare_analysis_sample()
    results = run_complementarity_analysis(df)
