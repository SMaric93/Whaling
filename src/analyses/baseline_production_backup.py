"""
Baseline production function estimation (R1).

Implements AKM-style decomposition with:
- Captain fixed effects (α_c)
- Agent fixed effects (γ_a)  
- Vessel×period fixed effects (δ)
- Route×time fixed effects (θ)
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

from .config import REGRESSIONS, DEFAULT_SAMPLE
from .connected_set import find_leave_one_out_connected_set


def build_sparse_design_matrix(
    df: pd.DataFrame,
    include_vessel_period: bool = True,
    include_route_time: bool = True,
) -> Tuple[sp.csr_matrix, Dict]:
    """
    Build sparse design matrix for AKM estimation.
    
    Parameters
    ----------
    df : pd.DataFrame
        Prepared voyage data with FE group columns.
    include_vessel_period : bool
        Whether to include vessel×period FEs.
    include_route_time : bool
        Whether to include route×time FEs.
        
    Returns
    -------
    Tuple[sp.csr_matrix, Dict]
        (design_matrix, index_maps)
    """
    n = len(df)
    matrices = []
    index_maps = {}
    
    # Captain FEs
    captain_ids = df["captain_id"].unique()
    captain_map = {c: i for i, c in enumerate(captain_ids)}
    captain_idx = df["captain_id"].map(captain_map).values
    X_captain = sp.csr_matrix(
        (np.ones(n), (np.arange(n), captain_idx)),
        shape=(n, len(captain_ids))
    )
    matrices.append(X_captain)
    index_maps["captain"] = {"ids": captain_ids, "map": captain_map, "n": len(captain_ids)}
    
    # Agent FEs (drop first for identification)
    agent_ids = df["agent_id"].unique()
    agent_map = {a: i for i, a in enumerate(agent_ids)}
    agent_idx = df["agent_id"].map(agent_map).values
    X_agent_full = sp.csr_matrix(
        (np.ones(n), (np.arange(n), agent_idx)),
        shape=(n, len(agent_ids))
    )
    X_agent = X_agent_full[:, 1:]  # Drop first
    matrices.append(X_agent)
    index_maps["agent"] = {"ids": agent_ids, "map": agent_map, "n": len(agent_ids)}
    
    # Vessel×period FEs (drop first)
    if include_vessel_period and "vessel_period" in df.columns:
        vp_ids = df["vessel_period"].unique()
        vp_map = {v: i for i, v in enumerate(vp_ids)}
        vp_idx = df["vessel_period"].map(vp_map).values
        X_vp_full = sp.csr_matrix(
            (np.ones(n), (np.arange(n), vp_idx)),
            shape=(n, len(vp_ids))
        )
        X_vp = X_vp_full[:, 1:]
        matrices.append(X_vp)
        index_maps["vessel_period"] = {"ids": vp_ids, "map": vp_map, "n": len(vp_ids)}
    
    # Route×time FEs (drop first)
    if include_route_time and "route_time" in df.columns:
        rt_ids = df["route_time"].unique()
        rt_map = {r: i for i, r in enumerate(rt_ids)}
        rt_idx = df["route_time"].map(rt_map).values
        X_rt_full = sp.csr_matrix(
            (np.ones(n), (np.arange(n), rt_idx)),
            shape=(n, len(rt_ids))
        )
        X_rt = X_rt_full[:, 1:]
        matrices.append(X_rt)
        index_maps["route_time"] = {"ids": rt_ids, "map": rt_map, "n": len(rt_ids)}
    
    # Continuous controls
    controls = []
    control_names = []
    
    for col in ["log_tonnage", "log_duration"]:
        if col in df.columns:
            controls.append(df[col].values)
            control_names.append(col)
    
    if controls:
        X_controls = sp.csr_matrix(np.column_stack(controls))
        matrices.append(X_controls)
        index_maps["controls"] = {"names": control_names, "n": len(control_names)}
    
    # Stack all
    X = sp.hstack(matrices)
    
    return X, index_maps


def estimate_r1(
    df: pd.DataFrame,
    dependent_var: str = "log_q",
    use_loo_sample: bool = True,
) -> Dict:
    """
    Estimate R1: Baseline production function.
    
    logQ_v = α_c + γ_a + δ_{vessel×period} + θ_{route×time} + Xβ + ε
    
    Parameters
    ----------
    df : pd.DataFrame
        Prepared voyage data.
    dependent_var : str
        Outcome variable column name.
    use_loo_sample : bool
        Whether to restrict to LOO connected set.
        
    Returns
    -------
    Dict
        Estimation results including FE estimates and diagnostics.
    """
    print("\n" + "=" * 60)
    print("ESTIMATING R1: BASELINE PRODUCTION FUNCTION")
    print("=" * 60)
    
    # Restrict to LOO connected set if requested
    if use_loo_sample:
        df_est, loo_diag = find_leave_one_out_connected_set(df)
    else:
        df_est = df.copy()
        loo_diag = None
    
    n = len(df_est)
    print(f"\nEstimation sample: {n:,} voyages")
    print(f"  Captains: {df_est['captain_id'].nunique():,}")
    print(f"  Agents: {df_est['agent_id'].nunique():,}")
    
    # Build design matrix
    X, index_maps = build_sparse_design_matrix(df_est)
    y = df_est[dependent_var].values
    
    print(f"\nDesign matrix: {X.shape}")
    print(f"  Captain FEs: {index_maps['captain']['n']}")
    print(f"  Agent FEs: {index_maps['agent']['n']} (−1 normalized)")
    if "vessel_period" in index_maps:
        print(f"  Vessel×period FEs: {index_maps['vessel_period']['n']} (−1 normalized)")
    if "route_time" in index_maps:
        print(f"  Route×time FEs: {index_maps['route_time']['n']} (−1 normalized)")
    if "controls" in index_maps:
        print(f"  Controls: {index_maps['controls']['names']}")
    
    # Solve sparse least squares
    print("\nSolving sparse least squares...")
    result = lsqr(X, y, iter_lim=10000, atol=1e-10, btol=1e-10)
    beta = result[0]
    
    # Extract coefficients
    idx = 0
    n_captains = index_maps["captain"]["n"]
    theta = beta[idx:idx + n_captains]
    idx += n_captains
    
    n_agents = index_maps["agent"]["n"]
    psi_est = beta[idx:idx + n_agents - 1]
    psi = np.concatenate([[0], psi_est])  # First agent normalized to 0
    idx += n_agents - 1
    
    # Vessel×period effects (if included)
    delta = None
    if "vessel_period" in index_maps:
        n_vp = index_maps["vessel_period"]["n"]
        delta_est = beta[idx:idx + n_vp - 1]
        delta = np.concatenate([[0], delta_est])
        idx += n_vp - 1
    
    # Route×time effects (if included)
    gamma = None
    if "route_time" in index_maps:
        n_rt = index_maps["route_time"]["n"]
        gamma_est = beta[idx:idx + n_rt - 1]
        gamma = np.concatenate([[0], gamma_est])
        idx += n_rt - 1
    
    # Control coefficients
    control_betas = {}
    if "controls" in index_maps:
        for i, name in enumerate(index_maps["controls"]["names"]):
            control_betas[name] = beta[idx + i]
    
    # Fitted values and residuals
    y_hat = X @ beta
    residuals = y - y_hat
    r2 = 1 - np.var(residuals) / np.var(y)
    rmse = np.std(residuals)
    
    print(f"\nModel fit:")
    print(f"  R² = {r2:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    
    # Create FE DataFrames
    captain_fe = pd.DataFrame({
        "captain_id": index_maps["captain"]["ids"],
        "alpha_hat": theta,
    })
    
    agent_fe = pd.DataFrame({
        "agent_id": index_maps["agent"]["ids"],
        "gamma_hat": psi,
    })
    
    # Merge FEs back to data
    df_est = df_est.merge(captain_fe, on="captain_id", how="left")
    df_est = df_est.merge(agent_fe, on="agent_id", how="left")
    df_est["residuals"] = residuals
    df_est["fitted"] = y_hat
    
    results = {
        "df": df_est,
        "captain_fe": captain_fe,
        "agent_fe": agent_fe,
        "alpha_hat": theta,
        "gamma_hat": psi,
        "delta": delta,
        "gamma_rt": gamma,
        "control_betas": control_betas,
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


def compute_kss_correction(results: Dict) -> Dict:
    """
    Compute KSS bias correction for variance components.
    
    Parameters
    ----------
    results : Dict
        Results from estimate_r1.
        
    Returns
    -------
    Dict
        Bias-corrected variance estimates.
    """
    print("\nComputing KSS bias correction...")
    
    df = results["df"]
    residuals = results["residuals"]
    n = len(df)
    
    alpha_i = df["alpha_hat"].values
    gamma_j = df["gamma_hat"].values
    
    # Leverages: approximate as 1/n_i for observation from captain with n_i obs
    captain_counts = df["captain_id"].value_counts()
    agent_counts = df["agent_id"].value_counts()
    
    h_captain = df["captain_id"].map(lambda c: 1.0 / captain_counts[c]).values
    h_agent = df["agent_id"].map(lambda a: 1.0 / agent_counts[a]).values
    
    # Residual variance (heteroskedastic)
    sigma_sq = residuals ** 2
    
    # Bias terms
    bias_var_alpha = np.sum(h_captain * sigma_sq) / n
    bias_var_gamma = np.sum(h_agent * sigma_sq) / n
    
    # Plug-in estimates
    var_alpha_plugin = np.var(alpha_i)
    var_gamma_plugin = np.var(gamma_j)
    cov_plugin = np.cov(alpha_i, gamma_j)[0, 1]
    
    # Corrected estimates
    var_alpha_kss = max(0, var_alpha_plugin - bias_var_alpha)
    var_gamma_kss = max(0, var_gamma_plugin - bias_var_gamma)
    cov_kss = cov_plugin  # Conservative: no correction for covariance
    
    print(f"  Var(α) plugin: {var_alpha_plugin:.4f}, bias: {bias_var_alpha:.4f}, corrected: {var_alpha_kss:.4f}")
    print(f"  Var(γ) plugin: {var_gamma_plugin:.4f}, bias: {bias_var_gamma:.4f}, corrected: {var_gamma_kss:.4f}")
    
    return {
        "var_alpha_plugin": var_alpha_plugin,
        "var_gamma_plugin": var_gamma_plugin,
        "cov_plugin": cov_plugin,
        "var_alpha_kss": var_alpha_kss,
        "var_gamma_kss": var_gamma_kss,
        "cov_kss": cov_kss,
        "bias_var_alpha": bias_var_alpha,
        "bias_var_gamma": bias_var_gamma,
    }


def variance_decomposition(results: Dict) -> pd.DataFrame:
    """
    Compute full variance decomposition with KSS correction.
    
    Parameters
    ----------
    results : Dict
        Results from estimate_r1.
        
    Returns
    -------
    pd.DataFrame
        Variance decomposition table.
    """
    print("\n" + "=" * 60)
    print("VARIANCE DECOMPOSITION (R1)")
    print("=" * 60)
    
    df = results["df"]
    
    alpha_i = df["alpha_hat"].values
    gamma_j = df["gamma_hat"].values
    y = results["y"]
    eps = results["residuals"]
    
    # Components
    var_y = np.var(y)
    var_alpha = np.var(alpha_i)
    var_gamma = np.var(gamma_j)
    cov_alpha_gamma = np.cov(alpha_i, gamma_j)[0, 1]
    var_eps = np.var(eps)
    
    # KSS correction
    kss = compute_kss_correction(results)
    total_bias = kss["bias_var_alpha"] + kss["bias_var_gamma"]
    var_eps_kss = var_eps + total_bias
    
    # Build decomposition table
    decomp = pd.DataFrame({
        "Component": [
            "Var(α) - Captain Skill",
            "Var(γ) - Agent Capability",
            "2×Cov(α,γ) - Sorting",
            "Var(ε) - Residual",
        ],
        "Plugin_Var": [var_alpha, var_gamma, 2*cov_alpha_gamma, var_eps],
        "KSS_Var": [kss["var_alpha_kss"], kss["var_gamma_kss"], 2*kss["cov_kss"], var_eps_kss],
    })
    
    decomp["Plugin_Share"] = decomp["Plugin_Var"] / var_y
    decomp["KSS_Share"] = decomp["KSS_Var"] / var_y
    
    print("\n--- Variance Decomposition ---")
    print(decomp.to_string(index=False))
    
    # Correlations
    corr_plugin = cov_alpha_gamma / (np.sqrt(var_alpha) * np.sqrt(var_gamma)) if var_alpha > 0 and var_gamma > 0 else 0
    corr_kss = kss["cov_kss"] / (np.sqrt(kss["var_alpha_kss"]) * np.sqrt(kss["var_gamma_kss"])) if kss["var_alpha_kss"] > 0 and kss["var_gamma_kss"] > 0 else 0
    
    print(f"\nCorr(α, γ) plugin = {corr_plugin:.4f}")
    print(f"Corr(α, γ) KSS    = {corr_kss:.4f}")
    
    results["kss"] = kss
    results["variance_decomposition"] = decomp
    results["corr_alpha_gamma_plugin"] = corr_plugin
    results["corr_alpha_gamma_kss"] = corr_kss
    
    return decomp


def run_r1(
    df: pd.DataFrame,
    save_outputs: bool = True,
    output_dir: Optional[str] = None,
) -> Dict:
    """
    Run full R1 baseline production function analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        Prepared voyage data.
    save_outputs : bool
        Whether to save outputs to files.
    output_dir : str, optional
        Output directory path.
        
    Returns
    -------
    Dict
        Full R1 results.
    """
    from .config import TABLES_DIR, FIGURES_DIR
    from pathlib import Path
    
    if output_dir is None:
        output_dir = TABLES_DIR
    output_dir = Path(output_dir)
    
    # Estimate model
    results = estimate_r1(df)
    
    # Variance decomposition
    decomp = variance_decomposition(results)
    
    # Save outputs
    if save_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Tables
        results["captain_fe"].to_csv(output_dir / "r1_captain_effects.csv", index=False)
        results["agent_fe"].to_csv(output_dir / "r1_agent_effects.csv", index=False)
        decomp.to_csv(output_dir / "r1_variance_decomposition.csv", index=False)
        
        print(f"\nSaved R1 outputs to {output_dir}")
    
    return results


if __name__ == "__main__":
    from .data_loader import prepare_analysis_sample
    
    df = prepare_analysis_sample()
    results = run_r1(df)
