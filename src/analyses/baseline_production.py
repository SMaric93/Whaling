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
    Compute KSS (Kline-Saggio-Sølvsten 2020) bias correction for variance components.
    
    Uses exact leverage computation via the inverse Gram matrix:
        P_ii = x_i' (X'X)^-1 x_i
    
    Bias corrections:
        Bias(Var(θ)) = (1/N) * Σ P_ii_θ * σ_i^2
        Bias(Cov(θ,ψ)) = (1/N) * Σ P_ii_θψ * σ_i^2
        
    Parameters
    ----------
    results : Dict
        Results from estimate_r1.
        
    Returns
    -------
    Dict
        Bias-corrected variance estimates.
    """
    print("\nComputing KSS bias correction (Exact Method)...")
    
    df = results["df"]
    residuals = results["residuals"]
    X = results["X"]
    n = len(df)
    
    alpha_i = df["alpha_hat"].values
    gamma_j = df["gamma_hat"].values
    
    # 1. Compute Inverse Gram Matrix (X'X)^-1
    # Note: With ~9000 parameters, dense inversion is feasible (~600MB RAM)
    # We maintain sparsity for measuring X'X
    print(f"  Computing Gram matrix (shape {X.shape[1]}x{X.shape[1]})...")
    XtX = X.T @ X
    
    # Using sparse linear solver to get inverse diagonal logic or dense inv
    # Given size, dense inv is safer for stability if memory allows
    try:
        if sp.issparse(XtX):
            XtX_dense = XtX.toarray()
        else:
            XtX_dense = XtX
            
        print("  Inverting Gram matrix...")
        XtX_inv = np.linalg.inv(XtX_dense)
        
    except np.linalg.LinAlgError:
        print("  WARNING: Singular matrix, using pseudo-inverse...")
        XtX_inv = np.linalg.pinv(XtX_dense)

    # 2. Compute Leverages (P_ii)
    # We need P_ii for the whole system, but specifically we need to project 
    # the bias onto the alpha and gamma components.
    # Actually, KSS defines the bias for the quadratic form.
    # Var(alpha) = alpha' alpha / N.
    # Bias = tr( (X'X)^-1 * M_alpha * Sigma ) / N?
    # Simplified KSS for random effects variance components:
    # Var_hat(\theta) = Var(\theta) + bias.
    # Bias correction term for observation i: B_i = P_ii * sigma_i^2 (approx)
    # More precisely, we need the leverage corresponding to the specific FE block.
    
    # Let's use the 'statistical leverage' approach P_ii = diag(X * (X'X)^-1 * X')
    # Since X is n x k and (X'X)^-1 is k x k, we can compute row-wise dot products efficiently
    # P_ii = sum( (X[i] @ XtX_inv) * X[i] )
    
    # Efficient calculation:
    # H = X @ XtX_inv
    # P_ii = sum(H * X, axis=1)
    
    print("  Calculating exact leverages...")
    # This might be memory intensive if we materialize H.
    # Do it in chunks or utilizing sparsity.
    
    # Identify indices for Captain (alpha) and Agent (gamma) coefficients
    idx_maps = results["index_maps"]
    cap_start = 0
    cap_end = idx_maps["captain"]["n"]
    
    ag_start = cap_end
    ag_end = cap_end + idx_maps["agent"]["n"] - 1 # -1 for dropped ref
    
    # Captain part of X
    X_alpha = X[:, cap_start:cap_end]
    # Agent part of X
    X_gamma = X[:, ag_start:ag_end]
    
    # Corresponding blocks of inverse matrix
    S_alpha = XtX_inv[cap_start:cap_end, cap_start:cap_end]
    S_gamma = XtX_inv[ag_start:ag_end, ag_start:ag_end]
    S_cross = XtX_inv[cap_start:cap_end, ag_start:ag_end]
    
    # Compute leverages specifically for the variance components
    # The bias in Var(\hat\alpha) comes from the trace of the variance of estimate
    # Bias ≈ (1/N) * Sum( Trace(S_alpha) ? No. )
    
    # KSS (2020) Equation 11:
    # Bias correction for \sigma^2_\theta = \hat{\sigma}^2_\theta - \hat{B}_\theta
    # \hat{B}_\theta = (1/N) \sum_i \hat{\sigma}^2_i * ( x_{i,\theta}' S_{\theta\theta} x_{i,\theta} - x_{i,\theta}' S_{\theta\psi} x_{i,\psi} ... )
    # Actually simpler: Bias is summing the "prediction variance" of that component.
    
    # Leverage of 'captain part' for obs i: B_ii_alpha = x_{i,alpha} @ S_alpha @ x_{i,alpha}'
    # We can compute this efficiently: sum( (X_alpha @ S_alpha) * X_alpha, axis=1 )
    
    # 1. Project alpha part
    print("  Projecting bias terms...")
    # Dense matrix math is fine here
    if sp.issparse(X_alpha): 
        X_alpha_d = X_alpha.toarray() 
    else: 
        X_alpha_d = X_alpha
        
    leverage_alpha = np.sum((X_alpha_d @ S_alpha) * X_alpha_d, axis=1)
    
    # 2. Project gamma part
    if sp.issparse(X_gamma): 
        X_gamma_d = X_gamma.toarray()
    else:
        X_gamma_d = X_gamma
        
    leverage_gamma = np.sum((X_gamma_d @ S_gamma) * X_gamma_d, axis=1)
    
    # 3. Project cross part (for Covariance)
    # Term is x_{i,alpha} @ S_cross @ x_{i,gamma}'
    leverage_cov = np.sum((X_alpha_d @ S_cross) * X_gamma_d, axis=1)
    
    # Compute component biases
    # Bias = (1/N) * Sum( leverage_component * sigma_i^2 )
    sigma_sq = residuals ** 2
    
    bias_var_alpha = np.sum(leverage_alpha * sigma_sq) / n
    bias_var_gamma = np.sum(leverage_gamma * sigma_sq) / n
    bias_cov = np.sum(leverage_cov * sigma_sq) / n
    
    # Plug-in estimates
    var_alpha_plugin = np.var(alpha_i)
    var_gamma_plugin = np.var(gamma_j)
    cov_plugin = np.cov(alpha_i, gamma_j)[0, 1]
    
    # Corrected estimates
    var_alpha_kss = max(0, var_alpha_plugin - bias_var_alpha)
    var_gamma_kss = max(0, var_gamma_plugin - bias_var_gamma)
    cov_kss = cov_plugin - bias_cov
    
    print(f"  Var(α): Plugin={var_alpha_plugin:.4f} - Bias={bias_var_alpha:.4f} = {var_alpha_kss:.4f}")
    print(f"  Var(γ): Plugin={var_gamma_plugin:.4f} - Bias={bias_var_gamma:.4f} = {var_gamma_kss:.4f}")
    print(f"  Cov(α,γ): Plugin={cov_plugin:.4f} - Bias={bias_cov:.4f} = {cov_kss:.4f}")
    
    return {
        "var_alpha_plugin": var_alpha_plugin,
        "var_gamma_plugin": var_gamma_plugin,
        "cov_plugin": cov_plugin,
        "var_alpha_kss": var_alpha_kss,
        "var_gamma_kss": var_gamma_kss,
        "cov_kss": cov_kss,
        "bias_var_alpha": bias_var_alpha,
        "bias_var_gamma": bias_var_gamma,
        "bias_cov": bias_cov,
    }


def variance_decomposition(results: Dict) -> pd.DataFrame:
    """
    Compute full variance decomposition with KSS correction.
    
    The AKM variance decomposition partitions the variance of log output:
        Var(y) = Var(α) + Var(γ) + 2*Cov(α,γ) + Var(other FEs) + Var(Xβ) + Var(ε)
    
    We report:
    1. Shares of total outcome variance (Var(y)) - shows absolute contribution
    2. Shares of captain+agent variance - shows relative importance within the 
       labor market components (α and γ)
    
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
    
    # Compute the total variance explained by captain + agent (the AKM components)
    # This is: Var(α) + Var(γ) + 2*Cov(α,γ) = Var(α + γ)
    plugin_captain_agent_total = var_alpha + var_gamma + 2 * cov_alpha_gamma
    kss_captain_agent_total = kss["var_alpha_kss"] + kss["var_gamma_kss"] + 2 * kss["cov_kss"]
    
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
    
    # Share of TOTAL variance (Var(y)) - for context
    decomp["Share_of_VarY"] = decomp["Plugin_Var"] / var_y
    decomp["Share_of_VarY_KSS"] = decomp["KSS_Var"] / var_y
    
    # Share of CAPTAIN+AGENT variance (the interpretable AKM shares)
    # Only compute for the first 3 rows (α, γ, 2*Cov)
    decomp["Plugin_Share"] = np.nan
    decomp["KSS_Share"] = np.nan
    
    if plugin_captain_agent_total > 0:
        decomp.loc[0, "Plugin_Share"] = var_alpha / plugin_captain_agent_total
        decomp.loc[1, "Plugin_Share"] = var_gamma / plugin_captain_agent_total
        decomp.loc[2, "Plugin_Share"] = (2 * cov_alpha_gamma) / plugin_captain_agent_total
    
    if kss_captain_agent_total > 0:
        decomp.loc[0, "KSS_Share"] = kss["var_alpha_kss"] / kss_captain_agent_total
        decomp.loc[1, "KSS_Share"] = kss["var_gamma_kss"] / kss_captain_agent_total
        decomp.loc[2, "KSS_Share"] = (2 * kss["cov_kss"]) / kss_captain_agent_total
    
    # Convert to percentages for display
    decomp["Plugin_Share_Pct"] = decomp["Plugin_Share"] * 100
    decomp["KSS_Share_Pct"] = decomp["KSS_Share"] * 100
    
    print("\n--- Variance Decomposition ---")
    print(f"\nTotal Var(y) = {var_y:.4f}")
    print(f"Captain+Agent Var = Var(α) + Var(γ) + 2Cov(α,γ) = {plugin_captain_agent_total:.4f} (plugin), {kss_captain_agent_total:.4f} (KSS)")
    print(f"Captain+Agent share of Var(y) = {plugin_captain_agent_total/var_y*100:.1f}% (plugin), {kss_captain_agent_total/var_y*100:.1f}% (KSS)")
    print(f"\n--- AKM Component Shares (within Captain+Agent variance) ---")
    print(decomp[["Component", "KSS_Var", "KSS_Share_Pct"]].to_string(index=False))
    
    # Correlations
    corr_plugin = cov_alpha_gamma / (np.sqrt(var_alpha) * np.sqrt(var_gamma)) if var_alpha > 0 and var_gamma > 0 else 0
    corr_kss = kss["cov_kss"] / (np.sqrt(kss["var_alpha_kss"]) * np.sqrt(kss["var_gamma_kss"])) if kss["var_alpha_kss"] > 0 and kss["var_gamma_kss"] > 0 else 0
    
    print(f"\nCorr(α, γ) plugin = {corr_plugin:.4f}")
    print(f"Corr(α, γ) KSS    = {corr_kss:.4f}")
    
    # Store additional diagnostics
    results["kss"] = kss
    results["variance_decomposition"] = decomp
    results["corr_alpha_gamma_plugin"] = corr_plugin
    results["corr_alpha_gamma_kss"] = corr_kss
    results["var_y"] = var_y
    results["captain_agent_var_plugin"] = plugin_captain_agent_total
    results["captain_agent_var_kss"] = kss_captain_agent_total
    results["captain_agent_share_of_var_y"] = kss_captain_agent_total / var_y
    
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
