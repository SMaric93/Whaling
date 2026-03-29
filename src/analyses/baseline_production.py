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
from .parallel_akm import parallel_kss_leverage


def build_fe_block(
    values: pd.Series,
    drop_first: bool = False,
) -> Tuple[sp.csr_matrix, np.ndarray, Dict]:
    """Build one sparse fixed-effect block using factorized codes."""
    codes, ids = pd.factorize(values, sort=False)
    offset = 1 if drop_first else 0
    mask = codes >= offset
    rows = np.flatnonzero(mask)
    cols = codes[mask] - offset
    matrix = sp.csr_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(len(values), max(len(ids) - offset, 0)),
    )
    index_map = {value: i for i, value in enumerate(ids)}
    return matrix, ids, index_map


def build_sparse_design_matrix(
    df: pd.DataFrame,
    include_vessel_period: bool = True,
    include_route_time: bool = True,
    include_port_time: bool = True,
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
    include_port_time : bool
        Whether to include port×time FEs (home port × decade).
        
    Returns
    -------
    Tuple[sp.csr_matrix, Dict]
        (design_matrix, index_maps)
    """
    n = len(df)
    matrices = []
    index_maps = {}
    
    # Captain FEs
    X_captain, captain_ids, captain_map = build_fe_block(df["captain_id"])
    matrices.append(X_captain)
    index_maps["captain"] = {"ids": captain_ids, "map": captain_map, "n": len(captain_ids)}
    
    # Agent FEs (drop first for identification)
    X_agent, agent_ids, agent_map = build_fe_block(df["agent_id"], drop_first=True)
    matrices.append(X_agent)
    index_maps["agent"] = {"ids": agent_ids, "map": agent_map, "n": len(agent_ids)}
    
    # Vessel×period FEs (drop first)
    if include_vessel_period and "vessel_period" in df.columns:
        X_vp, vp_ids, vp_map = build_fe_block(df["vessel_period"], drop_first=True)
        matrices.append(X_vp)
        index_maps["vessel_period"] = {"ids": vp_ids, "map": vp_map, "n": len(vp_ids)}
    
    # Route×time FEs (drop first)
    if include_route_time and "route_time" in df.columns:
        X_rt, rt_ids, rt_map = build_fe_block(df["route_time"], drop_first=True)
        matrices.append(X_rt)
        index_maps["route_time"] = {"ids": rt_ids, "map": rt_map, "n": len(rt_ids)}
    
    # Port×time FEs (drop first) - home port × decade
    if include_port_time and "port_time" in df.columns:
        X_pt, pt_ids, pt_map = build_fe_block(df["port_time"], drop_first=True)
        matrices.append(X_pt)
        index_maps["port_time"] = {"ids": pt_ids, "map": pt_map, "n": len(pt_ids)}
    
    # Continuous controls (standard + any column starting with '_fe_')
    controls = []
    control_names = []
    
    for col in ["log_tonnage", "log_duration"]:
        if col in df.columns:
            controls.append(df[col].values)
            control_names.append(col)
    
    # Additional controls: any column starting with '_fe_' (e.g. year dummies)
    for col in sorted(df.columns):
        if col.startswith("_fe_") and col not in control_names:
            controls.append(df[col].to_numpy(dtype=float, copy=False))
            control_names.append(col)
    
    if controls:
        X_controls = sp.csr_matrix(df[control_names].to_numpy(dtype=float, copy=False))
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
    if "port_time" in index_maps:
        print(f"  Port×time FEs: {index_maps['port_time']['n']} (−1 normalized)")
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
    captain_lookup = pd.Series(theta, index=index_maps["captain"]["ids"])
    agent_lookup = pd.Series(psi, index=index_maps["agent"]["ids"])
    df_est["alpha_hat"] = df_est["captain_id"].map(captain_lookup)
    df_est["gamma_hat"] = df_est["agent_id"].map(agent_lookup)
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


def compute_kss_correction(
    results: Dict,
    use_parallel: bool = True,
    n_workers: Optional[int] = None,
    homoskedastic: bool = False,
) -> Dict:
    """
    Compute KSS (Kline-Saggio-Sølvsten 2020) bias correction for variance components.
    
    Follows the KSS MATLAB reference implementation (leave_out_KSS.m, lines 448-462):
    
    1. Use the full model (with all FEs and controls) to estimate nuisance coefficients.
    2. Partial out nuisance FEs (vessel×period, route×time, port×time) and controls
       from the outcome: y_new = y - X_nuisance @ b_nuisance.
    3. Re-estimate a reduced model with only captain + agent FEs on y_new.
    4. Compute KSS leverages on the reduced design matrix, which is well-conditioned.
    
    This avoids the numerical instability that arises when the full design matrix
    has more columns than observations (which makes the Gram matrix singular).
    
    Parameters
    ----------
    results : Dict
        Results from estimate_r1 (which ran the full model).
    use_parallel : bool
        If True, use multithreaded leverage computation.
    n_workers : int, optional
        Number of worker threads (default: auto-detect CPU count).
    homoskedastic : bool
        If True, use σ̂² = RSS/(n-k) instead of observation-specific εᵢ².
        More stable at low obs/parameter ratios (< 10). This is the correction
        used in Card, Heining, Kline (2013).
        
    Returns
    -------
    Dict
        Bias-corrected variance estimates.
    """
    mode_str = "homoskedastic" if homoskedastic else "heteroskedastic"
    print(f"\nComputing KSS bias correction (partial-out, {mode_str}, parallel={use_parallel})...")
    
    df = results["df"]
    full_X = results["X"]
    y = results["y"]
    idx_maps = results["index_maps"]
    n = len(df)
    
    # =========================================================================
    # Step 1: Identify captain+agent vs nuisance columns in full design matrix
    # =========================================================================
    cap_n = idx_maps["captain"]["n"]
    ag_n = idx_maps["agent"]["n"] - 1  # minus 1 for dropped reference
    
    ca_end = cap_n + ag_n  # End of captain+agent columns
    
    # Nuisance columns: everything after captain+agent FEs
    nuisance_start = ca_end
    n_nuisance = full_X.shape[1] - nuisance_start
    
    print(f"  Full design: {full_X.shape[1]} columns")
    print(f"  Captain+Agent FEs: {cap_n + ag_n} columns")
    print(f"  Nuisance (vessel×period, route×time, port×time, controls): {n_nuisance} columns")
    
    # =========================================================================
    # Step 2: Partial out nuisance FEs from y
    # Following KSS MATLAB: y = y - X(:,N+J:end) * b(N+J:end)
    # =========================================================================
    if n_nuisance > 0:
        # Partial out nuisance by regressing y on nuisance columns ONLY.
        # This avoids collinearity between nuisance (e.g. decade dummies)
        # and the captain/agent FE block that can produce NaN in LSQR.
        X_nuisance = full_X[:, nuisance_start:]
        if sp.issparse(X_nuisance):
            X_nui_dense = X_nuisance.toarray()
        else:
            X_nui_dense = np.asarray(X_nuisance)
        
        # OLS via pseudo-inverse (handles rank-deficiency gracefully)
        XtX_nui = X_nui_dense.T @ X_nui_dense
        Xty_nui = X_nui_dense.T @ y
        try:
            b_nuisance = np.linalg.solve(XtX_nui, Xty_nui)
        except np.linalg.LinAlgError:
            b_nuisance = np.linalg.pinv(XtX_nui) @ Xty_nui
        nuisance_fitted = X_nui_dense @ b_nuisance
        
        y_partialled = y - nuisance_fitted
        print(f"  Partialled out {n_nuisance} nuisance columns from y")
        print(f"  Var(y_original) = {np.var(y):.4f}")
        print(f"  Var(y_partialled) = {np.var(y_partialled):.4f}")
    else:
        y_partialled = y
    
    # =========================================================================
    # Step 3: Build reduced design matrix (captain + agent FEs only)
    # =========================================================================
    X_reduced = full_X[:, :ca_end]
    print(f"  Reduced design matrix: ({n}, {ca_end})")
    print(f"  Observations per parameter: {n / ca_end:.1f}")
    
    # =========================================================================
    # Step 4: Re-estimate reduced model on partialled-out y
    # =========================================================================
    result_reduced = lsqr(X_reduced, y_partialled, iter_lim=10000, atol=1e-10, btol=1e-10)
    beta_reduced = result_reduced[0]
    residuals_reduced = y_partialled - X_reduced @ beta_reduced
    
    # Extract FE estimates from reduced model
    theta_reduced = beta_reduced[:cap_n]
    psi_reduced = np.concatenate([[0], beta_reduced[cap_n:ca_end]])
    
    # Map to observation level
    alpha_lookup = pd.Series(theta_reduced, index=idx_maps["captain"]["ids"])
    gamma_lookup = pd.Series(psi_reduced, index=idx_maps["agent"]["ids"])
    alpha_i = df["captain_id"].map(alpha_lookup).to_numpy(dtype=float, copy=False)
    gamma_j = df["agent_id"].map(gamma_lookup).to_numpy(dtype=float, copy=False)
    
    # =========================================================================
    # Step 5: Compute KSS leverages on the reduced (well-conditioned) system
    # =========================================================================
    print(f"  Computing Gram matrix ({ca_end}×{ca_end})...")
    XtX = X_reduced.T @ X_reduced
    
    if sp.issparse(XtX):
        XtX_dense = XtX.toarray()
    else:
        XtX_dense = XtX
    
    # Check condition number
    try:
        cond = np.linalg.cond(XtX_dense)
        print(f"  Condition number: {cond:.2e}")
        if cond > 1e15:
            print("  WARNING: Gram matrix is ill-conditioned, using pseudo-inverse")
    except Exception:
        cond = None
    
    print("  Inverting Gram matrix...")
    try:
        XtX_inv = np.linalg.inv(XtX_dense)
    except np.linalg.LinAlgError:
        print("  WARNING: Singular matrix, using pseudo-inverse...")
        XtX_inv = np.linalg.pinv(XtX_dense)
    
    # Captain and agent sub-blocks
    X_alpha = X_reduced[:, :cap_n]
    X_gamma = X_reduced[:, cap_n:ca_end]
    
    S_alpha = XtX_inv[:cap_n, :cap_n]
    S_gamma = XtX_inv[cap_n:ca_end, cap_n:ca_end]
    S_cross = XtX_inv[:cap_n, cap_n:ca_end]
    
    # Convert to dense for leverage computation
    print("  Calculating exact leverages...")
    if sp.issparse(X_alpha):
        X_alpha_d = X_alpha.toarray()
    else:
        X_alpha_d = X_alpha
    
    if sp.issparse(X_gamma):
        X_gamma_d = X_gamma.toarray()
    else:
        X_gamma_d = X_gamma
    
    # Compute leverages - parallel or sequential
    if use_parallel:
        leverage_alpha, leverage_gamma, leverage_cov = parallel_kss_leverage(
            X_alpha_d, S_alpha,
            X_gamma_d, S_gamma,
            S_cross,
            n_workers=n_workers,
        )
    else:
        leverage_alpha = np.sum((X_alpha_d @ S_alpha) * X_alpha_d, axis=1)
        leverage_gamma = np.sum((X_gamma_d @ S_gamma) * X_gamma_d, axis=1)
        leverage_cov = np.sum((X_alpha_d @ S_cross) * X_gamma_d, axis=1)
    
    # =========================================================================
    # Step 6: Compute bias corrections
    # =========================================================================
    if homoskedastic:
        # Card-Heining-Kline (2013) homoskedastic correction:
        # Use σ̂² = RSS / (n - k) as a single scalar instead of εᵢ²
        # This is much more stable at low obs/parameter ratios because
        # it prevents high-leverage, high-residual observations from
        # dominating the cross-term bias (which causes Corr > 1).
        rss = np.sum(residuals_reduced ** 2)
        dof = max(n - ca_end, 1)
        sigma_hat_sq = rss / dof
        print(f"  Homoskedastic σ̂² = {sigma_hat_sq:.4f} (RSS={rss:.1f}, dof={dof})")
        
        bias_var_alpha = sigma_hat_sq * np.mean(leverage_alpha)
        bias_var_gamma = sigma_hat_sq * np.mean(leverage_gamma)
        bias_cov = sigma_hat_sq * np.mean(leverage_cov)
    else:
        # KSS heteroskedastic correction: weight by observation-specific εᵢ²
        sigma_sq = residuals_reduced ** 2
        bias_var_alpha = np.sum(leverage_alpha * sigma_sq) / n
        bias_var_gamma = np.sum(leverage_gamma * sigma_sq) / n
        bias_cov = np.sum(leverage_cov * sigma_sq) / n
    
    # Plugin estimates (from partialled-out model)
    var_alpha_plugin = np.var(alpha_i)
    var_gamma_plugin = np.var(gamma_j)
    cov_plugin = np.cov(alpha_i, gamma_j)[0, 1]
    
    # Corrected estimates
    var_alpha_kss = max(0, var_alpha_plugin - bias_var_alpha)
    var_gamma_kss = max(0, var_gamma_plugin - bias_var_gamma)
    cov_kss = cov_plugin - bias_cov
    
    # Sanity: clamp correlation to [-1, 1]
    if var_alpha_kss > 0 and var_gamma_kss > 0:
        corr_kss = cov_kss / (np.sqrt(var_alpha_kss) * np.sqrt(var_gamma_kss))
        if abs(corr_kss) > 1.0:
            print(f"  WARNING: Corr(α,γ) = {corr_kss:.4f} exceeds [-1,1]. "
                  f"Clamping covariance.")
            cov_kss = np.sign(cov_kss) * np.sqrt(var_alpha_kss * var_gamma_kss)
    
    print(f"\n  --- KSS Results ({mode_str}) ---")
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
        "n_nuisance_partialled": n_nuisance,
        "reduced_design_shape": (n, ca_end),
        "condition_number": cond,
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
