"""
Robustness Specifications with Vessel Controls.

Addresses editorial concern C2.3: Vessel confounding.
Re-estimates Table 3 variance decomposition with tonnage + rig type controls.
"""

from typing import Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

from .config import TABLES_DIR

warnings.filterwarnings("ignore", category=FutureWarning)


def create_rig_dummies(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create rig type dummy variables for regression.
    
    Simplifies rig categories into major types to avoid thin cells.
    """
    df = df.copy()
    
    if "rig" not in df.columns:
        return df, []
    
    def simplify_rig(rig):
        if pd.isna(rig):
            return "Other"
        rig = str(rig).lower()
        if "ship" in rig:
            return "Ship"
        elif "bark" in rig:
            return "Bark"
        elif "brig" in rig:
            return "Brig"
        elif "schr" in rig or "schooner" in rig:
            return "Schooner"
        elif "sloop" in rig:
            return "Sloop"
        else:
            return "Other"
    
    df["rig_category"] = df["rig"].apply(simplify_rig)
    
    # Create dummies (drop first = "Bark" which is common)
    rig_dummies = pd.get_dummies(df["rig_category"], prefix="rig", drop_first=True)
    dummy_cols = list(rig_dummies.columns)
    
    df = pd.concat([df, rig_dummies], axis=1)
    
    return df, dummy_cols


def estimate_with_vessel_controls(
    df: pd.DataFrame,
    dependent_var: str = "log_q",
) -> Dict:
    """
    Re-estimate AKM model with full vessel controls (tonnage + rig).
    
    logQ = α_c + γ_a + θ_{route×time} + β_tonnage + rig_FE + ε
    
    Parameters
    ----------
    df : pd.DataFrame
        Voyage data.
    dependent_var : str
        Outcome variable.
        
    Returns
    -------
    Dict
        Estimation results including variance decomposition.
    """
    print("\n" + "=" * 60)
    print("TABLE 3 ROBUSTNESS: WITH VESSEL CONTROLS (TONNAGE + RIG)")
    print("=" * 60)
    
    # Add rig dummies
    df, rig_cols = create_rig_dummies(df)
    
    print(f"Sample: {len(df):,} voyages")
    print(f"Rig categories: {df['rig_category'].value_counts().to_dict()}")
    
    # Build design matrix
    n = len(df)
    matrices = []
    
    # Captain FEs
    captain_ids = df["captain_id"].unique()
    captain_map = {c: i for i, c in enumerate(captain_ids)}
    captain_idx = df["captain_id"].map(captain_map).values
    X_captain = sp.csr_matrix(
        (np.ones(n), (np.arange(n), captain_idx)),
        shape=(n, len(captain_ids))
    )
    matrices.append(X_captain)
    
    # Agent FEs (drop first)
    agent_ids = df["agent_id"].unique()
    agent_map = {a: i for i, a in enumerate(agent_ids)}
    agent_idx = df["agent_id"].map(agent_map).values
    X_agent = sp.csr_matrix(
        (np.ones(n), (np.arange(n), agent_idx)),
        shape=(n, len(agent_ids))
    )[:, 1:]
    matrices.append(X_agent)
    
    # Route×time FEs (drop first)
    if "route_time" in df.columns:
        rt_ids = df["route_time"].unique()
        rt_map = {r: i for i, r in enumerate(rt_ids)}
        rt_idx = df["route_time"].map(rt_map).values
        X_rt = sp.csr_matrix(
            (np.ones(n), (np.arange(n), rt_idx)),
            shape=(n, len(rt_ids))
        )[:, 1:]
        matrices.append(X_rt)
    
    # Controls: tonnage + rig dummies
    controls = []
    control_names = []
    
    if "log_tonnage" in df.columns:
        controls.append(df["log_tonnage"].fillna(df["log_tonnage"].mean()).values)
        control_names.append("log_tonnage")
    
    for col in rig_cols:
        controls.append(df[col].values.astype(float))
        control_names.append(col)
    
    if controls:
        X_controls = sp.csr_matrix(np.column_stack(controls))
        matrices.append(X_controls)
    
    X = sp.hstack(matrices)
    y = df[dependent_var].values
    
    print(f"\nDesign matrix: {X.shape}")
    print(f"Captain FEs: {len(captain_ids)}")
    print(f"Agent FEs: {len(agent_ids)-1}")
    print(f"Controls: {control_names}")
    
    # Solve
    result = lsqr(X, y, iter_lim=10000, atol=1e-10, btol=1e-10)
    beta = result[0]
    
    # Extract captain and agent effects
    n_captains = len(captain_ids)
    theta = beta[:n_captains]
    
    n_agents = len(agent_ids) - 1
    psi_est = beta[n_captains:n_captains + n_agents]
    psi = np.concatenate([[0], psi_est])
    
    # Map FEs to data
    df["alpha_hat"] = df["captain_id"].map(dict(zip(captain_ids, theta)))
    df["gamma_hat"] = df["agent_id"].map(dict(zip(agent_ids, psi)))
    
    # Compute variance decomposition
    y_hat = X @ beta
    residuals = y - y_hat
    r2 = 1 - np.var(residuals) / np.var(y)
    
    var_alpha = np.var(df["alpha_hat"])
    var_gamma = np.var(df["gamma_hat"])
    cov_alpha_gamma = np.cov(df["alpha_hat"], df["gamma_hat"])[0, 1]
    var_resid = np.var(residuals)
    
    # Total explained by captain + agent
    total_ca = var_alpha + var_gamma + 2 * cov_alpha_gamma
    
    print(f"\n--- Results ---")
    print(f"R² = {r2:.4f}")
    print(f"Var(α) = {var_alpha:.4f} ({var_alpha/total_ca*100:.1f}% of α+γ)")
    print(f"Var(γ) = {var_gamma:.4f} ({var_gamma/total_ca*100:.1f}% of α+γ)")
    print(f"2×Cov(α,γ) = {2*cov_alpha_gamma:.4f}")
    
    # Control coefficients
    control_betas = {}
    if controls:
        n_prior = n_captains + n_agents
        if "route_time" in df.columns:
            n_prior += len(rt_ids) - 1
        control_betas = dict(zip(control_names, beta[n_prior:n_prior+len(control_names)]))
        print(f"\nControl coefficients:")
        for name, coef in control_betas.items():
            print(f"  {name}: {coef:.4f}")
    
    return {
        "n": n,
        "r2": r2,
        "var_alpha": var_alpha,
        "var_gamma": var_gamma,
        "cov_alpha_gamma": cov_alpha_gamma,
        "var_residual": var_resid,
        "alpha_share": var_alpha / total_ca,
        "gamma_share": var_gamma / total_ca,
        "control_coefficients": control_betas,
        "alpha_hat": theta,
        "gamma_hat": psi,
        "df": df,
    }


def run_robustness_suite(
    df: pd.DataFrame,
    save_outputs: bool = True,
) -> Dict:
    """
    Run robustness suite with vessel controls.
    """
    print("\n" + "=" * 60)
    print("ROBUSTNESS: TABLE 3 WITH VESSEL CONTROLS")
    print("=" * 60)
    
    results = estimate_with_vessel_controls(df)
    
    # Save results
    if save_outputs:
        output_path = TABLES_DIR / "robustness_vessel_controls.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        summary = pd.DataFrame([{
            "Specification": "With Vessel Controls", 
            "gamma_share": results["gamma_share"],
            "n": results["n"],
            "r2": results["r2"],
        }])
        summary.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
    
    return results


if __name__ == "__main__":
    from .data_loader import prepare_analysis_sample
    
    df = prepare_analysis_sample()
    results = run_robustness_suite(df)
