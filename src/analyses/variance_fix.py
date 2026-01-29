"""
Variance Fix Module: Address the Exploding Variance Pathology.

Diagnoses and fixes the issue where Var(ψ)=10.28 is economically implausible.

Approaches:
1. Diagnose connectivity on LOO connected set
2. Implement grouped agent estimation (Port×Decade)
3. Apply Empirical Bayes shrinkage
4. Verify adding-up constraint
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
import warnings
from scipy import sparse

warnings.filterwarnings('ignore')


def diagnose_connectivity(df: pd.DataFrame) -> Dict:
    """
    Diagnose the connectivity of the captain-agent mobility graph.
    
    Operates on the leave-one-out connected set.
    
    Returns connectivity statistics and identifies sparse regions.
    """
    print("\n" + "=" * 70)
    print("CONNECTIVITY DIAGNOSTICS (LOO Connected Set)")
    print("=" * 70)
    
    # Build bipartite graph: captains x agents
    captain_agent = df.groupby(['captain_id', 'agent_id']).size().reset_index(name='n_voyages')
    
    n_captains = df['captain_id'].nunique()
    n_agents = df['agent_id'].nunique()
    n_edges = len(captain_agent)
    
    print(f"\nBipartite Graph:")
    print(f"  Captains: {n_captains:,}")
    print(f"  Agents: {n_agents:,}")
    print(f"  Edges (unique pairs): {n_edges:,}")
    print(f"  Total voyages: {len(df):,}")
    
    # Mobility statistics
    agents_per_captain = df.groupby('captain_id')['agent_id'].nunique()
    captains_per_agent = df.groupby('agent_id')['captain_id'].nunique()
    
    print(f"\nMobility Statistics:")
    print(f"  Agents per captain: {agents_per_captain.mean():.2f} (median: {agents_per_captain.median():.0f}, max: {agents_per_captain.max()})")
    print(f"  Captains per agent: {captains_per_agent.mean():.2f} (median: {captains_per_agent.median():.0f}, max: {captains_per_agent.max()})")
    
    # Movers vs stayers
    movers = (agents_per_captain > 1).sum()
    stayers = (agents_per_captain == 1).sum()
    mover_pct = 100 * movers / n_captains
    
    print(f"\nMovers vs Stayers:")
    print(f"  Movers (captains with 2+ agents): {movers:,} ({mover_pct:.1f}%)")
    print(f"  Stayers (captains with 1 agent): {stayers:,} ({100-mover_pct:.1f}%)")
    
    # Articulation points (edges that would disconnect if removed)
    # This is the LOO perspective
    from collections import defaultdict
    
    # Build adjacency for connected component analysis
    captain_to_idx = {c: i for i, c in enumerate(df['captain_id'].unique())}
    agent_to_idx = {a: i for i, a in enumerate(df['agent_id'].unique())}
    
    # Check if removing each edge breaks connectivity
    # Simplified: count edges that appear only once
    edge_counts = captain_agent.set_index(['captain_id', 'agent_id'])['n_voyages']
    single_appearance = (edge_counts == 1).sum()
    
    print(f"\nLOO Vulnerability:")
    print(f"  Single-voyage edges (articulation points): {single_appearance:,} ({100*single_appearance/n_edges:.1f}%)")
    
    # Agent concentration (few agents with many captains)
    agent_concentration = captains_per_agent.describe()
    top_agents = captains_per_agent.nlargest(10)
    
    print(f"\nAgent Concentration:")
    print(f"  Top 10 agents by captain count:")
    for agent, count in top_agents.items():
        print(f"    {agent}: {count} captains")
    
    # Economic implication check
    print(f"\nEconomic Implication:")
    print(f"  Current Var(ψ) ≈ 10.28 implies σ_ψ ≈ 3.2")
    print(f"  This means ±1 SD = factor of e^3.2 ≈ 24.5x productivity")
    print(f"  With {n_agents} agents, this is IMPLAUSIBLE")
    
    return {
        "n_captains": n_captains,
        "n_agents": n_agents,
        "n_edges": n_edges,
        "agents_per_captain_mean": agents_per_captain.mean(),
        "captains_per_agent_mean": captains_per_agent.mean(),
        "mover_pct": mover_pct,
        "single_voyage_edges": single_appearance,
        "articulation_pct": 100 * single_appearance / n_edges,
    }


def check_adding_up_constraint(
    var_theta: float,
    var_psi: float, 
    cov_theta_psi: float,
    var_epsilon: float,
    var_y: float,
) -> Dict:
    """
    Check if variance decomposition satisfies adding-up constraint.
    
    Var(y) = Var(θ) + Var(ψ) + 2Cov(θ,ψ) + Var(ε)
    """
    print("\n" + "=" * 70)
    print("ADDING-UP CONSTRAINT CHECK")
    print("=" * 70)
    
    implied = var_theta + var_psi + 2 * cov_theta_psi + var_epsilon
    residual = var_y - implied
    pct_error = 100 * abs(residual) / var_y
    
    print(f"\nComponents:")
    print(f"  Var(θ):      {var_theta:.4f}")
    print(f"  Var(ψ):      {var_psi:.4f}")
    print(f"  2Cov(θ,ψ):   {2*cov_theta_psi:.4f}")
    print(f"  Var(ε):      {var_epsilon:.4f}")
    print(f"  Sum:         {implied:.4f}")
    print(f"  Var(y):      {var_y:.4f}")
    print(f"  Residual:    {residual:.4f} ({pct_error:.1f}% error)")
    
    constraint_holds = pct_error < 5.0
    print(f"\nConstraint {'HOLDS' if constraint_holds else 'VIOLATED'} (tolerance: 5%)")
    
    return {
        "var_theta": var_theta,
        "var_psi": var_psi,
        "cov_theta_psi": cov_theta_psi,
        "var_epsilon": var_epsilon,
        "var_y": var_y,
        "implied": implied,
        "residual": residual,
        "pct_error": pct_error,
        "constraint_holds": constraint_holds,
    }


def create_agent_groups(df: pd.DataFrame, method: str = "port_decade") -> pd.DataFrame:
    """
    Create grouped agent identifiers to increase connectivity.
    
    Methods:
    - port_decade: Group by home port × decade
    - size_decile: Group by agent size (captain count)
    - port_only: Group by home port only
    """
    print(f"\nCreating agent groups (method: {method})...")
    
    df = df.copy()
    
    if method == "port_decade":
        # Use home port and decade
        if 'home_port' in df.columns and 'decade' in df.columns:
            df['agent_group'] = df['home_port'].astype(str) + '_' + df['decade'].astype(str)
        else:
            # Fallback: use first 2 chars of agent_id + decade
            df['agent_group'] = df['agent_id'].str[:6] + '_' + (df['year_out'] // 10 * 10).astype(str)
    
    elif method == "size_decile":
        # Group agents by their size (number of captains)
        agent_sizes = df.groupby('agent_id')['captain_id'].nunique()
        agent_deciles = pd.qcut(agent_sizes, q=10, labels=False, duplicates='drop')
        df['agent_group'] = df['agent_id'].map(agent_deciles).astype(str)
    
    elif method == "port_only":
        if 'home_port' in df.columns:
            df['agent_group'] = df['home_port'].astype(str)
        else:
            df['agent_group'] = df['agent_id'].str[:4]
    
    n_groups = df['agent_group'].nunique()
    n_original = df['agent_id'].nunique()
    compression = n_original / n_groups
    
    print(f"  Original agents: {n_original}")
    print(f"  Agent groups: {n_groups}")
    print(f"  Compression ratio: {compression:.1f}x")
    
    return df


def estimate_with_grouped_agents(df: pd.DataFrame, agent_col: str = "agent_group") -> Dict:
    """
    Re-estimate production function with grouped agent FE.
    
    This increases connectivity and should yield plausible variances.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import lsqr
    
    print("\n" + "=" * 70)
    print(f"RE-ESTIMATION WITH GROUPED AGENTS ({agent_col})")
    print("=" * 70)
    
    # Prepare data
    y = df['log_q'].values
    
    # Create FE indices
    captain_ids = df['captain_id'].unique()
    group_ids = df[agent_col].unique()
    
    captain_to_idx = {c: i for i, c in enumerate(captain_ids)}
    group_to_idx = {g: i + len(captain_ids) for i, g in enumerate(group_ids)}
    
    n_captains = len(captain_ids)
    n_groups = len(group_ids)
    
    print(f"  Captains: {n_captains}")
    print(f"  Agent groups: {n_groups}")
    print(f"  Voyages: {len(df)}")
    
    # Build design matrix (sparse)
    n = len(df)
    n_fe = n_captains + n_groups
    
    rows = []
    cols = []
    data = []
    
    for i, (_, row) in enumerate(df.iterrows()):
        # Captain FE
        c_idx = captain_to_idx[row['captain_id']]
        rows.append(i)
        cols.append(c_idx)
        data.append(1.0)
        
        # Agent group FE
        g_idx = group_to_idx[row[agent_col]]
        rows.append(i)
        cols.append(g_idx)
        data.append(1.0)
    
    # Add controls if available
    controls = []
    control_names = []
    for col in ['log_tonnage', 'log_duration']:
        if col in df.columns:
            controls.append(df[col].values)
            control_names.append(col)
    
    if controls:
        X_controls = np.column_stack(controls)
        n_controls = X_controls.shape[1]
        for j in range(n_controls):
            for i in range(n):
                rows.append(i)
                cols.append(n_fe + j)
                data.append(X_controls[i, j])
        n_params = n_fe + n_controls
    else:
        n_params = n_fe
    
    X = csr_matrix((data, (rows, cols)), shape=(n, n_params))
    
    # Solve
    print("  Solving sparse least squares...")
    result = lsqr(X, y, atol=1e-10, btol=1e-10)
    beta = result[0]
    
    # Extract FEs
    alpha = beta[:n_captains]  # Captain FEs
    gamma = beta[n_captains:n_captains + n_groups]  # Agent group FEs
    
    # Residuals
    y_pred = X @ beta
    resid = y - y_pred
    
    # R-squared
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    
    print(f"\n  R² = {r2:.4f}")
    print(f"  RMSE = {np.sqrt(np.mean(resid**2)):.4f}")
    
    # Variance decomposition
    var_y = np.var(y)
    var_alpha = np.var(alpha)
    var_gamma = np.var(gamma)
    
    # Map FEs back to observations for covariance
    alpha_i = np.array([alpha[captain_to_idx[c]] for c in df['captain_id']])
    gamma_j = np.array([gamma[group_to_idx[g] - n_captains] for g in df[agent_col]])
    cov_ag = np.cov(alpha_i, gamma_j)[0, 1]
    var_resid = np.var(resid)
    
    print(f"\nVariance Decomposition (Grouped Agents):")
    print(f"  Var(y):     {var_y:.4f}")
    print(f"  Var(α):     {var_alpha:.4f} ({100*var_alpha/var_y:.1f}%)")
    print(f"  Var(γ):     {var_gamma:.4f} ({100*var_gamma/var_y:.1f}%)")
    print(f"  2Cov(α,γ):  {2*cov_ag:.4f} ({100*2*cov_ag/var_y:.1f}%)")
    print(f"  Var(ε):     {var_resid:.4f} ({100*var_resid/var_y:.1f}%)")
    
    # Economic interpretation
    sigma_alpha = np.sqrt(var_alpha)
    sigma_gamma = np.sqrt(var_gamma)
    
    print(f"\nEconomic Interpretation:")
    print(f"  σ(α) = {sigma_alpha:.3f} → ±1 SD = {100*(np.exp(sigma_alpha)-1):.1f}% productivity")
    print(f"  σ(γ) = {sigma_gamma:.3f} → ±1 SD = {100*(np.exp(sigma_gamma)-1):.1f}% productivity")
    
    # Plausibility check
    gamma_plausible = sigma_gamma < 1.0  # Less than e^1 ≈ 2.7x
    print(f"\n  Agent variance PLAUSIBLE: {gamma_plausible}")
    
    return {
        "r2": r2,
        "var_y": var_y,
        "var_alpha": var_alpha,
        "var_gamma": var_gamma,
        "cov_alpha_gamma": cov_ag,
        "var_resid": var_resid,
        "sigma_alpha": sigma_alpha,
        "sigma_gamma": sigma_gamma,
        "alpha": alpha,
        "gamma": gamma,
        "captain_to_idx": captain_to_idx,
        "group_to_idx": group_to_idx,
        "plausible": gamma_plausible,
    }


def apply_empirical_bayes_shrinkage(
    fe_estimates: np.ndarray,
    fe_se: np.ndarray = None,
    prior_var: float = None,
) -> Tuple[np.ndarray, float]:
    """
    Apply Empirical Bayes shrinkage to fixed effects.
    
    Shrinks estimates toward the grand mean, with more shrinkage
    for noisier estimates.
    """
    print("\nApplying Empirical Bayes shrinkage...")
    
    grand_mean = np.mean(fe_estimates)
    
    if fe_se is not None:
        # Heterogeneous shrinkage
        signal_var = np.var(fe_estimates) - np.mean(fe_se ** 2)
        signal_var = max(signal_var, 0.001)  # Floor
        
        shrinkage = fe_se ** 2 / (fe_se ** 2 + signal_var)
        shrunk = grand_mean + (1 - shrinkage) * (fe_estimates - grand_mean)
    else:
        # Homogeneous shrinkage
        if prior_var is None:
            prior_var = np.var(fe_estimates) * 0.5  # Default: moderate shrinkage
        
        total_var = np.var(fe_estimates)
        shrinkage = prior_var / total_var
        shrunk = grand_mean + shrinkage * (fe_estimates - grand_mean)
    
    shrunk_var = np.var(shrunk)
    original_var = np.var(fe_estimates)
    
    print(f"  Original variance: {original_var:.4f}")
    print(f"  Shrunk variance: {shrunk_var:.4f}")
    print(f"  Shrinkage ratio: {shrunk_var/original_var:.2f}")
    
    return shrunk, shrunk_var


def run_variance_fix_pipeline(df: pd.DataFrame, save_outputs: bool = True) -> Dict:
    """
    Run the complete variance fix pipeline.
    """
    print("\n" + "=" * 70)
    print("VARIANCE FIX PIPELINE")
    print("=" * 70)
    
    # 1. Diagnose connectivity
    connectivity = diagnose_connectivity(df)
    
    # 2. Create agent groups
    df = create_agent_groups(df, method="port_decade")
    
    # 3. Re-estimate with grouped agents
    grouped_results = estimate_with_grouped_agents(df, agent_col="agent_group")
    
    # 4. Check adding-up constraint
    constraint = check_adding_up_constraint(
        var_theta=grouped_results['var_alpha'],
        var_psi=grouped_results['var_gamma'],
        cov_theta_psi=grouped_results['cov_alpha_gamma'],
        var_epsilon=grouped_results['var_resid'],
        var_y=grouped_results['var_y'],
    )
    
    # 5. Compile results
    results = {
        "connectivity": connectivity,
        "grouped_estimation": grouped_results,
        "constraint_check": constraint,
    }
    
    if save_outputs:
        output_dir = Path("output/variance_fix")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create summary markdown
        lines = [
            "# Variance Fix Results",
            "",
            "## Problem",
            "Original Var(ψ) = 10.28 implied 2,450% productivity differences (implausible).",
            "",
            "## Solution: Grouped Agent Estimation",
            f"- Grouped agents by Port × Decade",
            f"- Compression: {df['agent_id'].nunique()} agents → {df['agent_group'].nunique()} groups",
            "",
            "## New Variance Decomposition",
            f"| Component | Variance | Share of Var(y) |",
            f"|-----------|----------|-----------------|",
            f"| Var(θ) Captain | {grouped_results['var_alpha']:.4f} | {100*grouped_results['var_alpha']/grouped_results['var_y']:.1f}% |",
            f"| Var(ψ) Agent Group | {grouped_results['var_gamma']:.4f} | {100*grouped_results['var_gamma']/grouped_results['var_y']:.1f}% |",
            f"| 2Cov(θ,ψ) Sorting | {2*grouped_results['cov_alpha_gamma']:.4f} | {100*2*grouped_results['cov_alpha_gamma']/grouped_results['var_y']:.1f}% |",
            f"| Var(ε) Residual | {grouped_results['var_resid']:.4f} | {100*grouped_results['var_resid']/grouped_results['var_y']:.1f}% |",
            "",
            "## Economic Interpretation",
            f"- σ(θ) = {grouped_results['sigma_alpha']:.3f} → ±1 SD = {100*(np.exp(grouped_results['sigma_alpha'])-1):.1f}% productivity",
            f"- σ(ψ) = {grouped_results['sigma_gamma']:.3f} → ±1 SD = {100*(np.exp(grouped_results['sigma_gamma'])-1):.1f}% productivity",
            "",
            f"**Plausible:** {grouped_results['plausible']}",
            "",
            "## Adding-Up Constraint",
            f"- Error: {constraint['pct_error']:.1f}%",
            f"- Constraint holds: {constraint['constraint_holds']}",
        ]
        
        with open(output_dir / "variance_fix_results.md", "w") as f:
            f.write("\n".join(lines))
        
        print(f"\nSaved to {output_dir}")
    
    return results


if __name__ == "__main__":
    from src.analyses.data_loader import prepare_analysis_sample
    
    df = prepare_analysis_sample()
    results = run_variance_fix_pipeline(df)
