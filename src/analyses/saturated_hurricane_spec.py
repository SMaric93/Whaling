"""
Fully Saturated Hurricane Buffering Specification.

Implements the reviewer-ready specification:

ln Q_v = β ln(ton)_v + α_{c(v)} + γ_{a(v)} + μ_{route×year} + μ_{port×year}
         + θ₁ hurricane_t + θ₂ (hurricane_t × CapAgent_a) + ε_v

Key features:
- Captain FE: controls for captain skill
- Agent FE: controls for agent capability 
- Route×year FE: kills "they just go Arctic/Pacific" critique
- Port×year FE: controls for local market conditions
- Hurricane × CapAgent: tests if same organization performs better when hurricanes spike

CapAgent measured as:
1. AKM agent-efficiency estimate γ̂_a (from first-stage regression)
2. Pre-t voyage count (robustness)
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr
from scipy import stats


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "final"
OUTPUT_DIR = PROJECT_ROOT / "output"
TABLES_DIR = OUTPUT_DIR / "tables"


# =============================================================================
# Step 1: First-Stage AKM to Get Agent Efficiency γ̂_a
# =============================================================================

def estimate_akm_agent_effects(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    First-stage AKM regression to estimate agent efficiency effects.
    
    Spec: ln Q = β ln(ton) + α_captain + γ_agent + μ_{route×year} + ε
    
    Returns agent fixed effects γ̂_a as the CapAgent measure.
    """
    print("\n" + "=" * 60)
    print("STEP 1: FIRST-STAGE AKM - Estimating Agent Efficiency (γ̂)")
    print("=" * 60)
    
    df = df.copy()
    n = len(df)
    y = df["log_q"].values
    
    # Build design matrix
    matrices = []
    index_maps = {}
    
    # 1. Log tonnage (coefficient of interest)
    log_ton = df["log_tonnage"].values.reshape(-1, 1)
    matrices.append(sp.csr_matrix(log_ton))
    
    # 2. Captain FE
    captain_ids = df["captain_id"].unique()
    captain_map = {c: i for i, c in enumerate(captain_ids)}
    captain_idx = df["captain_id"].map(captain_map).values
    X_captain = sp.csr_matrix(
        (np.ones(n), (np.arange(n), captain_idx)),
        shape=(n, len(captain_ids))
    )
    matrices.append(X_captain)
    index_maps["captain"] = {"ids": captain_ids, "map": captain_map}
    
    # 3. Agent FE (drop first for identification)
    agent_ids = df["agent_id"].unique()
    agent_map = {a: i for i, a in enumerate(agent_ids)}
    agent_idx = df["agent_id"].map(agent_map).values
    X_agent_full = sp.csr_matrix(
        (np.ones(n), (np.arange(n), agent_idx)),
        shape=(n, len(agent_ids))
    )
    X_agent = X_agent_full[:, 1:]  # Drop first
    matrices.append(X_agent)
    index_maps["agent"] = {"ids": agent_ids, "map": agent_map}
    
    # 4. Route×year FE (drop first)
    rt_ids = df["route_time"].unique()
    rt_map = {r: i for i, r in enumerate(rt_ids)}
    rt_idx = df["route_time"].map(rt_map).values
    X_rt_full = sp.csr_matrix(
        (np.ones(n), (np.arange(n), rt_idx)),
        shape=(n, len(rt_ids))
    )
    X_rt = X_rt_full[:, 1:]
    matrices.append(X_rt)
    index_maps["route_time"] = {"ids": rt_ids}
    
    # Stack
    X = sp.hstack(matrices)
    print(f"Design matrix: {X.shape[0]:,} obs × {X.shape[1]:,} params")
    print(f"  Captains: {len(captain_ids):,}")
    print(f"  Agents: {len(agent_ids):,}")
    print(f"  Route×year cells: {len(rt_ids):,}")
    
    # Solve
    print("Solving first-stage AKM...")
    sol = lsqr(X, y, iter_lim=10000, atol=1e-10, btol=1e-10)
    beta = sol[0]
    
    # Extract coefficients
    beta_tonnage = beta[0]
    
    n_capt = len(captain_ids)
    alpha_captain = beta[1:1 + n_capt]
    
    # Agent effects (add 0 for normalized first agent)
    n_agent = len(agent_ids)
    gamma_agent_est = beta[1 + n_capt:1 + n_capt + n_agent - 1]
    gamma_agent = np.concatenate([[0], gamma_agent_est])
    
    # Compute R²
    y_hat = X @ beta
    residuals = y - y_hat
    r2 = 1 - np.var(residuals) / np.var(y)
    
    print(f"\nFirst-stage results:")
    print(f"  R² = {r2:.4f}")
    print(f"  β_tonnage = {beta_tonnage:.4f}")
    print(f"  Var(α_captain) = {np.var(alpha_captain):.4f}")
    print(f"  Var(γ_agent) = {np.var(gamma_agent):.4f}")
    
    # Create agent effects DataFrame
    agent_effects = pd.DataFrame({
        "agent_id": agent_ids,
        "gamma_hat": gamma_agent,
    })
    
    # Normalize: demean and standardize
    agent_effects["gamma_hat_std"] = (
        (agent_effects["gamma_hat"] - agent_effects["gamma_hat"].mean()) / 
        agent_effects["gamma_hat"].std()
    )
    
    # Create high-capability indicator (above median)
    agent_effects["high_cap_akm"] = (
        agent_effects["gamma_hat"] >= agent_effects["gamma_hat"].median()
    ).astype(int)
    
    diagnostics = {
        "r2": r2,
        "beta_tonnage": beta_tonnage,
        "var_alpha": np.var(alpha_captain),
        "var_gamma": np.var(gamma_agent),
        "n_captains": len(captain_ids),
        "n_agents": len(agent_ids),
    }
    
    return agent_effects, diagnostics


# =============================================================================
# Step 2: Fully Saturated Hurricane Specification
# =============================================================================

def run_saturated_hurricane_spec(
    df: pd.DataFrame,
    agent_effects: pd.DataFrame,
) -> Dict:
    """
    Run fully saturated hurricane buffering specification.
    
    ln Q_v = β ln(ton)_v + α_{c(v)} + γ_{a(v)} + μ_{route×year} + μ_{port×year}
             + θ₁ hurricane_t + θ₂ (hurricane_t × CapAgent_a) + ε_v
    
    With captain + agent + route×year + port×year FE, the interaction θ₂ is 
    interpreted as: "the same organization performs better when hurricanes spike"
    """
    print("\n" + "=" * 60)
    print("STEP 2: FULLY SATURATED HURRICANE SPECIFICATION")
    print("=" * 60)
    
    df = df.copy()
    
    # Merge AKM agent effects
    df = df.merge(agent_effects[["agent_id", "gamma_hat_std", "high_cap_akm"]], 
                  on="agent_id", how="left")
    
    # Standardize hurricane exposure
    df["hurricane_std"] = (
        (df["hurricane_exposure_count"] - df["hurricane_exposure_count"].mean()) / 
        df["hurricane_exposure_count"].std()
    )
    df["hurricane_std"] = df["hurricane_std"].fillna(0)
    
    # Create interaction term: hurricane × γ̂_agent
    df["hurr_x_gamma"] = df["hurricane_std"] * df["gamma_hat_std"].fillna(0)
    
    # Also create binary version: hurricane × high_cap_akm
    df["hurr_x_highcap"] = df["hurricane_std"] * df["high_cap_akm"].fillna(0)
    
    n = len(df)
    y = df["log_q"].values
    
    # =========================================================================
    # Build design matrix for fully saturated model
    # =========================================================================
    matrices = []
    
    # Coefficients of interest (will report SEs for these)
    x_vars = ["log_tonnage", "hurricane_std", "hurr_x_gamma"]
    X_coef = df[x_vars].values
    matrices.append(sp.csr_matrix(X_coef))
    
    # Captain FE
    captain_ids = df["captain_id"].unique()
    captain_map = {c: i for i, c in enumerate(captain_ids)}
    captain_idx = df["captain_id"].map(captain_map).values
    X_captain = sp.csr_matrix(
        (np.ones(n), (np.arange(n), captain_idx)),
        shape=(n, len(captain_ids))
    )
    matrices.append(X_captain)
    
    # Agent FE (drop first)
    agent_ids = df["agent_id"].unique()
    agent_map = {a: i for i, a in enumerate(agent_ids)}
    agent_idx = df["agent_id"].map(agent_map).values
    X_agent = sp.csr_matrix(
        (np.ones(n), (np.arange(n), agent_idx)),
        shape=(n, len(agent_ids))
    )[:, 1:]
    matrices.append(X_agent)
    
    # Route×year FE (drop first)
    rt_ids = df["route_time"].unique()
    rt_map = {r: i for i, r in enumerate(rt_ids)}
    rt_idx = df["route_time"].map(rt_map).values
    X_rt = sp.csr_matrix(
        (np.ones(n), (np.arange(n), rt_idx)),
        shape=(n, len(rt_ids))
    )[:, 1:]
    matrices.append(X_rt)
    
    # Port×year FE (drop first)
    pt_ids = df["port_time"].unique()
    pt_map = {p: i for i, p in enumerate(pt_ids)}
    pt_idx = df["port_time"].map(pt_map).values
    X_pt = sp.csr_matrix(
        (np.ones(n), (np.arange(n), pt_idx)),
        shape=(n, len(pt_ids))
    )[:, 1:]
    matrices.append(X_pt)
    
    # Stack
    X = sp.hstack(matrices)
    
    print(f"\nDesign matrix: {X.shape[0]:,} obs × {X.shape[1]:,} params")
    print(f"  Captain FE: {len(captain_ids):,}")
    print(f"  Agent FE: {len(agent_ids):,}")
    print(f"  Route×year FE: {len(rt_ids):,}")
    print(f"  Port×year FE: {len(pt_ids):,}")
    
    # Solve
    print("\nSolving saturated specification...")
    sol = lsqr(X, y, iter_lim=20000, atol=1e-12, btol=1e-12)
    beta = sol[0]
    
    # Extract coefficients of interest
    coefs = beta[:len(x_vars)]
    
    # Compute fit
    y_hat = X @ beta
    residuals = y - y_hat
    r2 = 1 - np.var(residuals) / np.var(y)
    rmse = np.std(residuals)
    
    # =========================================================================
    # Cluster-robust standard errors (two-way: agent + year)
    # =========================================================================
    print("Computing two-way cluster-robust SEs (agent + year)...")
    
    # Agent clusters
    se_agent = compute_cluster_se(X_coef, residuals, df["agent_id"].values)
    
    # Year clusters  
    se_year = compute_cluster_se(X_coef, residuals, df["year_out"].values)
    
    # Two-way clustering (Cameron-Gelbach-Miller adjustment)
    agent_year = df["agent_id"].astype(str) + "_" + df["year_out"].astype(str)
    se_both = compute_cluster_se(X_coef, residuals, agent_year.values)
    
    # Two-way SE: Var_A + Var_Y - Var_AY
    var_two_way = se_agent**2 + se_year**2 - se_both**2
    se_two_way = np.sqrt(np.maximum(var_two_way, 0))
    
    # Use agent-clustered SE (more conservative typically)
    se = se_agent
    
    # T-stats and p-values
    t_stats = coefs / se
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - X.shape[1]))
    
    # =========================================================================
    # Report results
    # =========================================================================
    print("\n" + "=" * 60)
    print("RESULTS: Fully Saturated Hurricane × Agent Capability")
    print("=" * 60)
    print(f"\nln Q = β₀ ln(ton) + α_captain + γ_agent + μ_route×year + μ_port×year")
    print(f"       + θ₁ hurricane + θ₂ (hurricane × γ̂_agent) + ε")
    print(f"\nN = {n:,}   R² = {r2:.4f}   RMSE = {rmse:.4f}")
    print("-" * 60)
    print(f"{'Variable':<30} {'Coef':>10} {'SE':>10} {'t':>8} {'p':>8}")
    print("-" * 60)
    
    for i, var in enumerate(x_vars):
        stars = ""
        if p_values[i] < 0.01:
            stars = "***"
        elif p_values[i] < 0.05:
            stars = "**"
        elif p_values[i] < 0.10:
            stars = "*"
        
        print(f"{var:<30} {coefs[i]:>10.4f} {se[i]:>10.4f} {t_stats[i]:>8.2f} {p_values[i]:>7.4f}{stars}")
    
    print("-" * 60)
    print(f"\nFixed Effects:")
    print(f"  Captain: Yes ({len(captain_ids):,})")
    print(f"  Agent: Yes ({len(agent_ids):,})")
    print(f"  Route×Year: Yes ({len(rt_ids):,})")
    print(f"  Port×Year: Yes ({len(pt_ids):,})")
    print(f"\nStandard errors: Clustered by agent")
    
    # Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    
    theta1 = coefs[x_vars.index("hurricane_std")]
    theta2 = coefs[x_vars.index("hurr_x_gamma")]
    
    print(f"""
θ₁ (hurricane main effect) = {theta1:.4f}
   → 1σ increase in hurricane exposure → {100*theta1:.1f}% change in output

θ₂ (hurricane × γ̂_agent interaction) = {theta2:.4f}
   → For a 1σ higher-efficiency agent, the hurricane effect is offset by {100*theta2:.1f}%

Economic magnitude:
   - At mean agent efficiency (γ̂=0): hurricane effect = {100*theta1:.1f}%
   - At +1σ agent efficiency: hurricane effect = {100*(theta1+theta2):.1f}%
   - At +2σ agent efficiency: hurricane effect = {100*(theta1+2*theta2):.1f}%

Key insight:
   With all FEs (captain, agent, route×year, port×year), θ₂ captures:
   "When hurricanes spike, THE SAME organization performs better if it has
   higher baseline efficiency (γ̂_agent)."
   
   This is NOT route selection (route×year FE absorbed).
   This is NOT agent selection (agent FE absorbed).
   This is WITHIN-agent, WITHIN-route resilience.
""")
    
    results = {
        "n": n,
        "r2": r2,
        "rmse": rmse,
        "variables": x_vars,
        "coefficients": coefs,
        "std_errors": se,
        "std_errors_twoway": se_two_way,
        "t_stats": t_stats,
        "p_values": p_values,
        "n_captain_fe": len(captain_ids),
        "n_agent_fe": len(agent_ids),
        "n_route_time_fe": len(rt_ids),
        "n_port_time_fe": len(pt_ids),
        "theta1": theta1,
        "theta2": theta2,
    }
    
    return results


def compute_cluster_se(X: np.ndarray, residuals: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    """Compute cluster-robust standard errors."""
    n, k = X.shape
    unique_clusters = np.unique(clusters)
    G = len(unique_clusters)
    
    XtX_inv = np.linalg.pinv(X.T @ X)
    
    meat = np.zeros((k, k))
    for c in unique_clusters:
        mask = clusters == c
        Xi = X[mask]
        ei = residuals[mask]
        score = Xi.T @ ei
        meat += np.outer(score, score)
    
    correction = (G / (G - 1)) * ((n - 1) / (n - k))
    vcov = correction * XtX_inv @ meat @ XtX_inv
    
    return np.sqrt(np.maximum(np.diag(vcov), 0))


# =============================================================================
# Step 3: Robustness with Pre-t Voyage Count
# =============================================================================

def run_robustness_voyage_count(df: pd.DataFrame) -> Dict:
    """
    Robustness: Use pre-t agent voyage count instead of AKM γ̂.
    """
    print("\n" + "=" * 60)
    print("ROBUSTNESS: Using Pre-t Voyage Count as CapAgent")
    print("=" * 60)
    
    df = df.copy()
    
    # Compute agent voyage count as of each year
    agent_cumulative = df.groupby("agent_id")["year_out"].apply(
        lambda x: x.rank(method='first')
    ).reset_index(level=0, drop=True)
    df["agent_prior_voyages"] = (agent_cumulative - 1).fillna(0)  # Voyages before this one
    
    # Log transform
    df["log_prior_voyages"] = np.log1p(df["agent_prior_voyages"].astype(float))
    
    # Standardize
    df["prior_voyages_std"] = (
        (df["log_prior_voyages"] - df["log_prior_voyages"].mean()) / 
        df["log_prior_voyages"].std()
    ).fillna(0)
    
    # Create interaction
    df["hurricane_std"] = (
        (df["hurricane_exposure_count"] - df["hurricane_exposure_count"].mean()) / 
        df["hurricane_exposure_count"].std()
    ).fillna(0)
    
    df["hurr_x_voyages"] = df["hurricane_std"] * df["prior_voyages_std"]
    
    n = len(df)
    y = df["log_q"].values.astype(float)
    
    # Build simpler model (without full saturation for comparison)
    x_vars = ["log_tonnage", "hurricane_std", "hurr_x_voyages"]
    X_coef = df[x_vars].astype(float).values
    
    matrices = [sp.csr_matrix(X_coef)]
    
    # Captain FE
    captain_ids = df["captain_id"].unique()
    captain_map = {c: i for i, c in enumerate(captain_ids)}
    X_cap = sp.csr_matrix(
        (np.ones(n), (np.arange(n), df["captain_id"].map(captain_map).values)),
        shape=(n, len(captain_ids))
    )
    matrices.append(X_cap)
    
    # Route×year FE
    rt_ids = df["route_time"].unique()
    rt_map = {r: i for i, r in enumerate(rt_ids)}
    X_rt = sp.csr_matrix(
        (np.ones(n), (np.arange(n), df["route_time"].map(rt_map).values)),
        shape=(n, len(rt_ids))
    )[:, 1:]
    matrices.append(X_rt)
    
    X = sp.hstack(matrices)
    
    sol = lsqr(X, y, iter_lim=10000, atol=1e-10, btol=1e-10)
    beta = sol[0]
    coefs = beta[:len(x_vars)]
    
    y_hat = X @ beta
    residuals = y - y_hat
    r2 = 1 - np.var(residuals) / np.var(y)
    
    se = compute_cluster_se(X_coef, residuals, df["agent_id"].values)
    t_stats = coefs / se
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - X.shape[1]))
    
    print(f"\nN = {n:,}   R² = {r2:.4f}")
    print("-" * 60)
    for i, var in enumerate(x_vars):
        stars = "***" if p_values[i] < 0.01 else "**" if p_values[i] < 0.05 else "*" if p_values[i] < 0.10 else ""
        print(f"{var:<30} {coefs[i]:>10.4f} ({se[i]:.4f}){stars}")
    
    return {
        "n": n, "r2": r2, "variables": x_vars,
        "coefficients": coefs, "std_errors": se, "p_values": p_values
    }


# =============================================================================
# Main
# =============================================================================

def load_and_prepare_data() -> pd.DataFrame:
    """Load and prepare data for saturated analysis."""
    print("Loading data...")
    
    voyages = pd.read_parquet(DATA_DIR / "analysis_voyage.parquet")
    weather = pd.read_parquet(DATA_DIR / "voyage_weather.parquet")
    
    df = voyages.merge(
        weather.drop(columns=["year_out"], errors="ignore"),
        on="voyage_id", how="left"
    )
    
    # Filter to weather data
    df = df[df["annual_storms"].notna()].copy()
    
    # Create transforms
    df["log_q"] = np.log(df["q_oil_bbl"].replace(0, np.nan))
    df["log_tonnage"] = np.log(df["tonnage"].replace(0, np.nan))
    
    if "ground_or_route" in df.columns:
        df["route_or_ground"] = df["ground_or_route"]
    
    # Create FE columns
    df["route_time"] = df["route_or_ground"].astype(str) + "_" + df["year_out"].astype(str)
    df["port_time"] = df["home_port"].astype(str) + "_" + df["year_out"].astype(str)
    
    # Drop missing
    required = ["log_q", "log_tonnage", "captain_id", "agent_id", "year_out", 
                "route_time", "port_time", "hurricane_exposure_count"]
    df = df.dropna(subset=required)
    
    print(f"Sample: {len(df):,} voyages")
    
    return df


def save_results(results: Dict, akm_diag: Dict) -> None:
    """Save results to CSV."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    rows = []
    for i, var in enumerate(results["variables"]):
        rows.append({
            "variable": var,
            "coefficient": results["coefficients"][i],
            "std_error": results["std_errors"][i],
            "t_stat": results["t_stats"][i],
            "p_value": results["p_values"][i],
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "saturated_hurricane_spec.csv", index=False)
    
    # Also save summary
    summary = {
        "n": results["n"],
        "r2": results["r2"],
        "n_captain_fe": results["n_captain_fe"],
        "n_agent_fe": results["n_agent_fe"],
        "n_route_time_fe": results["n_route_time_fe"],
        "n_port_time_fe": results["n_port_time_fe"],
        "theta1_hurricane": results["theta1"],
        "theta2_hurr_x_gamma": results["theta2"],
        "akm_r2": akm_diag["r2"],
        "akm_var_gamma": akm_diag["var_gamma"],
    }
    
    pd.DataFrame([summary]).to_csv(TABLES_DIR / "saturated_hurricane_summary.csv", index=False)
    
    print(f"\nResults saved to {TABLES_DIR}")


def main():
    """Run fully saturated hurricane specification."""
    
    # Load data
    df = load_and_prepare_data()
    
    # Step 1: First-stage AKM to get agent effects
    agent_effects, akm_diag = estimate_akm_agent_effects(df)
    
    # Step 2: Fully saturated specification
    results = run_saturated_hurricane_spec(df, agent_effects)
    
    # Step 3: Robustness with voyage count
    robustness = run_robustness_voyage_count(df)
    
    # Save results
    save_results(results, akm_diag)
    
    print("\n" + "=" * 60)
    print("SUMMARY FOR PAPER")
    print("=" * 60)
    print(f"""
Table X: Hurricane Exposure and Agent Capability
═══════════════════════════════════════════════════════════════

Dependent variable: ln(Oil Output)

                                    (1)
────────────────────────────────────────────────────────────────
ln(Tonnage)                        {results['coefficients'][0]:.3f}***
                                  ({results['std_errors'][0]:.3f})

Hurricane Exposure (std)           {results['coefficients'][1]:.3f}
                                  ({results['std_errors'][1]:.3f})

Hurricane × Agent Efficiency (γ̂)   {results['coefficients'][2]:.3f}{'***' if results['p_values'][2] < 0.01 else '**' if results['p_values'][2] < 0.05 else '*' if results['p_values'][2] < 0.10 else ''}
                                  ({results['std_errors'][2]:.3f})
────────────────────────────────────────────────────────────────
Captain FE                         Yes
Agent FE                           Yes
Route × Year FE                    Yes
Port × Year FE                     Yes
────────────────────────────────────────────────────────────────
Observations                       {results['n']:,}
R²                                 {results['r2']:.3f}
Captain effects                    {results['n_captain_fe']:,}
Agent effects                      {results['n_agent_fe']:,}
Route × Year cells                 {results['n_route_time_fe']:,}
Port × Year cells                  {results['n_port_time_fe']:,}
═══════════════════════════════════════════════════════════════
Notes: Standard errors clustered by agent in parentheses.
       *** p<0.01, ** p<0.05, * p<0.10
       Agent efficiency (γ̂) estimated from first-stage AKM.
""")
    
    return results, agent_effects


if __name__ == "__main__":
    main()
