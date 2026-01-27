"""
Exposure Test and Ex-Ante Capability Proxies.

Two approaches that avoid outcome contamination:

1. EXPOSURE TEST: Are better agents assigned to more hurricane exposure?
   hurricane_v = π γ̂^CF + μ_{route×year} + μ_{port×year} + f(ln ton) + ε
   
   If π > 0: better agents systematically get more hurricane-exposed voyages

2. EX-ANTE CAPABILITY PROXIES (not built from Q):
   - Network centrality (degree in captain-agent mobility graph)
   - Portfolio breadth (routes/ports handled pre-t)
   - Operational complexity (Arctic/Pacific/long voyage share pre-t)
   - Repeat-relationship intensity (repeat voyage share)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
# Data Loading
# =============================================================================

def load_data() -> pd.DataFrame:
    """Load and prepare data."""
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
    
    df["route_time"] = df["route_or_ground"].astype(str) + "_" + df["year_out"].astype(str)
    df["port_time"] = df["home_port"].astype(str) + "_" + df["year_out"].astype(str)
    df["decade"] = (df["year_out"] // 10 * 10).astype(int)
    
    # Standardize hurricane
    df["hurricane_std"] = (
        (df["hurricane_exposure_count"] - df["hurricane_exposure_count"].mean()) / 
        df["hurricane_exposure_count"].std()
    ).fillna(0)
    
    # Drop missing
    required = ["log_q", "log_tonnage", "captain_id", "agent_id", "year_out", "route_time"]
    df = df.dropna(subset=required)
    
    print(f"Sample: {len(df):,} voyages")
    
    return df


# =============================================================================
# Ex-Ante Capability Proxies
# =============================================================================

def compute_ex_ante_proxies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ex-ante agent capability proxies that don't use Q.
    
    All proxies computed using data strictly before each voyage's year.
    """
    print("\n" + "=" * 60)
    print("COMPUTING EX-ANTE CAPABILITY PROXIES")
    print("=" * 60)
    
    df = df.copy()
    
    # Sort by year for rolling computations
    df = df.sort_values(["agent_id", "year_out"])
    
    # Initialize proxy columns
    df["network_degree"] = np.nan
    df["portfolio_breadth"] = np.nan
    df["operational_complexity"] = np.nan
    df["repeat_intensity"] = np.nan
    
    # For each year, compute proxies using only prior data
    years = sorted(df["year_out"].unique())
    
    for year in years:
        # Get all prior data
        prior_data = df[df["year_out"] < year]
        
        if len(prior_data) < 50:
            continue
        
        # 1. Network degree: unique captains worked with
        network = prior_data.groupby("agent_id")["captain_id"].nunique()
        
        # 2. Portfolio breadth: unique routes × ports
        routes = prior_data.groupby("agent_id")["route_or_ground"].nunique()
        ports = prior_data.groupby("agent_id")["home_port"].nunique()
        portfolio = routes + ports
        
        # 3. Operational complexity: fraction of Arctic/Pacific/long voyages
        if "frac_days_in_arctic_polygon" in prior_data.columns:
            arctic_share = prior_data.groupby("agent_id")["frac_days_in_arctic_polygon"].mean()
        else:
            # Fallback: check if route contains Arctic/Pacific
            prior_data_copy = prior_data.copy()
            prior_data_copy["is_complex"] = prior_data_copy["route_or_ground"].str.contains(
                "Arctic|Pacific|Bering|Okhotsk|Japan", case=False, na=False
            ).astype(float)
            arctic_share = prior_data_copy.groupby("agent_id")["is_complex"].mean()
        
        # 4. Repeat intensity: fraction of captain-agent pairs that are repeats
        # Count voyages per captain-agent pair
        pair_counts = prior_data.groupby(["agent_id", "captain_id"]).size().reset_index(name="n")
        pair_counts["is_repeat"] = (pair_counts["n"] > 1).astype(float)
        repeat_share = pair_counts.groupby("agent_id")["is_repeat"].mean()
        
        # Apply to voyages in this year
        mask = df["year_out"] == year
        
        df.loc[mask, "network_degree"] = df.loc[mask, "agent_id"].map(network)
        df.loc[mask, "portfolio_breadth"] = df.loc[mask, "agent_id"].map(portfolio)
        df.loc[mask, "operational_complexity"] = df.loc[mask, "agent_id"].map(arctic_share)
        df.loc[mask, "repeat_intensity"] = df.loc[mask, "agent_id"].map(repeat_share)
    
    # Standardize proxies
    for proxy in ["network_degree", "portfolio_breadth", "operational_complexity", "repeat_intensity"]:
        col = f"{proxy}_std"
        valid = df[proxy].notna()
        if valid.sum() > 0:
            df[col] = (df[proxy] - df.loc[valid, proxy].mean()) / df.loc[valid, proxy].std()
        else:
            df[col] = np.nan
    
    # Report coverage
    print(f"\nProxy coverage:")
    for proxy in ["network_degree", "portfolio_breadth", "operational_complexity", "repeat_intensity"]:
        n = df[proxy].notna().sum()
        print(f"  {proxy}: {n}/{len(df)} ({100*n/len(df):.1f}%)")
    
    return df


def estimate_cross_fitted_gamma(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate cross-fitted γ̂ using pre-period data."""
    print("\n" + "=" * 60)
    print("ESTIMATING CROSS-FITTED γ̂ (Pre-1880)")
    print("=" * 60)
    
    df = df.copy()
    cutoff = 1880
    
    pre_mask = df["year_out"] <= cutoff
    pre_data = df[pre_mask]
    
    print(f"Pre-period sample: {len(pre_data):,} voyages")
    
    if len(pre_data) < 100:
        print("Insufficient data for γ̂ estimation")
        df["gamma_cf"] = np.nan
        return df
    
    n = len(pre_data)
    y = pre_data["log_q"].values.astype(float)
    
    # Build design
    matrices = []
    
    # Log tonnage
    matrices.append(sp.csr_matrix(pre_data["log_tonnage"].values.reshape(-1, 1)))
    
    # Captain FE
    captain_ids = pre_data["captain_id"].unique()
    captain_map = {c: i for i, c in enumerate(captain_ids)}
    X_cap = sp.csr_matrix(
        (np.ones(n), (np.arange(n), pre_data["captain_id"].map(captain_map).values)),
        shape=(n, len(captain_ids))
    )
    matrices.append(X_cap)
    
    # Agent FE
    agent_ids = pre_data["agent_id"].unique()
    agent_map = {a: i for i, a in enumerate(agent_ids)}
    X_agent = sp.csr_matrix(
        (np.ones(n), (np.arange(n), pre_data["agent_id"].map(agent_map).values)),
        shape=(n, len(agent_ids))
    )[:, 1:]
    matrices.append(X_agent)
    
    # Route×year FE
    rt_ids = pre_data["route_time"].unique()
    rt_map = {r: i for i, r in enumerate(rt_ids)}
    X_rt = sp.csr_matrix(
        (np.ones(n), (np.arange(n), pre_data["route_time"].map(rt_map).values)),
        shape=(n, len(rt_ids))
    )[:, 1:]
    matrices.append(X_rt)
    
    X = sp.hstack(matrices)
    
    sol = lsqr(X, y, iter_lim=10000, atol=1e-10, btol=1e-10)
    beta = sol[0]
    
    # Extract agent effects
    n_capt = len(captain_ids)
    n_agent = len(agent_ids)
    gamma_est = beta[1 + n_capt:1 + n_capt + n_agent - 1]
    gamma = np.concatenate([[0], gamma_est])
    
    # Create agent effects DataFrame
    agent_effects = pd.DataFrame({
        "agent_id": agent_ids,
        "gamma_hat": gamma,
    })
    
    # Standardize
    agent_effects["gamma_cf"] = (
        (agent_effects["gamma_hat"] - agent_effects["gamma_hat"].mean()) / 
        agent_effects["gamma_hat"].std()
    )
    
    print(f"Estimated γ̂ for {len(agent_effects)} agents")
    
    # Merge to full data
    df = df.merge(agent_effects[["agent_id", "gamma_cf"]], on="agent_id", how="left")
    
    n_matched = df["gamma_cf"].notna().sum()
    print(f"Voyages with γ̂^CF: {n_matched}/{len(df)} ({100*n_matched/len(df):.1f}%)")
    
    return df


# =============================================================================
# Exposure Test
# =============================================================================

def run_exposure_test(df: pd.DataFrame, capability_var: str, label: str) -> Dict:
    """
    Test: Are higher-capability agents assigned to more hurricane exposure?
    
    hurricane_v = π capability_a + μ_{route×year} + μ_{port×year} + β ln(ton) + ε
    
    If π > 0: better agents get more hurricane-exposed voyages
    """
    df = df.copy()
    df = df.dropna(subset=[capability_var, "hurricane_std", "log_tonnage", "route_time", "port_time"])
    
    if len(df) < 100:
        return None
    
    n = len(df)
    y = df["hurricane_std"].values.astype(float)
    
    # Covariates
    x_vars = [capability_var, "log_tonnage"]
    X_coef = df[x_vars].astype(float).values
    
    matrices = [sp.csr_matrix(X_coef)]
    
    # Route×year FE
    rt_ids = df["route_time"].unique()
    rt_map = {r: i for i, r in enumerate(rt_ids)}
    X_rt = sp.csr_matrix(
        (np.ones(n), (np.arange(n), df["route_time"].map(rt_map).values)),
        shape=(n, len(rt_ids))
    )[:, 1:]
    matrices.append(X_rt)
    
    # Port×year FE
    pt_ids = df["port_time"].unique()
    pt_map = {p: i for i, p in enumerate(pt_ids)}
    X_pt = sp.csr_matrix(
        (np.ones(n), (np.arange(n), df["port_time"].map(pt_map).values)),
        shape=(n, len(pt_ids))
    )[:, 1:]
    matrices.append(X_pt)
    
    X = sp.hstack(matrices)
    
    sol = lsqr(X, y, iter_lim=10000, atol=1e-10, btol=1e-10)
    beta = sol[0]
    coefs = beta[:len(x_vars)]
    
    y_hat = X @ beta
    residuals = y - y_hat
    r2 = 1 - np.var(residuals) / np.var(y)
    
    # Cluster-robust SE by agent
    clusters = df["agent_id"].values
    unique_clusters = np.unique(clusters)
    G = len(unique_clusters)
    
    if G > len(x_vars):
        XtX_inv = np.linalg.pinv(X_coef.T @ X_coef)
        meat = np.zeros((len(x_vars), len(x_vars)))
        for c in unique_clusters:
            mask = clusters == c
            Xi = X_coef[mask]
            ei = residuals[mask]
            score = Xi.T @ ei
            meat += np.outer(score, score)
        
        correction = (G / (G - 1)) * ((n - 1) / (n - len(x_vars)))
        vcov = correction * XtX_inv @ meat @ XtX_inv
        se = np.sqrt(np.maximum(np.diag(vcov), 0))
    else:
        se = np.full(len(x_vars), np.nan)
    
    t_stats = coefs / np.where(se > 0, se, np.nan)
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=max(n - X.shape[1], 1)))
    
    return {
        "label": label,
        "n": n,
        "r2": r2,
        "variables": x_vars,
        "coefficients": coefs,
        "std_errors": se,
        "t_stats": t_stats,
        "p_values": p_values,
        "pi": coefs[0],
        "pi_se": se[0],
        "pi_t": t_stats[0],
        "pi_p": p_values[0],
    }


def run_outcome_test(df: pd.DataFrame, capability_var: str, label: str) -> Dict:
    """
    Test: Does capability predict productivity after controlling for hurricane?
    
    ln Q_v = β ln(ton) + θ capability + φ hurricane + μ_{route×year} + α_captain + ε
    
    If θ > 0: capability predicts productivity beyond hurricane exposure
    """
    df = df.copy()
    df = df.dropna(subset=[capability_var, "hurricane_std", "log_q", "log_tonnage", "route_time"])
    
    if len(df) < 100:
        return None
    
    n = len(df)
    y = df["log_q"].values.astype(float)
    
    # Covariates
    x_vars = ["log_tonnage", capability_var, "hurricane_std"]
    X_coef = df[x_vars].astype(float).values
    
    matrices = [sp.csr_matrix(X_coef)]
    
    # Captain FE
    cap_ids = df["captain_id"].unique()
    cap_map = {c: i for i, c in enumerate(cap_ids)}
    X_cap = sp.csr_matrix(
        (np.ones(n), (np.arange(n), df["captain_id"].map(cap_map).values)),
        shape=(n, len(cap_ids))
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
    
    # Cluster-robust SE
    clusters = df["captain_id"].values
    unique_clusters = np.unique(clusters)
    G = len(unique_clusters)
    
    if G > len(x_vars):
        XtX_inv = np.linalg.pinv(X_coef.T @ X_coef)
        meat = np.zeros((len(x_vars), len(x_vars)))
        for c in unique_clusters:
            mask = clusters == c
            Xi = X_coef[mask]
            ei = residuals[mask]
            score = Xi.T @ ei
            meat += np.outer(score, score)
        
        correction = (G / (G - 1)) * ((n - 1) / (n - len(x_vars)))
        vcov = correction * XtX_inv @ meat @ XtX_inv
        se = np.sqrt(np.maximum(np.diag(vcov), 0))
    else:
        se = np.full(len(x_vars), np.nan)
    
    t_stats = coefs / np.where(se > 0, se, np.nan)
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=max(n - X.shape[1], 1)))
    
    return {
        "label": label,
        "n": n,
        "r2": r2,
        "variables": x_vars,
        "coefficients": coefs,
        "std_errors": se,
        "t_stats": t_stats,
        "p_values": p_values,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    """Run exposure tests and ex-ante proxy analysis."""
    
    # Load data
    df = load_data()
    
    # Compute ex-ante proxies
    df = compute_ex_ante_proxies(df)
    
    # Estimate cross-fitted γ̂
    df = estimate_cross_fitted_gamma(df)
    
    # =========================================================================
    # Part 1: Exposure Tests
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 1: EXPOSURE TESTS")
    print("Does higher capability → more hurricane exposure?")
    print("=" * 70)
    
    exposure_results = {}
    
    # Test with cross-fitted γ̂
    exposure_results["gamma_cf"] = run_exposure_test(
        df, "gamma_cf", "Cross-Fitted γ̂ (Pre-1880)"
    )
    
    # Test with ex-ante proxies
    for proxy in ["network_degree_std", "portfolio_breadth_std", 
                  "operational_complexity_std", "repeat_intensity_std"]:
        exposure_results[proxy] = run_exposure_test(df, proxy, proxy.replace("_std", ""))
    
    # Report
    print("\n" + "-" * 70)
    print(f"{'Capability Measure':<35} {'π':>10} {'SE':>10} {'t':>8} {'N':>8}")
    print("-" * 70)
    
    for key, res in exposure_results.items():
        if res is not None:
            stars = "***" if res["pi_p"] < 0.01 else "**" if res["pi_p"] < 0.05 else "*" if res["pi_p"] < 0.10 else ""
            print(f"{res['label']:<35} {res['pi']:>10.4f} {res['pi_se']:>10.4f} {res['pi_t']:>8.2f}{stars} {res['n']:>7}")
        else:
            print(f"{key:<35} (insufficient data)")
    
    # =========================================================================
    # Part 2: Outcome Tests with Ex-Ante Proxies
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 2: PRODUCTIVITY TESTS WITH EX-ANTE PROXIES")
    print("Does capability predict ln Q after controlling for hurricane?")
    print("=" * 70)
    
    outcome_results = {}
    
    for proxy in ["network_degree_std", "portfolio_breadth_std", 
                  "operational_complexity_std", "repeat_intensity_std"]:
        outcome_results[proxy] = run_outcome_test(df, proxy, proxy.replace("_std", ""))
    
    # Report
    print("\n" + "-" * 70)
    print(f"{'Capability Measure':<35} {'θ (cap)':>10} {'φ (hurr)':>10} {'N':>8}")
    print("-" * 70)
    
    for key, res in outcome_results.items():
        if res is not None:
            cap_idx = res["variables"].index(key)
            hurr_idx = res["variables"].index("hurricane_std")
            
            cap_stars = "***" if res["p_values"][cap_idx] < 0.01 else "**" if res["p_values"][cap_idx] < 0.05 else "*" if res["p_values"][cap_idx] < 0.10 else ""
            hurr_stars = "***" if res["p_values"][hurr_idx] < 0.01 else "**" if res["p_values"][hurr_idx] < 0.05 else "*" if res["p_values"][hurr_idx] < 0.10 else ""
            
            print(f"{res['label']:<35} {res['coefficients'][cap_idx]:>9.4f}{cap_stars} {res['coefficients'][hurr_idx]:>9.4f}{hurr_stars} {res['n']:>7}")
        else:
            print(f"{key:<35} (insufficient data)")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("""
EXPOSURE TEST (hurricane ~ capability):
- If π > 0: High-capability agents are assigned to MORE hurricane-exposed voyages
- If π < 0: High-capability agents are assigned to LESS hurricane-exposed voyages
- If π ≈ 0: Hurricane assignment is orthogonal to capability

OUTCOME TEST (ln Q ~ capability + hurricane):
- θ (capability): Direct effect of agent capability on productivity
- φ (hurricane): Effect of hurricane exposure on productivity

Key advantage: Ex-ante proxies (network, portfolio, complexity, repeat) 
are computed from pre-voyage data and do NOT use output Q.
""")
    
    # Save results
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    rows = []
    for key, res in {**exposure_results, **outcome_results}.items():
        if res is not None:
            for i, var in enumerate(res["variables"]):
                rows.append({
                    "test": "exposure" if key in exposure_results else "outcome",
                    "capability": res["label"],
                    "variable": var,
                    "coefficient": res["coefficients"][i],
                    "std_error": res["std_errors"][i],
                    "t_stat": res["t_stats"][i],
                    "p_value": res["p_values"][i],
                    "n": res["n"],
                    "r2": res["r2"],
                })
    
    pd.DataFrame(rows).to_csv(TABLES_DIR / "exposure_and_proxy_tests.csv", index=False)
    print(f"\nResults saved to {TABLES_DIR / 'exposure_and_proxy_tests.csv'}")
    
    return exposure_results, outcome_results, df


if __name__ == "__main__":
    main()
