"""
Weather-Controlled Productivity Regressions.

This module runs AKM-style productivity regressions with weather controls:
1. Baseline with NAO and hurricane exposure controls
2. Weather effect heterogeneity by agent capability
3. "Fair weather captain" tests

Key hypothesis: High-capability agents insulate captains from environmental shocks.
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
# Data Preparation
# =============================================================================

def load_analysis_data_with_weather() -> pd.DataFrame:
    """
    Load voyage data merged with weather controls.
    
    Returns
    -------
    pd.DataFrame
        Analysis-ready data with weather variables.
    """
    print("Loading analysis data with weather controls...")
    
    # Load main voyage data
    voyages_path = DATA_DIR / "analysis_voyage.parquet"
    if not voyages_path.exists():
        raise FileNotFoundError(f"Voyage data not found: {voyages_path}")
    
    voyages = pd.read_parquet(voyages_path)
    
    # Load weather data
    weather_path = DATA_DIR / "voyage_weather.parquet"
    if not weather_path.exists():
        raise FileNotFoundError(f"Weather data not found: {weather_path}")
    
    weather = pd.read_parquet(weather_path)
    
    # Merge on voyage_id
    df = voyages.merge(
        weather.drop(columns=["year_out"], errors="ignore"),
        on="voyage_id",
        how="left"
    )
    
    print(f"  Total voyages: {len(df):,}")
    print(f"  With NAO data: {df['nao_index'].notna().sum():,}")
    print(f"  With hurricane data: {df['annual_storms'].notna().sum():,}")
    
    return df


def prepare_regression_sample(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Prepare sample for weather regressions.
    
    Filters to:
    - Voyages with weather data (1851+)
    - Non-missing productivity
    - Connected set of captain-agent pairs
    
    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        (clean_data, diagnostics)
    """
    print("\nPreparing regression sample...")
    
    n_start = len(df)
    
    # Filter to voyages with weather data
    df = df[df["annual_storms"].notna()].copy()
    print(f"  With weather data: {len(df):,} (dropped {n_start - len(df):,})")
    
    # Create log transforms of productivity and tonnage
    df["log_q"] = np.log(df["q_oil_bbl"].replace(0, np.nan))
    df["log_tonnage"] = np.log(df["tonnage"].replace(0, np.nan))
    
    # Fix route column name
    if "ground_or_route" in df.columns:
        df["route_or_ground"] = df["ground_or_route"]
    
    # Require key variables
    required = ["log_q", "log_tonnage", "captain_id", "agent_id", "year_out"]
    df = df.dropna(subset=required)
    print(f"  With required vars: {len(df):,}")
    
    # Create FE interaction columns
    df["route_time"] = df["route_or_ground"].astype(str) + "_" + df["year_out"].astype(str)
    df["port_time"] = df["home_port"].astype(str) + "_" + df["year_out"].astype(str)
    
    # Standardize weather variables
    df["nao_std"] = (df["nao_index"] - df["nao_index"].mean()) / df["nao_index"].std()
    df["hurricane_std"] = (df["hurricane_exposure_count"] - df["hurricane_exposure_count"].mean()) / df["hurricane_exposure_count"].std()
    
    # Create high/low NAO indicator
    df["nao_negative"] = (df["nao_index"] < -0.5).astype(int)
    df["nao_strong_negative"] = (df["nao_index"] < -1.5).astype(int)
    
    # Create high hurricane exposure indicator
    df["high_hurricane_exposure"] = (df["hurricane_exposure_count"] >= df["hurricane_exposure_count"].median()).astype(int)
    
    diagnostics = {
        "n_voyages": len(df),
        "n_captains": df["captain_id"].nunique(),
        "n_agents": df["agent_id"].nunique(),
        "year_range": (df["year_out"].min(), df["year_out"].max()),
        "nao_coverage": df["nao_index"].notna().sum(),
        "hurricane_coverage": df["hurricane_exposure_count"].notna().sum(),
    }
    
    print(f"\nFinal sample: {diagnostics['n_voyages']:,} voyages")
    print(f"  Captains: {diagnostics['n_captains']:,}")
    print(f"  Agents: {diagnostics['n_agents']:,}")
    print(f"  Years: {diagnostics['year_range'][0]}-{diagnostics['year_range'][1]}")
    
    return df, diagnostics


# =============================================================================
# Regression Specifications
# =============================================================================

def cluster_robust_se(residuals: np.ndarray, X: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    """Compute cluster-robust standard errors."""
    n, k = X.shape
    unique_clusters = np.unique(clusters)
    G = len(unique_clusters)
    
    # Bread: (X'X)^-1
    XtX_inv = np.linalg.pinv(X.T @ X)
    
    # Meat: sum of cluster-level outer products
    meat = np.zeros((k, k))
    for c in unique_clusters:
        mask = clusters == c
        Xi = X[mask]
        ei = residuals[mask]
        score = Xi.T @ ei
        meat += np.outer(score, score)
    
    # Sandwich with small-sample correction
    correction = (G / (G - 1)) * ((n - 1) / (n - k))
    vcov = correction * XtX_inv @ meat @ XtX_inv
    
    return np.sqrt(np.diag(vcov))


def run_ols_with_fe(
    df: pd.DataFrame,
    y_var: str,
    x_vars: list,
    fe_vars: list,
    cluster_var: str = "captain_id",
    label: str = "OLS"
) -> Dict:
    """
    Run OLS regression with fixed effects absorbed.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data.
    y_var : str
        Dependent variable.
    x_vars : list
        Control variables (will get coefficients).
    fe_vars : list
        Fixed effect variables (absorbed, no coefficients).
    cluster_var : str
        Clustering variable.
    label : str
        Label for output.
        
    Returns
    -------
    Dict
        Regression results.
    """
    # Drop missing
    all_vars = [y_var] + x_vars + fe_vars + [cluster_var]
    df_clean = df.dropna(subset=[v for v in all_vars if v in df.columns])
    
    n = len(df_clean)
    y = df_clean[y_var].values
    
    # Build X matrix for coefficients of interest
    X_coef = df_clean[x_vars].values
    
    # Build FE design matrices
    fe_matrices = []
    for fe in fe_vars:
        ids = df_clean[fe].unique()
        id_map = {v: i for i, v in enumerate(ids)}
        idx = df_clean[fe].map(id_map).values
        X_fe = sp.csr_matrix(
            (np.ones(n), (np.arange(n), idx)),
            shape=(n, len(ids))
        )
        fe_matrices.append(X_fe)
    
    # Combine all
    X_fe_combined = sp.hstack(fe_matrices) if fe_matrices else None
    
    if X_fe_combined is not None:
        X_full = sp.hstack([sp.csr_matrix(X_coef), X_fe_combined])
    else:
        X_full = sp.csr_matrix(X_coef)
    
    # Solve
    sol = lsqr(X_full, y, iter_lim=10000, atol=1e-10, btol=1e-10)
    beta = sol[0]
    
    # Extract coefficients of interest
    coefs = beta[:len(x_vars)]
    
    # Fitted and residuals
    y_hat = X_full @ beta
    residuals = y - y_hat
    
    # R-squared
    r2 = 1 - np.var(residuals) / np.var(y)
    
    # Cluster-robust SEs for coefficients of interest
    se = cluster_robust_se(residuals, X_coef, df_clean[cluster_var].values)
    
    # T-stats and p-values
    t_stats = coefs / se
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - X_full.shape[1]))
    
    results = {
        "label": label,
        "n": n,
        "r2": r2,
        "variables": x_vars,
        "coefficients": coefs,
        "std_errors": se,
        "t_stats": t_stats,
        "p_values": p_values,
        "residuals": residuals,
    }
    
    return results


def format_results(results: Dict) -> str:
    """Format regression results for display."""
    lines = [
        f"\n{'=' * 60}",
        f"{results['label']}",
        f"{'=' * 60}",
        f"N = {results['n']:,}   R² = {results['r2']:.4f}",
        "-" * 60,
        f"{'Variable':<30} {'Coef':>10} {'SE':>10} {'t':>8} {'p':>8}",
        "-" * 60,
    ]
    
    for i, var in enumerate(results["variables"]):
        coef = results["coefficients"][i]
        se = results["std_errors"][i]
        t = results["t_stats"][i]
        p = results["p_values"][i]
        
        # Stars
        stars = ""
        if p < 0.01:
            stars = "***"
        elif p < 0.05:
            stars = "**"
        elif p < 0.10:
            stars = "*"
        
        lines.append(f"{var:<30} {coef:>10.4f} {se:>10.4f} {t:>8.2f} {p:>7.4f}{stars}")
    
    lines.append("-" * 60)
    
    return "\n".join(lines)


# =============================================================================
# Main Regression Suite
# =============================================================================

def run_weather_regressions(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Run suite of weather-controlled regressions.
    
    Specifications:
    W1: Baseline with NAO and hurricane controls
    W2: NAO effect (negative NAO = stormy Atlantic)
    W3: Hurricane exposure effect
    W4: Agent heterogeneity in weather response
    
    Returns
    -------
    Dict[str, Dict]
        All regression results.
    """
    print("\n" + "=" * 60)
    print("WEATHER-CONTROLLED PRODUCTIVITY REGRESSIONS")
    print("=" * 60)
    
    results = {}
    
    # Fill missing weather vars with 0 for interaction terms
    df = df.copy()
    df["nao_std"] = df["nao_std"].fillna(0)
    df["hurricane_std"] = df["hurricane_std"].fillna(0)
    df["nao_negative"] = df["nao_negative"].fillna(0)
    df["high_hurricane_exposure"] = df["high_hurricane_exposure"].fillna(0)
    
    # =========================================================================
    # W1: Baseline - Production function with weather controls
    # =========================================================================
    print("\n--- W1: Baseline with Weather Controls ---")
    
    results["W1"] = run_ols_with_fe(
        df=df,
        y_var="log_q",
        x_vars=["log_tonnage", "nao_std", "hurricane_std"],
        fe_vars=["captain_id", "agent_id", "route_time"],
        cluster_var="captain_id",
        label="W1: Baseline with Weather Controls (log_q ~ log_tonnage + NAO + hurricanes | captain + agent + route×year)"
    )
    print(format_results(results["W1"]))
    
    # =========================================================================
    # W2: NAO effect - Negative NAO = stormy Atlantic
    # =========================================================================
    print("\n--- W2: NAO Effect (Negative = Stormy) ---")
    
    results["W2"] = run_ols_with_fe(
        df=df,
        y_var="log_q",
        x_vars=["log_tonnage", "nao_negative"],
        fe_vars=["captain_id", "agent_id", "route_time"],
        cluster_var="captain_id",
        label="W2: NAO Negative Effect (log_q ~ nao_negative | FEs)"
    )
    print(format_results(results["W2"]))
    
    # =========================================================================
    # W3: Hurricane exposure effect
    # =========================================================================
    print("\n--- W3: Hurricane Exposure Effect ---")
    
    results["W3"] = run_ols_with_fe(
        df=df,
        y_var="log_q",
        x_vars=["log_tonnage", "high_hurricane_exposure"],
        fe_vars=["captain_id", "agent_id", "route_time"],
        cluster_var="captain_id",
        label="W3: Hurricane Exposure (log_q ~ high_hurricane | FEs)"
    )
    print(format_results(results["W3"]))
    
    # =========================================================================
    # W4: Agent heterogeneity - Do high-capability agents buffer weather shocks?
    # =========================================================================
    print("\n--- W4: Agent Heterogeneity in Weather Response ---")
    
    # First, compute agent effects from baseline
    # Use agent voyage count as proxy for capability
    agent_voyages = df.groupby("agent_id").size()
    df["agent_voyages"] = df["agent_id"].map(agent_voyages)
    df["high_cap_agent"] = (df["agent_voyages"] >= df["agent_voyages"].median()).astype(int)
    
    # Interaction: does high-cap agent buffer hurricane shock?
    df["hurricane_x_highcap"] = df["high_hurricane_exposure"] * df["high_cap_agent"]
    
    results["W4"] = run_ols_with_fe(
        df=df,
        y_var="log_q",
        x_vars=["log_tonnage", "high_hurricane_exposure", "hurricane_x_highcap"],
        fe_vars=["captain_id", "route_time"],
        cluster_var="captain_id",
        label="W4: Hurricane × Agent Capability (log_q ~ hurricane + hurricane×high_cap_agent | captain + route×year)"
    )
    print(format_results(results["W4"]))
    
    # =========================================================================
    # W5: Combined weather adversity
    # =========================================================================
    print("\n--- W5: Combined Weather Adversity ---")
    
    # Create combined adversity index
    df["weather_adversity"] = df["nao_negative"] + df["high_hurricane_exposure"]
    df["adversity_x_highcap"] = df["weather_adversity"] * df["high_cap_agent"]
    
    results["W5"] = run_ols_with_fe(
        df=df,
        y_var="log_q",
        x_vars=["log_tonnage", "weather_adversity", "adversity_x_highcap"],
        fe_vars=["captain_id", "route_time"],
        cluster_var="captain_id",
        label="W5: Weather Adversity × Agent Capability"
    )
    print(format_results(results["W5"]))
    
    return results


def save_results_table(results: Dict[str, Dict], output_path: Path) -> None:
    """Save regression results to CSV."""
    rows = []
    for spec_id, res in results.items():
        for i, var in enumerate(res["variables"]):
            rows.append({
                "specification": spec_id,
                "label": res["label"],
                "variable": var,
                "coefficient": res["coefficients"][i],
                "std_error": res["std_errors"][i],
                "t_stat": res["t_stats"][i],
                "p_value": res["p_values"][i],
                "n": res["n"],
                "r2": res["r2"],
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Run full weather regression suite."""
    
    # Load data
    df = load_analysis_data_with_weather()
    
    # Prepare sample
    df_clean, diagnostics = prepare_regression_sample(df)
    
    # Run regressions
    results = run_weather_regressions(df_clean)
    
    # Save results
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    save_results_table(results, TABLES_DIR / "weather_regressions.csv")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: KEY FINDINGS")
    print("=" * 60)
    
    print("\n1. NAO Effect (W1):")
    nao_idx = results["W1"]["variables"].index("nao_std")
    nao_coef = results["W1"]["coefficients"][nao_idx]
    nao_p = results["W1"]["p_values"][nao_idx]
    print(f"   β = {nao_coef:.4f}, p = {nao_p:.4f}")
    if nao_coef > 0:
        print("   → Positive NAO (calmer Atlantic) associated with higher productivity")
    else:
        print("   → Negative NAO (stormier Atlantic) associated with higher productivity")
    
    print("\n2. Hurricane Effect (W1):")
    hurr_idx = results["W1"]["variables"].index("hurricane_std")
    hurr_coef = results["W1"]["coefficients"][hurr_idx]
    hurr_p = results["W1"]["p_values"][hurr_idx]
    print(f"   β = {hurr_coef:.4f}, p = {hurr_p:.4f}")
    
    print("\n3. Agent Buffering Effect (W4):")
    if "hurricane_x_highcap" in results["W4"]["variables"]:
        inter_idx = results["W4"]["variables"].index("hurricane_x_highcap")
        inter_coef = results["W4"]["coefficients"][inter_idx]
        inter_p = results["W4"]["p_values"][inter_idx]
        print(f"   Hurricane × High-Cap Agent: β = {inter_coef:.4f}, p = {inter_p:.4f}")
        if inter_coef > 0:
            print("   → High-capability agents offset hurricane productivity losses")
    
    return results


if __name__ == "__main__":
    main()
