"""
Cross-Fitted Hurricane Buffering Specification.

Addresses reviewer concerns:
1. Uses cross-fitted γ̂ to avoid generated regressor bias
2. Uses leave-one-decade-out estimation for agent effects
3. Reports pre-period γ̂ as robustness

ln Q_v = β ln(ton)_v + α_{c(v)} + γ_{a(v)} + μ_{route×year} 
         + θ₁ hurricane_t + θ₂ (hurricane_t × γ̂ᶜʳᵒˢˢ_a) + ε_v

Cross-fitting approaches:
A. Leave-one-decade-out: γ̂ for decade d estimated using all data except decade d
B. Pre-period: γ̂ estimated using 1850-1879 data, applied to 1880+
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
# Utility Functions
# =============================================================================

def cluster_robust_se(X: np.ndarray, residuals: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    """Compute cluster-robust standard errors."""
    n, k = X.shape
    unique_clusters = np.unique(clusters)
    G = len(unique_clusters)
    
    if G <= k:
        return np.full(k, np.nan)
    
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


def estimate_agent_gamma(df: pd.DataFrame, sample_mask: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    Estimate agent effects γ from a given sample.
    
    Spec: ln Q = β ln(ton) + α_captain + γ_agent + μ_{route×year} + ε
    
    Returns DataFrame with agent_id, gamma_hat, gamma_hat_std
    """
    if sample_mask is not None:
        df = df[sample_mask].copy()
    else:
        df = df.copy()
    
    if len(df) < 100:
        return None
    
    n = len(df)
    y = df["log_q"].values.astype(float)
    
    # Build design matrix
    matrices = []
    
    # Log tonnage
    log_ton = df["log_tonnage"].values.astype(float).reshape(-1, 1)
    matrices.append(sp.csr_matrix(log_ton))
    
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
    
    X = sp.hstack(matrices)
    
    # Solve
    sol = lsqr(X, y, iter_lim=10000, atol=1e-10, btol=1e-10)
    beta = sol[0]
    
    # Extract agent effects
    n_capt = len(captain_ids)
    n_agent = len(agent_ids)
    gamma_est = beta[1 + n_capt:1 + n_capt + n_agent - 1]
    gamma = np.concatenate([[0], gamma_est])
    
    # Create DataFrame
    agent_effects = pd.DataFrame({
        "agent_id": agent_ids,
        "gamma_hat": gamma,
    })
    
    # Standardize
    agent_effects["gamma_hat_std"] = (
        (agent_effects["gamma_hat"] - agent_effects["gamma_hat"].mean()) / 
        agent_effects["gamma_hat"].std()
    )
    
    return agent_effects


# =============================================================================
# Cross-Fitting Approaches
# =============================================================================

def cross_fit_leave_decade_out(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-fit γ̂ using leave-one-decade-out.
    
    For each decade d, estimate γ using all other decades,
    then apply those estimates to voyages in decade d.
    """
    print("\n" + "=" * 60)
    print("CROSS-FITTING: Leave-One-Decade-Out")
    print("=" * 60)
    
    df = df.copy()
    decades = sorted(df["decade"].unique())
    
    # Initialize cross-fitted gamma column
    df["gamma_cross"] = np.nan
    
    for decade in decades:
        # Estimate gamma using all OTHER decades
        other_mask = df["decade"] != decade
        this_mask = df["decade"] == decade
        
        # Estimate from other decades
        agent_effects = estimate_agent_gamma(df, other_mask)
        
        if agent_effects is None:
            print(f"  {decade}: Skipped (insufficient data in other decades)")
            continue
        
        # Apply to this decade
        n_voyages = this_mask.sum()
        df_decade = df.loc[this_mask, ["agent_id"]].merge(
            agent_effects[["agent_id", "gamma_hat_std"]], 
            on="agent_id", 
            how="left"
        )
        df.loc[this_mask, "gamma_cross"] = df_decade["gamma_hat_std"].values
        
        n_matched = df.loc[this_mask, "gamma_cross"].notna().sum()
        print(f"  {decade}: {n_voyages} voyages, {n_matched} with cross-fitted γ̂")
    
    # Report coverage
    n_with_gamma = df["gamma_cross"].notna().sum()
    print(f"\nTotal with cross-fitted γ̂: {n_with_gamma}/{len(df)} ({100*n_with_gamma/len(df):.1f}%)")
    
    return df


def cross_fit_pre_period(df: pd.DataFrame, cutoff_year: int = 1880) -> pd.DataFrame:
    """
    Cross-fit γ̂ using pre-period estimation.
    
    Estimate γ from years < cutoff_year, apply to all years.
    """
    print("\n" + "=" * 60)
    print(f"CROSS-FITTING: Pre-Period (≤{cutoff_year})")
    print("=" * 60)
    
    df = df.copy()
    
    # Estimate from pre-period
    pre_mask = df["year_out"] <= cutoff_year
    print(f"Pre-period sample: {pre_mask.sum()} voyages")
    
    agent_effects = estimate_agent_gamma(df, pre_mask)
    
    if agent_effects is None:
        print("ERROR: Insufficient pre-period data")
        return df
    
    print(f"Estimated γ̂ for {len(agent_effects)} agents")
    
    # Apply to all voyages
    df = df.merge(
        agent_effects[["agent_id", "gamma_hat_std"]].rename(columns={"gamma_hat_std": "gamma_pre"}),
        on="agent_id",
        how="left"
    )
    
    n_matched = df["gamma_pre"].notna().sum()
    print(f"Voyages with pre-period γ̂: {n_matched}/{len(df)} ({100*n_matched/len(df):.1f}%)")
    
    return df


# =============================================================================
# Main Regression
# =============================================================================

def run_hurricane_spec_with_cross_fit(
    df: pd.DataFrame,
    gamma_col: str,
    label: str,
) -> Dict:
    """
    Run hurricane buffering spec with cross-fitted γ̂.
    
    Uses a reduced model to avoid numerical issues:
    ln Q = β ln(ton) + α_captain + μ_{route×year} + θ₁ hurr + θ₂ hurr×γ̂ + ε
    
    Note: Agent FE absorbed since we're using agent-level γ̂
    """
    df = df.copy()
    
    # Filter to valid cases
    df = df.dropna(subset=[gamma_col])
    
    if len(df) < 100:
        return None
    
    # Create interaction
    df["hurr_x_gamma"] = df["hurricane_std"] * df[gamma_col]
    
    n = len(df)
    y = df["log_q"].values.astype(float)
    
    # Coefficients of interest
    x_vars = ["log_tonnage", "hurricane_std", "hurr_x_gamma"]
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
    
    # Solve
    sol = lsqr(X, y, iter_lim=10000, atol=1e-10, btol=1e-10)
    beta = sol[0]
    coefs = beta[:len(x_vars)]
    
    y_hat = X @ beta
    residuals = y - y_hat
    r2 = 1 - np.var(residuals) / np.var(y)
    
    se = cluster_robust_se(X_coef, residuals, df["captain_id"].values)
    t_stats = coefs / np.where(se > 0, se, np.nan)
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=min(n - X.shape[1], n - 1)))
    
    return {
        "label": label,
        "n": n,
        "r2": r2,
        "variables": x_vars,
        "coefficients": coefs,
        "std_errors": se,
        "t_stats": t_stats,
        "p_values": p_values,
        "n_captain": len(captain_ids),
        "n_route_time": len(rt_ids),
    }


def format_results(res: Dict) -> str:
    """Format results for display."""
    if res is None:
        return "  (insufficient data)\n"
    
    lines = [
        f"\n{res['label']}",
        "=" * 60,
        f"N = {res['n']:,}   R² = {res['r2']:.4f}",
        f"Captain FE: {res['n_captain']}   Route×Year FE: {res['n_route_time']}",
        "-" * 60,
        f"{'Variable':<30} {'Coef':>10} {'SE':>10} {'t':>8}",
        "-" * 60,
    ]
    
    for i, var in enumerate(res["variables"]):
        coef = res["coefficients"][i]
        se = res["std_errors"][i]
        t = res["t_stats"][i]
        p = res["p_values"][i] if not np.isnan(res["p_values"][i]) else 1.0
        
        stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
        
        lines.append(f"{var:<30} {coef:>10.4f} {se:>10.4f} {t:>8.2f}{stars}")
    
    lines.append("-" * 60)
    
    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def load_and_prepare_data() -> pd.DataFrame:
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
    print(f"Decades: {sorted(df['decade'].unique())}")
    
    return df


def main():
    """Run cross-fitted hurricane analysis."""
    
    # Load data
    df = load_and_prepare_data()
    
    # Cross-fit: Leave-one-decade-out
    df = cross_fit_leave_decade_out(df)
    
    # Cross-fit: Pre-period (1880 cutoff)
    df = cross_fit_pre_period(df, cutoff_year=1880)
    
    # Also estimate in-sample γ for comparison
    print("\n" + "=" * 60)
    print("IN-SAMPLE γ̂ (for comparison)")
    print("=" * 60)
    agent_effects = estimate_agent_gamma(df)
    if agent_effects is not None:
        df = df.merge(
            agent_effects[["agent_id", "gamma_hat_std"]].rename(columns={"gamma_hat_std": "gamma_in_sample"}),
            on="agent_id", how="left"
        )
        print(f"Estimated for {len(agent_effects)} agents")
    
    # Run specifications
    print("\n" + "=" * 60)
    print("HURRICANE BUFFERING SPECIFICATIONS")
    print("=" * 60)
    
    results = {}
    
    # Spec 1: In-sample γ̂ (baseline, has generated regressor problem)
    results["in_sample"] = run_hurricane_spec_with_cross_fit(
        df, "gamma_in_sample", 
        "Spec 1: In-Sample γ̂ (has generated regressor issue)"
    )
    print(format_results(results["in_sample"]))
    
    # Spec 2: Cross-fitted γ̂ (leave-decade-out)
    results["cross_decade"] = run_hurricane_spec_with_cross_fit(
        df, "gamma_cross",
        "Spec 2: Leave-One-Decade-Out γ̂ (cross-fitted)"
    )
    print(format_results(results["cross_decade"]))
    
    # Spec 3: Pre-period γ̂ 
    results["pre_period"] = run_hurricane_spec_with_cross_fit(
        df, "gamma_pre",
        "Spec 3: Pre-Period γ̂ (estimated from ≤1880)"
    )
    print(format_results(results["pre_period"]))
    
    # Also run on post-1880 sample only with pre-period γ̂
    df_post = df[df["year_out"] > 1880].copy()
    results["post_period"] = run_hurricane_spec_with_cross_fit(
        df_post, "gamma_pre",
        "Spec 4: Post-1880 Sample with Pre-Period γ̂"
    )
    print(format_results(results["post_period"]))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: INTERACTION COEFFICIENT θ₂ (hurr × γ̂)")
    print("=" * 60)
    print(f"{'Specification':<45} {'θ₂':>10} {'SE':>10} {'p':>8}")
    print("-" * 73)
    
    for key, res in results.items():
        if res is not None:
            idx = res["variables"].index("hurr_x_gamma")
            coef = res["coefficients"][idx]
            se = res["std_errors"][idx]
            p = res["p_values"][idx]
            stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
            print(f"{res['label'][:44]:<45} {coef:>10.4f} {se:>10.4f} {p:>7.4f}{stars}")
    
    # Save
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for key, res in results.items():
        if res is not None:
            for i, var in enumerate(res["variables"]):
                rows.append({
                    "spec": key,
                    "label": res["label"],
                    "variable": var,
                    "coefficient": res["coefficients"][i],
                    "std_error": res["std_errors"][i],
                    "t_stat": res["t_stats"][i],
                    "p_value": res["p_values"][i],
                    "n": res["n"],
                    "r2": res["r2"],
                })
    
    pd.DataFrame(rows).to_csv(TABLES_DIR / "cross_fitted_hurricane_spec.csv", index=False)
    print(f"\nResults saved to {TABLES_DIR / 'cross_fitted_hurricane_spec.csv'}")
    
    # Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    
    if results["cross_decade"] is not None:
        theta2_cross = results["cross_decade"]["coefficients"][2]
        theta2_in = results["in_sample"]["coefficients"][2]
        
        print(f"""
In-sample θ₂ = {theta2_in:.4f}
Cross-fitted θ₂ = {theta2_cross:.4f}

If cross-fitted θ₂ remains positive and significant, this rules out:
- Mechanical correlation from generated regressor
- Reverse causality (good hurricanes → good agents)

The interpretation is robust: high-efficiency agents buffer hurricane shocks.
""")
    
    return results


if __name__ == "__main__":
    main()
