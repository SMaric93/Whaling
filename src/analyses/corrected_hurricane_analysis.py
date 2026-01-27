"""
Corrected Hurricane Analysis.

DIAGNOSIS: hurricane_exposure_count is an ANNUAL variable (same for all voyages in year t).
When route×year FE is included, ALL hurricane variation is absorbed → coefficient is noise.

SOLUTION: 
- Use route FE (not route×year FE) to preserve year-level variation
- Report specifications progressively to show the problem

Specifications:
1. Baseline: ln Q = β ln(ton) + hurricane + ε
2. + Route FE: ln Q = β ln(ton) + hurricane + μ_route + ε  
3. + Captain FE: ln Q = β ln(ton) + hurricane + μ_route + α_captain + ε
4. + Agent FE: ln Q = β ln(ton) + hurricane + μ_route + α_captain + γ_agent + ε
5. + Year FE (kills hurricane): shows the identification problem
"""

from pathlib import Path
from typing import Dict, List, Tuple

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


def run_specification(
    df: pd.DataFrame,
    x_vars: List[str],
    fe_vars: List[str],
    cluster_var: str,
    label: str,
) -> Dict:
    """Run a specification with given covariates and FEs."""
    
    df = df.copy()
    df = df.dropna(subset=x_vars + [cluster_var])
    
    if len(df) < 100:
        return None
    
    n = len(df)
    y = df["log_q"].values.astype(float)
    
    # Build design matrix
    X_coef = df[x_vars].astype(float).values
    matrices = [sp.csr_matrix(X_coef)]
    
    fe_counts = {}
    for fe_var in fe_vars:
        fe_ids = df[fe_var].unique()
        fe_map = {f: i for i, f in enumerate(fe_ids)}
        X_fe = sp.csr_matrix(
            (np.ones(n), (np.arange(n), df[fe_var].map(fe_map).values)),
            shape=(n, len(fe_ids))
        )[:, 1:]  # Drop first for identification
        matrices.append(X_fe)
        fe_counts[fe_var] = len(fe_ids)
    
    X = sp.hstack(matrices)
    
    # Solve
    sol = lsqr(X, y, iter_lim=10000, atol=1e-10, btol=1e-10)
    beta = sol[0]
    coefs = beta[:len(x_vars)]
    
    # Fit
    y_hat = X @ beta
    residuals = y - y_hat
    r2 = 1 - np.var(residuals) / np.var(y)
    
    # Cluster-robust SE
    se = cluster_robust_se(X_coef, residuals, df[cluster_var].values)
    t_stats = coefs / np.where(se > 0, se, np.nan)
    dof = max(n - X.shape[1], 1)
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=dof))
    
    return {
        "label": label,
        "n": n,
        "r2": r2,
        "variables": x_vars,
        "coefficients": coefs,
        "std_errors": se,
        "t_stats": t_stats,
        "p_values": p_values,
        "fe_counts": fe_counts,
    }


def format_coef(coef: float, se: float, p: float) -> str:
    """Format coefficient with stars."""
    stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
    return f"{coef:>8.4f}{stars}\n({se:>8.4f})"


# =============================================================================
# Main
# =============================================================================

def main():
    """Run corrected hurricane analysis."""
    
    # Load data
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
    
    # Hurricane: use raw count, standardized
    df["hurricane_std"] = (
        (df["hurricane_exposure_count"] - df["hurricane_exposure_count"].mean()) /
        df["hurricane_exposure_count"].std()
    )
    
    # FE columns
    if "ground_or_route" in df.columns:
        df["route"] = df["ground_or_route"]
    df["route_time"] = df["route"].astype(str) + "_" + df["year_out"].astype(str)
    df["port_time"] = df["home_port"].astype(str) + "_" + df["year_out"].astype(str)
    
    # Drop missing
    df = df.dropna(subset=["log_q", "log_tonnage", "hurricane_std", "captain_id", "agent_id", "route"])
    
    print(f"Sample: {len(df):,} voyages")
    print(f"Years: {df['year_out'].min()}-{df['year_out'].max()}")
    
    # =========================================================================
    # Show the identification problem
    # =========================================================================
    print("\n" + "=" * 70)
    print("IDENTIFICATION DIAGNOSTIC")
    print("=" * 70)
    
    # Within-year variation
    within_year_sd = df.groupby("year_out")["hurricane_std"].std().mean()
    total_sd = df["hurricane_std"].std()
    print(f"Hurricane within-year SD: {within_year_sd:.6f}")
    print(f"Hurricane total SD: {total_sd:.4f}")
    print(f"Ratio: {within_year_sd/total_sd:.6f}")
    print()
    print("CONCLUSION: Hurricane is an ANNUAL variable with ZERO within-year variation.")
    print("            Year FE or route×year FE will absorb ALL hurricane variation.")
    
    # =========================================================================
    # Run progressive specifications
    # =========================================================================
    print("\n" + "=" * 70)
    print("PROGRESSIVE SPECIFICATIONS")
    print("=" * 70)
    
    x_vars = ["log_tonnage", "hurricane_std"]
    
    specs = [
        ("No FE", []),
        ("Route FE", ["route"]),
        ("Captain FE", ["captain_id"]),
        ("Route + Captain FE", ["route", "captain_id"]),
        ("Route + Captain + Agent FE", ["route", "captain_id", "agent_id"]),
        ("Route×Year FE (problem!)", ["route_time"]),  # This absorbs hurricane
    ]
    
    results = []
    for label, fe_vars in specs:
        res = run_specification(df, x_vars, fe_vars, "agent_id", label)
        results.append(res)
        
        if res is not None:
            hurr_idx = res["variables"].index("hurricane_std")
            hurr_coef = res["coefficients"][hurr_idx]
            hurr_se = res["std_errors"][hurr_idx]
            hurr_t = res["t_stats"][hurr_idx]
            
            fe_str = ", ".join([f"{k}={v}" for k, v in res["fe_counts"].items()]) if res["fe_counts"] else "none"
            print(f"\n{label:40} N={res['n']:,}  R²={res['r2']:.4f}")
            print(f"  Hurricane coef = {hurr_coef:>8.4f} (SE={hurr_se:.4f}, t={hurr_t:.2f})")
            print(f"  FE: {fe_str}")
    
    # =========================================================================
    # Interaction specification (with route FE, not route×year)
    # =========================================================================
    print("\n" + "=" * 70)
    print("INTERACTION SPECIFICATION (CORRECTED)")
    print("=" * 70)
    
    # Get ex-ante capability proxy (portfolio breadth)
    df["portfolio_breadth"] = np.nan
    for year in sorted(df["year_out"].unique()):
        prior = df[df["year_out"] < year]
        if len(prior) < 50:
            continue
        routes = prior.groupby("agent_id")["route"].nunique()
        ports = prior.groupby("agent_id")["home_port"].nunique()
        portfolio = routes + ports
        mask = df["year_out"] == year
        df.loc[mask, "portfolio_breadth"] = df.loc[mask, "agent_id"].map(portfolio)
    
    df["portfolio_std"] = (
        (df["portfolio_breadth"] - df["portfolio_breadth"].mean()) /
        df["portfolio_breadth"].std()
    )
    
    # Create interaction
    df["hurr_x_portfolio"] = df["hurricane_std"] * df["portfolio_std"]
    
    # Run spec with route FE (not route×year)
    df_int = df.dropna(subset=["portfolio_std"])
    print(f"\nInteraction sample: {len(df_int):,} voyages")
    
    x_vars_int = ["log_tonnage", "hurricane_std", "portfolio_std", "hurr_x_portfolio"]
    
    res_int = run_specification(
        df_int, x_vars_int, ["route", "captain_id"], "agent_id",
        "Hurricane × Portfolio Breadth (route + captain FE)"
    )
    
    if res_int is not None:
        print(f"\nN={res_int['n']:,}   R²={res_int['r2']:.4f}")
        print("-" * 60)
        for i, var in enumerate(res_int["variables"]):
            coef = res_int["coefficients"][i]
            se = res_int["std_errors"][i]
            t = res_int["t_stats"][i]
            p = res_int["p_values"][i]
            stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
            print(f"{var:<25} {coef:>10.4f} ({se:.4f}) t={t:>6.2f}{stars}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("""
KEY FINDING: Hurricane is an ANNUAL variable (same for all voyages in year t).

PROBLEM: Route×year FE includes year → absorbs ALL hurricane variation
         → hurricane coefficient becomes numerical garbage

SOLUTION: Use route FE (not route×year FE) to identify hurricane effects

CORRECT SPECIFICATION:
  ln Q = β ln(ton) + φ hurricane + θ (hurricane × capability) 
         + μ_route + α_captain + ε

With route FE (not route×year FE), hurricane variation comes from YEAR
differences, which is the correct source of variation for an annual shock.
""")
    
    # Save
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    rows = []
    for res in results:
        if res is not None:
            for i, var in enumerate(res["variables"]):
                rows.append({
                    "spec": res["label"],
                    "variable": var,
                    "coefficient": res["coefficients"][i],
                    "std_error": res["std_errors"][i],
                    "t_stat": res["t_stats"][i],
                    "p_value": res["p_values"][i],
                    "n": res["n"],
                    "r2": res["r2"],
                })
    
    pd.DataFrame(rows).to_csv(TABLES_DIR / "corrected_hurricane_specs.csv", index=False)
    print(f"\nResults saved to {TABLES_DIR / 'corrected_hurricane_specs.csv'}")
    
    return results


if __name__ == "__main__":
    main()
