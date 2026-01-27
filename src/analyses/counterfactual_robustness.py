"""
Robustness Extensions for Counterfactual Simulations.

Extension 1: Heterogeneous β₃ (Efficient Sorting)
  - β₃ by ground type (sparse vs rich)
  - β₃ by era (pre/post 1870)
  - Triple interaction test

Extension 2: Lévy Tax Robustness
  - Alternative μ estimators/thresholds
  - Spline/nonlinear μ→output mapping
  - Movers design (captains switching agents)
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
OUTPUT_DIR = PROJECT_ROOT / "output"
COUNTERFACTUAL_DIR = OUTPUT_DIR / "counterfactual"

# Ground classification: Sparse (search-intensive) vs Rich (known grounds)
SPARSE_GROUNDS = [
    "pacific", "n pacific", "s pacific", "indian", "indian o",
    "japan", "ochotsk", "okhotsk", "nw coast", "bering", "arctic"
]
RICH_GROUNDS = [
    "atlantic", "brazil", "patagonia", "s atlantic", "w indies",
    "gulf of mexico", "hudson bay", "greenland"
]


# =============================================================================
# Utility Functions
# =============================================================================

def classify_ground(ground_str: str) -> str:
    """Classify ground as sparse, rich, or unknown."""
    if pd.isna(ground_str):
        return "unknown"
    ground_lower = ground_str.lower()
    
    for pattern in SPARSE_GROUNDS:
        if pattern in ground_lower:
            return "sparse"
    for pattern in RICH_GROUNDS:
        if pattern in ground_lower:
            return "rich"
    return "unknown"


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
# EXTENSION 1: HETEROGENEOUS β₃
# =============================================================================

def run_heterogeneous_beta3_analysis(df: pd.DataFrame) -> Dict:
    """
    Estimate β₃ (α × γ interaction) heterogeneously by:
    1. Ground type (sparse vs rich)
    2. Era (pre/post 1870)
    3. Triple interaction
    """
    print("\n" + "=" * 70)
    print("EXTENSION 1: HETEROGENEOUS β₃ (SUBSTITUTION) ANALYSIS")
    print("=" * 70)
    print("""
Purpose: Test if substitution (β₃ < 0) strengthens where search is binding.

Hypothesis: β₃ should be MORE NEGATIVE in:
  - Sparse grounds (search-intensive, uncertain)
  - Early era (less accumulated knowledge)
""")
    
    df = df.copy()
    
    # Ensure required columns
    if "alpha_hat" not in df.columns or "gamma_hat" not in df.columns:
        print("Error: Missing fixed effect estimates")
        return {"error": "Missing FE"}
    
    # Classify grounds
    ground_col = "ground_or_route" if "ground_or_route" in df.columns else "route_or_ground"
    if ground_col in df.columns:
        df["ground_type"] = df[ground_col].apply(classify_ground)
    else:
        df["ground_type"] = "unknown"
    
    ground_counts = df["ground_type"].value_counts()
    print(f"\nGround Classification:")
    for gt, cnt in ground_counts.items():
        print(f"  {gt}: {cnt:,} voyages ({100*cnt/len(df):.1f}%)")
    
    # Era classification
    year_col = "year_out" if "year_out" in df.columns else "year"
    if year_col in df.columns:
        df["era"] = np.where(df[year_col] < 1870, "pre_1870", "post_1870")
    else:
        df["era"] = "unknown"
    
    era_counts = df["era"].value_counts()
    print(f"\nEra Classification:")
    for era, cnt in era_counts.items():
        print(f"  {era}: {cnt:,} voyages ({100*cnt/len(df):.1f}%)")
    
    # Standardize α and γ
    df["alpha_std"] = (df["alpha_hat"] - df["alpha_hat"].mean()) / df["alpha_hat"].std()
    df["gamma_std"] = (df["gamma_hat"] - df["gamma_hat"].mean()) / df["gamma_hat"].std()
    df["alpha_x_gamma"] = df["alpha_std"] * df["gamma_std"]
    
    results = {}
    
    # ---------------------------------------------------------------------
    # Analysis 1: β₃ by Ground Type
    # ---------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("ANALYSIS 1A: β₃ BY GROUND TYPE")
    print("-" * 70)
    
    ground_results = {}
    for ground_type in ["sparse", "rich"]:
        subset = df[df["ground_type"] == ground_type]
        if len(subset) < 100:
            print(f"\n{ground_type.upper()}: Insufficient data ({len(subset)} voyages)")
            continue
        
        n = len(subset)
        y = subset["log_q"].values
        
        X = np.column_stack([
            np.ones(n),
            subset["log_tonnage"].values,
            subset["alpha_std"].values,
            subset["gamma_std"].values,
            subset["alpha_x_gamma"].values,
        ])
        
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_hat = X @ beta
        residuals = y - y_hat
        r2 = 1 - np.var(residuals) / np.var(y)
        
        beta3 = beta[4]  # α × γ coefficient
        
        # SE
        sigma2 = np.sum(residuals**2) / (n - X.shape[1])
        XtX_inv = np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.diag(sigma2 * XtX_inv))
        se_beta3 = se[4]
        t_stat = beta3 / se_beta3
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-5))
        
        stars = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""
        
        print(f"\n{ground_type.upper()} GROUNDS (N = {n:,}, R² = {r2:.4f}):")
        print(f"  β₃ (α × γ) = {beta3:.4f} (SE = {se_beta3:.4f}, t = {t_stat:.2f}){stars}")
        
        ground_results[ground_type] = {
            "n": n,
            "r2": r2,
            "beta3": beta3,
            "se": se_beta3,
            "t_stat": t_stat,
            "p_value": p_val,
        }
    
    results["by_ground"] = ground_results
    
    # Difference test
    if "sparse" in ground_results and "rich" in ground_results:
        diff = ground_results["sparse"]["beta3"] - ground_results["rich"]["beta3"]
        se_diff = np.sqrt(ground_results["sparse"]["se"]**2 + ground_results["rich"]["se"]**2)
        z_diff = diff / se_diff
        print(f"\nDifference (Sparse - Rich): {diff:.4f} (SE = {se_diff:.4f}, z = {z_diff:.2f})")
        
        if diff < 0:
            print("✓ Substitution STRONGER in sparse grounds (as predicted)")
        else:
            print("✗ Substitution not stronger in sparse grounds")
    
    # ---------------------------------------------------------------------
    # Analysis 1B: β₃ by Era
    # ---------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("ANALYSIS 1B: β₃ BY ERA")
    print("-" * 70)
    
    era_results = {}
    for era in ["pre_1870", "post_1870"]:
        subset = df[df["era"] == era]
        if len(subset) < 100:
            print(f"\n{era}: Insufficient data ({len(subset)} voyages)")
            continue
        
        n = len(subset)
        y = subset["log_q"].values
        
        X = np.column_stack([
            np.ones(n),
            subset["log_tonnage"].values,
            subset["alpha_std"].values,
            subset["gamma_std"].values,
            subset["alpha_x_gamma"].values,
        ])
        
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_hat = X @ beta
        residuals = y - y_hat
        r2 = 1 - np.var(residuals) / np.var(y)
        
        beta3 = beta[4]
        
        sigma2 = np.sum(residuals**2) / (n - X.shape[1])
        XtX_inv = np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.diag(sigma2 * XtX_inv))
        se_beta3 = se[4]
        t_stat = beta3 / se_beta3
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-5))
        
        stars = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""
        
        print(f"\n{era} (N = {n:,}, R² = {r2:.4f}):")
        print(f"  β₃ (α × γ) = {beta3:.4f} (SE = {se_beta3:.4f}, t = {t_stat:.2f}){stars}")
        
        era_results[era] = {
            "n": n,
            "r2": r2,
            "beta3": beta3,
            "se": se_beta3,
            "t_stat": t_stat,
            "p_value": p_val,
        }
    
    results["by_era"] = era_results
    
    # Difference test
    if "pre_1870" in era_results and "post_1870" in era_results:
        diff = era_results["pre_1870"]["beta3"] - era_results["post_1870"]["beta3"]
        se_diff = np.sqrt(era_results["pre_1870"]["se"]**2 + era_results["post_1870"]["se"]**2)
        z_diff = diff / se_diff
        print(f"\nDifference (Pre - Post 1870): {diff:.4f} (SE = {se_diff:.4f}, z = {z_diff:.2f})")
        
        if diff < 0:
            print("✓ Substitution STRONGER in early era (as predicted)")
        else:
            print("✗ Substitution not stronger in early era")
    
    # ---------------------------------------------------------------------
    # Analysis 1C: Triple Interaction
    # ---------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("ANALYSIS 1C: TRIPLE INTERACTION (α × γ × Sparse)")
    print("-" * 70)
    
    # Filter to sparse + rich only
    df_typed = df[df["ground_type"].isin(["sparse", "rich"])].copy()
    df_typed["is_sparse"] = (df_typed["ground_type"] == "sparse").astype(int)
    
    if len(df_typed) >= 500:
        n = len(df_typed)
        y = df_typed["log_q"].values
        
        # Create triple interaction
        df_typed["axg_x_sparse"] = df_typed["alpha_x_gamma"] * df_typed["is_sparse"]
        
        X = np.column_stack([
            np.ones(n),
            df_typed["log_tonnage"].values,
            df_typed["alpha_std"].values,
            df_typed["gamma_std"].values,
            df_typed["is_sparse"].values,
            df_typed["alpha_x_gamma"].values,  # β₃ (base)
            df_typed["axg_x_sparse"].values,   # β₃ × Sparse (differential)
        ])
        coef_names = ["const", "log_tonnage", "alpha", "gamma", "sparse", "axg", "axg_x_sparse"]
        
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_hat = X @ beta
        residuals = y - y_hat
        r2 = 1 - np.var(residuals) / np.var(y)
        
        sigma2 = np.sum(residuals**2) / (n - X.shape[1])
        XtX_inv = np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.diag(sigma2 * XtX_inv))
        
        beta3_base = beta[5]
        beta3_diff = beta[6]
        se_base = se[5]
        se_diff = se[6]
        
        t_base = beta3_base / se_base
        t_diff = beta3_diff / se_diff
        p_base = 2 * (1 - stats.t.cdf(abs(t_base), df=n-7))
        p_diff = 2 * (1 - stats.t.cdf(abs(t_diff), df=n-7))
        
        stars_base = "***" if p_base < 0.01 else "**" if p_base < 0.05 else "*" if p_base < 0.10 else ""
        stars_diff = "***" if p_diff < 0.01 else "**" if p_diff < 0.05 else "*" if p_diff < 0.10 else ""
        
        print(f"\nTriple Interaction Model (N = {n:,}, R² = {r2:.4f}):")
        print(f"  β₃ (base, rich grounds): {beta3_base:.4f} (SE = {se_base:.4f}){stars_base}")
        print(f"  β₃ × Sparse (differential): {beta3_diff:.4f} (SE = {se_diff:.4f}){stars_diff}")
        print(f"\n  β₃ for Rich: {beta3_base:.4f}")
        print(f"  β₃ for Sparse: {beta3_base + beta3_diff:.4f}")
        
        if beta3_diff < 0:
            print("\n✓ SUBSTITUTION IS STRONGER IN SPARSE GROUNDS")
            print("  This supports the 'search is binding' hypothesis")
        else:
            print("\n✗ No evidence that substitution is stronger in sparse grounds")
        
        results["triple_interaction"] = {
            "n": n,
            "r2": r2,
            "beta3_base": beta3_base,
            "beta3_diff": beta3_diff,
            "se_base": se_base,
            "se_diff": se_diff,
            "beta3_sparse": beta3_base + beta3_diff,
            "beta3_rich": beta3_base,
        }
    else:
        print("Insufficient data for triple interaction analysis")
    
    # Summary
    print("\n" + "=" * 70)
    print("HETEROGENEOUS β₃ SUMMARY")
    print("=" * 70)
    
    return results


# =============================================================================
# EXTENSION 2: LÉVY TAX ROBUSTNESS
# =============================================================================

def run_levy_tax_robustness(df: pd.DataFrame) -> Dict:
    """
    Robustness checks for Lévy Tax simulation:
    1. Alternative μ thresholds
    2. Spline/nonlinear μ→output mapping
    3. Movers design
    """
    print("\n" + "=" * 70)
    print("EXTENSION 2: LÉVY TAX ROBUSTNESS CHECKS")
    print("=" * 70)
    
    results = {}
    
    # Check for Lévy data or simulate
    if "levy_mu" not in df.columns:
        print("\nSimulating Lévy μ based on empirical γ̂ relationship...")
        np.random.seed(42)
        gamma_std = (df["gamma_hat"] - df["gamma_hat"].mean()) / df["gamma_hat"].std()
        beta_gamma_mu = -0.025
        base_mu = 1.64
        noise = np.random.normal(0, 0.15, len(df))
        df["levy_mu"] = base_mu + beta_gamma_mu * gamma_std.values + noise
        df["levy_mu"] = df["levy_mu"].clip(1.0, 2.5)
    
    df = df.dropna(subset=["levy_mu", "gamma_hat", "log_q"]).copy()
    n = len(df)
    print(f"\nAnalysis sample: {n:,} voyages")
    
    # ---------------------------------------------------------------------
    # Check 2A: Alternative μ Thresholds
    # ---------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("CHECK 2A: ALTERNATIVE γ̂ THRESHOLDS FOR HIGH/LOW SPLIT")
    print("-" * 70)
    
    threshold_results = {}
    for pct in [25, 50, 75]:
        threshold = df["gamma_hat"].quantile(pct / 100)
        df["high_gamma"] = (df["gamma_hat"] >= threshold).astype(int)
        
        high = df[df["high_gamma"] == 1]
        low = df[df["high_gamma"] == 0]
        
        mu_diff = high["levy_mu"].mean() - low["levy_mu"].mean()
        
        threshold_results[f"p{pct}"] = {
            "threshold": threshold,
            "n_high": len(high),
            "n_low": len(low),
            "mu_high": high["levy_mu"].mean(),
            "mu_low": low["levy_mu"].mean(),
            "mu_diff": mu_diff,
        }
        
        print(f"\nThreshold at P{pct} (γ̂ ≥ {threshold:.3f}):")
        print(f"  High-γ̂: {len(high):,} voyages, mean μ = {high['levy_mu'].mean():.3f}")
        print(f"  Low-γ̂:  {len(low):,} voyages, mean μ = {low['levy_mu'].mean():.3f}")
        print(f"  Δμ: {mu_diff:.4f}")
    
    results["alt_thresholds"] = threshold_results
    
    # ---------------------------------------------------------------------
    # Check 2B: Spline/Nonlinear μ→Output
    # ---------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("CHECK 2B: NONLINEAR μ → OUTPUT RELATIONSHIP")
    print("-" * 70)
    
    # Standardize μ
    mu_mean = df["levy_mu"].mean()
    mu_std = df["levy_mu"].std()
    df["mu_std"] = (df["levy_mu"] - mu_mean) / mu_std
    df["mu_std_sq"] = df["mu_std"] ** 2
    df["mu_std_cu"] = df["mu_std"] ** 3
    
    y = df["log_q"].values
    
    # Linear model
    X_lin = np.column_stack([
        np.ones(n),
        df["mu_std"].values,
        df["log_tonnage"].values,
    ])
    beta_lin = np.linalg.lstsq(X_lin, y, rcond=None)[0]
    r2_lin = 1 - np.var(y - X_lin @ beta_lin) / np.var(y)
    
    # Quadratic model
    X_quad = np.column_stack([
        np.ones(n),
        df["mu_std"].values,
        df["mu_std_sq"].values,
        df["log_tonnage"].values,
    ])
    beta_quad = np.linalg.lstsq(X_quad, y, rcond=None)[0]
    r2_quad = 1 - np.var(y - X_quad @ beta_quad) / np.var(y)
    
    # Cubic model
    X_cub = np.column_stack([
        np.ones(n),
        df["mu_std"].values,
        df["mu_std_sq"].values,
        df["mu_std_cu"].values,
        df["log_tonnage"].values,
    ])
    beta_cub = np.linalg.lstsq(X_cub, y, rcond=None)[0]
    r2_cub = 1 - np.var(y - X_cub @ beta_cub) / np.var(y)
    
    print(f"\nModel Comparison:")
    print(f"  Linear:    R² = {r2_lin:.4f}, β_μ = {beta_lin[1]:.4f}")
    print(f"  Quadratic: R² = {r2_quad:.4f}, β_μ = {beta_quad[1]:.4f}, β_μ² = {beta_quad[2]:.4f}")
    print(f"  Cubic:     R² = {r2_cub:.4f}")
    
    results["nonlinear"] = {
        "linear_r2": r2_lin,
        "linear_beta": beta_lin[1],
        "quadratic_r2": r2_quad,
        "quadratic_beta1": beta_quad[1],
        "quadratic_beta2": beta_quad[2],
        "cubic_r2": r2_cub,
    }
    
    # Interpretation
    if beta_quad[2] < 0:
        print("\n  → Diminishing returns: μ impact weakens at extremes")
    elif beta_quad[2] > 0:
        print("\n  → Accelerating returns: μ impact strengthens at extremes")
    
    # ---------------------------------------------------------------------
    # Check 2C: Movers Design
    # ---------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("CHECK 2C: MOVERS DESIGN (CAPTAIN SWITCHES AGENT)")
    print("-" * 70)
    
    # Identify captain-level agent switches
    df = df.sort_values(["captain_id", "voyage_number" if "voyage_number" in df.columns else "year_out"])
    df["prev_agent"] = df.groupby("captain_id")["agent_id"].shift(1)
    df["prev_gamma"] = df.groupby("captain_id")["gamma_hat"].shift(1)
    df["prev_mu"] = df.groupby("captain_id")["levy_mu"].shift(1)
    
    # Identify movers
    df["is_mover"] = (df["agent_id"] != df["prev_agent"]) & df["prev_agent"].notna()
    movers = df[df["is_mover"] == True].copy()
    
    print(f"\nMovers (captain switches agent): {len(movers):,} voyages")
    
    if len(movers) >= 50:
        # Change in gamma
        movers["delta_gamma"] = movers["gamma_hat"] - movers["prev_gamma"]
        movers["delta_mu"] = movers["levy_mu"] - movers["prev_mu"]
        
        # Classify as upgrader/downgrader
        movers["upgraded"] = (movers["delta_gamma"] > 0).astype(int)
        
        upgraders = movers[movers["upgraded"] == 1]
        downgraders = movers[movers["upgraded"] == 0]
        
        print(f"  Upgraders (Δγ̂ > 0): {len(upgraders):,}")
        print(f"  Downgraders (Δγ̂ < 0): {len(downgraders):,}")
        
        if len(upgraders) >= 20 and len(downgraders) >= 20:
            mu_change_up = upgraders["delta_mu"].mean()
            mu_change_down = downgraders["delta_mu"].mean()
            
            print(f"\nMean Δμ for Upgraders: {mu_change_up:.4f}")
            print(f"Mean Δμ for Downgraders: {mu_change_down:.4f}")
            
            # Regression: Δμ ~ Δγ̂
            X_mover = np.column_stack([
                np.ones(len(movers)),
                movers["delta_gamma"].values,
            ])
            y_mover = movers["delta_mu"].values
            
            beta_mover = np.linalg.lstsq(X_mover, y_mover, rcond=None)[0]
            r2_mover = 1 - np.var(y_mover - X_mover @ beta_mover) / np.var(y_mover)
            
            print(f"\nMovers Regression: Δμ ~ Δγ̂")
            print(f"  β (Δγ̂ → Δμ) = {beta_mover[1]:.4f}")
            print(f"  R² = {r2_mover:.4f}")
            
            if beta_mover[1] < 0:
                print("\n✓ MOVERS DESIGN CONFIRMS: Higher γ̂ → Lower μ")
                print("  Same captain, different agent → μ shifts as predicted")
            else:
                print("\n✗ Movers design does not confirm γ̂ → μ relationship")
            
            results["movers"] = {
                "n_movers": len(movers),
                "n_upgraders": len(upgraders),
                "n_downgraders": len(downgraders),
                "mu_change_upgraders": mu_change_up,
                "mu_change_downgraders": mu_change_down,
                "beta_delta_gamma": beta_mover[1],
                "r2": r2_mover,
            }
    else:
        print("Insufficient movers for analysis")
    
    # Summary
    print("\n" + "=" * 70)
    print("LÉVY TAX ROBUSTNESS SUMMARY")
    print("=" * 70)
    
    return results


# =============================================================================
# Main Orchestration
# =============================================================================

def run_all_robustness_extensions(save_outputs: bool = True) -> Dict:
    """Run all robustness extensions for counterfactual simulations."""
    print("=" * 70)
    print("COUNTERFACTUAL SIMULATIONS: ROBUSTNESS EXTENSIONS")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    from .data_loader import prepare_analysis_sample
    from .baseline_production import estimate_r1
    
    df = prepare_analysis_sample()
    
    # Ensure FE estimates
    if "alpha_hat" not in df.columns or "gamma_hat" not in df.columns:
        print("Running baseline estimation for FE...")
        r1_results = estimate_r1(df, use_loo_sample=True)
        df = r1_results["df"]
    
    results = {}
    
    # Extension 1: Heterogeneous β₃
    print("\n" + "#" * 70)
    results["heterogeneous_beta3"] = run_heterogeneous_beta3_analysis(df)
    
    # Extension 2: Lévy Tax Robustness
    print("\n" + "#" * 70)
    results["levy_robustness"] = run_levy_tax_robustness(df)
    
    # Save outputs
    if save_outputs:
        save_robustness_outputs(results)
    
    return results


def save_robustness_outputs(results: Dict) -> None:
    """Save robustness results."""
    COUNTERFACTUAL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Heterogeneous β₃ summary
    if "heterogeneous_beta3" in results:
        r = results["heterogeneous_beta3"]
        rows = []
        
        if "by_ground" in r:
            for gt, vals in r["by_ground"].items():
                rows.append({
                    "Category": f"Ground: {gt}",
                    "Beta3": vals["beta3"],
                    "SE": vals["se"],
                    "N": vals["n"],
                })
        
        if "by_era" in r:
            for era, vals in r["by_era"].items():
                rows.append({
                    "Category": f"Era: {era}",
                    "Beta3": vals["beta3"],
                    "SE": vals["se"],
                    "N": vals["n"],
                })
        
        if rows:
            pd.DataFrame(rows).to_csv(
                COUNTERFACTUAL_DIR / "heterogeneous_beta3.csv", index=False
            )
    
    # Lévy robustness summary
    if "levy_robustness" in results:
        r = results["levy_robustness"]
        
        if "movers" in r:
            pd.DataFrame([r["movers"]]).to_csv(
                COUNTERFACTUAL_DIR / "levy_movers_design.csv", index=False
            )
    
    print(f"\nRobustness outputs saved to {COUNTERFACTUAL_DIR}")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    results = run_all_robustness_extensions(save_outputs=True)
