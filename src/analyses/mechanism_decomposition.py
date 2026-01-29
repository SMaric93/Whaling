"""
Mechanism Decomposition Analysis.

Distinguishes between three competing hypotheses for the submodular (β₃ < 0)
interaction between Captain Skill (θ) and Agent Capability (ψ):

1. INSURANCE: High-ψ agents raise the floor for low-θ captains
2. REDUNDANCY: High-θ captains don't need maps (already know)
3. INTERFERENCE: Organization overrides expert intuition (hurts high-θ)

This module provides formal tests to distinguish between these mechanisms.
"""

import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from scipy.sparse.linalg import lsqr

from .config import TABLES_DIR
from .baseline_production import estimate_r1
from .data_loader import prepare_analysis_sample

warnings.filterwarnings("ignore", category=FutureWarning)

# Output directory
MECHANISM_DIR = TABLES_DIR.parent / "mechanism"


# =============================================================================
# Utility Functions
# =============================================================================

def standardize(x: np.ndarray) -> np.ndarray:
    """Standardize to mean 0, std 1."""
    std = np.std(x)
    if std < 1e-10:
        return np.zeros_like(x)
    return (x - np.mean(x)) / std


def compute_quartile_groups(df: pd.DataFrame, col: str) -> pd.Series:
    """Assign observations to quartile groups based on column values."""
    return pd.qcut(df[col], q=4, labels=["Q1", "Q2", "Q3", "Q4"])


# =============================================================================
# H1: Insurance Hypothesis Test
# =============================================================================

def test_insurance_hypothesis(df: pd.DataFrame) -> Dict:
    """
    H1: Insurance Hypothesis - Maps raise the floor for low-skill captains.
    
    Test: The substitution effect (β₃ < 0) should be STRONGER for low-θ captains.
    If true: High-ψ agents help low-θ captains more than high-θ captains.
    
    Interpretation: Organizations exist to standardize performance for the median worker.
    
    Parameters
    ----------
    df : pd.DataFrame
        Voyage data with theta_hat, psi_hat, log_q.
        
    Returns
    -------
    Dict
        Test results with stratified β₃ coefficients.
    """
    print("\n" + "=" * 70)
    print("H1: INSURANCE HYPOTHESIS")
    print("=" * 70)
    print("Prediction: β₃ more negative for LOW-θ captains (maps raise the floor)")
    
    df = df.dropna(subset=["theta_hat", "psi_hat", "log_q"]).copy()
    
    # Assign captain quartiles based on θ
    df["theta_quartile"] = compute_quartile_groups(df, "theta_hat")
    
    results = {"hypothesis": "Insurance", "description": "Maps raise floor for low-skill"}
    results["quartile_results"] = {}
    
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        df_q = df[df["theta_quartile"] == q].copy()
        
        if len(df_q) < 100:
            print(f"  Skipping {q} (n={len(df_q)})")
            continue
        
        # Standardize within quartile
        theta_std = standardize(df_q["theta_hat"].values)
        psi_std = standardize(df_q["psi_hat"].values)
        interaction = theta_std * psi_std
        
        # Regression: log_q ~ θ + ψ + θ×ψ + controls
        X = np.column_stack([
            np.ones(len(df_q)),
            df_q["log_tonnage"].values,
            theta_std,
            psi_std,
            interaction,
        ])
        y = df_q["log_q"].values
        
        beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        
        beta3 = beta[4]  # Interaction coefficient
        
        # Standard error via residual variance
        n, k = X.shape
        resid = y - X @ beta
        mse = np.sum(resid**2) / (n - k)
        var_beta = mse * np.linalg.inv(X.T @ X).diagonal()
        se_beta3 = np.sqrt(var_beta[4])
        t_stat = beta3 / se_beta3
        
        results["quartile_results"][q] = {
            "n": len(df_q),
            "beta3": beta3,
            "se": se_beta3,
            "t_stat": t_stat,
            "mean_theta": df_q["theta_hat"].mean(),
        }
        
        sig = "***" if abs(t_stat) > 2.58 else "**" if abs(t_stat) > 1.96 else "*" if abs(t_stat) > 1.64 else ""
        print(f"  {q} (θ̄={df_q['theta_hat'].mean():.3f}, N={len(df_q):,}): β₃ = {beta3:.4f} (SE={se_beta3:.4f}){sig}")
    
    # Test: Is β₃ more negative for Q1 than Q4?
    if "Q1" in results["quartile_results"] and "Q4" in results["quartile_results"]:
        beta3_q1 = results["quartile_results"]["Q1"]["beta3"]
        beta3_q4 = results["quartile_results"]["Q4"]["beta3"]
        se_q1 = results["quartile_results"]["Q1"]["se"]
        se_q4 = results["quartile_results"]["Q4"]["se"]
        
        diff = beta3_q1 - beta3_q4
        se_diff = np.sqrt(se_q1**2 + se_q4**2)
        z_stat = diff / se_diff
        
        results["q1_vs_q4_diff"] = diff
        results["q1_vs_q4_z"] = z_stat
        results["insurance_supported"] = diff < 0 and z_stat < -1.64
        
        print(f"\n  Insurance Test: β₃(Q1) - β₃(Q4) = {diff:.4f} (z = {z_stat:.2f})")
        if results["insurance_supported"]:
            print("  ✓ SUPPORTED: Substitution stronger for low-θ captains")
        else:
            print("  ✗ NOT SUPPORTED: Substitution not concentrated in low-θ")
    
    return results


# =============================================================================
# H2: Redundancy Hypothesis Test
# =============================================================================

def test_redundancy_hypothesis(df: pd.DataFrame) -> Dict:
    """
    H2: Redundancy Hypothesis - High-θ captains already know where whales are.
    
    Test: The marginal value of ψ (∂Q/∂ψ) should be LOWER for high-θ captains.
    If true: Maps add no value when captain already has the knowledge.
    
    Interpretation: Organizational knowledge is redundant with expert skill.
    
    Parameters
    ----------
    df : pd.DataFrame
        Voyage data with theta_hat, psi_hat, log_q.
        
    Returns
    -------
    Dict
        Test results.
    """
    print("\n" + "=" * 70)
    print("H2: REDUNDANCY HYPOTHESIS")
    print("=" * 70)
    print("Prediction: ∂Q/∂ψ lower for high-θ captains (maps redundant for experts)")
    
    df = df.dropna(subset=["theta_hat", "psi_hat", "log_q"]).copy()
    
    # Compute marginal effect of ψ at different θ levels
    # ∂Q/∂ψ = β₂ + β₃ × θ
    # With β₃ < 0, as θ increases, ∂Q/∂ψ decreases
    
    df["theta_quartile"] = compute_quartile_groups(df, "theta_hat")
    
    results = {"hypothesis": "Redundancy", "description": "Maps redundant for experts"}
    results["marginal_psi_by_theta"] = {}
    
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        df_q = df[df["theta_quartile"] == q].copy()
        
        if len(df_q) < 100:
            continue
        
        # Simple regression: log_q ~ ψ (within θ-quartile)
        psi_std = standardize(df_q["psi_hat"].values)
        X = np.column_stack([
            np.ones(len(df_q)),
            df_q["log_tonnage"].values,
            psi_std,
        ])
        y = df_q["log_q"].values
        
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        beta_psi = beta[2]
        
        n, k = X.shape
        resid = y - X @ beta
        mse = np.sum(resid**2) / (n - k)
        var_beta = mse * np.linalg.inv(X.T @ X).diagonal()
        se_psi = np.sqrt(var_beta[2])
        
        results["marginal_psi_by_theta"][q] = {
            "beta_psi": beta_psi,
            "se": se_psi,
            "n": len(df_q),
        }
        
        print(f"  {q}: ∂Q/∂ψ = {beta_psi:.4f} (SE={se_psi:.4f})")
    
    # Test: Is ∂Q/∂ψ declining as θ increases?
    betas = [results["marginal_psi_by_theta"][q]["beta_psi"] 
             for q in ["Q1", "Q2", "Q3", "Q4"] 
             if q in results["marginal_psi_by_theta"]]
    
    if len(betas) >= 4:
        monotonic_decline = all(betas[i] >= betas[i+1] for i in range(len(betas)-1))
        results["redundancy_supported"] = monotonic_decline and betas[0] > betas[-1]
        
        print(f"\n  Redundancy Test: ∂Q/∂ψ monotonically declining: {monotonic_decline}")
        if results["redundancy_supported"]:
            print("  ✓ SUPPORTED: Map value declines with captain skill")
        else:
            print("  ✗ NOT SUPPORTED: Map value does not decline monotonically")
    
    return results


# =============================================================================
# H3: Interference Hypothesis Test
# =============================================================================

def test_interference_hypothesis(df: pd.DataFrame) -> Dict:
    """
    H3: Interference Hypothesis - Organization overrides expert intuition.
    
    Test: For high-θ captains, MORE organization (high-ψ) should HURT performance.
    If true: ∂Q/∂ψ < 0 for θ > median.
    
    Interpretation: Organizations stifle "star" talent.
    
    Parameters
    ----------
    df : pd.DataFrame
        Voyage data with theta_hat, psi_hat, log_q.
        
    Returns
    -------
    Dict
        Test results.
    """
    print("\n" + "=" * 70)
    print("H3: INTERFERENCE HYPOTHESIS")
    print("=" * 70)
    print("Prediction: For high-θ captains, ψ has NEGATIVE effect (overrides intuition)")
    
    df = df.dropna(subset=["theta_hat", "psi_hat", "log_q"]).copy()
    
    theta_median = df["theta_hat"].median()
    df["high_theta"] = (df["theta_hat"] > theta_median).astype(int)
    
    results = {"hypothesis": "Interference", "description": "Maps stifle star talent"}
    
    # Test within high-θ subsample
    df_high = df[df["high_theta"] == 1].copy()
    
    psi_std = standardize(df_high["psi_hat"].values)
    X = np.column_stack([
        np.ones(len(df_high)),
        df_high["log_tonnage"].values,
        standardize(df_high["theta_hat"].values),
        psi_std,
    ])
    y = df_high["log_q"].values
    
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    beta_psi_high = beta[3]
    
    n, k = X.shape
    resid = y - X @ beta
    mse = np.sum(resid**2) / (n - k)
    var_beta = mse * np.linalg.inv(X.T @ X).diagonal()
    se_psi_high = np.sqrt(var_beta[3])
    t_stat = beta_psi_high / se_psi_high
    
    results["high_theta_psi_effect"] = {
        "beta_psi": beta_psi_high,
        "se": se_psi_high,
        "t_stat": t_stat,
        "n": len(df_high),
    }
    
    print(f"  High-θ sample (N={len(df_high):,}): β_ψ = {beta_psi_high:.4f} (SE={se_psi_high:.4f})")
    
    # Interference requires β_ψ < 0 for high-θ
    results["interference_supported"] = beta_psi_high < 0 and t_stat < -1.64
    
    if results["interference_supported"]:
        print("  ✓ SUPPORTED: Organization HURTS high-θ captains")
    else:
        print("  ✗ NOT SUPPORTED: Organization still helps high-θ captains")
        if beta_psi_high > 0:
            print(f"     (β_ψ is positive: {beta_psi_high:.4f})")
    
    return results


# =============================================================================
# Combined Mechanism Diagnosis
# =============================================================================

def diagnose_submodularity_mechanism(df: pd.DataFrame, save_outputs: bool = True) -> Dict:
    """
    Run all three hypothesis tests and determine the dominant mechanism.
    
    Parameters
    ----------
    df : pd.DataFrame
        Voyage data with theta_hat, psi_hat, log_q.
    save_outputs : bool
        Whether to save results to files.
        
    Returns
    -------
    Dict
        Combined results with mechanism diagnosis.
    """
    print("\n" + "=" * 70)
    print("MECHANISM DECOMPOSITION: WHY β₃ < 0?")
    print("=" * 70)
    
    # Ensure required columns exist
    if "theta_hat" not in df.columns:
        if "alpha_hat" in df.columns:
            df["theta_hat"] = df["alpha_hat"]
        else:
            raise ValueError("Need theta_hat or alpha_hat column")
    
    if "psi_hat" not in df.columns:
        if "gamma_hat" in df.columns:
            df["psi_hat"] = df["gamma_hat"]
        else:
            raise ValueError("Need psi_hat or gamma_hat column")
    
    results = {}
    
    # Run all hypothesis tests
    results["H1_insurance"] = test_insurance_hypothesis(df)
    results["H2_redundancy"] = test_redundancy_hypothesis(df)
    results["H3_interference"] = test_interference_hypothesis(df)
    
    # Determine dominant mechanism
    print("\n" + "=" * 70)
    print("MECHANISM DIAGNOSIS")
    print("=" * 70)
    
    supported = []
    if results["H1_insurance"].get("insurance_supported", False):
        supported.append("Insurance")
    if results["H2_redundancy"].get("redundancy_supported", False):
        supported.append("Redundancy")
    if results["H3_interference"].get("interference_supported", False):
        supported.append("Interference")
    
    if len(supported) == 0:
        results["dominant_mechanism"] = "Undetermined"
        print("  No single hypothesis clearly supported")
    elif len(supported) == 1:
        results["dominant_mechanism"] = supported[0]
        print(f"  Dominant mechanism: {supported[0]}")
    else:
        results["dominant_mechanism"] = " + ".join(supported)
        print(f"  Multiple mechanisms supported: {supported}")
    
    # Implications
    print("\n  Policy Implications:")
    if "Insurance" in supported:
        print("    → Organizations standardize performance for median workers")
    if "Redundancy" in supported:
        print("    → Organizational knowledge complements low-skill, substitutes high-skill")
    if "Interference" in supported:
        print("    → Organizations may stifle 'star' talent - need flexibility")
    
    # Save outputs
    if save_outputs:
        MECHANISM_DIR.mkdir(parents=True, exist_ok=True)
        
        # Summary table
        summary_rows = []
        for h, label in [("H1_insurance", "Insurance"), ("H2_redundancy", "Redundancy"), ("H3_interference", "Interference")]:
            summary_rows.append({
                "Hypothesis": label,
                "Description": results[h].get("description", ""),
                "Supported": results[h].get(f"{label.lower()}_supported", False),
            })
        
        pd.DataFrame(summary_rows).to_csv(MECHANISM_DIR / "mechanism_diagnosis_summary.csv", index=False)
        print(f"\n  Saved to {MECHANISM_DIR}")
    
    return results


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    df = prepare_analysis_sample()
    r1_results = estimate_r1(df, use_loo_sample=True)
    df = r1_results["df"]
    
    results = diagnose_submodularity_mechanism(df)
