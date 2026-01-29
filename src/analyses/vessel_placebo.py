"""
Vessel Placebo Tests (C2.3).

Tests whether vessel characteristics (tonnage, rig type) are systematically
related to organizational switches, which would indicate confounding.

Key Tests:
1. Tonnage Placebo: tonnage ~ Δψ (should be null)
   - Vessel tonnage is fixed at voyage start, cannot be affected by protocols
   - If γ ≠ 0 → vessels systematically differ across organizations
   
2. Rig Balance: Pr(switch) ~ rig_type (should be balanced)
   - Check if switches are balanced on vessel characteristics
"""

from typing import Dict, List
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

from .config import OUTPUT_DIR, TABLES_DIR

warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# C2.3a: Tonnage Placebo Test
# =============================================================================

def run_tonnage_placebo(df: pd.DataFrame) -> Dict:
    """
    Test whether tonnage predicts organizational switch.
    
    tonnage ~ Δψ + route×time FE + ε
    
    H0: γ = 0 (tonnage unrelated to agent switching)
    Pass: p > 0.10
    
    Parameters
    ----------
    df : pd.DataFrame
        Voyage data with tonnage and agent assignments.
        
    Returns
    -------
    Dict
        Test results including coefficient, SE, and pass/fail.
    """
    print("\n" + "=" * 60)
    print("C2.3a: TONNAGE PLACEBO TEST")
    print("=" * 60)
    
    df = df.copy()
    
    # Need agent FE estimates
    if "psi_hat" not in df.columns:
        agent_means = df.groupby("agent_id")["log_q"].mean()
        df["psi_hat"] = df["agent_id"].map(agent_means)
        df["psi_hat"] = df["psi_hat"].fillna(df["log_q"].mean())
    
    # Compute Δψ
    df = df.sort_values(["captain_id", "year_out"])
    df["prev_psi"] = df.groupby("captain_id")["psi_hat"].shift(1)
    df["delta_psi"] = df["psi_hat"] - df["prev_psi"]
    
    # Need tonnage
    if "tonnage" not in df.columns:
        print("ERROR: tonnage column not found")
        return {"error": "missing_tonnage"}
    
    # Sample: voyages with valid tonnage and delta_psi
    sample = df.dropna(subset=["tonnage", "delta_psi"]).copy()
    sample = sample[sample["tonnage"] > 0]
    
    print(f"Sample size: {len(sample):,}")
    print(f"Mean tonnage: {sample['tonnage'].mean():.1f}")
    
    if len(sample) < 100:
        print("Insufficient sample")
        return {"error": "insufficient_sample", "n": len(sample)}
    
    # Regression: tonnage ~ delta_psi + controls
    y = sample["tonnage"].values
    X = np.column_stack([
        np.ones(len(sample)),
        sample["delta_psi"].values,
    ])
    
    # Add year controls if available
    if "year_out" in sample.columns:
        year_dummies = pd.get_dummies(sample["year_out"], prefix="year", drop_first=True)
        if len(year_dummies.columns) > 0:
            X = np.column_stack([X, year_dummies.values])
    
    # OLS
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    y_hat = X @ beta
    resid = y - y_hat
    
    n, k = X.shape
    sigma_sq = np.sum(resid ** 2) / (n - k)
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(sigma_sq * XtX_inv))
    
    coef = beta[1]
    se_coef = se[1]
    t_stat = coef / se_coef
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n - k))
    
    stars = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.1 else ""
    
    print(f"\n--- Results ---")
    print(f"Coefficient on Δψ: {coef:.4f}{stars}")
    print(f"SE: {se_coef:.4f}")
    print(f"t-stat: {t_stat:.2f}")
    print(f"p-value: {p_value:.4f}")
    
    passed = p_value > 0.10
    
    print(f"\n--- RESULT ---")
    if passed:
        print("✓ PASS: Tonnage unrelated to agent switching (p > 0.10)")
        print("  → Vessel size does not confound organizational effects")
    else:
        print("✗ FAIL: Tonnage systematically differs with Δψ")
        print("  → Potential vessel-based confounding")
    
    return {
        "n": n,
        "coefficient": coef,
        "se": se_coef,
        "t_stat": t_stat,
        "p_value": p_value,
        "passed": passed,
    }


# =============================================================================
# C2.3b: Rig Type Balance Test
# =============================================================================

def run_rig_balance_test(df: pd.DataFrame) -> Dict:
    """
    Test whether switches are balanced on vessel rig type.
    
    Checks if Pr(switch | rig_type) is uniform across rig types.
    
    Parameters
    ----------
    df : pd.DataFrame
        Voyage data with rig type and switch indicators.
        
    Returns
    -------
    Dict
        Chi-square test results for balance.
    """
    print("\n" + "=" * 60)
    print("C2.3b: RIG TYPE BALANCE TEST")
    print("=" * 60)
    
    df = df.copy()
    
    # Check for rig column
    rig_col = None
    for col in ["rig", "rig_type", "vessel_rig"]:
        if col in df.columns:
            rig_col = col
            break
    
    if rig_col is None:
        print("No rig type column found")
        return {"error": "missing_rig_type"}
    
    # Need switch indicator
    if "switch_agent" not in df.columns:
        df = df.sort_values(["captain_id", "year_out"])
        df["prev_agent"] = df.groupby("captain_id")["agent_id"].shift(1)
        df["switch_agent"] = (df["agent_id"] != df["prev_agent"]).astype(float)
        first_voyage = df["prev_agent"].isna()
        df.loc[first_voyage, "switch_agent"] = np.nan
    
    # Sample: valid rig and switch
    sample = df.dropna(subset=[rig_col, "switch_agent"]).copy()
    
    print(f"Sample size: {len(sample):,}")
    print(f"\nRig type distribution:")
    rig_counts = sample[rig_col].value_counts()
    print(rig_counts.to_string())
    
    # Compute switch rate by rig type
    switch_by_rig = sample.groupby(rig_col)["switch_agent"].agg(["mean", "count", "sum"])
    switch_by_rig.columns = ["switch_rate", "n", "n_switches"]
    
    print(f"\nSwitch rates by rig type:")
    print(switch_by_rig.to_string())
    
    # Chi-square test for independence
    contingency = pd.crosstab(sample[rig_col], sample["switch_agent"])
    
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        print("Insufficient variation for chi-square test")
        return {"error": "insufficient_variation"}
    
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    
    print(f"\n--- Chi-square test ---")
    print(f"χ²({dof}) = {chi2:.3f}")
    print(f"p-value = {p_value:.4f}")
    
    passed = p_value > 0.10
    
    print(f"\n--- RESULT ---")
    if passed:
        print("✓ PASS: Switches balanced on rig type (p > 0.10)")
        print("  → No systematic vessel confounding by rig")
    else:
        print("✗ FAIL: Switches unbalanced on rig type")
        print("  → Potential confounding by vessel rig")
    
    return {
        "chi2": chi2,
        "dof": dof,
        "p_value": p_value,
        "passed": passed,
        "switch_by_rig": switch_by_rig.to_dict(),
    }


# =============================================================================
# Combined Test Suite
# =============================================================================

def run_vessel_placebo_tests(
    df: pd.DataFrame,
    save_outputs: bool = True,
) -> Dict:
    """
    Run all vessel placebo tests (C2.3).
    
    Parameters
    ----------
    df : pd.DataFrame
        Voyage data.
    save_outputs : bool
        Whether to save results.
        
    Returns
    -------
    Dict
        Combined results from all tests.
    """
    print("\n" + "=" * 60)
    print("C2.3: VESSEL CHARACTERISTIC PLACEBO TESTS")
    print("=" * 60)
    
    results = {}
    
    # C2.3a: Tonnage placebo
    results["tonnage"] = run_tonnage_placebo(df)
    
    # C2.3b: Rig balance
    results["rig_balance"] = run_rig_balance_test(df)
    
    # Summary
    print("\n" + "=" * 60)
    print("VESSEL PLACEBO SUMMARY")
    print("=" * 60)
    
    all_passed = True
    
    if "error" not in results["tonnage"]:
        tonnage_status = "✓ PASS" if results["tonnage"]["passed"] else "✗ FAIL"
        print(f"Tonnage placebo: {tonnage_status} (p = {results['tonnage']['p_value']:.4f})")
        all_passed = all_passed and results["tonnage"]["passed"]
    else:
        print(f"Tonnage placebo: SKIPPED ({results['tonnage']['error']})")
    
    if "error" not in results["rig_balance"]:
        rig_status = "✓ PASS" if results["rig_balance"]["passed"] else "✗ FAIL"
        print(f"Rig balance: {rig_status} (p = {results['rig_balance']['p_value']:.4f})")
        all_passed = all_passed and results["rig_balance"]["passed"]
    else:
        print(f"Rig balance: SKIPPED ({results['rig_balance']['error']})")
    
    results["all_passed"] = all_passed
    
    # Save results
    if save_outputs:
        output_path = TABLES_DIR / "c2_3_vessel_placebo.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        summary = []
        if "error" not in results["tonnage"]:
            summary.append({
                "Test": "Tonnage Placebo",
                "Statistic": results["tonnage"]["coefficient"],
                "p_value": results["tonnage"]["p_value"],
                "Passed": results["tonnage"]["passed"],
            })
        if "error" not in results["rig_balance"]:
            summary.append({
                "Test": "Rig Balance",
                "Statistic": results["rig_balance"]["chi2"],
                "p_value": results["rig_balance"]["p_value"],
                "Passed": results["rig_balance"]["passed"],
            })
        
        if summary:
            pd.DataFrame(summary).to_csv(output_path, index=False)
            print(f"\nResults saved to {output_path}")
    
    return results


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    from .data_loader import prepare_analysis_sample
    
    df = prepare_analysis_sample()
    results = run_vessel_placebo_tests(df)
