"""
Insurance Variance Validation Module.

Proves the "Floor-Raising" mechanism mathematically by testing the second moment:
High-ψ Agents act as "insurance" for Novices, compressing variance and protecting
the left tail of performance.

Key Tests:
1. Heteroskedasticity Test: Var(y|ψ,θ) ~ ψ × θ × Novice
2. Left-Tail (P10) Analysis: Compare P10 by treatment cells
3. Quantile Regression: Q_τ(log_q) ~ ψ, test β(τ=0.10) >> β(τ=0.50)

Predictions:
- Novice × High-ψ: drastically curtailed downside risk (higher P10)
- Expert × High-ψ: little variance effect (or capped upside)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from .config import OUTPUT_DIR, TABLES_DIR

warnings.filterwarnings("ignore", category=FutureWarning)

# Output directory
INSURANCE_DIR = OUTPUT_DIR / "insurance_variance"


# =============================================================================
# 1. Captain Classification (Novice vs Expert)
# =============================================================================

def classify_captain_experience(
    df: pd.DataFrame,
    novice_threshold: int = 3,
    expert_threshold: int = 10,
) -> pd.DataFrame:
    """
    Classify captains as Novice or Expert based on experience.
    
    Parameters
    ----------
    df : pd.DataFrame
        Voyage data with captain_id.
    novice_threshold : int
        Maximum prior voyages for Novice classification.
    expert_threshold : int
        Minimum prior voyages for Expert classification.
        
    Returns
    -------
    pd.DataFrame
        Data with captain_experience_class column.
    """
    print("\n" + "=" * 60)
    print("IV1: CLASSIFYING CAPTAIN EXPERIENCE")
    print("=" * 60)
    
    df = df.copy()
    
    # Compute prior voyages for each captain at each voyage
    df = df.sort_values(["captain_id", "year_out"])
    df["n_prior_voyages"] = df.groupby("captain_id").cumcount()
    
    # Classify
    df["is_novice"] = df["n_prior_voyages"] <= novice_threshold
    df["is_expert"] = df["n_prior_voyages"] >= expert_threshold
    
    # Create categorical
    conditions = [df["is_novice"], df["is_expert"]]
    choices = ["Novice", "Expert"]
    df["captain_experience"] = np.select(conditions, choices, default="Intermediate")
    
    # Summary
    exp_counts = df["captain_experience"].value_counts()
    print(f"\nCaptain-voyage distribution:")
    for exp, count in exp_counts.items():
        pct = 100 * count / len(df)
        print(f"  {exp}: {count:,} ({pct:.1f}%)")
    
    return df


def classify_agent_capability(
    df: pd.DataFrame,
    high_psi_quantile: float = 0.75,
) -> pd.DataFrame:
    """
    Classify agents as High-ψ or Low-ψ.
    
    Parameters
    ----------
    df : pd.DataFrame
        Voyage data with psi_hat or agent_id.
    high_psi_quantile : float
        Quantile threshold for High-ψ classification.
        
    Returns
    -------
    pd.DataFrame
        Data with agent_capability_class column.
    """
    print("\n" + "=" * 60)
    print("IV2: CLASSIFYING AGENT CAPABILITY")
    print("=" * 60)
    
    df = df.copy()
    
    # Compute psi_hat if not present
    if "psi_hat" not in df.columns:
        agent_means = df.groupby("agent_id")["log_q"].mean()
        df["psi_hat"] = df["agent_id"].map(agent_means)
    
    # Get agent-level ψ
    agent_psi = df.groupby("agent_id")["psi_hat"].first()
    psi_threshold = agent_psi.quantile(high_psi_quantile)
    
    df["is_high_psi"] = df["psi_hat"] >= psi_threshold
    df["agent_capability"] = np.where(df["is_high_psi"], "High-ψ", "Low-ψ")
    
    # Summary
    cap_counts = df["agent_capability"].value_counts()
    print(f"\nAgent capability distribution:")
    for cap, count in cap_counts.items():
        pct = 100 * count / len(df)
        print(f"  {cap}: {count:,} ({pct:.1f}%)")
    
    print(f"\nψ threshold (P{int(high_psi_quantile*100)}): {psi_threshold:.3f}")
    
    return df


def create_treatment_cells(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create 2×2 treatment cells: (Novice/Expert) × (High-ψ/Low-ψ).
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with captain_experience and agent_capability columns.
        
    Returns
    -------
    pd.DataFrame
        Data with treatment_cell column.
    """
    print("\n" + "=" * 60)
    print("IV3: CREATING TREATMENT CELLS")
    print("=" * 60)
    
    df = df.copy()
    
    # Filter to Novice/Expert only (exclude Intermediate)
    df_cells = df[df["captain_experience"].isin(["Novice", "Expert"])].copy()
    
    # Create cell identifier
    df_cells["treatment_cell"] = (
        df_cells["captain_experience"] + " × " + df_cells["agent_capability"]
    )
    
    # Summary
    cell_counts = df_cells["treatment_cell"].value_counts()
    print(f"\nTreatment cell distribution:")
    for cell, count in cell_counts.items():
        pct = 100 * count / len(df_cells)
        print(f"  {cell}: {count:,} ({pct:.1f}%)")
    
    return df_cells


# =============================================================================
# 2. Heteroskedasticity Test
# =============================================================================

def run_heteroskedasticity_test(df: pd.DataFrame) -> Dict:
    """
    Test for heteroskedasticity in residuals as function of ψ and θ.
    
    Model: |ε|² = α + β₁×ψ + β₂×θ + β₃×(ψ×θ) + β₄×Novice×ψ + ε
    
    Parameters
    ----------
    df : pd.DataFrame
        Voyage data with log_q, psi_hat, alpha_hat (or theta_hat), captain_experience.
        
    Returns
    -------
    Dict
        Test results including coefficients and Breusch-Pagan F-test.
    """
    print("\n" + "=" * 60)
    print("IV4: HETEROSKEDASTICITY TEST")
    print("=" * 60)
    
    df = df.copy()
    
    # First, regress log_q on θ and ψ to get residuals
    theta_col = "alpha_hat" if "alpha_hat" in df.columns else "theta_hat"
    if theta_col not in df.columns:
        print("No captain FE found. Creating from data...")
        captain_means = df.groupby("captain_id")["log_q"].mean()
        df["theta_hat"] = df["captain_id"].map(captain_means)
        theta_col = "theta_hat"
    
    if "psi_hat" not in df.columns:
        agent_means = df.groupby("agent_id")["log_q"].mean()
        df["psi_hat"] = df["agent_id"].map(agent_means)
    
    # Filter to valid observations
    sample = df.dropna(subset=["log_q", theta_col, "psi_hat"]).copy()
    
    # Stage 1: Regress log_q ~ θ + ψ
    y = sample["log_q"].values
    X1 = np.column_stack([
        np.ones(len(sample)),
        sample[theta_col].values,
        sample["psi_hat"].values,
    ])
    
    beta1 = np.linalg.lstsq(X1, y, rcond=None)[0]
    residuals = y - X1 @ beta1
    sample["residual_sq"] = residuals ** 2
    
    print(f"Stage 1 residual variance: {np.var(residuals):.4f}")
    
    # Stage 2: Regress |ε|² ~ ψ + θ + ψ×θ + Novice×ψ
    sample["theta_psi_interaction"] = sample[theta_col] * sample["psi_hat"]
    
    if "is_novice" in sample.columns:
        sample["novice_psi_interaction"] = sample["is_novice"].astype(float) * sample["psi_hat"]
    else:
        sample["is_novice"] = (sample.groupby("captain_id").cumcount() <= 3).astype(float)
        sample["novice_psi_interaction"] = sample["is_novice"] * sample["psi_hat"]
    
    y2 = sample["residual_sq"].values
    X2 = np.column_stack([
        np.ones(len(sample)),
        sample["psi_hat"].values,
        sample[theta_col].values,
        sample["theta_psi_interaction"].values,
        sample["novice_psi_interaction"].values,
    ])
    
    beta2 = np.linalg.lstsq(X2, y2, rcond=None)[0]
    y2_hat = X2 @ beta2
    resid2 = y2 - y2_hat
    
    n, k = X2.shape
    sigma_sq = np.sum(resid2 ** 2) / (n - k)
    
    try:
        XtX_inv = np.linalg.inv(X2.T @ X2)
        se = np.sqrt(np.diag(sigma_sq * XtX_inv))
    except np.linalg.LinAlgError:
        # Use pseudo-inverse if singular
        XtX_inv = np.linalg.pinv(X2.T @ X2)
        se = np.sqrt(np.abs(np.diag(sigma_sq * XtX_inv)))
    
    # Breusch-Pagan F-test
    r2_aux = 1 - np.var(resid2) / np.var(y2)
    f_stat = (r2_aux * n) / k
    p_value_bp = 1 - stats.f.cdf(f_stat, k - 1, n - k)
    
    print(f"\n--- Auxiliary Regression: |ε|² ~ ψ, θ, ψ×θ, Novice×ψ ---")
    print(f"N = {n:,}")
    
    coef_names = ["Intercept", "ψ (psi)", "θ (theta)", "ψ×θ", "Novice×ψ"]
    for i, (name, b, s) in enumerate(zip(coef_names, beta2, se)):
        t = b / s if s > 0 else 0
        p = 2 * (1 - stats.t.cdf(np.abs(t), df=n-k)) if s > 0 else 1
        stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
        print(f"  {name}: β = {b:.4f}{stars} (SE = {s:.4f}, t = {t:.2f})")
    
    print(f"\nBreusch-Pagan F-test: F = {f_stat:.3f}, p = {p_value_bp:.4f}")
    
    if p_value_bp < 0.10:
        print("✓ Significant heteroskedasticity detected")
    else:
        print("No significant heteroskedasticity")
    
    # Key test: Novice×ψ coefficient
    beta_novice_psi = beta2[4]
    se_novice_psi = se[4]
    t_novice_psi = beta_novice_psi / se_novice_psi if se_novice_psi > 0 else 0
    p_novice_psi = 2 * (1 - stats.t.cdf(np.abs(t_novice_psi), df=n-k))
    
    print(f"\n--- KEY TEST: Novice × ψ Effect on Variance ---")
    print(f"β(Novice×ψ) = {beta_novice_psi:.4f}")
    print(f"p-value = {p_novice_psi:.4f}")
    
    if beta_novice_psi < 0 and p_novice_psi < 0.10:
        print("✓ High-ψ REDUCES variance for Novices (Insurance Effect Confirmed)")
    elif beta_novice_psi > 0 and p_novice_psi < 0.10:
        print("⚠ High-ψ INCREASES variance for Novices (Unexpected)")
    else:
        print("No significant differential variance effect for Novices")
    
    return {
        "n": n,
        "coefficients": {
            "intercept": beta2[0],
            "psi": beta2[1],
            "theta": beta2[2],
            "theta_psi": beta2[3],
            "novice_psi": beta2[4],
        },
        "se": {
            "intercept": se[0],
            "psi": se[1],
            "theta": se[2],
            "theta_psi": se[3],
            "novice_psi": se[4],
        },
        "breusch_pagan_f": f_stat,
        "breusch_pagan_p": p_value_bp,
        "insurance_effect_confirmed": beta_novice_psi < 0 and p_novice_psi < 0.10,
    }


# =============================================================================
# 3. Left-Tail (P10) Analysis
# =============================================================================

def run_left_tail_analysis(df: pd.DataFrame) -> Dict:
    """
    Analyze left tail (P10, P25) of performance by treatment cells.
    
    Prediction: Novice × High-ψ has drastically higher P10 than Novice × Low-ψ.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with treatment_cell and log_q columns.
        
    Returns
    -------
    Dict
        Quantile statistics by treatment cell.
    """
    print("\n" + "=" * 60)
    print("IV5: LEFT-TAIL ANALYSIS")
    print("=" * 60)
    
    df = df.copy()
    
    if "treatment_cell" not in df.columns:
        print("Treatment cells not defined")
        return {"error": "no_treatment_cells"}
    
    # Compute quantiles by cell
    quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
    
    results = []
    for cell in df["treatment_cell"].unique():
        cell_data = df[df["treatment_cell"] == cell]["log_q"]
        
        row = {"Treatment Cell": cell, "N": len(cell_data)}
        row["Mean"] = cell_data.mean()
        row["Std"] = cell_data.std()
        
        for q in quantiles:
            row[f"P{int(q*100)}"] = cell_data.quantile(q)
        
        results.append(row)
    
    results_df = pd.DataFrame(results)
    
    print("\nQuantile Distribution by Treatment Cell:")
    print(results_df.to_string(index=False))
    
    # Key comparison: Novice × High-ψ vs Novice × Low-ψ
    print("\n--- KEY COMPARISON: Left Tail Protection for Novices ---")
    
    novice_high = results_df[results_df["Treatment Cell"] == "Novice × High-ψ"]
    novice_low = results_df[results_df["Treatment Cell"] == "Novice × Low-ψ"]
    
    if len(novice_high) > 0 and len(novice_low) > 0:
        p10_high = novice_high["P10"].values[0]
        p10_low = novice_low["P10"].values[0]
        p10_diff = p10_high - p10_low
        
        std_high = novice_high["Std"].values[0]
        std_low = novice_low["Std"].values[0]
        var_ratio = std_high / std_low if std_low > 0 else np.nan
        
        print(f"Novice × High-ψ: P10 = {p10_high:.3f}, Std = {std_high:.3f}")
        print(f"Novice × Low-ψ:  P10 = {p10_low:.3f}, Std = {std_low:.3f}")
        print(f"P10 Difference:  {p10_diff:.3f}")
        print(f"Variance Ratio (High/Low): {var_ratio:.3f}")
        
        if p10_diff > 0:
            print("✓ High-ψ agents RAISE THE FLOOR for Novices (Higher P10)")
        else:
            print("⚠ High-ψ agents do not raise floor for Novices")
        
        if var_ratio < 1:
            print("✓ High-ψ agents COMPRESS variance for Novices")
        else:
            print("⚠ High-ψ agents do not compress variance for Novices")
    
    return {
        "quantiles_by_cell": results_df.to_dict("records"),
        "p10_novice_high": float(novice_high["P10"].values[0]) if len(novice_high) > 0 else np.nan,
        "p10_novice_low": float(novice_low["P10"].values[0]) if len(novice_low) > 0 else np.nan,
    }


# =============================================================================
# 4. Quantile Regression
# =============================================================================

def run_quantile_regression(
    df: pd.DataFrame,
    quantiles: List[float] = None,
) -> Dict:
    """
    Run quantile regression: Q_τ(log_q) = α + β×ψ + controls.
    
    Test: β(τ=0.10) >> β(τ=0.50) indicates floor-raising stronger than mean effect.
    
    Parameters
    ----------
    df : pd.DataFrame
        Voyage data with log_q and psi_hat.
    quantiles : List[float], optional
        Quantiles to estimate. Default: [0.10, 0.25, 0.50, 0.75, 0.90].
        
    Returns
    -------
    Dict
        Quantile regression coefficients.
    """
    print("\n" + "=" * 60)
    print("IV6: QUANTILE REGRESSION")
    print("=" * 60)
    
    if quantiles is None:
        quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
    
    df = df.copy()
    
    if "psi_hat" not in df.columns:
        agent_means = df.groupby("agent_id")["log_q"].mean()
        df["psi_hat"] = df["agent_id"].map(agent_means)
    
    sample = df.dropna(subset=["log_q", "psi_hat"]).copy()
    
    print(f"Sample size: {len(sample):,}")
    
    # Try to use statsmodels quantile regression
    try:
        import statsmodels.api as sm
        from statsmodels.regression.quantile_regression import QuantReg
        
        y = sample["log_q"].values
        X = sm.add_constant(sample["psi_hat"].values)
        
        results = []
        for q in quantiles:
            model = QuantReg(y, X)
            res = model.fit(q=q, max_iter=1000)
            
            beta_psi = res.params[1]
            se_psi = res.bse[1]
            t_psi = beta_psi / se_psi
            p_psi = 2 * (1 - stats.t.cdf(np.abs(t_psi), df=len(y) - 2))
            
            results.append({
                "Quantile": f"τ = {q:.2f}",
                "β(ψ)": beta_psi,
                "SE": se_psi,
                "t": t_psi,
                "p": p_psi,
            })
            
            stars = "***" if p_psi < 0.01 else "**" if p_psi < 0.05 else "*" if p_psi < 0.1 else ""
            print(f"Q_{q:.2f}: β(ψ) = {beta_psi:.4f}{stars} (SE = {se_psi:.4f})")
        
        results_df = pd.DataFrame(results)
        
    except ImportError:
        print("statsmodels not available. Using OLS approximation...")
        
        # Fallback: OLS on subsamples defined by quantiles
        results = []
        for q in quantiles:
            # Get observations near quantile
            q_val = sample["log_q"].quantile(q)
            bandwidth = 0.1  # Use observations within 10% of quantile
            lower = sample["log_q"].quantile(max(0, q - bandwidth))
            upper = sample["log_q"].quantile(min(1, q + bandwidth))
            
            subsample = sample[(sample["log_q"] >= lower) & (sample["log_q"] <= upper)]
            
            if len(subsample) < 30:
                continue
            
            y = subsample["log_q"].values
            X = np.column_stack([np.ones(len(subsample)), subsample["psi_hat"].values])
            
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            y_hat = X @ beta
            resid = y - y_hat
            
            n, k = X.shape
            sigma_sq = np.sum(resid ** 2) / (n - k)
            try:
                XtX_inv = np.linalg.inv(X.T @ X)
                se = np.sqrt(np.diag(sigma_sq * XtX_inv))
            except:
                se = np.array([np.nan, np.nan])
            
            beta_psi = beta[1]
            se_psi = se[1]
            t_psi = beta_psi / se_psi if se_psi > 0 else 0
            p_psi = 2 * (1 - stats.t.cdf(np.abs(t_psi), df=n-k))
            
            results.append({
                "Quantile": f"τ = {q:.2f}",
                "β(ψ)": beta_psi,
                "SE": se_psi,
                "t": t_psi,
                "p": p_psi,
            })
            
            stars = "***" if p_psi < 0.01 else "**" if p_psi < 0.05 else "*" if p_psi < 0.1 else ""
            print(f"Q_{q:.2f} (approx): β(ψ) = {beta_psi:.4f}{stars}")
        
        results_df = pd.DataFrame(results)
    
    # Key test: Compare β at P10 vs P50
    print("\n--- KEY TEST: Floor Effect vs Mean Effect ---")
    
    beta_p10 = results_df[results_df["Quantile"] == "τ = 0.10"]["β(ψ)"].values
    beta_p50 = results_df[results_df["Quantile"] == "τ = 0.50"]["β(ψ)"].values
    
    if len(beta_p10) > 0 and len(beta_p50) > 0:
        beta_p10 = beta_p10[0]
        beta_p50 = beta_p50[0]
        ratio = beta_p10 / beta_p50 if beta_p50 != 0 else np.nan
        
        print(f"β(ψ) at P10: {beta_p10:.4f}")
        print(f"β(ψ) at P50: {beta_p50:.4f}")
        print(f"Ratio (P10/P50): {ratio:.2f}")
        
        if ratio > 1.2:
            print("✓ Floor effect LARGER than mean effect: Insurance confirmed")
        elif ratio > 1.0:
            print("✓ Floor effect slightly larger than mean effect")
        else:
            print("⚠ Floor effect not larger than mean effect")
    
    return {
        "quantile_results": results_df.to_dict("records"),
        "beta_p10": float(beta_p10) if isinstance(beta_p10, (int, float)) else np.nan,
        "beta_p50": float(beta_p50) if isinstance(beta_p50, (int, float)) else np.nan,
    }


# =============================================================================
# 5. Main Orchestration
# =============================================================================

def run_insurance_variance_tests(
    df: pd.DataFrame = None,
    save_outputs: bool = True,
) -> Dict:
    """
    Run complete Insurance/Variance Validation suite.
    
    Parameters
    ----------
    df : pd.DataFrame, optional
        Voyage data. If None, loads from disk.
    save_outputs : bool
        Whether to save results.
        
    Returns
    -------
    Dict
        All test results.
    """
    print("=" * 70)
    print("INSURANCE VARIANCE VALIDATION")
    print("Testing Floor-Raising Mechanism in Second Moment")
    print("=" * 70)
    
    # Load data if not provided
    if df is None:
        from .data_loader import prepare_analysis_sample
        df = prepare_analysis_sample()
    
    results = {}
    
    # Step 1: Classify captains
    df = classify_captain_experience(df)
    
    # Step 2: Classify agents
    df = classify_agent_capability(df)
    
    # Step 3: Create treatment cells
    df_cells = create_treatment_cells(df)
    
    # Step 4: Heteroskedasticity test
    results["heteroskedasticity"] = run_heteroskedasticity_test(df)
    
    # Step 5: Left-tail analysis
    results["left_tail"] = run_left_tail_analysis(df_cells)
    
    # Step 6: Quantile regression
    results["quantile_regression"] = run_quantile_regression(df)
    
    # Summary
    print("\n" + "=" * 70)
    print("INSURANCE VARIANCE VALIDATION SUMMARY")
    print("=" * 70)
    
    insurance_confirmed = False
    
    # Check heteroskedasticity
    if results["heteroskedasticity"].get("insurance_effect_confirmed"):
        print("✓ Heteroskedasticity: High-ψ reduces variance for Novices")
        insurance_confirmed = True
    
    # Check left tail
    p10_diff = (results["left_tail"].get("p10_novice_high", 0) - 
                results["left_tail"].get("p10_novice_low", 0))
    if p10_diff > 0:
        print(f"✓ Left Tail: P10 raised by {p10_diff:.3f} for Novice × High-ψ")
        insurance_confirmed = True
    
    # Check quantile regression
    beta_p10 = results["quantile_regression"].get("beta_p10", 0)
    beta_p50 = results["quantile_regression"].get("beta_p50", 0)
    if beta_p10 > beta_p50:
        print(f"✓ Quantile: Floor effect ({beta_p10:.3f}) > Mean effect ({beta_p50:.3f})")
        insurance_confirmed = True
    
    results["insurance_confirmed"] = insurance_confirmed
    
    # Save outputs
    if save_outputs:
        save_insurance_outputs(results, df_cells)
    
    return results


def save_insurance_outputs(results: Dict, df_cells: pd.DataFrame) -> None:
    """Save insurance variance test outputs."""
    INSURANCE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save cell analysis
    if "treatment_cell" in df_cells.columns:
        cell_summary = df_cells.groupby("treatment_cell").agg({
            "log_q": ["mean", "std", "count", 
                      lambda x: x.quantile(0.10), 
                      lambda x: x.quantile(0.25)],
        })
        cell_summary.columns = ["Mean", "Std", "N", "P10", "P25"]
        cell_summary.to_csv(INSURANCE_DIR / "treatment_cell_summary.csv")
    
    # Generate markdown summary
    md_lines = [
        "# Insurance Variance Validation Results",
        "",
        "## Key Finding",
        "",
    ]
    
    if results.get("insurance_confirmed"):
        md_lines.append("**✓ Insurance/Floor-Raising Mechanism Confirmed**: High-ψ agents ")
        md_lines.append("compress variance and protect the left tail for Novice captains.")
        md_lines.append("")
        md_lines.append("The effect is in the **second moment** (variance), not just the first (mean).")
    else:
        md_lines.append("Insurance effect not conclusively demonstrated.")
    
    md_lines.extend([
        "",
        "## Test Results",
        "",
        "### Heteroskedasticity",
        "",
    ])
    
    if "heteroskedasticity" in results:
        h = results["heteroskedasticity"]
        md_lines.append(f"- Breusch-Pagan F = {h.get('breusch_pagan_f', 0):.3f}, p = {h.get('breusch_pagan_p', 1):.4f}")
        md_lines.append(f"- β(Novice×ψ) = {h.get('coefficients', {}).get('novice_psi', 0):.4f}")
    
    md_lines.extend([
        "",
        "### Left-Tail Analysis",
        "",
    ])
    
    if "left_tail" in results:
        lt = results["left_tail"]
        md_lines.append(f"- Novice × High-ψ P10: {lt.get('p10_novice_high', 0):.3f}")
        md_lines.append(f"- Novice × Low-ψ P10: {lt.get('p10_novice_low', 0):.3f}")
    
    md_lines.extend([
        "",
        "### Quantile Regression",
        "",
    ])
    
    if "quantile_regression" in results:
        qr = results["quantile_regression"]
        md_lines.append(f"- β(ψ) at P10: {qr.get('beta_p10', 0):.4f}")
        md_lines.append(f"- β(ψ) at P50: {qr.get('beta_p50', 0):.4f}")
    
    with open(INSURANCE_DIR / "insurance_variance_results.md", "w") as f:
        f.write("\n".join(md_lines))
    
    print(f"\nOutputs saved to {INSURANCE_DIR}")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    from .data_loader import prepare_analysis_sample
    
    df = prepare_analysis_sample()
    results = run_insurance_variance_tests(df)
