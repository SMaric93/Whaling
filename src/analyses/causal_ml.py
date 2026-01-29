"""
Causal ML for Heterogeneous Treatment Effects.

Implements Causal Forest and Double ML to discover heterogeneity in the
θ×ψ interaction (β₃) without pre-specifying quartiles.

Methods:
- Causal Forest (econml.grf) — Non-parametric CATE estimation
- Double ML (econml.dml) — Debiased ML for treatment effects
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings

# Suppress convergence warnings
warnings.filterwarnings('ignore', category=UserWarning)


def estimate_causal_forest(
    df: pd.DataFrame,
    treatment_col: str = "gamma_hat",
    outcome_col: str = "log_q",
    effect_modifiers: list = None,
    controls: list = None,
    n_estimators: int = 200,
    save_outputs: bool = True,
) -> Dict:
    """
    Estimate heterogeneous treatment effects using Causal Forest.
    
    Parameters
    ----------
    df : pd.DataFrame
        Analysis data with treatment, outcome, and covariates.
    treatment_col : str
        Column name for treatment variable (agent capability ψ).
    outcome_col : str
        Column name for outcome (log output).
    effect_modifiers : list
        Variables over which to estimate heterogeneous effects.
    controls : list
        Control variables for the first stage.
    n_estimators : int
        Number of trees in the forest.
    save_outputs : bool
        Whether to save results to output/ml/.
        
    Returns
    -------
    Dict with CATE estimates, variable importance, and diagnostics.
    """
    from econml.dml import CausalForestDML
    from sklearn.ensemble import RandomForestRegressor
    
    print("\n" + "=" * 70)
    print("CAUSAL FOREST: HETEROGENEOUS TREATMENT EFFECTS")
    print("=" * 70)
    
    # Default effect modifiers
    if effect_modifiers is None:
        effect_modifiers = ["alpha_hat"]  # Heterogeneity by captain skill
    
    # Default controls
    if controls is None:
        controls = ["log_tonnage", "log_duration"]
    
    # Prepare data
    required_cols = [treatment_col, outcome_col] + effect_modifiers + controls
    df_valid = df.dropna(subset=required_cols).copy()
    
    print(f"Sample size: {len(df_valid):,}")
    print(f"Treatment: {treatment_col}")
    print(f"Outcome: {outcome_col}")
    print(f"Effect modifiers: {effect_modifiers}")
    print(f"Controls: {controls}")
    
    # Prepare arrays
    Y = df_valid[outcome_col].values
    T = df_valid[treatment_col].values
    X = df_valid[effect_modifiers].values  # Heterogeneity dimensions
    W = df_valid[controls].values if controls else None
    
    # Fit Causal Forest DML
    print("\nFitting Causal Forest DML...")
    cf = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1),
        model_t=RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1),
        n_estimators=n_estimators,
        min_samples_leaf=20,
        random_state=42,
    )
    
    cf.fit(Y, T, X=X, W=W)
    
    # Get CATE predictions
    cate = cf.effect(X).flatten()
    try:
        cate_intervals = cf.effect_interval(X, alpha=0.05)
        cate_lower = cate_intervals[0].flatten()
        cate_upper = cate_intervals[1].flatten()
    except:
        cate_lower = cate - 1.96 * np.std(cate)
        cate_upper = cate + 1.96 * np.std(cate)
    
    print(f"\nCATE Statistics:")
    print(f"  Mean CATE: {np.mean(cate):.4f}")
    print(f"  Std CATE: {np.std(cate):.4f}")
    print(f"  Min CATE: {np.min(cate):.4f}")
    print(f"  Max CATE: {np.max(cate):.4f}")
    
    # Analyze heterogeneity by effect modifier quantiles
    heterogeneity_results = {}
    for i, mod in enumerate(effect_modifiers):
        mod_values = X[:, i] if len(effect_modifiers) > 1 else X.flatten()
        quartiles = pd.qcut(mod_values, q=4, labels=["Q1", "Q2", "Q3", "Q4"])
        
        print(f"\nCATE by {mod} quartiles:")
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            mask = quartiles == q
            q_cate = cate[mask]
            q_mean = mod_values[mask].mean()
            print(f"  {q} ({mod}̄={q_mean:.3f}): CATE = {q_cate.mean():.4f} ± {q_cate.std():.4f}")
            heterogeneity_results[f"{mod}_{q}"] = {
                "mean_cate": float(q_cate.mean()),
                "std_cate": float(q_cate.std()),
                "mean_modifier": float(q_mean),
                "n": int(mask.sum()),
            }
    
    # Feature importance for heterogeneity
    try:
        importance = cf.feature_importances_
        print("\nFeature importance for heterogeneity:")
        for mod, imp in zip(effect_modifiers, importance):
            print(f"  {mod}: {imp:.4f}")
    except:
        importance = None
    
    results = {
        "method": "CausalForest",
        "n_obs": len(df_valid),
        "treatment": treatment_col,
        "outcome": outcome_col,
        "effect_modifiers": effect_modifiers,
        "controls": controls,
        "mean_cate": float(np.mean(cate)),
        "std_cate": float(np.std(cate)),
        "heterogeneity": heterogeneity_results,
        "feature_importance": dict(zip(effect_modifiers, importance)) if importance is not None else None,
        "cate_predictions": cate,
        "cate_lower": cate_lower,
        "cate_upper": cate_upper,
    }
    
    if save_outputs:
        output_dir = Path("output/ml")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary CSV
        summary_df = pd.DataFrame([
            {"modifier": k, **v} for k, v in heterogeneity_results.items()
        ])
        summary_df.to_csv(output_dir / "causal_forest_cate_by_quartile.csv", index=False)
        
        print(f"\nSaved to {output_dir}")
    
    return results


def estimate_double_ml(
    df: pd.DataFrame,
    treatment_col: str = "gamma_hat",
    outcome_col: str = "log_q",
    controls: list = None,
    model_type: str = "linear",
    cv_folds: int = 5,
    save_outputs: bool = True,
) -> Dict:
    """
    Estimate treatment effects using Double/Debiased ML.
    
    Uses cross-fitting to avoid overfitting bias.
    
    Parameters
    ----------
    df : pd.DataFrame
        Analysis data.
    treatment_col : str
        Treatment variable (agent capability ψ).
    outcome_col : str
        Outcome variable (log output).
    controls : list
        Control variables.
    model_type : str
        ML model for nuisance functions: "linear", "forest", "lasso".
    cv_folds : int
        Number of cross-validation folds.
    save_outputs : bool
        Whether to save results.
        
    Returns
    -------
    Dict with treatment effect estimates and confidence intervals.
    """
    from econml.dml import LinearDML, CausalForestDML
    from sklearn.linear_model import LassoCV, RidgeCV
    from sklearn.ensemble import RandomForestRegressor
    
    print("\n" + "=" * 70)
    print("DOUBLE ML: DEBIASED TREATMENT EFFECT")
    print("=" * 70)
    
    if controls is None:
        controls = ["alpha_hat", "log_tonnage", "log_duration"]
    
    # Prepare data
    required_cols = [treatment_col, outcome_col] + controls
    df_valid = df.dropna(subset=required_cols).copy()
    
    print(f"Sample size: {len(df_valid):,}")
    print(f"Treatment: {treatment_col}")
    print(f"Outcome: {outcome_col}")
    print(f"Controls: {controls}")
    print(f"Model type: {model_type}")
    
    Y = df_valid[outcome_col].values
    T = df_valid[treatment_col].values
    X = df_valid[controls].values
    
    # Select ML model for nuisance functions
    if model_type == "linear":
        model_y = RidgeCV()
        model_t = RidgeCV()
    elif model_type == "lasso":
        model_y = LassoCV(cv=3, max_iter=2000)
        model_t = LassoCV(cv=3, max_iter=2000)
    elif model_type == "forest":
        model_y = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        model_t = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Fit Double ML
    print(f"\nFitting Double ML with {cv_folds}-fold cross-fitting...")
    
    dml = LinearDML(
        model_y=model_y,
        model_t=model_t,
        cv=cv_folds,
        random_state=42,
    )
    
    dml.fit(Y=Y, T=T, W=X)
    
    # Get effect estimates
    ate = dml.ate()
    ate_interval = dml.ate_interval(alpha=0.05)
    
    print(f"\nAverage Treatment Effect (ATE):")
    print(f"  ATE = {ate:.4f}")
    print(f"  95% CI: [{ate_interval[0]:.4f}, {ate_interval[1]:.4f}]")
    
    # Compare with OLS
    from sklearn.linear_model import LinearRegression
    X_ols = np.column_stack([T, X])
    ols = LinearRegression().fit(X_ols, Y)
    ols_beta = ols.coef_[0]
    print(f"\nOLS comparison: β(ψ) = {ols_beta:.4f}")
    print(f"Difference from Double ML: {ate - ols_beta:.4f}")
    
    results = {
        "method": "DoubleMl",
        "model_type": model_type,
        "n_obs": len(df_valid),
        "treatment": treatment_col,
        "outcome": outcome_col,
        "controls": controls,
        "ate": float(ate),
        "ate_ci_lower": float(ate_interval[0]),
        "ate_ci_upper": float(ate_interval[1]),
        "ols_beta": float(ols_beta),
        "dml_vs_ols_diff": float(ate - ols_beta),
    }
    
    if save_outputs:
        output_dir = Path("output/ml")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        summary = pd.DataFrame([results])
        summary.to_csv(output_dir / "double_ml_results.csv", index=False)
        
        print(f"\nSaved to {output_dir}")
    
    return results


def estimate_heterogeneous_interaction(
    df: pd.DataFrame,
    save_outputs: bool = True,
) -> Dict:
    """
    Main function: Estimate heterogeneity in θ×ψ interaction using Causal ML.
    
    This addresses the reviewer critique about mechanism identification.
    """
    from econml.dml import CausalForestDML
    
    print("\n" + "=" * 70)
    print("CAUSAL ML: HETEROGENEOUS θ×ψ INTERACTION")
    print("=" * 70)
    
    # Prepare data
    required_cols = ["log_q", "alpha_hat", "gamma_hat", "log_tonnage", "log_duration"]
    df_valid = df.dropna(subset=required_cols).copy()
    
    # Create interaction term as treatment
    df_valid["theta_std"] = (df_valid["alpha_hat"] - df_valid["alpha_hat"].mean()) / df_valid["alpha_hat"].std()
    df_valid["psi_std"] = (df_valid["gamma_hat"] - df_valid["gamma_hat"].mean()) / df_valid["gamma_hat"].std()
    df_valid["theta_x_psi"] = df_valid["theta_std"] * df_valid["psi_std"]
    
    print(f"Sample size: {len(df_valid):,}")
    
    # 1. Causal Forest: Heterogeneity in ψ effect by θ
    print("\n--- Causal Forest: ψ effect heterogeneity by θ ---")
    cf_results = estimate_causal_forest(
        df_valid,
        treatment_col="psi_std",
        outcome_col="log_q",
        effect_modifiers=["theta_std"],
        controls=["log_tonnage", "log_duration"],
        save_outputs=False,
    )
    
    # 2. Double ML: Debiased interaction effect
    print("\n--- Double ML: Interaction effect ---")
    dml_results = estimate_double_ml(
        df_valid,
        treatment_col="theta_x_psi",
        outcome_col="log_q",
        controls=["theta_std", "psi_std", "log_tonnage", "log_duration"],
        model_type="forest",
        save_outputs=False,
    )
    
    # 3. Compare with ground type heterogeneity
    ground_results = {}
    if "ground_type" in df_valid.columns:
        print("\n--- Causal Forest by Ground Type ---")
        for gtype in ["sparse", "rich"]:
            df_g = df_valid[df_valid["ground_type"] == gtype]
            if len(df_g) > 100:
                print(f"\n  {gtype.upper()} grounds (N={len(df_g):,}):")
                cf_g = estimate_causal_forest(
                    df_g,
                    treatment_col="psi_std",
                    outcome_col="log_q",
                    effect_modifiers=["theta_std"],
                    controls=["log_tonnage", "log_duration"],
                    save_outputs=False,
                )
                ground_results[gtype] = cf_g
    
    # Compile results
    results = {
        "causal_forest": cf_results,
        "double_ml": dml_results,
        "by_ground_type": ground_results,
    }
    
    if save_outputs:
        output_dir = Path("output/ml")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create comprehensive summary
        summary_lines = [
            "# Causal ML: Heterogeneous Treatment Effects",
            "",
            "## Causal Forest Results",
            "",
            f"Treatment: Agent capability (ψ)",
            f"Outcome: log(output)",
            f"Heterogeneity dimension: Captain skill (θ)",
            "",
            "### CATE by θ Quartile",
            "",
            "| Quartile | Mean θ | CATE(ψ) | SE |",
            "|----------|--------|---------|-----|",
        ]
        
        for k, v in cf_results.get("heterogeneity", {}).items():
            summary_lines.append(
                f"| {k} | {v['mean_modifier']:.3f} | {v['mean_cate']:.4f} | {v['std_cate']:.4f} |"
            )
        
        summary_lines.extend([
            "",
            "## Double ML Results",
            "",
            f"**Interaction Effect (θ×ψ):** {dml_results['ate']:.4f}",
            f"**95% CI:** [{dml_results['ate_ci_lower']:.4f}, {dml_results['ate_ci_upper']:.4f}]",
            f"**OLS Comparison:** {dml_results['ols_beta']:.4f}",
            "",
            "## Interpretation",
            "",
            "The Causal Forest discovers the pattern of CATE(ψ) as a function of θ",
            "without pre-specifying quartile cutoffs. If Insurance hypothesis were true,",
            "CATE(ψ) would be highest for low-θ captains (maps help the weakest most).",
            "",
        ])
        
        with open(output_dir / "causal_ml_heterogeneity.md", "w") as f:
            f.write("\n".join(summary_lines))
        
        print(f"\nSaved to {output_dir}/causal_ml_heterogeneity.md")
    
    return results


if __name__ == "__main__":
    from src.analyses.data_loader import prepare_analysis_sample
    from src.analyses.baseline_production import estimate_r1
    
    df = prepare_analysis_sample()
    r1_results = estimate_r1(df, use_loo_sample=True)
    df = r1_results["df"]
    
    results = estimate_heterogeneous_interaction(df)
