"""
Strategy choice analysis (R10, R11, R12).

Implements:
- R10: Route choice as function of skill and capability
- R11: Downside risk / failure outcomes
- R12: Learning-by-doing: route experience
"""

from typing import Dict, Optional
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

from .config import DEFAULT_SAMPLE
from .baseline_production import estimate_r1

warnings.filterwarnings("ignore", category=FutureWarning)


def run_r10_route_choice(
    df: pd.DataFrame,
) -> Dict:
    """
    R10: Route choice as function of skill and capability.
    
    Pr(ArcticRoute_v = 1) = logit(b1·α̂ + b2·γ̂ + FE_{port×time} + FE_{vessel×period} + X)
    
    Parameters
    ----------
    df : pd.DataFrame
        Voyage data.
        
    Returns
    -------
    Dict
        Route choice results.
    """
    print("\n" + "=" * 60)
    print("R10: ROUTE CHOICE (RISK-TAKING)")
    print("=" * 60)
    
    # First get baseline FE estimates
    r1_results = estimate_r1(df, use_loo_sample=True)
    df_est = r1_results["df"]
    
    # Check if arctic_route exists
    if "arctic_route" not in df_est.columns:
        if "route_or_ground" in df_est.columns:
            arctic_keywords = ["arctic", "bering", "hudson", "bowhead", "polar", "ice"]
            df_est["arctic_route"] = df_est["route_or_ground"].str.lower().str.contains(
                "|".join(arctic_keywords), na=False
            ).astype(int)
        else:
            print("Cannot construct arctic_route indicator")
            return {"error": "missing_arctic_indicator"}
    
    print(f"\nSample: {len(df_est):,} voyages")
    print(f"Arctic routes: {df_est['arctic_route'].sum():,} ({100*df_est['arctic_route'].mean():.1f}%)")
    
    # Linear probability model (for simplicity with FEs)
    n = len(df_est)
    y = df_est["arctic_route"].values.astype(float)
    
    # Features
    X = np.column_stack([
        np.ones(n),
        df_est["alpha_hat"].values,  # Captain skill
        df_est["gamma_hat"].values,  # Agent capability
        df_est["log_tonnage"].values,
    ])
    
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    b_alpha = beta[1]
    b_gamma = beta[2]
    
    y_hat = X @ beta
    r2 = 1 - np.var(y - y_hat) / np.var(y)
    
    print(f"\n--- Route Choice Results (Linear Probability Model) ---")
    print(f"b1 (captain α̂ → Arctic): {b_alpha:.4f}")
    print(f"b2 (agent γ̂ → Arctic): {b_gamma:.4f}")
    print(f"R²: {r2:.4f}")
    
    if b_alpha > 0:
        print("\n  High-skill captains MORE likely to take risky routes")
    else:
        print("\n  High-skill captains LESS likely to take risky routes")
        
    if b_gamma > 0:
        print("  High-capability agents MORE likely to sponsor risky routes")
    else:
        print("  High-capability agents LESS likely to sponsor risky routes")
    
    # Quartile analysis
    df_est["alpha_q"] = pd.qcut(df_est["alpha_hat"], q=4, labels=[1, 2, 3, 4])
    df_est["gamma_q"] = pd.qcut(df_est["gamma_hat"], q=4, labels=[1, 2, 3, 4])
    
    arctic_by_skill = df_est.groupby("alpha_q")["arctic_route"].mean()
    arctic_by_cap = df_est.groupby("gamma_q")["arctic_route"].mean()
    
    print("\n--- Arctic Route Rate by Quartile ---")
    print("By Captain Skill:")
    print(arctic_by_skill.round(3).to_string())
    print("\nBy Agent Capability:")
    print(arctic_by_cap.round(3).to_string())
    
    results = {
        "b_alpha": b_alpha,
        "b_gamma": b_gamma,
        "r2": r2,
        "n": n,
        "arctic_by_skill": arctic_by_skill,
        "arctic_by_cap": arctic_by_cap,
    }
    
    return results


def run_r11_failure_risk(
    df: pd.DataFrame,
) -> Dict:
    """
    R11: Downside risk / failure outcomes.
    
    Pr(Failure_v = 1) = logit(b1·α̂ + b2·γ̂ + FE + X)
    
    Parameters
    ----------
    df : pd.DataFrame
        Voyage data.
        
    Returns
    -------
    Dict
        Failure risk results.
    """
    print("\n" + "=" * 60)
    print("R11: FAILURE / DOWNSIDE RISK")
    print("=" * 60)
    
    # First get baseline FE estimates
    r1_results = estimate_r1(df, use_loo_sample=True)
    df_est = r1_results["df"]
    
    # Create failure indicator
    if "failure_indicator" not in df_est.columns:
        if "voyage_outcome" in df_est.columns:
            df_est["failure_indicator"] = df_est["voyage_outcome"].isin(
                ["lost", "condemned", "wrecked", "missing"]
            ).astype(int)
        else:
            # Use very low output as proxy
            q5 = df_est["q_total_index"].quantile(0.05)
            df_est["failure_indicator"] = (df_est["q_total_index"] <= q5).astype(int)
    
    print(f"\nSample: {len(df_est):,} voyages")
    print(f"Failures: {df_est['failure_indicator'].sum():,} ({100*df_est['failure_indicator'].mean():.1f}%)")
    
    # Linear probability model
    n = len(df_est)
    y = df_est["failure_indicator"].values.astype(float)
    
    X = np.column_stack([
        np.ones(n),
        df_est["alpha_hat"].values,
        df_est["gamma_hat"].values,
        df_est["log_tonnage"].values,
        df_est.get("arctic_route", np.zeros(n)),  # Risk factor
    ])
    
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    b_alpha = beta[1]
    b_gamma = beta[2]
    
    y_hat = X @ beta
    r2 = 1 - np.var(y - y_hat) / np.var(y)
    
    print(f"\n--- Failure Risk Results ---")
    print(f"b1 (captain α̂ → failure): {b_alpha:.4f}")
    print(f"b2 (agent γ̂ → failure): {b_gamma:.4f}")
    print(f"R²: {r2:.4f}")
    
    if b_alpha < 0:
        print("\n  Higher-skill captains LESS likely to fail")
    if b_gamma < 0:
        print("  Higher-capability agents LESS likely to have failed voyages")
    
    # Quartile analysis
    df_est["alpha_q"] = pd.qcut(df_est["alpha_hat"], q=4, labels=[1, 2, 3, 4])
    df_est["gamma_q"] = pd.qcut(df_est["gamma_hat"], q=4, labels=[1, 2, 3, 4])
    
    failure_by_skill = df_est.groupby("alpha_q")["failure_indicator"].mean()
    failure_by_cap = df_est.groupby("gamma_q")["failure_indicator"].mean()
    
    print("\n--- Failure Rate by Quartile ---")
    print("By Captain Skill:")
    print(failure_by_skill.round(4).to_string())
    print("\nBy Agent Capability:")
    print(failure_by_cap.round(4).to_string())
    
    results = {
        "b_alpha": b_alpha,
        "b_gamma": b_gamma,
        "r2": r2,
        "n": n,
        "failure_rate": df_est["failure_indicator"].mean(),
        "failure_by_skill": failure_by_skill,
        "failure_by_cap": failure_by_cap,
    }
    
    return results


def run_r12_learning(
    df: pd.DataFrame,
) -> Dict:
    """
    R12: Learning-by-doing: route experience.
    
    logQ_v = α_c + γ_a + b·RouteExperience + δ_{vessel×period} + θ_{route×time} + ε
    
    Parameters
    ----------
    df : pd.DataFrame
        Voyage data with route_experience column.
        
    Returns
    -------
    Dict
        Learning results.
    """
    print("\n" + "=" * 60)
    print("R12: LEARNING-BY-DOING (ROUTE EXPERIENCE)")
    print("=" * 60)
    
    # First get baseline FE estimates
    r1_results = estimate_r1(df, use_loo_sample=True)
    df_est = r1_results["df"]
    
    # Check for route experience
    if "route_experience" not in df_est.columns:
        # Compute it
        df_est = df_est.sort_values(["captain_id", "year_out"])
        route_col = "route_or_ground" if "route_or_ground" in df_est.columns else "home_port"
        df_est["route_experience"] = df_est.groupby(["captain_id", route_col]).cumcount()
    
    print(f"\nSample: {len(df_est):,} voyages")
    print(f"Mean route experience: {df_est['route_experience'].mean():.2f}")
    print(f"Max route experience: {df_est['route_experience'].max()}")
    
    # Build design matrix
    n = len(df_est)
    y = df_est["log_q"].values
    
    matrices = []
    
    # Captain FEs
    captain_ids = df_est["captain_id"].unique()
    captain_map = {c: i for i, c in enumerate(captain_ids)}
    captain_idx = df_est["captain_id"].map(captain_map).values
    matrices.append(sp.csr_matrix(
        (np.ones(n), (np.arange(n), captain_idx)),
        shape=(n, len(captain_ids))
    ))
    
    # Agent FEs (drop first)
    agent_ids = df_est["agent_id"].unique()
    agent_map = {a: i for i, a in enumerate(agent_ids)}
    agent_idx = df_est["agent_id"].map(agent_map).values
    matrices.append(sp.csr_matrix(
        (np.ones(n), (np.arange(n), agent_idx)),
        shape=(n, len(agent_ids))
    )[:, 1:])
    
    # Vessel×period FEs (drop first)
    if "vessel_period" in df_est.columns:
        vp_ids = df_est["vessel_period"].unique()
        vp_map = {v: i for i, v in enumerate(vp_ids)}
        vp_idx = df_est["vessel_period"].map(vp_map).values
        matrices.append(sp.csr_matrix(
            (np.ones(n), (np.arange(n), vp_idx)),
            shape=(n, len(vp_ids))
        )[:, 1:])
    
    # Controls including route experience
    controls = np.column_stack([
        df_est["log_duration"].values,
        df_est["log_tonnage"].values,
        df_est["route_experience"].values,
    ])
    matrices.append(sp.csr_matrix(controls))
    
    X = sp.hstack(matrices)
    
    # Solve
    result = lsqr(X, y, iter_lim=10000, atol=1e-10, btol=1e-10)
    beta = result[0]
    
    # Route experience coefficient is last control
    b_experience = beta[-1]
    
    y_hat = X @ beta
    r2 = 1 - np.var(y - y_hat) / np.var(y)
    
    print(f"\n--- Learning Results ---")
    print(f"b (route experience → logQ): {b_experience:.4f}")
    print(f"R²: {r2:.4f}")
    
    if b_experience > 0:
        print(f"\n  Each additional voyage to same route increases output by {100*(np.exp(b_experience)-1):.1f}%")
    
    # Experience gradient by bins
    df_est["exp_bin"] = pd.cut(df_est["route_experience"], bins=[0, 1, 3, 5, 100], labels=["0", "1-2", "3-4", "5+"])
    exp_gradient = df_est.groupby("exp_bin")["log_q"].mean()
    
    print("\n--- Experience Gradient ---")
    print(exp_gradient.round(3).to_string())
    
    results = {
        "b_experience": b_experience,
        "r2": r2,
        "n": n,
        "mean_experience": df_est["route_experience"].mean(),
        "experience_gradient": exp_gradient,
    }
    
    return results


def run_strategy_analysis(
    df: pd.DataFrame,
    save_outputs: bool = True,
) -> Dict:
    """
    Run full strategy analysis (R10, R11, R12).
    
    Parameters
    ----------
    df : pd.DataFrame
        Voyage data.
    save_outputs : bool
        Whether to save outputs.
        
    Returns
    -------
    Dict
        Combined strategy results.
    """
    from .config import TABLES_DIR
    from pathlib import Path
    
    # R10: Route choice
    r10_results = run_r10_route_choice(df)
    
    # R11: Failure risk
    r11_results = run_r11_failure_risk(df)
    
    # R12: Learning
    r12_results = run_r12_learning(df)
    
    if save_outputs:
        summary = pd.DataFrame({
            "Specification": [
                "R10: α̂ → Arctic Route",
                "R10: γ̂ → Arctic Route",
                "R11: α̂ → Failure",
                "R11: γ̂ → Failure",
                "R12: Experience → logQ",
            ],
            "Coefficient": [
                r10_results.get("b_alpha", np.nan),
                r10_results.get("b_gamma", np.nan),
                r11_results.get("b_alpha", np.nan),
                r11_results.get("b_gamma", np.nan),
                r12_results.get("b_experience", np.nan),
            ],
        })
        
        output_path = TABLES_DIR / "r10_r11_r12_strategy.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_path, index=False)
        print(f"\nStrategy analysis saved to {output_path}")
    
    return {"r10": r10_results, "r11": r11_results, "r12": r12_results}


if __name__ == "__main__":
    from .data_loader import prepare_analysis_sample
    
    df = prepare_analysis_sample()
    results = run_strategy_analysis(df)
