"""
Labor market dynamics analysis (R13, R14, R15).

Implements:
- R13: Assortative matching (sorting)
- R14: Switching hazard
- R15: Talent acquisition advantage
"""

from typing import Dict, Optional
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr
from scipy import stats

from .config import DEFAULT_SAMPLE
from .baseline_production import estimate_r1

warnings.filterwarnings("ignore", category=FutureWarning)


def run_r13_sorting(
    df: pd.DataFrame,
) -> Dict:
    """
    R13: Assortative matching - do high-skill captains work with high-capability agents?
    
    α̂_c = b · γ̂_{agent_assigned} + FE_{port×time} + u
    
    Parameters
    ----------
    df : pd.DataFrame
        Voyage data.
        
    Returns
    -------
    Dict
        Sorting results.
    """
    print("\n" + "=" * 60)
    print("R13: ASSORTATIVE MATCHING (SORTING)")
    print("=" * 60)
    
    # First get baseline FE estimates
    r1_results = estimate_r1(df, use_loo_sample=True)
    df_est = r1_results["df"]
    
    print(f"\nSample: {len(df_est):,} voyages")
    
    # Simple regression of alpha on gamma
    n = len(df_est)
    y = df_est["alpha_hat"].values
    
    # With port×time FE
    if "port_time" in df_est.columns:
        pt_ids = df_est["port_time"].unique()
        pt_map = {p: i for i, p in enumerate(pt_ids)}
        pt_idx = df_est["port_time"].map(pt_map).values
        
        X_pt = sp.csr_matrix(
            (np.ones(n), (np.arange(n), pt_idx)),
            shape=(n, len(pt_ids))
        )[:, 1:]  # Drop first
        
        X_gamma = sp.csr_matrix(df_est["gamma_hat"].values.reshape(-1, 1))
        X = sp.hstack([X_gamma, X_pt])
    else:
        X = np.column_stack([
            np.ones(n),
            df_est["gamma_hat"].values,
        ])
    
    result = lsqr(X, y, iter_lim=10000, atol=1e-10, btol=1e-10) if sp.issparse(X) else None
    
    if result:
        beta = result[0]
        b_sorting = beta[0]  # gamma coefficient
    else:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        b_sorting = beta[1]
    
    y_hat = X @ beta if not sp.issparse(X) else X.dot(beta)
    r2 = 1 - np.var(y - y_hat) / np.var(y)
    
    # Raw correlation
    raw_corr = df_est["alpha_hat"].corr(df_est["gamma_hat"])
    
    print(f"\n--- Sorting Results ---")
    print(f"b (γ̂ → α̂): {b_sorting:.4f}")
    print(f"Raw correlation(α̂, γ̂): {raw_corr:.4f}")
    print(f"R²: {r2:.4f}")
    
    if b_sorting > 0.1:
        print("\n  Strong POSITIVE sorting: high-skill captains match with high-capability agents")
    elif b_sorting < -0.1:
        print("\n  NEGATIVE sorting: high-skill captains match with lower-capability agents")
    else:
        print("\n  Weak or no assortative matching")
    
    # Quintile analysis
    df_est["alpha_q"] = pd.qcut(df_est["alpha_hat"], q=5, labels=[1, 2, 3, 4, 5])
    df_est["gamma_q"] = pd.qcut(df_est["gamma_hat"], q=5, labels=[1, 2, 3, 4, 5])
    
    # Sorting matrix
    sorting_matrix = pd.crosstab(
        df_est["alpha_q"],
        df_est["gamma_q"],
        normalize="all"
    ) * 100
    
    print("\n--- Sorting Matrix (% of matches) ---")
    print("Rows = Captain α quintile, Cols = Agent γ quintile")
    print(sorting_matrix.round(1).to_string())
    
    # Diagonal share
    diagonal_share = sum(sorting_matrix.iloc[i, i] for i in range(5))
    print(f"\nDiagonal share: {diagonal_share:.1f}% (random = 20%)")
    
    # Time trend in sorting
    if "decade" in df_est.columns:
        decade_corr = df_est.groupby("decade").apply(
            lambda x: x["alpha_hat"].corr(x["gamma_hat"]) if len(x) > 10 else np.nan
        )
        print("\n--- Sorting by Decade ---")
        print(decade_corr.round(4).to_string())
    
    results = {
        "b_sorting": b_sorting,
        "raw_corr": raw_corr,
        "r2": r2,
        "n": n,
        "sorting_matrix": sorting_matrix,
        "diagonal_share": diagonal_share,
    }
    
    return results


def run_r14_switching_hazard(
    df: pd.DataFrame,
) -> Dict:
    """
    R14: Switching hazard - what predicts captain moves between agents?
    
    Pr(Switch_{c,t+1} = 1) = logit(b1·α̂ + b2·γ̂_current + b3·Shock + FE_{port×time})
    
    Parameters
    ----------
    df : pd.DataFrame
        Voyage data.
        
    Returns
    -------
    Dict
        Switching hazard results.
    """
    print("\n" + "=" * 60)
    print("R14: SWITCHING HAZARD")
    print("=" * 60)
    
    # First get baseline FE estimates
    r1_results = estimate_r1(df, use_loo_sample=True)
    df_est = r1_results["df"]
    
    # Check for switch_next
    if "switch_next" not in df_est.columns:
        df_est = df_est.sort_values(["captain_id", "year_out"])
        df_est["switch_next"] = df_est.groupby("captain_id")["switch_agent"].shift(-1)
    
    # Drop observations without next voyage
    df_hazard = df_est[df_est["switch_next"].notna()].copy()
    
    print(f"\nSample: {len(df_hazard):,} voyages (with known next voyage)")
    print(f"Switching rate: {df_hazard['switch_next'].mean()*100:.1f}%")
    
    # Linear probability model
    n = len(df_hazard)
    y = df_hazard["switch_next"].values.astype(float)
    
    # Create shock indicator (below-median performance for that voyage)
    median_residual = 0  # Residual from R1
    df_hazard["bad_voyage"] = (df_hazard["log_q"] < df_hazard["log_q"].median()).astype(int)
    
    X = np.column_stack([
        np.ones(n),
        df_hazard["alpha_hat"].values,
        df_hazard["gamma_hat"].values,
        df_hazard["bad_voyage"].values,
    ])
    
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    b_alpha = beta[1]
    b_gamma = beta[2]
    b_shock = beta[3]
    
    y_hat = X @ beta
    r2 = 1 - np.var(y - y_hat) / np.var(y)
    
    print(f"\n--- Switching Hazard Results ---")
    print(f"b1 (α̂ → switch): {b_alpha:.4f}")
    print(f"b2 (γ̂ → switch): {b_gamma:.4f}")
    print(f"b3 (bad voyage → switch): {b_shock:.4f}")
    print(f"R²: {r2:.4f}")
    
    if b_alpha > 0:
        print("\n  Higher-skill captains MORE likely to switch (mobile talent)")
    else:
        print("\n  Higher-skill captains LESS likely to switch (retention)")
        
    if b_gamma < 0:
        print("  Higher-capability agents have LOWER turnover (retention power)")
    else:
        print("  Higher-capability agents have HIGHER turnover")
        
    if b_shock > 0:
        print("  Bad voyages predict switching (accountability)")
    
    # Switching rate by quintile
    df_hazard["alpha_q"] = pd.qcut(df_hazard["alpha_hat"], q=4, labels=[1, 2, 3, 4])
    df_hazard["gamma_q"] = pd.qcut(df_hazard["gamma_hat"], q=4, labels=[1, 2, 3, 4])
    
    switch_by_skill = df_hazard.groupby("alpha_q")["switch_next"].mean()
    switch_by_cap = df_hazard.groupby("gamma_q")["switch_next"].mean()
    
    print("\n--- Switching Rate by Quartile ---")
    print("By Captain Skill:")
    print(switch_by_skill.round(3).to_string())
    print("\nBy Agent Capability:")
    print(switch_by_cap.round(3).to_string())
    
    results = {
        "b_alpha": b_alpha,
        "b_gamma": b_gamma,
        "b_shock": b_shock,
        "r2": r2,
        "n": n,
        "switching_rate": df_hazard["switch_next"].mean(),
        "switch_by_skill": switch_by_skill,
        "switch_by_cap": switch_by_cap,
    }
    
    return results


def run_r15_acquisition_advantage(
    df: pd.DataFrame,
) -> Dict:
    """
    R15: Talent acquisition advantage - do high-capability agents hire better captains?
    
    NextHireQuality_{a,t} = b · γ̂_a + FE_{port×time} + ε
    
    NextHireQuality = mean α̂ of first-time captains hired by agent in year
    
    Parameters
    ----------
    df : pd.DataFrame
        Voyage data.
        
    Returns
    -------
    Dict
        Acquisition advantage results.
    """
    print("\n" + "=" * 60)
    print("R15: TALENT ACQUISITION ADVANTAGE")
    print("=" * 60)
    
    # First get baseline FE estimates
    r1_results = estimate_r1(df, use_loo_sample=True)
    df_est = r1_results["df"]
    
    # Identify first voyage for each captain
    df_est = df_est.sort_values(["captain_id", "year_out"])
    df_est["is_first_voyage"] = ~df_est["captain_id"].duplicated()
    
    first_voyages = df_est[df_est["is_first_voyage"]].copy()
    print(f"\nFirst-time captain voyages: {len(first_voyages):,}")
    
    # Aggregate to agent-year level
    agent_year_hires = first_voyages.groupby(["agent_id", "year_out"]).agg({
        "alpha_hat": ["mean", "count"],
        "gamma_hat": "first",
    }).reset_index()
    
    agent_year_hires.columns = ["agent_id", "year_out", "hire_quality", "n_hires", "gamma_hat"]
    
    # Filter to agent-years with enough hires
    agent_year_hires = agent_year_hires[agent_year_hires["n_hires"] >= 1].copy()
    
    print(f"Agent-year observations: {len(agent_year_hires):,}")
    print(f"Mean hires per agent-year: {agent_year_hires['n_hires'].mean():.2f}")
    
    if len(agent_year_hires) < 50:
        print("Insufficient agent-year observations")
        return {"error": "insufficient_sample", "n": len(agent_year_hires)}
    
    # Simple regression
    n = len(agent_year_hires)
    y = agent_year_hires["hire_quality"].values
    
    X = np.column_stack([
        np.ones(n),
        agent_year_hires["gamma_hat"].values,
    ])
    
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    b_acquisition = beta[1]
    
    y_hat = X @ beta
    r2 = 1 - np.var(y - y_hat) / np.var(y)
    
    # Correlation
    corr = agent_year_hires["gamma_hat"].corr(agent_year_hires["hire_quality"])
    
    print(f"\n--- Acquisition Advantage Results ---")
    print(f"b (γ̂ → hire quality): {b_acquisition:.4f}")
    print(f"Correlation(γ̂, hire quality): {corr:.4f}")
    print(f"R²: {r2:.4f}")
    
    if b_acquisition > 0:
        print("\n  Higher-capability agents hire BETTER captains (screening/recruiting advantage)")
    else:
        print("\n  No evidence of acquisition advantage (or reverse)")
    
    # By capability quartile
    agent_year_hires["gamma_q"] = pd.qcut(agent_year_hires["gamma_hat"], q=4, labels=[1, 2, 3, 4])
    quality_by_cap = agent_year_hires.groupby("gamma_q")["hire_quality"].mean()
    
    print("\n--- Mean Hire Quality by Agent Capability Quartile ---")
    print(quality_by_cap.round(3).to_string())
    
    results = {
        "b_acquisition": b_acquisition,
        "correlation": corr,
        "r2": r2,
        "n": n,
        "quality_by_cap": quality_by_cap,
    }
    
    return results


def create_sorting_mobility_figure(
    r13_results: Dict,
    r14_results: Dict,
    output_path: Optional[str] = None,
) -> None:
    """
    Create sorting and mobility visualization.
    
    Parameters
    ----------
    r13_results : Dict
        Sorting results.
    r14_results : Dict
        Switching hazard results.
    output_path : str, optional
        Path to save figure.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not available, skipping figure generation")
        return
    
    from .config import FIGURES_DIR
    from pathlib import Path
    
    if output_path is None:
        output_path = FIGURES_DIR / "r13_r14_sorting_mobility.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Sorting matrix heatmap
    ax1 = axes[0]
    sorting_matrix = r13_results["sorting_matrix"]
    
    sns.heatmap(
        sorting_matrix,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        ax=ax1,
        cbar_kws={"label": "% of Matches"},
    )
    ax1.set_xlabel("Agent Capability Quintile", fontsize=11)
    ax1.set_ylabel("Captain Skill Quintile", fontsize=11)
    ax1.set_title(f"R13: Sorting Matrix\n(Diagonal Share: {r13_results['diagonal_share']:.1f}%, Corr: {r13_results['raw_corr']:.3f})", fontsize=12, fontweight="bold")
    
    # Right: Switching rates by capability
    ax2 = axes[1]
    switch_by_cap = r14_results["switch_by_cap"]
    
    bars = ax2.bar(
        switch_by_cap.index.astype(str),
        switch_by_cap.values * 100,
        color="steelblue",
        edgecolor="black",
    )
    
    ax2.set_xlabel("Agent Capability Quartile", fontsize=11)
    ax2.set_ylabel("Switching Rate (%)", fontsize=11)
    ax2.set_title("R14: Switching Hazard by Agent Capability", fontsize=12, fontweight="bold")
    
    # Add trend line
    x = np.arange(1, 5)
    slope, intercept = np.polyfit(x, switch_by_cap.values * 100, 1)
    ax2.plot(x - 1, slope * x + intercept, "r--", linewidth=2, label=f"Slope: {slope:.2f}")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"\nSorting/mobility figure saved to {output_path}")


def run_labor_market_analysis(
    df: pd.DataFrame,
    save_outputs: bool = True,
) -> Dict:
    """
    Run full labor market analysis (R13, R14, R15).
    
    Parameters
    ----------
    df : pd.DataFrame
        Voyage data.
    save_outputs : bool
        Whether to save outputs.
        
    Returns
    -------
    Dict
        Combined labor market results.
    """
    from .config import TABLES_DIR
    from pathlib import Path
    
    # R13: Sorting
    r13_results = run_r13_sorting(df)
    
    # R14: Switching hazard
    r14_results = run_r14_switching_hazard(df)
    
    # R15: Acquisition advantage
    r15_results = run_r15_acquisition_advantage(df)
    
    if save_outputs:
        # Create figure
        create_sorting_mobility_figure(r13_results, r14_results)
        
        # Save summary
        summary = pd.DataFrame({
            "Specification": [
                "R13: Sorting (b)",
                "R13: Raw Correlation",
                "R14: α̂ → Switch",
                "R14: γ̂ → Switch",
                "R15: Acquisition Advantage",
            ],
            "Coefficient": [
                r13_results.get("b_sorting", np.nan),
                r13_results.get("raw_corr", np.nan),
                r14_results.get("b_alpha", np.nan),
                r14_results.get("b_gamma", np.nan),
                r15_results.get("b_acquisition", np.nan),
            ],
        })
        
        output_path = TABLES_DIR / "r13_r14_r15_labor_market.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_path, index=False)
        print(f"\nLabor market analysis saved to {output_path}")
    
    return {"r13": r13_results, "r14": r14_results, "r15": r15_results}


if __name__ == "__main__":
    from .data_loader import prepare_analysis_sample
    
    df = prepare_analysis_sample()
    results = run_labor_market_analysis(df)
