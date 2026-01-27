"""
Risk Matching Theory Analysis.

Explains the negative sorting puzzle (Corr(α,γ)≈-0.05) by reframing 
matching around VARIANCE rather than MEAN performance:
- High-Skill Captains = "Variance Creators" (high risk/high reward)
- High-Capability Agents = "Variance Absorbers" (diversified portfolios)

Specifications:
- RM1: Captain Variance Decomposition (μ_α, σ²_α)
- RM2: Agent Portfolio Breadth Metrics
- RM3: Risk Sorting Regression (portfolio_breadth ~ captain_σ²)
- RM4: Sorting Correlation Comparison (mean vs risk sorting)
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .config import OUTPUT_DIR, TABLES_DIR
from .baseline_production import estimate_r1

# =============================================================================
# Configuration
# =============================================================================

RISK_MATCHING_DIR = OUTPUT_DIR / "risk_matching"


# =============================================================================
# RM1: Captain Variance Decomposition
# =============================================================================

def compute_captain_variance_decomposition(
    df: pd.DataFrame,
    min_voyages: int = 3,
) -> Tuple[pd.DataFrame, Dict]:
    """
    RM1: Decompose captain performance into Mean (μ_α) and Variance (σ²_α).
    
    Uses residuals from R1 baseline model to compute within-captain variance,
    isolating voyage-level "variance creation" from systematic skill.
    
    Parameters
    ----------
    df : pd.DataFrame
        Voyage data with R1 estimation results (residuals, alpha_hat).
    min_voyages : int
        Minimum voyages required for reliable variance estimation.
        
    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        (captain_variance_df, diagnostics)
    """
    print("\n" + "=" * 60)
    print("RM1: CAPTAIN VARIANCE DECOMPOSITION")
    print("=" * 60)
    
    # Check if residuals are already computed
    if "residuals" not in df.columns or "alpha_hat" not in df.columns:
        print("Running R1 baseline estimation first...")
        r1_results = estimate_r1(df, use_loo_sample=True)
        df = r1_results["df"]
    
    # Filter to captains with sufficient voyages
    captain_voyages = df.groupby("captain_id").size()
    valid_captains = captain_voyages[captain_voyages >= min_voyages].index
    df_valid = df[df["captain_id"].isin(valid_captains)].copy()
    
    print(f"\nCaptains with ≥{min_voyages} voyages: {len(valid_captains):,}")
    print(f"Voyages in sample: {len(df_valid):,}")
    
    # Compute captain-level statistics
    captain_stats = df_valid.groupby("captain_id").agg({
        "alpha_hat": "first",           # Captain FE (mean skill)
        "residuals": ["mean", "var"],   # Within-captain residual stats
        "log_q": ["mean", "var"],       # Raw output stats
        "voyage_id": "count",           # Number of voyages
    }).reset_index()
    
    captain_stats.columns = [
        "captain_id", 
        "alpha_hat",          # μ_α: Mean skill (FE estimate)
        "resid_mean",         # Should be ~0
        "sigma_sq_alpha",     # σ²_α: Within-captain variance
        "output_mean",        # Mean log output
        "output_var",         # Total output variance
        "n_voyages",          # Sample size
    ]
    
    # Handle NaN variances (captains with exactly min_voyages, var requires >1)
    captain_stats["sigma_sq_alpha"] = captain_stats["sigma_sq_alpha"].fillna(0)
    
    # Standardize variance for interpretability
    captain_stats["sigma_alpha_std"] = (
        (captain_stats["sigma_sq_alpha"] - captain_stats["sigma_sq_alpha"].mean()) /
        captain_stats["sigma_sq_alpha"].std()
    )
    
    # Identify "Variance Creators" (top tertile of σ²_α)
    captain_stats["variance_tertile"] = pd.qcut(
        captain_stats["sigma_sq_alpha"], 
        q=3, 
        labels=["Low", "Medium", "High"]
    )
    captain_stats["is_variance_creator"] = (
        captain_stats["variance_tertile"] == "High"
    ).astype(int)
    
    # Diagnostics
    diag = {
        "n_captains": len(captain_stats),
        "n_voyages": len(df_valid),
        "mean_sigma_sq": captain_stats["sigma_sq_alpha"].mean(),
        "median_sigma_sq": captain_stats["sigma_sq_alpha"].median(),
        "std_sigma_sq": captain_stats["sigma_sq_alpha"].std(),
        "n_variance_creators": captain_stats["is_variance_creator"].sum(),
        "corr_skill_variance": captain_stats["alpha_hat"].corr(
            captain_stats["sigma_sq_alpha"]
        ),
    }
    
    print(f"\n--- Captain Variance Statistics ---")
    print(f"Mean σ²_α: {diag['mean_sigma_sq']:.4f}")
    print(f"Median σ²_α: {diag['median_sigma_sq']:.4f}")
    print(f"Variance Creators (top tertile): {diag['n_variance_creators']:,}")
    print(f"Corr(α̂_mean, σ²_α): {diag['corr_skill_variance']:.4f}")
    
    if diag["corr_skill_variance"] > 0:
        print("  → High-skill captains tend to have higher variance (explorers)")
    else:
        print("  → High-skill captains tend to have lower variance (specialists)")
    
    return captain_stats, diag


# =============================================================================
# RM2: Agent Portfolio Breadth Metrics
# =============================================================================

def compute_agent_portfolio_breadth(
    df: pd.DataFrame,
    min_voyages: int = 3,
) -> Tuple[pd.DataFrame, Dict]:
    """
    RM2: Compute agent portfolio breadth (diversification metrics).
    
    Diversified agents can absorb high-variance captains via portfolio effects.
    
    Parameters
    ----------
    df : pd.DataFrame
        Voyage data with agent, route, port information.
    min_voyages : int
        Minimum voyages required for reliable metrics.
        
    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        (agent_portfolio_df, diagnostics)
    """
    print("\n" + "=" * 60)
    print("RM2: AGENT PORTFOLIO BREADTH")
    print("=" * 60)
    
    # Check for required columns
    route_col = "ground_or_route" if "ground_or_route" in df.columns else "route"
    if route_col not in df.columns:
        route_col = "route_or_ground" if "route_or_ground" in df.columns else None
    
    port_col = "home_port" if "home_port" in df.columns else "port"
    
    # Filter to agents with sufficient voyages
    agent_voyages = df.groupby("agent_id").size()
    valid_agents = agent_voyages[agent_voyages >= min_voyages].index
    df_valid = df[df["agent_id"].isin(valid_agents)].copy()
    
    print(f"\nAgents with ≥{min_voyages} voyages: {len(valid_agents):,}")
    print(f"Voyages in sample: {len(df_valid):,}")
    
    # Compute agent-level portfolio metrics
    agg_dict = {
        "voyage_id": "count",          # Number of voyages
        "captain_id": "nunique",       # Captain diversity
    }
    
    if route_col and route_col in df_valid.columns:
        agg_dict[route_col] = "nunique"  # Route diversity
    if port_col in df_valid.columns:
        agg_dict[port_col] = "nunique"   # Port diversity
    if "log_q" in df_valid.columns:
        agg_dict["log_q"] = ["mean", "var"]  # Output stats
    if "gamma_hat" in df_valid.columns:
        agg_dict["gamma_hat"] = "first"   # Agent FE
    
    agent_stats = df_valid.groupby("agent_id").agg(agg_dict).reset_index()
    
    # Flatten column names
    agent_stats.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col 
        for col in agent_stats.columns
    ]
    
    # Standardize column names
    col_renames = {
        "voyage_id_count": "n_voyages",
        "captain_id_nunique": "n_captains",
        f"{route_col}_nunique": "n_routes" if route_col else None,
        f"{port_col}_nunique": "n_ports",
        "log_q_mean": "output_mean",
        "log_q_var": "output_var",
        "gamma_hat_first": "gamma_hat",
    }
    col_renames = {k: v for k, v in col_renames.items() if k in agent_stats.columns and v}
    agent_stats = agent_stats.rename(columns=col_renames)
    
    # Compute portfolio breadth = routes + ports (diversification score)
    n_routes = agent_stats.get("n_routes", pd.Series(0, index=agent_stats.index))
    n_ports = agent_stats.get("n_ports", pd.Series(0, index=agent_stats.index))
    agent_stats["portfolio_breadth"] = n_routes.fillna(0) + n_ports.fillna(0)
    
    # Standardize for regression
    agent_stats["portfolio_breadth_std"] = (
        (agent_stats["portfolio_breadth"] - agent_stats["portfolio_breadth"].mean()) /
        agent_stats["portfolio_breadth"].std()
    )
    
    # Identify "Variance Absorbers" (top tertile of portfolio breadth)
    agent_stats["portfolio_tertile"] = pd.qcut(
        agent_stats["portfolio_breadth"].rank(method="first"), 
        q=3, 
        labels=["Narrow", "Medium", "Broad"]
    )
    agent_stats["is_variance_absorber"] = (
        agent_stats["portfolio_tertile"] == "Broad"
    ).astype(int)
    
    # Diagnostics
    diag = {
        "n_agents": len(agent_stats),
        "n_voyages": len(df_valid),
        "mean_portfolio_breadth": agent_stats["portfolio_breadth"].mean(),
        "median_portfolio_breadth": agent_stats["portfolio_breadth"].median(),
        "mean_captain_diversity": agent_stats.get("n_captains", pd.Series([0])).mean(),
        "n_variance_absorbers": agent_stats["is_variance_absorber"].sum(),
    }
    
    print(f"\n--- Agent Portfolio Statistics ---")
    print(f"Mean portfolio breadth: {diag['mean_portfolio_breadth']:.2f}")
    print(f"Median portfolio breadth: {diag['median_portfolio_breadth']:.2f}")
    print(f"Mean captain diversity: {diag['mean_captain_diversity']:.2f}")
    print(f"Variance Absorbers (top tertile): {diag['n_variance_absorbers']:,}")
    
    return agent_stats, diag


# =============================================================================
# RM3: Risk Sorting Regression
# =============================================================================

def run_risk_sorting_regression(
    df: pd.DataFrame,
    captain_variance: pd.DataFrame,
    agent_portfolio: pd.DataFrame,
) -> Dict:
    """
    RM3: Regress agent portfolio breadth on average captain variance.
    
    portfolio_breadth_a = β₀ + β₁·captain_σ²_weighted + FE_{decade} + ε
    
    Prediction: β₁ > 0 → Diversified agents hire high-variance captains
    
    Parameters
    ----------
    df : pd.DataFrame
        Voyage data with captain_id, agent_id.
    captain_variance : pd.DataFrame
        Captain-level variance decomposition from RM1.
    agent_portfolio : pd.DataFrame
        Agent-level portfolio metrics from RM2.
        
    Returns
    -------
    Dict
        Regression results.
    """
    print("\n" + "=" * 60)
    print("RM3: RISK SORTING REGRESSION")
    print("=" * 60)
    
    # Merge captain variance to voyage data
    df_merged = df.merge(
        captain_variance[["captain_id", "sigma_sq_alpha", "sigma_alpha_std", "alpha_hat"]],
        on="captain_id",
        how="inner",
        suffixes=("", "_capt")
    )
    
    # Compute agent-level weighted average of captain variance
    agent_captain_variance = df_merged.groupby("agent_id").agg({
        "sigma_sq_alpha": "mean",     # Mean σ²_α of captains hired
        "sigma_alpha_std": "mean",    # Standardized version
        "alpha_hat": "mean",          # Mean skill of captains hired (for comparison)
    }).reset_index()
    
    agent_captain_variance.columns = [
        "agent_id", 
        "captain_variance_mean",
        "captain_variance_std_mean", 
        "captain_skill_mean",
    ]
    
    # Merge with agent portfolio
    analysis_df = agent_portfolio.merge(
        agent_captain_variance,
        on="agent_id",
        how="inner"
    )
    
    print(f"\nAgent-level sample: {len(analysis_df):,} agents")
    
    # RM3a: Simple regression: portfolio_breadth ~ captain_variance
    valid = analysis_df.dropna(subset=["portfolio_breadth", "captain_variance_mean"])
    
    if len(valid) < 30:
        print(f"WARNING: Small sample size ({len(valid)} agents)")
    
    y = valid["portfolio_breadth"].values
    X = np.column_stack([
        np.ones(len(valid)),
        valid["captain_variance_mean"].values,
    ])
    
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    y_hat = X @ beta
    residuals = y - y_hat
    
    # Standard errors (OLS)
    n, k = X.shape
    sigma_sq = np.sum(residuals**2) / (n - k)
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(sigma_sq * XtX_inv))
    t_stats = beta / se
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - k))
    
    r2 = 1 - np.var(residuals) / np.var(y)
    
    # Key results
    b_risk_sorting = beta[1]
    se_risk = se[1]
    t_risk = t_stats[1]
    p_risk = p_values[1]
    
    print(f"\n--- Risk Sorting Regression Results ---")
    print(f"portfolio_breadth ~ captain_σ²_mean")
    print(f"  N = {n:,}")
    print(f"  R² = {r2:.4f}")
    print(f"\n  β₁ (captain_σ²) = {b_risk_sorting:.4f}")
    print(f"  SE = {se_risk:.4f}")
    print(f"  t = {t_risk:.2f}")
    print(f"  p = {p_risk:.4f}")
    
    stars = "***" if p_risk < 0.01 else "**" if p_risk < 0.05 else "*" if p_risk < 0.10 else ""
    print(f"\n  → β₁ = {b_risk_sorting:.4f}{stars}")
    
    if b_risk_sorting > 0 and p_risk < 0.10:
        print("  ✓ POSITIVE RISK SORTING: Diversified agents hire high-variance captains")
    elif b_risk_sorting < 0 and p_risk < 0.10:
        print("  ✗ NEGATIVE RISK SORTING: Diversified agents avoid high-variance captains")
    else:
        print("  ○ No significant risk sorting detected")
    
    # RM3b: Correlation-based test (robustness)
    corr_risk = valid["portfolio_breadth"].corr(valid["captain_variance_mean"])
    corr_skill = valid["portfolio_breadth"].corr(valid["captain_skill_mean"])
    
    print(f"\n--- Correlation Tests ---")
    print(f"Corr(portfolio_breadth, captain_σ²):  {corr_risk:.4f} (RISK sorting)")
    print(f"Corr(portfolio_breadth, captain_α̂):  {corr_skill:.4f} (SKILL sorting)")
    
    results = {
        "n_agents": n,
        "r2": r2,
        "b_risk_sorting": b_risk_sorting,
        "se_risk": se_risk,
        "t_risk": t_risk,
        "p_risk": p_risk,
        "corr_portfolio_variance": corr_risk,
        "corr_portfolio_skill": corr_skill,
        "analysis_df": analysis_df,
    }
    
    return results


# =============================================================================
# RM4: Sorting Correlation Comparison
# =============================================================================

def compare_sorting_correlations(
    df: pd.DataFrame,
    captain_variance: pd.DataFrame,
    agent_portfolio: pd.DataFrame,
    rm3_results: Dict,
) -> pd.DataFrame:
    """
    RM4: Compare sorting on mean (α̂,γ̂) vs sorting on risk (σ²_α, portfolio).
    
    Present three key correlations:
    1. Corr(α̂_mean, γ̂) = traditional sorting on MEAN (expect ~-0.05)
    2. Corr(σ²_α, portfolio_breadth) = sorting on RISK (expect positive)
    3. Corr(α̂_mean, σ²_α) = skill vs variance relationship
    
    Parameters
    ----------
    df : pd.DataFrame
        Voyage data with α̂ and γ̂.
    captain_variance : pd.DataFrame
        Captain variance stats from RM1.
    agent_portfolio : pd.DataFrame
        Agent portfolio stats from RM2.
    rm3_results : Dict
        Results from RM3.
        
    Returns
    -------
    pd.DataFrame
        Comparison table of sorting correlations.
    """
    print("\n" + "=" * 60)
    print("RM4: SORTING CORRELATION COMPARISON")
    print("=" * 60)
    
    comparisons = []
    
    # 1. Traditional sorting: Corr(α̂, γ̂) at voyage level
    if "alpha_hat" in df.columns and "gamma_hat" in df.columns:
        valid = df.dropna(subset=["alpha_hat", "gamma_hat"])
        corr_mean = valid["alpha_hat"].corr(valid["gamma_hat"])
        comparisons.append({
            "Type": "MEAN Sorting",
            "Variable_1": "α̂ (Captain Skill)",
            "Variable_2": "γ̂ (Agent Capability)",
            "Correlation": corr_mean,
            "Level": "Voyage",
            "N": len(valid),
            "Interpretation": "Negative = high-skill captains with low-cap agents",
        })
    
    # 2. Risk sorting: Corr(σ²_α, portfolio_breadth) at agent level
    corr_risk = rm3_results.get("corr_portfolio_variance", np.nan)
    n_agents = rm3_results.get("n_agents", 0)
    comparisons.append({
        "Type": "RISK Sorting",
        "Variable_1": "σ²_α (Captain Variance)",
        "Variable_2": "Portfolio Breadth (Agent)",
        "Correlation": corr_risk,
        "Level": "Agent",
        "N": n_agents,
        "Interpretation": "Positive = diversified agents hire high-variance captains",
    })
    
    # 3. Skill-variance relationship: Corr(α̂, σ²_α) at captain level
    corr_skill_var = captain_variance["alpha_hat"].corr(
        captain_variance["sigma_sq_alpha"]
    )
    comparisons.append({
        "Type": "Skill-Variance Link",
        "Variable_1": "α̂ (Captain Skill)",
        "Variable_2": "σ²_α (Captain Variance)",
        "Correlation": corr_skill_var,
        "Level": "Captain",
        "N": len(captain_variance),
        "Interpretation": "Positive = skilled captains are risk-takers",
    })
    
    # 4. Traditional agent-level sorting for comparison
    if "gamma_hat" in agent_portfolio.columns:
        analysis_df = rm3_results.get("analysis_df", pd.DataFrame())
        if "gamma_hat" in analysis_df.columns and "captain_skill_mean" in analysis_df.columns:
            corr_agent_skill = analysis_df["gamma_hat"].corr(
                analysis_df["captain_skill_mean"]
            )
            comparisons.append({
                "Type": "Agent-Level MEAN Sorting",
                "Variable_1": "γ̂ (Agent Capability)",
                "Variable_2": "Avg α̂ (Captain Skill)",
                "Correlation": corr_agent_skill,
                "Level": "Agent",
                "N": len(analysis_df),
                "Interpretation": "Traditional sorting at agent level",
            })
    
    comparison_df = pd.DataFrame(comparisons)
    
    print("\n--- Sorting Correlation Comparison ---")
    print(comparison_df[["Type", "Correlation", "Level", "N"]].to_string(index=False))
    
    # Key interpretation
    print("\n--- KEY FINDING ---")
    if len(comparisons) >= 2:
        mean_sort = comparisons[0]["Correlation"]
        risk_sort = comparisons[1]["Correlation"]
        
        print(f"Traditional MEAN sorting: Corr(α̂, γ̂) = {mean_sort:.4f}")
        print(f"Novel RISK sorting: Corr(σ², portfolio) = {risk_sort:.4f}")
        
        if mean_sort < 0 and risk_sort > 0:
            print("\n✓ RISK MATCHING THEORY SUPPORTED:")
            print("  - Negative sorting on MEAN is actually POSITIVE sorting on RISK")
            print("  - High-variance captains match with diversified agents")
            print("  - Project economies organize by matching Risk-Takers with Risk-Absorbers")
        elif mean_sort < 0 and risk_sort <= 0:
            print("\n✗ Risk matching theory not supported - sorting negative on both dimensions")
        elif mean_sort >= 0:
            print("\n○ No negative sorting puzzle to explain")
    
    return comparison_df


# =============================================================================
# Main Orchestration
# =============================================================================

def run_risk_matching_analysis(
    df: Optional[pd.DataFrame] = None,
    save_outputs: bool = True,
    min_voyages: int = 3,
) -> Dict:
    """
    Run full Risk Matching Theory analysis (RM1-RM4).
    
    Parameters
    ----------
    df : pd.DataFrame, optional
        Voyage data. If None, loads from prepare_analysis_sample().
    save_outputs : bool
        Whether to save outputs.
    min_voyages : int
        Minimum voyages for captain/agent inclusion.
        
    Returns
    -------
    Dict
        All Risk Matching analysis results.
    """
    print("=" * 70)
    print("RISK MATCHING THEORY ANALYSIS")
    print("Explaining Negative Sorting Through Variance/Risk Matching")
    print("=" * 70)
    
    # Load data if not provided
    if df is None:
        from .data_loader import prepare_analysis_sample
        df = prepare_analysis_sample()
    
    # Run R1 to get FE estimates if not present
    if "alpha_hat" not in df.columns or "gamma_hat" not in df.columns:
        print("\nRunning baseline AKM estimation...")
        r1_results = estimate_r1(df, use_loo_sample=True)
        df = r1_results["df"]
    
    print(f"\nAnalysis sample: {len(df):,} voyages")
    print(f"Captains: {df['captain_id'].nunique():,}")
    print(f"Agents: {df['agent_id'].nunique():,}")
    
    # RM1: Captain Variance Decomposition
    captain_variance, rm1_diag = compute_captain_variance_decomposition(
        df, min_voyages=min_voyages
    )
    
    # RM2: Agent Portfolio Breadth
    agent_portfolio, rm2_diag = compute_agent_portfolio_breadth(
        df, min_voyages=min_voyages
    )
    
    # RM3: Risk Sorting Regression
    rm3_results = run_risk_sorting_regression(
        df, captain_variance, agent_portfolio
    )
    
    # RM4: Sorting Correlation Comparison
    comparison_df = compare_sorting_correlations(
        df, captain_variance, agent_portfolio, rm3_results
    )
    
    # Save outputs
    if save_outputs:
        save_risk_matching_outputs(
            captain_variance, agent_portfolio, rm3_results, comparison_df,
            rm1_diag, rm2_diag
        )
    
    # Summary
    print_risk_matching_summary(rm1_diag, rm2_diag, rm3_results, comparison_df)
    
    results = {
        "captain_variance": captain_variance,
        "agent_portfolio": agent_portfolio,
        "rm1_diagnostics": rm1_diag,
        "rm2_diagnostics": rm2_diag,
        "rm3_results": rm3_results,
        "comparison_df": comparison_df,
    }
    
    return results


def save_risk_matching_outputs(
    captain_variance: pd.DataFrame,
    agent_portfolio: pd.DataFrame,
    rm3_results: Dict,
    comparison_df: pd.DataFrame,
    rm1_diag: Dict,
    rm2_diag: Dict,
) -> None:
    """Save all Risk Matching outputs to files."""
    RISK_MATCHING_DIR.mkdir(parents=True, exist_ok=True)
    
    # Captain variance
    captain_variance.to_csv(
        RISK_MATCHING_DIR / "captain_variance.csv", 
        index=False
    )
    
    # Agent portfolio
    agent_portfolio.to_csv(
        RISK_MATCHING_DIR / "agent_portfolio.csv", 
        index=False
    )
    
    # Comparison table
    comparison_df.to_csv(
        RISK_MATCHING_DIR / "sorting_comparison.csv", 
        index=False
    )
    
    # Summary table
    summary = pd.DataFrame({
        "Metric": [
            "N Captains (RM1)",
            "Captain Mean σ²_α",
            "Captain Corr(α̂, σ²)",
            "N Agents (RM2)",
            "Agent Mean Portfolio Breadth",
            "RM3: β₁ (Risk Sorting)",
            "RM3: p-value",
            "RM3: R²",
            "Corr(σ², portfolio)",
            "Corr(α̂, γ̂) [Traditional]",
        ],
        "Value": [
            rm1_diag["n_captains"],
            rm1_diag["mean_sigma_sq"],
            rm1_diag["corr_skill_variance"],
            rm2_diag["n_agents"],
            rm2_diag["mean_portfolio_breadth"],
            rm3_results["b_risk_sorting"],
            rm3_results["p_risk"],
            rm3_results["r2"],
            rm3_results["corr_portfolio_variance"],
            comparison_df.iloc[0]["Correlation"] if len(comparison_df) > 0 else np.nan,
        ],
    })
    
    summary.to_csv(
        RISK_MATCHING_DIR / "risk_matching_summary.csv", 
        index=False
    )
    
    print(f"\nOutputs saved to {RISK_MATCHING_DIR}")


def print_risk_matching_summary(
    rm1_diag: Dict,
    rm2_diag: Dict,
    rm3_results: Dict,
    comparison_df: pd.DataFrame,
) -> None:
    """Print summary of Risk Matching analysis."""
    print("\n" + "=" * 70)
    print("RISK MATCHING THEORY: SUMMARY")
    print("=" * 70)
    
    print("\n--- Key Results ---")
    
    # Captain variance creators
    print(f"1. VARIANCE CREATORS (Captains):")
    print(f"   • N = {rm1_diag['n_captains']:,} captains")
    print(f"   • Mean σ²_α = {rm1_diag['mean_sigma_sq']:.4f}")
    print(f"   • Corr(skill, variance) = {rm1_diag['corr_skill_variance']:.4f}")
    
    # Agent variance absorbers
    print(f"\n2. VARIANCE ABSORBERS (Agents):")
    print(f"   • N = {rm2_diag['n_agents']:,} agents")
    print(f"   • Mean portfolio breadth = {rm2_diag['mean_portfolio_breadth']:.2f}")
    
    # Risk sorting result
    b = rm3_results["b_risk_sorting"]
    p = rm3_results["p_risk"]
    stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
    
    print(f"\n3. RISK SORTING REGRESSION:")
    print(f"   • β₁ = {b:.4f}{stars} (p = {p:.4f})")
    
    if b > 0 and p < 0.10:
        print(f"   • ✓ POSITIVE risk sorting confirmed")
    else:
        print(f"   • Risk sorting not statistically significant")
    
    # Final verdict
    print("\n" + "-" * 70)
    print("THEORETICAL CONTRIBUTION:")
    print("-" * 70)
    
    if len(comparison_df) >= 2:
        mean_sort = comparison_df.iloc[0]["Correlation"]
        risk_sort = comparison_df.iloc[1]["Correlation"]
        
        if mean_sort < 0 and risk_sort > 0:
            print("""
The negative sorting puzzle (Corr(α,γ) ≈ -0.05) is EXPLAINED:

  • On MEAN performance: Negative sorting (puzzling)
  • On RISK/VARIANCE: POSITIVE sorting (theoretically sensible)

INTERPRETATION: Project economies organize by matching:
  - Risk-TAKERS (high-variance captains = explorers)
  - with Risk-ABSORBERS (diversified agents = portfolio managers)

This transforms a confusing descriptive result into a major theoretical
contribution about organizational design in high-uncertainty environments.
""")
        else:
            print(f"""
Risk Matching Theory: Mixed Evidence
  • Traditional sorting: {mean_sort:.4f}
  • Risk sorting: {risk_sort:.4f}
  
Further investigation needed if risk sorting is not positive.
""")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    results = run_risk_matching_analysis(save_outputs=True)
