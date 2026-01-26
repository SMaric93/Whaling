"""
Optional extensions (R16, R17).

Implements:
- R16: Revenue decomposition (requires price data)
- R17: Settlement governance (requires settlement data)

These are optional analyses that depend on data availability.
"""

from typing import Dict, Optional
import warnings

import numpy as np
import pandas as pd

from .baseline_production import estimate_r1

warnings.filterwarnings("ignore", category=FutureWarning)


def check_price_data_available(df: pd.DataFrame) -> bool:
    """Check if price/revenue data is available."""
    price_cols = ["price", "oil_price", "revenue", "gross_proceeds"]
    return any(col in df.columns for col in price_cols)


def check_settlement_data_available(df: pd.DataFrame) -> bool:
    """Check if settlement/governance data is available."""
    settlement_cols = ["net_proceeds", "settlement", "net_to_gross", "agent_share"]
    return any(col in df.columns for col in settlement_cols)


def run_r16_revenue(
    df: pd.DataFrame,
) -> Dict:
    """
    R16: Revenue decomposition.
    
    logR_v = α_c + γ_a + δ_{vessel×period} + θ_{route×time} + ε
    
    Economic significance with price data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Voyage data.
        
    Returns
    -------
    Dict
        Revenue decomposition results.
    """
    print("\n" + "=" * 60)
    print("R16: REVENUE DECOMPOSITION")
    print("=" * 60)
    
    if not check_price_data_available(df):
        print("Price/revenue data not available")
        print("Required columns: price, oil_price, revenue, or gross_proceeds")
        return {"error": "data_not_available", "required_cols": ["price", "revenue", "gross_proceeds"]}
    
    df = df.copy()
    
    # Find revenue column
    for col in ["revenue", "gross_proceeds"]:
        if col in df.columns:
            df["log_revenue"] = np.log(df[col].clip(lower=1))
            print(f"Using {col} as revenue measure")
            break
    
    # Or construct from quantity × price
    if "log_revenue" not in df.columns and "price" in df.columns:
        df["revenue"] = df["q_total_index"] * df["price"]
        df["log_revenue"] = np.log(df["revenue"].clip(lower=1))
        print("Constructed revenue from quantity × price")
    
    if "log_revenue" not in df.columns:
        return {"error": "could_not_construct_revenue"}
    
    # Filter to valid revenue
    df = df[df["log_revenue"].notna() & np.isfinite(df["log_revenue"])].copy()
    print(f"\nSample with valid revenue: {len(df):,} voyages")
    
    # Run R1-style estimation with revenue as outcome
    results = estimate_r1(df, dependent_var="log_revenue", use_loo_sample=True)
    
    print(f"\n--- Revenue Decomposition Results ---")
    print(f"R²: {results['r2']:.4f}")
    print(f"Captain effects explain: {np.var(results['alpha_hat'])/np.var(df['log_revenue'])*100:.1f}%")
    print(f"Agent effects explain: {np.var(results['gamma_hat'])/np.var(df['log_revenue'])*100:.1f}%")
    
    return results


def run_r17_governance(
    df: pd.DataFrame,
) -> Dict:
    """
    R17: Settlement governance / netting intensity.
    
    NetToGross_v = γ_a + f(GrossProceeds) + FE_{route×time} + FE_{vessel×period} + ε
    
    Parameters
    ----------
    df : pd.DataFrame
        Voyage data.
        
    Returns
    -------
    Dict
        Governance/extraction results.
    """
    print("\n" + "=" * 60)
    print("R17: SETTLEMENT GOVERNANCE")
    print("=" * 60)
    
    if not check_settlement_data_available(df):
        print("Settlement/governance data not available")
        print("Required columns: net_proceeds, settlement, net_to_gross, or agent_share")
        return {"error": "data_not_available", "required_cols": ["net_proceeds", "net_to_gross", "agent_share"]}
    
    df = df.copy()
    
    # Find or construct net-to-gross ratio
    if "net_to_gross" in df.columns:
        y_col = "net_to_gross"
    elif "net_proceeds" in df.columns and "gross_proceeds" in df.columns:
        df["net_to_gross"] = df["net_proceeds"] / df["gross_proceeds"].clip(lower=1)
        y_col = "net_to_gross"
    elif "agent_share" in df.columns:
        y_col = "agent_share"
    else:
        return {"error": "could_not_construct_governance_measure"}
    
    # Filter to valid observations
    df = df[df[y_col].notna() & np.isfinite(df[y_col])].copy()
    print(f"\nSample with valid {y_col}: {len(df):,} voyages")
    
    if len(df) < 100:
        return {"error": "insufficient_sample", "n": len(df)}
    
    # Simple regression
    n = len(df)
    y = df[y_col].values
    
    # Agent FEs + controls
    agent_ids = df["agent_id"].unique()
    agent_map = {a: i for i, a in enumerate(agent_ids)}
    agent_idx = df["agent_id"].map(agent_map).values
    
    # One-hot encoding for agents (drop first)
    X_agent = np.zeros((n, len(agent_ids) - 1))
    for i, idx in enumerate(agent_idx):
        if idx > 0:
            X_agent[i, idx - 1] = 1
    
    # Controls
    X_controls = np.column_stack([
        np.ones(n),
        np.log(df["q_total_index"].clip(lower=1)).values,  # Gross proceeds proxy
    ])
    
    X = np.hstack([X_controls, X_agent])
    
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    
    y_hat = X @ beta
    r2 = 1 - np.var(y - y_hat) / np.var(y)
    
    # Extract agent effects
    agent_effects = np.concatenate([[0], beta[2:]])  # First agent normalized to 0
    
    print(f"\n--- Governance Results ---")
    print(f"R²: {r2:.4f}")
    print(f"Mean {y_col}: {y.mean():.3f}")
    print(f"Std dev of agent effects: {np.std(agent_effects):.4f}")
    
    # Check if higher-capability agents have different governance
    agent_fe_df = pd.DataFrame({
        "agent_id": agent_ids,
        "governance_effect": agent_effects,
    })
    
    # Merge with capability from R1
    r1_results = estimate_r1(df, use_loo_sample=False)
    agent_fe_df = agent_fe_df.merge(
        r1_results["agent_fe"][["agent_id", "gamma_hat"]],
        on="agent_id",
        how="left"
    )
    
    corr = agent_fe_df["gamma_hat"].corr(agent_fe_df["governance_effect"])
    print(f"\nCorr(capability, governance): {corr:.4f}")
    
    if corr > 0:
        print("  Higher-capability agents take LARGER shares (rent extraction)")
    else:
        print("  Higher-capability agents take SMALLER shares (value sharing)")
    
    results = {
        "r2": r2,
        "n": n,
        "mean_outcome": y.mean(),
        "agent_effect_std": np.std(agent_effects),
        "capability_governance_corr": corr,
        "agent_fe": agent_fe_df,
    }
    
    return results


def run_extensions(
    df: pd.DataFrame,
    save_outputs: bool = True,
) -> Dict:
    """
    Run optional extension analyses (R16, R17).
    
    Parameters
    ----------
    df : pd.DataFrame
        Voyage data.
    save_outputs : bool
        Whether to save outputs.
        
    Returns
    -------
    Dict
        Extension results.
    """
    from .config import TABLES_DIR
    from pathlib import Path
    
    results = {}
    
    # R16: Revenue (if data available)
    r16_results = run_r16_revenue(df)
    results["r16"] = r16_results
    
    # R17: Governance (if data available)
    r17_results = run_r17_governance(df)
    results["r17"] = r17_results
    
    if save_outputs:
        # Summary of what's available
        summary_rows = []
        
        if "error" not in r16_results:
            summary_rows.append({
                "Specification": "R16: Revenue",
                "Status": "Completed",
                "R2": r16_results.get("r2", np.nan),
            })
        else:
            summary_rows.append({
                "Specification": "R16: Revenue",
                "Status": f"Skipped ({r16_results['error']})",
                "R2": np.nan,
            })
            
        if "error" not in r17_results:
            summary_rows.append({
                "Specification": "R17: Governance",
                "Status": "Completed",
                "R2": r17_results.get("r2", np.nan),
            })
        else:
            summary_rows.append({
                "Specification": "R17: Governance",
                "Status": f"Skipped ({r17_results['error']})",
                "R2": np.nan,
            })
        
        summary = pd.DataFrame(summary_rows)
        output_path = TABLES_DIR / "r16_r17_extensions.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_path, index=False)
        print(f"\nExtensions summary saved to {output_path}")
    
    return results


if __name__ == "__main__":
    from .data_loader import prepare_analysis_sample
    
    df = prepare_analysis_sample()
    results = run_extensions(df)
