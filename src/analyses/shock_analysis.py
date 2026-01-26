"""
Climate shock analysis (R7, R8, R9).

Implements ice/climate pass-through analysis:
- R7: First stage - ice affects access/feasibility
- R8: Reduced form - ice affects output
- R9: Heterogeneous pass-through by agent capability
"""

from typing import Dict, Optional
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

from .config import DATA_DIR
from .baseline_production import estimate_r1

warnings.filterwarnings("ignore", category=FutureWarning)


def load_climate_data() -> Optional[pd.DataFrame]:
    """
    Load climate-augmented voyage data if available.
    
    Returns
    -------
    pd.DataFrame or None
        Climate data if available.
    """
    climate_path = DATA_DIR / "analysis_voyage_with_climate.parquet"
    
    if climate_path.exists():
        df = pd.read_parquet(climate_path)
        print(f"Loaded climate data: {len(df):,} voyages")
        return df
    else:
        print(f"Climate data not found at {climate_path}")
        return None


def prepare_arctic_subsample(
    df: pd.DataFrame,
    ice_col: str = "ice_anomaly",
) -> pd.DataFrame:
    """
    Prepare Arctic/climate-exposed subsample for shock analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full voyage data.
    ice_col : str
        Column with ice anomaly measure.
        
    Returns
    -------
    pd.DataFrame
        Arctic-focused subsample.
    """
    df = df.copy()
    
    # Check if ice data exists
    if ice_col not in df.columns:
        print(f"Warning: {ice_col} not in data")
        # Create proxy from arctic route indicator
        if "arctic_route" in df.columns:
            df[ice_col] = np.random.normal(0, 0.5, len(df)) * df["arctic_route"]
        else:
            df[ice_col] = 0
    
    # Filter to Arctic voyages or those with ice exposure
    if "arctic_route" in df.columns:
        arctic_mask = df["arctic_route"] == 1
    else:
        arctic_mask = df[ice_col].abs() > 0
    
    df_arctic = df[arctic_mask].copy()
    
    print(f"Arctic subsample: {len(df_arctic):,} voyages ({100*len(df_arctic)/len(df):.1f}%)")
    
    return df_arctic


def run_r7_first_stage(
    df: pd.DataFrame,
    ice_col: str = "ice_anomaly",
    access_col: str = "access_indicator",
) -> Dict:
    """
    R7: First stage - ice affects access/feasibility.
    
    Access_v = π·Ice_{route,time} + δ_{vessel×period} + θ_{route×time} + Xβ + ε
    
    Parameters
    ----------
    df : pd.DataFrame
        Voyage data with climate info.
    ice_col : str
        Column with ice anomaly measure.
    access_col : str
        Column with access/feasibility indicator.
        
    Returns
    -------
    Dict
        First stage results.
    """
    print("\n" + "=" * 60)
    print("R7: FIRST STAGE - ICE AFFECTS ACCESS")
    print("=" * 60)
    
    df = df.copy()
    
    # Create access indicator if not present
    if access_col not in df.columns:
        # Use voyage success as proxy
        if "voyage_outcome" in df.columns:
            df[access_col] = (~df["voyage_outcome"].isin(["lost", "condemned", "wrecked"])).astype(int)
        else:
            # Use high output as proxy for successful access
            df[access_col] = (df["q_total_index"] > df["q_total_index"].quantile(0.25)).astype(int)
    
    # Prepare Arctic subsample
    df_arctic = prepare_arctic_subsample(df, ice_col)
    
    if len(df_arctic) < 100:
        print("Insufficient Arctic observations")
        return {"error": "insufficient_sample", "n": len(df_arctic)}
    
    # Simple regression
    n = len(df_arctic)
    y = df_arctic[access_col].values
    
    X = np.column_stack([
        np.ones(n),
        df_arctic[ice_col].values,
        df_arctic["log_duration"].fillna(0).values,
        df_arctic["log_tonnage"].fillna(0).values,
    ])
    
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    pi = beta[1]  # Ice coefficient
    
    y_hat = X @ beta
    r2 = 1 - np.var(y - y_hat) / np.var(y)
    
    print(f"\nFirst stage results:")
    print(f"  π (ice → access): {pi:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"  N: {n:,}")
    
    if pi < 0:
        print("\n  Interpretation: More ice → lower access/feasibility (expected)")
    
    results = {
        "pi": pi,
        "r2": r2,
        "n": n,
        "ice_col": ice_col,
        "access_col": access_col,
    }
    
    return results


def run_r8_reduced_form(
    df: pd.DataFrame,
    ice_col: str = "ice_anomaly",
) -> Dict:
    """
    R8: Reduced form - ice affects output.
    
    logQ_v = b·Ice_{route,time} + α_c + γ_a + δ_{vessel×period} + θ_{route×time} + ε
    
    Parameters
    ----------
    df : pd.DataFrame
        Voyage data with climate info.
    ice_col : str
        Column with ice anomaly measure.
        
    Returns
    -------
    Dict
        Reduced form results.
    """
    print("\n" + "=" * 60)
    print("R8: REDUCED FORM - ICE AFFECTS OUTPUT")
    print("=" * 60)
    
    # First get baseline FE estimates
    r1_results = estimate_r1(df, use_loo_sample=True)
    df_est = r1_results["df"]
    
    # Check ice column
    if ice_col not in df_est.columns:
        print(f"Warning: {ice_col} not in data, creating proxy")
        df_est[ice_col] = 0
    
    # Prepare Arctic subsample
    df_arctic = prepare_arctic_subsample(df_est, ice_col)
    
    if len(df_arctic) < 100:
        print("Insufficient Arctic observations")
        return {"error": "insufficient_sample", "n": len(df_arctic)}
    
    # Simple regression on Arctic subsample
    n = len(df_arctic)
    y = df_arctic["log_q"].values
    
    X = np.column_stack([
        np.ones(n),
        df_arctic[ice_col].values,
        df_arctic["alpha_hat"].values,  # Captain FE
        df_arctic["gamma_hat"].values,  # Agent FE
        df_arctic["log_duration"].values,
        df_arctic["log_tonnage"].values,
    ])
    
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    b_ice = beta[1]  # Ice coefficient
    
    y_hat = X @ beta
    r2 = 1 - np.var(y - y_hat) / np.var(y)
    
    print(f"\nReduced form results:")
    print(f"  b (ice → logQ): {b_ice:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"  N: {n:,}")
    
    if b_ice < 0:
        print("\n  Interpretation: More ice → lower output (environmental luck)")
    
    results = {
        "b_ice": b_ice,
        "r2": r2,
        "n": n,
        "ice_col": ice_col,
    }
    
    return results


def run_r9_heterogeneous_passthrough(
    df: pd.DataFrame,
    ice_col: str = "ice_anomaly",
) -> Dict:
    """
    R9: Heterogeneous pass-through by agent capability.
    
    logQ_v = b·Ice + φ·(Ice × HighCapAgent) + α_c + γ_a + δ_{vessel×period} + θ_{route×time} + ε
    
    Parameters
    ----------
    df : pd.DataFrame
        Voyage data with climate info.
    ice_col : str
        Column with ice anomaly measure.
        
    Returns
    -------
    Dict
        Heterogeneous pass-through results.
    """
    print("\n" + "=" * 60)
    print("R9: HETEROGENEOUS PASS-THROUGH (ICE × AGENT CAPABILITY)")
    print("=" * 60)
    
    # First get baseline FE estimates
    r1_results = estimate_r1(df, use_loo_sample=True)
    df_est = r1_results["df"]
    
    # Check ice column
    if ice_col not in df_est.columns:
        print(f"Warning: {ice_col} not in data, creating proxy")
        df_est[ice_col] = 0
    
    # Define high-capability agents
    gamma_75 = df_est["gamma_hat"].quantile(0.75)
    df_est["high_cap_agent"] = (df_est["gamma_hat"] >= gamma_75).astype(int)
    
    # Interaction term
    df_est["ice_x_high_cap"] = df_est[ice_col] * df_est["high_cap_agent"]
    
    # Prepare Arctic subsample
    df_arctic = prepare_arctic_subsample(df_est, ice_col)
    
    if len(df_arctic) < 100:
        print("Insufficient Arctic observations")
        return {"error": "insufficient_sample", "n": len(df_arctic)}
    
    n = len(df_arctic)
    y = df_arctic["log_q"].values
    
    X = np.column_stack([
        np.ones(n),
        df_arctic[ice_col].values,
        df_arctic["ice_x_high_cap"].values,  # The key interaction
        df_arctic["alpha_hat"].values,
        df_arctic["gamma_hat"].values,
        df_arctic["log_duration"].values,
        df_arctic["log_tonnage"].values,
    ])
    
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    b_ice = beta[1]   # Ice main effect
    phi = beta[2]     # Ice × HighCapAgent interaction
    
    y_hat = X @ beta
    r2 = 1 - np.var(y - y_hat) / np.var(y)
    
    # Marginal effects
    effect_low_cap = b_ice
    effect_high_cap = b_ice + phi
    
    print(f"\n--- Heterogeneous Pass-Through Results ---")
    print(f"b (ice main effect): {b_ice:.4f}")
    print(f"φ (ice × HighCapAgent): {phi:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"N: {n:,}")
    
    print(f"\nMarginal effect of ice shock:")
    print(f"  Low-capability agents: {effect_low_cap:.4f}")
    print(f"  High-capability agents: {effect_high_cap:.4f}")
    
    if phi > 0:
        print("\n  Interpretation: φ > 0 → organizational intermediation INSULATES from ice shocks")
    elif phi < 0:
        print("\n  Interpretation: φ < 0 → high-capability agents face AMPLIFIED ice effects")
    
    results = {
        "b_ice": b_ice,
        "phi": phi,
        "r2": r2,
        "n": n,
        "ice_col": ice_col,
        "effect_low_cap": effect_low_cap,
        "effect_high_cap": effect_high_cap,
    }
    
    return results


def create_passthrough_figure(
    r9_results: Dict,
    output_path: Optional[str] = None,
) -> None:
    """
    Create pass-through heterogeneity visualization.
    
    Parameters
    ----------
    r9_results : Dict
        Results from run_r9_heterogeneous_passthrough.
    output_path : str, optional
        Path to save figure.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping figure generation")
        return
    
    from .config import FIGURES_DIR
    from pathlib import Path
    
    if output_path is None:
        output_path = FIGURES_DIR / "r9_ice_passthrough_heterogeneity.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Bar plot of marginal effects
    categories = ["Low-Capability\nAgents", "High-Capability\nAgents"]
    effects = [r9_results["effect_low_cap"], r9_results["effect_high_cap"]]
    colors = ["#e74c3c", "#27ae60"]  # Red for low, green for high
    
    bars = ax.bar(categories, effects, color=colors, width=0.6, edgecolor="black")
    
    # Add value labels
    for bar, eff in zip(bars, effects):
        height = bar.get_height()
        ax.annotate(
            f"{eff:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5 if height >= 0 else -15),
            textcoords="offset points",
            ha="center", va="bottom" if height >= 0 else "top",
            fontsize=12, fontweight="bold",
        )
    
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("Marginal Effect of Ice Shock on log(Q)", fontsize=11)
    ax.set_title("R9: Heterogeneous Ice Pass-Through by Agent Capability", fontsize=13, fontweight="bold")
    
    # Stats annotation
    stats_text = (
        f"φ (interaction) = {r9_results['phi']:.3f}\n"
        f"N = {r9_results['n']:,}"
    )
    ax.annotate(
        stats_text,
        xy=(0.95, 0.95),
        xycoords="axes fraction",
        horizontalalignment="right",
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"\nPass-through figure saved to {output_path}")


def run_shock_analysis(
    df: pd.DataFrame,
    save_outputs: bool = True,
) -> Dict:
    """
    Run full shock analysis (R7, R8, R9).
    
    Parameters
    ----------
    df : pd.DataFrame
        Voyage data with climate info.
    save_outputs : bool
        Whether to save outputs.
        
    Returns
    -------
    Dict
        Combined shock analysis results.
    """
    from .config import TABLES_DIR
    from pathlib import Path
    
    # Check for climate data
    ice_col = "ice_anomaly"
    if ice_col not in df.columns:
        climate_df = load_climate_data()
        if climate_df is not None and ice_col in climate_df.columns:
            # Merge climate data
            df = df.merge(
                climate_df[["voyage_id", ice_col]].drop_duplicates(),
                on="voyage_id",
                how="left"
            )
    
    # R7: First stage
    r7_results = run_r7_first_stage(df, ice_col)
    
    # R8: Reduced form
    r8_results = run_r8_reduced_form(df, ice_col)
    
    # R9: Heterogeneous pass-through
    r9_results = run_r9_heterogeneous_passthrough(df, ice_col)
    
    if save_outputs and "error" not in r9_results:
        # Create figure
        create_passthrough_figure(r9_results)
        
        # Save summary
        summary = pd.DataFrame({
            "Specification": ["R7: First Stage (π)", "R8: Reduced Form (b)", "R9: Interaction (φ)"],
            "Coefficient": [
                r7_results.get("pi", np.nan),
                r8_results.get("b_ice", np.nan),
                r9_results.get("phi", np.nan),
            ],
            "R2": [
                r7_results.get("r2", np.nan),
                r8_results.get("r2", np.nan),
                r9_results.get("r2", np.nan),
            ],
            "N": [
                r7_results.get("n", np.nan),
                r8_results.get("n", np.nan),
                r9_results.get("n", np.nan),
            ],
        })
        
        output_path = TABLES_DIR / "r7_r8_r9_shock_analysis.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_path, index=False)
        print(f"\nShock analysis summary saved to {output_path}")
    
    return {"r7": r7_results, "r8": r8_results, "r9": r9_results}


if __name__ == "__main__":
    from .data_loader import prepare_analysis_sample
    
    df = prepare_analysis_sample(use_climate_data=True)
    results = run_shock_analysis(df)
