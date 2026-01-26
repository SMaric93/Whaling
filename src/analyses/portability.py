"""
Portability validation for captain and agent fixed effects (R2, R4).

Implements out-of-sample prediction tests to validate that estimated
fixed effects represent portable skill/capability rather than spurious
matching or measurement error.
"""

from typing import Dict, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

from .config import DEFAULT_SAMPLE
from .baseline_production import estimate_r1, build_sparse_design_matrix
from .data_loader import split_train_test

warnings.filterwarnings("ignore", category=FutureWarning)


def estimate_train_period_effects(
    train_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Estimate captain and agent effects on training period only.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training period voyages.
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, Dict]
        (captain_fe, agent_fe, full_results)
    """
    print("\n" + "=" * 60)
    print("ESTIMATING TRAINING PERIOD EFFECTS")
    print("=" * 60)
    
    results = estimate_r1(train_df, use_loo_sample=True)
    
    return results["captain_fe"], results["agent_fe"], results


def run_r2_captain_portability(
    df: pd.DataFrame,
    cutoff_year: Optional[int] = None,
) -> Dict:
    """
    R2: OOS prediction of output using pre-period captain effects.
    
    logQ_v (test) = b · α̂_train[c(v)] + δ_{vessel×period} + θ_{route×time} + Xβ + u_v
    
    Parameters
    ----------
    df : pd.DataFrame
        Full voyage data.
    cutoff_year : int, optional
        Year to split on.
        
    Returns
    -------
    Dict
        R2 results including portability coefficient and diagnostics.
    """
    if cutoff_year is None:
        cutoff_year = DEFAULT_SAMPLE.oos_cutoff_year
        
    print("\n" + "=" * 60)
    print(f"R2: CAPTAIN PORTABILITY (OOS Prediction)")
    print(f"Train/test split: {cutoff_year}")
    print("=" * 60)
    
    # Split data
    train_df, test_df = split_train_test(df, cutoff_year)
    
    # Estimate effects on training period
    captain_fe_train, agent_fe_train, train_results = estimate_train_period_effects(train_df)
    
    # Rename for clarity
    captain_fe_train = captain_fe_train.rename(columns={"alpha_hat": "alpha_hat_train"})
    
    # Merge training effects to test data
    test_df = test_df.merge(captain_fe_train, on="captain_id", how="left")
    
    # Only keep test captains that were in training
    test_with_alpha = test_df[test_df["alpha_hat_train"].notna()].copy()
    
    print(f"\nTest sample:")
    print(f"  Total test voyages: {len(test_df):,}")
    print(f"  With training α̂: {len(test_with_alpha):,} ({100*len(test_with_alpha)/len(test_df):.1f}%)")
    print(f"  Unique captains: {test_with_alpha['captain_id'].nunique():,}")
    
    # =========================================================================
    # Full test sample regression
    # =========================================================================
    print("\n--- Full Test Sample ---")
    
    # Build design matrix for test period (without captain FE, add α̂_train as control)
    n = len(test_with_alpha)
    y = test_with_alpha["log_q"].values
    
    # Vessel×period and route×time FEs
    matrices = []
    
    # Vessel×period (drop first)
    if "vessel_period" in test_with_alpha.columns:
        vp_ids = test_with_alpha["vessel_period"].unique()
        vp_map = {v: i for i, v in enumerate(vp_ids)}
        vp_idx = test_with_alpha["vessel_period"].map(vp_map).values
        X_vp = sp.csr_matrix(
            (np.ones(n), (np.arange(n), vp_idx)),
            shape=(n, len(vp_ids))
        )[:, 1:]
        matrices.append(X_vp)
    
    # Route×time (drop first)
    if "route_time" in test_with_alpha.columns:
        rt_ids = test_with_alpha["route_time"].unique()
        rt_map = {r: i for i, r in enumerate(rt_ids)}
        rt_idx = test_with_alpha["route_time"].map(rt_map).values
        X_rt = sp.csr_matrix(
            (np.ones(n), (np.arange(n), rt_idx)),
            shape=(n, len(rt_ids))
        )[:, 1:]
        matrices.append(X_rt)
    
    # Controls: α̂_train, log_duration, log_tonnage
    controls = np.column_stack([
        test_with_alpha["alpha_hat_train"].values,
        test_with_alpha["log_duration"].values,
        test_with_alpha["log_tonnage"].values,
    ])
    matrices.append(sp.csr_matrix(controls))
    
    # Intercept
    matrices.append(sp.csr_matrix(np.ones((n, 1))))
    
    X = sp.hstack(matrices)
    
    # Solve
    result = lsqr(X, y, iter_lim=10000, atol=1e-10, btol=1e-10)
    beta = result[0]
    
    # α̂_train coefficient is at the start of controls block
    n_vp = len(vp_ids) - 1 if "vessel_period" in test_with_alpha.columns else 0
    n_rt = len(rt_ids) - 1 if "route_time" in test_with_alpha.columns else 0
    b_alpha_full = beta[n_vp + n_rt]  # First control is alpha_hat_train
    
    # R² and residuals
    y_hat = X @ beta
    residuals = y - y_hat
    r2_full = 1 - np.var(residuals) / np.var(y)
    
    print(f"  b (α̂_train coefficient): {b_alpha_full:.4f}")
    print(f"  R²: {r2_full:.4f}")
    
    # =========================================================================
    # Switch-only sample
    # =========================================================================
    print("\n--- Switch-Only Sample ---")
    
    # Voyages where captain switched agent, vessel, or route
    switch_mask = test_with_alpha["any_switch"] == 1
    test_switchers = test_with_alpha[switch_mask].copy()
    
    if len(test_switchers) > 50:
        n_sw = len(test_switchers)
        y_sw = test_switchers["log_q"].values
        
        # Simple regression: logQ ~ α̂_train + controls
        X_sw = np.column_stack([
            np.ones(n_sw),
            test_switchers["alpha_hat_train"].values,
            test_switchers["log_duration"].values,
            test_switchers["log_tonnage"].values,
        ])
        
        beta_sw = np.linalg.lstsq(X_sw, y_sw, rcond=None)[0]
        b_alpha_switch = beta_sw[1]
        
        y_hat_sw = X_sw @ beta_sw
        r2_switch = 1 - np.var(y_sw - y_hat_sw) / np.var(y_sw)
        
        print(f"  Switch voyages: {n_sw:,}")
        print(f"  b (α̂_train coefficient): {b_alpha_switch:.4f}")
        print(f"  R²: {r2_switch:.4f}")
    else:
        b_alpha_switch = np.nan
        r2_switch = np.nan
        print(f"  Insufficient switch voyages: {len(test_switchers)}")
    
    # =========================================================================
    # Rank correlation validation
    # =========================================================================
    print("\n--- Rank Correlation Validation ---")
    
    # Average realized logQ by captain in test period
    captain_realized = test_with_alpha.groupby("captain_id").agg({
        "log_q": "mean",
        "alpha_hat_train": "first",
    }).reset_index()
    
    # Spearman correlation
    spearman_r, spearman_p = stats.spearmanr(
        captain_realized["alpha_hat_train"],
        captain_realized["log_q"]
    )
    
    # Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(
        captain_realized["alpha_hat_train"],
        captain_realized["log_q"]
    )
    
    print(f"  Captain-level correlations:")
    print(f"    Spearman ρ: {spearman_r:.4f} (p={spearman_p:.4f})")
    print(f"    Pearson r: {pearson_r:.4f} (p={pearson_p:.4f})")
    
    results = {
        "cutoff_year": cutoff_year,
        "n_train": len(train_df),
        "n_test_total": len(test_df),
        "n_test_with_alpha": len(test_with_alpha),
        "n_test_switchers": len(test_switchers) if "test_switchers" in dir() else 0,
        "b_alpha_full": b_alpha_full,
        "r2_full": r2_full,
        "b_alpha_switch": b_alpha_switch,
        "r2_switch": r2_switch,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "captain_fe_train": captain_fe_train,
        "captain_realized": captain_realized,
        "test_data": test_with_alpha,
        "train_results": train_results,
    }
    
    return results


def run_r4_agent_portability(
    df: pd.DataFrame,
    cutoff_year: Optional[int] = None,
) -> Dict:
    """
    R4: OOS prediction using pre-period agent effects.
    
    logQ_v (test) = b · γ̂_train[a(v)] + α_c + δ_{vessel×period} + θ_{route×time} + u_v
    
    Focus on voyages with captain turnover to avoid confounding.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full voyage data.
    cutoff_year : int, optional
        Year to split on.
        
    Returns
    -------
    Dict
        R4 results.
    """
    if cutoff_year is None:
        cutoff_year = DEFAULT_SAMPLE.oos_cutoff_year
        
    print("\n" + "=" * 60)
    print(f"R4: AGENT CAPABILITY PERSISTENCE (OOS Prediction)")
    print(f"Train/test split: {cutoff_year}")
    print("=" * 60)
    
    # Split data
    train_df, test_df = split_train_test(df, cutoff_year)
    
    # Estimate effects on training period
    captain_fe_train, agent_fe_train, train_results = estimate_train_period_effects(train_df)
    
    # Rename for clarity
    agent_fe_train = agent_fe_train.rename(columns={"gamma_hat": "gamma_hat_train"})
    
    # Merge training agent effects to test data
    test_df = test_df.merge(agent_fe_train, on="agent_id", how="left")
    
    # Only keep test voyages with agents from training
    test_with_gamma = test_df[test_df["gamma_hat_train"].notna()].copy()
    
    print(f"\nTest sample:")
    print(f"  Total test voyages: {len(test_df):,}")
    print(f"  With training γ̂: {len(test_with_gamma):,} ({100*len(test_with_gamma)/len(test_df):.1f}%)")
    print(f"  Unique agents: {test_with_gamma['agent_id'].nunique():,}")
    
    # Focus on voyages with NEW captains (not in training)
    train_captains = set(train_df["captain_id"])
    new_captain_mask = ~test_with_gamma["captain_id"].isin(train_captains)
    test_new_captains = test_with_gamma[new_captain_mask].copy()
    
    print(f"\nNew captain subsample (captain turnover):")
    print(f"  Voyages with new captains: {len(test_new_captains):,}")
    print(f"  New captains: {test_new_captains['captain_id'].nunique():,}")
    
    # Simple regression on new captain sample
    if len(test_new_captains) > 50:
        n = len(test_new_captains)
        y = test_new_captains["log_q"].values
        
        X = np.column_stack([
            np.ones(n),
            test_new_captains["gamma_hat_train"].values,
            test_new_captains["log_duration"].values,
            test_new_captains["log_tonnage"].values,
        ])
        
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        b_gamma = beta[1]
        
        y_hat = X @ beta
        r2 = 1 - np.var(y - y_hat) / np.var(y)
        
        print(f"\n  b (γ̂_train coefficient): {b_gamma:.4f}")
        print(f"  R²: {r2:.4f}")
        
        # Agent-level correlation
        agent_realized = test_new_captains.groupby("agent_id").agg({
            "log_q": "mean",
            "gamma_hat_train": "first",
        }).reset_index()
        
        spearman_r, spearman_p = stats.spearmanr(
            agent_realized["gamma_hat_train"],
            agent_realized["log_q"]
        )
        
        print(f"\n  Agent-level Spearman ρ: {spearman_r:.4f} (p={spearman_p:.4f})")
        
    else:
        b_gamma = np.nan
        r2 = np.nan
        spearman_r = np.nan
        spearman_p = np.nan
        print(f"  Insufficient new captain voyages for analysis")
    
    results = {
        "cutoff_year": cutoff_year,
        "n_test_with_gamma": len(test_with_gamma),
        "n_new_captain_voyages": len(test_new_captains),
        "b_gamma": b_gamma,
        "r2": r2,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "agent_fe_train": agent_fe_train,
    }
    
    return results


def create_portability_figure(
    r2_results: Dict,
    output_path: Optional[str] = None,
) -> None:
    """
    Create portability visualization (OOS prediction plot).
    
    Parameters
    ----------
    r2_results : Dict
        Results from run_r2_captain_portability.
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
        output_path = FIGURES_DIR / "r2_captain_portability_oos.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    captain_data = r2_results["captain_realized"]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Scatter plot
    ax1 = axes[0]
    ax1.scatter(
        captain_data["alpha_hat_train"],
        captain_data["log_q"],
        alpha=0.5,
        s=20,
        color="steelblue",
    )
    
    # Add regression line
    x = captain_data["alpha_hat_train"]
    y = captain_data["log_q"]
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax1.plot(x_line, p(x_line), "r--", linewidth=2, label=f"Slope: {z[0]:.3f}")
    
    ax1.set_xlabel("Training Period α̂ (Captain Effect)", fontsize=11)
    ax1.set_ylabel("Test Period Mean log(Q)", fontsize=11)
    ax1.set_title("R2: Captain Portability (OOS Prediction)", fontsize=12, fontweight="bold")
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Right: Bin scatter
    ax2 = axes[1]
    n_bins = 10
    captain_data["alpha_bin"] = pd.qcut(captain_data["alpha_hat_train"], q=n_bins, labels=False)
    bin_means = captain_data.groupby("alpha_bin").agg({
        "alpha_hat_train": "mean",
        "log_q": "mean",
    })
    
    ax2.scatter(bin_means["alpha_hat_train"], bin_means["log_q"], s=100, color="darkblue")
    ax2.plot(bin_means["alpha_hat_train"], bin_means["log_q"], "b-", linewidth=2)
    
    ax2.set_xlabel("Training Period α̂ (Decile Mean)", fontsize=11)
    ax2.set_ylabel("Test Period Mean log(Q)", fontsize=11)
    ax2.set_title("Binned Scatter (10 Deciles)", fontsize=12, fontweight="bold")
    ax2.grid(alpha=0.3)
    
    # Add statistics annotation
    stats_text = (
        f"Spearman ρ = {r2_results['spearman_r']:.3f}\n"
        f"b (full sample) = {r2_results['b_alpha_full']:.3f}\n"
        f"n = {r2_results['n_test_with_alpha']:,}"
    )
    ax2.annotate(
        stats_text,
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"\nPortability figure saved to {output_path}")


def run_portability_analysis(
    df: pd.DataFrame,
    save_outputs: bool = True,
) -> Dict:
    """
    Run full portability analysis (R2 and R4).
    
    Parameters
    ----------
    df : pd.DataFrame
        Full voyage data.
    save_outputs : bool
        Whether to save outputs.
        
    Returns
    -------
    Dict
        Combined R2 and R4 results.
    """
    from .config import TABLES_DIR
    from pathlib import Path
    
    # R2: Captain portability
    r2_results = run_r2_captain_portability(df)
    
    # R4: Agent persistence
    r4_results = run_r4_agent_portability(df)
    
    if save_outputs:
        # Create portability figure
        create_portability_figure(r2_results)
        
        # Save summary table
        summary = pd.DataFrame({
            "Specification": ["R2: Captain Portability", "R2: Switch-Only", "R4: Agent Persistence"],
            "Coefficient": [
                r2_results["b_alpha_full"],
                r2_results["b_alpha_switch"],
                r4_results["b_gamma"],
            ],
            "R2": [
                r2_results["r2_full"],
                r2_results["r2_switch"],
                r4_results["r2"],
            ],
            "Spearman_rho": [
                r2_results["spearman_r"],
                np.nan,
                r4_results["spearman_r"],
            ],
            "N": [
                r2_results["n_test_with_alpha"],
                r2_results.get("n_test_switchers", np.nan),
                r4_results["n_new_captain_voyages"],
            ],
        })
        
        output_path = TABLES_DIR / "r2_r4_portability_summary.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_path, index=False)
        print(f"\nPortability summary saved to {output_path}")
    
    return {"r2": r2_results, "r4": r4_results}


if __name__ == "__main__":
    from .data_loader import prepare_analysis_sample
    
    df = prepare_analysis_sample()
    results = run_portability_analysis(df)
