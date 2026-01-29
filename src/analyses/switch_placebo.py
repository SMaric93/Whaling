"""
Switch Placebo Test (C1.3).

Permutation test for validating that the observed agent effect κ
is not driven by spurious timing of switches.

Procedure:
1. For each captain, record their switch years
2. Randomly reassign switch timing within career (preserving N switches)
3. Re-estimate movers design regression on placebo switches
4. Build null distribution from 500 iterations
5. Compare observed κ to permutation distribution

Pass Criterion:
    Observed κ outside 95% CI of null distribution
"""

from typing import Dict, List, Optional
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

from .config import OUTPUT_DIR, FIGURES_DIR

warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# Movers Design Estimation (Simplified Table 3)
# =============================================================================

def estimate_movers_effect(df: pd.DataFrame) -> Dict:
    """
    Estimate agent effect κ from movers design.
    
    log_q = α_c + κ Δψ + controls + ε
    
    Where Δψ = change in agent FE upon switching.
    
    Parameters
    ----------
    df : pd.DataFrame
        Voyage data with agent assignments and switch indicators.
        
    Returns
    -------
    Dict
        Estimated κ and standard error.
    """
    df = df.copy()
    
    # Need agent FE estimates
    if "psi_hat" not in df.columns:
        # Simple estimation: mean productivity by agent
        agent_means = df.groupby("agent_id")["log_q"].mean()
        df["psi_hat"] = df["agent_id"].map(agent_means)
        df["psi_hat"] = df["psi_hat"].fillna(df["log_q"].mean())
    
    # Compute Δψ for switch voyages
    df = df.sort_values(["captain_id", "year_out"])
    df["prev_psi"] = df.groupby("captain_id")["psi_hat"].shift(1)
    df["delta_psi"] = df["psi_hat"] - df["prev_psi"]
    
    # Focus on switch voyages
    if "switch_agent" not in df.columns:
        df["prev_agent"] = df.groupby("captain_id")["agent_id"].shift(1)
        df["switch_agent"] = (df["agent_id"] != df["prev_agent"]).astype(float)
        first_voyage = df["prev_agent"].isna()
        df.loc[first_voyage, "switch_agent"] = np.nan
    
    # Sample: voyages with valid switch indicator and delta_psi
    sample = df.dropna(subset=["switch_agent", "delta_psi", "log_q"]).copy()
    
    if len(sample) < 50:
        return {"kappa": np.nan, "se": np.nan, "n": len(sample)}
    
    # Regression: log_q ~ delta_psi + captain FE
    n = len(sample)
    y = sample["log_q"].values
    
    # Build X: constant + delta_psi + captain FEs
    captain_ids = sample["captain_id"].unique()
    captain_map = {c: i for i, c in enumerate(captain_ids)}
    captain_idx = sample["captain_id"].map(captain_map).values
    
    X_captain = sp.csr_matrix(
        (np.ones(n), (np.arange(n), captain_idx)),
        shape=(n, len(captain_ids))
    )
    X_delta_psi = sample["delta_psi"].values.reshape(-1, 1)
    X = sp.hstack([sp.csr_matrix(X_delta_psi), X_captain])
    
    # Solve
    result = lsqr(X, y, iter_lim=5000, atol=1e-8, btol=1e-8)
    beta = result[0]
    
    kappa = beta[0]  # First coefficient is delta_psi
    
    # Approximate SE
    y_hat = X @ beta
    resid = y - y_hat
    sigma2 = np.sum(resid ** 2) / (n - X.shape[1])
    XtX_diag_inv = 1.0 / np.array(X.T.dot(X).diagonal()).flatten()
    se = np.sqrt(sigma2 * XtX_diag_inv[0])
    
    return {"kappa": kappa, "se": se, "n": n}


# =============================================================================
# Permutation Functions
# =============================================================================

def permute_switch_timing(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    Randomly reassign switch timing within each captain's career.
    
    Preserves the total number of switches per captain but changes
    when they occurred.
    
    Parameters
    ----------
    df : pd.DataFrame
        Voyage data with switch_agent indicator.
    seed : int
        Random seed for reproducibility.
        
    Returns
    -------
    pd.DataFrame
        Data with permuted switch indicators.
    """
    np.random.seed(seed)
    
    df = df.copy()
    df = df.sort_values(["captain_id", "year_out"])
    
    # For each captain, count switches and randomly reassign
    new_switches = []
    
    for captain_id, group in df.groupby("captain_id"):
        group = group.copy()
        
        # Can only have switches after first voyage
        if len(group) < 2:
            new_switches.extend([np.nan] * len(group))
            continue
        
        # Count actual switches (excluding first voyage)
        switches = group["switch_agent"].dropna()
        n_switches = int(switches.sum()) if len(switches) > 0 else 0
        
        # Create placebo switches: randomly assign n_switches to positions 1..N-1
        n_valid_positions = len(group) - 1  # Can switch on voyages 2, 3, ...
        
        if n_switches > n_valid_positions:
            n_switches = n_valid_positions
        
        # First voyage: always NaN
        placebo = [np.nan]
        
        # Random positions for switches
        switch_positions = set(np.random.choice(n_valid_positions, size=n_switches, replace=False))
        
        for i in range(n_valid_positions):
            if i in switch_positions:
                placebo.append(1.0)
            else:
                placebo.append(0.0)
        
        new_switches.extend(placebo)
    
    df["switch_agent"] = new_switches
    
    return df


def run_placebo_switch_test(
    df: pd.DataFrame,
    n_iterations: int = 500,
    save_outputs: bool = True,
) -> Dict:
    """
    C1.3: Permutation test for placebo switches.
    
    Parameters
    ----------
    df : pd.DataFrame
        Voyage data.
    n_iterations : int
        Number of permutation iterations.
    save_outputs : bool
        Whether to save figure and results.
        
    Returns
    -------
    Dict
        Results including observed κ, null distribution, and pass/fail.
    """
    print("\n" + "=" * 60)
    print("C1.3: PLACEBO SWITCH PERMUTATION TEST")
    print("=" * 60)
    
    # Step 1: Estimate observed κ using real switches
    print("\nStep 1: Estimating observed κ with real switches...")
    observed = estimate_movers_effect(df)
    observed_kappa = observed["kappa"]
    
    if np.isnan(observed_kappa):
        print("Could not estimate observed κ")
        return {"error": "estimation_failed"}
    
    print(f"Observed κ = {observed_kappa:.4f} (SE = {observed['se']:.4f})")
    
    # Step 2: Build null distribution
    print(f"\nStep 2: Running {n_iterations} permutations...")
    
    placebo_kappas = []
    for i in range(n_iterations):
        if (i + 1) % 100 == 0:
            print(f"  Iteration {i + 1}/{n_iterations}")
        
        df_placebo = permute_switch_timing(df, seed=i)
        result = estimate_movers_effect(df_placebo)
        
        if not np.isnan(result["kappa"]):
            placebo_kappas.append(result["kappa"])
    
    placebo_kappas = np.array(placebo_kappas)
    
    print(f"\nCompleted {len(placebo_kappas)} valid permutations")
    
    # Step 3: Compare observed to null distribution
    print("\nStep 3: Computing p-value...")
    
    # Two-tailed p-value
    p_value = np.mean(np.abs(placebo_kappas) >= np.abs(observed_kappa))
    
    # 95% CI
    ci_lower, ci_upper = np.percentile(placebo_kappas, [2.5, 97.5])
    
    # Pass: observed outside 95% CI
    passed = (observed_kappa < ci_lower) or (observed_kappa > ci_upper)
    
    print(f"\n--- Results ---")
    print(f"Observed κ: {observed_kappa:.4f}")
    print(f"Null distribution: mean = {placebo_kappas.mean():.4f}, std = {placebo_kappas.std():.4f}")
    print(f"95% CI of null: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"Permutation p-value: {p_value:.4f}")
    
    print(f"\n--- RESULT ---")
    if passed:
        print("✓ PASS: Observed κ outside 95% CI of placebo distribution")
        print("  → Agent effect is NOT spuriously driven by switch timing")
    else:
        print("✗ FAIL: Observed κ within 95% CI of placebo distribution")
        print("  → Agent effect may be artifact of switch timing")
    
    results = {
        "observed_kappa": observed_kappa,
        "observed_se": observed["se"],
        "placebo_mean": placebo_kappas.mean(),
        "placebo_std": placebo_kappas.std(),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "passed": passed,
        "n_permutations": len(placebo_kappas),
        "placebo_distribution": placebo_kappas,
    }
    
    # Save figure
    if save_outputs:
        create_placebo_figure(results)
    
    return results


def create_placebo_figure(results: Dict) -> None:
    """Create histogram of placebo distribution with observed value marked."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping figure")
        return
    
    from pathlib import Path
    
    output_path = FIGURES_DIR / "c1_3_placebo_switch_distribution.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram of placebo distribution
    ax.hist(
        results["placebo_distribution"],
        bins=50,
        alpha=0.7,
        color="steelblue",
        edgecolor="white",
        label="Placebo Distribution"
    )
    
    # Observed value line
    ax.axvline(
        results["observed_kappa"],
        color="red",
        linewidth=2,
        linestyle="--",
        label=f"Observed κ = {results['observed_kappa']:.4f}"
    )
    
    # 95% CI lines
    ax.axvline(
        results["ci_lower"],
        color="gray",
        linewidth=1,
        linestyle=":",
        label=f"95% CI: [{results['ci_lower']:.3f}, {results['ci_upper']:.3f}]"
    )
    ax.axvline(results["ci_upper"], color="gray", linewidth=1, linestyle=":")
    
    ax.set_xlabel("Agent Effect (κ)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        f"C1.3: Placebo Switch Permutation Test (n={results['n_permutations']})",
        fontsize=14,
        fontweight="bold"
    )
    
    # Result annotation
    status = "✓ PASS" if results["passed"] else "✗ FAIL"
    ax.annotate(
        f"p-value = {results['p_value']:.4f}\n{status}",
        xy=(0.95, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"\nFigure saved to {output_path}")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    from .data_loader import prepare_analysis_sample
    
    df = prepare_analysis_sample()
    results = run_placebo_switch_test(df, n_iterations=100)  # Use fewer for testing
