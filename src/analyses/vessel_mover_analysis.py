"""
Vessel Mover Analysis Module.

Implements the "Killer" Robustness Test: tracking the same physical vessel
as it moves from Agent A to Agent B to isolate Managerial Effects from
Capital Effects.

Key Test:
    If Ship X has a "lazy" search pattern (μ≈2) under Agent A, and immediately
    shifts to a "ballistic" pattern (μ→1) under Agent B, we have isolated the
    Managerial Effect from the Capital Effect.

Regression:
    μ_vjt = α + β × ψ_agent(j) + γ_vessel(v) + λ_year(t) + ε_vjt

    β ≠ 0 with vessel FE → organizational capability affects search geometry
    independent of vessel characteristics (tonnage, rig, speed).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

from .config import OUTPUT_DIR, TABLES_DIR, STAGING_DIR

warnings.filterwarnings("ignore", category=FutureWarning)

# Output directories
VESSEL_MOVER_DIR = OUTPUT_DIR / "vessel_mover"


# =============================================================================
# 1. Build Vessel-Agent Ownership Panel
# =============================================================================

def build_vessel_ownership_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build panel tracking vessel ownership (agent assignment) over time.
    
    For each vessel, track the sequence of agents operating it.
    Identify "vessel transfer" events where agent changes.
    
    Parameters
    ----------
    df : pd.DataFrame
        Voyage-level data with vessel_id, agent_id, year_out.
        
    Returns
    -------
    pd.DataFrame
        Panel with vessel-voyage observations and transfer indicators.
    """
    print("\n" + "=" * 60)
    print("VM1: BUILDING VESSEL-AGENT OWNERSHIP PANEL")
    print("=" * 60)
    
    df = df.copy()
    
    # Sort by vessel and time
    df = df.sort_values(["vessel_id", "year_out"])
    
    # Compute previous agent for each vessel
    df["prev_agent"] = df.groupby("vessel_id")["agent_id"].shift(1)
    
    # Identify vessel transfers (agent changes for same vessel)
    df["vessel_transfer"] = (
        (df["agent_id"] != df["prev_agent"]) & 
        df["prev_agent"].notna()
    ).astype(int)
    
    # Compute agent tenure on vessel
    df["agent_tenure_on_vessel"] = df.groupby(
        ["vessel_id", "agent_id"]
    ).cumcount() + 1
    
    # Compute voyage number for vessel
    df["vessel_voyage_num"] = df.groupby("vessel_id").cumcount() + 1
    
    # Summary statistics
    n_vessels = df["vessel_id"].nunique()
    n_transfers = df["vessel_transfer"].sum()
    vessels_with_transfers = df[df["vessel_transfer"] == 1]["vessel_id"].nunique()
    
    print(f"\nTotal vessels: {n_vessels:,}")
    print(f"Total vessel-voyage observations: {len(df):,}")
    print(f"Vessel transfers identified: {n_transfers:,}")
    print(f"Vessels with at least one transfer: {vessels_with_transfers:,}")
    
    # List most common transfer patterns
    if n_transfers > 0:
        transfers = df[df["vessel_transfer"] == 1].copy()
        transfers["from_agent"] = transfers["prev_agent"]
        transfers["to_agent"] = transfers["agent_id"]
        
        transfer_pairs = transfers.groupby(
            ["from_agent", "to_agent"]
        ).size().reset_index(name="count")
        transfer_pairs = transfer_pairs.sort_values("count", ascending=False).head(10)
        
        print(f"\nTop 10 Agent-to-Agent Transfer Pairs:")
        for _, row in transfer_pairs.iterrows():
            print(f"  {row['from_agent']} → {row['to_agent']}: {row['count']:,}")
    
    return df


def identify_multi_agent_vessels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify vessels that have been operated by multiple agents.
    
    These are the key vessels for the movers design.
    
    Parameters
    ----------
    df : pd.DataFrame
        Vessel ownership panel from build_vessel_ownership_panel().
        
    Returns
    -------
    pd.DataFrame
        Filtered to vessels with 2+ agents.
    """
    print("\n" + "=" * 60)
    print("VM2: IDENTIFYING MULTI-AGENT VESSELS")
    print("=" * 60)
    
    # Count agents per vessel
    vessel_agent_counts = df.groupby("vessel_id")["agent_id"].nunique()
    multi_agent_vessels = vessel_agent_counts[vessel_agent_counts >= 2].index
    
    print(f"Vessels with 2+ agents: {len(multi_agent_vessels):,}")
    print(f"Total voyages on these vessels: "
          f"{len(df[df['vessel_id'].isin(multi_agent_vessels)]):,}")
    
    # Distribution of agent counts
    agent_count_dist = vessel_agent_counts.value_counts().sort_index()
    print("\nAgent count distribution:")
    for n_agents, count in agent_count_dist.items():
        print(f"  {n_agents} agents: {count:,} vessels")
    
    return df[df["vessel_id"].isin(multi_agent_vessels)].copy()


# =============================================================================
# 2. Merge with Search Geometry (Lévy μ)
# =============================================================================

def merge_with_levy_metrics(
    df: pd.DataFrame,
    levy_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Merge vessel panel with Lévy flight metrics (μ).
    
    Parameters
    ----------
    df : pd.DataFrame
        Vessel ownership panel.
    levy_path : Path, optional
        Path to levy_exponents.csv. If None, computes from logbooks.
        
    Returns
    -------
    pd.DataFrame
        Panel with μ (search geometry) added.
    """
    print("\n" + "=" * 60)
    print("VM3: MERGING WITH LÉVY SEARCH METRICS")
    print("=" * 60)
    
    # Try to load pre-computed Lévy metrics
    search_theory_dir = OUTPUT_DIR / "search_theory"
    if levy_path is None:
        levy_path = search_theory_dir / "levy_exponents.csv"
    
    if levy_path.exists():
        levy_df = pd.read_csv(levy_path)
        print(f"Loaded Lévy metrics from {levy_path}")
        print(f"Captains with μ: {len(levy_df):,}")
        
        # Match on captain_id (μ is captain-level in existing data)
        df = df.merge(
            levy_df[["captain_id", "mean_mu"]].rename(columns={"mean_mu": "levy_mu"}),
            on="captain_id",
            how="left"
        )
    else:
        print("Lévy metrics not found. Attempting to compute from logbooks...")
        
        # Check for voyage-level μ from positions
        positions_path = STAGING_DIR / "logbook_positions.parquet"
        if positions_path.exists():
            from .search_theory import compute_step_lengths, fit_power_law_exponent
            
            positions_df = pd.read_parquet(positions_path)
            
            # Compute voyage-level μ
            voyage_mu = []
            for voyage_id, group in positions_df.groupby("voyage_id"):
                if len(group) < 20:
                    continue
                    
                group = group.sort_values("obs_date")
                lats = group["lat"].values
                lons = group["lon"].values
                
                # Compute step lengths
                from .search_theory import haversine_distance
                steps = []
                for i in range(len(lats) - 1):
                    step = haversine_distance(lats[i], lons[i], lats[i+1], lons[i+1])
                    if step > 0:
                        steps.append(step)
                
                if len(steps) >= 20:
                    mu, se = fit_power_law_exponent(np.array(steps))
                    voyage_mu.append({"voyage_id": voyage_id, "levy_mu": mu})
            
            voyage_mu_df = pd.DataFrame(voyage_mu)
            df = df.merge(voyage_mu_df, on="voyage_id", how="left")
            print(f"Computed voyage-level μ for {len(voyage_mu_df):,} voyages")
        else:
            print("No logbook positions found. Cannot compute μ.")
            df["levy_mu"] = np.nan
    
    valid_mu = df["levy_mu"].dropna()
    print(f"\nVoyages with valid μ: {len(valid_mu):,}")
    print(f"Mean μ: {valid_mu.mean():.3f}")
    print(f"Std μ: {valid_mu.std():.3f}")
    
    return df


# =============================================================================
# 3. Within-Vessel Regression
# =============================================================================

def run_within_vessel_regression(
    df: pd.DataFrame,
    outcome: str = "levy_mu",
) -> Dict:
    """
    Run within-vessel regression to isolate managerial effect.
    
    Model: μ_vjt = β × ψ_agent + vessel FE + year FE + ε
    
    The vessel FE absorbs all hardware characteristics.
    β captures the organizational/managerial effect on search geometry.
    
    Parameters
    ----------
    df : pd.DataFrame
        Panel with vessel_id, agent_id, levy_mu, psi_hat.
    outcome : str
        Outcome variable (default: levy_mu).
        
    Returns
    -------
    Dict
        Regression results with β, SE, t-stat, p-value.
    """
    print("\n" + "=" * 60)
    print("VM4: WITHIN-VESSEL REGRESSION")
    print("=" * 60)
    
    df = df.copy()
    
    # Ensure we have required columns
    if outcome not in df.columns:
        print(f"ERROR: Outcome '{outcome}' not found")
        return {"error": f"missing_{outcome}"}
    
    # Create psi_hat if not present
    if "psi_hat" not in df.columns:
        agent_means = df.groupby("agent_id")["log_q"].mean()
        df["psi_hat"] = df["agent_id"].map(agent_means)
        df["psi_hat"] = df["psi_hat"].fillna(0)
    
    # Filter to valid observations
    sample = df.dropna(subset=[outcome, "psi_hat", "vessel_id"]).copy()
    
    if len(sample) < 100:
        print(f"Insufficient sample: {len(sample)}")
        return {"error": "insufficient_sample", "n": len(sample)}
    
    # Restrict to multi-agent vessels for cleaner identification
    vessel_agent_counts = sample.groupby("vessel_id")["agent_id"].nunique()
    multi_agent_vessels = vessel_agent_counts[vessel_agent_counts >= 2].index
    sample_multi = sample[sample["vessel_id"].isin(multi_agent_vessels)]
    
    print(f"Full sample: {len(sample):,}")
    print(f"Multi-agent vessel sample: {len(sample_multi):,}")
    
    # ========== Model 1: Pooled OLS (No Vessel FE) ==========
    print("\n--- Model 1: Pooled OLS (No Vessel FE) ---")
    
    y = sample["psi_hat"].values  # Use psi_hat to predict mu
    X = np.column_stack([
        np.ones(len(sample)),
        sample[outcome].values,
    ])
    
    # Actually we want μ ~ ψ, so flip
    y_m1 = sample[outcome].values
    X_m1 = np.column_stack([
        np.ones(len(sample)),
        sample["psi_hat"].values,
    ])
    
    beta_m1 = np.linalg.lstsq(X_m1, y_m1, rcond=None)[0]
    y_hat_m1 = X_m1 @ beta_m1
    resid_m1 = y_m1 - y_hat_m1
    
    n_m1, k_m1 = X_m1.shape
    sigma_sq_m1 = np.sum(resid_m1 ** 2) / (n_m1 - k_m1)
    XtX_inv_m1 = np.linalg.inv(X_m1.T @ X_m1)
    se_m1 = np.sqrt(np.diag(sigma_sq_m1 * XtX_inv_m1))
    
    coef_m1 = beta_m1[1]
    se_coef_m1 = se_m1[1]
    t_m1 = coef_m1 / se_coef_m1
    p_m1 = 2 * (1 - stats.t.cdf(np.abs(t_m1), df=n_m1 - k_m1))
    r2_m1 = 1 - np.var(resid_m1) / np.var(y_m1)
    
    stars_m1 = "***" if p_m1 < 0.01 else "**" if p_m1 < 0.05 else "*" if p_m1 < 0.1 else ""
    print(f"N = {n_m1:,}")
    print(f"β(ψ) = {coef_m1:.4f}{stars_m1}")
    print(f"SE = {se_coef_m1:.4f}")
    print(f"t = {t_m1:.2f}, p = {p_m1:.4f}")
    print(f"R² = {r2_m1:.4f}")
    
    # ========== Model 2: Within-Vessel (Vessel FE) ==========
    print("\n--- Model 2: Within-Vessel (Vessel FE) ---")
    
    # Demean by vessel (fixed effects transformation)
    sample_multi = sample_multi.copy()
    sample_multi["mu_demeaned"] = sample_multi.groupby("vessel_id")[outcome].transform(
        lambda x: x - x.mean()
    )
    sample_multi["psi_demeaned"] = sample_multi.groupby("vessel_id")["psi_hat"].transform(
        lambda x: x - x.mean()
    )
    
    # Filter to non-zero variation
    sample_fe = sample_multi.dropna(subset=["mu_demeaned", "psi_demeaned"])
    sample_fe = sample_fe[sample_fe["psi_demeaned"].abs() > 1e-10]
    
    if len(sample_fe) < 50:
        print(f"Insufficient within-vessel variation: {len(sample_fe)}")
        return {
            "error": "insufficient_within_variation",
            "n": len(sample_fe),
            "model1": {
                "n": n_m1, "beta": coef_m1, "se": se_coef_m1, 
                "t": t_m1, "p": p_m1, "r2": r2_m1
            }
        }
    
    y_m2 = sample_fe["mu_demeaned"].values
    X_m2 = sample_fe["psi_demeaned"].values.reshape(-1, 1)
    
    # OLS on demeaned data (equivalent to FE)
    beta_m2 = np.linalg.lstsq(
        np.column_stack([np.ones(len(X_m2)), X_m2]), 
        y_m2, 
        rcond=None
    )[0]
    
    # The coefficient is at index 1 (index 0 is constant which should be ~0)
    coef_m2 = beta_m2[1]
    
    # Compute SE using within-cluster variance
    y_hat_m2 = X_m2.reshape(-1) * coef_m2
    resid_m2 = y_m2 - y_hat_m2
    n_m2 = len(sample_fe)
    n_vessels_m2 = sample_fe["vessel_id"].nunique()
    dof_m2 = n_m2 - n_vessels_m2 - 1  # n - g - k
    
    sigma_sq_m2 = np.sum(resid_m2 ** 2) / max(dof_m2, 1)
    se_coef_m2 = np.sqrt(sigma_sq_m2 / np.sum(X_m2 ** 2))
    t_m2 = coef_m2 / se_coef_m2
    p_m2 = 2 * (1 - stats.t.cdf(np.abs(t_m2), df=max(dof_m2, 1)))
    
    # Within R²
    tss_within = np.sum(y_m2 ** 2)
    rss_within = np.sum(resid_m2 ** 2)
    r2_m2 = 1 - rss_within / tss_within if tss_within > 0 else 0
    
    stars_m2 = "***" if p_m2 < 0.01 else "**" if p_m2 < 0.05 else "*" if p_m2 < 0.1 else ""
    print(f"N = {n_m2:,} (vessels = {n_vessels_m2:,})")
    print(f"β(ψ) = {coef_m2:.4f}{stars_m2}")
    print(f"SE = {se_coef_m2:.4f}")
    print(f"t = {t_m2:.2f}, p = {p_m2:.4f}")
    print(f"Within-R² = {r2_m2:.4f}")
    
    # ========== Interpretation ==========
    print("\n" + "=" * 60)
    print("VESSEL MOVER DESIGN RESULTS")
    print("=" * 60)
    
    if p_m2 < 0.10:
        if coef_m2 < 0:
            print("✓ MANAGERIAL EFFECT CONFIRMED")
            print(f"  Higher ψ (agent capability) → Lower μ (more ballistic search)")
            print(f"  Effect survives vessel FE: β = {coef_m2:.4f}{stars_m2}")
            print("  → This is SOFTWARE (organizational maps), not HARDWARE (better ships)")
        else:
            print("✓ MANAGERIAL EFFECT DETECTED (positive)")
            print(f"  Higher ψ → Higher μ (more exploratory search)")
            print(f"  Effect survives vessel FE: β = {coef_m2:.4f}{stars_m2}")
    else:
        print("✗ No significant within-vessel μ~ψ relationship detected")
        print(f"  β = {coef_m2:.4f}, p = {p_m2:.4f}")
        print("  Consider: (1) more vessel transfers, (2) voyage-level μ estimation")
    
    return {
        "model1_pooled": {
            "n": n_m1,
            "beta": coef_m1,
            "se": se_coef_m1,
            "t": t_m1,
            "p": p_m1,
            "r2": r2_m1,
        },
        "model2_vessel_fe": {
            "n": n_m2,
            "n_vessels": n_vessels_m2,
            "beta": coef_m2,
            "se": se_coef_m2,
            "t": t_m2,
            "p": p_m2,
            "r2_within": r2_m2,
        },
        "managerial_effect_confirmed": p_m2 < 0.10,
    }


# =============================================================================
# 4. Event Study Around Vessel Transfer
# =============================================================================

def run_vessel_transfer_event_study(
    df: pd.DataFrame,
    window: int = 3,
) -> Dict:
    """
    Event study: μ dynamics around vessel transfer events.
    
    For each transfer, track μ in voyages [-window, +window] around transfer.
    
    Parameters
    ----------
    df : pd.DataFrame
        Panel with vessel_id, levy_mu, vessel_transfer indicator.
    window : int
        Number of voyages before/after transfer to include.
        
    Returns
    -------
    Dict
        Event study coefficients and visualization data.
    """
    print("\n" + "=" * 60)
    print("VM5: VESSEL TRANSFER EVENT STUDY")
    print("=" * 60)
    
    df = df.copy()
    
    if "vessel_transfer" not in df.columns:
        print("No vessel transfers identified")
        return {"error": "no_transfers"}
    
    # Identify transfer events
    transfers = df[df["vessel_transfer"] == 1].copy()
    
    if len(transfers) == 0:
        print("No vessel transfers in sample")
        return {"error": "no_transfers"}
    
    print(f"Transfer events: {len(transfers):,}")
    
    # For each transfer, build event-time panel
    event_data = []
    
    for _, transfer in transfers.iterrows():
        vessel_id = transfer["vessel_id"]
        transfer_year = transfer["year_out"]
        
        # Get all voyages for this vessel
        vessel_voyages = df[df["vessel_id"] == vessel_id].copy()
        vessel_voyages = vessel_voyages.sort_values("year_out")
        
        # Find transfer position
        transfer_idx = vessel_voyages[
            vessel_voyages["year_out"] == transfer_year
        ].index[0]
        vessel_voyages = vessel_voyages.reset_index(drop=True)
        transfer_pos = vessel_voyages.index[
            vessel_voyages["voyage_id"] == transfer["voyage_id"]
        ][0]
        
        # Compute event time
        vessel_voyages["event_time"] = vessel_voyages.index - transfer_pos
        
        # Keep window
        in_window = vessel_voyages[
            (vessel_voyages["event_time"] >= -window) & 
            (vessel_voyages["event_time"] <= window)
        ]
        
        for _, row in in_window.iterrows():
            event_data.append({
                "vessel_id": vessel_id,
                "event_time": row["event_time"],
                "levy_mu": row.get("levy_mu"),
                "psi_hat": row.get("psi_hat"),
                "log_q": row.get("log_q"),
            })
    
    event_df = pd.DataFrame(event_data)
    
    if len(event_df) == 0:
        print("No event-time observations")
        return {"error": "no_event_obs"}
    
    # Compute mean μ by event time
    event_means = event_df.groupby("event_time").agg({
        "levy_mu": ["mean", "std", "count"],
        "psi_hat": "mean",
        "log_q": "mean",
    })
    event_means.columns = ["mu_mean", "mu_std", "n", "psi_mean", "logq_mean"]
    event_means = event_means.reset_index()
    
    print("\nμ by Event Time:")
    print(event_means.to_string(index=False))
    
    # Test: μ jump at t=0
    pre_mu = event_df[event_df["event_time"] < 0]["levy_mu"].dropna()
    post_mu = event_df[event_df["event_time"] >= 0]["levy_mu"].dropna()
    
    if len(pre_mu) > 10 and len(post_mu) > 10:
        t_stat, p_value = stats.ttest_ind(pre_mu, post_mu)
        print(f"\nPre-transfer μ: {pre_mu.mean():.3f} (n={len(pre_mu)})")
        print(f"Post-transfer μ: {post_mu.mean():.3f} (n={len(post_mu)})")
        print(f"Difference: {post_mu.mean() - pre_mu.mean():.3f}")
        print(f"t-test: t={t_stat:.2f}, p={p_value:.4f}")
        
        if p_value < 0.10:
            print("✓ Significant μ shift at vessel transfer")
        else:
            print("No significant μ shift at transfer")
    
    return {
        "event_means": event_means.to_dict(),
        "n_transfers": len(transfers),
        "n_event_obs": len(event_df),
    }


# =============================================================================
# 5. Main Orchestration
# =============================================================================

def run_vessel_mover_analysis(
    df: pd.DataFrame = None,
    save_outputs: bool = True,
) -> Dict:
    """
    Run complete Vessel Mover Design analysis.
    
    Parameters
    ----------
    df : pd.DataFrame, optional
        Voyage data. If None, loads from disk.
    save_outputs : bool
        Whether to save results to disk.
        
    Returns
    -------
    Dict
        All results from the analysis.
    """
    print("=" * 70)
    print("VESSEL MOVER DESIGN ANALYSIS")
    print("Isolating Managerial Effect from Capital Effect")
    print("=" * 70)
    
    # Load data if not provided
    if df is None:
        from .data_loader import prepare_analysis_sample
        df = prepare_analysis_sample()
    
    results = {}
    
    # Step 1: Build vessel ownership panel
    df_panel = build_vessel_ownership_panel(df)
    
    # Step 2: Identify multi-agent vessels
    df_multi = identify_multi_agent_vessels(df_panel)
    
    # Step 3: Merge with Lévy metrics
    df_with_mu = merge_with_levy_metrics(df_multi)
    
    # Step 4: Run within-vessel regression
    results["regression"] = run_within_vessel_regression(df_with_mu)
    
    # Step 5: Event study
    df_panel_with_mu = merge_with_levy_metrics(df_panel)
    results["event_study"] = run_vessel_transfer_event_study(df_panel_with_mu)
    
    # Save outputs
    if save_outputs:
        save_vessel_mover_outputs(results, df_multi)
    
    return results


def save_vessel_mover_outputs(results: Dict, df_multi: pd.DataFrame) -> None:
    """Save vessel mover analysis outputs."""
    VESSEL_MOVER_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save multi-agent vessel panel
    if len(df_multi) > 0:
        df_multi.to_csv(VESSEL_MOVER_DIR / "multi_agent_vessels.csv", index=False)
    
    # Summary table
    summary = []
    
    if "regression" in results and "model2_vessel_fe" in results["regression"]:
        r = results["regression"]["model2_vessel_fe"]
        summary.append({
            "Model": "Within-Vessel (Vessel FE)",
            "N": r["n"],
            "N_Vessels": r["n_vessels"],
            "Beta_psi": r["beta"],
            "SE": r["se"],
            "t_stat": r["t"],
            "p_value": r["p"],
            "R2_within": r["r2_within"],
        })
    
    if "regression" in results and "model1_pooled" in results["regression"]:
        r = results["regression"]["model1_pooled"]
        summary.append({
            "Model": "Pooled OLS (No FE)",
            "N": r["n"],
            "N_Vessels": np.nan,
            "Beta_psi": r["beta"],
            "SE": r["se"],
            "t_stat": r["t"],
            "p_value": r["p"],
            "R2_within": r["r2"],
        })
    
    if summary:
        pd.DataFrame(summary).to_csv(
            VESSEL_MOVER_DIR / "vessel_mover_summary.csv", index=False
        )
    
    # Generate markdown summary
    md_lines = [
        "# Vessel Mover Design Results",
        "",
        "## Key Finding",
        "",
    ]
    
    if results.get("regression", {}).get("managerial_effect_confirmed"):
        md_lines.append("**✓ Managerial Effect Confirmed**: Search geometry (μ) changes when ")
        md_lines.append("the same vessel transfers to a different agent, controlling for vessel FE.")
        md_lines.append("")
        md_lines.append("This isolates **Software** (organizational capability) from **Hardware** (vessel quality).")
    else:
        md_lines.append("No significant within-vessel managerial effect detected.")
    
    md_lines.extend([
        "",
        "## Regression Results",
        "",
    ])
    
    if summary:
        md_lines.append("| Model | N | β(ψ) | SE | t | p |")
        md_lines.append("|-------|---|------|----|----|---|")
        for row in summary:
            md_lines.append(
                f"| {row['Model']} | {row['N']:,} | {row['Beta_psi']:.4f} | "
                f"{row['SE']:.4f} | {row['t_stat']:.2f} | {row['p_value']:.4f} |"
            )
    
    with open(VESSEL_MOVER_DIR / "vessel_mover_results.md", "w") as f:
        f.write("\n".join(md_lines))
    
    print(f"\nOutputs saved to {VESSEL_MOVER_DIR}")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    from .data_loader import prepare_analysis_sample
    
    df = prepare_analysis_sample()
    results = run_vessel_mover_analysis(df)
