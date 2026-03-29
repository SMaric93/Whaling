"""
Phase 4: Vessel Mover Power (Authoritative Rerun)

Isolates Managerial Effect from Capital Effect by looking at vessels
that transferred from one agent to another.
Tests if search geometry (mu) shifts as a function of the new agent's authoritative psi_hat,
absorbing hardware (capital) effects via within-vessel fixed effects.

Critical fix: Uses KSS leave-one-out/EB `psi_hat` rather than raw within-sample means
to avoid mechanical data leakage.
"""

import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy import stats
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'post_connectivity'

def load_vessel_mover_data():
    """Load canonical set and authoritative agent types, merge with mu."""
    # 1. Load canonical voyages
    canonical_path = OUTPUT_DIR / 'manifests' / 'canonical_connected_set.parquet'
    types_path = OUTPUT_DIR / 'manifests' / 'type_file_authoritative.parquet'
    
    if not canonical_path.exists() or not types_path.exists():
        raise FileNotFoundError("Run Phase 0 and 1 to generate canonical manifests.")
        
    df = pd.read_parquet(canonical_path)
    df_types = pd.read_parquet(types_path)
    
    # Merge authoritative types
    df = df.merge(df_types[['voyage_id', 'theta_hat', 'psi_hat']], on='voyage_id', how='left')
    
    # 2. Merge with Levy mu (Search geometry)
    levy_path = PROJECT_ROOT / 'output' / 'search_theory' / 'levy_exponents.csv'
    if levy_path.exists():
        levy_df = pd.read_csv(levy_path)
        df = df.merge(
            levy_df[["captain_id", "mean_mu"]].rename(columns={"mean_mu": "levy_mu"}),
            on="captain_id",
            how="left"
        )
    else:
        print("WARNING: levy_exponents.csv not found. Vessel mover test requires mu.")
        df['levy_mu'] = np.nan
        
    # Standardize names and filter missing
    sample = df.dropna(subset=['levy_mu', 'psi_hat', 'vessel_id', 'agent_id']).copy()
    return sample

def run_vessel_mover_power():
    print("="*60)
    print("PHASE 4: VESSEL MOVER TEST (SOFTWARE VS HARDWARE)")
    print("="*60)
    
    df = load_vessel_mover_data()
    print(f"Sample with valid mu and types: {len(df):,} voyages")
    
    if len(df) == 0:
        print("No valid data for vessel mover test.")
        return
        
    # Sort and compute transfers
    df = df.sort_values(["vessel_id", "year_out"])
    
    # Restrict to multi-agent vessels for cleaner identification
    vessel_agent_counts = df.groupby("vessel_id")["agent_id"].nunique()
    multi_agent_vessels = vessel_agent_counts[vessel_agent_counts >= 2].index
    sample_multi = df[df["vessel_id"].isin(multi_agent_vessels)].copy()
    
    print(f"Multi-agent vessel sample: {len(sample_multi):,} voyages ({len(multi_agent_vessels)} vessels)")
    
    if len(sample_multi) < 20:
        print("Insufficient within-vessel variation to proceed.")
        return

    # Demean by vessel (fixed effects transformation)
    sample_fe = sample_multi.copy()
    sample_fe["mu_demeaned"] = sample_fe.groupby("vessel_id")['levy_mu'].transform(lambda x: x - x.mean())
    sample_fe["psi_demeaned"] = sample_fe.groupby("vessel_id")["psi_hat"].transform(lambda x: x - x.mean())
    
    # Filter to non-zero variation
    sample_fe = sample_fe.dropna(subset=["mu_demeaned", "psi_demeaned"])
    sample_fe = sample_fe[sample_fe["psi_demeaned"].abs() > 1e-10]
    
    print(f"Final effective sample after dropping collinearities: {len(sample_fe):,} voyages")
    
    if len(sample_fe) < 10:
        print("Insufficient within-vessel variance in psi_hat.")
        return
        
    y_m2 = sample_fe["mu_demeaned"].values
    X_m2 = sample_fe["psi_demeaned"].values.reshape(-1, 1)
    
    # OLS on demeaned data (equivalent to FE)
    beta_m2 = np.linalg.lstsq(
        np.column_stack([np.ones(len(X_m2)), X_m2]), 
        y_m2, 
        rcond=None
    )[0]
    
    coef_m2 = beta_m2[1]
    
    # Compute SE using within-cluster variance
    y_hat_m2 = X_m2.reshape(-1) * coef_m2
    resid_m2 = y_m2 - y_hat_m2
    n_m2 = len(sample_fe)
    n_vessels_m2 = sample_fe["vessel_id"].nunique()
    dof_m2 = n_m2 - n_vessels_m2 - 1
    
    sigma_sq_m2 = np.sum(resid_m2 ** 2) / max(dof_m2, 1)
    se_coef_m2 = np.sqrt(sigma_sq_m2 / np.sum(X_m2 ** 2))
    t_m2 = coef_m2 / se_coef_m2
    p_m2 = 2 * (1 - stats.t.cdf(np.abs(t_m2), df=max(dof_m2, 1)))
    
    # Within R²
    tss_within = np.sum(y_m2 ** 2)
    rss_within = np.sum(resid_m2 ** 2)
    r2_m2 = 1 - rss_within / tss_within if tss_within > 0 else 0
    
    stars_m2 = "***" if p_m2 < 0.01 else "**" if p_m2 < 0.05 else "*" if p_m2 < 0.1 else ""
    print("\n[WITHIN-VESSEL REGRESSION]: mu_vjt = α_v + β * psi_hat_j + ε")
    print(f"N = {n_m2:,} (vessels = {n_vessels_m2:,})")
    print(f"β(ψ) = {coef_m2:.4f}{stars_m2}")
    print(f"SE = {se_coef_m2:.4f}")
    print(f"t = {t_m2:.2f}, p = {p_m2:.4f}")
    print(f"Within-R² = {r2_m2:.4f}")
    
    # Save output
    out_dir = OUTPUT_DIR / 'tables'
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "Model": "Within-Vessel Authoritative",
        "N": n_m2,
        "N_Vessels": n_vessels_m2,
        "Beta_psi": coef_m2,
        "SE": se_coef_m2,
        "t_stat": t_m2,
        "p_value": p_m2,
        "R2_within": r2_m2,
    }]).to_csv(out_dir / 'vessel_mover_power.csv', index=False)
    
    print(f"\nSUCCESS: Vessel mover tests saved to {out_dir / 'vessel_mover_power.csv'}")

if __name__ == "__main__":
    run_vessel_mover_power()
