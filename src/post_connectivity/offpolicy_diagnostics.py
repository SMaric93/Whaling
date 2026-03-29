"""
Phase 4: Off-Policy Diagnostics (Authoritative)

Computes diagnostic metrics for Inverse Probability Weighting (IPW) and 
Doubly Robust (DR) estimation. 
Checks:
1. Effective Sample Size (ESS)
2. Propensity Score Overlap
3. Covariate Balance (Standardized Mean Differences)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'post_connectivity'

def load_canonical_treatment_data():
    canonical_path = OUTPUT_DIR / 'manifests' / 'canonical_connected_set.parquet'
    types_path = OUTPUT_DIR / 'manifests' / 'type_file_authoritative.parquet'
    
    if not canonical_path.exists() or not types_path.exists():
        raise FileNotFoundError("Run Phase 0 and 1 to generate canonical manifests.")
        
    df = pd.read_parquet(canonical_path)
    df_types = pd.read_parquet(types_path)
    
    df = df.merge(df_types[['voyage_id', 'theta_hat', 'psi_hat']], on='voyage_id', how='left')
    
    # Let's say the 'treatment' is picking a high-psi agent
    psi_threshold = df['psi_hat'].quantile(0.75)
    df['is_high_psi'] = (df['psi_hat'] >= psi_threshold).astype(int)
    
    # Feature set for propensity:
    # Just basic observable covariates present at the time a captain matches
    covariates = ["year_out", "tonnage", "theta_hat"]
    sample = df.dropna(subset=['is_high_psi'] + covariates).copy()
    
    return sample, covariates

def smd(x_treat, x_control):
    """Standardized Mean Difference"""
    mean_diff = np.mean(x_treat) - np.mean(x_control)
    pooled_sd = np.sqrt((np.var(x_treat) + np.var(x_control)) / 2)
    return np.abs(mean_diff / pooled_sd) if pooled_sd > 0 else 0

def w_smd(x_treat, w_treat, x_control, w_control):
    """Weighted Standardized Mean Difference"""
    mean_treat = np.average(x_treat, weights=w_treat)
    mean_ctrl = np.average(x_control, weights=w_control)
    
    var_treat = np.average((x_treat - mean_treat)**2, weights=w_treat)
    var_ctrl = np.average((x_control - mean_ctrl)**2, weights=w_control)
    
    pooled_sd = np.sqrt((var_treat + var_ctrl) / 2)
    return np.abs((mean_treat - mean_ctrl) / pooled_sd) if pooled_sd > 0 else 0

def run_offpolicy_diagnostics():
    print("="*60)
    print("PHASE 4: OFF-POLICY WEIGHTING DIAGNOSTICS")
    print("="*60)
    
    df, covariates = load_canonical_treatment_data()
    print(f"Sample: {len(df):,} matching pairs. Covariates: {covariates}")
    
    X = StandardScaler().fit_transform(df[covariates])
    W = df["is_high_psi"].values
    
    # Fit Propensity Model
    model = LogisticRegression(max_iter=500, class_weight='balanced')
    model.fit(X, W)
    df['propensity'] = model.predict_proba(X)[:, 1]
    
    # Clip to avoid extreme weights
    df['propensity_clipped'] = np.clip(df['propensity'], 0.05, 0.95)
    
    # Calculate IPW base weights: 1/e(x) for treated, 1/(1-e(x)) for control
    df['weight'] = np.where(df['is_high_psi'] == 1, 
                            1.0 / df['propensity_clipped'], 
                            1.0 / (1.0 - df['propensity_clipped']))
                            
    # 1. Effective Sample Size (ESS)
    N = len(df)
    sum_w = np.sum(df['weight'])
    sum_w2 = np.sum(df['weight']**2)
    ess = (sum_w ** 2) / sum_w2 if sum_w2 > 0 else N
    
    w_treat = df[df['is_high_psi'] == 1]['weight'].values
    w_ctrl = df[df['is_high_psi'] == 0]['weight'].values
    
    ess_treat = (np.sum(w_treat)**2) / np.sum(w_treat**2)
    ess_ctrl = (np.sum(w_ctrl)**2) / np.sum(w_ctrl**2)
    
    print(f"\n[Effective Sample Size (ESS)]")
    print(f"  Nominal N: {N:,} (Treated: {sum(W):,}, Control: {N - sum(W):,})")
    print(f"  Total ESS: {ess:,.1f} ({ess/N * 100:.1f}%)")
    print(f"  Treated ESS: {ess_treat:,.1f} ({ess_treat/sum(W) * 100:.1f}%)")
    print(f"  Control ESS: {ess_ctrl:,.1f} ({ess_ctrl/(N - sum(W)) * 100:.1f}%)")
    
    # 2. Propensity Score Overlap
    print(f"\n[Propensity Overlap]")
    p_treat = df[df['is_high_psi'] == 1]['propensity'].values
    p_ctrl = df[df['is_high_psi'] == 0]['propensity'].values
    
    print(f"  Treated mean propensity: {np.mean(p_treat):.4f} (min: {np.min(p_treat):.4f}, max: {np.max(p_treat):.4f})")
    print(f"  Control mean propensity: {np.mean(p_ctrl):.4f} (min: {np.min(p_ctrl):.4f}, max: {np.max(p_ctrl):.4f})")
    
    pct_overlap = np.mean((p_ctrl >= np.min(p_treat)) & (p_ctrl <= np.max(p_treat))) * 100
    print(f"  Control region covered by treated support: {pct_overlap:.1f}%")
    if pct_overlap < 80:
        print("  ⚠ Low Overlap! IPW estimators may be unstable.")
    else:
        print("  ✓ Strong Support Overlap.")
        
    # 3. Covariate Balance (Standardized Mean Differences)
    print(f"\n[Covariate Balance (SMD)]")
    balance_results = []
    
    for cov in covariates:
        x_all = df[cov].values
        x_treat = df[df['is_high_psi'] == 1][cov].values
        x_ctrl = df[df['is_high_psi'] == 0][cov].values
        
        # Unweighted SMD
        smd_unw = smd(x_treat, x_ctrl)
        
        # Weighted SMD
        smd_w = w_smd(x_treat, w_treat, x_ctrl, w_ctrl)
        
        print(f"  {cov:15s} | Unweighted SMD: {smd_unw:.3f} | Weighted SMD: {smd_w:.3f}")
        
        balance_results.append({
            "Covariate": cov,
            "Unweighted_SMD": smd_unw,
            "Weighted_SMD": smd_w,
            "Balanced": smd_w < 0.1
        })
        
    out_dir = OUTPUT_DIR / 'tables'
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(balance_results).to_csv(out_dir / 'offpolicy_diagnostics_smd.csv', index=False)
    
    max_smd = max([r["Weighted_SMD"] for r in balance_results])
    if max_smd < 0.1:
        print(f"\n✓ Excellent balance achieved! (Max SMD = {max_smd:.3f} < 0.10)")
    else:
        print(f"\n⚠ Imbalance remains! (Max SMD = {max_smd:.3f} > 0.10)")
        
    print(f"SUCCESS: Off-policy diagnostics saved to {out_dir / 'offpolicy_diagnostics_smd.csv'}")

if __name__ == "__main__":
    run_offpolicy_diagnostics()
