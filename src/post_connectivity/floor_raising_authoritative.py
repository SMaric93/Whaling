"""
Phase 4: Floor-Raising and Insurance Effects (Authoritative)

Unifies the testing of whether Agent Capability (psi_hat) acts as "insurance"
for low-skill or novice captains.

Tests:
1. Quantile Regression: Does psi_hat have a larger effect at the 10th percentile
   than at the 50th or 90th?
2. Heteroskedasticity: Does psi_hat compress the variance of outcomes, especially
   for low-theta captains?
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'post_connectivity'

def load_canonical_insurance_data():
    canonical_path = OUTPUT_DIR / 'manifests' / 'canonical_connected_set.parquet'
    types_path = OUTPUT_DIR / 'manifests' / 'type_file_authoritative.parquet'
    
    if not canonical_path.exists() or not types_path.exists():
        raise FileNotFoundError("Run Phase 0 and 1 to generate canonical manifests.")
        
    df = pd.read_parquet(canonical_path)
    df_types = pd.read_parquet(types_path)
    
    # Merge authoritative types
    df = df.merge(df_types[['voyage_id', 'theta_hat', 'psi_hat']], on='voyage_id', how='left')
    
    df['log_q'] = np.log(df['q_oil_bbl'] + 1)
    
    # Compute captain experience
    df = df.sort_values(["captain_id", "year_out"])
    df["captain_voyage_num"] = df.groupby("captain_id").cumcount() + 1
    df["is_novice"] = (df["captain_voyage_num"] <= 3).astype(int)
    
    # Standardize types and controls
    sample = df.dropna(subset=['log_q', 'theta_hat', 'psi_hat', 'tonnage']).copy()
    
    for col in ['theta_hat', 'psi_hat']:
        sample[col] = (sample[col] - sample[col].mean()) / sample[col].std()
        
    sample['log_tonnage'] = np.log(sample['tonnage'] + 1)
    sample['log_tonnage'] = (sample['log_tonnage'] - sample['log_tonnage'].mean()) / sample['log_tonnage'].std()
    
    # Define Low-Theta
    theta_threshold = sample['theta_hat'].quantile(0.25)
    sample['is_low_theta'] = (sample['theta_hat'] <= theta_threshold).astype(int)
    
    return sample

def run_insurance_tests():
    print("="*60)
    print("PHASE 4: FLOOR-RAISING / INSURANCE TESTS")
    print("="*60)
    
    df = load_canonical_insurance_data()
    print(f"Sample: {len(df):,} voyages")
    
    results = []
    
    # 1. Quantile Regression (Direct Floor-Raising Test)
    formula_qr = "log_q ~ psi_hat + theta_hat + log_tonnage"
    
    quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
    qr_res = {}
    
    print("\n[Quantile Regression: log_q ~ psi_hat + theta_hat]")
    for q in quantiles:
        try:
            res_qr = smf.quantreg(formula_qr, data=df).fit(q=q)
            beta_psi = res_qr.params.get("psi_hat")
            p_psi = res_qr.pvalues.get("psi_hat")
            
            qr_res[q] = beta_psi
            print(f"  Q{int(q*100)}: β(ψ) = {beta_psi:.4f} (p={p_psi:.4f})")
            
            results.append({
                "Test": "Quantile Reg",
                "Metric": f"Q{str(int(q*100))}",
                "Beta": beta_psi,
                "P_Value": p_psi
            })
        except Exception as e:
            print(f"  Q{int(q*100)} failed: {e}")

    if 0.10 in qr_res and 0.50 in qr_res:
        ratio = qr_res[0.10] / qr_res[0.50] if qr_res[0.50] != 0 else np.nan
        print(f"\n→ Ratio (Q10 / Q50) = {ratio:.2f}")
        if ratio > 1.2:
            print("✓ Floor-Raising Confirmed: Agent effect is much stronger at the bottom tail.")
            
    # 2. Heteroskedasticity Test (Variance Compression)
    # Step A: Get residuals
    res_ols = smf.ols(formula_qr, data=df).fit()
    df['resid_sq'] = res_ols.resid ** 2
    
    # Step B: Regress squared residuals on psi_hat and interactions
    formula_het = "resid_sq ~ psi_hat + theta_hat + is_low_theta*psi_hat"
    res_het = smf.ols(formula_het, data=df).fit(cov_type='HC3')
    
    print("\n[Heteroskedasticity: |ε|² ~ psi_hat × low_theta]")
    print(res_het.summary().tables[1])
    
    beta_het = res_het.params.get("is_low_theta:psi_hat", 0)
    p_het = res_het.pvalues.get("is_low_theta:psi_hat", 1)
    
    results.append({
        "Test": "Heteroskedasticity",
        "Metric": "Low_Theta X Psi_Hat",
        "Beta": beta_het,
        "P_Value": p_het
    })
    
    if beta_het < 0 and p_het < 0.10:
        print("✓ Variance Compression Confirmed: High psi_hat reduces variance specifically for Low-Theta captains.")
        
    out_dir = OUTPUT_DIR / 'tables'
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_dir / 'floor_raising_insurance.csv', index=False)
    
    print(f"\nSUCCESS: Floor-raising tests saved to {out_dir / 'floor_raising_insurance.csv'}")

if __name__ == "__main__":
    run_insurance_tests()
