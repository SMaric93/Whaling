"""
Phase 4: Production Surface & Submodularity (Authoritative)

Tests for structural submodularity:
    log(Q) = α + β1 θ + β2 ψ + β3 (θ × ψ) + ε
If β3 < 0, Captain Skill and Agent Capability are substitutes (Submodular).

Extends to Quantile and Spline/Polynomial surfaces.
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'post_connectivity'

def load_canonical_surface_data():
    canonical_path = OUTPUT_DIR / 'manifests' / 'canonical_connected_set.parquet'
    types_path = OUTPUT_DIR / 'manifests' / 'type_file_authoritative.parquet'
    
    if not canonical_path.exists() or not types_path.exists():
        raise FileNotFoundError("Run Phase 0 and 1 to generate canonical manifests.")
        
    df = pd.read_parquet(canonical_path)
    df_types = pd.read_parquet(types_path)
    
    # Merge authoritative types
    df = df.merge(df_types[['voyage_id', 'theta_hat', 'psi_hat']], on='voyage_id', how='left')
    
    df['log_q'] = np.log(df['q_oil_bbl'] + 1)
    
    # Standardize types and controls for cleanly scaled interactions
    sample = df.dropna(subset=['log_q', 'theta_hat', 'psi_hat', 'tonnage']).copy()
    
    # Normalize inputs to [0, 1] percentiles or z-scores for stability
    for col in ['theta_hat', 'psi_hat']:
        sample[col] = (sample[col] - sample[col].mean()) / sample[col].std()
        
    sample['log_tonnage'] = np.log(sample['tonnage'] + 1)
    sample['log_tonnage'] = (sample['log_tonnage'] - sample['log_tonnage'].mean()) / sample['log_tonnage'].std()
    
    return sample

def run_production_surface():
    print("="*60)
    print("PHASE 4: PRODUCTION SURFACE & SUBMODULARITY")
    print("="*60)
    
    df = load_canonical_surface_data()
    print(f"Sample: {len(df):,} voyages")
    
    results = []
    
    # 1. Linear OLS Interaction (Baseline Submodularity)
    formula_lin = "log_q ~ theta_hat * psi_hat + log_tonnage"
    res_ols = smf.ols(formula_lin, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['captain_id']})
    print("\n[Linear OLS]")
    print(res_ols.summary().tables[1])
    
    results.append({
        "Model": "Linear OLS",
        "Quantile": "Mean",
        "theta_coef": res_ols.params.get("theta_hat"),
        "psi_coef": res_ols.params.get("psi_hat"),
        "interaction_coef": res_ols.params.get("theta_hat:psi_hat"),
        "interaction_p": res_ols.pvalues.get("theta_hat:psi_hat"),
        "submodular": res_ols.params.get("theta_hat:psi_hat") < 0 and res_ols.pvalues.get("theta_hat:psi_hat") < 0.1
    })

    # 2. Polynomial Surface (Quadratic + Interaction)
    formula_quad = "log_q ~ theta_hat * psi_hat + I(theta_hat**2) + I(psi_hat**2) + log_tonnage"
    res_quad = smf.ols(formula_quad, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['captain_id']})
    print("\n[Quadratic Surface]")
    print(res_quad.summary().tables[1])
    
    results.append({
        "Model": "Quadratic OLS",
        "Quantile": "Mean",
        "theta_coef": res_quad.params.get("theta_hat"),
        "psi_coef": res_quad.params.get("psi_hat"),
        "interaction_coef": res_quad.params.get("theta_hat:psi_hat"),
        "interaction_p": res_quad.pvalues.get("theta_hat:psi_hat"),
        "submodular": res_quad.params.get("theta_hat:psi_hat") < 0 and res_quad.pvalues.get("theta_hat:psi_hat") < 0.1
    })

    # 3. Quantile Regression (Tail Submodularity)
    quantiles = [0.1, 0.5, 0.9]
    for q in quantiles:
        try:
            # Note: statsmodels qr doesn't support clustered SEs out of the box in summary, so we just fit
            res_qr = smf.quantreg(formula_lin, data=df).fit(q=q)
            print(f"\n[Quantile {q} Regression]")
            print(res_qr.summary().tables[1])
            
            results.append({
                "Model": "Quantile Reg",
                "Quantile": q,
                "theta_coef": res_qr.params.get("theta_hat"),
                "psi_coef": res_qr.params.get("psi_hat"),
                "interaction_coef": res_qr.params.get("theta_hat:psi_hat"),
                "interaction_p": res_qr.pvalues.get("theta_hat:psi_hat"),
                "submodular": res_qr.params.get("theta_hat:psi_hat") < 0 and res_qr.pvalues.get("theta_hat:psi_hat") < 0.1
            })
        except Exception as e:
            print(f"Quantile {q} failed: {e}")

    # Output to tables
    out_dir = OUTPUT_DIR / 'tables'
    out_dir.mkdir(parents=True, exist_ok=True)
    res_df = pd.DataFrame(results)
    
    res_df.to_csv(out_dir / 'production_surface_submodularity.csv', index=False)
    
    print("\nSUMMARY OF SUBMODULARITY (Interaction Coef < 0):")
    print(res_df[['Model', 'Quantile', 'interaction_coef', 'interaction_p', 'submodular']].to_markdown())
    print(f"\nSUCCESS: Production surface tests saved to {out_dir / 'production_surface_submodularity.csv'}")

if __name__ == "__main__":
    run_production_surface()
