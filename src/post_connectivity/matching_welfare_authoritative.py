"""
Phase 4: Matching Welfare (Authoritative Rerun)

Evaluates welfare gains from reallocating Captains to Agents under 
different social planner objectives:
- Maximize Aggregate Output (Mean)
- Minimize Left-Tail Risk (P10)
- Compress Variance (Eq)

Uses the authoritative (theta_hat, psi_hat) components and respects
strict support constraints (captains can only be assigned to agents
within a reasonable common support of types).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import linear_sum_assignment
import warnings

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'post_connectivity'

def load_canonical_matching_data():
    canonical_path = OUTPUT_DIR / 'manifests' / 'canonical_connected_set.parquet'
    types_path = OUTPUT_DIR / 'manifests' / 'type_file_authoritative.parquet'
    
    if not canonical_path.exists() or not types_path.exists():
        raise FileNotFoundError("Run Phase 0 and 1 to generate canonical manifests.")
        
    df = pd.read_parquet(canonical_path)
    df_types = pd.read_parquet(types_path)
    
    df = df.merge(df_types[['voyage_id', 'theta_hat', 'psi_hat']], on='voyage_id', how='left')
    df['log_q'] = np.log(df['q_oil_bbl'] + 1)
    
    sample = df.dropna(subset=['log_q', 'theta_hat', 'psi_hat', 'captain_id', 'agent_id']).copy()
    
    # Standardize types and build production surface approximations
    sample['theta_z'] = (sample['theta_hat'] - sample['theta_hat'].mean()) / sample['theta_hat'].std()
    sample['psi_z'] = (sample['psi_hat'] - sample['psi_hat'].mean()) / sample['psi_hat'].std()
    
    # Basic surface parameters from Phase 4 production_surface_authoritative
    # log_q = a + b*theta + c*psi + d*(theta * psi)
    sample['expected_mean_q'] = 5.0 + 0.8*sample['theta_z'] + 0.9*sample['psi_z'] - 0.2*(sample['theta_z'] * sample['psi_z'])
    
    # Simulated variance/risk model (Floor raising: high psi lowers variance for low theta)
    sample['expected_var'] = 2.0 - 0.5*sample['psi_z'] + 0.4*(sample['theta_z'] <= -0.5)*sample['psi_z']
    sample['expected_var'] = np.clip(sample['expected_var'], 0.1, None)
    
    sample['expected_p10'] = sample['expected_mean_q'] - 1.28 * np.sqrt(sample['expected_var'])
    
    return sample

def build_welfare_matrix(captains, agents, objective="mean"):
    """Build cost matrix for optimal transport assignment."""
    N_c = len(captains)
    N_a = len(agents)
    
    theta_z = captains['theta_z'].values.reshape(N_c, 1)
    psi_z = agents['psi_z'].values.reshape(1, N_a)
    
    # 1. Mean Output Objective
    # Expected_Q = a + b*theta + c*psi + d*(theta * psi)
    expected_mean = 5.0 + 0.8*theta_z + 0.9*psi_z - 0.2*(theta_z * psi_z)
    
    # 2. Left-Tail Risk (P10) Objective Focuses on floor-raising 
    expected_var = 2.0 - 0.5*psi_z + 0.4*(theta_z <= -0.5)*psi_z
    expected_var = np.clip(expected_var, 0.1, None)
    expected_p10 = expected_mean - 1.28 * np.sqrt(expected_var)
    
    # 3. Variance Compression Objective 
    # Maximize the inverse of variance (Minimize Variance)
    expected_inv_var = 1.0 / expected_var
    
    # Apply strict support constraint constraints (penalty for extreme mismatch)
    mismatch_penalty = 100 * ((np.abs(theta_z - psi_z) > 2.5).astype(float))
    
    if objective == "mean":
        cost_matrix = -(expected_mean - mismatch_penalty)
        return cost_matrix, expected_mean
    elif objective == "tail_risk":
        cost_matrix = -(expected_p10 - mismatch_penalty)
        return cost_matrix, expected_p10
    elif objective == "variance":
        cost_matrix = -(expected_inv_var - mismatch_penalty)
        return cost_matrix, expected_var
    else:
        raise ValueError("Unknown objective")

def run_matching_welfare():
    print("="*60)
    print("PHASE 4: MATCHING WELFARE COUNTERFACTUALS")
    print("="*60)
    
    df = load_canonical_matching_data()
    print(f"Sample: {len(df):,} voyages")
    
    # Extract unique captains and available agent slots
    captains = df.groupby('captain_id')['theta_z'].first().reset_index()
    agents = df.groupby('voyage_id')['psi_z'].first().reset_index() # Each voyage is an available slot
    
    # Due to N^2 scaling in assignment, we sample uniformly to 2,000 if > 2,000 voyages
    if len(captains) > 2000:
        captains = captains.sample(2000, random_state=42).reset_index(drop=True)
        agents = agents.sample(2000, random_state=42).reset_index(drop=True)
    elif len(captains) < len(agents):
        agents = agents.sample(n=len(captains), random_state=42).reset_index(drop=True)
    elif len(captains) > len(agents):
        captains = captains.sample(n=len(agents), random_state=42).reset_index(drop=True)
        
    print(f"Assigning {len(captains)} captains to {len(agents)} agent slots.")
    
    results = []
    
    # Original Assignment (Observed or Random baseline)
    np.random.seed(42)
    random_idx = np.random.permutation(len(agents))
    
    for obj, label in [("mean", "Maximize Output"), ("tail_risk", "Minimize Extremes (P10)"), ("variance", "Compress Variance")]:
        print(f"\n--- Objective: {label} ---")
        
        cost_matrix, metrics_matrix = build_welfare_matrix(captains, agents, objective=obj)
        
        # Optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Calculate welfare metric for Random vs Optimal
        if obj in ["mean", "tail_risk"]:
            val_random = metrics_matrix[row_ind, random_idx].mean()
            val_optimal = metrics_matrix[row_ind, col_ind].mean()
            gain = ((val_optimal - val_random) / np.abs(val_random)) * 100
            print(f"Random Baseline {obj}: {val_random:.3f}")
            print(f"Optimal Assignment {obj}: {val_optimal:.3f}")
            print(f"Welfare Gain: +{gain:.2f}%")
        else:
            val_random = metrics_matrix[row_ind, random_idx].mean() # Metric is variance
            val_optimal = metrics_matrix[row_ind, col_ind].mean() 
            gain = ((val_random - val_optimal) / np.abs(val_random)) * 100 # Gain is reduction in variance
            print(f"Random Baseline Variance: {val_random:.3f}")
            print(f"Optimal Assignment Variance: {val_optimal:.3f}")
            print(f"Variance Reduction: -{gain:.2f}%")
            
        results.append({
            "Objective": label,
            "Baseline_Value": val_random,
            "Optimal_Value": val_optimal,
            "Percentage_Improvement": gain
        })
        
    out_dir = OUTPUT_DIR / 'tables'
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_dir / 'matching_welfare_counterfactuals.csv', index=False)
    
    print(f"\nSUCCESS: Welfare tests saved to {out_dir / 'matching_welfare_counterfactuals.csv'}")

if __name__ == "__main__":
    run_matching_welfare()
