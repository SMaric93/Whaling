"""
Phase 3E: Assignment and Off-Policy Audit
Runs the off-policy evaluation and assignment optimizer using the newly
reconciled authoritative connected set to establish their statuses and 
provide an authoritative summary.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.ml.off_policy_eval import run_off_policy_evaluation
from src.ml.assignment_optimizer import solve_assignment

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'post_connectivity'

def reconcile_ml_outcome_dataset():
    """Build the outcome dataset and forcibly patch with authoritative types."""
    from src.ml.build_outcome_ml_dataset import build_outcome_ml_dataset
    df = build_outcome_ml_dataset(save=False)
    
    # Load authoritative types
    types_path = OUTPUT_DIR / 'manifests' / 'type_file_authoritative.parquet'
    if not types_path.exists():
        raise FileNotFoundError("Missing authoritative types manifest.")
        
    df_types = pd.read_parquet(types_path)
    
    # Merge and overwrite holdouts to use EB estimates for auditing
    df = df.drop(columns=['theta_hat_holdout', 'psi_hat_holdout'], errors='ignore')
    df = df.merge(df_types[['voyage_id', 'theta_hat', 'psi_hat']], on='voyage_id', how='left')
    df['theta_hat_holdout'] = df['theta_hat']
    df['psi_hat_holdout'] = df['psi_hat']
    
    return df

def run_assignment_audit():
    print("="*60)
    print("PHASE 3E: ASSIGNMENT & OFF-POLICY AUDIT")
    print("="*60)
    
    status_matrix = {}
    
    # 1. Dataset Generation
    try:
        df = reconcile_ml_outcome_dataset()
        status_matrix['Outcome Dataset Build'] = 'Pass'
        print(f"Outcome dataset cleanly built. Rows: {len(df)}")
    except Exception as e:
        status_matrix['Outcome Dataset Build'] = f"Fail: {e}"
        print(f"Failed to build outcome dataset: {e}")
        return
        
    # 2. Assignment Optimizer
    try:
        captains = df.groupby("captain_id").agg({
            "theta_hat_holdout": "mean",
            "captain_voyage_num": "max",
        }).reset_index()
    
        agents = df.groupby("agent_id").agg({
            "psi_hat_holdout": "mean",
        }).reset_index()
        
        # Build quick surrogate model
        from sklearn.ensemble import HistGradientBoostingRegressor
        feature_names = ["theta_hat_holdout", "psi_hat_holdout", "scarcity", "captain_voyage_num", "tonnage"]
        avail_features = [f for f in feature_names if f in df.columns]
        X = df[avail_features].fillna(0).values
        y = df["log_q"].values
        
        model = HistGradientBoostingRegressor(max_iter=200, random_state=42).fit(X, y)
        predict_fn = model.predict
        
        res_assign = solve_assignment(
            predict_fn, captains, agents,
            controls={"scarcity": df["scarcity"].median(), "tonnage": df["tonnage"].median()},
            feature_names=avail_features, save_outputs=False
        )
        
        status_matrix['Assignment Optimization'] = 'Pass'
        print("Assignment Optimizer cleanly ran.")
        print(res_assign['welfare_table'])
        
    except Exception as e:
        status_matrix['Assignment Optimization'] = f"Fail: {e}"
        print(f"Assignment optimizer failed: {e}")

    # 3. Off-Policy Eval
    try:
        import sys
        
        # We need to temporarily patch build_outcome_ml_dataset if off_policy_eval imports it directly inside 
        # run_off_policy_evaluation (which it does).
        # We will instead just run the code from off_policy_eval manually to pass our clean df.
        from src.ml.off_policy_eval import estimate_propensity, ipw_estimate, doubly_robust_estimate
        
        target_col = "log_q"
        psi_col = "psi_hat_holdout"
        df_valid = df.dropna(subset=[target_col, psi_col]).copy()
        
        psi_median = df_valid[psi_col].median()
        df_valid["treatment"] = (df_valid[psi_col] > psi_median).astype(int)
        
        covariates = ["theta_hat_holdout", "captain_voyage_num", "scarcity", "tonnage"]
        df_valid = df_valid.dropna(subset=covariates + [target_col, "treatment"]).reset_index(drop=True)
        
        y = df_valid[target_col].values
        treatment = df_valid["treatment"].values
        
        propensity = estimate_propensity(df_valid, "treatment", covariates)
        ipw_results = ipw_estimate(y, treatment, propensity)
        
        status_matrix['Off-Policy IPW Estimator'] = 'Pass'
        print("Off-Policy Evaluator cleanly ran.")
        print(f"IPW ATE: {ipw_results['ate_ipw']:.3f} (SE: {ipw_results['ate_se']:.3f})")
    except Exception as e:
        status_matrix['Off-Policy IPW Estimator'] = f"Fail: {e}"
        print(f"Off-policy evaluator failed: {e}")
        
    df_status = pd.DataFrame(list(status_matrix.items()), columns=["Module", "Status"])
    df_status.to_markdown(OUTPUT_DIR / 'memos' / 'assignment_offpolicy_status.md', index=False)
    print("\nSUCCESS: Phase 3E Assignment/Off-Policy audited.")
    print("Statuses written to assignment_offpolicy_status.md.")

if __name__ == "__main__":
    run_assignment_audit()
