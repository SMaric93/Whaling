"""
Phase 3D: ML Ablation Audit
Audits the feature matrices used in Policy Learning (Map vs Compass) to verify
that ablations are numerically distinct, encode categories properly, and avoid
data leakage (e.g., constants filled with zeros).
"""

import pandas as pd
import numpy as np
import hashlib
from collections import defaultdict
from pathlib import Path

from src.ml.policy_learning import (
    ABLATION_LADDER,
    _available_features,
    _encode_categoricals,
    ENVIRONMENT_FEATURES,
    STATE_FEATURES,
    CAPTAIN_FEATURES,
    AGENT_FEATURES
)
from src.ml.build_action_dataset import build_action_dataset

compass_features = ENVIRONMENT_FEATURES + STATE_FEATURES
COMPASS_ABLATION = {
    "env_state": compass_features,
    "env_state_captain": compass_features + CAPTAIN_FEATURES,
    "env_state_agent": compass_features + AGENT_FEATURES,
    "env_state_captain_agent": compass_features + CAPTAIN_FEATURES + AGENT_FEATURES,
    "env_state_captain_agent_types": compass_features + CAPTAIN_FEATURES + AGENT_FEATURES + ["theta_hat_holdout", "psi_hat_holdout"],
}

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'post_connectivity'

def hash_matrix(X: np.ndarray) -> str:
    """Return a strict hash of the matrix contents."""
    return hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest()

def run_ml_audit():
    print("="*60)
    print("PHASE 3D: ML ABLATION AUDIT")
    print("="*60)
    
    print("1. Building action dataset...")
    df = build_action_dataset(force_rebuild=True)
    print(f"Dataset shape: {df.shape}")
    
    # Check if theta_hat_holdout are all missing/zero
    for col in ['theta_hat_holdout', 'psi_hat_holdout']:
        if col in df.columns:
            n_nan = df[col].isna().sum()
            n_zero = (df[col] == 0).sum()
            var = df[col].var()
            print(f"Feature '{col}': {n_nan} NaNs, {n_zero} Zeros out of {len(df)}. Variance: {var:.4f}")
        else:
            print(f"Feature '{col}' is MISSING from the dataset entirely!")

    print("\n2. Auditing Map Model Ablations (Ground Choice)")
    
    # Imitate the map model behavior
    df_sorted = df.sort_values(["voyage_id", "obs_date"])
    df_sorted["_prev_ground"] = df_sorted.groupby("voyage_id")["ground_id"].shift(1)
    gc = df_sorted[
        (df_sorted["ground_id"] != df_sorted["_prev_ground"]) |
        df_sorted["_prev_ground"].isna()
    ].copy().dropna(subset=["ground_id"]).reset_index(drop=True)
    
    map_hashes = {}
    map_shapes = {}
    
    for abl_name, feature_set in ABLATION_LADDER.items():
        features = _available_features(gc, feature_set)
        gc_copy, enc_features = _encode_categoricals(gc.copy(), features)
        X = gc_copy[enc_features].fillna(0).values
        
        map_shapes[abl_name] = X.shape
        map_hashes[abl_name] = hash_matrix(X)
        
        print(f"Map [ {abl_name} ]: Shape {X.shape}, Features: {enc_features}")

    print("\n--- Map Model Distinction Check ---")
    distinct_hashes = set(map_hashes.values())
    if len(distinct_hashes) < len(map_hashes):
        print("CRITICAL WARNING: Some Map ablations have IDENTICAL feature matrices (likely due to missing columns or all-0 fillna)!")
        for abl_name, h in map_hashes.items():
            print(f"  {abl_name}: Hash {h[:8]}")
    else:
        print("PASS: Map feature matrices are strictly distinct.")

    
    print("\n3. Auditing Compass Model Ablations (Within Ground)")
    if "active_search_flag" in df.columns:
        df_comp = df[df["active_search_flag"] == 1].copy()
    else:
        df_comp = df.copy()
        
    comp_hashes = {}
    
    for abl_name, feature_set in COMPASS_ABLATION.items():
        features = _available_features(df_comp, feature_set)
        df_comp_copy, enc_features = _encode_categoricals(df_comp.copy(), features)
        X = df_comp_copy[enc_features].fillna(0).values
        comp_hashes[abl_name] = hash_matrix(X)
        print(f"Compass [ {abl_name} ]: Shape {X.shape}, Features: {enc_features[:3]}...")

    print("\n--- Compass Model Distinction Check ---")
    distinct_compass_hashes = set(comp_hashes.values())
    if len(distinct_compass_hashes) < len(comp_hashes):
        print("CRITICAL WARNING: Some Compass ablations have IDENTICAL feature matrices (likely due to missing columns or all-0 fillna)!")
        for abl_name, h in comp_hashes.items():
            print(f"  {abl_name}: Hash {h[:8]}")
    else:
        print("PASS: Compass feature matrices are strictly distinct.")
        
    # Categorical verification
    from sklearn.model_selection import train_test_split
    print("\n4. Categorical Unseen / One-Hot verification")
    obj_cols = [c for c in ENVIRONMENT_FEATURES if c in df.columns and df[c].dtype in ("object", "category")]
    if not obj_cols:
        print("No categorical object features in ENVIRONMENT_FEATURES. Only continuous/label-encoded used.")
    else:
        print(f"Categoricals found: {obj_cols}")
        for c in obj_cols:
            print(f"  {c}: {df[c].nunique()} unique values")
            # Unseen check
            train, test = train_test_split(df, test_size=0.2, random_state=42)
            train_vals = set(train[c].dropna().unique())
            test_vals = set(test[c].dropna().unique())
            unseen = test_vals - train_vals
            if unseen:
                print(f"  CRITICAL: {c} has {len(unseen)} unseen categories in a random 20% test split!")
                
    if len(distinct_hashes) == len(map_hashes) and len(distinct_compass_hashes) == len(comp_hashes):
        print("\nSUCCESS: Phase 3D ML Ablation Audit pass.")
    else:
        print("\nFAILURE: Phase 3D ML Ablation Audit failed. You MUST fix data leakage/missing vars before running ML.")


if __name__ == '__main__':
    run_ml_audit()
