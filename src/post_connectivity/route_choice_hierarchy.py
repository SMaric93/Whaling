"""
Phase 4: Route Choice Hierarchy (Authoritative Rerun)

Estimates multinomial logit models for destination choice at three levels:
1. Basin
2. Theater (conditional on Basin)
3. Major Ground (conditional on Basin + Theater)

Uses the canonical connected set and authoritative EB estimators (theta_hat, psi_hat)
to answer: Who controls where the ship goes?
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import warnings

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'final'
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'post_connectivity'

ENV_NUMERIC = ["year_out", "tonnage", "duration_days"]
SPECIFICATIONS = [
    ("1. Environment only", []),
    ("2. + Captain ID", ["captain_id"]),
    ("3. + Agent ID", ["agent_id"]),
    ("4. + Captain + Agent ID", ["captain_id", "agent_id"]),
    ("5. + Authoritative Types", ["theta_hat", "psi_hat"])
]

def load_canonical_destination_sample():
    """Load canonical set and join ontology constraints."""
    # 1. Load canonical voyages
    canonical_path = OUTPUT_DIR / 'manifests' / 'canonical_connected_set.parquet'
    types_path = OUTPUT_DIR / 'manifests' / 'type_file_authoritative.parquet'
    
    if not canonical_path.exists() or not types_path.exists():
        raise FileNotFoundError("Run Phase 0 and 1 to generate canonical manifests.")
        
    df = pd.read_parquet(canonical_path)
    df_types = pd.read_parquet(types_path)
    
    # Merge authoritative types
    df = df.merge(df_types[['voyage_id', 'theta_hat', 'psi_hat']], on='voyage_id', how='left')
    
    # 2. Join ontology
    ontology_path = PROJECT_ROOT / 'data' / 'derived' / 'destination_ontology.parquet'
    if ontology_path.exists():
        ontology = pd.read_parquet(ontology_path)
        keep = ["ground_or_route", "basin", "theater", "major_ground", "ground_for_model"]
        df = df.merge(ontology[keep].drop_duplicates("ground_or_route"), on="ground_or_route", how="left")
        df["major_ground_model"] = df["ground_for_model"].where(df["ground_for_model"].notna(), df["major_ground"])
    else:
        print("WARNING: destination_ontology.parquet not found. Creating naive bins.")
        df['basin'] = df['ground_or_route'].astype(str).str[0]
        df['theater'] = df['ground_or_route'].astype(str).str[:2]
        df['major_ground_model'] = df['ground_or_route']
        
    return df

def build_pipeline(numeric_features, categorical_features):
    transformers = []
    if numeric_features:
        transformers.append((
            "num",
            Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
            numeric_features
        ))
    if categorical_features:
        transformers.append((
            "cat",
            Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
            categorical_features
        ))
        
    preprocessor = ColumnTransformer(transformers=transformers)
    model = LogisticRegression(solver="saga", C=0.25, max_iter=500, tol=1e-3, random_state=42)
    return Pipeline([("prep", preprocessor), ("model", model)])

def fit_level(df, panel, level_label, target_col, parent_features, min_count=20):
    sample = df.dropna(subset=[target_col]).copy()
    sample = sample[sample[target_col].astype(str) != "Unknown"].copy()
    
    # Filter rare classes
    counts = sample[target_col].value_counts()
    sample = sample[sample[target_col].isin(counts[counts >= min_count].index)].copy()
    
    if sample[target_col].nunique() < 2 or len(sample) < 100:
        return []

    # Simple time split
    time_cutoff = sample["year_out"].quantile(0.8)
    train = sample[sample["year_out"] <= time_cutoff].copy()
    test = sample[sample["year_out"] > time_cutoff].copy()
    
    # Target encoding
    train['_y'] = pd.factorize(train[target_col])[0]
    valid_classes = train[target_col].unique()
    test = test[test[target_col].isin(valid_classes)].copy()
    
    if len(test) < 10:
        return []
        
    class_map = {k: v for v, k in enumerate(valid_classes)}
    test['_y'] = test[target_col].map(class_map)
    n_classes = len(valid_classes)

    null_probs = train["_y"].value_counts(normalize=True).sort_index().values
    null_array = np.tile(null_probs, (len(test), 1))
    
    try:
        null_log_loss = log_loss(test["_y"], null_array, labels=list(range(n_classes)))
    except Exception:
        null_log_loss = np.nan

    output_rows = []
    
    for spec_label, extras in SPECIFICATIONS:
        numeric = ENV_NUMERIC + [c for c in extras if c in {"theta_hat", "psi_hat"}]
        numeric = [c for c in numeric if c in train.columns]
        
        categorical = parent_features + [c for c in extras if c in {"captain_id", "agent_id"}]
        categorical = [c for c in categorical if c in train.columns]
        
        pipe = build_pipeline(numeric, categorical)
        try:
            pipe.fit(train[numeric + categorical], train["_y"])
            proba = pipe.predict_proba(test[numeric + categorical])
            loss = log_loss(test["_y"], proba, labels=list(range(n_classes)))
            
            # top-1 acc
            pred = pipe.predict(test[numeric + categorical])
            acc = (pred == test["_y"]).mean()
            
            output_rows.append({
                "panel": panel,
                "level": level_label,
                "specification": spec_label,
                "n_classes": n_classes,
                "n_obs_test": len(test),
                "test_log_loss": loss,
                "accuracy": acc,
                "pseudo_r2": 1.0 - (loss / null_log_loss) if pd.notna(null_log_loss) and null_log_loss > 0 else np.nan
            })
        except Exception as e:
            print(f"Skipping {spec_label} for {level_label} due to error: {e}")
            
    return output_rows

def run_route_choice():
    print("="*60)
    print("PHASE 4: ROUTE CHOICE HIERARCHY")
    print("="*60)
    
    df = load_canonical_destination_sample()
    print(f"Loaded {len(df)} canonical voyages for routing split.")
    
    results = []
    results.extend(fit_level(df, "Panel A", "Basin", "basin", [], min_count=40))
    results.extend(fit_level(df, "Panel B", "Theater | Basin", "theater", ["basin"], min_count=20))
    results.extend(fit_level(df, "Panel C", "Ground | Theater", "major_ground_model", ["basin", "theater"], min_count=15))
    
    res_df = pd.DataFrame(results)
    
    out_dir = OUTPUT_DIR / 'tables'
    out_dir.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out_dir / 'route_choice_hierarchy.csv', index=False)
    
    print("\nRESULTS (Pseudo-R2 across Hierarchy):")
    if not res_df.empty:
        summary = res_df.pivot(index='specification', columns='level', values='pseudo_r2')
        print(summary.to_markdown())
    else:
        print("No valid routing models were fitted (insufficient classes).")
        
    print(f"\nSUCCESS: Route choice tests saved to {out_dir / 'route_choice_hierarchy.csv'}")

if __name__ == "__main__":
    run_route_choice()
