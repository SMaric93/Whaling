"""
ML-Based Entity Resolution for Captain Matching.

Implements probabilistic record linkage for:
1. Captain name matching across voyages
2. Captain-Census linkage

Methods:
- Feature engineering (name similarity, temporal overlap)
- Random Forest classifier for match probability
- Blocking to reduce comparison space
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')


def jaro_winkler_similarity(s1: str, s2: str) -> float:
    """Compute Jaro-Winkler similarity between two strings."""
    try:
        import jellyfish
        return jellyfish.jaro_winkler_similarity(s1, s2)
    except ImportError:
        from difflib import SequenceMatcher
        return SequenceMatcher(None, s1, s2).ratio()


def normalize_name(name: str) -> str:
    """Normalize a name for comparison."""
    if pd.isna(name):
        return ""
    name = str(name).lower().strip()
    for suffix in [" jr", " sr", " ii", " iii"]:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    name = "".join(c for c in name if c.isalnum() or c.isspace())
    return name.strip()


def extract_name_parts(name: str) -> Dict[str, str]:
    """Extract first name, last name, and initials."""
    parts = normalize_name(name).split()
    if len(parts) == 0:
        return {"first": "", "last": "", "initials": ""}
    elif len(parts) == 1:
        return {"first": "", "last": parts[0], "initials": parts[0][0] if parts[0] else ""}
    else:
        return {"first": parts[0], "last": parts[-1], "initials": "".join(p[0] for p in parts if p)}


def compute_name_features(name1: str, name2: str) -> Dict[str, float]:
    """Compute similarity features between two names."""
    n1 = normalize_name(name1)
    n2 = normalize_name(name2)
    
    jw_full = jaro_winkler_similarity(n1, n2)
    parts1 = extract_name_parts(name1)
    parts2 = extract_name_parts(name2)
    
    jw_last = jaro_winkler_similarity(parts1["last"], parts2["last"])
    jw_first = jaro_winkler_similarity(parts1["first"], parts2["first"]) if parts1["first"] and parts2["first"] else 0.5
    
    return {
        "jw_full": jw_full,
        "jw_last": jw_last,
        "jw_first": jw_first,
        "exact_full": float(n1 == n2),
        "exact_last": float(parts1["last"] == parts2["last"]),
        "first_initial_match": 1.0 if (parts1["first"] and parts2["first"] and parts1["first"][0] == parts2["first"][0]) else 0.0,
        "len_ratio": min(len(n1), len(n2)) / max(len(n1), len(n2), 1),
    }


def generate_candidate_pairs(df: pd.DataFrame, name_col: str, max_pairs: int = 50000) -> List[Tuple[str, str]]:
    """Generate candidate pairs using blocking by first two letters."""
    print("Generating candidate pairs...")
    
    unique = df.drop_duplicates(subset=["captain_id"]).copy()
    unique["name_norm"] = unique[name_col].apply(normalize_name)
    unique["block"] = unique["name_norm"].str[:2]
    
    print(f"  Unique entities: {len(unique):,}")
    
    pairs = []
    for _, block_df in unique.groupby("block"):
        ids = block_df["captain_id"].values
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pairs.append((ids[i], ids[j]))
                if len(pairs) >= max_pairs:
                    print(f"  Reached max pairs: {max_pairs:,}")
                    return pairs
    
    print(f"  Generated {len(pairs):,} pairs")
    return pairs


def create_training_data(df: pd.DataFrame, name_col: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """Create training data using pseudo-labels from exact name matches."""
    print("Creating training data...")
    
    unique = df.groupby("captain_id").agg({
        name_col: "first",
        "year_out": ["min", "max"],
    }).reset_index()
    unique.columns = ["captain_id", "name", "year_min", "year_max"]
    unique["name_norm"] = unique["name"].apply(normalize_name)
    
    # Positives: exact name matches with different IDs
    positives = []
    for name, group in unique.groupby("name_norm"):
        if len(group) > 1 and len(name) > 3:
            ids = group["captain_id"].values
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    positives.append((ids[i], ids[j], 1))
                    if len(positives) >= 500:
                        break
                if len(positives) >= 500:
                    break
        if len(positives) >= 500:
            break
    
    # Negatives: different names
    all_ids = unique["captain_id"].values
    negatives = []
    np.random.seed(42)
    id_to_name = dict(zip(unique["captain_id"], unique["name_norm"]))
    attempts = 0
    while len(negatives) < 500 and attempts < 5000:
        i, j = np.random.choice(len(all_ids), 2, replace=False)
        id1, id2 = all_ids[i], all_ids[j]
        if id_to_name[id1] != id_to_name[id2]:
            negatives.append((id1, id2, 0))
        attempts += 1
    
    print(f"  Positives: {len(positives)}, Negatives: {len(negatives)}")
    
    # Build feature matrix
    labeled_pairs = positives + negatives
    id_lookup = unique.set_index("captain_id")
    
    features, labels = [], []
    for id1, id2, label in labeled_pairs:
        try:
            r1, r2 = id_lookup.loc[id1], id_lookup.loc[id2]
            feats = compute_name_features(r1["name"], r2["name"])
            feats["year_overlap"] = max(0, min(r1["year_max"], r2["year_max"]) - max(r1["year_min"], r2["year_min"])) / 50
            feats["year_gap"] = abs(r1["year_min"] - r2["year_min"]) / 50
            features.append(feats)
            labels.append(label)
        except KeyError:
            continue
    
    return pd.DataFrame(features), np.array(labels)


def train_matcher_model(X: pd.DataFrame, y: np.ndarray) -> Dict:
    """Train a Random Forest classifier for name matching."""
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    print("\nTraining matcher model...")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="f1")
    print(f"  CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    importance = dict(zip(X.columns, model.feature_importances_))
    
    return {
        "model": model,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cv_f1_mean": cv_scores.mean(),
        "cv_f1_std": cv_scores.std(),
        "feature_importance": importance,
        "feature_names": list(X.columns),
    }


def predict_matches(model_results: Dict, df: pd.DataFrame, pairs: List[Tuple[str, str]], name_col: str) -> pd.DataFrame:
    """Predict match probabilities for candidate pairs."""
    print(f"\nPredicting matches for {len(pairs):,} pairs...")
    
    model = model_results["model"]
    feature_names = model_results["feature_names"]
    
    unique = df.groupby("captain_id").agg({name_col: "first", "year_out": ["min", "max"]}).reset_index()
    unique.columns = ["captain_id", "name", "year_min", "year_max"]
    id_lookup = unique.set_index("captain_id")
    
    results = []
    for id1, id2 in pairs:
        try:
            r1, r2 = id_lookup.loc[id1], id_lookup.loc[id2]
            feats = compute_name_features(r1["name"], r2["name"])
            feats["year_overlap"] = max(0, min(r1["year_max"], r2["year_max"]) - max(r1["year_min"], r2["year_min"])) / 50
            feats["year_gap"] = abs(r1["year_min"] - r2["year_min"]) / 50
            X = pd.DataFrame([feats])[feature_names]
            prob = model.predict_proba(X)[0, 1]
            results.append({"id1": id1, "id2": id2, "name1": r1["name"], "name2": r2["name"], "probability": prob})
        except (KeyError, IndexError):
            continue
    
    df_results = pd.DataFrame(results)
    n_matches = (df_results["probability"] > 0.8).sum()
    print(f"  High-probability matches (>0.8): {n_matches:,}")
    
    return df_results


def run_entity_resolution(df: pd.DataFrame, name_col: str = "captain_name_clean", save_outputs: bool = True) -> Dict:
    """Run complete entity resolution pipeline."""
    print("\n" + "=" * 70)
    print("ML ENTITY RESOLUTION: CAPTAIN MATCHING")
    print("=" * 70)
    
    X, y = create_training_data(df, name_col=name_col)
    model_results = train_matcher_model(X, y)
    pairs = generate_candidate_pairs(df, name_col=name_col)
    predictions = predict_matches(model_results, df, pairs, name_col=name_col)
    
    high_prob = predictions[predictions["probability"] > 0.8].sort_values("probability", ascending=False)
    
    print(f"\nHigh-probability matches (>0.8): {len(high_prob):,}")
    if len(high_prob) > 0:
        print("Top 10 likely duplicates:")
        for _, row in high_prob.head(10).iterrows():
            print(f"  '{row['name1']}' <-> '{row['name2']}': {row['probability']:.3f}")
    
    if save_outputs:
        output_dir = Path("output/ml")
        output_dir.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(output_dir / "entity_resolution_predictions.csv", index=False)
        high_prob.to_csv(output_dir / "likely_duplicates.csv", index=False)
        
        lines = [
            "# ML Entity Resolution: Captain Matching",
            "",
            "## Model Performance",
            f"- Precision: {model_results['precision']:.4f}",
            f"- Recall: {model_results['recall']:.4f}",
            f"- F1: {model_results['f1']:.4f}",
            f"- CV F1: {model_results['cv_f1_mean']:.4f} ± {model_results['cv_f1_std']:.4f}",
            "",
            "## Feature Importance",
        ]
        for feat, imp in sorted(model_results["feature_importance"].items(), key=lambda x: -x[1]):
            lines.append(f"- {feat}: {imp:.4f}")
        lines.extend(["", f"## Results", f"- Candidate pairs: {len(pairs):,}", f"- Likely duplicates: {len(high_prob):,}"])
        
        with open(output_dir / "entity_resolution_report.md", "w") as f:
            f.write("\n".join(lines))
        
        print(f"\nSaved to {output_dir}")
    
    return {"model": model_results, "predictions": predictions, "high_prob_matches": high_prob}


if __name__ == "__main__":
    from src.analyses.data_loader import prepare_analysis_sample
    df = prepare_analysis_sample()
    results = run_entity_resolution(df)
