"""
Phase 1B: Out-of-Sample Route-Choice Prediction Benchmark (Table 2D).

Compares predictive accuracy for ground_or_route using:
  1. Environment only (port + decade)
  2. + Captain ID
  3. + Agent ID
  4. + Captain + Agent

Uses multinomial logistic regression with L2 regularization,
time-split at 1870, and reports log loss, top-3 accuracy, deviance improvement.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss

from .utils.io import write_table

warnings.filterwarnings("ignore", category=UserWarning)


def _top_k_accuracy(y_true: np.ndarray, y_proba: np.ndarray, k: int = 3) -> float:
    """Fraction of observations where the true label is in the top-k predicted."""
    top_k = np.argsort(y_proba, axis=1)[:, -k:]
    return float(np.mean([y_true[i] in top_k[i] for i in range(len(y_true))]))


def run_route_prediction_oos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 2D: Out-of-sample route-choice prediction benchmark.
    """
    print("\n" + "=" * 70)
    print("PHASE 1B: OUT-OF-SAMPLE ROUTE PREDICTION (TABLE 2D)")
    print("=" * 70)

    # Filter to observations with ground
    sample = df[df["ground_or_route"].notna()].copy()

    # Need year for time split
    sample = sample[sample["year_out"].notna()].copy()
    sample["year_out"] = sample["year_out"].astype(int)

    # Restrict ground to labels appearing ≥5 times (avoid degenerate classes)
    ground_counts = sample["ground_or_route"].value_counts()
    valid_grounds = ground_counts[ground_counts >= 5].index
    sample = sample[sample["ground_or_route"].isin(valid_grounds)].copy()

    # Encode target
    le_target = LabelEncoder()
    sample["y"] = le_target.fit_transform(sample["ground_or_route"])
    n_classes = len(le_target.classes_)
    print(f"  Sample: {len(sample):,} voyages, {n_classes} ground categories")

    # Time split
    cutoff = 1870
    train = sample[sample["year_out"] < cutoff].copy()
    test = sample[sample["year_out"] >= cutoff].copy()

    # Restrict test to labels seen in training
    train_labels = set(train["y"].unique())
    test = test[test["y"].isin(train_labels)].copy()

    print(f"  Train: {len(train):,} (before {cutoff})")
    print(f"  Test:  {len(test):,} (from {cutoff})")

    if len(train) < 100 or len(test) < 50:
        print("  ⚠ Insufficient data for OOS prediction")
        return pd.DataFrame()

    # Encode predictors
    le_port = LabelEncoder()
    sample_ports = pd.concat([train["home_port"], test["home_port"]]).fillna("Unknown")
    le_port.fit(sample_ports)
    train["port_enc"] = le_port.transform(train["home_port"].fillna("Unknown"))
    test["port_enc"] = le_port.transform(test["home_port"].fillna("Unknown"))

    le_captain = LabelEncoder()
    all_captains = pd.concat([train["captain_id"], test["captain_id"]])
    le_captain.fit(all_captains)
    train["captain_enc"] = le_captain.transform(train["captain_id"])
    test["captain_enc"] = le_captain.transform(test["captain_id"])

    le_agent = LabelEncoder()
    all_agents = pd.concat([train["agent_id"], test["agent_id"]])
    le_agent.fit(all_agents)
    train["agent_enc"] = le_agent.transform(train["agent_id"])
    test["agent_enc"] = le_agent.transform(test["agent_id"])

    # Decade
    train["decade"] = (train["year_out"] // 10) * 10
    test["decade"] = (test["year_out"] // 10) * 10

    # Define models
    models = {
        "Environment only (port + decade)": ["port_enc", "decade"],
        "+ Captain ID": ["port_enc", "decade", "captain_enc"],
        "+ Agent ID": ["port_enc", "decade", "agent_enc"],
        "+ Captain + Agent": ["port_enc", "decade", "captain_enc", "agent_enc"],
    }

    results = []
    baseline_ll = None

    for model_name, features in models.items():
        X_train = train[features].values.astype(float)
        X_test = test[features].values.astype(float)
        y_train = train["y"].values
        y_test = test["y"].values

        # Fit multinomial logistic with L2
        clf = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=2000,
            C=1.0,
            random_state=42,
            n_jobs=-1,
        )

        try:
            clf.fit(X_train, y_train)
            y_proba = clf.predict_proba(X_test)

            # Ensure probabilities cover the right labels
            ll = log_loss(y_test, y_proba, labels=clf.classes_)
            top3 = _top_k_accuracy(y_test, y_proba, k=3)

            if baseline_ll is None:
                baseline_ll = ll
                dev_improvement = "—"
            else:
                dev_improvement = f"{100 * (1 - ll / baseline_ll):.1f}%"

            results.append({
                "Model": model_name,
                "Log Loss": f"{ll:.3f}",
                "Top-3 Accuracy": f"{top3:.3f}",
                "Deviance Improvement": dev_improvement,
                "N (test)": len(y_test),
            })
            print(f"    {model_name}: LL={ll:.3f}, Top-3={top3:.3f}, Dev↓={dev_improvement}")

        except Exception as e:
            print(f"    {model_name}: FAILED ({e})")
            results.append({
                "Model": model_name,
                "Log Loss": "—",
                "Top-3 Accuracy": "—",
                "Deviance Improvement": "—",
                "N (test)": len(y_test),
            })

    df_results = pd.DataFrame(results)
    write_table(
        df_results,
        "table2_route_prediction_oos",
        caption="Table 2D: Out-of-Sample Route Prediction Benchmark",
        notes=(
            "Multinomial logistic regression with L2 regularization. "
            f"Time split at {cutoff}. Train: {len(train):,}, Test: {len(test):,}. "
            "Deviance improvement relative to environment-only baseline. "
            "Grounds with <5 voyages excluded."
        ),
    )
    return df_results
