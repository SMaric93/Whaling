from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from .._table_common import save_table_outputs
from ..config import BuildContext
from ..data import load_connected_sample, load_destination_ontology
from ..utils.footnotes import standard_footnote


ENV_NUMERIC = ["year_out", "tonnage", "duration_days"]
SPECIFICATIONS = [
    ("1. environment only", []),
    ("2. + captain", ["captain_id"]),
    ("3. + agent", ["agent_id"]),
    ("4. + captain + agent", ["captain_id", "agent_id"]),
    ("5. + theta_hat + psi_hat", ["theta", "psi"]),
]


def _load_destination_sample(context: BuildContext) -> pd.DataFrame:
    connected = load_connected_sample(context).copy()
    ontology = load_destination_ontology(context)
    keep = ["ground_or_route", "basin", "theater", "major_ground", "ground_for_model"]
    merged = connected.merge(ontology[keep].drop_duplicates("ground_or_route"), on="ground_or_route", how="left")
    merged["major_ground_model"] = merged["ground_for_model"].where(merged["ground_for_model"].notna(), merged["major_ground"])
    return merged


def _top3_accuracy(proba: np.ndarray, y_true: np.ndarray) -> float:
    if len(proba) == 0:
        return np.nan
    top_k = min(3, proba.shape[1])
    top_idx = np.argsort(proba, axis=1)[:, -top_k:]
    return float(np.mean([truth in preds for truth, preds in zip(y_true, top_idx)]))


def _time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = pd.to_numeric(df["year_out"], errors="coerce")
    cutoff = float(years.quantile(0.8))
    train = df.loc[years <= cutoff].copy()
    test = df.loc[years > cutoff].copy()
    if train.empty or test.empty:
        n_train = int(0.8 * len(df))
        train = df.iloc[:n_train].copy()
        test = df.iloc[n_train:].copy()
    return train, test


def _build_model(train: pd.DataFrame, test: pd.DataFrame, numeric_features: list[str], categorical_features: list[str], regularized: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    numeric_features = [c for c in numeric_features if c in train.columns]
    categorical_features = [c for c in categorical_features if c in train.columns]

    transformers = []
    if numeric_features:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers)
    estimator = LogisticRegression(
        solver="saga",
        C=0.25 if regularized else 5.0,
        max_iter=1500,
        tol=1e-3,
        random_state=0,
    )
    pipeline = Pipeline(steps=[("prep", preprocessor), ("model", estimator)])
    pipeline.fit(train[numeric_features + categorical_features], train["_y"])
    proba = pipeline.predict_proba(test[numeric_features + categorical_features])
    pred = pipeline.predict(test[numeric_features + categorical_features])
    return proba, pred, pipeline.named_steps["model"].classes_


def _fit_level(df: pd.DataFrame, *, panel: str, level_label: str, target_col: str, parent_features: list[str], min_count: int) -> list[dict]:
    sample = df.dropna(subset=[target_col]).copy()
    sample = sample[sample[target_col].astype(str).ne("Unknown")].copy()
    counts = sample[target_col].value_counts()
    sample = sample[sample[target_col].isin(counts[counts >= min_count].index)].copy()
    if sample[target_col].nunique() < 2 or len(sample) < 200:
        return []

    train, test = _time_split(sample)
    encoder = LabelEncoder()
    train["_y"] = encoder.fit_transform(train[target_col].astype(str))
    known_test = test[target_col].astype(str).isin(encoder.classes_)
    test = test.loc[known_test].copy()
    if test.empty:
        return []
    test["_y"] = encoder.transform(test[target_col].astype(str))

    if parent_features:
        for parent in parent_features:
            train[parent] = train[parent].astype(str)
            test[parent] = test[parent].astype(str)

    baseline_key = {}
    output_rows = []
    null_probs = train["_y"].value_counts(normalize=True).sort_index()
    null_array = np.tile(null_probs.reindex(range(len(encoder.classes_)), fill_value=0).to_numpy(), (len(test), 1))
    null_log_loss = float(log_loss(test["_y"], null_array, labels=list(range(len(encoder.classes_)))))

    for model_name, regularized in [("multinomial_logit", False), ("regularized_mnl", True)]:
        model_rows = []
        for spec_label, extras in SPECIFICATIONS:
            numeric = ENV_NUMERIC + [c for c in extras if c in {"theta", "psi"}]
            categorical = parent_features + [c for c in extras if c in {"captain_id", "agent_id"}]
            proba, pred, _ = _build_model(train, test, numeric, categorical, regularized)
            model_log_loss = float(log_loss(test["_y"], proba, labels=list(range(len(encoder.classes_)))))
            row = {
                "panel": panel,
                "level": level_label,
                "model": model_name,
                "specification": spec_label,
                "test_log_loss": model_log_loss,
                "top3_accuracy": _top3_accuracy(proba, test["_y"].to_numpy()),
                "pseudo_r2": 1.0 - model_log_loss / null_log_loss if null_log_loss > 0 else np.nan,
                "n_obs": int(len(test)),
                "note": "Time-split multinomial benchmark built from the destination ontology and connected-set voyages.",
            }
            model_rows.append(row)
            baseline_key[(model_name, spec_label)] = model_log_loss

        env_loss = next(row["test_log_loss"] for row in model_rows if row["specification"] == "1. environment only")
        captain_loss = next(row["test_log_loss"] for row in model_rows if row["specification"] == "2. + captain")
        agent_loss = next(row["test_log_loss"] for row in model_rows if row["specification"] == "3. + agent")
        captain_gain = env_loss - captain_loss
        agent_gain = env_loss - agent_loss

        for row in model_rows:
            row["deviance_improvement_vs_env_only"] = 2.0 * row["n_obs"] * (env_loss - row["test_log_loss"])
            row["captain_marginal_contribution"] = captain_gain if "captain" in row["specification"] else 0.0
            row["agent_marginal_contribution"] = agent_gain if "agent" in row["specification"] else 0.0
            output_rows.append(row)

    return output_rows


def build(context: BuildContext):
    df = _load_destination_sample(context)

    frame = pd.DataFrame(
        _fit_level(df, panel="Panel A", level_label="basin choice", target_col="basin", parent_features=[], min_count=50)
        + _fit_level(df, panel="Panel B", level_label="theater choice conditional on basin", target_col="theater", parent_features=["basin"], min_count=25)
        + _fit_level(df, panel="Panel C", level_label="major-ground choice conditional on theater", target_col="major_ground_model", parent_features=["basin", "theater"], min_count=20)
    )

    memo = standard_footnote(
        sample="Connected-set voyages merged to the repository's destination ontology.",
        unit="Voyage-level destination decision.",
        types_note="The captain and agent identity specifications use one-hot captain_id and agent_id features; the type specification uses theta_hat and psi_hat directly.",
        fe="No fixed effects; destination choice is estimated with multinomial classifiers on a time split.",
        cluster="Not reported in the predictive benchmark table.",
        controls="Environment controls are year_out, tonnage, and duration_days, with basin/theater conditioning variables included in Panels B and C.",
        interpretation="Destination control is hierarchical and environment-heavy: broad location choices are increasingly predictable once the ontology is respected, while captain and agent contributions vary by decision level.",
        caution="This table is a predictive decomposition rather than a causal estimate, and the current build uses time-split validation rather than the full captain-group and agent-group holdout battery.",
    )
    memo = (
        "# table03_hierarchical_map\n\n"
        + memo
        + "\n\nImplementation notes:\n"
        + "- The paper layer rebuilds all three hierarchy levels directly because the shipped next-round CSV only preserved a stale basin benchmark.\n"
        + "- Panel B conditions on basin and Panel C conditions on basin plus theater using the repaired destination ontology.\n"
        + "- The captain and agent specifications use identity features; the final paper-facing specification uses `theta_hat` and `psi_hat` directly.\n"
    )

    return save_table_outputs(
        name="table03_hierarchical_map",
        frame=frame,
        out_dir=context.outputs / "tables",
        context=context,
        memo=memo,
        title="Table 3. Hierarchical Destination Choice",
    )
