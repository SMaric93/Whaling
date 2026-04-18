"""Observation encoder: learned emission model for the HSMM.

Trains a LightGBM multiclass classifier on anchor-labeled observation-weeks
to produce calibrated state probabilities.  These replace the hand-crafted
``_emission_probabilities()`` function in ``voyage_state_model.py``.

Usage::

    from .emission_model import train_observation_encoder, predict_state_emissions
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

from .state_space import NUM_STATES, STATE_INDEX, STATE_NAMES

logger = logging.getLogger(__name__)

# Try to import LightGBM; fall back to sklearn HistGradientBoosting
try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

# Feature blocks from the spec
_EVENT_TYPES = ["dep", "arr", "spk", "rpt", "inp", "wrk"]

_NUMERIC_FEATURES = [
    "oil_total",
    "oil_delta",
    "oil_delta_per_elapsed_week",
    "bone_lbs",
    "days_out",
    "weeks_since_departure",
    "weeks_since_last_observation",
    "report_lag_days",
]

_QUALITY_FEATURES = [
    "mean_confidence",
    "min_confidence",
    "quality_weight",
    "missing_fraction",
    "link_probability_mean",
]

_REMARKS_PROB_PREFIXES = ["p_primary__", "p_tag__"]


def build_state_features(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """Engineer the feature matrix from the weekly observation panel.

    Returns a DataFrame with one row per observation-week, containing:
    - One-hot event type counts
    - Remarks probability vectors
    - Source mode fraction
    - Staleness / lag features
    - Numeric cargo and days_out with missing indicators
    - Quality and confidence features
    """
    features = pd.DataFrame(index=weekly_df.index)

    # Event type counts
    for event_type in _EVENT_TYPES:
        features[f"evt_count_{event_type}"] = 0

    if "event_type_counts_json" in weekly_df.columns:
        for idx, val in weekly_df["event_type_counts_json"].items():
            if isinstance(val, str):
                try:
                    counts = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    counts = {}
            elif isinstance(val, dict):
                counts = val
            else:
                counts = {}
            for et in _EVENT_TYPES:
                features.at[idx, f"evt_count_{et}"] = counts.get(et, 0)

    # Source mode
    features["source_mode_flow_fraction"] = pd.to_numeric(
        weekly_df.get("source_mode_flow_fraction"), errors="coerce"
    ).fillna(0.5)

    # Numeric features with missing indicators
    for col in _NUMERIC_FEATURES:
        if col in weekly_df.columns:
            vals = pd.to_numeric(weekly_df[col], errors="coerce")
            features[col] = vals.fillna(0.0)
            features[f"{col}_missing"] = vals.isna().astype(float)
        else:
            features[col] = 0.0
            features[f"{col}_missing"] = 1.0

    # Quality features
    for col in _QUALITY_FEATURES:
        if col in weekly_df.columns:
            features[col] = pd.to_numeric(weekly_df[col], errors="coerce").fillna(0.0)
        else:
            features[col] = 0.0

    # Remarks probability vectors
    for col in weekly_df.columns:
        for prefix in _REMARKS_PROB_PREFIXES:
            if col.startswith(prefix):
                features[col] = pd.to_numeric(weekly_df[col], errors="coerce").fillna(0.0)

    # Distress and productivity scores
    for col in ["distress_severity_max", "productivity_polarity_mean"]:
        if col in weekly_df.columns:
            features[col] = pd.to_numeric(weekly_df[col], errors="coerce").fillna(0.0)

    # N events
    features["n_events"] = pd.to_numeric(weekly_df.get("n_events"), errors="coerce").fillna(1.0)

    return features


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------


def _build_training_labels(
    weekly_df: pd.DataFrame,
    anchor_df: pd.DataFrame,
    min_strength: float = 0.55,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Join weekly panel with anchors and extract labeled examples.

    Returns (merged_df, y_labels, sample_weights) for rows with
    sufficient anchor strength.
    """
    prior_cols = [f"state_prior_{s}" for s in STATE_NAMES]
    anchor_subset = anchor_df[
        ["voyage_id", "week_idx", "anchor_strength"] + prior_cols
    ].copy()

    merged = weekly_df.merge(anchor_subset, on=["voyage_id", "week_idx"], how="left")

    # Keep rows with strong enough anchors
    mask = merged["anchor_strength"].fillna(0) >= min_strength
    labeled = merged[mask].copy()

    if labeled.empty:
        return labeled, np.array([]), np.array([])

    # Hard label = argmax of anchor posterior
    prior_matrix = labeled[prior_cols].to_numpy(dtype=np.float64)
    y_labels = prior_matrix.argmax(axis=1)

    # Sample weights = anchor strength × quality weight
    quality_w = pd.to_numeric(labeled.get("quality_weight"), errors="coerce").fillna(0.5)
    sample_weights = labeled["anchor_strength"].to_numpy(dtype=np.float64) * quality_w.to_numpy(dtype=np.float64)

    return labeled, y_labels, sample_weights


def train_observation_encoder(
    weekly_df: pd.DataFrame,
    anchor_df: pd.DataFrame,
    feature_spec: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train the observation encoder that outputs state probabilities.

    Parameters
    ----------
    weekly_df : pd.DataFrame
        Weekly observation panel.
    anchor_df : pd.DataFrame
        Anchor posteriors (from ``build_anchor_posteriors``).
    feature_spec : dict, optional
        Override feature parameters. Currently unused.
    config : dict, optional
        Training configuration overrides.

    Returns
    -------
    dict
        ``model``: fitted classifier,
        ``feature_columns``: ordered feature names,
        ``metrics``: evaluation metrics,
        ``calibrated``: whether isotonic calibration was applied.
    """
    config = config or {}
    min_anchor_strength = config.get("min_anchor_strength", 0.55)

    # Build training data
    labeled, y_labels, sample_weights = _build_training_labels(
        weekly_df, anchor_df, min_strength=min_anchor_strength
    )

    if len(labeled) < 50:
        logger.warning(
            "[emission] Only %d labeled examples; returning uniform encoder",
            len(labeled),
        )
        return {
            "model": None,
            "feature_columns": [],
            "metrics": {"n_train": 0, "status": "insufficient_data"},
            "calibrated": False,
        }

    features = build_state_features(labeled)
    feature_columns = list(features.columns)
    X = features.to_numpy(dtype=np.float64)

    # Replace any remaining NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info(
        "[emission] Training on %d labeled weeks, %d features, %d states represented",
        len(X),
        X.shape[1],
        len(np.unique(y_labels)),
    )

    # Build classifier
    n_classes = len(np.unique(y_labels))
    if _HAS_LGBM:
        base_clf = LGBMClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier
        base_clf = HistGradientBoostingClassifier(
            max_iter=300,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
        )

    # Train with calibration if enough data
    min_class_count = int(pd.Series(y_labels).value_counts().min())
    if n_classes >= 2 and min_class_count >= 5 and len(X) >= 100:
        n_folds = min(3, min_class_count)
        try:
            model = CalibratedClassifierCV(
                estimator=base_clf,
                method="isotonic",
                cv=n_folds,
            )
            model.fit(X, y_labels, sample_weight=sample_weights)
            calibrated = True
        except Exception as e:
            logger.warning("[emission] Calibration failed (%s); fitting uncalibrated", e)
            if _HAS_LGBM:
                base_clf.fit(X, y_labels, sample_weight=sample_weights)
            else:
                base_clf.fit(X, y_labels, sample_weight=sample_weights)
            model = base_clf
            calibrated = False
    else:
        if _HAS_LGBM:
            base_clf.fit(X, y_labels, sample_weight=sample_weights)
        else:
            base_clf.fit(X, y_labels, sample_weight=sample_weights)
        model = base_clf
        calibrated = False

    # Evaluate on training data (in-sample; held-out eval happens via HSMM)
    y_pred = model.predict(X)
    accuracy = float((y_pred == y_labels).mean())
    y_prob = model.predict_proba(X)

    # Ensure all states are represented in output
    model_classes = list(model.classes_) if hasattr(model, "classes_") else list(range(NUM_STATES))

    metrics = {
        "n_train": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_states_observed": int(n_classes),
        "accuracy_insample": accuracy,
        "calibrated": calibrated,
        "model_classes": [int(c) for c in model_classes],
        "class_distribution": {
            STATE_NAMES[int(c)]: int((y_labels == c).sum())
            for c in np.unique(y_labels)
        },
    }

    logger.info(
        "[emission] Trained: accuracy=%.3f, calibrated=%s, %d classes",
        accuracy,
        calibrated,
        n_classes,
    )

    return {
        "model": model,
        "feature_columns": feature_columns,
        "metrics": metrics,
        "calibrated": calibrated,
        "model_classes": model_classes,
    }


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def predict_state_emissions(
    weekly_df: pd.DataFrame,
    encoder_bundle: dict[str, Any],
) -> pd.DataFrame:
    """Predict per-state emission probabilities for all observation-weeks.

    Parameters
    ----------
    weekly_df : pd.DataFrame
        Weekly observation panel.
    encoder_bundle : dict
        Output of ``train_observation_encoder``.

    Returns
    -------
    pd.DataFrame
        Original columns plus ``state_prob_<name>`` and ``encoder_entropy``.
    """
    model = encoder_bundle.get("model")
    feature_columns = encoder_bundle.get("feature_columns", [])

    result = weekly_df.copy()

    if model is None or not feature_columns:
        # Uniform emissions fallback
        logger.warning("[emission] No trained encoder; using uniform emissions")
        for sname in STATE_NAMES:
            result[f"state_prob_{sname}"] = 1.0 / NUM_STATES
        result["encoder_entropy"] = np.log(NUM_STATES)
        return result

    features = build_state_features(weekly_df)

    # Ensure feature alignment
    for col in feature_columns:
        if col not in features.columns:
            features[col] = 0.0
    features = features[feature_columns]

    X = features.to_numpy(dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Predict probabilities
    probs = model.predict_proba(X)
    model_classes = encoder_bundle.get("model_classes", list(range(probs.shape[1])))

    # Map model classes to full state space
    full_probs = np.full((len(X), NUM_STATES), 1e-4, dtype=np.float64)
    for j, cls in enumerate(model_classes):
        if int(cls) < NUM_STATES:
            full_probs[:, int(cls)] = probs[:, j]

    # Renormalize
    full_probs /= full_probs.sum(axis=1, keepdims=True)

    # Assign columns
    for si, sname in enumerate(STATE_NAMES):
        result[f"state_prob_{sname}"] = full_probs[:, si]

    # Entropy
    entropy = -np.sum(full_probs * np.log(np.clip(full_probs, 1e-12, 1.0)), axis=1)
    result["encoder_entropy"] = entropy

    logger.info(
        "[emission] Predicted emissions for %d weeks; mean entropy=%.3f",
        len(result),
        float(entropy.mean()),
    )

    return result


def evaluate_observation_encoder(
    weekly_df: pd.DataFrame,
    anchor_df: pd.DataFrame,
    encoder_bundle: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate encoder against anchor posteriors on a held-out set."""
    config = config or {}
    min_strength = config.get("eval_min_anchor_strength", 0.70)

    labeled, y_true, weights = _build_training_labels(
        weekly_df, anchor_df, min_strength=min_strength
    )

    if len(labeled) < 10:
        return {"status": "insufficient_eval_data", "n_eval": len(labeled)}

    predictions = predict_state_emissions(labeled, encoder_bundle)
    prob_cols = [f"state_prob_{s}" for s in STATE_NAMES]
    pred_probs = predictions[prob_cols].to_numpy(dtype=np.float64)
    y_pred = pred_probs.argmax(axis=1)

    accuracy = float((y_pred == y_true).mean())
    weighted_accuracy = float(np.average(y_pred == y_true, weights=weights))

    return {
        "n_eval": int(len(labeled)),
        "accuracy": accuracy,
        "weighted_accuracy": weighted_accuracy,
        "min_anchor_strength": min_strength,
    }
