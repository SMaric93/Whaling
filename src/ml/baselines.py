"""
ML Layer — Baseline Models.

Every ML module must compare against simple baselines.
Provides consistent wrappers for:
- Mean / naive prediction
- Logistic regression (binary & multinomial)
- Linear regression (with optional spline features)
- Cox PH (via lifelines)
- Discrete-time logit hazard
- Fixed-effects / grouped linear models
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.ml.config import ML_CFG

logger = logging.getLogger(__name__)


class LogisticBaselinePipeline(Pipeline):
    """Scaler + logistic pipeline with estimator-like compatibility accessors."""

    @property
    def classes_(self):
        return self.named_steps["logit"].classes_

    @property
    def coef_(self):
        return self.named_steps["logit"].coef_

    @property
    def intercept_(self):
        return self.named_steps["logit"].intercept_

    @property
    def n_iter_(self):
        return self.named_steps["logit"].n_iter_

    @property
    def n_jobs(self):
        return self.named_steps["logit"].n_jobs


# ═══════════════════════════════════════════════════════════════════════════
# Mean / Naive Baselines
# ═══════════════════════════════════════════════════════════════════════════

class MeanBaseline(BaseEstimator, RegressorMixin):
    """Predict the training-set mean."""

    def __init__(self):
        self.mean_ = None

    def fit(self, X, y):
        self.mean_ = float(np.nanmean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


class MajorityClassBaseline(BaseEstimator, ClassifierMixin):
    """Predict the most frequent class; return uniform class probabilities from train."""

    def __init__(self):
        self.majority_class_ = None
        self.class_probs_ = None
        self.classes_ = None

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        counts = np.array([np.sum(y == c) for c in self.classes_])
        self.majority_class_ = self.classes_[np.argmax(counts)]
        self.class_probs_ = counts / counts.sum()
        return self

    def predict(self, X):
        return np.full(len(X), self.majority_class_)

    def predict_proba(self, X):
        return np.tile(self.class_probs_, (len(X), 1))


# ═══════════════════════════════════════════════════════════════════════════
# Linear / Logistic Baselines
# ═══════════════════════════════════════════════════════════════════════════

def fit_logistic_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    max_iter: int = 2000,
    penalty: str = "l2",
    C: float = 1.0,
    multi_class: str = "auto",
) -> Any:
    """
    Fit logistic regression baseline.

    For binary targets, returns LogisticRegression.
    For multiclass, returns multinomial LogisticRegression.
    """
    from sklearn.linear_model import LogisticRegression

    n_classes = len(np.unique(y_train))
    if n_classes > 2:
        multi_class = "multinomial"

    logistic_kwargs = {
        "penalty": penalty,
        "C": C,
        "max_iter": max_iter,
        "solver": "lbfgs",
        "random_state": ML_CFG.random_seed,
        # Restricted environments can reject the internal parallel setup used
        # by some sklearn estimators. Keep the baseline single-process.
        "n_jobs": 1,
    }
    if multi_class not in {None, "auto", "multinomial"}:
        logistic_kwargs["multi_class"] = multi_class

    model = LogisticBaselinePipeline([
        ("scaler", StandardScaler()),
        ("logit", LogisticRegression(**logistic_kwargs)),
    ])

    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    logger.info(
        "Logistic baseline: %d classes, n=%d, p=%d, %.1fs",
        n_classes, len(y_train), X_train.shape[1], elapsed,
    )
    return model


def fit_linear_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    with_splines: bool = False,
    spline_cols: List[int] = None,
    spline_df: int = 4,
    alpha: float = 0.01,
) -> Any:
    """
    Fit linear regression baseline.

    Parameters
    ----------
    with_splines : bool
        If True, add cubic spline basis expansion for selected columns.
    spline_cols : list of int
        Column indices to apply spline expansion to.
    spline_df : int
        Degrees of freedom for spline basis.
    alpha : float
        Ridge regularization strength.
    """
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if with_splines and spline_cols:
        try:
            from sklearn.preprocessing import SplineTransformer
            from sklearn.compose import ColumnTransformer

            spline_tf = ColumnTransformer(
                transformers=[
                    ("spline", SplineTransformer(n_knots=spline_df + 1, degree=3), spline_cols),
                ],
                remainder="passthrough",
            )
            pipeline = Pipeline([
                ("spline", spline_tf),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=alpha)),
            ])
        except ImportError:
            logger.warning("SplineTransformer not available; using plain linear")
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=alpha)),
            ])
    else:
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ])

    t0 = time.time()
    pipeline.fit(X_train, y_train)
    elapsed = time.time() - t0
    logger.info(
        "Linear baseline: n=%d, p=%d, splines=%s, %.1fs",
        len(y_train), X_train.shape[1], with_splines, elapsed,
    )
    return pipeline


# ═══════════════════════════════════════════════════════════════════════════
# Discrete-Time Hazard (Logit)
# ═══════════════════════════════════════════════════════════════════════════

def fit_discrete_hazard_logit(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    max_iter: int = 2000,
) -> Any:
    """
    Discrete-time hazard model using logistic regression.

    y_train should be 0/1 exit indicator on person-period data.
    """
    return fit_logistic_baseline(
        X_train, y_train,
        max_iter=max_iter,
        penalty="l2",
        C=1.0,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Cox PH Baseline
# ═══════════════════════════════════════════════════════════════════════════

def fit_cox_baseline(
    df_train: pd.DataFrame,
    duration_col: str,
    event_col: str,
    *,
    covariate_cols: List[str] = None,
    penalizer: float = None,
) -> Optional[Any]:
    """
    Cox PH model via lifelines.

    Returns the fitted CoxPHFitter or None if lifelines unavailable.
    """
    try:
        from lifelines import CoxPHFitter
    except ImportError:
        logger.warning("lifelines not installed; skipping Cox baseline")
        return None

    penalizer = penalizer or ML_CFG.cox_penalizer

    cols_to_use = [duration_col, event_col]
    if covariate_cols:
        cols_to_use += covariate_cols

    df_sub = df_train[cols_to_use].dropna()

    cph = CoxPHFitter(penalizer=penalizer)
    t0 = time.time()
    cph.fit(df_sub, duration_col=duration_col, event_col=event_col)
    elapsed = time.time() - t0

    logger.info(
        "Cox baseline: n=%d, covariates=%d, concordance=%.3f, %.1fs",
        len(df_sub),
        len(covariate_cols or []),
        cph.concordance_index_,
        elapsed,
    )
    return cph


# ═══════════════════════════════════════════════════════════════════════════
# Grouped / Fixed-Effects Linear Model
# ═══════════════════════════════════════════════════════════════════════════

def fit_grouped_linear(
    df_train: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    group_col: str,
    *,
    alpha: float = 0.01,
) -> Dict[str, Any]:
    """
    Linear model with fixed effects for groups (demeaned regression).

    Demeans features and target within groups, then fits Ridge.
    """
    from sklearn.linear_model import Ridge

    df = df_train[feature_cols + [target_col, group_col]].dropna()

    # Demean within groups
    group_means = df.groupby(group_col)[feature_cols + [target_col]].transform("mean")
    df_demeaned = df[feature_cols + [target_col]] - group_means

    X = df_demeaned[feature_cols].values
    y = df_demeaned[target_col].values

    model = Ridge(alpha=alpha)
    model.fit(X, y)

    logger.info(
        "Grouped linear: groups=%d, n=%d, p=%d",
        df[group_col].nunique(), len(df), len(feature_cols),
    )

    return {
        "model": model,
        "feature_cols": feature_cols,
        "group_col": group_col,
        "group_means": df.groupby(group_col)[feature_cols + [target_col]].mean(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Baseline Registry
# ═══════════════════════════════════════════════════════════════════════════

BASELINE_REGISTRY = {
    "mean": MeanBaseline,
    "majority_class": MajorityClassBaseline,
    "logistic": fit_logistic_baseline,
    "multinomial_logistic": lambda X, y, **kw: fit_logistic_baseline(X, y, multi_class="multinomial", **kw),
    "linear": fit_linear_baseline,
    "linear_spline": lambda X, y, **kw: fit_linear_baseline(X, y, with_splines=True, **kw),
    "discrete_hazard": fit_discrete_hazard_logit,
    "cox": fit_cox_baseline,
    "grouped_linear": fit_grouped_linear,
}


def get_baseline(name: str) -> Any:
    """Get a baseline model constructor by name."""
    if name not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline '{name}'. Options: {list(BASELINE_REGISTRY)}")
    return BASELINE_REGISTRY[name]


def fit_all_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task: str = "regression",
) -> Dict[str, Any]:
    """
    Fit all appropriate baselines for a task type.

    Returns dict of name → fitted model.
    """
    models = {}

    if task == "regression":
        models["mean"] = MeanBaseline().fit(X_train, y_train)
        models["linear"] = fit_linear_baseline(X_train, y_train)
    elif task == "classification":
        models["majority_class"] = MajorityClassBaseline().fit(X_train, y_train)
        models["logistic"] = fit_logistic_baseline(X_train, y_train)
    elif task == "survival":
        models["discrete_hazard"] = fit_discrete_hazard_logit(X_train, y_train)

    return models
