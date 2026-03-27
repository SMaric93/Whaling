"""
Shared anomaly-detection helpers for QA.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from config import ML_SHIFT_CONFIG

try:
    from sklearn.ensemble import IsolationForest
except ImportError:  # pragma: no cover
    IsolationForest = None


def compute_anomaly_scores(
    df: pd.DataFrame,
    numeric_cols: list[str],
    *,
    contamination: Optional[float] = None,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    Fit IsolationForest on selected numeric columns and return anomaly scores.
    """
    contamination = (
        ML_SHIFT_CONFIG.anomaly_contamination
        if contamination is None else contamination
    )
    random_state = (
        ML_SHIFT_CONFIG.random_state
        if random_state is None else random_state
    )

    cols = [col for col in numeric_cols if col in df.columns]
    if IsolationForest is None or len(cols) == 0:
        return pd.DataFrame(columns=["anomaly_score", "anomaly_flag"])

    work = df[cols].replace([np.inf, -np.inf], np.nan)
    keep = work.notna().any(axis=1)
    if keep.sum() < ML_SHIFT_CONFIG.anomaly_min_rows:
        return pd.DataFrame(index=df.index, columns=["anomaly_score", "anomaly_flag"])

    filled = work.loc[keep].fillna(work.loc[keep].median(numeric_only=True))
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=200,
    )
    scores = model.fit_predict(filled)
    decision = model.decision_function(filled)

    result = pd.DataFrame(index=df.index, columns=["anomaly_score", "anomaly_flag"])
    result["anomaly_score"] = np.nan
    result["anomaly_flag"] = False
    result.loc[keep, "anomaly_score"] = -decision
    result.loc[keep, "anomaly_flag"] = scores == -1
    return result
