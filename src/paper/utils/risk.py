from __future__ import annotations

import numpy as np
import pandas as pd


def lower_tail_reference(values: pd.Series, quantile: float = 0.10) -> float:
    series = pd.to_numeric(values, errors="coerce").dropna()
    if series.empty:
        return np.nan

    cutoff = float(series.quantile(quantile))
    if cutoff > 0:
        return cutoff

    positive = series[series > 0]
    if positive.empty:
        return cutoff
    return float(positive.quantile(quantile))


def expected_shortfall_proxy(values: pd.Series, quantile: float = 0.10) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    cutoff = lower_tail_reference(numeric, quantile=quantile)
    if not np.isfinite(cutoff):
        return pd.Series(np.nan, index=numeric.index, dtype=float)
    return pd.Series(np.maximum(cutoff - numeric.to_numpy(dtype=float), 0.0), index=numeric.index, dtype=float)
