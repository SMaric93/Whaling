from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.reinforcement.utils import absorb_fixed_effects, cluster_se


def numeric(series: pd.Series | None, index: pd.Index | None = None) -> pd.Series:
    """Coerce a series to numeric, returning NaN-filled series if input is None."""
    if series is None:
        return pd.Series(np.nan, index=index, dtype=float)
    return pd.to_numeric(series, errors="coerce")


def normal_pvalue(coef: float, se: float) -> float:
    if not np.isfinite(coef) or not np.isfinite(se) or se <= 0:
        return np.nan
    return float(2 * stats.norm.sf(abs(coef / se)))


def clustered_ols(
    df: pd.DataFrame,
    *,
    outcome: str,
    regressors: list[str],
    cluster_col: str,
    fe_cols: list[str] | None = None,
) -> dict[str, object]:
    fe_cols = fe_cols or []
    required = [outcome, cluster_col] + regressors + fe_cols
    # If any required column is missing, return the empty-result sentinel
    missing = [c for c in required if c not in df.columns]
    if missing:
        return {"coef": {}, "se": {}, "p": {}, "n_obs": 0, "r_squared": np.nan}
    clean = df.dropna(subset=required).copy()
    if clean.empty:
        return {"coef": {}, "se": {}, "p": {}, "n_obs": 0, "r_squared": np.nan}

    y = clean[outcome].to_numpy(dtype=float)
    X = clean[regressors].to_numpy(dtype=float)
    clusters = clean[cluster_col].to_numpy()

    if fe_cols:
        fe_groups = [clean[col].to_numpy() for col in fe_cols]
        result = absorb_fixed_effects(y, X, fe_groups, return_residuals=True)
        mask = result["_mask"]
        coefs = result["coefficients"]
        residuals = result["residuals"]
        X_used = X[mask]
        clusters_used = clusters[mask]
        r_squared = result["r_squared"]
        n_obs = int(result["n_obs"])
    else:
        X_used = np.column_stack([np.ones(len(clean)), X])
        beta = np.linalg.lstsq(X_used, y, rcond=None)[0]
        residuals = y - X_used @ beta
        coefs = beta[1:]
        clusters_used = clusters
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        n_obs = int(len(clean))

    se = cluster_se(X_used, residuals, clusters_used)
    if not fe_cols:
        se = se[1:]

    coef_map = dict(zip(regressors, coefs))
    se_map = dict(zip(regressors, se))
    p_map = {name: normal_pvalue(coef_map[name], se_map[name]) for name in regressors}

    return {
        "coef": coef_map,
        "se": se_map,
        "p": p_map,
        "n_obs": n_obs,
        "r_squared": r_squared,
    }
