"""
Compass Pipeline — Index Construction & Early Window (Steps 6–7).

* **Step 6** — Standardise features (z-score within strata), fit PCA, and
  produce ``CompassIndex1 … CompassIndexN`` per voyage.
* **Step 7** — Re-compute features and index using only the *first N
  search steps* after arrival to ground (early-window compass).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from compass.config import CompassConfig

logger = logging.getLogger(__name__)


# ── feature columns (must match features.py output) ─────────────────────────

FEATURE_COLS = [
    "hill_tail_index",
    "step_length_p50",
    "step_length_p75",
    "step_length_p90",
    "step_length_p95",
    "share_top_decile",
    "mean_resultant_length",
    "heading_autocorr_lag1",
    "heading_run_length_mean",
    "net_to_gross_ratio",
    "grid_cells_visited",
    "recurrence_rate",
    "loiter_fraction",
    "median_speed_mps",
]


# ── standardisation ─────────────────────────────────────────────────────────

def standardize_features(
    df: pd.DataFrame,
    group_col: Optional[str] = None,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    """
    Z-score features within *group_col* (e.g. ``state_time_cell_id``).

    If *group_col* is ``None`` or absent from *df*, standardise globally.

    Returns
    -------
    (df_standardised, stats)
        *stats* maps ``feature → (mean, std)`` (global, for reference).
    """
    fcols = feature_cols or [c for c in FEATURE_COLS if c in df.columns]
    df = df.copy()

    stats: Dict[str, Tuple[float, float]] = {}

    if group_col and group_col in df.columns:
        for col in fcols:
            grp = df.groupby(group_col)[col]
            mu = grp.transform("mean")
            sd = grp.transform("std").replace(0, 1)
            df[col] = (df[col] - mu) / sd
            stats[col] = (float(df[col].mean()), float(df[col].std()))
    else:
        for col in fcols:
            mu = df[col].mean()
            sd = df[col].std()
            if sd == 0:
                sd = 1.0
            df[col] = (df[col] - mu) / sd
            stats[col] = (float(mu), float(sd))

    return df, stats


# ── PCA ─────────────────────────────────────────────────────────────────────

def fit_pca(
    df: pd.DataFrame,
    n_components: int = 3,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[PCA, Dict]:
    """
    Fit PCA on the (already-standardised) feature matrix.

    Returns
    -------
    (pca_model, loadings_dict)
    """
    fcols = feature_cols or [c for c in FEATURE_COLS if c in df.columns]
    X = df[fcols].dropna()

    n_comp = min(n_components, len(fcols), len(X))
    pca = PCA(n_components=n_comp)
    pca.fit(X)

    loadings: Dict[str, list] = {}
    for i in range(n_comp):
        key = f"CompassIndex{i + 1}"
        loadings[key] = {
            col: float(w)
            for col, w in zip(fcols, pca.components_[i])
        }
        loadings[f"{key}_explained_var"] = float(pca.explained_variance_ratio_[i])

    logger.info(
        "PCA: %d components, explained var = %s",
        n_comp,
        [round(v, 3) for v in pca.explained_variance_ratio_],
    )
    return pca, loadings


def compute_compass_index(
    features_df: pd.DataFrame,
    cfg: CompassConfig,
) -> Tuple[pd.DataFrame, Dict]:
    """
    End-to-end: standardise → PCA → add CompassIndex columns.

    Returns
    -------
    (df_with_indices, loadings_dict)
    """
    df_std, _ = standardize_features(
        features_df,
        group_col=cfg.standardize_group_col,
    )

    fcols = [c for c in FEATURE_COLS if c in df_std.columns]
    valid = df_std[fcols].notna().all(axis=1)
    df_valid = df_std.loc[valid]

    if len(df_valid) < cfg.pca_n_components + 1:
        logger.warning("Too few valid voyages (%d) for PCA.", len(df_valid))
        return features_df, {}

    pca, loadings = fit_pca(df_valid, n_components=cfg.pca_n_components, feature_cols=fcols)
    scores = pca.transform(df_valid[fcols])

    out = features_df.copy()
    for i in range(scores.shape[1]):
        col = f"CompassIndex{i + 1}"
        out[col] = np.nan
        out.loc[df_valid.index, col] = scores[:, i]

    logger.info("Added %d CompassIndex columns.", scores.shape[1])
    return out, loadings


def save_loadings(loadings: Dict, path: Path) -> None:
    """Save PCA loadings to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(loadings, f, indent=2)
    logger.info("Saved loadings to %s.", path)


# ── early-window compass (Step 7) ──────────────────────────────────────────

def _first_n_search_steps(
    steps_df: pd.DataFrame,
    n: int,
    arrival_col: Optional[str] = "arrival_ground_timestamp_utc",
) -> pd.DataFrame:
    """
    Return the first *n* search-regime steps per voyage,
    optionally restricted to after ``arrival_col``.
    """
    search = steps_df.loc[steps_df["regime_label"] == "search"].copy()

    if arrival_col and arrival_col in search.columns:
        search = search.loc[
            search["timestamp_utc"] >= search[arrival_col]
        ]

    parts: list[pd.DataFrame] = []
    for _vid, sub in search.groupby("voyage_id", sort=False):
        parts.append(sub.sort_values("timestamp_utc").head(n))

    if not parts:
        return pd.DataFrame(columns=search.columns)
    return pd.concat(parts, ignore_index=True)


def compute_early_window(
    steps_df: pd.DataFrame,
    meta_df: Optional[pd.DataFrame],
    cfg: CompassConfig,
) -> pd.DataFrame:
    """
    Compute compass features and index on the early-window subset.

    Uses ``cfg.early_window_search_steps`` (list of N values).
    Merges ``arrival_ground_timestamp_utc`` from *meta_df* if available.

    Returns
    -------
    pd.DataFrame
        One row per (voyage_id × window_size) with suffix columns.
    """
    from compass.features import compute_compass_features

    # optionally merge arrival time
    df = steps_df.copy()
    if (
        meta_df is not None
        and "arrival_ground_timestamp_utc" in meta_df.columns
    ):
        df = df.merge(
            meta_df[["voyage_id", "arrival_ground_timestamp_utc"]],
            on="voyage_id",
            how="left",
        )

    parts: list[pd.DataFrame] = []

    for n in cfg.early_window_search_steps:
        window_df = _first_n_search_steps(df, n)
        feats = compute_compass_features(window_df, cfg)
        if feats.empty:
            continue

        # re-index
        _, loadings = compute_compass_index(feats, cfg)
        feats = compute_compass_index(feats, cfg)[0]  # need the actual df
        feats["early_window_n"] = n
        parts.append(feats)

    if not parts:
        logger.warning("No early-window features computed.")
        return pd.DataFrame()

    result = pd.concat(parts, ignore_index=True)
    logger.info(
        "Early-window compass: %d rows across window sizes %s.",
        len(result), cfg.early_window_search_steps,
    )
    return result
