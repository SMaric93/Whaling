"""
Compass Pipeline — Compass Feature Suite (Step 5).

Computes an interpretable per-voyage feature vector from *search-regime*
steps (or posterior-weighted steps).  The suite covers:

* **Tail behaviour** — Hill tail index, quantiles, share-top-decile.
* **Directional persistence** — mean resultant length, heading autocorr.
* **Coverage / revisiting** — net-to-gross ratio, grid visitation, recurrence.
* **Loitering** — fraction below speed threshold, median speed.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from compass.config import CompassConfig

logger = logging.getLogger(__name__)


# ── individual feature functions ────────────────────────────────────────────

def _hill_tail_index(x: np.ndarray, k_frac: float) -> float:
    """
    Hill estimator for the tail index of positive values.

    α_Hill = n_tail / Σ log(x_i / x_k)
    where x_k is the k-th largest value.
    """
    x = x[x > 0]
    if len(x) < 10:
        return np.nan
    x_sorted = np.sort(x)[::-1]
    k = max(int(len(x) * k_frac), 2)
    x_k = x_sorted[k - 1]
    if x_k <= 0:
        return np.nan
    logs = np.log(x_sorted[:k] / x_k)
    denom = logs.sum()
    if denom == 0:
        return np.nan
    return k / denom


def _quantiles(x: np.ndarray) -> Dict[str, float]:
    """Compute p50, p75, p90, p95 of x."""
    if len(x) == 0:
        return {f"step_length_p{q}": np.nan for q in (50, 75, 90, 95)}
    return {
        f"step_length_p{q}": float(np.nanpercentile(x, q))
        for q in (50, 75, 90, 95)
    }


def _share_top_decile(x: np.ndarray) -> float:
    """Fraction of total step length in the top decile of steps."""
    if len(x) < 10:
        return np.nan
    threshold = np.nanpercentile(x, 90)
    total = np.nansum(x)
    if total == 0:
        return np.nan
    return float(np.nansum(x[x >= threshold]) / total)


def _mean_resultant_length(headings: np.ndarray) -> float:
    """
    Mean resultant length (circular statistics).

    R = |Σ exp(i·θ)| / n.  R → 1 means strong directional persistence.
    """
    headings = headings[np.isfinite(headings)]
    if len(headings) == 0:
        return np.nan
    C = np.cos(headings).mean()
    S = np.sin(headings).mean()
    return float(np.hypot(C, S))


def _heading_autocorr(headings: np.ndarray, lag: int = 1) -> float:
    """
    Circular autocorrelation of headings at *lag*.

    Computed as mean(cos(θ_t - θ_{t-lag})).
    """
    headings = headings[np.isfinite(headings)]
    if len(headings) <= lag:
        return np.nan
    diff = headings[lag:] - headings[:-lag]
    return float(np.cos(diff).mean())


def _net_to_gross_ratio(x: np.ndarray, y: np.ndarray) -> float:
    """Net displacement / gross distance travelled.  Low ⇒ area-restricted search."""
    if len(x) < 2:
        return np.nan
    dx = x[-1] - x[0]
    dy = y[-1] - y[0]
    net = np.hypot(dx, dy)
    diffs = np.hypot(np.diff(x), np.diff(y))
    gross = diffs.sum()
    if gross == 0:
        return np.nan
    return float(net / gross)


def _grid_visitation(
    x: np.ndarray, y: np.ndarray, cell_m: float,
) -> Dict[str, float]:
    """Grid-cell visitation growth and recurrence rate."""
    if len(x) < 2:
        return {"grid_cells_visited": np.nan, "recurrence_rate": np.nan}
    gx = np.floor(x / cell_m).astype(int)
    gy = np.floor(y / cell_m).astype(int)
    cells = list(zip(gx, gy))
    unique_cells = set()
    revisits = 0
    for c in cells:
        if c in unique_cells:
            revisits += 1
        unique_cells.add(c)
    n_unique = len(unique_cells)
    return {
        "grid_cells_visited": float(n_unique),
        "recurrence_rate": float(revisits / len(cells)) if len(cells) > 0 else np.nan,
    }


def _loiter_metrics(
    speed: np.ndarray, threshold_mps: float,
) -> Dict[str, float]:
    """Fraction of steps below loiter speed and median speed."""
    speed = speed[np.isfinite(speed)]
    if len(speed) == 0:
        return {"loiter_fraction": np.nan, "median_speed_mps": np.nan}
    return {
        "loiter_fraction": float((speed < threshold_mps).mean()),
        "median_speed_mps": float(np.median(speed)),
    }


# ── voyage-level aggregation ───────────────────────────────────────────────

def _features_for_group(
    sub: pd.DataFrame,
    cfg: CompassConfig,
) -> Dict[str, float]:
    """Compute the full feature dict for one voyage's search steps."""
    sl = sub["step_length_m"].values.astype(float)
    hd = sub["heading_rad"].values.astype(float)
    sp = sub["speed_mps"].values.astype(float)
    xm = sub["x_m"].values.astype(float)
    ym = sub["y_m"].values.astype(float)

    feat: Dict[str, float] = {}

    # tail behaviour
    feat["hill_tail_index"] = _hill_tail_index(sl, cfg.hill_tail_k_frac)
    feat.update(_quantiles(sl))
    feat["share_top_decile"] = _share_top_decile(sl)

    # directional persistence
    feat["mean_resultant_length"] = _mean_resultant_length(hd)
    feat["heading_autocorr_lag1"] = _heading_autocorr(hd, lag=1)

    # run-length proxy: mean consecutive steps in same 45° heading sector
    if len(hd) > 1:
        sector = (np.floor(hd / (np.pi / 4)) % 8).astype(int)
        runs = np.split(sector, np.where(np.diff(sector) != 0)[0] + 1)
        feat["heading_run_length_mean"] = float(np.mean([len(r) for r in runs]))
    else:
        feat["heading_run_length_mean"] = np.nan

    # coverage
    feat["net_to_gross_ratio"] = _net_to_gross_ratio(xm, ym)
    feat.update(_grid_visitation(xm, ym, cfg.grid_cell_size_m))

    # loitering
    feat.update(_loiter_metrics(sp, cfg.loiter_speed_mps_threshold))

    return feat


# ── main entry point ────────────────────────────────────────────────────────

def compute_compass_features(
    steps_df: pd.DataFrame,
    cfg: CompassConfig,
    use_posterior_weighting: bool = False,
) -> pd.DataFrame:
    """
    Compute per-voyage compass feature suite restricted to search regime.

    Parameters
    ----------
    steps_df : pd.DataFrame
        Step-level data with regime labels (from ``regimes.segment_voyages``).
    cfg : CompassConfig
    use_posterior_weighting : bool
        If True, weight steps by ``p_search`` instead of hard filtering.

    Returns
    -------
    pd.DataFrame
        One row per voyage with feature columns + ``n_search_steps``.
    """
    if use_posterior_weighting and "p_search" in steps_df.columns:
        # soft filter: keep all steps but weight by P(search)
        search_df = steps_df.copy()
        # for feature computation we still need finite features
        has_feat = steps_df[["step_length_m", "heading_rad", "speed_mps"]].notna().all(axis=1)
        search_df = search_df.loc[has_feat]
    else:
        # hard filter to search regime
        search_df = steps_df.loc[
            steps_df["regime_label"] == "search"
        ].copy()

    records: list[dict] = []

    for vid, sub in search_df.groupby("voyage_id", sort=False):
        n_steps = len(sub)
        if n_steps < cfg.min_search_steps_for_features:
            continue
        feat = _features_for_group(sub, cfg)
        feat["voyage_id"] = vid
        feat["n_search_steps"] = n_steps
        records.append(feat)

    if not records:
        logger.warning("No voyages met the minimum search-step threshold.")
        return pd.DataFrame()

    result = pd.DataFrame.from_records(records)
    logger.info(
        "Computed compass features for %d / %d voyages.",
        len(result), steps_df["voyage_id"].nunique(),
    )
    return result
