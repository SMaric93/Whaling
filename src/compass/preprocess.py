"""
Compass Pipeline — Projection & Cleaning (Step 2).

Projects lat/lon to a local metric coordinate system (UTM) per voyage
and optionally applies light smoothing.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

from compass.config import CompassConfig

logger = logging.getLogger(__name__)

# ── lazy pyproj import ──────────────────────────────────────────────────────

def _get_pyproj():
    try:
        import pyproj
        return pyproj
    except ImportError:
        raise ImportError(
            "pyproj is required for coordinate projection. "
            "Install with: pip install pyproj"
        )


# ── UTM zone selection ─────────────────────────────────────────────────────

def select_utm_epsg(lat: float, lon: float) -> int:
    """
    Pick the UTM EPSG code for a given lat/lon.

    Returns
    -------
    int
        EPSG code (e.g. 32617 for UTM zone 17N).
    """
    zone_number = int((lon + 180) / 6) + 1
    if lat >= 0:
        return 32600 + zone_number   # north
    return 32700 + zone_number       # south


def project_points(
    lats: np.ndarray,
    lons: np.ndarray,
    epsg: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project lat/lon arrays to metric (x_m, y_m) using *epsg*.

    Returns
    -------
    (x_m, y_m) : tuple of ndarray
    """
    pyproj = _get_pyproj()
    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326", f"EPSG:{epsg}", always_xy=True,
    )
    x_m, y_m = transformer.transform(lons, lats)
    return np.asarray(x_m, dtype="float64"), np.asarray(y_m, dtype="float64")


# ── per-voyage projection ──────────────────────────────────────────────────

def project_all_voyages(
    df: pd.DataFrame,
    cfg: CompassConfig,
) -> pd.DataFrame:
    """
    Add ``x_m``, ``y_m``, ``epsg`` columns by projecting each voyage
    into its own UTM zone (chosen by median lat/lon).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``voyage_id``, ``lat``, ``lon``.
    cfg : CompassConfig

    Returns
    -------
    pd.DataFrame
        Same rows with three new columns.
    """
    x_all = np.empty(len(df), dtype="float64")
    y_all = np.empty(len(df), dtype="float64")
    epsg_all = np.empty(len(df), dtype="int32")

    for vid, idx in df.groupby("voyage_id").groups.items():
        sub = df.loc[idx]
        med_lat = sub["lat"].median()
        med_lon = sub["lon"].median()
        epsg = select_utm_epsg(med_lat, med_lon)

        x, y = project_points(
            sub["lat"].values, sub["lon"].values, epsg,
        )
        x_all[idx] = x
        y_all[idx] = y
        epsg_all[idx] = epsg

    df = df.copy()
    df["x_m"] = x_all
    df["y_m"] = y_all
    df["epsg"] = epsg_all

    logger.info(
        "Projected %d voyages into per-voyage UTM zones.", df["voyage_id"].nunique(),
    )
    return df


# ── optional smoothing ─────────────────────────────────────────────────────

def smooth_positions(
    df: pd.DataFrame,
    cfg: CompassConfig,
) -> pd.DataFrame:
    """
    Apply light centred moving-average to ``x_m, y_m`` per voyage.

    Preserves turns by using a short window (default 5).
    Disabled when ``cfg.smoothing_enabled is False``.
    """
    if not cfg.smoothing_enabled:
        return df

    df = df.copy()
    w = cfg.smoothing_window

    for vid, idx in df.groupby("voyage_id").groups.items():
        for col in ("x_m", "y_m"):
            series = df.loc[idx, col]
            smoothed = series.rolling(w, center=True, min_periods=1).mean()
            df.loc[idx, col] = smoothed.values

    logger.info("Smoothed positions (window=%d).", w)
    return df
