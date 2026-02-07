"""
Compass Pipeline — Step Construction (Step 3).

Converts point-level trajectory data into step-level representations.
Offers two alternative step definitions for robustness:

A. **Time-resampled steps** — resample to a fixed interval via
   interpolation (only across short gaps).
B. **Distance-thinned steps** — keep points when cumulative distance
   exceeds a threshold.

Both produce DataFrames with identical column schemas so that all
downstream code is step-definition–agnostic.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from compass.config import CompassConfig

logger = logging.getLogger(__name__)


# ── core step-level computations ────────────────────────────────────────────

def _wrap_angle(a: np.ndarray) -> np.ndarray:
    """Wrap angle to [-π, π]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def compute_raw_steps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build step-level features from consecutive points.

    Requires columns: ``voyage_id, timestamp_utc, x_m, y_m``.

    Adds per-step:
        dx, dy, step_length_m, dt_seconds, speed_mps,
        heading_rad, turning_angle_rad.

    The first point of each voyage has NaN step values.
    """
    df = df.copy()

    # per-voyage diffs
    grp = df.groupby("voyage_id", sort=False)
    df["dx"] = grp["x_m"].diff()
    df["dy"] = grp["y_m"].diff()
    df["dt_seconds"] = grp["timestamp_utc"].diff().dt.total_seconds()

    df["step_length_m"] = np.hypot(df["dx"], df["dy"])
    df["speed_mps"] = df["step_length_m"] / df["dt_seconds"].replace(0, np.nan)

    # heading: angle from east (atan2(dy, dx)), so North = π/2
    df["heading_rad"] = np.arctan2(df["dy"], df["dx"])

    # turning angle = signed difference between consecutive headings
    heading_diff = grp["heading_rad"].diff()
    df["turning_angle_rad"] = _wrap_angle(heading_diff)

    logger.info("Computed raw steps: %d rows.", len(df))
    return df


# ── time resampling ─────────────────────────────────────────────────────────

def resample_time(
    df: pd.DataFrame,
    hours: float,
    max_gap_hours: float,
) -> pd.DataFrame:
    """
    Resample to fixed time interval via linear interpolation.

    Only interpolates across gaps shorter than *max_gap_hours*; longer
    gaps produce NaN rows that downstream code can filter.

    Parameters
    ----------
    df : pd.DataFrame
        Point-level data with ``voyage_id, timestamp_utc, x_m, y_m``.
    hours : float
        Desired fixed interval in hours.
    max_gap_hours : float
        Gaps longer than this are not interpolated.

    Returns
    -------
    pd.DataFrame
        Resampled points (no step features yet — call
        ``compute_raw_steps`` afterwards).
    """
    freq = pd.Timedelta(hours=hours)
    parts: list[pd.DataFrame] = []

    for vid, sub in df.groupby("voyage_id", sort=False):
        sub = sub.set_index("timestamp_utc").sort_index()

        # create uniform time grid
        t0 = sub.index.min().ceil(freq)
        t1 = sub.index.max()
        if t0 >= t1:
            continue
        grid = pd.date_range(t0, t1, freq=freq)

        # interpolate x_m, y_m
        combined = sub[["x_m", "y_m"]].reindex(sub.index.union(grid))
        combined = combined.interpolate(method="time", limit_direction="forward")
        resampled = combined.loc[grid].copy()

        # mask large gaps: compute nearest original timestamp distance
        orig_times = sub.index.values.astype("int64")
        grid_times = grid.values.astype("int64")
        # for each grid point find the nearest original point
        idx_nearest = np.searchsorted(orig_times, grid_times, side="left")
        idx_nearest = np.clip(idx_nearest, 0, len(orig_times) - 1)
        dt_prev = np.abs(grid_times - orig_times[np.clip(idx_nearest - 1, 0, None)])
        dt_next = np.abs(grid_times - orig_times[idx_nearest])
        min_dist_ns = np.minimum(dt_prev, dt_next)
        max_gap_ns = max_gap_hours * 3.6e12
        mask = min_dist_ns > max_gap_ns
        resampled.loc[resampled.index[mask], ["x_m", "y_m"]] = np.nan

        resampled = resampled.dropna(subset=["x_m", "y_m"])
        resampled["voyage_id"] = vid
        resampled = resampled.reset_index().rename(columns={"index": "timestamp_utc"})
        parts.append(resampled)

    if not parts:
        return pd.DataFrame(columns=["voyage_id", "timestamp_utc", "x_m", "y_m"])

    result = pd.concat(parts, ignore_index=True)
    logger.info(
        "Time-resampled (%gh): %d rows, %d voyages.",
        hours, len(result), result["voyage_id"].nunique(),
    )
    return result


# ── distance thinning ───────────────────────────────────────────────────────

def thin_distance(
    df: pd.DataFrame,
    threshold_m: float,
) -> pd.DataFrame:
    """
    Thin trajectory by keeping a point when cumulative distance exceeds
    *threshold_m* since the last kept point.

    Parameters
    ----------
    df : pd.DataFrame
        Point-level data with ``voyage_id, timestamp_utc, x_m, y_m``.
    threshold_m : float
        Distance threshold in metres.

    Returns
    -------
    pd.DataFrame
        Thinned points (call ``compute_raw_steps`` afterwards).
    """
    keep_idx: list[int] = []

    for _vid, sub in df.groupby("voyage_id", sort=False):
        sub = sub.sort_values("timestamp_utc")
        xs = sub["x_m"].values
        ys = sub["y_m"].values
        idxs = sub.index.values

        keep_idx.append(idxs[0])
        last_x, last_y = xs[0], ys[0]

        for i in range(1, len(xs)):
            d = np.hypot(xs[i] - last_x, ys[i] - last_y)
            if d >= threshold_m:
                keep_idx.append(idxs[i])
                last_x, last_y = xs[i], ys[i]

    result = df.loc[keep_idx].copy().reset_index(drop=True)
    logger.info(
        "Distance-thinned (%.0f m): %d → %d rows.",
        threshold_m, len(df), len(result),
    )
    return result


# ── convenience orchestrator ────────────────────────────────────────────────

def build_all_step_variants(
    df: pd.DataFrame,
    cfg: CompassConfig,
) -> dict[str, pd.DataFrame]:
    """
    Build both time-resampled and distance-thinned step DataFrames.

    Returns a dict keyed by variant name (e.g. ``"time_6h"``,
    ``"dist_5000m"``).
    """
    variants: dict[str, pd.DataFrame] = {}

    for h in cfg.time_resample_hours:
        key = f"time_{int(h)}h"
        resampled = resample_time(df, hours=h, max_gap_hours=cfg.interp_max_gap_hours)
        variants[key] = compute_raw_steps(resampled)

    for d in cfg.distance_thin_meters:
        key = f"dist_{int(d)}m"
        thinned = thin_distance(df, threshold_m=d)
        variants[key] = compute_raw_steps(thinned)

    logger.info("Built %d step variants: %s", len(variants), list(variants.keys()))
    return variants
