"""
Reinforcement Test Suite — Search Behavior Metrics.

Reusable search path metrics computed at multiple granularities:
voyage, ground-spell, and rolling windows.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import CFG, COLS
from .utils import haversine_nm

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Core Metric Functions (operate on arrays)
# ═══════════════════════════════════════════════════════════════════════════

def levy_exponent(step_lengths: np.ndarray, x_min: float = 1.0) -> Tuple[float, float]:
    """
    Fit power-law exponent via MLE: P(x) ∝ x^{-μ} for x ≥ x_min.

    Returns (mu, standard_error).
    """
    x = step_lengths[step_lengths >= x_min]
    n = len(x)
    if n < 10:
        return np.nan, np.nan
    log_ratio = np.log(x / x_min)
    denom = log_ratio.sum()
    if denom <= 0:
        return np.nan, np.nan
    mu = 1.0 + n / denom
    se = (mu - 1.0) / np.sqrt(n)
    return mu, se


def straightness_index(lats: np.ndarray, lons: np.ndarray) -> float:
    """
    Net displacement / total distance (0 = circular, 1 = straight line).
    """
    if len(lats) < 2:
        return np.nan
    # Step distances
    step_dists = haversine_nm(lats[:-1], lons[:-1], lats[1:], lons[1:])
    gross = step_dists.sum()
    if gross == 0:
        return np.nan
    net = haversine_nm(
        np.array([lats[0]]), np.array([lons[0]]),
        np.array([lats[-1]]), np.array([lons[-1]]),
    )[0]
    return float(net / gross)


def turning_angle_concentration(headings_rad: np.ndarray) -> float:
    """
    Mean resultant length of heading changes (turning angles).

    R → 1 means mostly straight, R → 0 means high turning variability.
    """
    if len(headings_rad) < 3:
        return np.nan
    # Turning angles = successive heading differences
    turns = np.diff(headings_rad)
    # Wrap to [-pi, pi]
    turns = (turns + np.pi) % (2 * np.pi) - np.pi
    C = np.cos(turns).mean()
    S = np.sin(turns).mean()
    return float(np.hypot(C, S))


def heading_from_positions(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Compute bearing (radians) between consecutive positions."""
    if len(lats) < 2:
        return np.array([])
    lat1 = np.radians(lats[:-1])
    lat2 = np.radians(lats[1:])
    dlon = np.radians(lons[1:] - lons[:-1])
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.arctan2(x, y)


def revisit_rate(
    lats: np.ndarray,
    lons: np.ndarray,
    cell_size_deg: float = 0.5,
) -> float:
    """
    Fraction of positions in previously-visited grid cells.
    """
    if len(lats) < 2:
        return np.nan
    cells_lat = np.floor(lats / cell_size_deg).astype(int)
    cells_lon = np.floor(lons / cell_size_deg).astype(int)
    visited = set()
    revisits = 0
    for i in range(len(lats)):
        cell = (cells_lat[i], cells_lon[i])
        if cell in visited:
            revisits += 1
        visited.add(cell)
    return revisits / len(lats)


def average_daily_distance(
    lats: np.ndarray,
    lons: np.ndarray,
    dates: np.ndarray,
) -> float:
    """Average daily distance in nautical miles."""
    if len(lats) < 2:
        return np.nan
    step_dists = haversine_nm(lats[:-1], lons[:-1], lats[1:], lons[1:])
    # Days between observations
    dates = pd.to_datetime(dates)
    day_diffs = np.diff(dates).astype("timedelta64[D]").astype(float)
    day_diffs = np.clip(day_diffs, 1, None)  # avoid division by zero
    daily = step_dists / day_diffs
    return float(np.nanmean(daily))


def first_passage_time(
    lats: np.ndarray,
    lons: np.ndarray,
    radius_nm: float = 50.0,
) -> float:
    """
    Mean first-passage time: average number of steps to exit a circle
    of given radius centered on the starting position.

    Computed over multiple starting points.
    """
    if len(lats) < 10:
        return np.nan
    n = len(lats)
    fpt_values = []
    # Sample starting points every 10 positions
    for start in range(0, n - 5, 10):
        for t in range(start + 1, min(start + 100, n)):
            d = haversine_nm(
                np.array([lats[start]]), np.array([lons[start]]),
                np.array([lats[t]]), np.array([lons[t]]),
            )[0]
            if d > radius_nm:
                fpt_values.append(t - start)
                break
    if not fpt_values:
        return np.nan
    return float(np.mean(fpt_values))


def local_redundancy(
    lats: np.ndarray,
    lons: np.ndarray,
    cell_size_deg: float = 0.5,
) -> float:
    """
    Fraction of total positions spent in cells that have already been
    visited. Higher = more area-restricted search.
    """
    return revisit_rate(lats, lons, cell_size_deg)


# ═══════════════════════════════════════════════════════════════════════════
# Aggregate Metric Computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_all_metrics(
    lats: np.ndarray,
    lons: np.ndarray,
    dates: np.ndarray = None,
) -> Dict[str, float]:
    """
    Compute all search metrics for a trajectory segment.

    Parameters
    ----------
    lats, lons : arrays of positions
    dates : array of dates (optional, for daily distance)

    Returns
    -------
    dict of metric name → value
    """
    step_dists = haversine_nm(lats[:-1], lons[:-1], lats[1:], lons[1:]) if len(lats) > 1 else np.array([])
    headings = heading_from_positions(lats, lons)

    mu, mu_se = levy_exponent(step_dists, x_min=1.0)

    metrics = {
        "levy_mu": mu,
        "levy_mu_se": mu_se,
        "straightness_index": straightness_index(lats, lons),
        "turning_concentration": turning_angle_concentration(headings),
        "revisit_rate": revisit_rate(lats, lons),
        "first_passage_time_50nm": first_passage_time(lats, lons, 50.0),
        "local_redundancy": local_redundancy(lats, lons),
        "n_positions": len(lats),
        "total_distance_nm": float(step_dists.sum()) if len(step_dists) > 0 else np.nan,
    }

    if dates is not None and len(dates) > 1:
        metrics["avg_daily_distance_nm"] = average_daily_distance(lats, lons, dates)

    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# Voyage-Level Metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_voyage_search_metrics(
    positions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute search metrics at the voyage level.

    Parameters
    ----------
    positions : pd.DataFrame
        Daily positions with voyage_id, lat, lon, obs_date.

    Returns
    -------
    pd.DataFrame
        One row per voyage with all search metrics.
    """
    records = []
    for vid, vdf in positions.groupby(COLS.voyage_id, sort=False):
        vdf = vdf.sort_values("obs_date")
        lats = vdf[COLS.lat].dropna().values
        lons = vdf[COLS.lon].dropna().values
        dates = vdf["obs_date"].values

        if len(lats) < 10:
            continue

        metrics = compute_all_metrics(lats, lons, dates)
        metrics[COLS.voyage_id] = vid
        records.append(metrics)

    result = pd.DataFrame(records)
    logger.info("Computed voyage-level search metrics for %d voyages", len(result))
    return result


def compute_spell_search_metrics(
    positions: pd.DataFrame,
    spells: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute search metrics at the ground-spell level.

    Parameters
    ----------
    positions : pd.DataFrame
        Daily positions.
    spells : pd.DataFrame
        Ground spell panel with entry_date, exit_date.

    Returns
    -------
    pd.DataFrame
        One row per spell with all search metrics.
    """
    pos = positions.copy()
    pos["obs_date"] = pd.to_datetime(pos["obs_date"])

    records = []
    for _, spell in spells.iterrows():
        vid = spell[COLS.voyage_id]
        entry = pd.to_datetime(spell["entry_date"])
        exit_ = pd.to_datetime(spell["exit_date"])

        mask = (
            (pos[COLS.voyage_id] == vid)
            & (pos["obs_date"] >= entry)
            & (pos["obs_date"] <= exit_)
        )
        spos = pos.loc[mask].sort_values("obs_date")

        if len(spos) < 5:
            continue

        lats = spos[COLS.lat].dropna().values
        lons = spos[COLS.lon].dropna().values
        dates = spos["obs_date"].values

        metrics = compute_all_metrics(lats, lons, dates)
        metrics["spell_id"] = spell["spell_id"]
        metrics[COLS.voyage_id] = vid
        records.append(metrics)

    result = pd.DataFrame(records)
    logger.info("Computed spell-level search metrics for %d spells", len(result))
    return result
