"""
Compass Pipeline — Data I/O & Validation (Step 1).

Loads trajectory point and voyage metadata tables, enforces dtypes, drops
duplicates and impossible coordinates, flags time gaps, and ensures each
voyage meets the minimum-points threshold.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from compass.config import CompassConfig

logger = logging.getLogger(__name__)

# ── required / optional column sets ─────────────────────────────────────────

_TRAJ_REQUIRED = {"voyage_id", "timestamp_utc", "lat", "lon"}
_TRAJ_OPTIONAL = {
    "captain_id", "agent_id", "firm_env_id", "ground_id",
    "state_time_cell_id", "catch_event_flag", "catch_timestamp_utc",
    "speed_knots", "heading_degrees",
}

_META_REQUIRED = {"voyage_id"}
_META_OPTIONAL = {
    "captain_id", "agent_id", "firm_env_id", "ground_id",
    "departure_timestamp_utc", "arrival_ground_timestamp_utc",
    "return_timestamp_utc", "ship_id", "port_id", "year",
}


# ── loaders ─────────────────────────────────────────────────────────────────

def _read_file(path: str | Path) -> pd.DataFrame:
    """Read parquet or CSV."""
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_trajectory_points(
    path: str | Path,
    cfg: Optional[CompassConfig] = None,
) -> pd.DataFrame:
    """
    Load and type-enforce *trajectory_points*.

    Parameters
    ----------
    path : str | Path
        File path (parquet or csv).
    cfg : CompassConfig, optional

    Returns
    -------
    pd.DataFrame
        With at least the required columns, timestamps tz-aware UTC.
    """
    df = _read_file(path)

    missing = _TRAJ_REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"trajectory_points missing columns: {missing}")

    # enforce types
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce").astype("float64")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce").astype("float64")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["voyage_id"] = df["voyage_id"].astype(str)

    # optional timestamp columns
    for col in ("catch_timestamp_utc",):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    logger.info(
        "Loaded trajectory_points: %d rows, %d voyages.",
        len(df), df["voyage_id"].nunique(),
    )
    return df


def load_voyage_metadata(path: str | Path) -> pd.DataFrame:
    """Load *voyage_metadata* with dtype enforcement."""
    df = _read_file(path)

    missing = _META_REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"voyage_metadata missing columns: {missing}")

    df["voyage_id"] = df["voyage_id"].astype(str)

    for col in (
        "departure_timestamp_utc",
        "arrival_ground_timestamp_utc",
        "return_timestamp_utc",
    ):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    logger.info(
        "Loaded voyage_metadata: %d voyages.", df["voyage_id"].nunique(),
    )
    return df


# ── validation ──────────────────────────────────────────────────────────────

def validate_trajectories(
    df: pd.DataFrame,
    cfg: CompassConfig,
) -> pd.DataFrame:
    """
    Clean and validate trajectory points.

    Operations (in order):
    1. Drop exact duplicates.
    2. Drop rows with NaN lat/lon or impossible coords.
    3. Sort by (voyage_id, timestamp_utc).
    4. Compute time deltas; flag gaps > threshold.
    5. Drop voyages with fewer than *minimum_points_per_voyage* points.

    Returns a copy with added columns:
    ``dt_seconds``, ``dt_hours``, ``gap_flag``.
    """
    n0 = len(df)

    # 1. duplicates
    df = df.drop_duplicates(subset=["voyage_id", "timestamp_utc", "lat", "lon"])
    logger.info("Dropped %d duplicate rows.", n0 - len(df))

    # 2. impossible coords
    valid = (
        df["lat"].between(-90, 90)
        & df["lon"].between(-180, 180)
        & df["lat"].notna()
        & df["lon"].notna()
    )
    n_bad = (~valid).sum()
    if n_bad:
        logger.warning("Dropping %d rows with impossible coordinates.", n_bad)
    df = df.loc[valid].copy()

    # 3. sort
    df.sort_values(["voyage_id", "timestamp_utc"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 4. time deltas
    df["dt_seconds"] = (
        df.groupby("voyage_id")["timestamp_utc"]
        .diff()
        .dt.total_seconds()
    )
    df["dt_hours"] = df["dt_seconds"] / 3600.0
    df["gap_flag"] = df["dt_hours"] > cfg.gap_threshold_hours

    n_gaps = df["gap_flag"].sum()
    if n_gaps:
        logger.info(
            "%d time gaps > %.0f h detected.", n_gaps, cfg.gap_threshold_hours,
        )

    # 5. minimum points
    counts = df.groupby("voyage_id")["lat"].transform("count")
    too_short = counts < cfg.minimum_points_per_voyage
    n_drop_voyages = df.loc[too_short, "voyage_id"].nunique()
    if n_drop_voyages:
        logger.warning(
            "Dropping %d voyages with < %d points.",
            n_drop_voyages,
            cfg.minimum_points_per_voyage,
        )
    df = df.loc[~too_short].copy()

    logger.info(
        "Validated trajectory: %d rows, %d voyages.",
        len(df), df["voyage_id"].nunique(),
    )
    return df
