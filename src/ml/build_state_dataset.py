"""
ML Layer — State Dataset Builder.

Unit: ordered trajectory window for latent state modelling.

Computes rolling-window features from the action dataset:
- Mean/variance of speed, turn angle
- Revisit rate, local redundancy
- Productivity indicators
- Time since last success
- Patch residence duration
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import pandas as pd

from src.ml.config import ML_CFG, ML_DATA_DIR

logger = logging.getLogger(__name__)

OUTPUT_PATH = ML_DATA_DIR / "state_dataset.parquet"


def build_state_dataset(
    action_df: pd.DataFrame = None,
    *,
    window_size: int = 7,
    min_window_obs: int = 3,
    force_rebuild: bool = False,
    save: bool = True,
) -> pd.DataFrame:
    """
    Build state dataset from rolling trajectory windows.

    Parameters
    ----------
    action_df : pd.DataFrame
        Action dataset. Loaded from cache if None.
    window_size : int
        Rolling window size in days.
    min_window_obs : int
        Minimum non-NaN observations in a window.
    force_rebuild : bool
        Force rebuild even if cache exists.
    save : bool
        Save to parquet.

    Returns
    -------
    pd.DataFrame
        One row per valid trajectory window.
    """
    if OUTPUT_PATH.exists() and not force_rebuild:
        logger.info("Loading cached state dataset from %s", OUTPUT_PATH)
        return pd.read_parquet(OUTPUT_PATH)

    t0 = time.time()
    logger.info("Building state dataset (window=%d)...", window_size)

    if action_df is None:
        from src.ml.build_action_dataset import build_action_dataset
        action_df = build_action_dataset()

    # Only use active search observations for state modelling
    df = action_df.copy()
    if "active_search_flag" in df.columns:
        df = df[df["active_search_flag"] == 1].copy()

    df = df.sort_values(["voyage_id", "obs_date"]).reset_index(drop=True)

    # ── Rolling window features ─────────────────────────────────────
    grouped = df.groupby("voyage_id")

    features = pd.DataFrame(index=df.index)

    # Speed statistics
    if "speed" in df.columns:
        features["avg_speed"] = grouped["speed"].transform(
            lambda s: s.rolling(window_size, min_periods=min_window_obs).mean()
        )
        features["var_speed"] = grouped["speed"].transform(
            lambda s: s.rolling(window_size, min_periods=min_window_obs).var()
        )

    # Move length statistics
    if "move_length" in df.columns:
        features["avg_move_length"] = grouped["move_length"].transform(
            lambda s: s.rolling(window_size, min_periods=min_window_obs).mean()
        )
        features["var_move_length"] = grouped["move_length"].transform(
            lambda s: s.rolling(window_size, min_periods=min_window_obs).var()
        )

    # Turn angle statistics
    if "turn_angle" in df.columns:
        features["avg_turn_angle"] = grouped["turn_angle"].transform(
            lambda s: s.rolling(window_size, min_periods=min_window_obs).mean()
        )
        features["var_turn_angle"] = grouped["turn_angle"].transform(
            lambda s: s.rolling(window_size, min_periods=min_window_obs).var()
        )

    # Revisit measures
    if "revisit_indicator" in df.columns:
        features["revisit_rate"] = grouped["revisit_indicator"].transform(
            lambda s: s.rolling(window_size, min_periods=min_window_obs).mean()
        )

    # Productivity indicators
    if "consecutive_empty_days" in df.columns:
        features["max_empty_streak_window"] = grouped["consecutive_empty_days"].transform(
            lambda s: s.rolling(window_size, min_periods=min_window_obs).max()
        )

    if "days_since_last_success" in df.columns:
        features["time_since_success"] = df["days_since_last_success"]

    # Patch residence
    if "days_in_ground" in df.columns:
        features["patch_residence"] = df["days_in_ground"]

    if "days_in_patch" in df.columns:
        features["patch_duration"] = df["days_in_patch"]

    # ── Loop / local measures ───────────────────────────────────────
    if "net_displacement" in df.columns and "move_length" in df.columns:
        cumul_dist = grouped["move_length"].transform(
            lambda s: s.rolling(window_size, min_periods=min_window_obs).sum()
        )
        # Rolling net displacement approximation
        features["local_loop_ratio"] = np.where(
            cumul_dist > 0,
            1 - (df["net_displacement"] / cumul_dist.clip(lower=1)),
            0,
        )

    # ── Combine with identifiers from action dataset ────────────────
    id_cols = [
        "voyage_id", "captain_id", "agent_id", "vessel_id",
        "obs_date", "date", "year", "ground_id", "patch_id",
        "lat", "lon", "voyage_day",
        "theta_hat_holdout", "psi_hat_holdout",
        "active_search_flag", "transit_flag", "homebound_flag",
        "scarcity", "novice",
    ]
    available_ids = [c for c in id_cols if c in df.columns]

    result = pd.concat([df[available_ids].reset_index(drop=True),
                        features.reset_index(drop=True)], axis=1)

    # Drop rows with all NaN features
    feature_cols = [c for c in features.columns]
    result = result.dropna(subset=feature_cols, how="all").reset_index(drop=True)

    elapsed = time.time() - t0
    logger.info(
        "State dataset built: %d rows, %d columns, %.1fs",
        len(result), len(result.columns), elapsed,
    )

    if save:
        result.to_parquet(OUTPUT_PATH, index=False)
        logger.info("Saved to %s", OUTPUT_PATH)

    return result
