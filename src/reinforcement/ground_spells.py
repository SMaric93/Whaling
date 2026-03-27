"""
Reinforcement Test Suite — Ground Spell Construction.

Builds voyage-ground spells from daily logbook positions:
- Classifies each position into whaling grounds
- Groups consecutive same-ground days into spells
- Computes spell-level metrics (duration, productivity, encounter stats)
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
# Whaling Ground Definitions
# ═══════════════════════════════════════════════════════════════════════════

# Approximate bounding boxes for major whaling grounds.
# Source: historical whaling ground classifications (Townsend 1935, Lund 2001).
WHALING_GROUNDS = {
    "atlantic_north": {"lat_min": 30, "lat_max": 60, "lon_min": -80, "lon_max": -10},
    "atlantic_south": {"lat_min": -60, "lat_max": -10, "lon_min": -70, "lon_max": 20},
    "pacific_north": {"lat_min": 20, "lat_max": 60, "lon_min": -180, "lon_max": -100},
    "pacific_south": {"lat_min": -60, "lat_max": -10, "lon_min": -180, "lon_max": -70},
    "indian_ocean": {"lat_min": -50, "lat_max": 10, "lon_min": 20, "lon_max": 120},
    "western_pacific": {"lat_min": -50, "lat_max": 40, "lon_min": 100, "lon_max": 180},
    "arctic": {"lat_min": 60, "lat_max": 90, "lon_min": -180, "lon_max": 180},
    "cape_horn": {"lat_min": -60, "lat_max": -50, "lon_min": -80, "lon_max": -60},
    "japan_ground": {"lat_min": 25, "lat_max": 45, "lon_min": 130, "lon_max": 180},
    "kodiak_ground": {"lat_min": 50, "lat_max": 62, "lon_min": -170, "lon_max": -140},
    "on_the_line": {"lat_min": -5, "lat_max": 5, "lon_min": -180, "lon_max": -80},
}


def classify_ground(lat: float, lon: float) -> Optional[str]:
    """
    Classify a position into a whaling ground.

    Returns ground name or None if not in any defined ground.
    Uses most specific match (smallest area) if overlapping.
    """
    if pd.isna(lat) or pd.isna(lon):
        return None

    matches = []
    for name, bounds in WHALING_GROUNDS.items():
        if (bounds["lat_min"] <= lat <= bounds["lat_max"]
                and bounds["lon_min"] <= lon <= bounds["lon_max"]):
            area = (
                (bounds["lat_max"] - bounds["lat_min"])
                * (bounds["lon_max"] - bounds["lon_min"])
            )
            matches.append((name, area))

    if not matches:
        return None

    # Return most specific (smallest area) match
    matches.sort(key=lambda x: x[1])
    return matches[0][0]


def classify_ground_vectorized(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Vectorized ground classification for arrays."""
    result = np.full(len(lats), None, dtype=object)

    for name, bounds in WHALING_GROUNDS.items():
        in_ground = (
            (lats >= bounds["lat_min"])
            & (lats <= bounds["lat_max"])
            & (lons >= bounds["lon_min"])
            & (lons <= bounds["lon_max"])
        )
        # Only assign if not already assigned to a more specific ground
        mask = in_ground & pd.isna(pd.Series(result))
        # We'll handle specificity by sorting grounds smallest-first
        pass

    # Simpler: apply per-row
    for i in range(len(lats)):
        result[i] = classify_ground(lats[i], lons[i])

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Spell Construction
# ═══════════════════════════════════════════════════════════════════════════

def build_ground_spells(
    positions: pd.DataFrame,
    *,
    min_spell_days: int = None,
    max_gap_days: int = None,
) -> pd.DataFrame:
    """
    Build voyage-ground spells from daily logbook positions.

    A ground spell is a continuous period where a vessel stays within
    the same whaling ground. Short gaps (≤ max_gap_days) within the
    same ground are bridged.

    Parameters
    ----------
    positions : pd.DataFrame
        Daily positions with: voyage_id, obs_date, lat, lon,
        and optionally encounter, species, n_struck, n_tried.
    min_spell_days : int
        Minimum spell duration to keep. Default from config.
    max_gap_days : int
        Maximum gap (in days) to bridge within same ground.

    Returns
    -------
    pd.DataFrame
        Ground spell panel with columns:
        - voyage_id, ground, spell_id, spell_num
        - entry_date, exit_date, duration_days
        - n_positions, n_encounters, n_sightings, n_strikes
        - total_struck, total_tried
        - productive (bool), n_empty_days, max_empty_streak
        - cumulative_ground_days (within voyage)
    """
    min_spell_days = min_spell_days or CFG.min_ground_spell_days
    max_gap_days = max_gap_days or CFG.max_ground_transit_gap_days

    # Ensure sorted
    pos = positions.copy()
    pos["obs_date"] = pd.to_datetime(pos["obs_date"])
    pos = pos.sort_values([COLS.voyage_id, "obs_date"])

    # Classify grounds
    pos["ground"] = classify_ground_vectorized(
        pos[COLS.lat].values, pos[COLS.lon].values
    )

    # Encounter flags (if available)
    has_encounters = COLS.encounter in pos.columns
    if has_encounters:
        pos["is_encounter"] = (
            pos[COLS.encounter].notna()
            & (pos[COLS.encounter] != "NoEnc")
        )
        pos["is_sighting"] = pos[COLS.encounter] == "Sight"
        pos["is_strike"] = pos[COLS.encounter] == "Strike"
    else:
        pos["is_encounter"] = False
        pos["is_sighting"] = False
        pos["is_strike"] = False

    spells = []

    for voyage_id, vdf in pos.groupby(COLS.voyage_id, sort=False):
        vdf = vdf.sort_values("obs_date").reset_index(drop=True)

        if len(vdf) == 0 or vdf["ground"].isna().all():
            continue

        # Identify spell boundaries
        ground_changes = vdf["ground"] != vdf["ground"].shift(1)
        # Also break on large gaps
        date_gaps = vdf["obs_date"].diff().dt.days > max_gap_days
        spell_breaks = ground_changes | date_gaps
        vdf["spell_group"] = spell_breaks.cumsum()

        spell_num = 0
        cumulative_days = 0

        for spell_group, sdf in vdf.groupby("spell_group"):
            ground = sdf["ground"].mode()
            if len(ground) == 0 or pd.isna(ground.iloc[0]):
                continue
            ground = ground.iloc[0]

            entry = sdf["obs_date"].min()
            exit_ = sdf["obs_date"].max()
            duration = (exit_ - entry).days + 1

            if duration < min_spell_days:
                continue

            spell_num += 1
            cumulative_days += duration

            # Encounter stats
            n_enc = int(sdf["is_encounter"].sum())
            n_sight = int(sdf["is_sighting"].sum())
            n_strike = int(sdf["is_strike"].sum())
            total_struck = int(sdf.get(COLS.n_struck, pd.Series(0)).fillna(0).sum())
            total_tried = int(sdf.get(COLS.n_tried, pd.Series(0)).fillna(0).sum())

            # Empty day streaks
            empty_days = (~sdf["is_encounter"]).astype(int).values
            n_empty = int(empty_days.sum())
            max_streak = _max_consecutive(empty_days)

            spells.append({
                COLS.voyage_id: voyage_id,
                "ground": ground,
                "spell_id": f"{voyage_id}_{ground}_{spell_num}",
                "spell_num": spell_num,
                "entry_date": entry,
                "exit_date": exit_,
                "duration_days": duration,
                "n_positions": len(sdf),
                "n_encounters": n_enc,
                "n_sightings": n_sight,
                "n_strikes": n_strike,
                "total_struck": total_struck,
                "total_tried": total_tried,
                "productive": n_strike > 0,
                "n_empty_days": n_empty,
                "max_empty_streak": max_streak,
                "cumulative_ground_days": cumulative_days,
            })

    result = pd.DataFrame(spells)
    logger.info(
        "Built %d ground spells from %d voyages (%.1f spells/voyage)",
        len(result),
        result[COLS.voyage_id].nunique() if len(result) > 0 else 0,
        len(result) / max(result[COLS.voyage_id].nunique(), 1) if len(result) > 0 else 0,
    )
    return result


def _max_consecutive(arr: np.ndarray) -> int:
    """Maximum consecutive run of 1s in a binary array."""
    if len(arr) == 0:
        return 0
    max_run = 0
    current = 0
    for v in arr:
        if v == 1:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run


# ═══════════════════════════════════════════════════════════════════════════
# Spell-Level Enrichment
# ═══════════════════════════════════════════════════════════════════════════

def enrich_spells_with_voyage_data(
    spells: pd.DataFrame,
    voyages: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge voyage-level metadata (captain_id, agent_id, theta, psi, etc.)
    into the ground spell panel.
    """
    merge_cols = [
        COLS.voyage_id, COLS.captain_id, COLS.agent_id, COLS.vessel_id,
        COLS.year_out, COLS.home_port, COLS.ground_or_route,
        COLS.log_q, COLS.q_oil_bbl,
    ]
    # Include AKM effects if available
    for col in ["theta", "psi", "theta_quintile", "psi_quintile",
                 "novice", "expert", COLS.captain_experience, "decade",
                 "vessel_period", COLS.route_time, COLS.switch_agent]:
        if col in voyages.columns:
            merge_cols.append(col)

    merge_cols = [c for c in merge_cols if c in voyages.columns]
    merge_cols = list(dict.fromkeys(merge_cols))  # dedupe preserving order

    enriched = spells.merge(
        voyages[merge_cols],
        on=COLS.voyage_id,
        how="left",
    )
    logger.info("Enriched %d spells with voyage data", len(enriched))
    return enriched


# ═══════════════════════════════════════════════════════════════════════════
# Sparse / Rich Classification
# ═══════════════════════════════════════════════════════════════════════════

def classify_ground_scarcity(
    spells: pd.DataFrame,
    *,
    method: str = "encounter_rate",
) -> pd.DataFrame:
    """
    Tag ground-spells as sparse or rich based on encounter rates.

    Parameters
    ----------
    spells : pd.DataFrame
        Ground spell panel (enriched with voyage data).
    method : str
        "encounter_rate" — below/above median encounters per day
        "decade_ground" — uses ground × decade median

    Returns
    -------
    pd.DataFrame with "scarcity" and "sparse" columns added.
    """
    df = spells.copy()
    df["encounter_rate"] = df["n_encounters"] / df["duration_days"].clip(lower=1)

    if method == "decade_ground" and "decade" in df.columns:
        medians = df.groupby(["ground", "decade"])["encounter_rate"].transform("median")
        df["scarcity"] = -df["encounter_rate"]  # higher = scarcer
        df["sparse"] = (df["encounter_rate"] <= medians).astype(int)
    else:
        median_rate = df["encounter_rate"].median()
        df["scarcity"] = -df["encounter_rate"]
        df["sparse"] = (df["encounter_rate"] <= median_rate).astype(int)

    logger.info(
        "Scarcity classification: %d sparse, %d rich",
        df["sparse"].sum(), (1 - df["sparse"]).sum(),
    )
    return df
