"""
Reinforcement Test Suite — Patch Spell Construction.

Builds local patch visits within whaling grounds:
- Groups positions into spatial patches via sequential radius clustering
- Computes patch-level encounter stats and stopping-rule metrics
- Multiple radius definitions for robustness
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import CFG, COLS
from .utils import haversine_nm

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Patch Construction
# ═══════════════════════════════════════════════════════════════════════════

def build_patch_spells(
    positions: pd.DataFrame,
    *,
    radius_nm: float = None,
    min_duration_days: int = None,
) -> pd.DataFrame:
    """
    Build patch spells from daily logbook positions.

    A patch is a contiguous cluster of positions within `radius_nm`
    of the patch centroid. When the vessel moves beyond the radius,
    the current patch ends and a new one begins.

    Parameters
    ----------
    positions : pd.DataFrame
        Daily positions with: voyage_id, obs_date, lat, lon,
        encounter, species, n_struck, n_tried.
    radius_nm : float
        Patch radius in nautical miles.
    min_duration_days : int
        Minimum patch duration to keep.

    Returns
    -------
    pd.DataFrame
        Patch spell panel with:
        - voyage_id, patch_id, ground, centroid_lat, centroid_lon
        - entry_date, exit_date, duration_days, n_positions
        - n_encounters, n_sightings, n_strikes, total_struck
        - n_empty_days, max_empty_streak, days_since_last_success
        - productive (bool), encounter_rate
    """
    radius_nm = radius_nm or CFG.default_patch_radius_nm
    min_duration_days = min_duration_days or CFG.min_patch_duration_days

    pos = positions.copy()
    pos["obs_date"] = pd.to_datetime(pos["obs_date"])
    pos = pos.sort_values([COLS.voyage_id, "obs_date"])

    # Encounter flags
    has_enc = COLS.encounter in pos.columns
    if has_enc:
        pos["is_encounter"] = (
            pos[COLS.encounter].notna()
            & (pos[COLS.encounter] != "NoEnc")
        )
        pos["is_strike"] = pos[COLS.encounter] == "Strike"
        pos["is_sighting"] = pos[COLS.encounter] == "Sight"
    else:
        pos["is_encounter"] = False
        pos["is_strike"] = False
        pos["is_sighting"] = False

    patches = []

    for voyage_id, vdf in pos.groupby(COLS.voyage_id, sort=False):
        vdf = vdf.sort_values("obs_date").reset_index(drop=True)
        if len(vdf) < 2:
            continue

        lats = vdf[COLS.lat].values
        lons = vdf[COLS.lon].values
        dates = vdf["obs_date"].values
        encounters = vdf["is_encounter"].values
        strikes = vdf["is_strike"].values
        sightings = vdf["is_sighting"].values
        n_struck_arr = vdf.get(COLS.n_struck, pd.Series(0, index=vdf.index)).fillna(0).values
        n_tried_arr = vdf.get(COLS.n_tried, pd.Series(0, index=vdf.index)).fillna(0).values

        # Sequential radius clustering
        patch_labels = _sequential_cluster(lats, lons, radius_nm)
        vdf["patch_label"] = patch_labels

        last_success_date = None
        patch_num = 0

        for label, pdf in vdf.groupby("patch_label"):
            if pd.isna(label) or label < 0:
                continue

            entry = pdf["obs_date"].min()
            exit_ = pdf["obs_date"].max()
            duration = (exit_ - entry).days + 1

            if duration < min_duration_days:
                continue

            patch_num += 1
            centroid_lat = pdf[COLS.lat].mean()
            centroid_lon = pdf[COLS.lon].mean()

            # Encounter stats
            p_enc = pdf["is_encounter"].values
            p_strike = pdf["is_strike"].values
            p_sight = pdf["is_sighting"].values
            n_enc = int(p_enc.sum())
            n_strikes = int(p_strike.sum())
            n_sights = int(p_sight.sum())
            total_struck = int(pdf.get(COLS.n_struck, pd.Series(0)).fillna(0).sum())
            total_tried = int(pdf.get(COLS.n_tried, pd.Series(0)).fillna(0).sum())

            # Empty day analysis
            empty_flags = (~p_enc).astype(int)
            n_empty = int(empty_flags.sum())
            max_streak = _max_consecutive(empty_flags)

            # Days since last success
            if last_success_date is not None:
                days_since = (entry - last_success_date).days
            else:
                days_since = np.nan

            # Update last success
            if n_strikes > 0:
                strike_dates = pdf.loc[pdf["is_strike"], "obs_date"]
                last_success_date = strike_dates.max()

            # Ground classification
            from .ground_spells import classify_ground
            ground = classify_ground(centroid_lat, centroid_lon)

            patches.append({
                COLS.voyage_id: voyage_id,
                "patch_id": f"{voyage_id}_p{patch_num}",
                "patch_num": patch_num,
                "ground": ground,
                "centroid_lat": centroid_lat,
                "centroid_lon": centroid_lon,
                "entry_date": entry,
                "exit_date": exit_,
                "duration_days": duration,
                "n_positions": len(pdf),
                "n_encounters": n_enc,
                "n_sightings": n_sights,
                "n_strikes": n_strikes,
                "total_struck": total_struck,
                "total_tried": total_tried,
                "productive": n_strikes > 0,
                "n_empty_days": n_empty,
                "max_empty_streak": max_streak,
                "days_since_last_success": days_since,
                "encounter_rate": n_enc / max(duration, 1),
                "radius_nm": radius_nm,
            })

    result = pd.DataFrame(patches)
    logger.info(
        "Built %d patch spells (radius=%.0fnm) from %d voyages",
        len(result), radius_nm,
        result[COLS.voyage_id].nunique() if len(result) > 0 else 0,
    )
    return result


def _sequential_cluster(
    lats: np.ndarray,
    lons: np.ndarray,
    radius_nm: float,
) -> np.ndarray:
    """
    Sequential radius clustering: assign patch labels based on distance
    from running centroid.

    When distance from current centroid > radius, start a new patch.
    """
    n = len(lats)
    labels = np.full(n, -1, dtype=int)

    if n == 0:
        return labels

    current_patch = 0
    centroid_lat = lats[0]
    centroid_lon = lons[0]
    patch_count = 1
    labels[0] = current_patch

    for i in range(1, n):
        if pd.isna(lats[i]) or pd.isna(lons[i]):
            labels[i] = -1
            continue

        dist = haversine_nm(
            np.array([centroid_lat]),
            np.array([centroid_lon]),
            np.array([lats[i]]),
            np.array([lons[i]]),
        )[0]

        if dist <= radius_nm:
            labels[i] = current_patch
            # Update running centroid
            centroid_lat = (centroid_lat * patch_count + lats[i]) / (patch_count + 1)
            centroid_lon = (centroid_lon * patch_count + lons[i]) / (patch_count + 1)
            patch_count += 1
        else:
            current_patch += 1
            centroid_lat = lats[i]
            centroid_lon = lons[i]
            patch_count = 1
            labels[i] = current_patch

    return labels


def _max_consecutive(arr: np.ndarray) -> int:
    """Maximum consecutive run of 1s."""
    if len(arr) == 0:
        return 0
    max_run = current = 0
    for v in arr:
        if v == 1:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run


# ═══════════════════════════════════════════════════════════════════════════
# Multi-Radius Robustness
# ═══════════════════════════════════════════════════════════════════════════

def build_patch_spells_multi_radius(
    positions: pd.DataFrame,
    radii: List[float] = None,
) -> pd.DataFrame:
    """
    Build patch spells at multiple radius definitions for robustness.

    Returns a stacked DataFrame with a `radius_nm` column.
    """
    radii = radii or CFG.patch_radii_nm
    parts = []
    for r in radii:
        df = build_patch_spells(positions, radius_nm=r)
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# Patch-Day Panel (for hazard models)
# ═══════════════════════════════════════════════════════════════════════════

def expand_to_patch_days(
    patches: pd.DataFrame,
    positions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Expand patch spells to patch-day observations for hazard models.

    Each row = one day within a patch, with:
    - day_in_patch (1, 2, 3, ...)
    - exit_tomorrow (0/1)
    - encounter_today (0/1)
    - strike_today (0/1)
    - consecutive_empty_days (running count)
    - days_since_last_strike (running count)

    Parameters
    ----------
    patches : pd.DataFrame
        Patch spell panel.
    positions : pd.DataFrame
        Daily positions with encounter data.

    Returns
    -------
    pd.DataFrame
        Patch-day panel ready for hazard model estimation.
    """
    pos = positions.copy()
    pos["obs_date"] = pd.to_datetime(pos["obs_date"])

    has_enc = COLS.encounter in pos.columns

    patch_days = []

    for _, patch in patches.iterrows():
        vid = patch[COLS.voyage_id]
        entry = pd.to_datetime(patch["entry_date"])
        exit_ = pd.to_datetime(patch["exit_date"])

        # Get positions for this voyage within the patch date range
        mask = (
            (pos[COLS.voyage_id] == vid)
            & (pos["obs_date"] >= entry)
            & (pos["obs_date"] <= exit_)
        )
        ppos = pos.loc[mask].sort_values("obs_date")

        if len(ppos) == 0:
            continue

        consec_empty = 0
        days_since_strike = np.nan
        last_strike_seen = False

        for day_num, (_, row) in enumerate(ppos.iterrows(), 1):
            is_last = day_num == len(ppos)

            if has_enc:
                enc_today = row.get(COLS.encounter, "NoEnc") not in (None, "NoEnc")
                strike_today = row.get(COLS.encounter, "") == "Strike"
            else:
                enc_today = False
                strike_today = False

            if enc_today:
                consec_empty = 0
            else:
                consec_empty += 1

            if strike_today:
                days_since_strike = 0
                last_strike_seen = True
            elif last_strike_seen:
                days_since_strike += 1

            patch_days.append({
                COLS.voyage_id: vid,
                "patch_id": patch["patch_id"],
                "obs_date": row["obs_date"],
                "day_in_patch": day_num,
                "exit_tomorrow": int(is_last),
                "encounter_today": int(enc_today),
                "strike_today": int(strike_today),
                "consecutive_empty_days": consec_empty,
                "days_since_last_strike": days_since_strike,
                "duration_days": patch["duration_days"],
                "ground": patch.get("ground"),
            })

    result = pd.DataFrame(patch_days)
    logger.info(
        "Expanded %d patches to %d patch-day observations",
        len(patches), len(result),
    )
    return result
