"""
Reinforcement Test Suite — Type Measures & Experience Builders.

Constructs captain experience, novice/expert bins, scarcity measures,
and fixed-effect cell identifiers.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .config import CFG, COLS

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Scarcity Measures
# ═══════════════════════════════════════════════════════════════════════════

def compute_scarcity_index(
    voyages: pd.DataFrame,
    *,
    method: str = "ground_decade",
) -> pd.DataFrame:
    """
    Compute scarcity proxy for each voyage based on whale density.

    Parameters
    ----------
    voyages : pd.DataFrame
        Voyage panel with ground_or_route, year_out, and optionally
        Maury/Townsend sighting counts.
    method : str
        "ground_decade" — within ground×decade z-score of production
        "total_sightings" — from augmented data (maury_total_sightings)
        "encounter_rate" — from logbook encounter data

    Returns
    -------
    pd.DataFrame with scarcity_index and scarcity_bin columns added.
    """
    df = voyages.copy()

    if method == "total_sightings" and "maury_total_sightings" in df.columns:
        # Use Maury sighting data
        df["_raw_density"] = df["maury_total_sightings"].fillna(0)
        group = ["ground_or_route", "decade"] if "decade" in df.columns else ["ground_or_route"]
        median_density = df.groupby(group)["_raw_density"].transform("median")
        df["scarcity_index"] = -(df["_raw_density"] - median_density)
        df["scarcity_index"] = (
            df["scarcity_index"] - df["scarcity_index"].mean()
        ) / max(df["scarcity_index"].std(), 0.01)
    else:
        # Ground × decade average production as proxy
        df["decade"] = df.get("decade", (df[COLS.year_out] // 10) * 10)
        group_avg = df.groupby([COLS.ground_or_route, "decade"])[COLS.q_oil_bbl].transform("mean")
        df["scarcity_index"] = -(group_avg - group_avg.mean()) / max(group_avg.std(), 0.01)

    # Bin into terciles
    df["scarcity_bin"] = pd.qcut(
        df["scarcity_index"],
        q=CFG.n_scarcity_bins,
        labels=[f"S{i+1}" for i in range(CFG.n_scarcity_bins)],
        duplicates="drop",
    )

    logger.info(
        "Scarcity index computed (method=%s): mean=%.2f, std=%.2f",
        method,
        df["scarcity_index"].mean(),
        df["scarcity_index"].std(),
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Experience Bins
# ═══════════════════════════════════════════════════════════════════════════

def compute_experience_bins(
    df: pd.DataFrame,
    *,
    method: str = "voyage_count",
    n_bins: int = 3,
) -> pd.DataFrame:
    """
    Create experience tercile bins for captains.

    Parameters
    ----------
    df : pd.DataFrame
        Must have captain_experience or captain_voyage_num.
    method : str
        "voyage_count" or "years" (years since first voyage).
    n_bins : int
        Number of experience bins.

    Returns
    -------
    pd.DataFrame with experience_bin column added.
    """
    df = df.copy()

    if method == "years":
        first_year = df.groupby(COLS.captain_id)[COLS.year_out].transform("min")
        df["experience_years"] = df[COLS.year_out] - first_year
        exp_col = "experience_years"
    else:
        exp_col = COLS.captain_voyage_num

    df["experience_bin"] = pd.qcut(
        df[exp_col],
        q=n_bins,
        labels=[f"E{i+1}" for i in range(n_bins)],
        duplicates="drop",
    )

    logger.info(
        "Experience bins: %s",
        df["experience_bin"].value_counts().to_dict(),
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Capability Bins
# ═══════════════════════════════════════════════════════════════════════════

def compute_capability_bins(
    df: pd.DataFrame,
    *,
    n_bins: int = None,
    col: str = "psi",
) -> pd.DataFrame:
    """
    Create capability bins for agents based on psi.

    Parameters
    ----------
    df : pd.DataFrame
        Must have psi column.
    n_bins : int
        Number of capability bins.
    col : str
        Column to bin.

    Returns
    -------
    pd.DataFrame with high_psi and psi_bin columns added.
    """
    n_bins = n_bins or CFG.n_capability_bins
    df = df.copy()

    if col not in df.columns:
        logger.warning("Column '%s' not found, skipping capability bins", col)
        return df

    # Binary high/low
    median_psi = df[col].median()
    df["high_psi"] = (df[col] >= median_psi).astype(int)

    # Finer bins
    df["psi_bin"] = pd.qcut(
        df[col],
        q=n_bins,
        labels=[f"P{i+1}" for i in range(n_bins)],
        duplicates="drop",
    )

    logger.info(
        "Capability bins: high_psi=%d, low_psi=%d",
        df["high_psi"].sum(),
        (1 - df["high_psi"]).sum(),
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Mover / Stayer Classification
# ═══════════════════════════════════════════════════════════════════════════

def classify_movers_stayers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify captains as movers (switched agents) or stayers.

    Returns
    -------
    pd.DataFrame with:
    - is_mover: captain ever switched agents
    - n_agents: number of distinct agents
    - n_switches: number of agent switches
    """
    df = df.copy()

    captain_stats = df.groupby(COLS.captain_id).agg(
        n_agents=(COLS.agent_id, "nunique"),
        n_voyages=(COLS.voyage_id, "count"),
    ).reset_index()

    if COLS.switch_agent in df.columns:
        n_switches = df.groupby(COLS.captain_id)[COLS.switch_agent].sum().reset_index()
        n_switches.columns = [COLS.captain_id, "n_switches"]
        captain_stats = captain_stats.merge(n_switches, on=COLS.captain_id, how="left")
    else:
        captain_stats["n_switches"] = captain_stats["n_agents"] - 1

    captain_stats["is_mover"] = (captain_stats["n_agents"] > 1).astype(int)

    df = df.merge(
        captain_stats[[COLS.captain_id, "is_mover", "n_agents", "n_switches"]],
        on=COLS.captain_id,
        how="left",
    )

    n_movers = captain_stats["is_mover"].sum()
    n_stayers = len(captain_stats) - n_movers
    logger.info(
        "Movers: %d captains (%.1f%%), Stayers: %d",
        n_movers,
        100 * n_movers / max(len(captain_stats), 1),
        n_stayers,
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Switch-Sample Construction (for Test 1)
# ═══════════════════════════════════════════════════════════════════════════

def build_switch_sample(
    df: pd.DataFrame,
    *,
    require_same_ground: bool = True,
    min_ground_visits: int = 2,
) -> pd.DataFrame:
    """
    Construct the same-captain, same-ground, different-agent sample.

    Parameters
    ----------
    df : pd.DataFrame
        Voyage panel with captain_id, agent_id, ground_or_route.
    require_same_ground : bool
        Require captain visits same ground under different agents.
    min_ground_visits : int
        Minimum visits to same ground.

    Returns
    -------
    pd.DataFrame
        Subset of voyages for the switch design.
    """
    if not require_same_ground:
        # Just movers
        movers = df.groupby(COLS.captain_id)[COLS.agent_id].nunique()
        mover_ids = movers[movers > 1].index
        switch_df = df[df[COLS.captain_id].isin(mover_ids)].copy()
        logger.info("Switch sample (any ground): %d voyages, %d captains",
                     len(switch_df), switch_df[COLS.captain_id].nunique())
        return switch_df

    # Captain × ground combinations with multiple agents
    cg = df.groupby([COLS.captain_id, COLS.ground_or_route]).agg(
        n_agents=(COLS.agent_id, "nunique"),
        n_voyages=(COLS.voyage_id, "count"),
    ).reset_index()

    # Keep captain-ground pairs with ≥2 agents and ≥ min visits
    valid_cg = cg[
        (cg["n_agents"] >= 2) & (cg["n_voyages"] >= min_ground_visits)
    ]

    # Filter to these captain-ground pairs
    valid_keys = set(
        zip(valid_cg[COLS.captain_id], valid_cg[COLS.ground_or_route])
    )
    mask = df.apply(
        lambda r: (r[COLS.captain_id], r[COLS.ground_or_route]) in valid_keys,
        axis=1,
    )
    switch_df = df[mask].copy()

    # Create captain × ground FE
    switch_df["captain_ground"] = (
        switch_df[COLS.captain_id].astype(str)
        + "_"
        + switch_df[COLS.ground_or_route].fillna("UNK").astype(str)
    )

    logger.info(
        "Switch sample (same ground): %d voyages, %d captains, "
        "%d captain-ground pairs",
        len(switch_df),
        switch_df[COLS.captain_id].nunique(),
        len(valid_keys),
    )
    return switch_df
