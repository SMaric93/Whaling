"""
Reinforcement Test Suite — Master Data Builder.

Loads and merges all data sources into a single analysis-ready DataFrame:
- analysis_voyage.parquet (main voyage data)
- AKM captain/agent effects (theta/psi)
- Logbook features
- Switch indicators and experience measures
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import CFG, COLS, DATA_DIR, STAGING_DIR

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Core Voyage Panel
# ═══════════════════════════════════════════════════════════════════════════

def load_voyage_panel(
    *,
    include_climate: bool = False,
    include_augmented: bool = True,
) -> pd.DataFrame:
    """
    Load the main voyage-level panel with all controls.

    Parameters
    ----------
    include_climate : bool
        Use the climate-augmented version.
    include_augmented : bool
        Use the augmented version (Maury/Townsend sightings).

    Returns
    -------
    pd.DataFrame
        Voyage-level panel.
    """
    if include_climate:
        path = DATA_DIR / "analysis_voyage_with_climate.parquet"
    elif include_augmented:
        path = DATA_DIR / "analysis_voyage_augmented.parquet"
    else:
        path = DATA_DIR / "analysis_voyage.parquet"

    df = pd.read_parquet(path)
    logger.info("Loaded %d voyages from %s", len(df), path.name)
    return df


def merge_akm_effects(df: pd.DataFrame) -> pd.DataFrame:
    """Merge in-sample AKM captain (theta) and agent (psi) effects."""
    # Captain effects
    captain_path = DATA_DIR / "akm_captain_effects.csv"
    if captain_path.exists():
        akm_c = pd.read_csv(captain_path)
        df = df.merge(
            akm_c[[COLS.captain_id, "theta", "theta_quintile"]],
            on=COLS.captain_id,
            how="left",
        )
        n_matched = df["theta"].notna().sum()
        logger.info("Merged captain theta: %d / %d matched", n_matched, len(df))
    else:
        logger.warning("AKM captain effects not found at %s", captain_path)

    # Agent effects
    agent_path = DATA_DIR / "akm_agent_effects.csv"
    if agent_path.exists():
        akm_a = pd.read_csv(agent_path)
        df = df.merge(
            akm_a[[COLS.agent_id, "psi", "psi_quintile"]],
            on=COLS.agent_id,
            how="left",
        )
        n_matched = df["psi"].notna().sum()
        logger.info("Merged agent psi: %d / %d matched", n_matched, len(df))
    else:
        logger.warning("AKM agent effects not found at %s", agent_path)

    return df


def merge_logbook_features(df: pd.DataFrame) -> pd.DataFrame:
    """Merge voyage-level logbook route features."""
    path = DATA_DIR / "voyage_logbook_features.parquet"
    if not path.exists():
        logger.warning("Logbook features not found at %s", path)
        return df

    lf = pd.read_parquet(path)
    # Avoid column collisions
    existing = set(df.columns)
    merge_cols = [c for c in lf.columns if c not in existing or c == COLS.voyage_id]
    df = df.merge(lf[merge_cols], on=COLS.voyage_id, how="left")
    n_matched = df["route_efficiency"].notna().sum() if "route_efficiency" in df.columns else 0
    logger.info("Merged logbook features: %d / %d matched", n_matched, len(df))
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Derived Variables
# ═══════════════════════════════════════════════════════════════════════════

def construct_derived_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct all derived variables needed for reinforcement tests.

    Adds: log_q, decade, vessel_period, route_time, captain_experience,
    captain_voyage_num, novice, expert, switch indicators, captain_agent.
    """
    df = df.copy()

    # Log output
    df[COLS.log_q] = np.log(df[COLS.q_oil_bbl].clip(lower=1))

    # Decade
    df["decade"] = (df[COLS.year_out] // 10) * 10

    # Vessel × period
    period_bin = 5
    df["period"] = (df[COLS.year_out] // period_bin) * period_bin
    df["vessel_period"] = (
        df[COLS.vessel_id].astype(str) + "_" + df["period"].astype(str)
    )

    # Route × time
    if "route_year_cell" not in df.columns:
        df["route_year_cell"] = (
            df[COLS.ground_or_route].fillna("UNK").astype(str)
            + "_"
            + df[COLS.year_out].astype(str)
        )
    df[COLS.route_time] = df["route_year_cell"]

    # Captain experience (cumulative voyage count)
    df = df.sort_values([COLS.captain_id, COLS.year_out, COLS.date_out])
    df[COLS.captain_voyage_num] = df.groupby(COLS.captain_id).cumcount() + 1
    df[COLS.captain_experience] = df[COLS.captain_voyage_num] - 1  # 0-indexed prior voyages

    # Novice / expert bins
    df["novice"] = (df[COLS.captain_voyage_num] <= CFG.novice_max_voyages).astype(int)
    df["expert"] = (df[COLS.captain_voyage_num] >= CFG.expert_min_voyages).astype(int)

    # Switch indicators
    df = _compute_switch_indicators(df)

    # Captain-agent interaction ID for two-way clustering
    df["captain_agent"] = (
        df[COLS.captain_id].astype(str) + "_" + df[COLS.agent_id].astype(str)
    )

    # Arctic route indicator
    if "frac_days_in_arctic_polygon" in df.columns:
        df["arctic_route"] = (df["frac_days_in_arctic_polygon"] > 0.1).astype(int)
    else:
        df["arctic_route"] = 0

    return df


def _compute_switch_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute agent and vessel switch indicators."""
    df = df.sort_values([COLS.captain_id, COLS.year_out])

    # Previous agent/vessel for each captain
    df["prev_agent"] = df.groupby(COLS.captain_id)[COLS.agent_id].shift(1)
    df["prev_vessel"] = df.groupby(COLS.captain_id)[COLS.vessel_id].shift(1)

    # Switch flags
    df[COLS.switch_agent] = (
        df["prev_agent"].notna() & (df[COLS.agent_id] != df["prev_agent"])
    ).astype(int)
    df[COLS.switch_vessel] = (
        df["prev_vessel"].notna() & (df[COLS.vessel_id] != df["prev_vessel"])
    ).astype(int)

    # Clean up temp columns
    df = df.drop(columns=["prev_agent", "prev_vessel"])

    return df


# ═══════════════════════════════════════════════════════════════════════════
# Sample Restrictions
# ═══════════════════════════════════════════════════════════════════════════

def apply_sample_restrictions(
    df: pd.DataFrame,
    *,
    require_akm: bool = True,
    require_logbook: bool = False,
    trim_output: bool = True,
) -> pd.DataFrame:
    """
    Apply standard sample restrictions.

    Parameters
    ----------
    require_akm : bool
        Require non-missing theta and psi.
    require_logbook : bool
        Require logbook position data.
    trim_output : bool
        Winsorize output variable.

    Returns
    -------
    pd.DataFrame
        Restricted sample.
    """
    n_start = len(df)

    # Year range
    df = df[
        (df[COLS.year_out] >= CFG.min_year)
        & (df[COLS.year_out] <= CFG.max_year)
    ]

    # Minimum voyage counts
    captain_counts = df[COLS.captain_id].value_counts()
    valid_captains = captain_counts[captain_counts >= CFG.min_captain_voyages].index
    df = df[df[COLS.captain_id].isin(valid_captains)]

    agent_counts = df[COLS.agent_id].value_counts()
    valid_agents = agent_counts[agent_counts >= CFG.min_agent_voyages].index
    df = df[df[COLS.agent_id].isin(valid_agents)]

    # AKM effects
    if require_akm:
        df = df[df["theta"].notna() & df["psi"].notna()]

    # Logbook data
    if require_logbook:
        if "has_route_data" in df.columns:
            df = df[df["has_route_data"]]
        elif "route_efficiency" in df.columns:
            df = df[df["route_efficiency"].notna()]

    # Trim output
    if trim_output and COLS.log_q in df.columns:
        lo = df[COLS.log_q].quantile(CFG.output_trim_lower_pct / 100)
        hi = df[COLS.log_q].quantile(CFG.output_trim_upper_pct / 100)
        df = df[(df[COLS.log_q] >= lo) & (df[COLS.log_q] <= hi)]

    logger.info(
        "Sample: %d → %d voyages after restrictions (akm=%s, logbook=%s)",
        n_start, len(df), require_akm, require_logbook,
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def build_analysis_panel(
    *,
    require_akm: bool = True,
    require_logbook: bool = False,
    include_climate: bool = False,
) -> pd.DataFrame:
    """
    Full pipeline to build the analysis-ready voyage panel.

    Returns
    -------
    pd.DataFrame
        Merged, derived, restricted analysis panel.
    """
    logger.info("Building analysis panel...")

    df = load_voyage_panel(include_climate=include_climate)
    df = merge_akm_effects(df)
    df = merge_logbook_features(df)
    df = construct_derived_variables(df)
    df = apply_sample_restrictions(
        df,
        require_akm=require_akm,
        require_logbook=require_logbook,
    )

    logger.info(
        "Final panel: %d voyages, %d captains, %d agents",
        len(df),
        df[COLS.captain_id].nunique(),
        df[COLS.agent_id].nunique(),
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Logbook Position Panel (with encounters)
# ═══════════════════════════════════════════════════════════════════════════

def load_logbook_positions() -> pd.DataFrame:
    """
    Load daily logbook positions with encounter data.

    Returns DataFrame with: voyage_id, obs_date, lat, lon, year,
    encounter, species, n_struck, n_tried, place, remarks.
    """
    path = STAGING_DIR / "logbook_positions.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Logbook positions not found at {path}. "
            "Run the logbook parser first to generate this file."
        )

    df = pd.read_parquet(path)

    # Check if encounter columns are present (parser may not have been re-run)
    if COLS.encounter not in df.columns:
        logger.warning(
            "Encounter columns not found in logbook_positions.parquet. "
            "Re-run the logbook parser to include encounter data."
        )
        # Load directly from raw as fallback
        df = _load_raw_logbook_with_encounters()

    logger.info(
        "Loaded %d logbook positions (%d voyages, %d encounter events)",
        len(df),
        df[COLS.voyage_id].nunique(),
        (df.get(COLS.encounter, pd.Series(dtype=str)) != "NoEnc").sum(),
    )
    return df


def _load_raw_logbook_with_encounters() -> pd.DataFrame:
    """Fallback: load raw AOWL data directly with encounter columns."""
    from pathlib import Path as _Path

    raw_dir = _Path(__file__).resolve().parents[2] / "data" / "raw" / "logbooks"
    raw_files = list(raw_dir.glob("aowl_*.txt"))
    if not raw_files:
        raise FileNotFoundError(f"No AOWL files found in {raw_dir}")

    raw = pd.read_csv(raw_files[0], sep="\t", low_memory=False)

    # Standardize column names
    col_map = {
        "VoyageID": "voyage_id",
        "Lat": "lat",
        "Lon": "lon",
        "Encounter": "encounter",
        "Species": "species",
        "NStruck": "n_struck",
        "NTried": "n_tried",
        "Place": "place",
        "Remarks": "remarks",
    }
    df = raw.rename(columns=col_map)

    # Build date
    from datetime import date

    def _make_date(row):
        try:
            return date(int(row["Year"]), int(row["Month"]), int(row["Day"]))
        except (ValueError, TypeError):
            return None

    df["obs_date"] = df.apply(_make_date, axis=1)
    df["year"] = pd.to_numeric(df.get("Year"), errors="coerce")

    # Clean NULL strings
    for col in ["encounter", "species", "place", "remarks"]:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].replace({"NULL": None, "null": None, "": None})

    for col in ["n_struck", "n_tried"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep standardized columns
    keep = [
        "voyage_id", "obs_date", "lat", "lon", "year",
        "encounter", "species", "n_struck", "n_tried", "place", "remarks",
    ]
    df = df[[c for c in keep if c in df.columns]]

    logger.info("Loaded %d positions from raw AOWL with encounter data", len(df))
    return df
