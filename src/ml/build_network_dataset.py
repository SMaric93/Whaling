"""
ML Layer — Network Dataset Builder.

Unit: career edge or co-working relation.

Constructs network from the voyage panel:
- captain-agent edges
- captain-vessel edges
- other pairwise relations if data available

Conditional on data existence.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import pandas as pd

from src.ml.config import ML_CFG, ML_DATA_DIR

logger = logging.getLogger(__name__)

OUTPUT_PATH = ML_DATA_DIR / "network_dataset.parquet"


def build_network_dataset(
    *,
    force_rebuild: bool = False,
    save: bool = True,
) -> pd.DataFrame:
    """
    Build the network dataset from voyage-level relations.

    Creates edges for:
    - captain-agent (worked together)
    - captain-vessel (sailed on)

    Each edge has start/end dates, exposure counts, and early-career flags.

    Returns
    -------
    pd.DataFrame with columns:
        person_id_1, person_type_1, person_id_2, person_type_2,
        relationship_type, start_date, end_date, exposure_count,
        early_career_flag_1, early_career_flag_2
    """
    if OUTPUT_PATH.exists() and not force_rebuild:
        logger.info("Loading cached network dataset from %s", OUTPUT_PATH)
        return pd.read_parquet(OUTPUT_PATH)

    t0 = time.time()
    logger.info("Building network dataset...")

    from src.reinforcement.data_builder import build_analysis_panel

    voyages = build_analysis_panel(require_akm=True, require_logbook=False)

    edges = []

    # ── Captain-Agent edges ─────────────────────────────────────────
    if "captain_id" in voyages.columns and "agent_id" in voyages.columns:
        ca = voyages.dropna(subset=["captain_id", "agent_id"]).copy()

        # Compute captain voyage number for early-career flag
        if "captain_voyage_num" not in ca.columns:
            ca["captain_voyage_num"] = ca.groupby("captain_id").cumcount() + 1

        ca_edges = ca.groupby(["captain_id", "agent_id"]).agg(
            start_year=("year_out", "min"),
            end_year=("year_out", "max"),
            exposure_count=("voyage_id", "count"),
            early_career_count=("captain_voyage_num",
                                lambda x: (x <= ML_CFG.experience_bins["novice_max"]).sum()),
        ).reset_index()

        ca_edges["person_id_1"] = ca_edges["captain_id"]
        ca_edges["person_type_1"] = "captain"
        ca_edges["person_id_2"] = ca_edges["agent_id"]
        ca_edges["person_type_2"] = "agent"
        ca_edges["relationship_type"] = "captain_agent"
        ca_edges["early_career_flag"] = (ca_edges["early_career_count"] > 0).astype(int)

        edges.append(ca_edges[[
            "person_id_1", "person_type_1", "person_id_2", "person_type_2",
            "relationship_type", "start_year", "end_year",
            "exposure_count", "early_career_flag",
        ]])
        logger.info("Built %d captain-agent edges", len(ca_edges))

    # ── Captain-Vessel edges ────────────────────────────────────────
    if "captain_id" in voyages.columns and "vessel_id" in voyages.columns:
        cv = voyages.dropna(subset=["captain_id", "vessel_id"]).copy()

        if "captain_voyage_num" not in cv.columns:
            cv["captain_voyage_num"] = cv.groupby("captain_id").cumcount() + 1

        cv_edges = cv.groupby(["captain_id", "vessel_id"]).agg(
            start_year=("year_out", "min"),
            end_year=("year_out", "max"),
            exposure_count=("voyage_id", "count"),
            early_career_count=("captain_voyage_num",
                                lambda x: (x <= ML_CFG.experience_bins["novice_max"]).sum()),
        ).reset_index()

        cv_edges["person_id_1"] = cv_edges["captain_id"]
        cv_edges["person_type_1"] = "captain"
        cv_edges["person_id_2"] = cv_edges["vessel_id"]
        cv_edges["person_type_2"] = "vessel"
        cv_edges["relationship_type"] = "captain_vessel"
        cv_edges["early_career_flag"] = (cv_edges["early_career_count"] > 0).astype(int)

        edges.append(cv_edges[[
            "person_id_1", "person_type_1", "person_id_2", "person_type_2",
            "relationship_type", "start_year", "end_year",
            "exposure_count", "early_career_flag",
        ]])
        logger.info("Built %d captain-vessel edges", len(cv_edges))

    # ── Vessel-Agent edges ──────────────────────────────────────────
    if "vessel_id" in voyages.columns and "agent_id" in voyages.columns:
        va = voyages.dropna(subset=["vessel_id", "agent_id"]).copy()

        va_edges = va.groupby(["vessel_id", "agent_id"]).agg(
            start_year=("year_out", "min"),
            end_year=("year_out", "max"),
            exposure_count=("voyage_id", "count"),
        ).reset_index()

        va_edges["person_id_1"] = va_edges["vessel_id"]
        va_edges["person_type_1"] = "vessel"
        va_edges["person_id_2"] = va_edges["agent_id"]
        va_edges["person_type_2"] = "agent"
        va_edges["relationship_type"] = "vessel_agent"
        va_edges["early_career_flag"] = 0

        edges.append(va_edges[[
            "person_id_1", "person_type_1", "person_id_2", "person_type_2",
            "relationship_type", "start_year", "end_year",
            "exposure_count", "early_career_flag",
        ]])
        logger.info("Built %d vessel-agent edges", len(va_edges))

    if not edges:
        logger.warning("No network edges could be constructed")
        result = pd.DataFrame()
    else:
        result = pd.concat(edges, ignore_index=True)

    elapsed = time.time() - t0
    logger.info(
        "Network dataset built: %d edges, %.1fs",
        len(result), elapsed,
    )

    if save and len(result) > 0:
        result.to_parquet(OUTPUT_PATH, index=False)
        logger.info("Saved to %s", OUTPUT_PATH)

    return result
