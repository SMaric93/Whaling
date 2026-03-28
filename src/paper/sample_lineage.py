from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import BuildContext, MAIN_TABLES
from .data import (
    load_action_dataset,
    load_connected_sample,
    load_logbook_features,
    load_patch_sample,
    load_positions,
    load_survival_dataset,
    load_universe,
)


def build_master_sample_lineage(context: BuildContext) -> Path:
    out = context.outputs / "manifests" / "master_sample_lineage.parquet"

    universe = load_universe(context)
    connected = load_connected_sample(context)
    logbook_features = load_logbook_features(context)
    positions = load_positions(context)
    patches = load_patch_sample(context)
    survival = load_survival_dataset(context)
    action = load_action_dataset(context)

    df = universe[
        [
            "voyage_id",
            "captain_id",
            "agent_id",
            "vessel_id",
            "year_out",
            "ground_or_route",
            "has_labor_data",
            "has_route_data",
            "has_vqi_data",
            "has_logbook_data",
            "logbook_source_count",
            "tonnage",
            "crew_count",
        ]
    ].copy()
    df["in_universe"] = True

    connected_cols = [
        "voyage_id",
        "theta",
        "psi",
        "switch_agent",
        "switch_vessel",
        "scarcity",
        "n_positions",
        "n_grounds_visited",
        "ground_or_route",
        "has_logbook_data",
        "logbook_source_count",
    ]
    connected_cols = [c for c in connected_cols if c in connected.columns]
    connected_flags = connected[connected_cols].copy()
    connected_flags["in_connected_set"] = True
    df = df.merge(connected_flags, on="voyage_id", how="left", suffixes=("", "_connected"))

    if "ground_or_route_connected" in df.columns:
        df["ground_or_route"] = df["ground_or_route"].fillna(df["ground_or_route_connected"])
        df = df.drop(columns=["ground_or_route_connected"])
    if "has_logbook_data_connected" in df.columns:
        df["has_logbook_data"] = df["has_logbook_data"].fillna(df["has_logbook_data_connected"])
        df = df.drop(columns=["has_logbook_data_connected"])
    if "logbook_source_count_connected" in df.columns:
        df["logbook_source_count"] = df["logbook_source_count"].fillna(df["logbook_source_count_connected"])
        df = df.drop(columns=["logbook_source_count_connected"])

    position_counts = (
        positions[positions["voyage_id"].notna()]
        .groupby("voyage_id")
        .size()
        .rename("coordinate_observations")
        .reset_index()
    )
    df = df.merge(position_counts, on="voyage_id", how="left")

    if not logbook_features.empty:
        logbook_subset = logbook_features[
            [c for c in ["voyage_id", "n_positions", "primary_ground"] if c in logbook_features.columns]
        ]
        df = df.merge(logbook_subset, on="voyage_id", how="left", suffixes=("", "_logbook"))
        if "n_positions_logbook" in df.columns:
            df["n_positions"] = df["n_positions"].fillna(df["n_positions_logbook"])
            df = df.drop(columns=["n_positions_logbook"])

    patch_counts = (
        patches.groupby("voyage_id")
        .size()
        .rename("patch_observations")
        .reset_index()
        if not patches.empty
        else pd.DataFrame(columns=["voyage_id", "patch_observations"])
    )
    df = df.merge(patch_counts, on="voyage_id", how="left")

    patch_day_counts = (
        survival.groupby("voyage_id")
        .size()
        .rename("patch_day_observations")
        .reset_index()
        if not survival.empty
        else pd.DataFrame(columns=["voyage_id", "patch_day_observations"])
    )
    df = df.merge(patch_day_counts, on="voyage_id", how="left")

    encounter_counts = (
        action.groupby("voyage_id")
        .size()
        .rename("encounter_observations")
        .reset_index()
        if not action.empty
        else pd.DataFrame(columns=["voyage_id", "encounter_observations"])
    )
    df = df.merge(encounter_counts, on="voyage_id", how="left")

    df["in_connected_set"] = df["in_connected_set"].fillna(False).astype(bool)
    df["has_coordinates"] = (
        df["has_route_data"].fillna(False)
        | df["coordinate_observations"].fillna(0).gt(0)
        | df["n_positions"].fillna(0).gt(0)
    )
    df["has_ground_labels"] = (
        df["ground_or_route"].notna()
        | df.get("primary_ground", pd.Series(index=df.index, dtype=object)).notna()
    )
    df["has_patch_data"] = df["patch_observations"].fillna(0).gt(0)
    df["has_encounter_data"] = df["encounter_observations"].fillna(0).gt(0)
    df["has_switch_event"] = df.get("switch_agent", 0).fillna(0).astype(float).gt(0)
    df["has_archival_data"] = (
        df["has_labor_data"].fillna(False)
        | df["has_vqi_data"].fillna(False)
        | df["has_logbook_data"].fillna(False)
        | df["logbook_source_count"].fillna(0).gt(0)
    )

    df["used_in_table01"] = df["in_universe"]
    df["used_in_table02"] = df["in_connected_set"]
    df["used_in_table03"] = df["in_connected_set"] & df["has_ground_labels"]
    df["used_in_table04"] = df["in_connected_set"] & df["has_patch_data"] & df["has_encounter_data"]
    df["used_in_table05"] = df["in_connected_set"] & df["has_switch_event"]
    df["used_in_table06"] = df["in_connected_set"] & df["has_encounter_data"]
    df["used_in_table07"] = df["in_connected_set"] & df["tonnage"].notna()
    df["used_in_table08"] = df["in_connected_set"]
    df["used_in_table09"] = df["in_connected_set"] & df["has_patch_data"]
    df["used_in_table10"] = df["in_connected_set"]

    for t in MAIN_TABLES:
        col = f"used_in_{t}"
        if col not in df.columns:
            df[col] = df["in_connected_set"]

    for i in range(1, 19):
        appendix_col = f"used_in_appendix_A{i:02d}"
        if i == 1:
            df[appendix_col] = df["in_universe"]
        elif i == 2:
            df[appendix_col] = df["has_ground_labels"]
        elif i == 3:
            df[appendix_col] = df["in_connected_set"]
        elif i == 5:
            df[appendix_col] = df["has_switch_event"]
        elif i == 8:
            df[appendix_col] = df["has_patch_data"]
        elif i == 10:
            df[appendix_col] = df["used_in_table04"]
        elif i == 18:
            df[appendix_col] = df["has_archival_data"]
        else:
            df[appendix_col] = df["in_connected_set"]

    df["exclusion_connected_set_reason"] = np.where(~df["in_connected_set"], "missing_theta_or_psi", "")
    df["exclusion_coordinates_reason"] = np.where(~df["has_coordinates"], "no_position_history", "")
    df["exclusion_ground_reason"] = np.where(~df["has_ground_labels"], "no_ground_label", "")
    df["exclusion_patch_reason"] = np.where(~df["has_patch_data"], "no_patch_panel", "")
    df["exclusion_encounter_reason"] = np.where(~df["has_encounter_data"], "no_encounter_panel", "")
    df["exclusion_archival_reason"] = np.where(~df["has_archival_data"], "no_archival_proxy", "")
    df["primary_exclusion_reason"] = np.select(
        [
            ~df["in_connected_set"],
            ~df["has_coordinates"],
            ~df["has_ground_labels"],
            ~df["has_patch_data"],
            ~df["has_encounter_data"],
        ],
        [
            "missing_connected_set",
            "missing_coordinates",
            "missing_ground_labels",
            "missing_patch_data",
            "missing_encounter_data",
        ],
        default="",
    )

    bool_cols = [c for c in df.columns if c.startswith("used_in_") or c.startswith("has_") or c.startswith("in_")]
    for col in bool_cols:
        df[col] = df[col].fillna(False).astype(bool)

    df.to_parquet(out, index=False)
    return out
