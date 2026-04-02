from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import MAIN_TABLES, BuildContext
from .data import (
    load_action_dataset,
    load_connected_sample,
    load_logbook_features,
    load_patch_sample,
    load_positions,
    load_survival_dataset,
    load_universe,
)


def _voyage_lookup(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return a voyage-indexed lookup table for the requested columns."""
    available = [c for c in columns if c in frame.columns]
    if frame.empty or "voyage_id" not in frame.columns or not available:
        return pd.DataFrame(columns=available)
    return frame.loc[frame["voyage_id"].notna(), ["voyage_id", *available]].drop_duplicates("voyage_id").set_index("voyage_id")


def _voyage_counts(frame: pd.DataFrame, output_col: str) -> pd.Series:
    """Count voyage-level observations without building an intermediate frame."""
    if frame.empty or "voyage_id" not in frame.columns:
        return pd.Series(dtype=float, name=output_col)
    return (
        frame.loc[frame["voyage_id"].notna()]
        .groupby("voyage_id", sort=False)
        .size()
        .rename(output_col)
    )


def _ensure_columns(frame: pd.DataFrame, defaults: dict[str, object]) -> pd.DataFrame:
    """Backfill optional columns that are absent in older intermediate datasets."""
    frame = frame.copy()
    for column, default in defaults.items():
        if column not in frame.columns:
            frame[column] = default
    return frame


def build_master_sample_lineage(context: BuildContext) -> Path:
    out = context.outputs / "manifests" / "master_sample_lineage.parquet"

    universe = _ensure_columns(
        load_universe(context),
        {
            "has_logbook_data": False,
            "logbook_source_count": 0,
            "has_route_data": False,
            "has_labor_data": False,
            "has_vqi_data": False,
        },
    )
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
    connected_flags = _voyage_lookup(connected, connected_cols)
    connected_flags["in_connected_set"] = True
    df = df.join(connected_flags, on="voyage_id", rsuffix="_connected")

    if "ground_or_route_connected" in df.columns:
        df["ground_or_route"] = df["ground_or_route"].combine_first(df["ground_or_route_connected"])
        df = df.drop(columns=["ground_or_route_connected"])
    if "has_logbook_data_connected" in df.columns:
        df["has_logbook_data"] = df["has_logbook_data"].combine_first(df["has_logbook_data_connected"])
        df = df.drop(columns=["has_logbook_data_connected"])
    if "logbook_source_count_connected" in df.columns:
        df["logbook_source_count"] = df["logbook_source_count"].combine_first(df["logbook_source_count_connected"])
        df = df.drop(columns=["logbook_source_count_connected"])

    df = _ensure_columns(
        df,
        {
            "theta": np.nan,
            "psi": np.nan,
            "scarcity": np.nan,
            "switch_agent": 0,
            "switch_vessel": 0,
            "n_positions": np.nan,
            "n_grounds_visited": np.nan,
        },
    )

    df["coordinate_observations"] = df["voyage_id"].map(_voyage_counts(positions, "coordinate_observations"))

    if not logbook_features.empty:
        logbook_subset = _voyage_lookup(logbook_features, ["n_positions", "primary_ground"])
        df = df.join(logbook_subset, on="voyage_id", rsuffix="_logbook")
        if "n_positions_logbook" in df.columns:
            df["n_positions"] = df["n_positions"].combine_first(df["n_positions_logbook"])
            df = df.drop(columns=["n_positions_logbook"])

    df["patch_observations"] = df["voyage_id"].map(_voyage_counts(patches, "patch_observations"))
    df["patch_day_observations"] = df["voyage_id"].map(_voyage_counts(survival, "patch_day_observations"))
    df["encounter_observations"] = df["voyage_id"].map(_voyage_counts(action, "encounter_observations"))

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
