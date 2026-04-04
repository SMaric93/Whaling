from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.reinforcement.data_builder import build_analysis_panel

from .config import BuildContext


def _dedupe_voyages(df: pd.DataFrame) -> pd.DataFrame:
    if "voyage_id" not in df.columns:
        return df.copy()
    return df[df["voyage_id"].notna()].drop_duplicates("voyage_id").copy()


def _empty_frame(*columns: str) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _read_optional_table(
    path: Path,
    *,
    reader: Callable[[Path], pd.DataFrame],
    empty_columns: tuple[str, ...] = ("voyage_id",),
    dedupe_voyages: bool = False,
) -> pd.DataFrame:
    if not path.exists():
        return _empty_frame(*empty_columns)
    frame = reader(path)
    if dedupe_voyages:
        return _dedupe_voyages(frame)
    return frame


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except TypeError:
        return False


def load_universe(context: BuildContext) -> pd.DataFrame:
    path = context.root / "data" / "final" / "analysis_voyage_augmented.parquet"
    return _dedupe_voyages(pd.read_parquet(path))


def load_connected_sample(context: BuildContext) -> pd.DataFrame:
    cached = context.root / "outputs" / "datasets" / "ml" / "outcome_ml_dataset.parquet"
    if cached.exists():
        df = _dedupe_voyages(pd.read_parquet(cached))
    else:
        df = _dedupe_voyages(build_analysis_panel(require_akm=True))
    # Guarantee AKM columns exist for downstream table builders
    for col in ["theta", "psi", "scarcity"]:
        if col not in df.columns:
            df[col] = np.nan
    return df


def load_action_dataset(context: BuildContext) -> pd.DataFrame:
    path = context.root / "outputs" / "datasets" / "ml" / "action_dataset.parquet"
    return _read_optional_table(path, reader=pd.read_parquet)


def load_state_dataset(context: BuildContext) -> pd.DataFrame:
    path = context.root / "outputs" / "datasets" / "ml" / "state_dataset.parquet"
    return _read_optional_table(path, reader=pd.read_parquet)


def load_survival_dataset(context: BuildContext) -> pd.DataFrame:
    path = context.root / "outputs" / "datasets" / "ml" / "survival_dataset.parquet"
    return _read_optional_table(path, reader=pd.read_parquet)


def load_logbook_features(context: BuildContext) -> pd.DataFrame:
    path = context.root / "data" / "final" / "voyage_logbook_features.parquet"
    return _read_optional_table(
        path,
        reader=pd.read_parquet,
        dedupe_voyages=True,
    )


def load_ground_quality(context: BuildContext) -> pd.DataFrame:
    path = context.root / "data" / "derived" / "ground_quality_loo.parquet"
    return _read_optional_table(
        path,
        reader=pd.read_parquet,
        dedupe_voyages=True,
    )


def load_destination_ontology(context: BuildContext) -> pd.DataFrame:
    path = context.root / "data" / "derived" / "destination_ontology.parquet"
    return _read_optional_table(
        path,
        reader=pd.read_parquet,
        empty_columns=(
            "ground_or_route",
            "basin",
            "theater",
            "major_ground",
            "ground_for_model",
        ),
    )


def load_positions(context: BuildContext) -> pd.DataFrame:
    path = context.root / "data" / "staging" / "logbook_positions.parquet"
    return _read_optional_table(path, reader=pd.read_parquet)


def load_patch_sample(context: BuildContext) -> pd.DataFrame:
    path = context.root / "output" / "stopping_rule" / "patches.csv"
    return _read_optional_table(path, reader=pd.read_csv)


def load_akm_variance_decomposition(context: BuildContext) -> pd.DataFrame:
    path = context.root / "data" / "final" / "akm_variance_decomposition.csv"
    return _read_optional_table(
        path,
        reader=pd.read_csv,
        empty_columns=("Component", "Type", "Variance", "Share"),
    )


def load_split_sample_stability(context: BuildContext) -> pd.DataFrame:
    path = context.root / "output" / "figures" / "akm_tails" / "split_sample_stability.csv"
    return _read_optional_table(
        path,
        reader=pd.read_csv,
        empty_columns=("entity_type", "n_bin", "split_corr", "n_entities"),
    )


def load_test3_stopping_output(context: BuildContext) -> pd.DataFrame:
    path = context.root / "output" / "reinforcement" / "tables" / "test3_stopping_rule.csv"
    return _read_optional_table(
        path,
        reader=pd.read_csv,
        empty_columns=("specification", "variable", "coefficient", "std_error", "p_value", "n_obs"),
    )


def load_rational_exit_output(context: BuildContext) -> pd.DataFrame:
    path = context.root / "outputs" / "tables" / "next_round" / "rational_exit_tests.csv"
    return _read_optional_table(
        path,
        reader=pd.read_csv,
        empty_columns=("test", "exit_rate"),
    )


def load_next_round_output(context: BuildContext, filename: str) -> pd.DataFrame:
    path = context.root / "outputs" / "tables" / "next_round" / filename
    return _read_optional_table(path, reader=pd.read_csv, empty_columns=())


def years_label(values: pd.Series) -> str:
    years = pd.to_numeric(values, errors="coerce").dropna()
    if years.empty:
        return ""
    return f"{int(years.min())}-{int(years.max())}"


def infer_basin(values: pd.Series) -> pd.Series:
    def _map(value: object) -> object:
        if _is_missing(value):
            return np.nan
        text = str(value).upper()
        if "PACIFIC" in text:
            return "Pacific"
        if any(token in text for token in ["ATLANTIC", "BRAZIL", "GRAND BANKS", "W INDIES"]):
            return "Atlantic"
        if "INDIAN" in text:
            return "Indian"
        if any(token in text for token in ["ARCTIC", "BERING"]):
            return "Arctic/Bering"
        return "Other"

    return values.map(_map)
