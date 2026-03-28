from __future__ import annotations

import numpy as np
import pandas as pd

from src.reinforcement.data_builder import build_analysis_panel

from .config import BuildContext


def _dedupe_voyages(df: pd.DataFrame) -> pd.DataFrame:
    if "voyage_id" not in df.columns:
        return df.copy()
    return df[df["voyage_id"].notna()].drop_duplicates("voyage_id").copy()


def load_universe(context: BuildContext) -> pd.DataFrame:
    path = context.root / "data" / "final" / "analysis_voyage_augmented.parquet"
    return _dedupe_voyages(pd.read_parquet(path))


def load_connected_sample(context: BuildContext) -> pd.DataFrame:
    cached = context.root / "outputs" / "datasets" / "ml" / "outcome_ml_dataset.parquet"
    if cached.exists():
        return _dedupe_voyages(pd.read_parquet(cached))
    return _dedupe_voyages(build_analysis_panel(require_akm=True))


def load_action_dataset(context: BuildContext) -> pd.DataFrame:
    path = context.root / "outputs" / "datasets" / "ml" / "action_dataset.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["voyage_id"])
    return pd.read_parquet(path)


def load_state_dataset(context: BuildContext) -> pd.DataFrame:
    path = context.root / "outputs" / "datasets" / "ml" / "state_dataset.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["voyage_id"])
    return pd.read_parquet(path)


def load_survival_dataset(context: BuildContext) -> pd.DataFrame:
    path = context.root / "outputs" / "datasets" / "ml" / "survival_dataset.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["voyage_id"])
    return pd.read_parquet(path)


def load_logbook_features(context: BuildContext) -> pd.DataFrame:
    path = context.root / "data" / "final" / "voyage_logbook_features.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["voyage_id"])
    return _dedupe_voyages(pd.read_parquet(path))


def load_ground_quality(context: BuildContext) -> pd.DataFrame:
    path = context.root / "data" / "derived" / "ground_quality_loo.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["voyage_id"])
    return _dedupe_voyages(pd.read_parquet(path))


def load_positions(context: BuildContext) -> pd.DataFrame:
    path = context.root / "data" / "staging" / "logbook_positions.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["voyage_id"])
    return pd.read_parquet(path)


def load_patch_sample(context: BuildContext) -> pd.DataFrame:
    path = context.root / "output" / "stopping_rule" / "patches.csv"
    if not path.exists():
        return pd.DataFrame(columns=["voyage_id"])
    return pd.read_csv(path)


def load_akm_variance_decomposition(context: BuildContext) -> pd.DataFrame:
    return pd.read_csv(context.root / "data" / "final" / "akm_variance_decomposition.csv")


def load_split_sample_stability(context: BuildContext) -> pd.DataFrame:
    return pd.read_csv(context.root / "output" / "figures" / "akm_tails" / "split_sample_stability.csv")


def load_test3_stopping_output(context: BuildContext) -> pd.DataFrame:
    path = context.root / "output" / "reinforcement" / "tables" / "test3_stopping_rule.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_rational_exit_output(context: BuildContext) -> pd.DataFrame:
    path = context.root / "outputs" / "tables" / "next_round" / "rational_exit_tests.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_next_round_output(context: BuildContext, filename: str) -> pd.DataFrame:
    path = context.root / "outputs" / "tables" / "next_round" / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def years_label(values: pd.Series) -> str:
    years = pd.to_numeric(values, errors="coerce").dropna()
    if years.empty:
        return ""
    return f"{int(years.min())}-{int(years.max())}"


def infer_basin(values: pd.Series) -> pd.Series:
    def _map(value: object) -> object:
        if value is None:
            return np.nan
        try:
            if pd.isna(value):
                return np.nan
        except TypeError:
            pass
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
