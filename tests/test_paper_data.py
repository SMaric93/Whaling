from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paper.config import BuildContext
from src.paper.data import (
    infer_basin,
    load_action_dataset,
    load_akm_variance_decomposition,
    load_destination_ontology,
    load_ground_quality,
    load_logbook_features,
    load_rational_exit_output,
    load_split_sample_stability,
    years_label,
)


def _context(tmp_path: Path) -> BuildContext:
    root = tmp_path / "repo"
    return BuildContext(
        root=root,
        outputs=root / "outputs" / "paper",
        docs=root / "docs" / "paper",
    )


def test_optional_loaders_return_schema_when_files_are_missing(tmp_path: Path) -> None:
    context = _context(tmp_path)

    assert list(load_action_dataset(context).columns) == ["voyage_id"]
    assert list(load_logbook_features(context).columns) == ["voyage_id"]
    assert list(load_ground_quality(context).columns) == ["voyage_id"]
    assert list(load_destination_ontology(context).columns) == [
        "ground_or_route",
        "basin",
        "theater",
        "major_ground",
        "ground_for_model",
    ]
    assert list(load_akm_variance_decomposition(context).columns) == [
        "Component",
        "Type",
        "Variance",
        "Share",
    ]
    assert list(load_split_sample_stability(context).columns) == [
        "entity_type",
        "n_bin",
        "split_corr",
        "n_entities",
    ]
    assert list(load_rational_exit_output(context).columns) == ["test", "exit_rate"]


def test_optional_loaders_dedupe_voyage_level_parquet_inputs(tmp_path: Path) -> None:
    context = _context(tmp_path)
    logbook_path = context.root / "data" / "final"
    quality_path = context.root / "data" / "derived"
    logbook_path.mkdir(parents=True, exist_ok=True)
    quality_path.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "voyage_id": ["V1", "V1", "V2", None],
            "n_positions": [2, 3, 4, 5],
        }
    ).to_parquet(logbook_path / "voyage_logbook_features.parquet", index=False)
    pd.DataFrame(
        {
            "voyage_id": ["V1", "V1", "V3"],
            "quality_loo_ground_year": [0.2, 0.3, 0.4],
        }
    ).to_parquet(quality_path / "ground_quality_loo.parquet", index=False)

    logbook = load_logbook_features(context)
    quality = load_ground_quality(context)

    assert logbook["voyage_id"].tolist() == ["V1", "V2"]
    assert quality["voyage_id"].tolist() == ["V1", "V3"]


def test_years_label_and_infer_basin_handle_blank_values() -> None:
    years = pd.Series([1835, np.nan, 1820, 1841])
    labels = pd.Series(
        [
            "North Pacific grounds",
            "Brazil Banks",
            "Indian Ocean",
            "Bering cruise",
            "",
            None,
        ]
    )

    assert years_label(years) == "1820-1841"
    assert years_label(pd.Series([np.nan])) == ""

    mapped = infer_basin(labels)
    assert mapped.iloc[0] == "Pacific"
    assert mapped.iloc[1] == "Atlantic"
    assert mapped.iloc[2] == "Indian"
    assert mapped.iloc[3] == "Arctic/Bering"
    assert mapped.iloc[4] == "Other"
    assert pd.isna(mapped.iloc[5])
