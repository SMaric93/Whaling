from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_build_augmented_voyage_fills_boolean_flags_without_futurewarning():
    from src.assembly.voyage_augmentor import build_augmented_voyage

    base_voyage = pd.DataFrame({"voyage_id": ["V1", "V2"]})
    wsl_features = pd.DataFrame(
        {
            "voyage_id": ["V1"],
            "n_wsl_events_total": [3],
            "has_wsl_loss": [True],
        }
    )
    route_validation = pd.DataFrame(
        {
            "voyage_id": ["V1"],
            "route_discrepancy_flag": [True],
        }
    )
    icoads_controls = pd.DataFrame(columns=["voyage_id"])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FutureWarning)
        result = build_augmented_voyage(
            base_voyage,
            wsl_features,
            route_validation,
            icoads_controls,
        )

    assert not [w for w in caught if issubclass(w.category, FutureWarning)]
    assert result["has_wsl_loss"].dtype == bool
    assert result["route_discrepancy_flag"].dtype == bool
    assert result["has_wsl_loss"].tolist() == [True, False]
    assert result["route_discrepancy_flag"].tolist() == [True, False]
