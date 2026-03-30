from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_compute_date_score_accepts_timestamp_inputs():
    from src.entities.wsl_voyage_matcher import compute_date_score

    score = compute_date_score(
        "1843-03-17",
        pd.Timestamp("1842-01-01"),
        pd.Timestamp("1844-01-01"),
    )

    assert score == 1.0
