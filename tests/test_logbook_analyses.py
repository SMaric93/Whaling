from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_route_efficiency_skill_skips_when_columns_are_missing():
    from src.analyses import logbook_analyses

    results = logbook_analyses.test_route_efficiency_skill(
        pd.DataFrame({"log_output": [1.0, 2.0, 3.0]})
    )

    assert results["status"] == "skipped"
    assert results["reason"] == "Missing columns: ['route_efficiency']"


def test_route_efficiency_skill_reports_completed_regression():
    from src.analyses import logbook_analyses

    df = pd.DataFrame({
        "log_output": [2.0, 4.0, 6.0, 8.0, 10.0],
        "route_efficiency": [1.0, 2.0, 3.0, 4.0, 5.0],
    })

    results = logbook_analyses.test_route_efficiency_skill(df)

    assert results["status"] == "completed"
    assert results["efficiency_coefficient"] == 2.0
    assert results["efficiency_r_squared"] == 1.0
    assert results["n_observations"] == 5
    assert "significant positive effect" in results["interpretation"]


def test_run_all_logbook_tests_uses_registry_and_serializes_results(tmp_path, monkeypatch):
    from src.analyses import logbook_analyses

    def fake_test(_: pd.DataFrame):
        return {
            "test_name": "fake_metric",
            "hypothesis": "registry executes configured tests",
            "status": "completed",
            "score": np.float64(0.5),
            "interpretation": "Fake interpretation",
        }

    monkeypatch.setattr(
        logbook_analyses,
        "LOGBOOK_TEST_CATEGORIES",
        (
            logbook_analyses.LogbookCategory(
                title="CATEGORY X: TEST",
                tests=(("fake_metric", fake_test),),
            ),
        ),
    )
    monkeypatch.setattr(logbook_analyses, "OUTPUT_DIR", tmp_path)

    results = logbook_analyses.run_all_logbook_tests(
        df=pd.DataFrame({"voyage_id": [1, 2]}),
        save_outputs=True,
    )

    assert results["fake_metric"]["score"] == np.float64(0.5)

    output_path = tmp_path / "logbook_analyses_results.json"
    assert output_path.exists()
    assert json.loads(output_path.read_text()) == {
        "fake_metric": {
            "test_name": "fake_metric",
            "hypothesis": "registry executes configured tests",
            "status": "completed",
            "score": 0.5,
            "interpretation": "Fake interpretation",
        }
    }
