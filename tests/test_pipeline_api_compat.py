from __future__ import annotations

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_stage1_download_api_compat():
    from src.download.aowv_downloader import download_all_aowv_data

    assert callable(download_all_aowv_data)


def test_stage4_analysis_api_compat():
    from src.analyses.insurance_variance_test import run_insurance_variances
    from src.analyses.mechanism_crew import run_crew_mechanism_analysis
    from src.analyses.vessel_mover_analysis import run_vessel_mover_robustness

    assert callable(run_vessel_mover_robustness)
    assert callable(run_insurance_variances)
    assert callable(run_crew_mechanism_analysis)


def test_stage5_paper_layer_api_compat():
    """Verify the data-driven paper layer is importable."""
    from src.paper.config import BuildContext, MAIN_TABLES, APPENDIX_TABLES

    assert len(MAIN_TABLES) >= 10
    assert len(APPENDIX_TABLES) >= 15

    context = BuildContext()
    assert context.root.exists()


def test_weather_runner_accepts_default_df():
    from src.analyses.weather_regressions import run_weather_regressions

    signature = inspect.signature(run_weather_regressions)
    assert signature.parameters["df"].default is None
