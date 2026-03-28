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


def test_stage5_paper_table_api_compat():
    from src.analyses.paper_tables import (
        generate_all_latex_tables,
        generate_all_markdown_tables,
        generate_table_1,
        generate_table_a1,
        generate_table_a9,
    )

    assert generate_table_1().startswith("## Table 1")
    assert generate_table_a1().startswith("## Table A1")
    assert generate_table_a9().startswith("## Table A9")
    assert "\\documentclass" in generate_all_latex_tables()
    assert "# Maps of the Sea: Publication Tables" in generate_all_markdown_tables()



def test_weather_runner_accepts_default_df():
    from src.analyses.weather_regressions import run_weather_regressions

    signature = inspect.signature(run_weather_regressions)
    assert signature.parameters["df"].default is None
