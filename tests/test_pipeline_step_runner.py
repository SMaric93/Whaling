from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _raise_runtime_error(message: str):
    raise RuntimeError(message)


def test_run_steps_updates_successes_and_preserves_failure_defaults(caplog):
    from src.pipeline._runner import StepSpec, run_steps

    results = {
        "success": False,
        "failure": "keep-default",
    }

    with caplog.at_level(logging.WARNING):
        run_steps(
            results,
            [
                StepSpec("success", lambda: {"ok": True}, "success failed"),
                StepSpec("failure", lambda: _raise_runtime_error("boom"), "failure failed"),
            ],
            logger=logging.getLogger("pipeline-test"),
        )

    assert results["success"] == {"ok": True}
    assert results["failure"] == "keep-default"
    assert "failure failed: boom" in caplog.text


def test_run_pull_continues_after_core_failure(monkeypatch):
    from src.pipeline import stage1_pull

    calls: list[tuple[str, bool]] = []

    monkeypatch.setattr(
        stage1_pull,
        "pull_aowv",
        lambda force=False: calls.append(("aowv", force)) or True,
    )
    monkeypatch.setattr(
        stage1_pull,
        "pull_online_sources",
        lambda force=False: calls.append(("online_sources", force)) or _raise_runtime_error("network"),
    )
    monkeypatch.setattr(
        stage1_pull,
        "pull_weather",
        lambda force=False: calls.append(("weather", force)) or True,
    )
    monkeypatch.setattr(
        stage1_pull,
        "pull_wsl_pdfs",
        lambda force=False: calls.append(("wsl_pdfs", force)) or True,
    )
    monkeypatch.setattr(
        stage1_pull,
        "pull_economic",
        lambda force=False: calls.append(("economic", force)) or False,
    )

    results = stage1_pull.run_pull(force=True, skip_optional=False)

    assert results == {
        "aowv": True,
        "online_sources": False,
        "wsl_pdfs": True,
        "weather": True,
        "economic": False,
    }
    assert calls == [
        ("aowv", True),
        ("online_sources", True),
        ("weather", True),
        ("wsl_pdfs", True),
        ("economic", True),
    ]


def test_run_analyze_falls_back_when_baseline_suite_fails(monkeypatch):
    from src.pipeline import stage4_analyze

    monkeypatch.setattr(
        stage4_analyze,
        "run_full_baseline_suite",
        lambda: _raise_runtime_error("baseline down"),
    )
    monkeypatch.setattr(stage4_analyze, "run_baseline_akm", lambda: {"akm": 1})
    monkeypatch.setattr(
        stage4_analyze,
        "run_complementarity_analysis",
        lambda: {"complementarity": 1},
    )
    monkeypatch.setattr(
        stage4_analyze,
        "run_event_studies",
        lambda: pytest.fail("quick mode should skip extended analyses"),
    )

    results = stage4_analyze.run_analyze(quick=True)

    assert "baseline" not in results
    assert results["akm"] == {"akm": 1}
    assert results["complementarity"] == {"complementarity": 1}


def test_run_output_continues_after_individual_failures(monkeypatch):
    from src.pipeline import stage5_output

    monkeypatch.setattr(stage5_output, "ensure_output_dirs", lambda: None)
    monkeypatch.setattr(stage5_output, "generate_main_tables_md", lambda: ["table_1.md"])
    monkeypatch.setattr(
        stage5_output,
        "generate_appendix_tables_md",
        lambda: _raise_runtime_error("appendix failed"),
    )
    monkeypatch.setattr(stage5_output, "generate_all_tables_tex", lambda: "all_tables.tex")
    monkeypatch.setattr(stage5_output, "generate_all_tables_md", lambda: "all_tables.md")
    monkeypatch.setattr(stage5_output, "copy_figures_to_paper", lambda: 3)
    monkeypatch.setattr(stage5_output, "generate_robustness_summary", lambda: None)
    monkeypatch.setattr(stage5_output, "generate_mechanism_summary", lambda: None)

    results = stage5_output.run_output(include_figures=True)

    assert results["main_tables_md"] == ["table_1.md"]
    assert results["appendix_tables_md"] == []
    assert results["all_tables_tex"] == "all_tables.tex"
    assert results["all_tables_md"] == "all_tables.md"
    assert results["figures_copied"] == 3
