from __future__ import annotations

import logging
import os
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


def test_summarize_step_results_counts_successes_skips_and_failures():
    from src.pipeline._runner import summarize_step_results

    summary = summarize_step_results(
        {
            "success": True,
            "skipped": None,
            "failure": False,
            "payload": {"ok": True},
        }
    )

    assert summary == (2, 1, 1)


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


def test_stage4_sets_writable_matplotlib_cache(monkeypatch, tmp_path):
    from src.pipeline import stage4_analyze

    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    monkeypatch.setattr(stage4_analyze.tempfile, "gettempdir", lambda: str(tmp_path))

    stage4_analyze._ensure_runtime_cache_dirs()

    assert os.environ["MPLCONFIGDIR"] == str(tmp_path / "whaling-mpl")
    assert (tmp_path / "whaling-mpl").exists()


def test_run_output_continues_after_individual_failures(monkeypatch):
    from src.pipeline import stage5_output

    monkeypatch.setattr(stage5_output, "ensure_output_dirs", lambda: None)
    monkeypatch.setattr(
        stage5_output,
        "generate_paper_outputs",
        lambda: _raise_runtime_error("paper build failed"),
    )
    monkeypatch.setattr(stage5_output, "copy_figures_to_paper", lambda: 3)
    monkeypatch.setattr(stage5_output, "generate_robustness_summary", lambda: None)
    monkeypatch.setattr(stage5_output, "generate_mechanism_summary", lambda: None)

    results = stage5_output.run_output(include_figures=True)

    # Paper outputs should remain at default because the step failed
    assert results["paper_outputs"] is None
    assert results["figures_copied"] == 3


def test_clean_wsl_saves_compatibility_alias_and_runs_crosswalk(monkeypatch, tmp_path):
    from src.pipeline import stage2_clean

    raw_wsl = tmp_path / "raw_wsl"
    (raw_wsl / "1843").mkdir(parents=True)
    (raw_wsl / "1843" / "sample.pdf").write_bytes(b"%PDF-1.4")

    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "voyages_master.parquet").write_bytes(b"voyages")

    df = __import__("pandas").DataFrame(
        [{"wsl_event_id": "evt1", "voyage_id": None, "event_type": "OTHER"}]
    )

    saved_paths = []
    crosswalk_calls = []

    def fake_save_output_variants(frame, primary, *aliases):
        saved_paths.append((primary, aliases))
        primary.parent.mkdir(parents=True, exist_ok=True)
        primary.write_bytes(b"stub")
        for alias in aliases:
            alias.write_bytes(b"stub")

    monkeypatch.setattr(stage2_clean, "_save_output_variants", fake_save_output_variants)

    def fake_extract_all_wsl_events(path):
        assert path == raw_wsl
        return df

    def fake_run_wsl_crosswalk(events_path=None, voyages_path=None, confidence_threshold=0.7):
        crosswalk_calls.append((events_path, voyages_path, confidence_threshold))
        return (
            __import__("pandas").DataFrame({"voyage_id": ["AV1"]}),
            __import__("pandas").DataFrame({"voyage_id": ["AV1"]}),
        )

    monkeypatch.setattr("src.config.RAW_WSL", raw_wsl)
    monkeypatch.setattr("src.config.STAGING_DIR", staging)

    import src.parsing.wsl_event_extractor as extractor
    import src.entities.wsl_voyage_matcher as matcher

    monkeypatch.setattr(extractor, "extract_all_wsl_events", fake_extract_all_wsl_events)
    monkeypatch.setattr(matcher, "run_wsl_crosswalk", fake_run_wsl_crosswalk)

    assert stage2_clean.clean_wsl() is True
    assert saved_paths == [
        (
            staging / "wsl_events.parquet",
            (staging / "wsl_extracted_events.parquet",),
        )
    ]
    assert crosswalk_calls == [
        (
            staging / "wsl_events.parquet",
            staging / "voyages_master.parquet",
            0.7,
        )
    ]


def test_run_clean_summary_reports_skipped_steps(monkeypatch, caplog):
    from src.pipeline import stage2_clean

    monkeypatch.setattr(stage2_clean, "clean_voyages", lambda: True)
    monkeypatch.setattr(stage2_clean, "clean_crew", lambda: True)
    monkeypatch.setattr(stage2_clean, "clean_logbooks", lambda: True)
    monkeypatch.setattr(stage2_clean, "clean_registers", lambda: None)
    monkeypatch.setattr(stage2_clean, "clean_starbuck", lambda: True)
    monkeypatch.setattr(stage2_clean, "clean_maury", lambda: None)
    monkeypatch.setattr(stage2_clean, "clean_wsl", lambda: True)

    with caplog.at_level(logging.INFO):
        results = stage2_clean.run_clean()

    assert results["registers"] is None
    assert results["maury"] is None
    assert "Stage 2 complete: 5 successful, 2 skipped, 0 failed" in caplog.text


def test_run_merge_summary_reports_skipped_steps(monkeypatch, caplog):
    from src.pipeline import stage3_merge

    monkeypatch.setattr(stage3_merge, "compute_labor_metrics", lambda: True)
    monkeypatch.setattr(stage3_merge, "compute_route_exposure", lambda: True)
    monkeypatch.setattr(stage3_merge, "resolve_entities", lambda: True)
    monkeypatch.setattr(stage3_merge, "build_entity_crosswalks", lambda: True)
    monkeypatch.setattr(stage3_merge, "assemble_voyages", lambda: True)
    monkeypatch.setattr(stage3_merge, "link_captains", lambda: None)
    monkeypatch.setattr(stage3_merge, "assemble_captains", lambda: True)
    monkeypatch.setattr(stage3_merge, "augment_voyages", lambda: True)
    monkeypatch.setattr(stage3_merge, "merge_climate_data", lambda: True)

    with caplog.at_level(logging.INFO):
        results = stage3_merge.run_merge()

    assert results["captain_linkage"] is None
    assert "Stage 3 complete: 8 successful, 1 skipped, 0 failed" in caplog.text
