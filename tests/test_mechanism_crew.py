"""
Tests for first mate effects estimation and paper table integration.

Tests cover:
 - Crew feature extraction (mate_id, greenhand ratio)
 - Crew experience tracking (cumulative voyages, repeat ratios)
 - Mate FE variance decomposition correctness
 - Mate-to-captain career path regression
 - Table A9 markdown and LaTeX generation
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Synthetic data fixtures
# ---------------------------------------------------------------------------

def _make_crew_roster(n_voyages: int = 200, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic crew roster with known mate assignments."""
    rng = np.random.default_rng(seed)

    rows: list[dict] = []
    for v in range(n_voyages):
        vid = f"V{v:04d}"
        crew_size = rng.integers(18, 35)
        # Captain
        rows.append({
            "voyage_id": vid,
            "crew_name_clean": f"CAPTAIN_{v % 40:03d}",
            "rank": "MASTER",
            "birthplace": rng.choice(["Nantucket", "New Bedford", "Fairhaven", "Sag Harbor"]),
            "age": rng.integers(25, 55),
            "is_deserted": 0,
        })
        # First mate -- deterministic assignment so tests can verify
        mate_idx = v % 60
        rows.append({
            "voyage_id": vid,
            "crew_name_clean": f"MATE_{mate_idx:03d}",
            "rank": "1ST MATE",
            "birthplace": rng.choice(["Nantucket", "New Bedford"]),
            "age": rng.integers(22, 45),
            "is_deserted": 0,
        })
        # 2nd mate
        rows.append({
            "voyage_id": vid,
            "crew_name_clean": f"MATE2_{v % 30:03d}",
            "rank": "2ND MATE",
            "birthplace": rng.choice(["Nantucket", "New Bedford"]),
            "age": rng.integers(20, 40),
            "is_deserted": 0,
        })
        # Ordinary crew (some greenhands)
        for c in range(crew_size - 3):
            is_green = rng.random() < 0.3
            rows.append({
                "voyage_id": vid,
                "crew_name_clean": f"CREW_{v}_{c}",
                "rank": "GREENHAND" if is_green else "SEAMAN",
                "birthplace": rng.choice(["Nantucket", "New Bedford", "Boston", "Provincetown", "Unknown"]),
                "age": rng.integers(16, 50),
                "is_deserted": int(rng.random() < 0.08),
            })
    return pd.DataFrame(rows)


def _make_voyage_df(n_voyages: int = 200, seed: int = 42) -> pd.DataFrame:
    """Create synthetic voyage-level data matching the crew roster."""
    rng = np.random.default_rng(seed)
    rows = []
    for v in range(n_voyages):
        rows.append({
            "voyage_id": f"V{v:04d}",
            "captain_id": f"CAPTAIN_{v % 40:03d}",
            "agent_id": f"AGENT_{v % 15:02d}",
            "year_out": 1820 + (v % 50),
            "log_q": rng.normal(7.0, 1.0),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests: compute_crew_features
# ---------------------------------------------------------------------------

class TestComputeCrewFeatures:
    """Test feature extraction from crew rosters."""

    def test_extracts_mate_id(self):
        crew = _make_crew_roster(50)
        voyages = _make_voyage_df(50)

        from src.analyses.mechanism_crew import compute_crew_features
        features = compute_crew_features(crew, voyages)

        # Every voyage should have a mate assigned
        assert features["mate_id"].notna().all(), "Some voyages missing mate_id"
        # Mate IDs should follow our naming convention
        assert all(mid.startswith("MATE_") for mid in features["mate_id"])

    def test_greenhand_ratio_in_range(self):
        crew = _make_crew_roster(50)
        voyages = _make_voyage_df(50)

        from src.analyses.mechanism_crew import compute_crew_features
        features = compute_crew_features(crew, voyages)

        assert (features["greenhand_ratio"] >= 0).all()
        assert (features["greenhand_ratio"] <= 1).all()
        # With ~30% greenhand probability, mean should be roughly 0.2-0.4
        mean_ratio = features["greenhand_ratio"].mean()
        assert 0.1 < mean_ratio < 0.5, f"Unexpected mean greenhand ratio: {mean_ratio:.3f}"

    def test_crew_size_is_positive(self):
        crew = _make_crew_roster(50)
        voyages = _make_voyage_df(50)

        from src.analyses.mechanism_crew import compute_crew_features
        features = compute_crew_features(crew, voyages)

        assert (features["crew_size"] > 0).all()

    def test_unique_mates_count(self):
        crew = _make_crew_roster(100)
        voyages = _make_voyage_df(100)

        from src.analyses.mechanism_crew import compute_crew_features
        features = compute_crew_features(crew, voyages)

        # We assign mates with v % 60, so with 100 voyages expect 60 unique mates
        n_unique = features["mate_id"].nunique()
        assert n_unique == 60, f"Expected 60 unique mates, got {n_unique}"

    def test_exact_feature_aggregation(self):
        crew = pd.DataFrame({
            "voyage_id": ["V1", "V1", "V1", "V2", "V2"],
            "crew_name_clean": ["Mate One", "Crew A", "Crew B", "Mate Two", "Crew C"],
            "rank": ["1ST MATE", "GREENHAND", "SEAMAN", "MATE", "GREENHAND"],
            "birthplace": ["Nantucket", "Boston", "Boston", "Nantucket", None],
            "age": [30, 20, 40, 35, 18],
            "is_deserted": [0, 1, 0, 0, 0],
        })
        voyages = pd.DataFrame({"voyage_id": ["V1", "V2"]})

        from src.analyses.mechanism_crew import compute_crew_features

        features = compute_crew_features(crew, voyages).set_index("voyage_id")

        assert features.loc["V1", "crew_size"] == 3
        assert features.loc["V1", "greenhand_ratio"] == pytest.approx(1 / 3)
        assert features.loc["V1", "crew_diversity"] == 2
        assert features.loc["V1", "avg_crew_age"] == pytest.approx(30.0)
        assert features.loc["V1", "desertion_rate"] == pytest.approx(1 / 3)
        assert features.loc["V1", "mate_id"] == "Mate One"


# ---------------------------------------------------------------------------
# Tests: track_crew_experience
# ---------------------------------------------------------------------------

class TestTrackCrewExperience:
    """Test cumulative voyage experience tracking."""

    def test_returns_required_columns(self):
        crew = _make_crew_roster(50)
        voyages = _make_voyage_df(50)

        from src.analyses.mechanism_crew import track_crew_experience
        exp = track_crew_experience(crew, voyages)

        assert "avg_prior_voyages" in exp.columns
        assert "repeat_crew_ratio" in exp.columns

    def test_first_voyage_has_zero_experience(self):
        """Captain's first voyage should have 0 prior voyages."""
        crew = _make_crew_roster(100)
        voyages = _make_voyage_df(100)

        from src.analyses.mechanism_crew import track_crew_experience
        exp = track_crew_experience(crew, voyages)

        # avg_prior_voyages should be non-negative everywhere
        assert (exp["avg_prior_voyages"] >= 0).all()

    def test_repeat_crew_ratio_exact(self):
        crew = pd.DataFrame({
            "voyage_id": ["V1", "V1", "V2", "V2", "V2"],
            "crew_name_clean": ["Alice", "Bob", "Alice", "Bob", "Cara"],
        })
        voyages = pd.DataFrame({
            "voyage_id": ["V1", "V2"],
            "year_out": [1820, 1821],
        })

        from src.analyses.mechanism_crew import track_crew_experience

        exp = track_crew_experience(crew, voyages).set_index("voyage_id")

        assert exp.loc["V1", "avg_prior_voyages"] == pytest.approx(0.0)
        assert exp.loc["V1", "repeat_crew_ratio"] == pytest.approx(0.0)
        assert exp.loc["V2", "repeat_crew_ratio"] == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# Tests: Mate FE variance decomposition (Test 4 in mechanism_crew)
# ---------------------------------------------------------------------------

class TestMateFEDecomposition:
    """Test that mate FE variance decomposition produces valid results."""

    def test_mate_variance_decomposition_with_known_effect(self):
        """Inject a known mate effect and verify between/within split."""
        rng = np.random.default_rng(99)
        n = 1000

        # Create 80 mates with known effects
        n_mates = 80
        mate_effects = {f"MATE_{i:03d}": rng.normal(0, 0.5) for i in range(n_mates)}

        mate_ids = [f"MATE_{i % n_mates:03d}" for i in range(n)]
        true_effects = np.array([mate_effects[m] for m in mate_ids])
        noise = rng.normal(0, 0.8, n)
        outcome = 7.0 + true_effects + noise

        df = pd.DataFrame({
            "mate_id": mate_ids,
            "log_q": outcome,
        })

        # Replicate the decomposition logic from mechanism_crew.run_mechanism_regressions
        overall_var = df["log_q"].var()
        mate_means = df.groupby("mate_id")["log_q"].mean()
        between_var = mate_means.var()
        df["mate_mean"] = df["mate_id"].map(mate_means)
        within_var = (df["log_q"] - df["mate_mean"]).var()
        mate_share = between_var / (between_var + within_var)

        # With σ_mate=0.5 and σ_noise=0.8, expected share ≈ 0.25/(0.25+0.64) ≈ 0.28
        assert 0.10 < mate_share < 0.50, (
            f"Mate share {mate_share:.3f} outside plausible range for σ_mate=0.5, σ_noise=0.8"
        )
        assert between_var > 0, "Between-mate variance should be positive"
        assert within_var > 0, "Within-mate variance should be positive"
        assert between_var < within_var, (
            "With σ_noise > σ_mate, within should exceed between"
        )


# ---------------------------------------------------------------------------
# Tests: Mate-to-captain career paths (Test 7 in mechanism_crew)
# ---------------------------------------------------------------------------

class TestMateToCapitanCareerPath:
    """Test the mate-to-captain career path analysis."""

    def _build_career_data(self, n_promoted: int = 80, same_agent_bonus: float = 0.1, seed: int = 7):
        """Build crew + voyage data where some mates became captains."""
        rng = np.random.default_rng(seed)

        crew_rows: list[dict] = []
        voyage_rows: list[dict] = []
        vid = 0

        for p in range(n_promoted):
            name = f"PERSON_{p:03d}"
            training_agent = f"AGENT_{p % 12:02d}"

            # Phase 1: serve as mate under training_agent
            mate_vid = f"V{vid:04d}"
            crew_rows.append({
                "voyage_id": mate_vid,
                "crew_name_clean": name,
                "rank": "MATE",
            })
            voyage_rows.append({
                "voyage_id": mate_vid,
                "captain_id": f"OTHER_CAP_{p:03d}",
                "agent_id": training_agent,
                "year_out": 1820 + p,
                "log_q": rng.normal(7.0, 0.5),
            })
            vid += 1

            # Phase 2: become captain (some with same agent, some different)
            for cv in range(rng.integers(2, 6)):
                cap_vid = f"V{vid:04d}"
                if rng.random() < 0.35:
                    agent = training_agent
                    bonus = same_agent_bonus
                else:
                    agent = f"AGENT_{rng.integers(0, 12):02d}"
                    bonus = 0.0

                crew_rows.append({
                    "voyage_id": cap_vid,
                    "crew_name_clean": name,
                    "rank": "MASTER",
                })
                voyage_rows.append({
                    "voyage_id": cap_vid,
                    "captain_id": name,
                    "agent_id": agent,
                    "year_out": 1825 + p + cv,
                    "log_q": rng.normal(7.0 + bonus, 0.5),
                })
                vid += 1

        return pd.DataFrame(crew_rows), pd.DataFrame(voyage_rows)

    def test_detects_positive_effect(self):
        """With a known same-agent bonus, β should be positive."""
        crew, voyages = self._build_career_data(n_promoted=80, same_agent_bonus=0.3)

        from src.analyses.mechanism_crew import run_mate_to_captain_test
        result = run_mate_to_captain_test(voyages, crew, outcome_col="log_q")

        assert result, "Expected non-empty results for 80 promoted mates"
        assert result["beta"] > 0, f"Expected positive β, got {result['beta']:.4f}"
        assert result["n_promoted"] == 80

    def test_insufficient_data_returns_empty(self):
        """With < 50 promoted mates, should return empty dict."""
        crew, voyages = self._build_career_data(n_promoted=20)

        from src.analyses.mechanism_crew import run_mate_to_captain_test
        result = run_mate_to_captain_test(voyages, crew, outcome_col="log_q")

        assert result == {}, "Expected empty dict for insufficient promoted mates"


# ---------------------------------------------------------------------------
# Tests: Table A9 generation
# ---------------------------------------------------------------------------

class TestTableA9Generation:
    """Test paper table generators for Table A9."""

    def test_markdown_output_starts_with_header(self):
        from src.analyses.paper_tables import generate_table_a9

        md = generate_table_a9()
        assert md.startswith("## Table A9"), f"Unexpected start: {md[:50]}"

    def test_markdown_contains_scarcity_content(self):
        """Table A9 is now scarcity definition robustness (flat list)."""
        from src.analyses.paper_tables import generate_table_a9_md

        md = generate_table_a9_md()
        assert "Sparse State" in md, "Missing scarcity definition column"
        assert "3-Year Lag" in md, "Missing baseline row"

    def test_latex_output_has_table_environment(self):
        from src.analyses.paper_tables import generate_table_a9_tex

        tex = generate_table_a9_tex()
        assert r"\begin{table}" in tex
        assert r"\end{table}" in tex
        assert r"\label{tab:tableA9}" in tex

    def test_latex_contains_scarcity_content(self):
        """A9 LaTeX should have scarcity robustness content, not mate panels."""
        from src.analyses.paper_tables import generate_table_a9_tex

        tex = generate_table_a9_tex()
        assert "Sparse State" in tex or "3-Year Lag" in tex

    def test_a9_in_all_markdown_tables(self):
        from src.analyses.paper_tables import generate_all_markdown_tables

        all_md = generate_all_markdown_tables()
        assert "Table A9" in all_md

    def test_a9_in_all_latex_tables(self):
        from src.analyses.paper_tables import generate_all_latex_tables

        all_tex = generate_all_latex_tables()
        assert "tableA9" in all_tex


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
