from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _lower_ml_thresholds(monkeypatch):
    from config import ML_SHIFT_CONFIG

    monkeypatch.setattr(ML_SHIFT_CONFIG, "min_training_rows", 4)
    monkeypatch.setattr(ML_SHIFT_CONFIG, "min_text_training_rows", 2)
    monkeypatch.setattr(ML_SHIFT_CONFIG, "anomaly_min_rows", 8)


def test_entity_resolver_reports_observation_counts(monkeypatch):
    _lower_ml_thresholds(monkeypatch)

    from entities.entity_resolver import EntityResolver

    voyages = pd.DataFrame({
        "vessel_name_clean": ["AWASHONKS", "AWASHONKS", "COLUMBIA"],
        "captain_name_clean": ["WOOD", "WOOD", "SWAIN"],
        "agent_name_clean": ["ROTCH", "ROTCH", "HUSSEY"],
        "home_port": ["NEW BEDFORD", "NEW BEDFORD", "NANTUCKET"],
        "rig": ["SHIP", "SHIP", "BARK"],
        "year_out": [1850, 1851, 1849],
        "port_out": ["NEW BEDFORD", "NEW BEDFORD", "NANTUCKET"],
    })

    crosswalk = EntityResolver().resolve_vessels(voyages)
    counts = crosswalk.set_index("vessel_name_clean")["observation_count"].to_dict()

    assert counts["AWASHONKS"] == 2
    assert counts["COLUMBIA"] == 1


def test_record_linker_applies_match_probabilities(monkeypatch):
    _lower_ml_thresholds(monkeypatch)

    from linkage.record_linker import RecordLinker

    results = pd.DataFrame({
        "captain_id": ["C1"] * 4 + ["C2"] * 4,
        "match_score": [0.99, 0.95, 0.90, 0.89, 0.25, 0.20, 0.10, 0.05],
        "name_score": [0.99, 0.96, 0.91, 0.88, 0.30, 0.25, 0.15, 0.10],
        "age_score": [1.0, 0.9, 0.8, 0.9, 0.2, 0.3, 0.1, 0.2],
        "geo_score": [0.9, 0.85, 0.80, 0.75, 0.1, 0.2, 0.1, 0.1],
        "occ_score": [0.8, 0.85, 0.70, 0.75, 0.1, 0.1, 0.0, 0.0],
        "spouse_validated": [1, 1, 0, 1, 0, 0, 0, 0],
        "match_method": [
            "deterministic", "scored", "scored", "scored",
            "scored", "scored", "scored", "scored",
        ],
    })

    enhanced = RecordLinker()._apply_ml_match_probabilities(results)

    assert "match_probability" in enhanced.columns
    assert enhanced["match_model_trained"].all()
    assert "ml_probabilistic" in enhanced["match_method"].values


def test_wsl_event_extractor_emits_ml_columns(monkeypatch):
    _lower_ml_thresholds(monkeypatch)

    from parsing.wsl_event_extractor import events_to_dataframe, extract_events_from_text

    sample_text = """
    The ship AWASHONKS, Capt. Wood, arrived at New Bedford on March 15.
    Sailed from Nantucket, March 10 - Ship COLUMBIA, Swain, Indian Ocean.
    LOST - The brig HESPER of Sag Harbor was wrecked on the coast of Patagonia.
    """

    events = extract_events_from_text(
        text=sample_text,
        issue_id="wsl_1850_03_18",
        issue_year=1850,
        issue_month=3,
        issue_day=18,
    )
    df = events_to_dataframe(events)

    assert len(df) >= 2
    assert {"heuristic_event_type", "event_type_probability", "event_type_model_trained"}.issubset(df.columns)
    assert df["event_type_probability"].notna().all()


def test_wsl_voyage_matcher_returns_probability_columns(monkeypatch):
    _lower_ml_thresholds(monkeypatch)

    from entities.wsl_voyage_matcher import match_events_to_voyages

    events_df = pd.DataFrame({
        "wsl_event_id": ["E1"],
        "vessel_name_clean": ["AWASHONKS"],
        "captain_name_clean": ["WOOD"],
        "event_date": ["1850-03-15"],
        "port_name_clean": ["NEW BEDFORD"],
    })
    voyages_df = pd.DataFrame({
        "voyage_id": ["V1"],
        "vessel_name_clean": ["AWASHONKS"],
        "captain_name_clean": ["WOOD"],
        "date_out": ["1850-01-01"],
        "date_in": ["1850-07-01"],
        "port_out": ["NANTUCKET"],
        "port_in": ["NEW BEDFORD"],
    })

    crosswalk_df, diagnostics_df = match_events_to_voyages(events_df, voyages_df)

    assert crosswalk_df.loc[0, "voyage_id"] == "V1"
    assert "match_probability" in crosswalk_df.columns
    assert "match_probability" in diagnostics_df.columns


def test_parse_crew_lists_adds_desertion_probability(monkeypatch, tmp_path):
    _lower_ml_thresholds(monkeypatch)

    from parsing.crew_parser import parse_crew_lists

    crew_csv = tmp_path / "crew.csv"
    crew_csv.write_text(
        "voyage_id,name,remarks,age\n"
        "V1,John Doe,Deserted at Honolulu,24\n"
        "V1,James Hall,Died at sea,31\n"
        "V2,Will Smith,Run away,19\n"
        "V2,Henry Brown,Discharged,28\n",
        encoding="utf-8",
    )

    df = parse_crew_lists(crew_csv)

    assert "desertion_probability" in df.columns
    assert "desertion_model_trained" in df.columns
    assert df["desertion_probability"].between(0, 1).all()
    assert df["is_deserted"].sum() >= 2


def test_parse_registers_adds_parse_quality_probability(monkeypatch, tmp_path):
    _lower_ml_thresholds(monkeypatch)

    from parsing.register_parser import parse_registers

    register_txt = tmp_path / "register.txt"
    register_txt.write_text(
        "1845\n"
        "AWASHONKS A1 12000\n"
        "MINERVA B 9000\n",
        encoding="utf-8",
    )

    df = parse_registers(register_txt)

    assert len(df) >= 2
    assert "parse_quality_probability" in df.columns
    assert df["parse_quality_probability"].between(0, 1).all()


def test_parse_starbuck_adds_parse_probability(monkeypatch, tmp_path):
    _lower_ml_thresholds(monkeypatch)

    import parsing.starbuck_parser as starbuck_parser

    monkeypatch.setattr(
        starbuck_parser,
        "save_starbuck_voyages",
        lambda voyages, output_path=None: tmp_path / "starbuck.parquet",
    )

    ocr_txt = tmp_path / "starbuck.txt"
    ocr_txt.write_text(
        "VOYAGES FROM NEW BEDFORD\n"
        "AWASHONKS, 1845-1848\n"
        "MINERVA sailed 1852\n",
        encoding="utf-8",
    )

    df = starbuck_parser.parse_starbuck(ocr_txt)

    assert len(df) >= 2
    assert "parse_probability" in df.columns
    assert df["parse_probability"].between(0, 1).all()


def test_maury_matcher_exposes_probability_columns(monkeypatch):
    _lower_ml_thresholds(monkeypatch)

    from entities.maury_voyage_matcher import match_positions_to_voyages

    positions_df = pd.DataFrame({
        "maury_obs_id": ["M1", "M2"],
        "obs_date": ["1850-06-01", "1850-06-15"],
        "vessel_name_clean": ["AWASHONKS", "AWASHONKS"],
        "captain_name_clean": ["WOOD", "WOOD"],
        "lat": [65.0, 66.0],
    })
    voyages_df = pd.DataFrame({
        "voyage_id": ["V1"],
        "vessel_name_clean": ["AWASHONKS"],
        "captain_name_clean": ["WOOD"],
        "date_out": ["1850-01-01"],
        "date_in": ["1851-01-01"],
    })

    crosswalk_df = match_positions_to_voyages(positions_df, voyages_df)

    assert "match_probability" in crosswalk_df.columns
    assert "heuristic_match_score" in crosswalk_df.columns
    assert set(crosswalk_df["voyage_id"]) == {"V1"}


def test_starbuck_reconciler_exposes_probability_columns(monkeypatch):
    _lower_ml_thresholds(monkeypatch)

    from entities.starbuck_reconciler import match_starbuck_to_aowv

    starbuck_df = pd.DataFrame({
        "starbuck_row_id": ["S1"],
        "vessel_name_clean": ["AWASHONKS"],
        "departure_year": [1850],
        "return_year": [1852],
        "home_port_clean": ["NEW BEDFORD"],
    })
    aowv_df = pd.DataFrame({
        "voyage_id": ["V1"],
        "vessel_name_clean": ["AWASHONKS"],
        "date_out": ["1850-01-01"],
        "date_in": ["1852-06-01"],
        "home_port": ["NEW BEDFORD"],
    })

    reconciliation_df = match_starbuck_to_aowv(starbuck_df, aowv_df)

    assert reconciliation_df.loc[0, "best_voyage_id"] == "V1"
    assert "match_probability" in reconciliation_df.columns
    assert "heuristic_match_score" in reconciliation_df.columns


def test_validate_analysis_voyage_adds_anomaly_check(monkeypatch):
    _lower_ml_thresholds(monkeypatch)

    from qa.validators import validate_analysis_voyage

    df = pd.DataFrame({
        "voyage_id": [f"V{i}" for i in range(12)],
        "date_out": pd.date_range("1850-01-01", periods=12, freq="MS").strftime("%Y-%m-%d"),
        "date_in": pd.date_range("1850-02-01", periods=12, freq="MS").strftime("%Y-%m-%d"),
        "q_oil_bbl": [1200] * 11 + [12000],
        "q_bone_lbs": [500] * 11 + [9000],
        "desertion_rate": [0.1] * 11 + [0.95],
        "duration_days": [400] * 11 + [2500],
        "year_out": [1850] * 11 + [1900],
    })

    results = validate_analysis_voyage(df)

    assert any(result.check_name == "voyage_multivariate_anomalies" for result in results)
