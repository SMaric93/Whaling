from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analyses.run_wsl_reliability_ml import _prepare_predeparture_panel, _prepare_triage_panel
from src.analyses.wsl_reliability_ml.departure_information import compute_information_stock_features
from src.analyses.wsl_reliability_ml.policy_layer import (
    evaluate_policy_frontiers,
    fit_predeparture_policy_models,
    score_predeparture_policies,
)
from src.analyses.wsl_reliability_ml.remarks_taxonomy import (
    _apply_rule_taxonomy,
    build_remarks_canonical_text,
    train_remarks_models,
)
from src.analyses.wsl_reliability_ml.utils import (
    WSLReliabilityConfig,
    attach_voyage_linkage,
    build_voyage_linkage,
    load_wsl_cleaned_events,
    save_dataframe,
)
from src.analyses.wsl_reliability_ml.voyage_state_model import (
    create_state_anchor_labels,
    fit_voyage_state_model,
    infer_voyage_states,
)


def test_load_wsl_cleaned_events_maps_registry_pages(tmp_path: Path) -> None:
    cleaned_path = tmp_path / "events.jsonl"
    record = {
        "pdf": "wsl_1843_03_17.pdf",
        "page": 5,
        "page_key": "wsl_1843_03_17.pdf:p5",
        "page_type": "sparse",
        "page_route": "default",
        "events": [
            {
                "vessel_name": "Barclay",
                "captain": "Cook",
                "agent": "Davis & Corey",
                "event_type": "dep",
                "port": "Westport",
                "date": "Dec 27, 42",
                "home_port": "Westport",
                "destination": "Atlantic",
                "remarks": None,
                "_raw": {"v": "Barclay"},
                "_flags": ["likely_registry_not_weekly"],
                "_confidence": 0.95,
                "panel_include": False,
                "validation_status": "suspicious",
            }
        ],
    }
    cleaned_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    cfg = WSLReliabilityConfig(
        cleaned_events_path=cleaned_path,
        issue_index_path=tmp_path / "missing_issue_index.parquet",
        output_root=tmp_path / "outputs",
    )
    df = load_wsl_cleaned_events(cfg)
    assert df.loc[0, "page_type"] == "fleet_registry_stock"
    assert str(df.loc[0, "issue_date"].date()) == "1843-03-17"
    assert 0.0 < df.loc[0, "row_weight"] < 0.4


def test_load_wsl_cleaned_events_cache_roundtrip_preserves_nested_fields(tmp_path: Path) -> None:
    cleaned_path = tmp_path / "events.jsonl"
    records = [
        {
            "pdf": "wsl_1843_03_17.pdf",
            "page": 5,
            "page_key": "wsl_1843_03_17.pdf:p5",
            "page_type": "shipping_table",
            "page_route": "default",
            "events": [
                {
                    "vessel_name": "Barclay",
                    "captain": "Cook",
                    "agent": "Davis & Corey",
                    "event_type": "dep",
                    "port": "Westport",
                    "date": "Dec 27, 42",
                    "home_port": "Westport",
                    "destination": "Atlantic",
                    "remarks": "clean",
                    "_raw": {"v": "Barclay"},
                    "_flags": ["likely_registry_not_weekly"],
                    "_confidence": 0.95,
                    "panel_include": True,
                }
            ],
        },
        {
            "pdf": "wsl_1843_03_24.pdf",
            "page": 6,
            "page_key": "wsl_1843_03_24.pdf:p6",
            "page_type": "shipping_table",
            "page_route": "default",
            "events": [
                {
                    "vessel_name": "Coronet",
                    "captain": "Nye",
                    "agent": "Howland",
                    "event_type": "rpt",
                    "port": "New Bedford",
                    "date": "Mar 21, 43",
                    "home_port": "New Bedford",
                    "destination": "Pacific",
                    "remarks": "good catch",
                    "_raw": ["overflow", "notes"],
                    "_flags": [],
                    "_confidence": 0.88,
                    "panel_include": True,
                }
            ],
        },
    ]
    cleaned_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    cfg = WSLReliabilityConfig(
        cleaned_events_path=cleaned_path,
        issue_index_path=tmp_path / "missing_issue_index.parquet",
        output_root=tmp_path / "outputs",
    )

    first = load_wsl_cleaned_events(cfg)
    second = load_wsl_cleaned_events(cfg)

    assert isinstance(first.loc[0, "_raw"], dict)
    assert isinstance(first.loc[1, "_raw"], list)
    assert isinstance(second.loc[0, "_raw"], dict)
    assert isinstance(second.loc[1, "_raw"], list)
    assert isinstance(second.loc[0, "_flags"], list)
    assert isinstance(second.loc[0, "destination_basin_probs"], dict)
    assert len(second) == 2


def test_load_wsl_cleaned_events_repairs_duplicate_event_row_ids(tmp_path: Path) -> None:
    cleaned_path = tmp_path / "events.jsonl"
    record = {
        "pdf": "wsl_1843_03_17.pdf",
        "page": 5,
        "page_key": "wsl_1843_03_17.pdf:p5",
        "page_type": "shipping_table",
        "page_route": "default",
        "events": [
            {
                "vessel_name": "Barclay",
                "captain": "Cook",
                "agent": "Davis & Corey",
                "event_type": "dep",
                "port": "Westport",
                "date": "Dec 27, 42",
                "home_port": "Westport",
                "destination": "Atlantic",
                "remarks": "clean",
                "_raw": {"v": "Barclay"},
                "_flags": [],
                "_confidence": 0.95,
                "panel_include": True,
            }
        ],
    }
    cleaned_path.write_text("\n".join([json.dumps(record), json.dumps(record)]) + "\n", encoding="utf-8")
    cfg = WSLReliabilityConfig(
        cleaned_events_path=cleaned_path,
        issue_index_path=tmp_path / "missing_issue_index.parquet",
        output_root=tmp_path / "outputs",
    )
    df = load_wsl_cleaned_events(cfg)
    assert len(df) == 2
    assert df["event_row_id"].nunique() == 2


def test_remarks_canonical_text_uses_overflow_and_rules() -> None:
    events = pd.DataFrame(
        [
            {
                "event_row_id": "evt_1",
                "remarks": None,
                "destination": "Pacific",
                "agent": "Smith & Co",
                "reported_by": "reported leaking and bound home",
                "_raw": {"date": "lat 42 s"},
                "_flags": ["status_in_port_field"],
                "event_type": "rpt",
                "page_type": "weekly_event_flow",
                "page_type_confidence": 0.9,
                "row_weight": 0.8,
                "_confidence": 0.9,
                "validation_status": "suspicious",
            }
        ]
    )
    canonical = build_remarks_canonical_text(events)
    ruled = _apply_rule_taxonomy(canonical)
    assert "reported leaking and bound home" in ruled.loc[0, "remarks_canonical_text"]
    assert "lat 42 s" in ruled.loc[0, "remarks_canonical_text"]
    assert ruled.loc[0, "remarks_from_overflow_only"]
    assert ruled.loc[0, "rule_primary_class"] == "distress_hazard"
    assert "bound_home" in ruled.loc[0, "rule_secondary_tags"]


def test_build_voyage_linkage_tolerates_missing_optional_voyage_columns() -> None:
    events = pd.DataFrame(
        [
            {
                "event_row_id": "evt_1",
                "vessel_name_norm": "BARCLAY",
                "captain_norm": "COOK",
                "home_port_norm": "WESTPORT",
                "port_norm": np.nan,
                "parsed_event_date_if_available": pd.Timestamp("1843-03-01"),
                "issue_date": pd.Timestamp("1843-03-17"),
                "row_weight": 0.9,
            }
        ]
    )
    voyages = pd.DataFrame(
        [
            {
                "voyage_id": "V1",
                "vessel_name_norm": "BARCLAY",
                "captain_name_norm": "COOK",
                "agent_name_norm": "DAVIS & COREY",
                "home_port_norm": "WESTPORT",
                "port_out_norm": "WESTPORT",
                "port_in_norm": "WESTPORT",
                "date_out": pd.Timestamp("1842-12-27"),
                "date_in": pd.Timestamp("1844-01-01"),
                "captain_id": "C1",
                "agent_id": "A1",
                "vessel_id": "VV1",
            }
        ]
    )
    linkage = build_voyage_linkage(events, voyages, WSLReliabilityConfig())
    assert linkage.loc[0, "voyage_id"] == "V1"
    assert pd.isna(linkage.loc[0, "voyage_basin"])
    assert pd.isna(linkage.loc[0, "voyage_theater"])
    assert pd.isna(linkage.loc[0, "voyage_major_ground"])


def test_train_remarks_models_handles_downsampled_weak_rule_training() -> None:
    rows = []
    for idx in range(40):
        rows.append(
            {
                "event_row_id": f"r_{idx}",
                "remarks": "good catch and whales sighted",
                "destination": "Pacific",
                "agent": "Agent A",
                "reported_by": None,
                "_raw": {},
                "_flags": [],
                "event_type": "rpt",
                "page_type": "weekly_event_flow",
                "page_type_confidence": 0.9,
                "row_weight": 0.9,
                "_confidence": 0.9,
                "validation_status": "valid",
            }
        )
    for idx in range(40, 80):
        rows.append(
            {
                "event_row_id": f"r_{idx}",
                "remarks": "repairing in port after leaking",
                "destination": "Atlantic",
                "agent": "Agent B",
                "reported_by": None,
                "_raw": {},
                "_flags": [],
                "event_type": "inp",
                "page_type": "weekly_event_flow",
                "page_type_confidence": 0.9,
                "row_weight": 0.8,
                "_confidence": 0.85,
                "validation_status": "valid",
            }
        )
    for idx in range(80, 82):
        rows.append(
            {
                "event_row_id": f"r_{idx}",
                "remarks": "wrecked and abandoned",
                "destination": "Atlantic",
                "agent": "Agent C",
                "reported_by": None,
                "_raw": {},
                "_flags": [],
                "event_type": "wrk",
                "page_type": "weekly_event_flow",
                "page_type_confidence": 0.95,
                "row_weight": 0.95,
                "_confidence": 0.95,
                "validation_status": "valid",
            }
        )
    cfg = WSLReliabilityConfig(remarks_max_train_rows=50, remarks_min_tag_support=2)
    models = train_remarks_models(pd.DataFrame(rows), cfg)
    assert models["metrics"]["n_train"] > 0
    assert set(models["class_order"]) >= {"positive_productivity", "distress_hazard"}


def test_save_dataframe_serializes_nested_object_columns_for_parquet(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "event_row_id": ["a", "b", "c"],
            "_raw": [{"a": 1}, ["overflow"], "clean"],
            "_flags": [["x"], [], ["y", "z"]],
            "secondary_tags": [["good_catch"], ["repairing"], []],
        }
    )
    path = tmp_path / "nested.parquet"
    save_dataframe(frame, path)
    loaded = pd.read_parquet(path)
    assert loaded.loc[0, "_raw"] == '{"a": 1}'
    assert loaded.loc[1, "_raw"] == '["overflow"]'
    assert loaded.loc[2, "_raw"] == "clean"
    assert loaded.loc[0, "_flags"] == '["x"]'


def test_attach_voyage_linkage_is_idempotent_for_existing_linkage_columns() -> None:
    events = pd.DataFrame(
        [
            {
                "event_row_id": "evt_1",
                "voyage_id": pd.NA,
                "episode_fallback_key": "episode_1",
                "row_weight": 1.0,
                "entity_link_uncertain": True,
            }
        ]
    )
    linkage = pd.DataFrame(
        [
            {
                "event_row_id": "evt_1",
                "voyage_id": "V1",
                "episode_fallback_key": "episode_1",
                "entity_link_uncertain": True,
                "linkage_confidence": 0.6,
            }
        ]
    )
    cfg = WSLReliabilityConfig()
    first = attach_voyage_linkage(events, linkage, cfg)
    second = attach_voyage_linkage(first, linkage, cfg)
    expected_weight = cfg.flag_penalties["entity_link_uncertain"]
    assert first.loc[0, "voyage_id"] == "V1"
    assert second.loc[0, "voyage_id"] == "V1"
    assert first.loc[0, "row_weight"] == pytest.approx(expected_weight)
    assert second.loc[0, "row_weight"] == pytest.approx(expected_weight)


def test_prepare_predeparture_panel_handles_missing_home_port_norm() -> None:
    departure = pd.DataFrame(
        [
            {
                "departure_id": "D1",
                "voyage_id": "V1",
                "departure_issue_date": pd.Timestamp("1850-01-01"),
            }
        ]
    )
    features = pd.DataFrame(
        [
            {
                "departure_id": "D1",
                "voyage_id": "V1",
                "departure_issue_date": pd.Timestamp("1850-01-01"),
                "portfolio_information_index": 0.2,
            }
        ]
    )
    voyage_ref = pd.DataFrame([{"voyage_id": "V1", "home_port": "NB", "q_total_index": 10.0}])
    voyage_summary = pd.DataFrame([{"voyage_id": "V1", "ever_bad_state": 0}])
    panel = _prepare_predeparture_panel(departure, features, voyage_ref, voyage_summary)
    assert panel.loc[0, "home_port"] == "NB"
    assert panel.loc[0, "zero_catch_or_failure"] == 0


def test_score_predeparture_policies_handles_missing_optional_columns() -> None:
    scored_df = pd.DataFrame(
        {
            "propensity_score": [0.2, 0.8],
            "mu1__log_output": [1.5, 1.2],
            "mu0__log_output": [1.0, 1.0],
            "mu1__zero_catch_or_failure": [0.2, 0.4],
            "mu0__zero_catch_or_failure": [0.3, 0.5],
            "mu1__distress_burden_days": [10.0, 8.0],
            "mu0__distress_burden_days": [12.0, 9.0],
        }
    )
    scored = score_predeparture_policies(
        pd.DataFrame(),
        {"scored_df": scored_df, "diagnostics": {}},
        WSLReliabilityConfig(),
    )
    assert "score_top_skill_first" in scored.columns
    assert scored["score_top_skill_first"].tolist() == [0.0, 0.0]
    assert scored["score_weak_first"].tolist() == [0.0, 0.0]


def test_prepare_triage_panel_handles_missing_home_port() -> None:
    states = pd.DataFrame(
        [
            {
                "episode_id": "E1",
                "voyage_id": "V1",
                "issue_date": pd.Timestamp("1850-03-01"),
                "most_likely_state": "distress_at_sea",
                "captain_id": "C1",
                "agent_id": "A1",
                "vessel_id": "VV1",
                "p_state__distress_at_sea": 0.9,
                "p_state__in_port_interruption_or_repair": 0.1,
                "p_state__terminal_loss": 0.0,
            }
        ]
    )
    voyage_summary = pd.DataFrame(
        [
            {
                "episode_id": "E1",
                "voyage_id": "V1",
                "captain_id": "C1",
                "agent_id": "A1",
                "vessel_id": "VV1",
            }
        ]
    )
    voyage_ref = pd.DataFrame([{"voyage_id": "V1", "theta_hat_holdout": 0.2}])
    info_panel = pd.DataFrame([{"voyage_id": "V1", "risk_index": 0.4}])
    triage = _prepare_triage_panel(states, voyage_summary, voyage_ref, info_panel)
    assert triage.loc[0, "home_port"] == "UNK"


def test_state_model_keeps_completed_arrival_absorbing() -> None:
    episode = pd.DataFrame(
        [
            {
                "event_row_id": "e1",
                "episode_id": "V1",
                "voyage_id": "V1",
                "issue_date": pd.Timestamp("1850-01-01"),
                "event_type": "dep",
                "mention_order": 0,
                "row_weight": 1.0,
                "page_type": "weekly_event_flow",
                "primary_class": "routine_info",
                "secondary_tags": [],
                "distress_severity_0_4": 0,
                "productivity_polarity_m2_p2": 0,
                "contamination_score_0_3": 0,
            },
            {
                "event_row_id": "e2",
                "episode_id": "V1",
                "voyage_id": "V1",
                "issue_date": pd.Timestamp("1850-02-01"),
                "event_type": "rpt",
                "mention_order": 1,
                "row_weight": 1.0,
                "page_type": "weekly_event_flow",
                "primary_class": "positive_productivity",
                "secondary_tags": ["good_catch"],
                "distress_severity_0_4": 0,
                "productivity_polarity_m2_p2": 1,
                "contamination_score_0_3": 0,
            },
            {
                "event_row_id": "e3",
                "episode_id": "V1",
                "voyage_id": "V1",
                "issue_date": pd.Timestamp("1850-03-01"),
                "event_type": "arr",
                "mention_order": 2,
                "row_weight": 1.0,
                "page_type": "weekly_event_flow",
                "primary_class": "routine_info",
                "secondary_tags": [],
                "distress_severity_0_4": 0,
                "productivity_polarity_m2_p2": 0,
                "contamination_score_0_3": 0,
            },
            {
                "event_row_id": "e4",
                "episode_id": "V1",
                "voyage_id": "V1",
                "issue_date": pd.Timestamp("1850-03-15"),
                "event_type": "rpt",
                "mention_order": 3,
                "row_weight": 1.0,
                "page_type": "weekly_event_flow",
                "primary_class": "routine_info",
                "secondary_tags": [],
                "distress_severity_0_4": 0,
                "productivity_polarity_m2_p2": 0,
                "contamination_score_0_3": 0,
            },
        ]
    )
    cfg = WSLReliabilityConfig()
    anchors = create_state_anchor_labels(episode, episode, cfg)
    model = fit_voyage_state_model(episode.merge(anchors, on="event_row_id"), anchors, cfg)
    inferred = infer_voyage_states(episode.merge(anchors, on="event_row_id"), model, cfg)
    assert inferred.loc[inferred["event_row_id"] == "e3", "most_likely_state"].item() == "completed_arrival"
    assert inferred.loc[inferred["event_row_id"] == "e4", "most_likely_state"].item() == "completed_arrival"


def test_information_stock_uses_only_prior_issue_dates() -> None:
    departure = pd.DataFrame(
        [
            {
                "departure_id": "D1",
                "voyage_id": "V1",
                "departure_issue_date": pd.Timestamp("1850-01-10"),
                "home_port_norm": "NEW BEDFORD",
                "agent_norm": "AGENT A",
                "departure_destination_basin": "Pacific",
            }
        ]
    )
    prior_events = pd.DataFrame(
        [
            {
                "event_row_id": "before",
                "issue_date": pd.Timestamp("1850-01-05"),
                "page_type": "weekly_event_flow",
                "destination_basin": "Pacific",
                "primary_class": "positive_productivity",
                "row_weight": 1.0,
                "_confidence": 1.0,
                "home_port_norm": "NEW BEDFORD",
                "agent_norm": "AGENT A",
                "agent_id": "A1",
                "vessel_name_norm": "SHIP ONE",
                "vessel_id": "VV1",
            },
            {
                "event_row_id": "after",
                "issue_date": pd.Timestamp("1850-01-20"),
                "page_type": "weekly_event_flow",
                "destination_basin": "Pacific",
                "primary_class": "positive_productivity",
                "row_weight": 1.0,
                "_confidence": 1.0,
                "home_port_norm": "NEW BEDFORD",
                "agent_norm": "AGENT A",
                "agent_id": "A1",
                "vessel_name_norm": "SHIP TWO",
                "vessel_id": "VV2",
            },
        ]
    )
    cfg = WSLReliabilityConfig()
    features = compute_information_stock_features(departure, prior_events, cfg)
    expected = np.exp(-5 / 30.0)
    assert features.loc[0, "pub_basin_reports_tau30"] == pytest.approx(expected)
    assert features.loc[0, "pub_basin_positive_mass_tau30"] == pytest.approx(expected)


def test_predeparture_policy_pipeline_smoke() -> None:
    rng = np.random.default_rng(42)
    n = 150
    df = pd.DataFrame(
        {
            "departure_issue_date": pd.date_range("1850-01-01", periods=n, freq="7D"),
            "theta_hat_holdout": rng.normal(size=n),
            "novice": rng.integers(0, 2, size=n),
            "tonnage": rng.normal(300, 40, size=n),
            "home_port": rng.choice(["NB", "NANTUCKET", "NEW LONDON"], size=n),
            "departure_destination_basin": rng.choice(["Pacific", "Atlantic"], size=n),
            "departure_month": rng.integers(1, 13, size=n),
            "departure_decade": 1850,
            "public_information_index": rng.normal(size=n),
            "portfolio_information_index": rng.normal(size=n),
            "risk_index": rng.normal(size=n),
            "information_advantage_index": rng.normal(size=n),
            "agent_recent_bad_state_rate_tau180": rng.uniform(0, 1, size=n),
            "agent_recent_recovery_rate_tau180": rng.uniform(0, 1, size=n),
        }
    )
    support_score = (
        df["portfolio_information_index"] + df["information_advantage_index"] + df["public_information_index"] - df["risk_index"]
    ) / 4
    df["high_support_context_top_quartile"] = (support_score >= support_score.quantile(0.75)).astype(int)
    df["zero_catch_or_failure"] = (
        0.8
        - 0.6 * df["high_support_context_top_quartile"]
        + 0.25 * df["novice"]
        + 0.25 * (df["risk_index"] > 0).astype(int)
        + rng.normal(0, 0.1, size=n)
        > 0.65
    ).astype(int)
    df["log_output"] = (
        3.5
        + 0.5 * df["high_support_context_top_quartile"]
        + 0.3 * df["portfolio_information_index"]
        - 0.2 * df["risk_index"]
        + rng.normal(0, 0.2, size=n)
    )
    df["ever_bad_state"] = (
        0.5
        - 0.2 * df["high_support_context_top_quartile"]
        + 0.2 * (df["risk_index"] > 0).astype(int)
        + rng.normal(0, 0.1, size=n)
        > 0.55
    ).astype(int)
    df["distress_burden_days"] = np.maximum(
        0,
        25
        - 6 * df["high_support_context_top_quartile"]
        + 4 * df["risk_index"]
        + rng.normal(0, 2, size=n),
    )
    cfg = WSLReliabilityConfig(policy_bootstrap_reps=10)
    bundle = fit_predeparture_policy_models(df, cfg)
    scored = score_predeparture_policies(df, bundle, cfg)
    frontier = evaluate_policy_frontiers(scored, cfg)
    assert "utility_uplift" in scored.columns
    assert not frontier.empty
    assert "highest_uplift_first" in set(frontier["policy_name"])
