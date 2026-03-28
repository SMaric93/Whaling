from __future__ import annotations

import csv
from pathlib import Path

from .config import APPENDIX_TABLES, BuildContext, MAIN_TABLES


TEST_DEFINITIONS = [
    (1, "Connected-set representativeness", "src.paper.appendix.tableA03_connected_set", "analysis_voyage_augmented.parquet; outcome_ml_dataset.parquet", "paper_builder", "outputs/paper/appendix/tableA03_connected_set.csv", "appendix"),
    (2, "Mover/switcher representativeness", "src.paper.appendix.tableA05_mover_switcher_balance", "outcome_ml_dataset.parquet", "paper_builder", "outputs/paper/appendix/tableA05_mover_switcher_balance.csv", "appendix"),
    (3, "Held-out reliability for theta and psi", "src.paper.tables.table02_types", "akm effects; split_sample_stability.csv", "paper_builder", "outputs/paper/tables/table02_types.csv", "main"),
    (4, "Destination ontology sensitivity", "src.paper.appendix.tableA02_destination_ontology", "data/derived/destination_ontology.parquet", "paper_builder", "outputs/paper/appendix/tableA02_destination_ontology.csv", "appendix"),
    (5, "Leave-one-out ground-quality construction", "src.next_round.repairs.ground_quality_loo", "analysis_voyage_augmented.parquet", "existing_output", "data/derived/ground_quality_loo.parquet", "appendix"),
    (6, "Missingness audit by year/port/captain/agent", "src.paper.tables.table01_sample", "analysis_voyage_augmented.parquet", "paper_builder", "outputs/paper/tables/table01_sample.csv", "main"),
    (7, "Leave-one-agent/ground/era influence analysis", "src.paper.appendix.tableA16_influence", "outputs/datasets/ml/outcome_ml_dataset.parquet", "paper_builder", "outputs/paper/appendix/tableA16_influence.csv", "appendix"),
    (8, "Basin-choice multinomial/nested logit", "src.paper.tables.table03_hierarchical_map", "outputs/datasets/ml/outcome_ml_dataset.parquet; data/derived/destination_ontology.parquet", "paper_builder", "outputs/paper/tables/table03_hierarchical_map.csv", "main"),
    (9, "Theater-choice conditional model", "src.paper.tables.table03_hierarchical_map", "outputs/datasets/ml/outcome_ml_dataset.parquet; data/derived/destination_ontology.parquet", "paper_builder", "outputs/paper/tables/table03_hierarchical_map.csv", "main"),
    (10, "Major-ground conditional model", "src.paper.tables.table03_hierarchical_map", "outputs/datasets/ml/outcome_ml_dataset.parquet; data/derived/destination_ontology.parquet", "paper_builder", "outputs/paper/tables/table03_hierarchical_map.csv", "main"),
    (11, "Conditional deviance / Shapley decomposition", "src.paper.tables.table03_hierarchical_map", "outputs/datasets/ml/outcome_ml_dataset.parquet; data/derived/destination_ontology.parquet", "paper_builder", "outputs/paper/tables/table03_hierarchical_map.csv", "main"),
    (12, "Captain-group holdout", "src.paper.appendix.tableA13_portability", "outputs/datasets/ml/outcome_ml_dataset.parquet; data/derived/destination_ontology.parquet", "paper_builder", "outputs/paper/appendix/tableA13_portability.csv", "appendix"),
    (13, "Agent-group holdout", "src.paper.appendix.tableA13_portability", "outputs/datasets/ml/outcome_ml_dataset.parquet; data/derived/destination_ontology.parquet", "paper_builder", "outputs/paper/appendix/tableA13_portability.csv", "appendix"),
    (14, "Information-refresh timing test", "src.paper.appendix.tableA12_info_vs_routine", "outputs/datasets/ml/action_dataset.parquet", "paper_builder", "outputs/paper/appendix/tableA12_info_vs_routine.csv", "appendix"),
    (15, "Discrete-time exit hazard with psi × negative signal", "src.paper.tables.table04_stopping", "test3_stopping_rule.csv; survival_dataset.parquet", "paper_builder", "outputs/paper/tables/table04_stopping.csv", "main"),
    (16, "Positive-signal placebo", "src.paper.tables.table04_stopping", "test3_stopping_rule.csv; action_dataset.parquet", "paper_builder", "outputs/paper/tables/table04_stopping.csv", "main"),
    (17, "Transit/homebound/productive-state placebos", "src.paper.tables.table04_stopping", "action_dataset.parquet", "paper_builder", "outputs/paper/tables/table04_stopping.csv", "main"),
    (18, "Rational-exit interactions with outside options and season remaining", "src.paper.appendix.tableA10_rational_exit", "outputs/tables/next_round/rational_exit_tests.csv", "paper_builder", "outputs/paper/appendix/tableA10_rational_exit.csv", "appendix"),
    (19, "Captain FE stopping rule", "src.paper.tables.table04_stopping", "outputs/datasets/ml/survival_dataset.parquet; outputs/datasets/ml/outcome_ml_dataset.parquet", "paper_builder", "outputs/paper/tables/table04_stopping.csv", "main"),
    (20, "Agent-clustered inference", "src.paper.tables.table04_stopping", "outputs/datasets/ml/survival_dataset.parquet; outputs/datasets/ml/outcome_ml_dataset.parquet", "paper_builder", "outputs/paper/tables/table04_stopping.csv", "main"),
    (21, "Cox robustness", "src.paper.tables.table04_stopping", "outputs/datasets/ml/survival_dataset.parquet", "paper_builder", "outputs/paper/tables/table04_stopping.csv", "main"),
    (22, "AFT robustness", "src.paper.tables.table04_stopping", "outputs/datasets/ml/survival_dataset.parquet", "paper_builder", "outputs/paper/tables/table04_stopping.csv", "main"),
    (23, "HMM / latent-state estimation", "src.paper.tables.table05_state_switching", "outputs/datasets/ml/state_dataset.parquet", "paper_builder", "outputs/paper/tables/table05_state_switching.csv", "main"),
    (24, "State-transition multinomial/hazard model", "src.paper.tables.table05_state_switching", "outputs/datasets/ml/state_dataset.parquet; outputs/datasets/ml/action_dataset.parquet", "paper_builder", "outputs/paper/tables/table05_state_switching.csv", "main"),
    (25, "Same-captain matched-state pre/post switch FE", "src.paper.tables.table05_state_switching", "outputs/datasets/ml/action_dataset.parquet; outcome_ml_dataset.parquet", "paper_builder", "outputs/paper/tables/table05_state_switching.csv", "main"),
    (26, "Switch event study", "src.paper.tables.table05_state_switching", "outputs/datasets/ml/action_dataset.parquet; outcome_ml_dataset.parquet", "paper_builder", "outputs/paper/tables/table05_state_switching.csv", "main"),
    (27, "Placebo switch-date test", "src.paper.appendix.tableA05_mover_switcher_balance", "outputs/tables/next_round/switch_policy_change.csv; outputs/datasets/ml/outcome_ml_dataset.parquet", "paper_builder", "outputs/paper/appendix/tableA05_mover_switcher_balance.csv", "appendix"),
    (28, "Switchback reversibility", "src.paper.appendix.tableA05_mover_switcher_balance", "outputs/datasets/ml/outcome_ml_dataset.parquet", "paper_builder", "outputs/paper/appendix/tableA05_mover_switcher_balance.csv", "appendix"),
    (29, "Encounter hazard model", "src.paper.tables.table06_search_execution_exitvalue", "outputs/datasets/ml/action_dataset.parquet", "paper_builder", "outputs/paper/tables/table06_search_execution_exitvalue.csv", "main"),
    (30, "Strike conditional on encounter", "src.paper.tables.table06_search_execution_exitvalue", "outputs/datasets/ml/action_dataset.parquet", "paper_builder", "outputs/paper/tables/table06_search_execution_exitvalue.csv", "main"),
    (31, "Capture conditional on strike", "src.paper.tables.table06_search_execution_exitvalue", "outputs/datasets/ml/action_dataset.parquet", "paper_builder", "outputs/paper/tables/table06_search_execution_exitvalue.csv", "main"),
    (32, "Yield conditional on capture", "src.paper.tables.table06_search_execution_exitvalue", "outputs/datasets/ml/action_dataset.parquet; outcome_ml_dataset.parquet", "paper_builder", "outputs/paper/tables/table06_search_execution_exitvalue.csv", "main"),
    (33, "Value-of-exit matched barren-state comparison", "src.paper.tables.table06_search_execution_exitvalue", "outputs/datasets/ml/action_dataset.parquet", "paper_builder", "outputs/paper/tables/table06_search_execution_exitvalue.csv", "main"),
    (34, "IPW exit-value estimator", "src.paper.tables.table06_search_execution_exitvalue", "outputs/datasets/ml/action_dataset.parquet", "paper_builder", "outputs/paper/tables/table06_search_execution_exitvalue.csv", "main"),
    (35, "Doubly robust exit-value estimator", "src.paper.tables.table06_search_execution_exitvalue", "outputs/datasets/ml/action_dataset.parquet", "paper_builder", "outputs/paper/tables/table06_search_execution_exitvalue.csv", "main"),
    (36, "Sequential horse race with vessel controls", "src.paper.tables.table07_hardware_staffing", "outcome_ml_dataset.parquet", "paper_builder", "outputs/paper/tables/table07_hardware_staffing.csv", "main"),
    (37, "Sequential horse race with crew/officer controls", "src.paper.tables.table07_hardware_staffing", "outcome_ml_dataset.parquet", "paper_builder", "outputs/paper/tables/table07_hardware_staffing.csv", "main"),
    (38, "Sequential horse race with disruption controls", "src.paper.tables.table07_hardware_staffing", "outcome_ml_dataset.parquet", "paper_builder", "outputs/paper/tables/table07_hardware_staffing.csv", "main"),
    (39, "Sequential horse race with incentives/lay controls", "src.paper.tables.table07_hardware_staffing", "outcome_ml_dataset.parquet", "paper_builder", "outputs/paper/tables/table07_hardware_staffing.csv", "main"),
    (40, "Same-vessel FE robustness", "src.paper.appendix.tableA07_same_vessel_same_captain", "outputs/datasets/ml/outcome_ml_dataset.parquet", "paper_builder", "outputs/paper/appendix/tableA07_same_vessel_same_captain.csv", "appendix"),
    (41, "Negative controls on transit/homebound speed", "src.paper.tables.table07_hardware_staffing", "outputs/datasets/ml/action_dataset.parquet", "paper_builder", "outputs/paper/tables/table07_hardware_staffing.csv", "main"),
    (42, "Within-agent SD compression", "src.paper.appendix.tableA11_policy_entropy", "output/reinforcement/tables/test4_within_agent.csv", "paper_builder", "outputs/paper/appendix/tableA11_policy_entropy.csv", "appendix"),
    (43, "Conditional action entropy", "src.paper.appendix.tableA11_policy_entropy", "outputs/tables/next_round/policy_entropy.csv", "paper_builder", "outputs/paper/appendix/tableA11_policy_entropy.csv", "appendix"),
    (44, "Cross-captain policy divergence within agent", "src.paper.appendix.tableA11_policy_entropy", "outputs/tables/next_round/policy_entropy.csv; output/reinforcement/tables/test4_residual_variance.csv", "paper_builder", "outputs/paper/appendix/tableA11_policy_entropy.csv", "appendix"),
    (45, "Pre/post switch change in policy dispersion", "src.paper.appendix.tableA11_policy_entropy", "outputs/tables/next_round/switch_policy_change.csv", "paper_builder", "outputs/paper/appendix/tableA11_policy_entropy.csv", "appendix"),
    (46, "Bottom-decile model", "src.paper.tables.table08_floor_raising", "outcome_ml_dataset.parquet", "paper_builder", "outputs/paper/tables/table08_floor_raising.csv", "main"),
    (47, "Bottom-5% model", "src.paper.tables.table08_floor_raising", "outcome_ml_dataset.parquet", "paper_builder", "outputs/paper/tables/table08_floor_raising.csv", "main"),
    (48, "Catastrophic-voyage model", "src.paper.tables.table08_floor_raising", "outcome_ml_dataset.parquet", "paper_builder", "outputs/paper/tables/table08_floor_raising.csv", "main"),
    (49, "Long-dry-spell hazard", "src.paper.tables.table08_floor_raising", "outputs/datasets/ml/action_dataset.parquet", "paper_builder", "outputs/paper/tables/table08_floor_raising.csv", "main"),
    (50, "Expected shortfall model", "src.paper.tables.table08_floor_raising", "outcome_ml_dataset.parquet", "paper_builder", "outputs/paper/tables/table08_floor_raising.csv", "main"),
    (51, "Quantile regressions at P10/P25/P50/P75/P90", "src.paper.tables.table08_floor_raising", "outcome_ml_dataset.parquet", "paper_builder", "outputs/paper/tables/table08_floor_raising.csv", "main"),
    (52, "Within-captain variance pre/post switch", "src.paper.tables.table08_floor_raising", "outcome_ml_dataset.parquet", "paper_builder", "outputs/paper/tables/table08_floor_raising.csv", "main"),
    (53, "Heterogeneity by novice/expert", "src.paper.tables.table08_floor_raising", "outcome_ml_dataset.parquet", "paper_builder", "outputs/paper/tables/table08_floor_raising.csv", "main"),
    (54, "Heterogeneity by theta", "src.paper.tables.table08_floor_raising", "outcome_ml_dataset.parquet", "paper_builder", "outputs/paper/tables/table08_floor_raising.csv", "main"),
    (55, "Heterogeneity by prior volatility", "src.paper.tables.table08_floor_raising", "outcome_ml_dataset.parquet", "paper_builder", "outputs/paper/tables/table08_floor_raising.csv", "main"),
    (56, "Heterogeneity by scarcity", "src.paper.tables.table08_floor_raising", "outcome_ml_dataset.parquet", "paper_builder", "outputs/paper/tables/table08_floor_raising.csv", "main"),
    (57, "Mediation through voyage duration", "src.paper.tables.table09_mediation", "outcome_ml_dataset.parquet; outputs/datasets/ml/action_dataset.parquet", "paper_builder", "outputs/paper/tables/table09_mediation.csv", "main"),
    (58, "Mediation through barren-search time", "src.paper.tables.table09_mediation", "outputs/datasets/ml/action_dataset.parquet", "paper_builder", "outputs/paper/tables/table09_mediation.csv", "main"),
    (59, "Mediation through exploitation time", "src.paper.tables.table09_mediation", "outputs/datasets/ml/action_dataset.parquet", "paper_builder", "outputs/paper/tables/table09_mediation.csv", "main"),
    (60, "Mediation through number of grounds visited", "src.paper.tables.table09_mediation", "outcome_ml_dataset.parquet; outputs/datasets/ml/action_dataset.parquet", "paper_builder", "outputs/paper/tables/table09_mediation.csv", "main"),
    (61, "Mediation through destination diversification", "src.paper.tables.table09_mediation", "outcome_ml_dataset.parquet; outputs/datasets/ml/action_dataset.parquet", "paper_builder", "outputs/paper/tables/table09_mediation.csv", "main"),
    (62, "Mean-output theta × psi interaction", "src.paper.tables.table10_tail_matching", "outcome_ml_dataset.parquet", "paper_builder", "outputs/paper/tables/table10_tail_matching.csv", "main"),
    (63, "Scarcity-interacted theta × psi", "src.paper.tables.table10_tail_matching", "outcome_ml_dataset.parquet", "paper_builder", "outputs/paper/tables/table10_tail_matching.csv", "main"),
    (64, "Tail-risk theta × psi for bottom-decile risk", "src.paper.tables.table10_tail_matching", "outcome_ml_dataset.parquet; outputs/datasets/ml/action_dataset.parquet", "paper_builder", "outputs/paper/tables/table10_tail_matching.csv", "main"),
    (65, "Tail-risk theta × psi for expected shortfall", "src.paper.tables.table10_tail_matching", "outcome_ml_dataset.parquet", "paper_builder", "outputs/paper/tables/table10_tail_matching.csv", "main"),
    (66, "Sorting moments by era and scarcity", "src.paper.tables.table10_tail_matching", "outcome_ml_dataset.parquet", "paper_builder", "outputs/paper/tables/table10_tail_matching.csv", "main"),
    (67, "Constrained PAM", "src.paper.tables.table10_tail_matching", "outputs/datasets/ml/outcome_ml_dataset.parquet; outputs/datasets/ml/action_dataset.parquet", "paper_builder", "outputs/paper/tables/table10_tail_matching.csv", "main"),
    (68, "Constrained AAM/NAM", "src.paper.tables.table10_tail_matching", "outputs/datasets/ml/outcome_ml_dataset.parquet; outputs/datasets/ml/action_dataset.parquet", "paper_builder", "outputs/paper/tables/table10_tail_matching.csv", "main"),
    (69, "Mean-optimal assignment", "src.paper.tables.table10_tail_matching", "outputs/datasets/ml/outcome_ml_dataset.parquet; outputs/datasets/ml/action_dataset.parquet", "paper_builder", "outputs/paper/tables/table10_tail_matching.csv", "main"),
    (70, "CVaR-optimal assignment", "src.paper.tables.table10_tail_matching", "outputs/datasets/ml/outcome_ml_dataset.parquet; outputs/datasets/ml/action_dataset.parquet", "paper_builder", "outputs/paper/tables/table10_tail_matching.csv", "main"),
    (71, "Certainty-equivalent-optimal assignment", "src.paper.tables.table10_tail_matching", "outputs/datasets/ml/outcome_ml_dataset.parquet; outputs/datasets/ml/action_dataset.parquet", "paper_builder", "outputs/paper/tables/table10_tail_matching.csv", "main"),
    (72, "Matching uncertainty bootstrap", "src.paper.appendix.tableA15_matching_robustness", "outputs/datasets/ml/outcome_ml_dataset.parquet; outputs/datasets/ml/action_dataset.parquet", "paper_builder", "outputs/paper/appendix/tableA15_matching_robustness.csv", "appendix"),
    (73, "Support/overlap audit for matching counterfactuals", "src.paper.appendix.tableA15_matching_robustness", "outputs/datasets/ml/outcome_ml_dataset.parquet; outputs/datasets/ml/action_dataset.parquet", "paper_builder", "outputs/paper/appendix/tableA15_matching_robustness.csv", "appendix"),
    (74, "Instructions index -> stopping rules", "src.paper.appendix.tableA18_archival_mechanisms", "data/final/analysis_voyage_augmented.parquet", "paper_builder", "outputs/paper/appendix/tableA18_archival_mechanisms.csv", "appendix"),
    (75, "Officer pipeline -> governance", "src.paper.appendix.tableA18_archival_mechanisms", "data/final/analysis_voyage_augmented.parquet", "paper_builder", "outputs/paper/appendix/tableA18_archival_mechanisms.csv", "appendix"),
    (76, "Lay/incentive intensity -> stopping and duration", "src.paper.appendix.tableA18_archival_mechanisms", "data/final/analysis_voyage_augmented.parquet", "paper_builder", "outputs/paper/appendix/tableA18_archival_mechanisms.csv", "appendix"),
    (77, "Outfit/refit intensity -> stopping and output", "src.paper.appendix.tableA18_archival_mechanisms", "data/final/analysis_voyage_augmented.parquet", "paper_builder", "outputs/paper/appendix/tableA18_archival_mechanisms.csv", "appendix"),
    (78, "Intelligence-sharing exposure -> destination and exit policy", "src.paper.appendix.tableA18_archival_mechanisms", "data/final/analysis_voyage_augmented.parquet", "paper_builder", "outputs/paper/appendix/tableA18_archival_mechanisms.csv", "appendix"),
]


def build_test_registry(context: BuildContext) -> Path:
    registry = context.outputs / "manifests" / "test_registry.csv"
    rows = [
        {
            "test_id": test_id,
            "test_name": test_name,
            "module_name": module_name,
            "input_dataset": input_dataset,
            "status": status,
            "output_files": output_files,
            "feeds": feeds,
        }
        for test_id, test_name, module_name, input_dataset, status, output_files, feeds in TEST_DEFINITIONS
    ]
    with registry.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return registry


def build_table_maps(context: BuildContext) -> tuple[Path, Path]:
    main_path = context.docs / "main_text_table_map.md"
    app_path = context.docs / "appendix_table_map.md"
    main_path.write_text(
        "# Main-text Table Map\n\n" + "\n".join(f"- `{t}`" for t in MAIN_TABLES) + "\n",
        encoding="utf-8",
    )
    app_path.write_text(
        "# Appendix Table Map\n\n" + "\n".join(f"- `{t}`" for t in APPENDIX_TABLES) + "\n",
        encoding="utf-8",
    )
    return main_path, app_path
