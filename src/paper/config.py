from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "outputs" / "paper"
DOCS_ROOT = ROOT / "docs" / "paper"

THETA_PSI_CROSSWALK = {
    "alpha_hat": "theta_hat",
    "gamma_hat": "psi_hat",
}

MAIN_TABLES = [
    "table01_sample",
    "table02_types",
    "table03_hierarchical_map",
    "table04_stopping",
    "table05_state_switching",
    "table06_search_execution_exitvalue",
    "table07_hardware_staffing",
    "table08_floor_raising",
    "table09_mediation",
    "table10_tail_matching",
]

APPENDIX_TABLES = [
    "tableA01_included_excluded",
    "tableA02_destination_ontology",
    "tableA03_connected_set",
    "tableA04_type_robustness",
    "tableA05_mover_switcher_balance",
    "tableA06_ml_ablation_audit",
    "tableA07_same_vessel_same_captain",
    "tableA08_patch_definition",
    "tableA09_scarcity_definition",
    "tableA10_rational_exit",
    "tableA11_policy_entropy",
    "tableA12_info_vs_routine",
    "tableA13_portability",
    "tableA14_lower_tail_robustness",
    "tableA15_matching_robustness",
    "tableA16_influence",
    "tableA17_regime_heterogeneity",
    "tableA18_archival_mechanisms",
]

FIGURES = [
    "fig01_sample_flow",
    "fig02_map_hierarchy",
    "fig03_stopping_margins",
    "fig04_state_transitions",
    "fig05_switch_event_study",
    "fig06_exit_value",
    "fig07_search_vs_execution",
    "fig08_floor_raising",
    "fig09_tail_submodularity",
    "fig10_matching_welfare",
]

DEFAULT_SAMPLE_FLAGS = [
    "in_universe",
    "in_connected_set",
    "has_coordinates",
    "has_ground_labels",
    "has_patch_data",
    "has_encounter_data",
    "has_switch_event",
    "has_archival_data",
]

@dataclass(frozen=True)
class BuildContext:
    root: Path = ROOT
    outputs: Path = OUTPUT_ROOT
    docs: Path = DOCS_ROOT
