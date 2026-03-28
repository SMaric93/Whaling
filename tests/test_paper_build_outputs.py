from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paper.config import BuildContext
from src.paper.sample_lineage import build_master_sample_lineage
from src.paper.tables import (
    table01_sample,
    table02_types,
    table03_hierarchical_map,
    table04_stopping,
    table05_state_switching,
    table06_search_execution_exitvalue,
    table07_hardware_staffing,
    table08_floor_raising,
    table09_mediation,
    table10_tail_matching,
)


def _context(tmp_path: Path) -> BuildContext:
    return BuildContext(
        root=Path(__file__).resolve().parents[1],
        outputs=tmp_path / "paper_outputs",
        docs=tmp_path / "paper_docs",
    )


def test_master_sample_lineage_is_data_driven(tmp_path):
    context = _context(tmp_path)
    (context.outputs / "manifests").mkdir(parents=True, exist_ok=True)
    path = build_master_sample_lineage(context)
    lineage = pd.read_parquet(path)

    assert len(lineage) > 10000
    assert lineage["voyage_id"].notna().all()
    assert "primary_exclusion_reason" in lineage.columns
    assert lineage["in_connected_set"].sum() > 5000


def test_table01_sample_has_real_panels(tmp_path):
    context = _context(tmp_path)
    for subdir in ["tables", "appendix", "figures", "manifests", "memos", "latex"]:
        (context.outputs / subdir).mkdir(parents=True, exist_ok=True)
    result = table01_sample.build(context)
    table = pd.read_csv(result["csv"])

    assert {"Panel A", "Panel B"} <= set(table["panel"].unique())
    assert "Universe of voyages" in set(table["row_label"])
    assert "Log output" in set(table["row_label"])


def test_table02_and_table04_are_not_placeholder_module_maps(tmp_path):
    context = _context(tmp_path)
    for subdir in ["tables", "appendix", "figures", "manifests", "memos", "latex"]:
        (context.outputs / subdir).mkdir(parents=True, exist_ok=True)

    table02 = pd.read_csv(table02_types.build(context)["csv"])
    table04 = pd.read_csv(table04_stopping.build(context)["csv"])

    assert "Captain variance" in set(table02.get("row_label", []))
    assert "theta split-half correlation" in set(table02.get("row_label", []))
    assert "psi_hat × negative signal" in set(table04.get("row_label", []))
    assert "mapped_integration" not in table02.to_string()
    assert "mapped_integration" not in table04.to_string()


def test_remaining_main_tables_are_data_driven(tmp_path):
    context = _context(tmp_path)
    for subdir in ["tables", "appendix", "figures", "manifests", "memos", "latex"]:
        (context.outputs / subdir).mkdir(parents=True, exist_ok=True)

    built_tables = {
        "table03": pd.read_csv(table03_hierarchical_map.build(context)["csv"]),
        "table05": pd.read_csv(table05_state_switching.build(context)["csv"]),
        "table06": pd.read_csv(table06_search_execution_exitvalue.build(context)["csv"]),
        "table07": pd.read_csv(table07_hardware_staffing.build(context)["csv"]),
        "table08": pd.read_csv(table08_floor_raising.build(context)["csv"]),
        "table09": pd.read_csv(table09_mediation.build(context)["csv"]),
        "table10": pd.read_csv(table10_tail_matching.build(context)["csv"]),
    }

    assert {
        "basin choice",
        "theater choice conditional on basin",
        "major-ground choice conditional on theater",
    } <= set(built_tables["table03"].get("level", []))
    assert {"barren_search -> exit/relocation", "post-switch", "switch", "t+2"} <= set(
        built_tables["table05"].get("row_label", [])
    )
    assert "future output (90)" in set(built_tables["table06"].get("row_label", []))
    assert "psi_hat × negative signal" in set(built_tables["table07"].get("row_label", []))
    assert "expected shortfall proxy" in set(built_tables["table08"].get("row_label", []))
    assert "destination diversification" in set(built_tables["table09"].get("row_label", []))
    assert {
        "constrained mean-optimal",
        "constrained CVaR-optimal",
        "constrained certainty-equivalent-optimal",
    } <= set(built_tables["table10"].get("row_label", []))
    matching_rows = built_tables["table10"].set_index("row_label")
    assert pd.notna(matching_rows.loc["constrained mean-optimal", "mean_output"])
    assert pd.notna(matching_rows.loc["constrained certainty-equivalent-optimal", "certainty_equivalent"])
    interaction_rows = built_tables["table10"].set_index("row_label")
    assert pd.notna(interaction_rows.loc["expected shortfall: theta × psi", "coefficient"])
    assert matching_rows.loc["PAM", "pct_reassigned_outside_observed_support"] == 0
    assert matching_rows.loc["AAM/NAM", "pct_reassigned_outside_observed_support"] == 0
    assert matching_rows.loc["constrained mean-optimal", "pct_reassigned_outside_observed_support"] == 0
    assert matching_rows.loc["constrained CVaR-optimal", "pct_reassigned_outside_observed_support"] == 0
    assert matching_rows.loc["constrained certainty-equivalent-optimal", "pct_reassigned_outside_observed_support"] == 0

    for name, table in built_tables.items():
        assert "mapped_integration" not in table.to_string(), name


def test_latex_outputs_do_not_render_literal_nan(tmp_path):
    context = _context(tmp_path)
    for subdir in ["tables", "appendix", "figures", "manifests", "memos", "latex"]:
        (context.outputs / subdir).mkdir(parents=True, exist_ok=True)

    table01_sample.build(context)
    table10_tail_matching.build(context)

    for name in ["table01_sample.tex", "table10_tail_matching.tex"]:
        tex = (context.outputs / "latex" / name).read_text(encoding="utf-8").lower()
        assert " nan " not in tex
        assert "& nan &" not in tex


def test_transit_placebo_row_has_observations(tmp_path):
    context = _context(tmp_path)
    for subdir in ["tables", "appendix", "figures", "manifests", "memos", "latex"]:
        (context.outputs / subdir).mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(table04_stopping.build(context)["csv"]).set_index("row_label")
    assert table.loc["transit", "n_obs"] > 0
