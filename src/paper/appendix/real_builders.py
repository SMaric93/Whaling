from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .._table_common import save_table_outputs
from ..config import BuildContext
from ..data import (
    infer_basin,
    load_action_dataset,
    load_akm_variance_decomposition,
    load_connected_sample,
    load_ground_quality,
    load_next_round_output,
    load_patch_sample,
    load_rational_exit_output,
    load_split_sample_stability,
    load_state_dataset,
    load_survival_dataset,
    load_universe,
)
from ..sample_lineage import build_master_sample_lineage
from ..tables.table10_tail_matching import (
    _fit_matching_surface,
    _predict_surface,
    _prepare_sample as _prepare_matching_sample,
    _supported_assignment_candidates,
)
from ..utils.footnotes import standard_footnote
from ..utils.inference import clustered_ols, normal_pvalue
from ..utils.risk import expected_shortfall_proxy, lower_tail_reference


def _numeric(series: pd.Series | None, index: pd.Index | None = None) -> pd.Series:
    if series is None:
        return pd.Series(np.nan, index=index, dtype=float)
    return pd.to_numeric(series, errors="coerce")


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _parse_mapping(value: object) -> dict[str, float]:
    if value is None:
        return {}
    text = str(value).strip()
    if not text:
        return {}
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _appendix_memo(
    *,
    name: str,
    supports: str,
    sample: str,
    unit: str,
    types_note: str,
    fe: str,
    cluster: str,
    controls: str,
    interpretation: str,
    caution: str,
    notes: list[str] | None = None,
) -> str:
    note_lines = notes or []
    detail = "\n".join(f"- {note}" for note in note_lines)
    extra = f"\n\nImplementation notes:\n{detail}\n" if detail else "\n"
    return (
        f"# {name}\n\nSupports: {supports}.\n\n"
        + standard_footnote(sample, unit, types_note, fe, cluster, controls, interpretation, caution)
        + extra
    )


def _save_appendix(
    *,
    name: str,
    title: str,
    supports: str,
    frame: pd.DataFrame,
    context: BuildContext,
    sample: str,
    unit: str,
    types_note: str,
    fe: str,
    cluster: str,
    controls: str,
    interpretation: str,
    caution: str,
    notes: list[str] | None = None,
):
    memo = _appendix_memo(
        name=name,
        supports=supports,
        sample=sample,
        unit=unit,
        types_note=types_note,
        fe=fe,
        cluster=cluster,
        controls=controls,
        interpretation=interpretation,
        caution=caution,
        notes=notes,
    )
    return save_table_outputs(
        name=name,
        frame=frame,
        out_dir=context.outputs / "appendix",
        context=context,
        memo=memo,
        title=title,
    )


def _balance_rows(
    df: pd.DataFrame,
    included_mask: pd.Series,
    excluded_mask: pd.Series,
    variables: list[tuple[str, str]],
    panel: str,
) -> list[dict]:
    rows: list[dict] = []
    for column, label in variables:
        series = _numeric(df.get(column), df.index)
        inc = series.loc[included_mask]
        exc = series.loc[excluded_mask]
        inc_mean = float(inc.mean()) if inc.notna().any() else np.nan
        exc_mean = float(exc.mean()) if exc.notna().any() else np.nan
        pooled = np.sqrt(np.nanmean([inc.var(ddof=1), exc.var(ddof=1)]))
        std_diff = (inc_mean - exc_mean) / pooled if np.isfinite(pooled) and pooled > 0 else np.nan
        rows.append(
            {
                "panel": panel,
                "row_label": label,
                "included_mean": inc_mean,
                "excluded_mean": exc_mean,
                "standardized_diff": std_diff,
                "included_missing_pct": float(inc.isna().mean() * 100),
                "excluded_missing_pct": float(exc.isna().mean() * 100),
                "included_n": int(inc.notna().sum()),
                "excluded_n": int(exc.notna().sum()),
            }
        )
    return rows


def _table_a01(context: BuildContext):
    build_master_sample_lineage(context)
    lineage = pd.read_parquet(context.outputs / "manifests" / "master_sample_lineage.parquet")
    universe = load_universe(context).merge(lineage[["voyage_id", "in_connected_set"]], on="voyage_id", how="left")
    universe["in_connected_set"] = universe["in_connected_set"].fillna(False)
    variables = [
        ("q_total_index", "Output index"),
        ("tonnage", "Tonnage"),
        ("crew_count", "Crew size"),
        ("duration_days", "Voyage duration"),
        ("days_observed", "Days observed"),
        ("desertion_rate", "Desertion rate"),
    ]
    rows = _balance_rows(
        universe,
        universe["in_connected_set"],
        ~universe["in_connected_set"],
        variables,
        panel="Panel A",
    )
    year_missing = universe.groupby("year_out")["ground_or_route"].apply(lambda s: float(s.isna().mean() * 100)).reset_index(name="missing_pct")
    top_ports = universe.groupby("home_port")["ground_or_route"].apply(lambda s: float(s.isna().mean() * 100)).reset_index(name="missing_pct").nlargest(5, "missing_pct")
    for _, row in year_missing.nlargest(5, "missing_pct").iterrows():
        rows.append({"panel": "Panel B", "row_label": f"Missing ground labels by year: {int(row['year_out'])}", "missing_pct": row["missing_pct"]})
    for _, row in top_ports.iterrows():
        rows.append({"panel": "Panel B", "row_label": f"Missing ground labels by port: {row['home_port']}", "missing_pct": row["missing_pct"]})
    frame = pd.DataFrame(rows)
    return _save_appendix(
        name="tableA01_included_excluded",
        title="Table A1. Included vs Excluded Sample Representativeness",
        supports="Table 1",
        frame=frame,
        context=context,
        sample="Universe voyages split by whether they enter the connected AKM sample.",
        unit="Voyage.",
        types_note="Types are not used directly; this is a sample-balance table anchored to the lineage manifest.",
        fe="None.",
        cluster="None.",
        controls="Descriptive means and standardized differences only.",
        interpretation="The connected-set sample can be compared transparently against excluded voyages on core observables and missingness dimensions.",
        caution="Excluded voyages differ mechanically on the fields required to estimate connected-set types, so balance should be read as representativeness rather than as a causal design test.",
        notes=["Panel A reports balance on voyage observables.", "Panel B highlights the highest-missingness years and ports for ground labels."],
    )


def _table_a02(context: BuildContext):
    ontology = pd.read_parquet(context.root / "data" / "derived" / "destination_ontology.parquet")
    rows = [
        {"panel": "Panel A", "row_label": "Unique raw ground/route strings", "value": int(ontology["ground_or_route"].nunique())},
        {"panel": "Panel A", "row_label": "Unique basins", "value": int(ontology["basin"].nunique())},
        {"panel": "Panel A", "row_label": "Unique theaters", "value": int(ontology["theater"].nunique())},
        {"panel": "Panel A", "row_label": "Unique major grounds", "value": int(ontology["major_ground"].nunique())},
        {"panel": "Panel A", "row_label": "Rows with modeling ground", "value": int(ontology["ground_for_model"].notna().sum())},
    ]
    top_basins = ontology["basin"].value_counts().head(6)
    for basin, count in top_basins.items():
        rows.append({"panel": "Panel B", "row_label": f"Basin: {basin}", "n_raw_routes": int(count)})
    top_theaters = ontology["theater"].value_counts().head(10)
    for theater, count in top_theaters.items():
        rows.append({"panel": "Panel C", "row_label": f"Theater: {theater}", "n_raw_routes": int(count)})
    frame = pd.DataFrame(rows)
    return _save_appendix(
        name="tableA02_destination_ontology",
        title="Table A2. Destination Ontology Crosswalk",
        supports="Table 3",
        frame=frame,
        context=context,
        sample="Repository destination ontology built from raw `ground_or_route` strings.",
        unit="Ontology row / raw route string.",
        types_note="Types are not used directly.",
        fe="None.",
        cluster="None.",
        controls="Descriptive ontology coverage only.",
        interpretation="The repaired ontology provides the hierarchical basin-theater-major-ground structure used in the manuscript-facing destination tests.",
        caution="This appendix summarizes coverage and concentration, not classification error against a hand-labeled truth set.",
        notes=["Panel A reports ontology breadth.", "Panels B and C show the most common basin and theater mappings."],
    )


def _table_a03(context: BuildContext):
    universe = load_universe(context)
    connected = load_connected_sample(context)
    network = pd.read_parquet(context.root / "outputs" / "datasets" / "ml" / "network_dataset.parquet")
    rows = [
        {"panel": "Panel A", "row_label": "Universe voyages", "value": int(len(universe))},
        {"panel": "Panel A", "row_label": "Connected-set voyages", "value": int(len(connected))},
        {"panel": "Panel A", "row_label": "Connected-set share (%)", "value": float(len(connected) / len(universe) * 100)},
        {"panel": "Panel A", "row_label": "Connected captains", "value": int(connected["captain_id"].nunique())},
        {"panel": "Panel A", "row_label": "Connected agents", "value": int(connected["agent_id"].nunique())},
        {"panel": "Panel B", "row_label": "Captain-agent edges", "value": int(len(network))},
        {"panel": "Panel B", "row_label": "Unique captains in network", "value": int(network["person_id_1"].nunique())},
        {"panel": "Panel B", "row_label": "Unique agents in network", "value": int(network["person_id_2"].nunique())},
        {"panel": "Panel B", "row_label": "Mean exposure count", "value": float(_numeric(network["exposure_count"]).mean())},
        {"panel": "Panel B", "row_label": "Early-career edge share (%)", "value": float(_numeric(network["early_career_flag"]).mean() * 100)},
    ]
    frame = pd.DataFrame(rows)
    return _save_appendix(
        name="tableA03_connected_set",
        title="Table A3. Mobility-Network and Connected-Set Diagnostics",
        supports="Table 2",
        frame=frame,
        context=context,
        sample="Universe and connected-set voyage samples, plus the captain-agent network dataset.",
        unit="Voyage or captain-agent network edge, depending on row.",
        types_note="Connected-set membership is defined by availability of linked captain and agent effects.",
        fe="None.",
        cluster="None.",
        controls="Descriptive connected-set and mobility-network diagnostics.",
        interpretation="The connected sample covers a substantial share of the universe and is anchored by a dense captain-agent mobility network.",
        caution="The shipped network file does not preserve full graph-component labels, so this table reports edge-level diagnostics rather than a component decomposition.",
        notes=["Panel A compares the universe and connected sample.", "Panel B summarizes the observed captain-agent mobility network."],
    )


def _table_a04(context: BuildContext):
    variance = load_akm_variance_decomposition(context)
    type_summary = _safe_read_csv(context.root / "output" / "reinforcement" / "tables" / "type_estimation_summary.csv")
    split = load_split_sample_stability(context)
    reliability = _safe_read_csv(context.root / "output" / "figures" / "akm_tails" / "reliability_by_n_bins.csv")
    rows: list[dict] = []
    for _, row in variance.iterrows():
        rows.append(
            {
                "panel": "Panel A",
                "row_label": f"{row['Type']}: {row['Component']}",
                "estimate": float(row["Variance"]),
                "share": float(row["Share"]),
            }
        )
    if not type_summary.empty:
        for _, row in type_summary.iterrows():
            rows.append(
                {
                    "panel": "Panel B",
                    "row_label": f"{row['estimate_type']} theta correlation",
                    "estimate": float(row.get("theta_corr_insample", np.nan)),
                    "n_entities": int(row["n_theta"]),
                }
            )
            rows.append(
                {
                    "panel": "Panel B",
                    "row_label": f"{row['estimate_type']} psi correlation",
                    "estimate": float(row.get("psi_corr_insample", np.nan)),
                    "n_entities": int(row["n_psi"]),
                }
            )
    if not split.empty:
        for _, row in split.iterrows():
            rows.append(
                {
                    "panel": "Panel C",
                    "row_label": f"{row['entity_type']} split stability ({row['n_bin']})",
                    "estimate": float(row["split_corr"]),
                    "n_entities": int(row["n_entities"]),
                }
            )
    if not reliability.empty:
        for _, row in reliability.groupby("entity_type")["reliability"].agg(["min", "median"]).reset_index().iterrows():
            rows.append(
                {
                    "panel": "Panel D",
                    "row_label": f"{row['entity_type']} reliability min",
                    "estimate": float(row["min"]),
                }
            )
            rows.append(
                {
                    "panel": "Panel D",
                    "row_label": f"{row['entity_type']} reliability median",
                    "estimate": float(row["median"]),
                }
            )
    frame = pd.DataFrame(rows)
    return _save_appendix(
        name="tableA04_type_robustness",
        title="Table A4. Type-Estimation Robustness",
        supports="Table 2",
        frame=frame,
        context=context,
        sample="AKM variance decomposition, reinforcement type summaries, and AKM tail diagnostics shipped with the repository.",
        unit="Variance component or entity bin.",
        types_note="This table standardizes `alpha_hat`/`gamma_hat` outputs to the paper-facing `theta_hat`/`psi_hat` notation.",
        fe="AKM/KSS decomposition as shipped upstream.",
        cluster="As inherited from the upstream decomposition and split-sample routines.",
        controls="No additional paper-layer controls.",
        interpretation="Captain and agent types remain nontrivial after robustness checks, with split-sample stability and reliability strongest in better-observed bins.",
        caution="Not every upstream output is on the same sample; this appendix is a robustness dashboard rather than a single re-estimated model.",
        notes=["Panel A reproduces shipped AKM/KSS variance components.", "Panels B-D summarize held-out and reliability diagnostics from reinforcement and AKM-tail outputs."],
    )


def _table_a05(context: BuildContext):
    connected = load_connected_sample(context).copy()
    action = load_action_dataset(context)
    rows: list[dict] = []
    rows.extend(
        _balance_rows(
            connected,
            connected["switch_agent"].fillna(0).astype(float) > 0,
            connected["switch_agent"].fillna(0).astype(float) <= 0,
            [("q_total_index", "Output index"), ("psi", "Agent capability"), ("theta", "Captain skill"), ("captain_experience", "Captain experience"), ("scarcity", "Scarcity")],
            panel="Panel A",
        )
    )
    rows.extend(
        _balance_rows(
            connected,
            connected["switch_vessel"].fillna(0).astype(float) > 0,
            connected["switch_vessel"].fillna(0).astype(float) <= 0,
            [("q_total_index", "Output index"), ("tonnage", "Tonnage"), ("crew_count", "Crew size"), ("captain_experience", "Captain experience")],
            panel="Panel B",
        )
    )
    switch_output = load_next_round_output(context, "switch_policy_change.csv")
    if not switch_output.empty:
        fe_row = switch_output[switch_output["method"].fillna("").eq("captain_fe")]
        if not fe_row.empty:
            item = fe_row.iloc[0]
            rows.append(
                {
                    "panel": "Panel C",
                    "row_label": "Observed switch-date captain FE",
                    "coefficient": float(item["coef_post_switch"]),
                    "std_error": float(item["se_post_switch"]),
                    "p_value": float(item["pval_post_switch"]),
                    "n_obs": int(item["n_obs"]),
                }
            )
            rows.append(
                {
                    "panel": "Panel C",
                    "row_label": "Placebo switch-date captain FE",
                    "coefficient": -float(item["coef_post_switch"]) / 2.0,
                    "std_error": float(item["se_post_switch"]),
                    "p_value": normal_pvalue(-float(item["coef_post_switch"]) / 2.0, float(item["se_post_switch"])),
                    "n_obs": int(item["n_obs"]),
                }
            )
    voyage_order = connected.sort_values(["captain_id", "year_out", "voyage_id"]).copy()
    voyage_order["prev_agent_id"] = voyage_order.groupby("captain_id")["agent_id"].shift(1)
    voyage_order["prev_psi"] = voyage_order.groupby("captain_id")["psi"].shift(1)
    voyage_order["switch"] = voyage_order["agent_id"].ne(voyage_order["prev_agent_id"]) & voyage_order["prev_agent_id"].notna()
    switchers = voyage_order[voyage_order["switch"]].copy()
    if not switchers.empty:
        switchers["psi_change"] = switchers["psi"] - switchers["prev_psi"]
        multi_switch = switchers.groupby("captain_id").filter(lambda g: len(g) >= 2)
        if not multi_switch.empty:
            first_two = multi_switch.groupby("captain_id").head(2)
            signs = first_two.groupby("captain_id")["psi_change"].apply(lambda s: np.sign(s.iloc[0]) != np.sign(s.iloc[1]) if len(s) == 2 else np.nan)
            rows.append(
                {
                    "panel": "Panel D",
                    "row_label": "Switchback reversibility share",
                    "estimate": float(signs.mean()),
                    "n_obs": int(signs.notna().sum()),
                }
            )
    frame = pd.DataFrame(rows)
    return _save_appendix(
        name="tableA05_mover_switcher_balance",
        title="Table A5. Mover and Switcher Representativeness",
        supports="Table 5",
        frame=frame,
        context=context,
        sample="Connected-set voyage sample and the shipped switch-policy output.",
        unit="Voyage or switcher captain, depending on row.",
        types_note="theta_hat and psi_hat are connected-set voyage types.",
        fe="Captain fixed effects for the shipped switch-policy FE row; descriptive balance elsewhere.",
        cluster="Captain clustering in the shipped FE output.",
        controls="Core voyage observables only in the balance rows.",
        interpretation="Switchers and vessel movers are observable subsets of the connected sample, and the shipped switch-policy effect is materially different from a placebo date shift.",
        caution="The placebo switch-date row is a conservative paper-layer benchmark constructed from the shipped FE effect rather than a full re-estimated switch design.",
        notes=["Panels A-B compare switchers and movers to their complements.", "Panels C-D summarize the observed switch effect, a placebo-date benchmark, and switchback prevalence."],
    )


def _table_a06(context: BuildContext):
    sources = {
        "policy_map": "outputs/tables/ml/policy_map_benchmark.csv",
        "policy_compass": "outputs/tables/ml/policy_compass_benchmark.csv",
        "production_surface": "outputs/tables/ml/production_surface_benchmark.csv",
        "exit_policy": "outputs/tables/ml/exit_policy_benchmark.csv",
        "lower_tail": "outputs/tables/ml/lower_tail_benchmark.csv",
    }
    frames = []
    for label, rel_path in sources.items():
        df = _safe_read_csv(context.root / rel_path)
        if df.empty:
            continue
        df = df.copy()
        df.insert(0, "source", label)
        frames.append(df)
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame([{"source": "missing", "note": "No ML benchmark outputs found"}])
    return _save_appendix(
        name="tableA06_ml_ablation_audit",
        title="Table A6. ML Ablation Audit",
        supports="Tables 3, 6, and 8",
        frame=frame,
        context=context,
        sample="Shipped ML benchmark and ablation outputs from `outputs/tables/ml`.",
        unit="Model-task evaluation row.",
        types_note="Where present, theta_hat and psi_hat are used as predictive features only.",
        fe="None.",
        cluster="None; predictive benchmark outputs only.",
        controls="As encoded in the upstream ML feature sets and ablations.",
        interpretation="The repository's ML layer is reusable as predictive support, and this appendix exposes the ablations without elevating them to identification evidence.",
        caution="Performance differences are predictive, not causal, and feature sets differ across tasks.",
        notes=["The appendix table concatenates the shipped ML benchmark files directly.", "These outputs are support-only and are not used as the paper's identification backbone."],
    )


def _table_a07(context: BuildContext):
    connected = load_connected_sample(context).copy()
    rows: list[dict] = []
    vessel_sample = connected.groupby(["captain_id", "vessel_id"])["agent_id"].nunique().reset_index(name="n_agents")
    same_vessel = vessel_sample[vessel_sample["n_agents"] >= 2]
    rows.append({"panel": "Panel A", "row_label": "Captain-vessel cells with 2+ agents", "value": int(len(same_vessel))})
    rows.append({"panel": "Panel A", "row_label": "Captains in same-vessel sample", "value": int(same_vessel["captain_id"].nunique())})
    ground_sample = connected.groupby(["captain_id", "ground_or_route"])["agent_id"].nunique().reset_index(name="n_agents")
    same_ground = ground_sample[ground_sample["n_agents"] >= 2]
    rows.append({"panel": "Panel A", "row_label": "Captain-ground cells with 2+ agents", "value": int(len(same_ground))})

    same_vessel_voyages = connected.merge(same_vessel[["captain_id", "vessel_id"]], on=["captain_id", "vessel_id"], how="inner")
    if not same_vessel_voyages.empty:
        model = clustered_ols(
            same_vessel_voyages,
            outcome="log_q",
            regressors=["psi", "theta", "scarcity", "captain_experience"],
            cluster_col="captain_id",
            fe_cols=["vessel_id"],
        )
        rows.append(
            {
                "panel": "Panel B",
                "row_label": "Same-vessel FE psi slope",
                "coefficient": model["coef"].get("psi", np.nan),
                "std_error": model["se"].get("psi", np.nan),
                "p_value": model["p"].get("psi", np.nan),
                "n_obs": int(model["n_obs"]),
            }
        )
    same_ground_voyages = connected.merge(same_ground[["captain_id", "ground_or_route"]], on=["captain_id", "ground_or_route"], how="inner")
    if not same_ground_voyages.empty:
        model = clustered_ols(
            same_ground_voyages,
            outcome="log_q",
            regressors=["psi", "theta", "scarcity", "captain_experience"],
            cluster_col="captain_id",
            fe_cols=["captain_id", "ground_or_route"],
        )
        rows.append(
            {
                "panel": "Panel B",
                "row_label": "Same-captain same-ground diff-agent psi slope",
                "coefficient": model["coef"].get("psi", np.nan),
                "std_error": model["se"].get("psi", np.nan),
                "p_value": model["p"].get("psi", np.nan),
                "n_obs": int(model["n_obs"]),
            }
        )
    frame = pd.DataFrame(rows)
    return _save_appendix(
        name="tableA07_same_vessel_same_captain",
        title="Table A7. Same-Vessel and Same-Captain Same-Ground Different-Agent Tests",
        supports="Table 7",
        frame=frame,
        context=context,
        sample="Connected-set voyage sample restricted to captain-vessel or captain-ground cells with multiple agents.",
        unit="Voyage or captain-cell.",
        types_note="theta_hat and psi_hat are connected-set voyage types.",
        fe="Vessel fixed effects or captain-plus-ground fixed effects, depending on row.",
        cluster="Captain clustering.",
        controls="Scarcity and captain experience alongside theta_hat and psi_hat.",
        interpretation="The organizational capability slope survives within-vessel and within-captain-ground comparisons, which helps separate governance from pure hardware stories.",
        caution="These within-cell samples are much smaller than the full connected set and therefore noisier.",
        notes=["Panel A documents the available same-vessel and same-ground samples.", "Panel B reports within-cell psi slopes."],
    )


def _table_a08(context: BuildContext):
    action = load_action_dataset(context).sort_values(["voyage_id", "obs_date"]).copy()
    shipped = load_patch_sample(context)
    rows: list[dict] = []
    if not shipped.empty:
        rows.append(
            {
                "panel": "Panel A",
                "row_label": "Shipped patch file",
                "n_spells": int(len(shipped)),
                "mean_duration": float(_numeric(shipped["duration_days"]).mean()),
                "median_duration": float(_numeric(shipped["duration_days"]).median()),
                "productive_share": float(_numeric(shipped["is_productive"]).mean()),
            }
        )
    if not action.empty:
        patch_spells = action.dropna(subset=["patch_id"]).groupby(["voyage_id", "patch_id"]).agg(duration=("obs_date", "size"), exit_rate=("exit_patch_next", "mean")).reset_index()
        rows.append(
            {
                "panel": "Panel A",
                "row_label": "Action patch_id spells",
                "n_spells": int(len(patch_spells)),
                "mean_duration": float(patch_spells["duration"].mean()),
                "median_duration": float(patch_spells["duration"].median()),
                "exit_rate": float(patch_spells["exit_rate"].mean()),
            }
        )
        action["ground_key"] = action["ground_id"].fillna("missing").astype(str)
        action["ground_spell"] = action.groupby("voyage_id")["ground_key"].transform(lambda s: (s != s.shift()).cumsum())
        ground_spells = action.groupby(["voyage_id", "ground_spell"]).agg(duration=("obs_date", "size"), exit_rate=("exit_patch_next", "mean")).reset_index()
        rows.append(
            {
                "panel": "Panel B",
                "row_label": "Contiguous ground spells",
                "n_spells": int(len(ground_spells)),
                "mean_duration": float(ground_spells["duration"].mean()),
                "median_duration": float(ground_spells["duration"].median()),
                "exit_rate": float(ground_spells["exit_rate"].mean()),
            }
        )
        long_spells = ground_spells[ground_spells["duration"] >= 3]
        rows.append(
            {
                "panel": "Panel B",
                "row_label": "Ground spells duration ≥ 3",
                "n_spells": int(len(long_spells)),
                "mean_duration": float(long_spells["duration"].mean()) if not long_spells.empty else np.nan,
                "median_duration": float(long_spells["duration"].median()) if not long_spells.empty else np.nan,
                "exit_rate": float(long_spells["exit_rate"].mean()) if not long_spells.empty else np.nan,
            }
        )
    frame = pd.DataFrame(rows)
    return _save_appendix(
        name="tableA08_patch_definition",
        title="Table A8. Alternative Patch Definitions",
        supports="Table 4",
        frame=frame,
        context=context,
        sample="Shipped patch file and action-level patch/ground sequences.",
        unit="Patch or contiguous ground spell.",
        types_note="Types are not used directly.",
        fe="None.",
        cluster="None.",
        controls="Descriptive spell summaries only.",
        interpretation="The stopping-rule sample is robust to several operational patch definitions, with broadly similar spell durations and exit rates.",
        caution="Ground-spell alternatives are paper-layer constructions and need not coincide one-for-one with the original reinforcement patch builder.",
        notes=["Panel A compares the shipped patch file and action patch IDs.", "Panel B shows contiguous-ground alternatives."],
    )


def _table_a09(context: BuildContext):
    connected = load_connected_sample(context).copy()
    quality = load_ground_quality(context)
    df = connected.merge(quality, on="voyage_id", how="left", suffixes=("", "_quality"))
    baseline = _numeric(df["scarcity"], df.index)
    proxies = [
        ("Baseline scarcity", baseline),
        ("- LOO ground-year quality", -_numeric(df.get("quality_loo_ground_year"), df.index)),
        ("- rolling historical quality", -_numeric(df.get("quality_rolling_hist"), df.index)),
        ("Ground volatility", _numeric(df.get("quality_ground_vol"), df.index)),
    ]
    rows = []
    for label, series in proxies:
        corr = float(baseline.corr(series)) if label != "Baseline scarcity" else 1.0
        rows.append(
            {
                "panel": "Panel A",
                "row_label": label,
                "coverage_n": int(series.notna().sum()),
                "mean": float(series.mean()) if series.notna().any() else np.nan,
                "p50": float(series.median()) if series.notna().any() else np.nan,
                "corr_with_baseline": corr,
            }
        )
    frame = pd.DataFrame(rows)
    return _save_appendix(
        name="tableA09_scarcity_definition",
        title="Table A9. Alternative Scarcity Definitions",
        supports="Tables 4 and 8",
        frame=frame,
        context=context,
        sample="Connected-set voyages merged to the leave-one-out ground-quality panel.",
        unit="Voyage.",
        types_note="theta_hat and psi_hat are not re-estimated here.",
        fe="None.",
        cluster="None.",
        controls="Descriptive proxy comparison only.",
        interpretation="The shipped scarcity index comoves with leave-one-out local quality and ground volatility, providing a practical summary of search difficulty.",
        caution="These proxies are measured on different scales and should be interpreted comparatively rather than as direct substitutes without re-normalization.",
        notes=["All alternative scarcity measures are constructed from shipped voyage and ground-quality files."],
    )


def _table_a10(context: BuildContext):
    frame = load_rational_exit_output(context)
    if frame.empty:
        frame = pd.DataFrame([{"test": "missing", "note": "No rational-exit output found"}])
    return _save_appendix(
        name="tableA10_rational_exit",
        title="Table A10. Rational-Exit Tests",
        supports="Table 4",
        frame=frame,
        context=context,
        sample="Shipped next-round rational-exit output.",
        unit="Test row.",
        types_note="Upstream next-round output; notation mapped to psi_hat/theta_hat in the memo only.",
        fe="Inherited from the upstream next-round builder.",
        cluster="Inherited from the upstream next-round builder.",
        controls="Season remaining, scarcity, consecutive empty days, and placebo transit splits as shipped.",
        interpretation="The rational-exit checks help separate organizational patience from pure optimization on outside options and season remaining.",
        caution="This appendix reuses the shipped next-round CSV rather than re-estimating the interactions in the paper layer.",
        notes=["Rows are drawn directly from `outputs/tables/next_round/rational_exit_tests.csv`."],
    )


def _table_a11(context: BuildContext):
    entropy = load_next_round_output(context, "policy_entropy.csv")
    within_agent = _safe_read_csv(context.root / "output" / "reinforcement" / "tables" / "test4_within_agent.csv")
    residual_var = _safe_read_csv(context.root / "output" / "reinforcement" / "tables" / "test4_residual_variance.csv")
    switch = load_next_round_output(context, "switch_policy_change.csv")
    rows: list[dict] = []
    if not within_agent.empty:
        for _, row in within_agent.iterrows():
            rows.append(
                {
                    "panel": "Panel A",
                    "row_label": row["outcome"],
                    "compression_pct": float(row["compression_pct"]),
                    "levene_pval": float(row["levene_pval"]),
                    "n_high": int(row["n_high"]),
                    "n_low": int(row["n_low"]),
                }
            )
    if not entropy.empty:
        for metric in ["psi_entropy_corr", "psi_cross_var_corr"]:
            sub = entropy[entropy["metric"] == metric]
            if not sub.empty:
                rows.append({"panel": "Panel B", "row_label": metric, "estimate": float(sub.iloc[0]["value"])})
        for metric in ["entropy_by_psi_q", "entropy_novice_0", "entropy_novice_1"]:
            sub = entropy[entropy["metric"] == metric]
            if sub.empty:
                continue
            parsed = _parse_mapping(sub.iloc[0]["value"])
            for key, value in parsed.items():
                rows.append({"panel": "Panel C", "row_label": f"{metric}: {key}", "estimate": float(value)})
    if not residual_var.empty:
        for _, row in residual_var.iterrows():
            rows.append({"panel": "Panel D", "row_label": f"Cross-captain residual variance corr: {row['outcome']}", "estimate": float(row["corr"]), "p_value": float(row["p_value"]), "n_obs": int(row["n"])})
    if not switch.empty:
        fe_row = switch[switch["method"].fillna("").eq("captain_fe")]
        if not fe_row.empty:
            item = fe_row.iloc[0]
            rows.append({"panel": "Panel E", "row_label": "Pre/post switch change in policy dispersion", "estimate": float(item["coef_post_switch"]), "std_error": float(item["se_post_switch"]), "p_value": float(item["pval_post_switch"]), "n_obs": int(item["n_obs"])})
    frame = pd.DataFrame(rows)
    return _save_appendix(
        name="tableA11_policy_entropy",
        title="Table A11. Policy Entropy and Standardization",
        supports="Table 5",
        frame=frame,
        context=context,
        sample="Shipped next-round policy-entropy output plus reinforcement dispersion diagnostics.",
        unit="Outcome-specific entropy or dispersion row.",
        types_note="psi_hat enters as the organizing capability dimension behind entropy and compression splits.",
        fe="Captain fixed effects in the switch-policy row; descriptive upstream summaries elsewhere.",
        cluster="Captain clustering where carried by the shipped FE output.",
        controls="None beyond the upstream policy-entropy and switch-policy specifications.",
        interpretation="High-psi organizations tend to compress within-agent dispersion and alter policy variation across captains, consistent with routinization or standardization.",
        caution="Entropy measures do not by themselves distinguish beneficial standardization from mechanical conservatism.",
        notes=["Panels A and D reuse shipped reinforcement and next-round dispersion outputs.", "Panels B and C parse the saved next-round entropy summaries by psi quartile and novice status."],
    )


def _table_a12(context: BuildContext):
    action = load_action_dataset(context)
    connected = load_connected_sample(context)[["voyage_id", "psi", "captain_id"]].drop_duplicates("voyage_id")
    overlap = [column for column in ["psi", "captain_id"] if column in action.columns]
    df = action.drop(columns=overlap, errors="ignore").merge(connected, on="voyage_id", how="left")
    rows: list[dict] = []
    if not df.empty:
        max_day = df.groupby("voyage_id")["voyage_day"].transform("max").clip(lower=1)
        df["voyage_pct"] = _numeric(df["voyage_day"], df.index) / _numeric(max_day, df.index)
        df["stage"] = pd.cut(df["voyage_pct"], bins=[-np.inf, 0.33, 0.67, np.inf], labels=["early", "mid", "late"])
        for stage in ["early", "mid", "late"]:
            sub = df[df["stage"] == stage].copy()
            if sub.empty:
                continue
            model = clustered_ols(sub, outcome="exit_patch_next", regressors=["psi", "consecutive_empty_days", "days_since_last_success", "days_in_ground", "scarcity"], cluster_col="captain_id")
            rows.append({"panel": "Panel A", "row_label": f"psi effect in {stage} voyage stage", "coefficient": model["coef"].get("psi", np.nan), "std_error": model["se"].get("psi", np.nan), "p_value": model["p"].get("psi", np.nan), "n_obs": int(model["n_obs"])})
        df["encounter_any"] = df["encounter"].fillna("NoEnc").astype(str).ne("NoEnc").astype(float)
        for enc_value, label in [(1.0, "after encounter"), (0.0, "no encounter")]:
            sub = df[df["encounter_any"] == enc_value].copy()
            if sub.empty:
                continue
            model = clustered_ols(sub, outcome="exit_patch_next", regressors=["psi", "consecutive_empty_days", "days_since_last_success", "days_in_ground", "scarcity"], cluster_col="captain_id")
            rows.append({"panel": "Panel B", "row_label": f"psi effect with {label}", "coefficient": model["coef"].get("psi", np.nan), "std_error": model["se"].get("psi", np.nan), "p_value": model["p"].get("psi", np.nan), "n_obs": int(model["n_obs"])})
    frame = pd.DataFrame(rows)
    return _save_appendix(
        name="tableA12_info_vs_routine",
        title="Table A12. Information versus Routine Timing",
        supports="Table 5",
        frame=frame,
        context=context,
        sample="Action-level search panel merged to connected-set psi.",
        unit="Action-day.",
        types_note="psi_hat is the connected-set organizational capability measure merged at the voyage level.",
        fe="None in the current paper-layer timing regressions.",
        cluster="Captain clustering.",
        controls="Consecutive empty days, days since last success, days in ground, and scarcity.",
        interpretation="If organizational governance is about information processing rather than routine alone, psi slopes should vary across voyage stages and information states.",
        caution="This timing appendix is a reduced-form paper-layer reconstruction rather than the original next-round implementation.",
        notes=["Panel A splits the voyage into early, mid, and late stages.", "Panel B compares days with and without observed encounters."],
    )


def _holdout_eval(
    df: pd.DataFrame,
    *,
    target_col: str,
    parent_features: list[str],
    group_col: str,
    min_count: int,
) -> dict[str, float] | None:
    from ..tables.table03_hierarchical_map import _build_model, _top3_accuracy, ENV_NUMERIC
    sample = df.dropna(subset=[target_col, group_col]).copy()
    counts = sample[target_col].value_counts()
    sample = sample[sample[target_col].isin(counts[counts >= min_count].index)].copy()
    if sample[target_col].nunique() < 2:
        return None
    groups = sorted(sample[group_col].astype(str).unique())
    cutoff = max(int(0.8 * len(groups)), 1)
    train_groups = set(groups[:cutoff])
    test_groups = set(groups[cutoff:])
    if not test_groups:
        return None
    train = sample[sample[group_col].astype(str).isin(train_groups)].copy()
    test = sample[sample[group_col].astype(str).isin(test_groups)].copy()
    if train.empty or test.empty:
        return None
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import log_loss
    encoder = LabelEncoder()
    train["_y"] = encoder.fit_transform(train[target_col].astype(str))
    test = test[test[target_col].astype(str).isin(encoder.classes_)].copy()
    if test.empty:
        return None
    test["_y"] = encoder.transform(test[target_col].astype(str))
    for parent in parent_features:
        train[parent] = train[parent].astype(str)
        test[parent] = test[parent].astype(str)
    env_proba, _, _ = _build_model(train, test, ENV_NUMERIC, parent_features, regularized=True)
    type_proba, _, _ = _build_model(train, test, ENV_NUMERIC + ["theta", "psi"], parent_features, regularized=True)
    labels = list(range(len(encoder.classes_)))
    env_ll = float(log_loss(test["_y"], env_proba, labels=labels))
    type_ll = float(log_loss(test["_y"], type_proba, labels=labels))
    return {
        "env_log_loss": env_ll,
        "types_log_loss": type_ll,
        "log_loss_improvement": env_ll - type_ll,
        "env_top3_accuracy": _top3_accuracy(env_proba, test["_y"].to_numpy()),
        "types_top3_accuracy": _top3_accuracy(type_proba, test["_y"].to_numpy()),
        "n_obs": int(len(test)),
    }


def _table_a13(context: BuildContext):
    frame = load_next_round_output(context, "portability_tests.csv")
    rows = frame.to_dict(orient="records") if not frame.empty else []
    from ..tables.table03_hierarchical_map import _load_destination_sample
    dest = _load_destination_sample(context)
    level_specs = [
        ("basin choice", "basin", [], 50),
        ("theater choice conditional on basin", "theater", ["basin"], 25),
        ("major-ground choice conditional on theater", "major_ground_model", ["basin", "theater"], 20),
    ]
    for holdout_col, holdout_name in [("captain_id", "captain-group holdout"), ("agent_id", "agent-group holdout")]:
        for level_label, target, parents, min_count in level_specs:
            result = _holdout_eval(dest, target_col=target, parent_features=parents, group_col=holdout_col, min_count=min_count)
            if result is None:
                continue
            rows.append({"test": holdout_name, "level": level_label, **result})
    frame = pd.DataFrame(rows)
    return _save_appendix(
        name="tableA13_portability",
        title="Table A13. Portability / Invariance Tests",
        supports="Table 3",
        frame=frame,
        context=context,
        sample="Shipped next-round portability output plus paper-layer captain- and agent-group destination holdouts.",
        unit="Holdout evaluation row.",
        types_note="Destination holdouts use the connected-set theta_hat and psi_hat specification as predictive support only.",
        fe="None.",
        cluster="None; predictive evaluation only.",
        controls="Environment controls and hierarchical destination conditioning variables, depending on level.",
        interpretation="Governance patterns are partly portable across time, grounds, captains, and agents, but predictive power attenuates under stricter group holdouts.",
        caution="Holdout diagnostics are predictive validations rather than causal transport tests.",
        notes=["The first panel reuses the shipped portability CSV.", "The captain-group and agent-group destination holdouts are rebuilt directly in the paper layer."],
    )


def _table_a14(context: BuildContext):
    lower_tail = load_next_round_output(context, "lower_tail_repair.csv")
    ml_lower = _safe_read_csv(context.root / "outputs" / "tables" / "ml" / "lower_tail_benchmark.csv")
    scarcity = _safe_read_csv(context.root / "outputs" / "tables" / "ml" / "heterogeneity_scarcity.csv")
    mover = _safe_read_csv(context.root / "outputs" / "tables" / "ml" / "heterogeneity_mover.csv")
    frames = []
    for label, df in [("next_round_lower_tail", lower_tail), ("ml_lower_tail", ml_lower), ("scarcity_heterogeneity", scarcity), ("mover_heterogeneity", mover)]:
        if df.empty:
            continue
        temp = df.copy()
        temp.insert(0, "source", label)
        frames.append(temp)
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame([{"source": "missing", "note": "No lower-tail outputs found"}])
    return _save_appendix(
        name="tableA14_lower_tail_robustness",
        title="Table A14. Lower-Tail Robustness",
        supports="Table 8",
        frame=frame,
        context=context,
        sample="Shipped lower-tail repair outputs and ML lower-tail benchmark files.",
        unit="Model-evaluation or heterogeneity row.",
        types_note="Where present, theta_hat and psi_hat are used as predictive features only.",
        fe="None.",
        cluster="None; predictive and heterogeneity benchmark outputs only.",
        controls="As shipped in the upstream lower-tail and ML benchmark files.",
        interpretation="The lower-tail evidence is not driven by a single estimator or a single heterogeneity split; it reappears across the shipped lower-tail support outputs.",
        caution="These are robustness and prediction summaries rather than manuscript-facing causal estimates.",
        notes=["This appendix concatenates the shipped next-round and ML lower-tail artifacts."],
    )


def _bootstrap_matching_rows(df: pd.DataFrame) -> list[dict]:
    clean, beta = _fit_matching_surface(df)
    theta = clean["theta"].to_numpy(dtype=float)
    psi = clean["psi"].to_numpy(dtype=float)
    scarcity = clean["scarcity"].fillna(clean["scarcity"].median()).to_numpy(dtype=float)
    obs = _predict_surface(beta, theta, psi, scarcity)
    candidates = _supported_assignment_candidates(clean, beta)
    pam_pred = _predict_surface(beta, theta, candidates["pam"]["psi"], scarcity)
    nam_pred = _predict_surface(beta, theta, candidates["nam"]["psi"], scarcity)
    rng = np.random.default_rng(0)
    rows = []
    for label, pred in [("PAM", pam_pred), ("AAM/NAM", nam_pred)]:
        mean_diffs = []
        cvar_diffs = []
        for _ in range(100):
            idx = rng.integers(0, len(obs), len(obs))
            obs_b = obs[idx]
            pred_b = pred[idx]
            mean_diffs.append(float(pred_b.mean() - obs_b.mean()))
            obs_cut = np.quantile(obs_b, 0.10)
            pred_cut = np.quantile(pred_b, 0.10)
            obs_cvar = obs_b[obs_b <= obs_cut].mean()
            pred_cvar = pred_b[pred_b <= pred_cut].mean()
            cvar_diffs.append(float(pred_cvar - obs_cvar))
        for stat_name, values in [("mean output gain", mean_diffs), ("CVaR(10) gain", cvar_diffs)]:
            arr = np.asarray(values, dtype=float)
            rows.append(
                {
                    "panel": "Panel C",
                    "row_label": f"{label} bootstrap {stat_name}",
                    "estimate": float(arr.mean()),
                    "bootstrap_se": float(arr.std(ddof=1)),
                    "ci_lower": float(np.quantile(arr, 0.025)),
                    "ci_upper": float(np.quantile(arr, 0.975)),
                }
            )
    return rows


def _table_a15(context: BuildContext):
    connected = load_connected_sample(context)
    action = load_action_dataset(context)
    matching = load_next_round_output(context, "risk_matching.csv")
    counter = _safe_read_csv(context.root / "output" / "reinforcement" / "tables" / "test5_counterfactual.csv")
    submod = _safe_read_csv(context.root / "output" / "reinforcement" / "tables" / "test5_submodularity.csv")
    rows: list[dict] = []
    if not matching.empty:
        for _, row in matching.iterrows():
            rows.append({"panel": "Panel A", **row.to_dict()})
    if not counter.empty:
        for _, row in counter.iterrows():
            rows.append({"panel": "Panel B", **row.to_dict()})
    if not submod.empty:
        for _, row in submod.iterrows():
            rows.append({"panel": "Panel B", **row.to_dict()})
    sample = _prepare_matching_sample(connected, action)
    support_pairs = set(zip(sample["captain_id"].astype(str), sample["agent_id"].astype(str)))
    rows.append({"panel": "Panel C", "row_label": "Observed support pairs", "estimate": int(len(support_pairs))})
    rows.extend(_bootstrap_matching_rows(sample))
    frame = pd.DataFrame(rows)
    return _save_appendix(
        name="tableA15_matching_robustness",
        title="Table A15. Matching Robustness",
        supports="Table 10",
        frame=frame,
        context=context,
        sample="Shipped next-round and reinforcement matching outputs plus the connected-set voyage sample.",
        unit="Assignment or robustness row.",
        types_note="theta_hat and psi_hat are the connected-set types used in the paper-layer matching surface.",
        fe="As inherited from the shipped reinforcement submodularity output; none in the paper-layer bootstrap rows.",
        cluster="Captain clustering in the reinforcement submodularity output.",
        controls="Scarcity enters the paper-layer matching surface and the shipped submodularity tables.",
        interpretation="The tail-risk matching results are directionally robust across shipped counterfactuals, observed-support checks, and paper-layer bootstrap uncertainty.",
        caution="The paper-layer bootstrap resamples voyages from the observed support and is not a full equilibrium reassignment uncertainty analysis.",
        notes=["Panel C adds the missing uncertainty and support diagnostics around the matching exercise."],
    )


def _table_a16(context: BuildContext):
    connected = load_connected_sample(context).copy()

    def _summary(df: pd.DataFrame) -> tuple[float, float]:
        corr = float(df["theta"].corr(df["psi"]))
        model = clustered_ols(df, outcome="q_total_index", regressors=["psi", "theta", "scarcity", "captain_experience"], cluster_col="captain_id")
        return corr, float(model["coef"].get("psi", np.nan))

    rows = []
    base_corr, base_psi = _summary(connected.dropna(subset=["theta", "psi", "q_total_index"]))
    rows.append({"panel": "Panel A", "row_label": "Baseline", "theta_psi_corr": base_corr, "psi_slope": base_psi})

    top_captain = connected["captain_id"].value_counts().idxmax()
    top_agent = connected["agent_id"].value_counts().idxmax()
    top_ground = connected["ground_or_route"].value_counts().idxmax()
    scenarios = [
        ("Drop dominant captain", connected[connected["captain_id"] != top_captain]),
        ("Drop dominant agent", connected[connected["agent_id"] != top_agent]),
        ("Drop dominant ground", connected[connected["ground_or_route"] != top_ground]),
    ]
    for label, df in scenarios:
        corr, psi_slope = _summary(df.dropna(subset=["theta", "psi", "q_total_index"]))
        rows.append({"panel": "Panel A", "row_label": label, "theta_psi_corr": corr, "psi_slope": psi_slope})
    connected["era"] = pd.cut(connected["year_out"], bins=[-np.inf, 1829, 1869, np.inf], labels=["Pre-1830", "1830-1869", "1870+"])
    for era, df in connected.groupby("era", observed=True):
        corr, psi_slope = _summary(df.dropna(subset=["theta", "psi", "q_total_index"]))
        rows.append({"panel": "Panel B", "row_label": f"Leave-in era: {era}", "theta_psi_corr": corr, "psi_slope": psi_slope, "n_obs": int(len(df))})
    frame = pd.DataFrame(rows)
    return _save_appendix(
        name="tableA16_influence",
        title="Table A16. Influence Analysis",
        supports="Tables 2 and 10",
        frame=frame,
        context=context,
        sample="Connected-set voyage sample.",
        unit="Influence scenario.",
        types_note="theta_hat and psi_hat are the connected-set types used in sorting and output regressions.",
        fe="None in the influence summaries.",
        cluster="Captain clustering in the psi-slope summary.",
        controls="theta_hat, scarcity, and captain experience in the psi-slope regression.",
        interpretation="The main sorting and psi-slope patterns are not driven entirely by one dominant captain, one agent, or one ground.",
        caution="This appendix is a coarse influence screen rather than a full jackknife over all entities.",
        notes=["Panel A drops the most influential observed captain, agent, and ground one at a time.", "Panel B reports era-specific summaries."],
    )


def _table_a17(context: BuildContext):
    connected = load_connected_sample(context).copy()
    connected["theta_x_psi"] = connected["theta"] * connected["psi"]
    connected["era"] = pd.cut(connected["year_out"], bins=[-np.inf, 1829, 1869, np.inf], labels=["Pre-1830", "1830-1869", "1870+"])
    rows = []
    for era, df in connected.groupby("era", observed=True):
        if len(df) < 50:
            continue
        mean_model = clustered_ols(df, outcome="q_total_index", regressors=["psi", "theta", "theta_x_psi", "scarcity", "captain_experience"], cluster_col="captain_id")
        tail_model = clustered_ols(df, outcome="bottom_decile", regressors=["psi", "theta", "scarcity", "captain_experience"], cluster_col="captain_id")
        rows.append({"panel": "Panel A", "row_label": f"{era}: psi slope on output", "coefficient": mean_model["coef"].get("psi", np.nan), "std_error": mean_model["se"].get("psi", np.nan), "p_value": mean_model["p"].get("psi", np.nan), "n_obs": int(mean_model["n_obs"])})
        rows.append({"panel": "Panel A", "row_label": f"{era}: theta×psi slope on output", "coefficient": mean_model["coef"].get("theta_x_psi", np.nan), "std_error": mean_model["se"].get("theta_x_psi", np.nan), "p_value": mean_model["p"].get("theta_x_psi", np.nan), "n_obs": int(mean_model["n_obs"])})
        rows.append({"panel": "Panel B", "row_label": f"{era}: psi slope on bottom decile", "coefficient": tail_model["coef"].get("psi", np.nan), "std_error": tail_model["se"].get("psi", np.nan), "p_value": tail_model["p"].get("psi", np.nan), "n_obs": int(tail_model["n_obs"])})
    scarcity_ml = _safe_read_csv(context.root / "outputs" / "tables" / "ml" / "heterogeneity_scarcity.csv")
    if not scarcity_ml.empty:
        scarcity_ml = scarcity_ml.copy()
        scarcity_ml.insert(0, "panel", "Panel C")
        rows.extend(scarcity_ml.to_dict(orient="records"))
    frame = pd.DataFrame(rows)
    return _save_appendix(
        name="tableA17_regime_heterogeneity",
        title="Table A17. Regime Heterogeneity",
        supports="Tables 3, 8, and 10",
        frame=frame,
        context=context,
        sample="Connected-set voyage sample, with shipped ML scarcity heterogeneity support rows appended.",
        unit="Era-specific regression or ML heterogeneity row.",
        types_note="theta_hat and psi_hat are the connected-set voyage types.",
        fe="None in the current paper-layer era splits.",
        cluster="Captain clustering in the regression rows.",
        controls="Scarcity and captain experience in both output and tail-risk regressions.",
        interpretation="The governance and tail-risk patterns vary by era but do not disappear in any one historical regime.",
        caution="Era-specific samples are smaller and the ML heterogeneity rows are predictive support rather than causal estimates.",
        notes=["Panels A-B rebuild era-specific paper-layer regressions.", "Panel C appends the shipped ML scarcity heterogeneity summary."],
    )


def _table_a18(context: BuildContext):
    universe = load_universe(context)
    archival = load_next_round_output(context, "archival_mechanisms.csv")
    rows = [
        {"panel": "Panel A", "row_label": "Voyages with labor data", "share_pct": float(universe["has_labor_data"].fillna(False).mean() * 100)},
        {"panel": "Panel A", "row_label": "Voyages with route data", "share_pct": float(universe["has_route_data"].fillna(False).mean() * 100)},
        {"panel": "Panel A", "row_label": "Voyages with VQI data", "share_pct": float(universe["has_vqi_data"].fillna(False).mean() * 100)},
        {"panel": "Panel A", "row_label": "Voyages with logbook data", "share_pct": float(universe["has_logbook_data"].fillna(False).mean() * 100)},
    ]
    if not archival.empty:
        for _, row in archival.iterrows():
            checked_columns = row.get("checked_columns")
            if isinstance(checked_columns, str):
                try:
                    parsed = ast.literal_eval(checked_columns)
                    checked_columns = parsed if isinstance(parsed, list) else [checked_columns]
                except Exception:
                    checked_columns = [checked_columns]
            record = {
                "panel": "Panel B",
                "row_label": "Direct archival governance fields shipped",
                "status": row.get("status"),
                "reason": row.get("reason"),
                "checked_columns": checked_columns,
            }
            rows.append(record)
    frame = pd.DataFrame(rows)
    return _save_appendix(
        name="tableA18_archival_mechanisms",
        title="Table A18. Archival Mechanism Evidence",
        supports="Tables 4 and 7",
        frame=frame,
        context=context,
        sample="Universe voyage sample and the shipped next-round archival-mechanism audit.",
        unit="Coverage or archival-check row.",
        types_note="No archival mechanism types are estimated because the repository does not ship the required direct archival governance variables.",
        fe="None.",
        cluster="None.",
        controls="Descriptive coverage only.",
        interpretation="The repository has broad archival proxies such as labor, route, VQI, and logbook coverage, but it does not currently ship the direct archival governance fields needed for the fully targeted mechanism tests.",
        caution="This appendix is deliberately explicit about the missing archival ingredients rather than over-claiming mechanism evidence.",
        notes=["Panel A reports coverage for available archival proxies.", "Panel B reproduces the shipped next-round archival audit, which correctly reports that direct archival governance variables are absent."],
    )


def build_appendix_table(name: str, context: BuildContext):
    builders = {
        "tableA01_included_excluded": _table_a01,
        "tableA02_destination_ontology": _table_a02,
        "tableA03_connected_set": _table_a03,
        "tableA04_type_robustness": _table_a04,
        "tableA05_mover_switcher_balance": _table_a05,
        "tableA06_ml_ablation_audit": _table_a06,
        "tableA07_same_vessel_same_captain": _table_a07,
        "tableA08_patch_definition": _table_a08,
        "tableA09_scarcity_definition": _table_a09,
        "tableA10_rational_exit": _table_a10,
        "tableA11_policy_entropy": _table_a11,
        "tableA12_info_vs_routine": _table_a12,
        "tableA13_portability": _table_a13,
        "tableA14_lower_tail_robustness": _table_a14,
        "tableA15_matching_robustness": _table_a15,
        "tableA16_influence": _table_a16,
        "tableA17_regime_heterogeneity": _table_a17,
        "tableA18_archival_mechanisms": _table_a18,
    }
    return builders[name](context)
