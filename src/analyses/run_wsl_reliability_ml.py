from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analyses.wsl_reliability_ml.departure_information import (
    build_departure_panel,
    build_information_indexes,
    compute_information_stock_features,
    export_information_stock_outputs,
)
from src.analyses.wsl_reliability_ml.policy_layer import (
    evaluate_policy_frontiers,
    fit_predeparture_policy_models,
    fit_triage_policy_models,
    run_information_equalization_counterfactual,
    score_predeparture_policies,
    score_triage_policies,
)
from src.analyses.wsl_reliability_ml.remarks_taxonomy import (
    export_remarks_taxonomy_outputs,
    predict_remarks_annotations,
    sample_remarks_goldset,
    train_remarks_models,
)
from src.analyses.wsl_reliability_ml.utils import (
    PerfTracer,
    WSLReliabilityConfig,
    build_manifest_payload,
    build_voyage_linkage,
    compute_config_hash,
    ensure_output_dirs,
    load_default_config,
    load_voyage_reference,
    load_wsl_cleaned_events,
    save_dataframe,
    write_json,
    write_markdown,
)
from src.analyses.wsl_reliability_ml.voyage_state_model import (
    build_voyage_episodes,
    create_state_anchor_labels,
    fit_voyage_state_model,
    infer_voyage_states,
    summarize_voyage_state_outputs,
)

logger = logging.getLogger(__name__)


def _build_information_audit_sample(
    departure_df: pd.DataFrame,
    remarks_df: pd.DataFrame,
    *,
    sample_n: int = 30,
) -> pd.DataFrame:
    if departure_df.empty or remarks_df.empty:
        return pd.DataFrame(columns=["departure_id"])
    sampled = departure_df.sample(min(sample_n, len(departure_df)), random_state=42)
    rows: list[dict[str, Any]] = []
    for row in sampled.itertuples(index=False):
        relevant = remarks_df[
            (remarks_df["issue_date"] <= row.departure_issue_date)
            & (remarks_df["destination_basin"] == row.departure_destination_basin)
        ].copy()
        if relevant.empty:
            continue
        delta = (row.departure_issue_date - relevant["issue_date"]).dt.days
        relevant["contribution_tau90"] = relevant["row_weight"] * np.exp(-delta / 90.0)
        for source in relevant.sort_values("contribution_tau90", ascending=False).head(10).itertuples(index=False):
            rows.append(
                {
                    "departure_id": row.departure_id,
                    "departure_issue_date": row.departure_issue_date,
                    "departure_basin": row.departure_destination_basin,
                    "source_event_row_id": source.event_row_id,
                    "source_issue_date": source.issue_date,
                    "source_page_type": source.page_type,
                    "source_primary_class": source.primary_class,
                    "source_reason_codes": source.reason_codes,
                    "contribution_tau90": source.contribution_tau90,
                }
            )
    return pd.DataFrame(rows)


def _prepare_predeparture_panel(
    departure_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    voyage_ref: pd.DataFrame,
    voyage_summary: pd.DataFrame,
) -> pd.DataFrame:
    panel = departure_df.merge(feature_df, on=["departure_id", "voyage_id", "departure_issue_date"], how="left")
    keep = [
        "voyage_id",
        "q_total_index",
        "tonnage",
        "home_port",
        "ground_or_route",
        "theta",
        "psi",
        "theta_hat_holdout",
        "psi_hat_holdout",
        "novice",
        "expert",
        "captain_voyage_num",
        "zero_catch_or_failure",
        "log_output",
    ]
    keep = [column for column in keep if column in voyage_ref.columns]
    panel = panel.merge(voyage_ref[keep].drop_duplicates("voyage_id"), on="voyage_id", how="left")
    panel = panel.merge(voyage_summary, on="voyage_id", how="left")
    panel["departure_month"] = pd.to_datetime(panel["departure_issue_date"]).dt.month
    if "zero_catch_or_failure" not in panel.columns:
        panel["zero_catch_or_failure"] = (
            pd.to_numeric(panel.get("q_total_index"), errors="coerce").fillna(0) <= 0
        ).astype(int)
    if "log_output" not in panel.columns:
        panel["log_output"] = np.log1p(pd.to_numeric(panel.get("q_total_index"), errors="coerce").fillna(0).clip(lower=0))
    if "home_port" not in panel.columns:
        panel["home_port"] = pd.NA
    if "home_port_norm" in panel.columns:
        panel["home_port"] = panel["home_port"].combine_first(panel["home_port_norm"])
    return panel


def _prepare_triage_panel(
    states_df: pd.DataFrame,
    voyage_summary: pd.DataFrame,
    voyage_ref: pd.DataFrame,
    info_panel: pd.DataFrame,
) -> pd.DataFrame:
    bad_rows = states_df[states_df["most_likely_state"].isin({"distress_at_sea", "in_port_interruption_or_repair", "terminal_loss"})].copy()
    if bad_rows.empty:
        return pd.DataFrame()
    first_bad = bad_rows.sort_values(["episode_id", "issue_date"]).drop_duplicates("episode_id")
    keep_state = [
        "episode_id",
        "voyage_id",
        "issue_date",
        "most_likely_state",
        "captain_id",
        "agent_id",
        "vessel_id",
        "p_state__distress_at_sea",
        "p_state__in_port_interruption_or_repair",
        "p_state__terminal_loss",
    ]
    triage = first_bad[keep_state].rename(
        columns={
            "issue_date": "triage_issue_date",
            "most_likely_state": "triage_state",
        }
    )
    triage = triage.merge(voyage_summary, on=["episode_id", "voyage_id", "captain_id", "agent_id", "vessel_id"], how="left")
    triage = triage.merge(
        voyage_ref[
            [
                column
                for column in [
                    "voyage_id",
                    "theta_hat_holdout",
                    "novice",
                    "tonnage",
                    "home_port",
                    "zero_catch_or_failure",
                    "log_output",
                ]
                if column in voyage_ref.columns
            ]
        ].drop_duplicates("voyage_id"),
        on="voyage_id",
        how="left",
    )
    triage = triage.merge(
        info_panel[
            [
                column
                for column in [
                    "voyage_id",
                    "departure_destination_basin",
                    "risk_index",
                    "portfolio_information_index",
                    "information_advantage_index",
                    "agent_recent_bad_state_rate_tau180",
                    "agent_recent_recovery_rate_tau180",
                ]
                if column in info_panel.columns
            ]
        ].drop_duplicates("voyage_id"),
        on="voyage_id",
        how="left",
    )
    if "home_port" not in triage.columns:
        triage["home_port"] = "UNK"
    else:
        triage["home_port"] = triage["home_port"].fillna("UNK")
    return triage


def _build_policy_diagnostics(
    pre_bundle: dict[str, Any],
    triage_bundle: dict[str, Any],
    pre_frontier: pd.DataFrame,
    triage_frontier: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "predeparture": pre_bundle.get("diagnostics", {}),
        "triage": triage_bundle.get("diagnostics", {}),
        "predeparture_frontier_best": pre_frontier.sort_values("policy_value", ascending=False).head(5).to_dict(orient="records"),
        "triage_frontier_best": triage_frontier.sort_values("policy_value", ascending=False).head(5).to_dict(orient="records"),
    }


def _write_results_summary(
    config: WSLReliabilityConfig,
    output_paths: dict[str, Path],
    events_df: pd.DataFrame,
    remarks_df: pd.DataFrame,
    voyage_summary: pd.DataFrame,
    state_diagnostics: dict[str, Any],
    pre_bundle: dict[str, Any],
    triage_bundle: dict[str, Any],
) -> Path:
    metrics_path = output_paths["remarks"] / "remarks_model_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    failures: list[str] = []
    if metrics.get("label_source") != "manual":
        failures.append("Remarks models are currently trained and evaluated on weak-rule labels, not adjudicated gold annotations.")
    if metrics.get("distress_recall", 0.0) < 0.90:
        failures.append(f"Distress recall is below target at {metrics.get('distress_recall', float('nan')):.3f}.")
    if metrics.get("terminal_loss_precision", 0.0) < 0.95:
        failures.append(f"Terminal-loss precision is below target at {metrics.get('terminal_loss_precision', float('nan')):.3f}.")
    contamination_auc = metrics.get("contamination_auroc")
    if contamination_auc is None or (isinstance(contamination_auc, float) and np.isnan(contamination_auc)) or contamination_auc < 0.90:
        failures.append(f"Contamination AUROC is below target or unavailable ({contamination_auc}).")
    if state_diagnostics.get("absorbing_state_violations", 0) > 0:
        failures.append("Absorbing-state violations were detected in the state decoder diagnostics.")
    pre_overlap = pre_bundle.get("diagnostics", {}).get("overlap", {})
    triage_overlap = triage_bundle.get("diagnostics", {}).get("overlap", {})
    if pre_overlap.get("share_outside_clip", 0.0) > 0.25:
        failures.append(f"Predeparture overlap is thin: {pre_overlap.get('share_outside_clip', 0.0):.1%} outside the clip range.")
    if triage_overlap.get("share_outside_clip", 0.0) > 0.25:
        failures.append(f"Triage overlap is thin: {triage_overlap.get('share_outside_clip', 0.0):.1%} outside the clip range.")

    remarks_high_conf = remarks_df[remarks_df["_confidence"] >= 0.90]
    uncertainty_lines = []
    for label in ["distress_hazard", "terminal_loss", "positive_productivity"]:
        all_share = float((remarks_df["primary_class"] == label).mean()) if len(remarks_df) else np.nan
        hc_share = float((remarks_high_conf["primary_class"] == label).mean()) if len(remarks_high_conf) else np.nan
        uncertainty_lines.append(f"- `{label}` share: all-data={all_share:.3f}, high-confidence-only={hc_share:.3f}")

    summary = [
        "# WSL Reliability ML Results Summary",
        "",
        f"- Config hash: `{compute_config_hash(config)}`",
        f"- Events loaded: {len(events_df):,}",
        f"- Remarks rows scored: {len(remarks_df):,}",
        f"- Voyage episodes: {voyage_summary['episode_id'].nunique() if 'episode_id' in voyage_summary.columns else len(voyage_summary):,}",
        f"- Linked voyages with state summaries: {voyage_summary['voyage_id'].notna().sum() if 'voyage_id' in voyage_summary.columns else 0:,}",
        "",
        "## QA Checks",
        "",
        f"- Page-type separation audit: weekly rows={int((events_df['page_type'] == 'weekly_event_flow').sum()):,}, registry rows={int((events_df['page_type'] == 'fleet_registry_stock').sum()):,}, non-table rows={int((events_df['page_type'] == 'non_table_or_other').sum()):,}.",
        "- Timing audit: departure and policy features were constructed only from source rows with `issue_date <= snapshot_date`; audit samples are exported in `information_stock_audit_sample.csv`.",
        "- Uncertainty audit:",
        *uncertainty_lines,
        "- Remarks audit: `remarks_error_audit.csv` contains the highest-distress, highest-contamination, and routine review slices.",
        f"- State audit: absorbing violations={state_diagnostics.get('absorbing_state_violations', 'NA')}, impossible transitions={state_diagnostics.get('impossible_transition_count', 'NA')}.",
        f"- Policy audit: predeparture overlap outside clip={pre_overlap.get('share_outside_clip', float('nan')):.1%}, triage overlap outside clip={triage_overlap.get('share_outside_clip', float('nan')):.1%}.",
        "",
        "## Major Check Failures",
        "",
    ]
    if failures:
        summary.extend([f"- {failure}" for failure in failures])
    else:
        summary.append("- No major automated QA failure triggered on this run.")
    path = config.output_root / "RESULTS_SUMMARY.md"
    write_markdown(path, "\n".join(summary))
    return path


def run_pipeline(config: WSLReliabilityConfig) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    tracer = PerfTracer()
    output_dirs = ensure_output_dirs(config)

    with tracer.span("load_events"):
        events_df = load_wsl_cleaned_events(config)
        if getattr(config, 'max_events', None) and len(events_df) > config.max_events:
            logger.info("Subsampling from %d to %d events", len(events_df), config.max_events)
            events_df = events_df.iloc[:config.max_events].copy()
        tracer.set_metadata(n_events=len(events_df))
        logger.info("Loaded %d flattened WSL events", len(events_df))

    with tracer.span("voyage_linkage"):
        voyage_ref = load_voyage_reference(config)
        tracer.set_metadata(n_voyages=len(voyage_ref))
        linkage_df = build_voyage_linkage(events_df, voyage_ref, config)
        linked_events = events_df.merge(linkage_df, on="event_row_id", how="left")
        linked_events["episode_id"] = linked_events["voyage_id"].fillna(linked_events["episode_fallback_key"])
        tracer.set_metadata(
            n_linked=int(linkage_df["voyage_id"].notna().sum()),
            n_unlinked=int(linkage_df["voyage_id"].isna().sum()),
        )

    with tracer.span("remarks_taxonomy"):
        with tracer.span("remarks_gold_sample"):
            gold_sample = sample_remarks_goldset(linked_events, config)
            label_source = "manual" if "primary_class" in gold_sample.columns and gold_sample["primary_class"].notna().sum() >= 100 else "weak_rules"
            train_df = gold_sample if label_source == "manual" else linked_events
            tracer.set_metadata(label_source=label_source, n_gold=len(gold_sample))
        with tracer.span("remarks_train"):
            remarks_models = train_remarks_models(train_df, config)
            tracer.set_metadata(
                n_train=remarks_models["metrics"]["n_train"],
                n_test=remarks_models["metrics"]["n_test"],
                macro_f1=remarks_models["metrics"]["primary_class_macro_f1"],
            )
        with tracer.span("remarks_predict"):
            remarks_predictions = predict_remarks_annotations(linked_events, remarks_models, config)
            tracer.set_metadata(n_predicted=len(remarks_predictions))
        with tracer.span("remarks_export"):
            remarks_outputs = export_remarks_taxonomy_outputs(output_dirs["remarks"], gold_sample, remarks_predictions, remarks_models)

    with tracer.span("voyage_state_model"):
        with tracer.span("build_episodes"):
            episode_df = build_voyage_episodes(remarks_predictions, linkage_df, config)
            tracer.set_metadata(n_episodes=int(episode_df["episode_id"].nunique()), n_mentions=len(episode_df))
        with tracer.span("state_anchors"):
            anchors_df = create_state_anchor_labels(episode_df, remarks_predictions, config)
            tracer.set_metadata(n_anchored=int(anchors_df["anchor_state"].notna().sum()))
        with tracer.span("fit_state_model"):
            state_model = fit_voyage_state_model(episode_df.merge(anchors_df, on="event_row_id", how="left"), anchors_df, config)
        with tracer.span("hmm_inference"):
            state_posteriors = infer_voyage_states(
                episode_df.merge(anchors_df, on="event_row_id", how="left"),
                state_model,
                config,
                tracer=tracer,
            )
        with tracer.span("state_summary"):
            voyage_summary, state_diagnostics, transition_df = summarize_voyage_state_outputs(state_posteriors, config)
            voyage_summary = voyage_summary.merge(
                voyage_ref[
                    [
                        column
                        for column in [
                            "voyage_id",
                            "theta_hat_holdout",
                            "novice",
                            "tonnage",
                            "q_total_index",
                            "home_port",
                        ]
                        if column in voyage_ref.columns
                    ]
                ].drop_duplicates("voyage_id"),
                on="voyage_id",
                how="left",
            )
            tracer.set_metadata(
                n_bad_state_episodes=int(voyage_summary["ever_bad_state"].sum()),
                absorbing_violations=state_diagnostics.get("absorbing_state_violations", 0),
            )
        with tracer.span("state_export"):
            state_paths = {
                "episode_table": output_dirs["states"] / "voyage_episode_table.parquet",
                "anchors": output_dirs["states"] / "state_anchor_labels.parquet",
                "posteriors": output_dirs["states"] / "voyage_state_posteriors.parquet",
                "summary": output_dirs["states"] / "voyage_state_summary.parquet",
                "diagnostics": output_dirs["states"] / "state_model_diagnostics.json",
                "transitions": output_dirs["states"] / "state_transition_matrix.csv",
                "first_bad_timing": output_dirs["states"] / "first_bad_state_timing_distribution.csv",
            }
            save_dataframe(episode_df, state_paths["episode_table"])
            save_dataframe(anchors_df, state_paths["anchors"])
            save_dataframe(state_posteriors, state_paths["posteriors"])
            save_dataframe(voyage_summary, state_paths["summary"])
            transition_df.to_csv(state_paths["transitions"])
            first_bad_timing = (
                voyage_summary["time_to_first_bad_state_days"]
                .dropna()
                .pipe(lambda series: pd.DataFrame({"time_to_first_bad_state_days": series}))
            )
            first_bad_timing.to_csv(state_paths["first_bad_timing"], index=False)
            write_json(state_paths["diagnostics"], state_diagnostics | state_model["diagnostics"])

    with tracer.span("departure_information"):
        with tracer.span("build_departure_panel"):
            state_enriched = state_posteriors.merge(
                voyage_summary[["episode_id", "voyage_id", "ever_bad_state", "ever_recovered_after_bad_state"]],
                on=["episode_id", "voyage_id"],
                how="left",
            )
            departure_panel = build_departure_panel(state_enriched, linkage_df, config)
            tracer.set_metadata(n_departures=len(departure_panel))
        with tracer.span("info_stock_features"):
            info_raw = compute_information_stock_features(departure_panel, state_enriched, config, tracer=tracer)
        with tracer.span("info_indexes"):
            info_indexes = build_information_indexes(info_raw, config)
            info_panel = _prepare_predeparture_panel(departure_panel, info_indexes, voyage_ref, voyage_summary)
        with tracer.span("info_export"):
            info_audit = _build_information_audit_sample(departure_panel, remarks_predictions)
            info_metadata = {
                "config_hash": compute_config_hash(config),
                "n_departures": int(len(departure_panel)),
                "n_matched_departures": int(departure_panel["matched_to_voyage"].sum()) if "matched_to_voyage" in departure_panel.columns else 0,
                "feature_columns": sorted([column for column in info_raw.columns if column not in {"departure_id", "voyage_id", "departure_issue_date"}]),
            }
            info_outputs = export_information_stock_outputs(
                output_dirs["information"],
                departure_panel,
                info_raw,
                info_indexes,
                info_metadata,
                info_audit,
            )
            group_dist = (
                info_indexes.assign(agent_group=info_panel.get("agent_norm"), port_group=info_panel.get("home_port_norm"))
                .groupby("departure_decade")[
                    ["public_information_index", "portfolio_information_index", "risk_index", "information_advantage_index"]
                ]
                .agg(["mean", "std"])
            )
            group_dist.to_csv(output_dirs["information"] / "information_index_distributions_by_group.csv")

    with tracer.span("policy_layer"):
        with tracer.span("predeparture_policy"):
            pre_bundle = fit_predeparture_policy_models(info_panel, config)
            pre_scores = score_predeparture_policies(info_panel, pre_bundle, config)
            pre_frontier = evaluate_policy_frontiers(pre_scores, config)
            tracer.set_metadata(n_pre_scores=len(pre_scores), n_pre_frontier=len(pre_frontier))
        with tracer.span("triage_policy"):
            triage_panel = _prepare_triage_panel(state_posteriors, voyage_summary, voyage_ref, info_panel)
            triage_bundle = fit_triage_policy_models(triage_panel, config) if not triage_panel.empty else {"scored_df": pd.DataFrame(), "diagnostics": {}}
            triage_scores = score_triage_policies(triage_panel, triage_bundle, config) if not triage_panel.empty else pd.DataFrame()
            triage_frontier = evaluate_policy_frontiers(triage_scores, config) if not triage_scores.empty else pd.DataFrame()
            tracer.set_metadata(n_triage_scores=len(triage_scores), n_triage_panel=len(triage_panel))
        with tracer.span("equalization"):
            equalization = run_information_equalization_counterfactual(info_panel, config)
        with tracer.span("policy_export"):
            policy_paths = {
                "pre_scores": output_dirs["policy"] / "predeparture_policy_scores.parquet",
                "pre_frontier": output_dirs["policy"] / "predeparture_policy_frontier.csv",
                "triage_scores": output_dirs["policy"] / "triage_policy_scores.parquet",
                "triage_frontier": output_dirs["policy"] / "triage_policy_frontier.csv",
                "equalization": output_dirs["policy"] / "information_equalization_counterfactual.csv",
                "diagnostics": output_dirs["policy"] / "policy_diagnostics.json",
            }
            save_dataframe(pre_scores, policy_paths["pre_scores"])
            pre_frontier.to_csv(policy_paths["pre_frontier"], index=False)
            if not triage_scores.empty:
                save_dataframe(triage_scores, policy_paths["triage_scores"])
                triage_frontier.to_csv(policy_paths["triage_frontier"], index=False)
            else:
                pd.DataFrame().to_parquet(policy_paths["triage_scores"], index=False)
                pd.DataFrame().to_csv(policy_paths["triage_frontier"], index=False)
            equalization.to_csv(policy_paths["equalization"], index=False)
            policy_diagnostics = _build_policy_diagnostics(pre_bundle, triage_bundle, pre_frontier, triage_frontier if not triage_frontier.empty else pd.DataFrame(columns=pre_frontier.columns))
            write_json(policy_paths["diagnostics"], policy_diagnostics)

    with tracer.span("finalize"):
        summary_path = _write_results_summary(
            config,
            output_dirs,
            events_df,
            remarks_predictions,
            voyage_summary,
            state_diagnostics,
            pre_bundle,
            triage_bundle,
        )
        output_files = [
            *remarks_outputs.values(),
            *state_paths.values(),
            *info_outputs.values(),
            *policy_paths.values(),
            summary_path,
        ]
        manifest_payload = build_manifest_payload(
            config,
            [path for path in output_files if Path(path).exists()],
            extra={
                "counts": {
                    "events": int(len(events_df)),
                    "remarks_rows": int(len(remarks_predictions)),
                    "episodes": int(episode_df["episode_id"].nunique()),
                    "departures": int(len(departure_panel)),
                    "predeparture_policy_rows": int(len(pre_scores)),
                    "triage_policy_rows": int(len(triage_scores)),
                },
                "perf_summary": tracer.summary_table(),
            },
        )
        manifest_path = config.output_root / "manifest.json"
        write_json(manifest_path, manifest_payload)
        trace_path = config.output_root / "perf_trace.json"
        tracer.export_json(trace_path)

    return {
        "manifest_path": manifest_path,
        "summary_path": summary_path,
        "remarks_outputs": remarks_outputs,
        "state_paths": state_paths,
        "information_outputs": info_outputs,
        "policy_paths": policy_paths,
        "perf_trace": tracer.to_dict(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the WSL reliability ML pipeline")
    parser.add_argument("--cleaned-events-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--gold-sample-size-target", type=int, default=6000)
    parser.add_argument("--remarks-max-train-rows", type=int, default=40000)
    parser.add_argument("--max-events", type=int, default=None,
                        help="Limit events for fast iteration (e.g. 10000)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_default_config()
    if args.cleaned_events_path is not None:
        config.cleaned_events_path = args.cleaned_events_path
    if args.output_root is not None:
        config.output_root = args.output_root
    config.gold_sample_size_target = args.gold_sample_size_target
    config.remarks_max_train_rows = args.remarks_max_train_rows
    if args.max_events:
        config.max_events = args.max_events
    outputs = run_pipeline(config)
    print(f"WSL reliability ML outputs written to {config.output_root}")
    print(f"Manifest: {outputs['manifest_path']}")
    print(f"Summary: {outputs['summary_path']}")
    perf = outputs.get("perf_trace", {})
    print(f"Total time: {perf.get('total_elapsed_seconds', 0):.1f}s")
    print(f"Peak memory: {perf.get('peak_memory_mb', 0):.0f} MB")
    for row in perf.get("summary", []):
        print(f"  {row['stage']:.<40s} {row['elapsed_seconds']:>8.1f}s  {row.get('memory_delta_mb', 0):>+7.0f} MB")

if __name__ == "__main__":
    main()
