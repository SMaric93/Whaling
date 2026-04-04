"""
Step 1 — Sample lineage and filter audit.

Traces every key sample from raw panel through to each regression sample.
Reports exact observation counts and reason codes for every drop.
"""
from __future__ import annotations
import logging
import numpy as np, pandas as pd
from .config import CFG, INTERMEDIATES_DIR, TABLES_DIR, P1_INTERMEDIATES
from .helpers import (load_raw_panel, load_broad_analysis_panel,
                      load_positive_analysis_panel, load_connected_set_panel,
                      save_table, save_md)

logger = logging.getLogger(__name__)

def _sample_stats(df, name):
    q = df.get("q_raw", df.get("q_total_index"))
    conn_psi = P1_INTERMEDIATES / "psi_hat_leave_one_captain_out.parquet"
    psi_cov = 0.0
    if conn_psi.exists() and "voyage_id" in df.columns:
        psi = pd.read_parquet(conn_psi)
        merged = df[["voyage_id"]].merge(psi[["voyage_id","psi_hat_loo"]], on="voyage_id", how="left")
        psi_cov = float(merged["psi_hat_loo"].notna().mean())
    cap_agents = df.groupby("captain_id")["agent_id"].nunique() if "captain_id" in df.columns and "agent_id" in df.columns else pd.Series(dtype=int)
    return dict(
        sample_name=name,
        n_voyages=len(df),
        n_captains=int(df["captain_id"].nunique()) if "captain_id" in df.columns else 0,
        n_agents=int(df["agent_id"].nunique()) if "agent_id" in df.columns else 0,
        n_vessels=int(df["vessel_id"].nunique()) if "vessel_id" in df.columns else 0,
        zero_catch_pct=float((q<=0).mean()*100) if q is not None else None,
        near_zero_pct=float(df["near_zero"].mean()*100) if "near_zero" in df.columns else None,
        mean_log_q=float(df["log_q"].dropna().mean()) if "log_q" in df.columns else None,
        sd_log_q=float(df["log_q"].dropna().std()) if "log_q" in df.columns else None,
        mean_tonnage=float(df["tonnage"].mean()) if "tonnage" in df.columns else None,
        mover_pct=float((cap_agents>=2).mean()*100) if len(cap_agents)>0 else None,
        psi_connected_loo_coverage_pct=round(psi_cov*100,1),
    )

def build_sample_lineage():
    logger.info("="*60+"\nSTEP 1: SAMPLE LINEAGE AND FILTER AUDIT\n"+"="*60)
    df_raw = load_raw_panel()
    df_broad = load_broad_analysis_panel()
    df_pos = load_positive_analysis_panel()
    df_conn = load_connected_set_panel()

    # Effect-covered connected set
    psi_path = P1_INTERMEDIATES / "psi_hat_leave_one_captain_out.parquet"
    if psi_path.exists():
        psi = pd.read_parquet(psi_path)
        df_eff = df_conn.merge(psi[["voyage_id","psi_hat_loo"]], on="voyage_id", how="inner")
        df_eff = df_eff[df_eff["psi_hat_loo"].notna()].copy()
    else:
        df_eff = df_conn.copy()

    # Mover/switcher sample
    df_switch = df_pos.sort_values(["captain_id","year_out"]).copy()
    df_switch["prev_agent"] = df_switch.groupby("captain_id")["agent_id"].shift(1)
    df_switch["switched"] = ((df_switch["agent_id"]!=df_switch["prev_agent"])&df_switch["prev_agent"].notna()).astype(int)
    cap_agents = df_switch.groupby("captain_id")["agent_id"].nunique()
    movers = cap_agents[cap_agents>=2].index
    df_movers = df_switch[df_switch["captain_id"].isin(movers)].copy()

    samples = [
        _sample_stats(df_raw,"1_raw_panel"),
        _sample_stats(df_broad,"2_broad_with_zeros"),
        _sample_stats(df_pos,"3_positive_output"),
        _sample_stats(df_conn,"4_connected_set_loo"),
        _sample_stats(df_eff,"5_effect_covered_connected"),
        _sample_stats(df_movers,"6_mover_switcher"),
    ]
    lineage = pd.DataFrame(samples)
    save_table(lineage, "table_sample_lineage")
    save_md(lineage, "table_sample_lineage")
    logger.info("  Lineage table: %d samples", len(lineage))

    # ── Filter-loss table ────────────────────────────────────────────────
    # Track every observation from connected set → effect-covered → main regression
    conn_ids = set(df_conn["voyage_id"])
    eff_ids = set(df_eff["voyage_id"]) if "voyage_id" in df_eff.columns else set()
    lost = df_conn[~df_conn["voyage_id"].isin(eff_ids)].copy() if "voyage_id" in df_conn.columns else pd.DataFrame()
    reason_rows = []
    if len(lost) > 0:
        # Missing psi_hat
        if psi_path.exists():
            psi_ids = set(pd.read_parquet(psi_path)["voyage_id"])
            for _,r in lost.iterrows():
                reasons = []
                if r["voyage_id"] not in psi_ids:
                    reasons.append("missing_psi_hat_loo")
                if "log_tonnage" in r and pd.isna(r.get("log_tonnage")):
                    reasons.append("missing_controls")
                reason_rows.append(dict(voyage_id=r["voyage_id"],
                                        captain_id=r.get("captain_id"),
                                        agent_id=r.get("agent_id"),
                                        reason="; ".join(reasons) if reasons else "outlier_trim"))
    filter_df = pd.DataFrame(reason_rows) if reason_rows else pd.DataFrame(columns=["voyage_id","captain_id","agent_id","reason"])
    save_table(filter_df, "table_filter_lineage_connected_to_effects")
    logger.info("  Filter-loss rows: %d", len(filter_df))

    # Save sample IDs
    id_dict = {}
    for name, df_s in [("raw",df_raw),("broad",df_broad),("positive",df_pos),
                        ("connected",df_conn),("effect_covered",df_eff),("movers",df_movers)]:
        if "voyage_id" in df_s.columns:
            id_dict[name] = df_s["voyage_id"].values
    max_len = max(len(v) for v in id_dict.values())
    id_df = pd.DataFrame({k: pd.Series(v) for k,v in id_dict.items()})
    id_df.to_parquet(INTERMEDIATES_DIR / "sample_ids_by_stage.parquet", index=False)

    return lineage
