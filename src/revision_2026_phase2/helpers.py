"""
Shared helpers: data loaders, output schema (re-exported from Phase 1),
and common panel builders used across all Phase 2 steps.
"""
from __future__ import annotations
import logging, json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence
import numpy as np, pandas as pd
from .config import (CFG, PROJECT_ROOT, DATA_FINAL, VOYAGE_PARQUET,
                     P1_INTERMEDIATES, TABLES_DIR, FIGURES_DIR, INTERMEDIATES_DIR)

logger = logging.getLogger(__name__)

# ── Re-export Phase 1 tidy schema ───────────────────────────────────────
TIDY_COLUMNS = [
    "outcome","sample_name","spec_name","term","estimate","std_error",
    "test_stat","p_value","ci_low","ci_high","n_obs","n_captains",
    "n_agents","n_vessels","fixed_effects","cluster_scheme",
    "effect_object_used","trim_rule","notes",
]

def build_tidy_row(*, outcome, sample_name, spec_name, term, estimate,
                   std_error, n_obs, n_captains=0, n_agents=0, n_vessels=0,
                   fixed_effects="", cluster_scheme="captain_id",
                   effect_object_used="", trim_rule="", notes="",
                   alpha=0.05) -> Dict[str, Any]:
    from scipy.stats import norm
    z = 1.96 if alpha == 0.05 else abs(float(norm.ppf(alpha / 2)))
    t = estimate / std_error if std_error > 0 else np.nan
    p = float(2 * (1 - norm.cdf(abs(t)))) if not np.isnan(t) else np.nan
    return dict(outcome=outcome, sample_name=sample_name, spec_name=spec_name,
                term=term, estimate=estimate, std_error=std_error,
                test_stat=t, p_value=p, ci_low=estimate - z*std_error,
                ci_high=estimate + z*std_error, n_obs=n_obs,
                n_captains=n_captains, n_agents=n_agents, n_vessels=n_vessels,
                fixed_effects=fixed_effects, cluster_scheme=cluster_scheme,
                effect_object_used=effect_object_used, trim_rule=trim_rule,
                notes=notes)

def rows_to_tidy(rows: Sequence[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for c in TIDY_COLUMNS:
        if c not in df.columns: df[c] = ""
    return df[TIDY_COLUMNS]

def save_table(df, name, *, output_dir=None, meta=None):
    d = output_dir or TABLES_DIR; d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.csv"; df.to_csv(p, index=False)
    if meta:
        with open(d / f"{name}_meta.json","w") as f: json.dump(meta, f, indent=2, default=str)
    return p

def save_md(df, name, *, output_dir=None):
    d = output_dir or TABLES_DIR; d.mkdir(parents=True, exist_ok=True)
    cols = list(df.columns)
    lines = ["| "+" | ".join(str(c) for c in cols)+" |",
             "|"+"|".join(["---"]*len(cols))+"|"]
    for _,r in df.iterrows():
        lines.append("| "+" | ".join(str(r[c]) for c in cols)+" |")
    (d / f"{name}.md").write_text("\n".join(lines)+"\n")

# ── Panel builders ──────────────────────────────────────────────────────
def load_raw_panel() -> pd.DataFrame:
    """Load voyage parquet with no filters. Adds log_q, controls, decade."""
    df = pd.read_parquet(VOYAGE_PARQUET)
    q_col = "q_total_index"
    df["q_raw"] = df[q_col].fillna(0)
    df["positive_catch"] = (df["q_raw"] > 0).astype(int)
    df["zero_catch"] = (df["q_raw"] == 0).astype(int)
    thresh = df.loc[df["q_raw"]>0, "q_raw"].quantile(CFG.near_zero_percentile / 100)
    df["near_zero"] = (df["q_raw"] <= thresh).astype(int)
    df["log_q"] = np.where(df["q_raw"]>0, np.log(df["q_raw"]), np.nan)
    if "tonnage" in df.columns:
        df["log_tonnage"] = np.log(df["tonnage"].clip(lower=1))
    if "duration_days" in df.columns:
        df["log_duration"] = np.log(df["duration_days"].clip(lower=1))
    for c in ["log_tonnage","log_duration"]:
        if c in df.columns: df[c] = df[c].fillna(df[c].median())
    if "year_out" in df.columns:
        df["decade"] = (df["year_out"]//10)*10
    return df

def load_broad_analysis_panel() -> pd.DataFrame:
    """Full panel with valid IDs and year, INCLUDING zeros."""
    df = load_raw_panel()
    df = df.dropna(subset=["captain_id","agent_id","year_out"]).copy()
    return df

def load_positive_analysis_panel() -> pd.DataFrame:
    """Positive-output panel (canonical analysis sample, pre-connected-set)."""
    df = load_broad_analysis_panel()
    return df[df["positive_catch"]==1].copy()

def load_connected_set_panel() -> pd.DataFrame:
    """LOO connected-set panel from Phase 1 or rebuilt on the fly."""
    from src.analyses.data_loader import load_voyage_data, construct_variables
    from src.analyses.connected_set import find_connected_set, find_leave_one_out_connected_set
    df = load_voyage_data()
    df = construct_variables(df)
    df_cc, _ = find_connected_set(df)
    df_conn, _ = find_leave_one_out_connected_set(df_cc)
    if "log_q" not in df_conn.columns:
        df_conn["log_q"] = np.log(df_conn["q_total_index"].clip(lower=1e-6))
    if "log_tonnage" not in df_conn.columns and "tonnage" in df_conn.columns:
        df_conn["log_tonnage"] = np.log(df_conn["tonnage"].clip(lower=1))
    if "log_duration" not in df_conn.columns and "duration_days" in df_conn.columns:
        df_conn["log_duration"] = np.log(df_conn["duration_days"].clip(lower=1))
    for c in ["log_tonnage","log_duration"]:
        if c in df_conn.columns: df_conn[c] = df_conn[c].fillna(df_conn[c].median())
    if "decade" not in df_conn.columns and "year_out" in df_conn.columns:
        df_conn["decade"] = (df_conn["year_out"]//10)*10
    return df_conn
