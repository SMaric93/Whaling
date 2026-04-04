"""
Connected-set representativeness checks (Step 3).

Compares the LOO connected set to the broader analysis sample on key
observables, decade composition, and route composition.  Optionally
computes IPW-reweighted means via a logistic selection model.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import CFG, DATA_FINAL, VOYAGE_PARQUET, CANONICAL_CONNECTED_SET, TABLES_DIR
from .output_schema import save_result_table, save_markdown_table

logger = logging.getLogger(__name__)


def _standardized_mean_diff(mean1: float, mean0: float, std_pool: float) -> float:
    """Compute standardized mean difference (Cohen's d)."""
    if std_pool == 0:
        return 0.0
    return (mean1 - mean0) / std_pool


def run_representativeness_checks() -> pd.DataFrame:
    """Compare connected set to full analysis sample."""
    logger.info("=" * 60)
    logger.info("STEP 3: CONNECTED-SET REPRESENTATIVENESS")
    logger.info("=" * 60)

    # Load full analysis sample
    df_full = pd.read_parquet(VOYAGE_PARQUET)
    df_full = df_full.dropna(subset=["captain_id", "agent_id", "year_out"]).copy()
    q_col = "q_total_index" if "q_total_index" in df_full.columns else "q_oil_bbl"
    if q_col in df_full.columns:
        df_full = df_full[df_full[q_col] > 0].copy()
        df_full["log_q"] = np.log(df_full[q_col].clip(lower=1e-6))

    # Load connected set
    if CANONICAL_CONNECTED_SET.exists():
        df_conn = pd.read_parquet(CANONICAL_CONNECTED_SET)
    else:
        from src.analyses.connected_set import find_connected_set, find_leave_one_out_connected_set
        df_cc, _ = find_connected_set(df_full)
        df_conn, _ = find_leave_one_out_connected_set(df_cc)

    # Tag membership
    conn_voyages = set(df_conn["voyage_id"]) if "voyage_id" in df_conn.columns else set()
    df_full["in_connected_set"] = df_full["voyage_id"].isin(conn_voyages).astype(int)

    logger.info("  Full analysis: %d, Connected set: %d (%.1f%%)",
                len(df_full), df_full["in_connected_set"].sum(),
                100 * df_full["in_connected_set"].mean())

    # Variables to compare
    compare_cols = []
    for col in ["log_q", "tonnage", "crew", "crew_size", "duration_days", "year_out"]:
        if col in df_full.columns:
            compare_cols.append(col)

    rows = []
    for col in compare_cols:
        valid = df_full[col].dropna()
        in_conn = valid[df_full.loc[valid.index, "in_connected_set"] == 1]
        not_conn = valid[df_full.loc[valid.index, "in_connected_set"] == 0]

        if len(in_conn) == 0 or len(not_conn) == 0:
            continue

        pooled_std = float(valid.std())
        smd = _standardized_mean_diff(float(in_conn.mean()), float(not_conn.mean()), pooled_std)

        rows.append({
            "variable": col,
            "mean_connected": float(in_conn.mean()),
            "sd_connected": float(in_conn.std()),
            "mean_not_connected": float(not_conn.mean()),
            "sd_not_connected": float(not_conn.std()),
            "standardized_mean_diff": smd,
            "n_connected": len(in_conn),
            "n_not_connected": len(not_conn),
            "balance_ok": abs(smd) < 0.25,
        })

    # Decade composition comparison
    if "year_out" in df_full.columns:
        df_full["decade"] = (df_full["year_out"] // 10) * 10
        for decade in sorted(df_full["decade"].unique()):
            full_share = (df_full["decade"] == decade).mean()
            conn_share = (df_full[df_full["in_connected_set"] == 1]["decade"] == decade).mean() \
                if df_full["in_connected_set"].sum() > 0 else 0
            rows.append({
                "variable": f"decade_{decade}_share",
                "mean_connected": float(conn_share),
                "sd_connected": 0.0,
                "mean_not_connected": float(full_share),
                "sd_not_connected": 0.0,
                "standardized_mean_diff": float(conn_share - full_share),
                "n_connected": int(df_full["in_connected_set"].sum()),
                "n_not_connected": int((1 - df_full["in_connected_set"]).sum()),
                "balance_ok": abs(conn_share - full_share) < 0.10,
            })

    # Route composition (if available)
    route_col = "ground_or_route" if "ground_or_route" in df_full.columns else None
    if route_col:
        top_routes = df_full[route_col].value_counts().head(10).index
        for route in top_routes:
            full_share = (df_full[route_col] == route).mean()
            conn_sub = df_full[df_full["in_connected_set"] == 1]
            conn_share = (conn_sub[route_col] == route).mean() if len(conn_sub) > 0 else 0
            rows.append({
                "variable": f"route_{route}_share",
                "mean_connected": float(conn_share),
                "sd_connected": 0.0,
                "mean_not_connected": float(full_share),
                "sd_not_connected": 0.0,
                "standardized_mean_diff": float(conn_share - full_share),
                "n_connected": int(df_full["in_connected_set"].sum()),
                "n_not_connected": int((1 - df_full["in_connected_set"]).sum()),
                "balance_ok": abs(conn_share - full_share) < 0.10,
            })

    result = pd.DataFrame(rows)

    # IPW reweighting (optional, logistic selection model)
    try:
        result_ipw = _compute_ipw_reweighted_means(df_full, compare_cols)
        if result_ipw is not None:
            result = pd.concat([result, result_ipw], ignore_index=True)
    except Exception as e:
        logger.warning("  IPW reweighting failed: %s", e)

    # Save
    save_result_table(result, "table_connected_set_representativeness", metadata={
        "description": "Connected set vs full sample balance checks",
        "n_imbalanced": int((~result["balance_ok"]).sum()) if "balance_ok" in result.columns else 0,
    })
    save_markdown_table(result, "table_connected_set_representativeness")

    n_imbalanced = (~result["balance_ok"]).sum() if "balance_ok" in result.columns else 0
    if n_imbalanced > 0:
        logger.warning("  %d variables show imbalance (|SMD| >= 0.25)", n_imbalanced)
    else:
        logger.info("  All variables balanced (|SMD| < 0.25)")

    return result


def _compute_ipw_reweighted_means(
    df: pd.DataFrame,
    compare_cols: List[str],
) -> Optional[pd.DataFrame]:
    """Compute IPW-reweighted means using logistic propensity score."""
    from sklearn.linear_model import LogisticRegression

    features = [c for c in compare_cols if c != "log_q" and c in df.columns]
    if not features or "in_connected_set" not in df.columns:
        return None

    df_valid = df.dropna(subset=features + ["in_connected_set"]).copy()
    if len(df_valid) < 100:
        return None

    X = df_valid[features].values
    y = df_valid["in_connected_set"].values

    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(X, y)
    p_hat = lr.predict_proba(X)[:, 1].clip(0.05, 0.95)

    # IPW weights for non-connected observations
    df_valid["ipw_weight"] = np.where(
        df_valid["in_connected_set"] == 1, 1.0, p_hat / (1 - p_hat)
    )

    rows = []
    for col in compare_cols:
        if col not in df_valid.columns:
            continue
        not_conn = df_valid[df_valid["in_connected_set"] == 0]
        if len(not_conn) == 0:
            continue
        weighted_mean = float(np.average(not_conn[col], weights=not_conn["ipw_weight"]))
        conn_mean = float(df_valid[df_valid["in_connected_set"] == 1][col].mean())
        pooled_std = float(df_valid[col].std())
        smd = _standardized_mean_diff(conn_mean, weighted_mean, pooled_std)
        rows.append({
            "variable": f"{col}_ipw_reweighted",
            "mean_connected": conn_mean,
            "sd_connected": float(df_valid[df_valid["in_connected_set"] == 1][col].std()),
            "mean_not_connected": weighted_mean,
            "sd_not_connected": 0.0,
            "standardized_mean_diff": smd,
            "n_connected": int(df_valid["in_connected_set"].sum()),
            "n_not_connected": int((1 - df_valid["in_connected_set"]).sum()),
            "balance_ok": abs(smd) < 0.25,
        })

    return pd.DataFrame(rows) if rows else None
