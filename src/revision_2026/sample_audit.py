"""
Sample Crosswalk and Audit Table (Step 2).

Builds the authoritative sample crosswalk showing N voyages, N captains,
N agents, N vessels, zero-catch share, and key moments for every analysis
sample used in the revision.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .config import CFG, DATA_FINAL, VOYAGE_PARQUET, CANONICAL_CONNECTED_SET, TABLES_DIR
from .output_schema import save_result_table, save_markdown_table

logger = logging.getLogger(__name__)


# ── Sample definitions ───────────────────────────────────────────────────

def _load_raw_panel() -> pd.DataFrame:
    """Load the raw voyage panel (pre-filter, preserving zeros)."""
    df = pd.read_parquet(VOYAGE_PARQUET)
    return df


def _compute_sample_stats(df: pd.DataFrame, sample_name: str) -> Dict[str, Any]:
    """Compute descriptive statistics for a sample."""
    n = len(df)
    stats: Dict[str, Any] = {"sample_name": sample_name, "n_voyages": n}

    for col, label in [
        ("captain_id", "n_captains"),
        ("agent_id", "n_agents"),
        ("vessel_id", "n_vessels"),
    ]:
        if col in df.columns:
            stats[label] = int(df[col].nunique())
        else:
            stats[label] = None

    # Output variables
    q_col = "q_total_index" if "q_total_index" in df.columns else "q_oil_bbl"
    if q_col in df.columns:
        q = df[q_col]
        stats["zero_catch_share"] = float((q <= 0).mean())
        stats["near_zero_share"] = float(
            (q <= q.quantile(CFG.near_zero_percentile / 100)).mean()
        ) if n > 10 else None
        stats["mean_q"] = float(q.mean())
        stats["sd_q"] = float(q.std())
        # log output (positive only)
        q_pos = q[q > 0]
        if len(q_pos) > 0:
            log_q = np.log(q_pos)
            stats["mean_log_q"] = float(log_q.mean())
            stats["sd_log_q"] = float(log_q.std())
        else:
            stats["mean_log_q"] = stats["sd_log_q"] = None
    else:
        for k in ["zero_catch_share", "near_zero_share", "mean_q", "sd_q", "mean_log_q", "sd_log_q"]:
            stats[k] = None

    # Tonnage
    if "tonnage" in df.columns:
        stats["mean_tonnage"] = float(df["tonnage"].mean())
        stats["sd_tonnage"] = float(df["tonnage"].std())
    else:
        stats["mean_tonnage"] = stats["sd_tonnage"] = None

    # Crew
    if "crew" in df.columns:
        stats["mean_crew"] = float(df["crew"].mean())
        stats["sd_crew"] = float(df["crew"].std())
    elif "crew_size" in df.columns:
        stats["mean_crew"] = float(df["crew_size"].mean())
        stats["sd_crew"] = float(df["crew_size"].std())
    else:
        stats["mean_crew"] = stats["sd_crew"] = None

    # Decade composition
    if "year_out" in df.columns:
        decade = (df["year_out"] // 10) * 10
        stats["decade_min"] = int(decade.min())
        stats["decade_max"] = int(decade.max())
        stats["decade_mode"] = int(decade.mode().iloc[0]) if len(decade.mode()) > 0 else None
    else:
        stats["decade_min"] = stats["decade_max"] = stats["decade_mode"] = None

    # Movers
    if "captain_id" in df.columns and "agent_id" in df.columns:
        cap_agents = df.groupby("captain_id")["agent_id"].nunique()
        stats["n_movers"] = int((cap_agents >= 2).sum())
        stats["mover_share"] = float((cap_agents >= 2).mean())
    else:
        stats["n_movers"] = stats["mover_share"] = None

    return stats


def build_sample_crosswalk() -> pd.DataFrame:
    """Build the authoritative crosswalk table across all analysis samples."""
    logger.info("=" * 60)
    logger.info("STEP 2: BUILDING SAMPLE CROSSWALK AND AUDIT TABLE")
    logger.info("=" * 60)

    all_stats: List[Dict[str, Any]] = []

    # 1. Raw panel (pre-filter)
    df_raw = _load_raw_panel()
    all_stats.append(_compute_sample_stats(df_raw, "1_raw_panel"))
    logger.info("  Raw panel: %d voyages", len(df_raw))

    # 2. Full analysis sample (after standard filters)
    q_col = "q_total_index" if "q_total_index" in df_raw.columns else "q_oil_bbl"
    df_analysis = df_raw.dropna(subset=["captain_id", "agent_id", "year_out"]).copy()
    if q_col in df_analysis.columns:
        df_analysis = df_analysis[df_analysis[q_col] > 0].copy()
    all_stats.append(_compute_sample_stats(df_analysis, "2_full_analysis"))
    logger.info("  Full analysis: %d voyages", len(df_analysis))

    # 3. Extended analysis (including zeros and near-zeros)
    df_with_zeros = df_raw.dropna(subset=["captain_id", "agent_id", "year_out"]).copy()
    all_stats.append(_compute_sample_stats(df_with_zeros, "3_with_zeros"))
    logger.info("  With zeros: %d voyages", len(df_with_zeros))

    # 4. Connected set (LOO/KSS)
    if CANONICAL_CONNECTED_SET.exists():
        df_conn = pd.read_parquet(CANONICAL_CONNECTED_SET)
    else:
        from src.analyses.connected_set import find_connected_set, find_leave_one_out_connected_set
        df_cc, _ = find_connected_set(df_analysis)
        df_conn, _ = find_leave_one_out_connected_set(df_cc)
    all_stats.append(_compute_sample_stats(df_conn, "4_connected_set_loo"))
    logger.info("  Connected set (LOO): %d voyages", len(df_conn))

    # 5. Search geometry sample (requires Lévy exponent data)
    if "levy_exponent" in df_raw.columns or "mu" in df_raw.columns:
        mu_col = "levy_exponent" if "levy_exponent" in df_raw.columns else "mu"
        df_search = df_analysis[df_analysis[mu_col].notna()].copy()
        all_stats.append(_compute_sample_stats(df_search, "5_search_geometry"))
        logger.info("  Search geometry: %d voyages", len(df_search))
    else:
        logger.info("  Search geometry: column not found in main panel, will use positions data")

    # 6. Officer pipeline (mate-to-captain)
    captain_year_path = DATA_FINAL / "analysis_captain_year.parquet"
    if captain_year_path.exists():
        df_cy = pd.read_parquet(captain_year_path)
        all_stats.append(_compute_sample_stats(df_cy, "6_captain_year_panel"))
        logger.info("  Captain-year panel: %d obs", len(df_cy))

    crosswalk = pd.DataFrame(all_stats)

    # Save
    save_result_table(crosswalk, "table_sample_crosswalk", metadata={
        "description": "Authoritative sample crosswalk for revision 2026",
        "n_samples": len(all_stats),
    })
    save_markdown_table(crosswalk, "table_sample_crosswalk")

    # Audit flags
    flags = _check_audit_flags(crosswalk)
    if flags:
        flags_df = pd.DataFrame(flags)
        save_result_table(flags_df, "table_sample_audit_flags")
        for f in flags:
            logger.warning("  AUDIT FLAG: %s", f["message"])
    else:
        logger.info("  All audit checks passed.")

    logger.info("  Saved crosswalk to %s", TABLES_DIR)
    return crosswalk


def _check_audit_flags(crosswalk: pd.DataFrame) -> List[Dict[str, str]]:
    """Run consistency checks on the crosswalk table."""
    flags = []

    if len(crosswalk) == 0:
        return [{"sample": "all", "level": "ERROR", "message": "Empty crosswalk table"}]

    # Check connected set is subset of full analysis
    full = crosswalk[crosswalk["sample_name"] == "2_full_analysis"]
    conn = crosswalk[crosswalk["sample_name"] == "4_connected_set_loo"]

    if len(full) > 0 and len(conn) > 0:
        n_full = full.iloc[0]["n_voyages"]
        n_conn = conn.iloc[0]["n_voyages"]
        if n_conn > n_full:
            flags.append({
                "sample": "connected_set",
                "level": "ERROR",
                "message": f"Connected set ({n_conn}) larger than full analysis ({n_full})"
            })
        coverage = n_conn / n_full if n_full > 0 else 0
        if coverage < 0.30:
            flags.append({
                "sample": "connected_set",
                "level": "WARNING",
                "message": f"Connected set covers only {coverage:.1%} of full analysis"
            })

    # Check zero-catch share
    with_zeros = crosswalk[crosswalk["sample_name"] == "3_with_zeros"]
    if len(with_zeros) > 0:
        zs = with_zeros.iloc[0].get("zero_catch_share")
        if zs is not None and zs > 0.30:
            flags.append({
                "sample": "with_zeros",
                "level": "WARNING",
                "message": f"Zero-catch share is {zs:.1%} — verify data quality"
            })

    return flags
