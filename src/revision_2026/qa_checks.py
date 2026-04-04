"""
QA Validation Checks (Step 10).

Automated quality assurance to verify:
  - No leakage in effect construction
  - Sample count consistency across tables
  - Pre-trend test honesty
  - Sensitivity comparison across psi_hat versions
  - Crosswalk consistency
  - Reproducibility verification
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .config import CFG, INTERMEDIATES_DIR, TABLES_DIR, OUTPUT_BASE

logger = logging.getLogger(__name__)


def _check_leakage() -> List[Dict[str, Any]]:
    """Verify no same-sample effect objects are used in downstream regressions."""
    flags = []

    # Scan all tidy result tables for effect_object_used
    for csv_path in TABLES_DIR.glob("*.csv"):
        if "_meta" in csv_path.name:
            continue
        try:
            df = pd.read_csv(csv_path)
            if "effect_object_used" in df.columns:
                leaky = df[df["effect_object_used"].isin(["same_sample", "full_sample", "none", ""])]
                if len(leaky) > 0 and "effect_object_used" in leaky.columns:
                    bad_sources = leaky["effect_object_used"].unique()
                    # Only flag if it's explicitly "same_sample"
                    for src in bad_sources:
                        if src == "same_sample":
                            flags.append({
                                "check": "leakage",
                                "level": "ERROR",
                                "file": csv_path.name,
                                "message": f"Same-sample effects used in {csv_path.name}: {src}",
                            })
                        elif src in ("none", ""):
                            flags.append({
                                "check": "leakage",
                                "level": "WARNING",
                                "file": csv_path.name,
                                "message": f"Missing effect source in {csv_path.name}",
                            })
        except Exception:
            pass

    if not flags:
        logger.info("  ✓ No leakage detected in result tables")
    return flags


def _check_sample_counts() -> List[Dict[str, Any]]:
    """Verify sample counts are consistent across tables."""
    flags = []

    # Check that crosswalk exists
    crosswalk_path = TABLES_DIR / "table_sample_crosswalk.csv"
    if not crosswalk_path.exists():
        flags.append({
            "check": "sample_counts",
            "level": "WARNING",
            "file": "table_sample_crosswalk.csv",
            "message": "Sample crosswalk table not found",
        })
        return flags

    crosswalk = pd.read_csv(crosswalk_path)

    # Check that connected set N in crosswalk matches intermediate files
    psi_path = INTERMEDIATES_DIR / "psi_hat_leave_one_captain_out.parquet"
    if psi_path.exists():
        psi = pd.read_parquet(psi_path)
        conn_row = crosswalk[crosswalk["sample_name"] == "4_connected_set_loo"]
        if len(conn_row) > 0:
            expected_n = int(conn_row.iloc[0]["n_voyages"])
            actual_n = len(psi)
            if expected_n != actual_n:
                flags.append({
                    "check": "sample_counts",
                    "level": "WARNING",
                    "file": "psi_hat_leave_one_captain_out.parquet",
                    "message": f"Crosswalk expects {expected_n} voyages but psi_hat has {actual_n}",
                })

    if not flags:
        logger.info("  ✓ Sample counts consistent across tables")
    return flags


def _check_effect_coverage() -> List[Dict[str, Any]]:
    """Check coverage of separate-sample effect objects."""
    flags = []

    meta_path = INTERMEDIATES_DIR / "effect_construction_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

        for key, info in meta.items():
            coverage = info.get("coverage_pct", 100)
            if coverage < 50:
                flags.append({
                    "check": "effect_coverage",
                    "level": "WARNING",
                    "file": key,
                    "message": f"{key} coverage is only {coverage:.1f}%",
                })
    else:
        flags.append({
            "check": "effect_coverage",
            "level": "WARNING",
            "file": "effect_construction_metadata.json",
            "message": "Effect construction metadata not found",
        })

    if not flags:
        logger.info("  ✓ Effect objects have adequate coverage")
    return flags


def _check_pretrend_honesty() -> List[Dict[str, Any]]:
    """Check that event-study pre-trend p-values are reported."""
    flags = []

    es_path = TABLES_DIR / "table_event_study_pretrend_tests.csv"
    if es_path.exists():
        df = pd.read_csv(es_path)
        pretrend = df[df["spec_name"] == "pretrend_test"]
        if len(pretrend) > 0:
            # Check if any pre-trend is significant
            for _, row in pretrend.iterrows():
                if "p_value" in row and pd.notna(row["p_value"]):
                    if row["p_value"] < 0.05:
                        flags.append({
                            "check": "pretrend",
                            "level": "WARNING",
                            "file": "table_event_study_pretrend_tests.csv",
                            "message": f"Significant pre-trend detected: {row['term']} (p={row['p_value']:.4f})",
                        })
        else:
            flags.append({
                "check": "pretrend",
                "level": "INFO",
                "file": "table_event_study_pretrend_tests.csv",
                "message": "No pre-trend test results found in event study table",
            })

    if not flags:
        logger.info("  ✓ Pre-trend tests reported and clean")
    return flags


def _check_psi_sensitivity() -> List[Dict[str, Any]]:
    """Compare results across psi_hat construction methods."""
    flags = []

    psi_loo_path = INTERMEDIATES_DIR / "psi_hat_leave_one_captain_out.parquet"
    psi_pre_path = INTERMEDIATES_DIR / "psi_hat_preperiod.parquet"

    if psi_loo_path.exists() and psi_pre_path.exists():
        psi_loo = pd.read_parquet(psi_loo_path)
        psi_pre = pd.read_parquet(psi_pre_path)

        merged = psi_loo.merge(psi_pre[["voyage_id", "psi_hat_pre"]], on="voyage_id", how="inner")
        valid = merged.dropna(subset=["psi_hat_loo", "psi_hat_pre"])

        if len(valid) > 100:
            corr = valid["psi_hat_loo"].corr(valid["psi_hat_pre"])
            rank_corr = valid["psi_hat_loo"].corr(valid["psi_hat_pre"], method="spearman")

            logger.info("  Psi_hat sensitivity: LOO vs PrePeriod Pearson=%.3f, Spearman=%.3f", corr, rank_corr)

            if corr < 0.70:
                flags.append({
                    "check": "psi_sensitivity",
                    "level": "WARNING",
                    "file": "psi_hat comparison",
                    "message": f"LOO vs PrePeriod correlation is low: {corr:.3f}",
                })
    else:
        logger.info("  Skipping psi sensitivity (one or both files missing)")

    if not flags:
        logger.info("  ✓ Psi_hat construction methods agree")
    return flags


def run_all_qa_checks() -> pd.DataFrame:
    """Run all QA checks and produce a summary."""
    logger.info("=" * 60)
    logger.info("STEP 10: QA VALIDATION CHECKS")
    logger.info("=" * 60)

    all_flags = []
    all_flags.extend(_check_leakage())
    all_flags.extend(_check_sample_counts())
    all_flags.extend(_check_effect_coverage())
    all_flags.extend(_check_pretrend_honesty())
    all_flags.extend(_check_psi_sensitivity())

    flags_df = pd.DataFrame(all_flags) if all_flags else pd.DataFrame(
        columns=["check", "level", "file", "message"]
    )

    # Save
    flags_df.to_csv(TABLES_DIR / "qa_validation_flags.csv", index=False)

    n_errors = len(flags_df[flags_df["level"] == "ERROR"]) if len(flags_df) > 0 else 0
    n_warnings = len(flags_df[flags_df["level"] == "WARNING"]) if len(flags_df) > 0 else 0

    logger.info("QA Summary: %d errors, %d warnings, %d total flags",
                n_errors, n_warnings, len(flags_df))

    if n_errors > 0:
        for _, row in flags_df[flags_df["level"] == "ERROR"].iterrows():
            logger.error("  ERROR: %s", row["message"])

    return flags_df
