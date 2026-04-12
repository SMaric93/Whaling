"""
Data validation functions for analysis files.

Implements plausibility and consistency checks.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import re
from config import VALIDATION_CONFIG
from ml.anomaly_detection import compute_anomaly_scores

def validate_schema_against_dictionary(df: pd.DataFrame, file_name: str) -> List[ValidationResult]:
    """Validates that dataframe columns match those documented in data_dictionary.md."""
    dict_path = Path(__file__).parent.parent.parent / "docs" / "data_dictionary.md"
    if not dict_path.exists():
        return []
    
    content = dict_path.read_text()
    section_pattern = re.compile(rf"## {file_name}(.*?)(?:## |$)", re.DOTALL)
    match = section_pattern.search(content)
    if not match:
        return []
    
    section_content = match.group(1)
    var_pattern = re.compile(r"\|\s*`([a-zA-Z0-9_]+)`\s*\|")
    expected_vars = set(var_pattern.findall(section_content))
    
    actual_vars = set(df.columns)
    missing_in_df = expected_vars - actual_vars
    missing_in_dict = actual_vars - expected_vars
    
    results = []
    if missing_in_df:
        results.append(ValidationResult(
            check_name="schema_missing_in_df",
            passed=False,
            message=f"Missing {len(missing_in_df)} documented columns: {list(missing_in_df)[:5]}",
            affected_count=len(missing_in_df),
            details={"missing": list(missing_in_df)}
        ))
    if missing_in_dict:
        results.append(ValidationResult(
            check_name="schema_missing_in_dictionary",
            passed=True,  # Warning only
            message=f"Found {len(missing_in_dict)} undocumented columns: {list(missing_in_dict)[:5]}...",
            affected_count=len(missing_in_dict),
            details={"undocumented": list(missing_in_dict)}
        ))
    if not missing_in_df and not missing_in_dict:
        results.append(ValidationResult(
            check_name="schema_match",
            passed=True,
            message="Dataframe schema perfectly matches data_dictionary.md"
        ))
    return results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""
    check_name: str
    passed: bool
    message: str
    affected_count: int = 0
    affected_fraction: float = 0.0
    details: Dict[str, Any] = None


def _append_anomaly_check(
    results: List[ValidationResult],
    df: pd.DataFrame,
    *,
    check_name: str,
    numeric_cols: List[str],
) -> None:
    """Append an IsolationForest anomaly summary when enough numeric data exists."""
    anomaly_df = compute_anomaly_scores(df, numeric_cols)
    if len(anomaly_df) == 0 or "anomaly_flag" not in anomaly_df.columns:
        return

    flags = anomaly_df["anomaly_flag"].fillna(False)
    if len(flags) == 0 or flags.isna().all():
        return

    flagged = int(flags.sum())
    results.append(ValidationResult(
        check_name=check_name,
        passed=(flagged == 0),
        message=f"{flagged} rows flagged by multivariate anomaly detection",
        affected_count=flagged,
        affected_fraction=flagged / len(df) if len(df) > 0 else 0,
        details={
            "numeric_cols": [col for col in numeric_cols if col in df.columns],
            "model": "IsolationForest",
        },
    ))


def validate_analysis_voyage(df: pd.DataFrame) -> List[ValidationResult]:
    """
    Validate the analysis_voyage table.
    
    Checks:
    - voyage_id uniqueness
    - date consistency (date_in >= date_out)
    - quantity ranges
    - rate bounds (desertion_rate in [0,1])
    
    Returns:
        List of ValidationResult objects
    """
    results = []
    
    # Check voyage_id uniqueness
    if "voyage_id" in df.columns:
        duplicates = df["voyage_id"].duplicated().sum()
        results.append(ValidationResult(
            check_name="voyage_id_uniqueness",
            passed=(duplicates == 0),
            message=f"{duplicates} duplicate voyage_ids found",
            affected_count=duplicates,
            affected_fraction=duplicates / len(df) if len(df) > 0 else 0,
        ))
    
    # Check date consistency
    if "date_out" in df.columns and "date_in" in df.columns:
        valid_dates = df[df["date_out"].notna() & df["date_in"].notna()]
        if len(valid_dates) > 0:
            inconsistent = (valid_dates["date_in"] < valid_dates["date_out"]).sum()
            results.append(ValidationResult(
                check_name="date_consistency",
                passed=(inconsistent == 0),
                message=f"{inconsistent} voyages with date_in < date_out",
                affected_count=inconsistent,
                affected_fraction=inconsistent / len(valid_dates),
            ))
    
    # Check oil quantity range
    if "q_oil_bbl" in df.columns:
        q = df["q_oil_bbl"]
        negative = (q < 0).sum()
        extreme = (q > VALIDATION_CONFIG.max_oil_barrels).sum()
        results.append(ValidationResult(
            check_name="oil_quantity_range",
            passed=(negative == 0 and extreme == 0),
            message=f"{negative} negative, {extreme} extreme (>{VALIDATION_CONFIG.max_oil_barrels})",
            affected_count=negative + extreme,
            affected_fraction=(negative + extreme) / q.notna().sum() if q.notna().sum() > 0 else 0,
        ))
    
    # Check bone quantity range
    if "q_bone_lbs" in df.columns:
        q = df["q_bone_lbs"]
        negative = (q < 0).sum()
        extreme = (q > VALIDATION_CONFIG.max_bone_lbs).sum()
        results.append(ValidationResult(
            check_name="bone_quantity_range",
            passed=(negative == 0 and extreme == 0),
            message=f"{negative} negative, {extreme} extreme (>{VALIDATION_CONFIG.max_bone_lbs})",
            affected_count=negative + extreme,
            affected_fraction=(negative + extreme) / q.notna().sum() if q.notna().sum() > 0 else 0,
        ))
    
    # Check desertion rate bounds
    if "desertion_rate" in df.columns:
        r = df["desertion_rate"]
        out_of_bounds = ((r < 0) | (r > 1)).sum()
        results.append(ValidationResult(
            check_name="desertion_rate_bounds",
            passed=(out_of_bounds == 0),
            message=f"{out_of_bounds} values outside [0,1]",
            affected_count=out_of_bounds,
            affected_fraction=out_of_bounds / r.notna().sum() if r.notna().sum() > 0 else 0,
        ))
    
    # Check duration range
    if "duration_days" in df.columns:
        d = df["duration_days"]
        too_short = (d < VALIDATION_CONFIG.min_voyage_days).sum()
        too_long = (d > VALIDATION_CONFIG.max_voyage_days).sum()
        results.append(ValidationResult(
            check_name="duration_range",
            passed=(too_short + too_long == 0),
            message=f"{too_short} too short (<{VALIDATION_CONFIG.min_voyage_days}d), {too_long} too long (>{VALIDATION_CONFIG.max_voyage_days}d)",
            affected_count=too_short + too_long,
            affected_fraction=(too_short + too_long) / d.notna().sum() if d.notna().sum() > 0 else 0,
        ))
    
    # Check year range plausibility
    if "year_out" in df.columns:
        y = df["year_out"]
        out_of_era = ((y < 1700) | (y > 1930)).sum()
        results.append(ValidationResult(
            check_name="year_range",
            passed=(out_of_era == 0),
            message=f"{out_of_era} years outside whaling era (1700-1930)",
            affected_count=out_of_era,
            affected_fraction=out_of_era / y.notna().sum() if y.notna().sum() > 0 else 0,
        ))

    _append_anomaly_check(
        results,
        df,
        check_name="voyage_multivariate_anomalies",
        numeric_cols=[
            "q_oil_bbl",
            "q_bone_lbs",
            "duration_days",
            "desertion_rate",
            "year_out",
        ],
    )
    
    return results


def validate_analysis_captain_year(df: pd.DataFrame) -> List[ValidationResult]:
    """
    Validate the analysis_captain_year table.
    
    Checks:
    - (captain_id, census_year) uniqueness
    - Voyage count positivity
    - Wealth non-negativity
    - Age plausibility
    
    Returns:
        List of ValidationResult objects
    """
    results = []
    
    # Check key uniqueness
    if "captain_id" in df.columns and "census_year" in df.columns:
        duplicates = df.duplicated(subset=["captain_id", "census_year"]).sum()
        results.append(ValidationResult(
            check_name="captain_year_uniqueness",
            passed=(duplicates == 0),
            message=f"{duplicates} duplicate (captain_id, census_year) pairs",
            affected_count=duplicates,
            affected_fraction=duplicates / len(df) if len(df) > 0 else 0,
        ))
    
    # Check voyage count positivity
    if "whaling_voyages_count" in df.columns:
        v = df["whaling_voyages_count"]
        non_positive = (v <= 0).sum()
        results.append(ValidationResult(
            check_name="voyage_count_positive",
            passed=(non_positive == 0),
            message=f"{non_positive} captain-years with non-positive voyage count",
            affected_count=non_positive,
            affected_fraction=non_positive / len(df) if len(df) > 0 else 0,
        ))
    
    # Check wealth non-negativity
    for col in ["REALPROP", "PERSPROP", "total_wealth"]:
        if col in df.columns:
            w = df[col]
            negative = (w < 0).sum()
            results.append(ValidationResult(
                check_name=f"{col.lower()}_non_negative",
                passed=(negative == 0),
                message=f"{negative} negative {col} values",
                affected_count=negative,
                affected_fraction=negative / w.notna().sum() if w.notna().sum() > 0 else 0,
            ))
    
    # Check age plausibility
    if "AGE" in df.columns:
        a = df["AGE"]
        implausible = ((a < 18) | (a > 100)).sum()
        results.append(ValidationResult(
            check_name="age_plausibility",
            passed=(implausible == 0),
            message=f"{implausible} ages outside 18-100 range",
            affected_count=implausible,
            affected_fraction=implausible / a.notna().sum() if a.notna().sum() > 0 else 0,
        ))

    _append_anomaly_check(
        results,
        df,
        check_name="captain_year_multivariate_anomalies",
        numeric_cols=[
            "AGE",
            "REALPROP",
            "PERSPROP",
            "total_wealth",
            "whaling_voyages_count",
        ],
    )
    
    results.extend(validate_schema_against_dictionary(df, "analysis_captain_year.parquet"))
    
    return results


def run_all_validations(
    voyage_path: Optional[Path] = None,
    captain_path: Optional[Path] = None,
) -> Dict[str, List[ValidationResult]]:
    """
    Run all validations on final analysis files.
    
    Returns:
        Dict mapping file type to list of validation results
    """
    from config import FINAL_DIR
    
    results = {}
    
    # Validate voyage file
    voyage_path = voyage_path or (FINAL_DIR / "analysis_voyage.parquet")
    if voyage_path.exists():
        df = pd.read_parquet(voyage_path)
        results["analysis_voyage"] = validate_analysis_voyage(df)
    
    # Validate captain file
    captain_path = captain_path or (FINAL_DIR / "analysis_captain_year.parquet")
    if captain_path.exists():
        df = pd.read_parquet(captain_path)
        results["analysis_captain_year"] = validate_analysis_captain_year(df)
    
    return results


if __name__ == "__main__":
    results = run_all_validations()
    
    for file_type, checks in results.items():
        print(f"\n=== {file_type} ===")
        for check in checks:
            status = "✓" if check.passed else "✗"
            print(f"  {status} {check.check_name}: {check.message}")
