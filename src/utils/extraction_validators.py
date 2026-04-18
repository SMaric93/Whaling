"""
Validation gate for VLM-extracted WSL events.

Runs schema, distribution, and biophysical-sanity checks on the event-level
parquet produced by ``src.parsing.wsl_v4_postprocess``. Fatal failures raise
``ExtractionValidationError`` so the pipeline halts before bad data flows into
Stage 2 and onwards. Soft issues are returned as warnings.

Thresholds are driven by ``VALIDATION_CONFIG`` in ``src/config.py`` plus a few
extraction-specific knobs defined here. No external dependencies (pandera,
great_expectations) — keep the research codebase lean.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from src.config import VALIDATION_CONFIG, WSL_EVENT_TYPES

logger = logging.getLogger(__name__)


class ExtractionValidationError(Exception):
    """Raised when extracted events fail a fatal validation check."""


@dataclass
class ExtractionValidationConfig:
    """Thresholds for the VLM extraction validation gate."""

    required_columns: Set[str] = field(default_factory=lambda: {
        "wsl_event_id", "page_id", "pdf", "year", "page", "event_idx",
        "vessel_name", "event_type", "_confidence",
    })

    min_events: int = 10_000
    min_vessels: int = 500
    min_years_covered: int = 30
    max_other_event_share: float = 0.05
    max_vessel_is_port_share: float = 0.05
    min_mean_confidence: float = 0.70
    min_vessel_name_fill: float = 0.95
    min_date_fill: float = 0.60
    min_port_fill: float = 0.50

    known_port_coverage_floor: float = 0.80


VALIDATION_THRESHOLDS = ExtractionValidationConfig()


KNOWN_PORTS_LOWER: Set[str] = {
    "new bedford", "nantucket", "new london", "sag harbor", "fairhaven",
    "provincetown", "edgartown", "mattapoisett", "westport", "dartmouth",
    "stonington", "mystic", "warren", "bristol", "newport", "providence",
    "honolulu", "lahaina", "hilo", "talcahuano", "paita", "tumbes",
    "valparaiso", "callao", "fayal", "cape verde", "st. helena",
    "st. catharines", "hobart town", "sydney", "bay of islands",
    "san francisco", "tahiti", "guam", "pernambuco", "rio de janeiro",
    "boston", "new york", "philadelphia", "baltimore",
}


@dataclass
class ValidationResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        head = f"{'PASS' if self.ok else 'FAIL'} — {len(self.errors)} errors, {len(self.warnings)} warnings"
        lines = [head]
        for err in self.errors:
            lines.append(f"  ✗ {err}")
        for warn in self.warnings:
            lines.append(f"  ! {warn}")
        return "\n".join(lines)


def _check_schema(events: pd.DataFrame, cfg: ExtractionValidationConfig,
                  result: ValidationResult) -> None:
    missing = cfg.required_columns - set(events.columns)
    if missing:
        result.errors.append(f"missing required columns: {sorted(missing)}")


def _check_volume(events: pd.DataFrame, cfg: ExtractionValidationConfig,
                  result: ValidationResult) -> None:
    n = len(events)
    result.metrics["n_events"] = n
    if n < cfg.min_events:
        result.errors.append(f"event count {n:,} below floor {cfg.min_events:,}")

    if "vessel_name" in events.columns:
        n_vessels = events["vessel_name"].dropna().nunique()
        result.metrics["n_unique_vessels"] = n_vessels
        if n_vessels < cfg.min_vessels:
            result.errors.append(
                f"unique vessel count {n_vessels:,} below floor {cfg.min_vessels:,}"
            )

    if "year" in events.columns:
        n_years = events["year"].dropna().nunique()
        result.metrics["n_years"] = n_years
        if n_years < cfg.min_years_covered:
            result.warnings.append(
                f"only {n_years} years covered (expected ≥ {cfg.min_years_covered})"
            )


def _check_event_types(events: pd.DataFrame, cfg: ExtractionValidationConfig,
                       result: ValidationResult) -> None:
    if "event_type" not in events.columns or events.empty:
        return

    vc = events["event_type"].value_counts(dropna=False)
    share = vc / len(events)

    other = float(share.get("OTHER", 0.0))
    result.metrics["other_event_share"] = round(other, 4)
    if other > cfg.max_other_event_share:
        result.errors.append(
            f"OTHER event share {other:.2%} exceeds ceiling {cfg.max_other_event_share:.2%}"
        )

    unknown = set(vc.index) - set(WSL_EVENT_TYPES) - {None}
    if unknown:
        result.warnings.append(f"event_type values outside canonical set: {sorted(unknown)}")


def _check_cargo_ranges(events: pd.DataFrame, result: ValidationResult) -> None:
    for col, ceiling in (
        ("oil_sperm_bbls", VALIDATION_CONFIG.max_oil_barrels),
        ("oil_whale_bbls", VALIDATION_CONFIG.max_oil_barrels),
        ("bone_lbs", VALIDATION_CONFIG.max_bone_lbs),
    ):
        if col not in events.columns:
            continue
        series = pd.to_numeric(events[col], errors="coerce")
        if series.dropna().empty:
            continue
        n_over = int((series > ceiling).sum())
        n_neg = int((series < 0).sum())
        result.metrics[f"{col}_over_ceiling"] = n_over
        if n_over > 0:
            result.errors.append(
                f"{col}: {n_over} rows exceed biophysical ceiling {ceiling}"
            )
        if n_neg > 0:
            result.errors.append(f"{col}: {n_neg} rows are negative")


def _check_fill_rates(events: pd.DataFrame, cfg: ExtractionValidationConfig,
                      result: ValidationResult) -> None:
    if events.empty:
        return
    total = len(events)
    for col, floor, level in (
        ("vessel_name", cfg.min_vessel_name_fill, "error"),
        ("date", cfg.min_date_fill, "warning"),
        ("port", cfg.min_port_fill, "warning"),
    ):
        if col not in events.columns:
            continue
        filled = events[col].notna().sum()
        if pd.api.types.is_string_dtype(events[col]):
            filled = events[col].astype(str).str.strip().replace({"": None}).notna().sum()
        rate = float(filled) / total
        result.metrics[f"{col}_fill_rate"] = round(rate, 4)
        if rate < floor:
            msg = f"{col} fill rate {rate:.2%} below floor {floor:.2%}"
            (result.errors if level == "error" else result.warnings).append(msg)


def _check_confidence(events: pd.DataFrame, cfg: ExtractionValidationConfig,
                      result: ValidationResult) -> None:
    if "_confidence" not in events.columns:
        return
    conf = pd.to_numeric(events["_confidence"], errors="coerce").dropna()
    if conf.empty:
        result.warnings.append("_confidence column is entirely null")
        return
    mean_conf = float(conf.mean())
    result.metrics["mean_confidence"] = round(mean_conf, 4)
    if mean_conf < cfg.min_mean_confidence:
        result.errors.append(
            f"mean confidence {mean_conf:.3f} below floor {cfg.min_mean_confidence:.3f}"
        )


def _check_port_whitelist(events: pd.DataFrame, cfg: ExtractionValidationConfig,
                          result: ValidationResult) -> None:
    if "port" not in events.columns:
        return
    ports = events["port"].dropna().astype(str).str.strip().str.lower()
    ports = ports[ports != ""]
    if ports.empty:
        return
    in_whitelist = ports.isin(KNOWN_PORTS_LOWER).sum()
    rate = float(in_whitelist) / len(ports)
    result.metrics["known_port_coverage"] = round(rate, 4)
    if rate < cfg.known_port_coverage_floor:
        result.warnings.append(
            f"known-port coverage {rate:.2%} below floor {cfg.known_port_coverage_floor:.2%} "
            f"— review post-processor port corrections"
        )


def _check_contamination_flags(events: pd.DataFrame, cfg: ExtractionValidationConfig,
                               result: ValidationResult) -> None:
    if "_vessel_is_port_name" not in events.columns:
        return
    share = float(events["_vessel_is_port_name"].fillna(False).mean())
    result.metrics["vessel_is_port_share"] = round(share, 4)
    if share > cfg.max_vessel_is_port_share:
        result.warnings.append(
            f"vessel_is_port_name share {share:.2%} exceeds soft ceiling {cfg.max_vessel_is_port_share:.2%}"
        )


def validate_wsl_events(
    events: pd.DataFrame,
    config: Optional[ExtractionValidationConfig] = None,
    strict: bool = True,
) -> ValidationResult:
    """
    Validate a VLM-extracted events DataFrame.

    Args:
        events: The event-level DataFrame (one row per event mention).
        config: Threshold overrides. Defaults to ``VALIDATION_THRESHOLDS``.
        strict: If True, raise ``ExtractionValidationError`` on any fatal error.

    Returns:
        A ``ValidationResult`` with errors, warnings, and aggregate metrics.
    """
    cfg = config or VALIDATION_THRESHOLDS
    result = ValidationResult(ok=True)

    if events is None or events.empty:
        result.errors.append("events DataFrame is empty")
        result.ok = False
        if strict:
            raise ExtractionValidationError(result.summary())
        return result

    _check_schema(events, cfg, result)
    _check_volume(events, cfg, result)
    _check_event_types(events, cfg, result)
    _check_cargo_ranges(events, result)
    _check_fill_rates(events, cfg, result)
    _check_confidence(events, cfg, result)
    _check_port_whitelist(events, cfg, result)
    _check_contamination_flags(events, cfg, result)

    result.ok = not result.errors

    logger.info("Extraction validation: %s", result.summary())

    if strict and not result.ok:
        raise ExtractionValidationError(result.summary())

    return result


def validate_parquet(path, strict: bool = True) -> Tuple[ValidationResult, pd.DataFrame]:
    """Convenience: load the parquet and validate. Returns (result, df)."""
    df = pd.read_parquet(path)
    return validate_wsl_events(df, strict=strict), df
