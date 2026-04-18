from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import resource
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.download.manifest import compute_file_hash
from src.entities.wsl_voyage_matcher import (
    compute_date_score,
    compute_name_similarity,
    compute_port_score,
)
from src.ml.build_outcome_ml_dataset import build_outcome_ml_dataset
from src.next_round.repairs.destination_ontology import BASIN_RULES
from src.parsing.string_normalizer import (
    normalize_name,
    normalize_port_name,
    normalize_vessel_name,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Performance tracing
# ---------------------------------------------------------------------------


def _current_memory_mb() -> float:
    """Return current process RSS in megabytes (best-effort, POSIX)."""
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # macOS reports maxrss in bytes; Linux in kilobytes.
        if hasattr(os, "uname") and os.uname().sysname == "Darwin":
            return usage.ru_maxrss / (1024 * 1024)
        return usage.ru_maxrss / 1024
    except Exception:
        return float("nan")


@dataclass
class _Span:
    """A single timed span in the performance trace."""

    name: str
    started_at: float
    ended_at: float | None = None
    memory_mb_start: float = 0.0
    memory_mb_end: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    children: list[_Span] = field(default_factory=list)

    @property
    def elapsed_seconds(self) -> float:
        if self.ended_at is None:
            return time.monotonic() - self.started_at
        return self.ended_at - self.started_at

    @property
    def memory_delta_mb(self) -> float | None:
        if self.memory_mb_end is None:
            return None
        return self.memory_mb_end - self.memory_mb_start

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "memory_mb_start": round(self.memory_mb_start, 1),
        }
        if self.memory_mb_end is not None:
            result["memory_mb_end"] = round(self.memory_mb_end, 1)
            result["memory_delta_mb"] = round(self.memory_delta_mb or 0, 1)
        if self.metadata:
            result["metadata"] = self.metadata
        if self.children:
            result["children"] = [child.to_dict() for child in self.children]
        return result


class PerfTracer:
    """Lightweight wall-clock + memory tracer for the WSL reliability ML pipeline.

    Usage::

        tracer = PerfTracer()
        with tracer.span("load_events"):
            events = load_events()
            tracer.set_metadata(n_events=len(events))

        with tracer.span("hmm_inference"):
            for i, ep in enumerate(episodes):
                tracer.tick("hmm_inference", i, len(episodes), every=500)
                infer(ep)

        tracer.export_json(output_path)
    """

    def __init__(self) -> None:
        self._root_spans: list[_Span] = []
        self._stack: list[_Span] = []
        self._wall_start = time.monotonic()

    @contextmanager
    def span(self, name: str, **initial_metadata: Any) -> Generator[None, None, None]:
        """Context manager that times a named block and records memory."""
        span = _Span(
            name=name,
            started_at=time.monotonic(),
            memory_mb_start=_current_memory_mb(),
            metadata=dict(initial_metadata) if initial_metadata else {},
        )
        if self._stack:
            self._stack[-1].children.append(span)
        else:
            self._root_spans.append(span)
        self._stack.append(span)
        logger.info("[perf] ▶ %s", name)
        try:
            yield
        finally:
            span.ended_at = time.monotonic()
            span.memory_mb_end = _current_memory_mb()
            self._stack.pop()
            logger.info(
                "[perf] ◀ %s  %.1fs  (%.0f MB → %.0f MB)",
                name,
                span.elapsed_seconds,
                span.memory_mb_start,
                span.memory_mb_end,
            )

    def set_metadata(self, **kwargs: Any) -> None:
        """Attach key-value metadata to the currently open span."""
        if self._stack:
            self._stack[-1].metadata.update(kwargs)

    def tick(
        self,
        span_name: str,
        current: int,
        total: int,
        *,
        every: int = 500,
    ) -> None:
        """Log progress for a long-running loop inside the current span.

        Logs every *every* iterations and on the final iteration.
        """
        if current == 0 or (current + 1) % every == 0 or current + 1 == total:
            elapsed = time.monotonic() - (self._stack[-1].started_at if self._stack else self._wall_start)
            rate = (current + 1) / max(elapsed, 1e-6)
            remaining = (total - current - 1) / max(rate, 1e-6)
            logger.info(
                "[perf]   %s  %d/%d  (%.1f/s, ~%.0fs remaining)",
                span_name,
                current + 1,
                total,
                rate,
                remaining,
            )

    @property
    def total_elapsed_seconds(self) -> float:
        return time.monotonic() - self._wall_start

    def summary_table(self) -> list[dict[str, Any]]:
        """Return a flat list of top-level span timings for quick inspection."""
        rows: list[dict[str, Any]] = []
        for span in self._root_spans:
            rows.append(
                {
                    "stage": span.name,
                    "elapsed_seconds": round(span.elapsed_seconds, 3),
                    "memory_delta_mb": round(span.memory_delta_mb or 0, 1),
                    **{k: v for k, v in span.metadata.items() if not isinstance(v, (dict, list))},
                }
            )
        rows.append(
            {
                "stage": "__total__",
                "elapsed_seconds": round(self.total_elapsed_seconds, 3),
                "memory_delta_mb": round(_current_memory_mb() - (self._root_spans[0].memory_mb_start if self._root_spans else 0), 1),
            }
        )
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_elapsed_seconds": round(self.total_elapsed_seconds, 3),
            "peak_memory_mb": round(_current_memory_mb(), 1),
            "spans": [span.to_dict() for span in self._root_spans],
            "summary": self.summary_table(),
        }

    def export_json(self, path: Path) -> None:
        """Write the full trace to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=False, default=_json_default),
            encoding="utf-8",
        )
        logger.info("[perf] Trace exported to %s", path)

EVENT_TYPES = ["dep", "arr", "spk", "rpt", "inp", "wrk"]
PAGE_TYPES = [
    "weekly_event_flow",
    "fleet_registry_stock",
    "non_table_or_other",
]

PRIMARY_CLASSES = [
    "routine_info",
    "positive_productivity",
    "weak_or_empty_productivity",
    "distress_hazard",
    "interruption_repair",
    "homebound_or_termination",
    "terminal_loss",
    "assistance_transfer_coordination",
    "commercial_admin_status",
    "extraction_noise_or_uncertain",
]

SECONDARY_TAGS = [
    "coordinates_only",
    "multi_port_list",
    "spoken_or_seen",
    "received_report",
    "whales_sighted",
    "good_catch",
    "full_or_nearly_full",
    "small_catch",
    "no_whales_or_clean",
    "leaking_or_damaged",
    "dismasted_or_disabled",
    "ashore_or_aground",
    "storm_or_weather",
    "ice_or_frozen",
    "fire_or_collision",
    "sick_or_injured",
    "death_or_mortality",
    "desertion_or_crew_shortage",
    "mutiny_or_discipline",
    "short_provisions_or_water",
    "in_port",
    "repairing",
    "wintering_or_delayed",
    "bound_home",
    "ordered_home_or_recalled",
    "wrecked_or_condemned",
    "abandoned_or_lost",
    "assisted_or_towed",
    "transferred_oil_or_bone",
    "transferred_crew_or_boats",
    "received_orders_or_intelligence",
    "sold_or_withdrawn",
    "non_whaling_trade",
    "legal_or_customs",
    "ocr_or_layout_noise",
]

STATE_SPACE = [
    "outbound_initial_transit",
    "active_search_neutral",
    "productive_search",
    "low_yield_or_stalled_search",
    "distress_at_sea",
    "in_port_interruption_or_repair",
    "homebound_or_terminated",
    "completed_arrival",
    "terminal_loss",
]

BAD_STATES = {
    "distress_at_sea",
    "in_port_interruption_or_repair",
    "terminal_loss",
}

DEFAULT_ALLOWED_TRANSITIONS = {
    "outbound_initial_transit": {
        "outbound_initial_transit",
        "active_search_neutral",
        "productive_search",
        "low_yield_or_stalled_search",
        "in_port_interruption_or_repair",
        "terminal_loss",
    },
    "active_search_neutral": {
        "active_search_neutral",
        "productive_search",
        "low_yield_or_stalled_search",
        "distress_at_sea",
        "homebound_or_terminated",
    },
    "productive_search": {
        "productive_search",
        "active_search_neutral",
        "low_yield_or_stalled_search",
        "distress_at_sea",
        "homebound_or_terminated",
    },
    "low_yield_or_stalled_search": {
        "low_yield_or_stalled_search",
        "active_search_neutral",
        "productive_search",
        "distress_at_sea",
        "homebound_or_terminated",
    },
    "distress_at_sea": {
        "distress_at_sea",
        "in_port_interruption_or_repair",
        "homebound_or_terminated",
        "terminal_loss",
        "active_search_neutral",
    },
    "in_port_interruption_or_repair": {
        "in_port_interruption_or_repair",
        "active_search_neutral",
        "homebound_or_terminated",
        "terminal_loss",
    },
    "homebound_or_terminated": {
        "homebound_or_terminated",
        "completed_arrival",
        "terminal_loss",
    },
    "completed_arrival": {"completed_arrival"},
    "terminal_loss": {"terminal_loss"},
}

DEFAULT_SOURCE_WEIGHTS = {
    "weekly_event_flow": 1.0,
    "fleet_registry_stock": 0.35,
    "non_table_or_other": 0.0,
}

DEFAULT_FLAG_PENALTIES = {
    "page_type_mismatch": 0.5,
    "field_contamination": 0.7,
    "entity_link_uncertain": 0.7,
    "date_parse_uncertain": 0.8,
    "ocr_noise": 0.6,
}

REMARK_TEXT_HINT = re.compile(
    r"\b("
    r"bound home|ordered home|repair|repairing|leak|leaking|"
    r"dismast|disabled|ashore|aground|wreck|condemn|lost|abandon|"
    r"clean\b|no whales|small catch|full cargo|full\b|good catch|"
    r"tow|towed|assist|transfer|orders|intelligence|winter|"
    r"ice|storm|fire|collision|sick|injured|died|dead|desert|mutiny|"
    r"short water|short provisions|spoke|spoken|seen|reported|"
    r"lat\b|lon\b|\d+\s*(?:sp|wh|bn|bbl)"
    r")\b",
    re.IGNORECASE,
)
COORDINATE_HINT = re.compile(
    r"(?:\blat\b|\blon\b|\b\d+\s*[ns]\b|\b\d+\s*[ew]\b|\d+\s*°)",
    re.IGNORECASE,
)
MULTI_PORT_HINT = re.compile(
    r"\b(?:at|off|from|to)\s+[a-z][a-z .'-]+(?:,\s*[a-z][a-z .'-]+){1,}",
    re.IGNORECASE,
)


@dataclass
class WSLReliabilityConfig:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[3])
    cleaned_events_path: Path | None = None
    issue_index_path: Path | None = None
    voyage_panel_path: Path | None = None
    destination_ontology_path: Path | None = None
    output_root: Path | None = None
    random_seed: int = 42
    gold_sample_size_target: int = 6000
    remarks_max_train_rows: int = 40000
    remarks_min_tag_support: int = 50
    linkage_threshold: float = 0.72
    linkage_low_conf_threshold: float = 0.55
    page_registry_share_threshold: float = 0.90
    page_panel_share_threshold: float = 0.25
    policy_propensity_clip_low: float = 0.05
    policy_propensity_clip_high: float = 0.95
    policy_bootstrap_reps: int = 100
    policy_budget_grid: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.40)
    information_windows: tuple[int, ...] = (30, 90, 180)
    utility_alpha: float = 1.0
    utility_beta: float = 2.0
    utility_gamma: float = 0.25
    source_weight_defaults: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_SOURCE_WEIGHTS)
    )
    flag_penalties: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_FLAG_PENALTIES)
    )
    allowed_transitions: Mapping[str, set[str]] = field(
        default_factory=lambda: {
            key: set(value) for key, value in DEFAULT_ALLOWED_TRANSITIONS.items()
        }
    )

    def __post_init__(self) -> None:
        if self.cleaned_events_path is None:
            self.cleaned_events_path = self.project_root / "data" / "wsl" / "cleaned" / "wsl_events_all_cleaned.jsonl"
        if self.issue_index_path is None:
            self.issue_index_path = self.project_root / "data" / "staging" / "wsl_issue_index.parquet"
        if self.voyage_panel_path is None:
            self.voyage_panel_path = self.project_root / "data" / "final" / "analysis_voyage_augmented.parquet"
        if self.destination_ontology_path is None:
            self.destination_ontology_path = self.project_root / "data" / "derived" / "destination_ontology.parquet"
        if self.output_root is None:
            self.output_root = self.project_root / "outputs" / "wsl_reliability_ml"


def load_default_config() -> WSLReliabilityConfig:
    return WSLReliabilityConfig()


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def compute_config_hash(config: WSLReliabilityConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, default=_json_default)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def stable_hash(parts: Sequence[Any], *, prefix: str = "") -> str:
    joined = "||".join("" if part is None else str(part) for part in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}{digest}" if prefix else digest


def ensure_output_dirs(config: WSLReliabilityConfig) -> dict[str, Path]:
    paths = {
        "root": config.output_root,
        "remarks": config.output_root / "remarks",
        "states": config.output_root / "states",
        "information": config.output_root / "information",
        "policy": config.output_root / "policy",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_json(path: Path, payload: Mapping[str, Any] | Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def sanitize_missing_strings(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    cleaned = frame.copy()
    sentinels = {"", "NAN", "NONE", "NULL", "N/A", "NA"}
    for column in columns:
        if column not in cleaned.columns:
            continue
        cleaned[column] = cleaned[column].map(
            lambda value: pd.NA
            if value is None or (isinstance(value, str) and value.strip().upper() in sentinels)
            else value
        )
    return cleaned


def parse_issue_date_from_issue_id(issue_id: str | None) -> pd.Timestamp:
    if issue_id is None:
        return pd.NaT
    match = re.search(r"wsl_(\d{4})_(\d{2})_(\d{2})", str(issue_id))
    if not match:
        return pd.NaT
    year, month, day = match.groups()
    return pd.Timestamp(year=int(year), month=int(month), day=int(day))


def load_issue_index(config: WSLReliabilityConfig) -> pd.DataFrame:
    if not config.issue_index_path.exists():
        return pd.DataFrame(columns=["wsl_issue_id", "issue_date"])
    index_df = pd.read_parquet(config.issue_index_path)
    if "issue_date" not in index_df.columns:
        index_df["issue_date"] = pd.to_datetime(
            dict(year=index_df["year"], month=index_df["month"], day=index_df["day"]),
            errors="coerce",
        )
    return index_df[["wsl_issue_id", "issue_date"]].drop_duplicates("wsl_issue_id")


def parse_partial_event_date(raw_value: Any, issue_date: Any) -> pd.Timestamp:
    if raw_value is None or (isinstance(raw_value, float) and np.isnan(raw_value)):
        return pd.NaT
    text = str(raw_value).strip()
    if not text:
        return pd.NaT

    issue_date = pd.to_datetime(issue_date, errors="coerce")
    issue_year = int(issue_date.year) if pd.notna(issue_date) else None
    cleaned = (
        text.replace("Sept.", "Sep ")
        .replace("Sept", "Sep")
        .replace("Mch", "Mar")
        .replace("Inst.", "")
        .replace("inst.", "")
        .replace("Ult.", "")
        .replace("ult.", "")
        .replace(".", " ")
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;")
    if not cleaned:
        return pd.NaT
    if re.search(r"(today|yesterday|monday|tuesday|wednesday|thursday|friday|saturday|sunday)", cleaned, re.IGNORECASE):
        return pd.NaT
    match = re.match(
        r"(?P<month>[A-Za-z]+)\s+(?P<day>\d{1,2})(?:,?\s*(?P<year>\d{2,4}))?$",
        cleaned,
    )
    if match:
        month = match.group("month")
        day = int(match.group("day"))
        year_text = match.group("year")
        if year_text is None:
            if issue_year is None:
                return pd.NaT
            year = issue_year
        else:
            year = int(year_text)
            if year < 100:
                if issue_year is None:
                    year += 1800
                else:
                    century = (issue_year // 100) * 100
                    year = century + year
                    if year - issue_year > 3:
                        year -= 100
        try:
            return pd.to_datetime(f"{month} {day} {year}", errors="coerce")
        except Exception:
            return pd.NaT
    return pd.to_datetime(cleaned, errors="coerce")


def classify_page_process(page_record: Mapping[str, Any], config: WSLReliabilityConfig) -> tuple[str, float, str, float, float]:
    raw_page_type = str(page_record.get("page_type") or "")
    events = list(page_record.get("events") or [])
    if raw_page_type == "skip" or not events:
        return ("non_table_or_other", 0.99, "skip_or_empty_page", 0.0, 0.0)
    n_events = max(len(events), 1)
    registry_hits = sum(
        1 for event in events if "likely_registry_not_weekly" in (event.get("_flags") or [])
    )
    panel_valid = sum(1 for event in events if event.get("panel_include") is True)
    registry_share = registry_hits / n_events
    panel_share = panel_valid / n_events

    # V4 detection: if no event has panel_include set, panel_share is meaningless;
    # classify based on registry_share threshold and extraction router page_type only.
    has_panel_include = any(event.get("panel_include") is not None for event in events)

    if registry_share >= config.page_registry_share_threshold:
        confidence = min(0.65 + 0.35 * registry_share, 0.99)
        reason = f"registry_flag_share={registry_share:.2f}"
        return ("fleet_registry_stock", confidence, reason, registry_share, panel_share)

    # Panel-share checks only apply when panel_include is populated (V3 data)
    if has_panel_include:
        if raw_page_type == "sparse" and panel_share < config.page_panel_share_threshold:
            confidence = 0.72
            reason = f"sparse_low_panel_share={panel_share:.2f}"
            return ("fleet_registry_stock", confidence, reason, registry_share, panel_share)
        if panel_share <= 0.05 and raw_page_type in {"shipping_table", "mixed"}:
            return (
                "fleet_registry_stock",
                0.68,
                f"near_zero_panel_share={panel_share:.2f}",
                registry_share,
                panel_share,
            )

    confidence = 0.80
    if raw_page_type == "shipping_table":
        confidence = 0.90
    elif raw_page_type == "mixed":
        confidence = 0.82
    elif raw_page_type == "sparse":
        confidence = 0.75
    reason = f"event_flow_candidate::{raw_page_type}"
    return ("weekly_event_flow", confidence, reason, registry_share, panel_share)


def map_penalty_reasons(row: Mapping[str, Any]) -> list[str]:
    flags = set(row.get("_flags") or [])
    reasons: list[str] = []
    if row.get("page_type") == "fleet_registry_stock" and row.get("page_type_raw") in {"shipping_table", "mixed"}:
        reasons.append("page_type_mismatch")
    if any(
        flag.startswith("vessel_is_port_name")
        or flag.startswith("captain_is_port_name")
        or flag == "status_in_port_field"
        or flag == "vessel_captain_echo"
        for flag in flags
    ):
        reasons.append("field_contamination")
    if row.get("parsed_event_date_if_available") is pd.NaT or pd.isna(row.get("parsed_event_date_if_available")):
        if row.get("date") not in (None, "", pd.NA):
            reasons.append("date_parse_uncertain")
    if any(flag.startswith("invalid_event_type") for flag in flags) or row.get("validation_status") == "invalid":
        reasons.append("ocr_noise")
    return reasons


def compute_row_weight(row: Mapping[str, Any], config: WSLReliabilityConfig) -> float:
    confidence = float(np.clip(pd.to_numeric(row.get("_confidence"), errors="coerce"), 0.05, 1.0))
    weight = confidence
    page_type = row.get("page_type")
    weight *= float(config.source_weight_defaults.get(page_type, 0.0))
    for reason in map_penalty_reasons(row):
        weight *= float(config.flag_penalties.get(reason, 1.0))
    return float(np.clip(weight, 0.0, 1.0))


def infer_basin_probabilities(text: Any) -> dict[str, float]:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return {"Unknown": 1.0}
    raw_text = str(text).strip()
    if not raw_text:
        return {"Unknown": 1.0}
    matches: list[str] = []
    for pattern, basin, _theater in BASIN_RULES:
        if re.search(pattern, raw_text.lower()):
            matches.append(basin)
    if not matches:
        return {"Unknown": 1.0}
    basin_counts = pd.Series(matches).value_counts(normalize=True)
    return {str(key): float(value) for key, value in basin_counts.items()}


def expected_basin(probabilities: Mapping[str, float]) -> str:
    if not isinstance(probabilities, Mapping) or not probabilities:
        return "Unknown"
    return max(probabilities.items(), key=lambda item: item[1])[0]


def load_destination_ontology(config: WSLReliabilityConfig) -> pd.DataFrame:
    if not config.destination_ontology_path.exists():
        return pd.DataFrame(
            columns=["ground_or_route", "basin", "theater", "major_ground", "ground_for_model"]
        )
    ontology = pd.read_parquet(config.destination_ontology_path)
    keep = ["ground_or_route", "basin", "theater", "major_ground", "ground_for_model"]
    keep = [column for column in keep if column in ontology.columns]
    return ontology[keep].drop_duplicates("ground_or_route")


def load_voyage_reference(config: WSLReliabilityConfig) -> pd.DataFrame:
    voyage_df = pd.read_parquet(config.voyage_panel_path)
    voyage_df = sanitize_missing_strings(
        voyage_df,
        [
            "vessel_name_clean",
            "captain_name_clean",
            "agent_name_clean",
            "captain_id",
            "agent_id",
            "vessel_id",
            "home_port",
            "port_out",
            "port_in",
            "ground_or_route",
        ],
    )
    voyage_df["date_out"] = pd.to_datetime(voyage_df.get("date_out"), errors="coerce")
    voyage_df["date_in"] = pd.to_datetime(voyage_df.get("date_in"), errors="coerce")
    voyage_df["home_port_norm"] = voyage_df.get("home_port").map(normalize_port_name)
    voyage_df["port_out_norm"] = voyage_df.get("port_out").map(normalize_port_name)
    voyage_df["port_in_norm"] = voyage_df.get("port_in").map(normalize_port_name)
    voyage_df["vessel_name_norm"] = voyage_df.get("vessel_name_clean").map(normalize_vessel_name)
    voyage_df["captain_name_norm"] = voyage_df.get("captain_name_clean").map(normalize_name)
    voyage_df["agent_name_norm"] = voyage_df.get("agent_name_clean").map(normalize_name)
    voyage_df["year_out"] = pd.to_numeric(voyage_df.get("year_out"), errors="coerce")
    voyage_df["year_in"] = pd.to_numeric(voyage_df.get("year_in"), errors="coerce")
    voyage_df["decade"] = (voyage_df["year_out"] // 10) * 10

    ontology = load_destination_ontology(config)
    if not ontology.empty and "ground_or_route" in voyage_df.columns:
        voyage_df = voyage_df.merge(
            ontology,
            on="ground_or_route",
            how="left",
        )

    try:
        outcome_df = build_outcome_ml_dataset(force_rebuild=False, save=True)
        keep = [
            "voyage_id",
            "theta",
            "psi",
            "theta_hat_holdout",
            "psi_hat_holdout",
            "novice",
            "expert",
            "captain_voyage_num",
            "scarcity",
            "log_q",
            "bottom_decile",
            "bottom_5pct",
            "switch_agent",
            "switch_vessel",
        ]
        keep = [column for column in keep if column in outcome_df.columns]
        voyage_df = voyage_df.merge(
            outcome_df[keep].drop_duplicates("voyage_id"),
            on="voyage_id",
            how="left",
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not load outcome ML dataset for voyage reference: %s", exc)

    if "q_total_index" in voyage_df.columns and "zero_catch_or_failure" not in voyage_df.columns:
        voyage_df["zero_catch_or_failure"] = (voyage_df["q_total_index"].fillna(0) <= 0).astype(int)
    if "log_output" not in voyage_df.columns:
        base_output = pd.to_numeric(voyage_df.get("q_total_index"), errors="coerce").fillna(0)
        voyage_df["log_output"] = np.log1p(base_output.clip(lower=0))
    return voyage_df


def _serialize_nested_for_cache(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (Mapping, list, tuple, set, np.ndarray)):
        payload = value.tolist() if isinstance(value, np.ndarray) else value
        if isinstance(payload, set):
            payload = sorted(payload)
        return json.dumps(payload, sort_keys=True, default=_json_default)
    return value


def _deserialize_nested_from_cache(value: Any, default: Any) -> Any:
    if value is None or value is pd.NA:
        return default
    if isinstance(value, float) and np.isnan(value):
        return default
    if isinstance(value, (Mapping, list)):
        return value
    if not isinstance(value, str):
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def _prepare_events_cache_frame(events_df: pd.DataFrame) -> pd.DataFrame:
    cache_df = events_df.copy()
    for column in ["_raw", "_flags", "destination_basin_probs"]:
        if column in cache_df.columns:
            cache_df[column] = cache_df[column].map(_serialize_nested_for_cache)
    return cache_df


def _restore_events_cache_frame(events_df: pd.DataFrame) -> pd.DataFrame:
    restored = events_df.copy()
    if "_raw" in restored.columns:
        restored["_raw"] = restored["_raw"].map(lambda value: _deserialize_nested_from_cache(value, {}))
    if "_flags" in restored.columns:
        restored["_flags"] = restored["_flags"].map(lambda value: _deserialize_nested_from_cache(value, []))
    if "destination_basin_probs" in restored.columns:
        restored["destination_basin_probs"] = restored["destination_basin_probs"].map(
            lambda value: _deserialize_nested_from_cache(value, {"Unknown": 1.0})
        )
    return restored


def _ensure_unique_event_row_ids(events_df: pd.DataFrame) -> pd.DataFrame:
    if "event_row_id" not in events_df.columns or not events_df["event_row_id"].duplicated().any():
        return events_df
    repaired = events_df.copy()
    duplicate_rank = repaired.groupby("event_row_id").cumcount()
    mask = duplicate_rank > 0
    repaired.loc[mask, "event_row_id"] = [
        stable_hash([event_id, int(rank)], prefix="wslr_evt_")
        for event_id, rank in zip(repaired.loc[mask, "event_row_id"], duplicate_rank[mask])
    ]
    logger.warning("Resolved %d duplicate event_row_id collisions in flattened WSL events", int(mask.sum()))
    return repaired


def _coerce_match_string(value: Any) -> str | None:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    text = str(value).strip()
    return text or None


def _coerce_match_date(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def load_wsl_cleaned_events(config: WSLReliabilityConfig) -> pd.DataFrame:
    cache_dir = config.output_root / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"flattened_events_{compute_config_hash(config)}.parquet"
    if cache_path.exists():
        try:
            logger.info("Loading cached flattened WSL events from %s", cache_path)
            cached = _ensure_unique_event_row_ids(_restore_events_cache_frame(pd.read_parquet(cache_path)))
            return cached
        except Exception as exc:
            logger.warning("Could not read flattened WSL cache at %s (%s); rebuilding it", cache_path, exc)
            cache_path.unlink(missing_ok=True)

    issue_index = load_issue_index(config)
    issue_lookup = dict(zip(issue_index["wsl_issue_id"], issue_index["issue_date"]))
    rows: list[dict[str, Any]] = []
    with config.cleaned_events_path.open("r", encoding="utf-8") as handle:
        for page_num, line in enumerate(handle, start=1):
            try:
                page_record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping corrupt JSONL line at page_num=%d", page_num)
                continue
            page_key = page_record.get("page_key")
            pdf_name = page_record.get("pdf")
            issue_id = (
                re.sub(r"\.pdf$", "", str(pdf_name)).replace("-", "_")
                if pdf_name
                else str(page_key).split(".pdf")[0]
            )
            issue_date = issue_lookup.get(issue_id) or parse_issue_date_from_issue_id(issue_id)
            page_type, page_type_conf, page_reason, registry_share, panel_share = classify_page_process(page_record, config)
            raw_page_type = page_record.get("page_type")
            for event_index, event in enumerate(page_record.get("events") or []):
                raw_payload = event.get("_raw") or {}
                remarks = event.get("remarks")
                event_row_id = stable_hash(
                    [
                        issue_id,
                        page_key,
                        page_num,
                        page_record.get("page_route"),
                        event_index,
                        json.dumps(raw_payload, sort_keys=True, default=str),
                    ],
                    prefix="wslr_evt_",
                )
                event_row = {
                    "event_row_id": event_row_id,
                    "page_key": page_key,
                    "issue_id": issue_id,
                    "issue_date": pd.to_datetime(issue_date, errors="coerce"),
                    "pdf": pdf_name,
                    "page": page_record.get("page"),
                    "page_type_raw": raw_page_type,
                    "page_route": page_record.get("page_route"),
                    "page_type": page_type,
                    "page_type_confidence": page_type_conf,
                    "page_type_reason": page_reason,
                    "page_registry_share": registry_share,
                    "page_panel_share": panel_share,
                    "event_index": event_index,
                    "event_type": str(event.get("event_type") or "").lower(),
                    "vessel_name": event.get("vessel_name"),
                    "captain": event.get("captain"),
                    "agent": event.get("agent"),
                    "reported_by": event.get("reported_by"),
                    "port": event.get("port"),
                    "home_port": event.get("home_port"),
                    "destination": event.get("destination"),
                    "remarks": remarks,
                    "remarks_raw": remarks,
                    "date": event.get("date"),
                    "parsed_event_date_if_available": parse_partial_event_date(event.get("date"), issue_date),
                    "oil_sperm_bbls": pd.to_numeric(event.get("oil_sperm_bbls"), errors="coerce"),
                    "oil_whale_bbls": pd.to_numeric(event.get("oil_whale_bbls"), errors="coerce"),
                    "bone_lbs": pd.to_numeric(event.get("bone_lbs"), errors="coerce"),
                    "days_out": pd.to_numeric(event.get("days_out"), errors="coerce"),
                    "latitude": pd.to_numeric(event.get("latitude"), errors="coerce"),
                    "longitude": pd.to_numeric(event.get("longitude"), errors="coerce"),
                    "_raw": raw_payload,
                    "_flags": event.get("_flags") or [],
                    "_confidence": pd.to_numeric(event.get("_confidence"), errors="coerce"),
                    "_page_type": event.get("_page_type"),
                    "validation_status": event.get("validation_status"),
                    "panel_include": bool(event.get("panel_include")) if event.get("panel_include") is not None else False,
                    "vessel_name_raw": event.get("vessel_name_raw"),
                    "captain_raw": event.get("captain_raw"),
                    "agent_raw": event.get("agent_raw"),
                    "port_raw": event.get("port_raw"),
                    "home_port_raw": event.get("home_port_raw"),
                    "structured_field_pollution_present": False,
                }
                rows.append(event_row)
            if page_num % 1000 == 0:
                logger.info("Flattened %d WSL pages into %d event rows", page_num, len(rows))

    events_df = pd.DataFrame(rows)
    if events_df.empty:
        return events_df

    events_df = sanitize_missing_strings(
        events_df,
        [
            "vessel_name",
            "captain",
            "agent",
            "reported_by",
            "port",
            "home_port",
            "destination",
            "remarks",
        ],
    )
    for source_col, target_col, normalizer in [
        ("vessel_name", "vessel_name_norm", normalize_vessel_name),
        ("captain", "captain_norm", normalize_name),
        ("agent", "agent_norm", normalize_name),
        ("reported_by", "reported_by_norm", normalize_name),
        ("port", "port_norm", normalize_port_name),
        ("home_port", "home_port_norm", normalize_port_name),
    ]:
        unique_values = pd.Series(events_df[source_col].dropna().unique())
        mapping = {value: normalizer(value) for value in unique_values}
        events_df[target_col] = events_df[source_col].map(mapping)
    destination_values = pd.Series(events_df["destination"].dropna().unique())
    basin_mapping = {value: infer_basin_probabilities(value) for value in destination_values}
    basin_probabilities = events_df["destination"].map(basin_mapping)
    events_df["destination_basin_probs"] = basin_probabilities
    events_df["destination_basin"] = basin_probabilities.map(expected_basin)
    events_df["decade"] = (events_df["issue_date"].dt.year // 10) * 10
    confidence = pd.to_numeric(events_df["_confidence"], errors="coerce").clip(lower=0.05, upper=1.0).fillna(0.05)
    source_weight = events_df["page_type"].map(config.source_weight_defaults).fillna(0.0)
    field_contam = events_df["_flags"].map(
        lambda flags: any(
            str(flag).startswith("vessel_is_port_name")
            or str(flag).startswith("captain_is_port_name")
            or flag in {"status_in_port_field", "vessel_captain_echo"}
            for flag in (flags or [])
        )
    )
    page_type_mismatch = (
        (events_df["page_type"] == "fleet_registry_stock")
        & (events_df["page_type_raw"].isin(["shipping_table", "mixed"]))
    )
    date_parse_uncertain = events_df["parsed_event_date_if_available"].isna() & events_df["date"].notna()
    ocr_noise = events_df["_flags"].map(
        lambda flags: any(str(flag).startswith("invalid_event_type") for flag in (flags or []))
    ) | events_df["validation_status"].eq("invalid")
    row_weight = confidence * source_weight
    row_weight *= np.where(field_contam, float(config.flag_penalties["field_contamination"]), 1.0)
    row_weight *= np.where(page_type_mismatch, float(config.flag_penalties["page_type_mismatch"]), 1.0)
    row_weight *= np.where(date_parse_uncertain, float(config.flag_penalties["date_parse_uncertain"]), 1.0)
    row_weight *= np.where(ocr_noise, float(config.flag_penalties["ocr_noise"]), 1.0)
    events_df["row_weight"] = row_weight.clip(lower=0.0, upper=1.0)
    events_df["drop_hard"] = (
        (events_df["page_type"] == "non_table_or_other") & (events_df["page_type_confidence"] >= 0.95)
    )
    events_df["drop_reason"] = np.where(events_df["drop_hard"], "page_non_table_high_confidence", "")
    events_df["remarks_length"] = events_df["remarks"].fillna("").astype(str).str.len()
    events_df = _ensure_unique_event_row_ids(events_df)
    _prepare_events_cache_frame(events_df).to_parquet(cache_path, index=False)
    logger.info("Cached flattened WSL events at %s", cache_path)
    return events_df


def build_voyage_linkage(events_df: pd.DataFrame, voyages_df: pd.DataFrame, config: WSLReliabilityConfig) -> pd.DataFrame:
    # Known whaling port names — events with these as vessel_name are section headers
    _PORT_NAMES = frozenset({
        "NEW BEDFORD", "NANTUCKET", "MATTAPOISETT", "FAIRHAVEN",
        "EDGARTOWN", "PROVINCETOWN", "SIPPICAN", "WESTPORT",
        "DARTMOUTH", "HOLMES HOLE", "WARREN", "BRISTOL",
        "NEWPORT", "SAG HARBOR", "GREENPORT", "COLD SPRING",
        "MYSTIC", "STONINGTON", "NEW LONDON", "SALEM",
        "SANDWICH", "WAREHAM", "MARION", "TISBURY",
        "FALL RIVER", "DORCHESTER", "LYNN", "PLYMOUTH",
        "SAN FRANCISCO", "HONOLULU", "HUDSON", "POUGHKEEPSIE",
        "GAY HEAD",
    })

    vessel_index: dict[str, list[dict[str, Any]]] = {}
    base_columns = [
        "voyage_id",
        "vessel_name_norm",
        "captain_name_norm",
        "agent_name_norm",
        "home_port_norm",
        "port_out_norm",
        "port_in_norm",
        "date_out",
        "date_in",
        "captain_id",
        "agent_id",
        "vessel_id",
        "basin",
        "theater",
        "major_ground",
        "ground_or_route",
    ]
    available_columns = [column for column in base_columns if column in voyages_df.columns]
    optional_columns = {"captain_id", "agent_id", "vessel_id", "basin", "theater", "major_ground", "ground_or_route"}

    # Re-normalize panel vessel names with the updated normalizer (abbreviation expansion)
    for record in voyages_df[available_columns].dropna(subset=["voyage_id", "vessel_name_norm"]).to_dict("records"):
        for column in optional_columns:
            record.setdefault(column, pd.NA)
        raw_norm = str(record["vessel_name_norm"])
        re_normed = normalize_vessel_name(raw_norm) or raw_norm
        record["vessel_name_norm_original"] = raw_norm
        record["vessel_name_norm"] = re_normed
        vessel_index.setdefault(re_normed, []).append(record)
        # Also index under original name if different (backward compat)
        if raw_norm != re_normed:
            vessel_index.setdefault(raw_norm, []).append(record)

    # Build a secondary index mapping "first N tokens" → full names for substring matching
    _token_index: dict[str, set[str]] = {}
    for full_name in vessel_index:
        tokens = full_name.split()
        for n in range(1, len(tokens) + 1):
            prefix = " ".join(tokens[:n])
            _token_index.setdefault(prefix, set()).add(full_name)

    linkage_rows: list[dict[str, Any]] = []
    for event in events_df.itertuples(index=False):
        raw_vessel = str(event.vessel_name_norm) if pd.notna(event.vessel_name_norm) else None

        # Port-name contamination: skip linkage for section headers
        if raw_vessel and raw_vessel in _PORT_NAMES:
            fallback_episode = stable_hash(
                [event.vessel_name_norm, event.home_port_norm,
                 event.issue_date.year if pd.notna(event.issue_date) else ""],
                prefix="episode_",
            )
            linkage_rows.append({
                "event_row_id": event.event_row_id,
                "voyage_id": pd.NA,
                "linkage_method": "port_name_contamination",
                "linkage_confidence": 0.0,
                "episode_fallback_key": fallback_episode,
                "top_candidates": "[]",
                "captain_id": pd.NA, "agent_id": pd.NA, "vessel_id": pd.NA,
                "voyage_basin": pd.NA, "voyage_theater": pd.NA, "voyage_major_ground": pd.NA,
            })
            continue

        # Re-normalize the event vessel name with abbreviation expansion
        re_normed_vessel = normalize_vessel_name(raw_vessel) if raw_vessel else None

        # Look up candidates: try exact match first, then substring
        candidate_records = []
        if re_normed_vessel and pd.notna(event.vessel_name_norm):
            candidate_records = vessel_index.get(re_normed_vessel, [])
            # Fallback: if no exact match, try substring (e.g., "GAY HEAD" → "GAY HEAD I")
            if not candidate_records and re_normed_vessel in _token_index:
                for full_name in _token_index[re_normed_vessel]:
                    if full_name != re_normed_vessel:
                        candidate_records.extend(vessel_index.get(full_name, []))
            # Also try: panel name is a prefix of WSL name (e.g., "GOSNOLD" matches "BART GOSNOLD")
            if not candidate_records:
                for token_start in [re_normed_vessel.split()[-1]]:
                    if len(token_start) >= 4 and token_start in _token_index:
                        for full_name in _token_index[token_start]:
                            candidate_records.extend(vessel_index.get(full_name, []))

        match_date = _coerce_match_date(
            event.parsed_event_date_if_available if pd.notna(event.parsed_event_date_if_available) else event.issue_date
        )
        captain_norm = _coerce_match_string(event.captain_norm)
        port_norm = _coerce_match_string(event.port_norm)
        home_port_norm = _coerce_match_string(event.home_port_norm)
        best_score = -1.0
        best_record: dict[str, Any] | None = None
        top_scores: list[dict[str, Any]] = []
        for record in candidate_records:
            date_score = compute_date_score(
                match_date,
                _coerce_match_date(record.get("date_out")),
                _coerce_match_date(record.get("date_in")),
                tolerance_days=180,
            )
            captain_score = compute_name_similarity(
                captain_norm,
                _coerce_match_string(record.get("captain_name_norm")),
            )
            port_score = max(
                compute_port_score(
                    port_norm,
                    _coerce_match_string(record.get("port_out_norm")),
                    _coerce_match_string(record.get("port_in_norm")),
                ),
                compute_port_score(
                    home_port_norm,
                    _coerce_match_string(record.get("home_port_norm")),
                    _coerce_match_string(record.get("port_out_norm")),
                ),
            )
            composite = 0.45 + 0.30 * date_score + 0.15 * captain_score + 0.10 * port_score
            if pd.notna(event.row_weight):
                composite *= 0.75 + 0.25 * float(event.row_weight)
            top_scores.append(
                {
                    "voyage_id": record["voyage_id"],
                    "score": float(composite),
                    "date_score": float(date_score),
                    "captain_score": float(captain_score),
                    "port_score": float(port_score),
                }
            )
            if composite > best_score:
                best_score = composite
                best_record = record

        top_scores = sorted(top_scores, key=lambda item: item["score"], reverse=True)[:3]
        method = "no_candidates"
        voyage_id = pd.NA
        if best_record is not None and best_score >= config.linkage_threshold:
            method = "exact_vessel_scored"
            voyage_id = best_record["voyage_id"]
        elif best_record is not None and best_score >= config.linkage_low_conf_threshold:
            method = "exact_vessel_low_confidence"
        elif best_record is not None:
            method = "below_threshold"

        fallback_episode = stable_hash(
            [
                event.vessel_name_norm,
                event.home_port_norm,
                event.issue_date.year if pd.notna(event.issue_date) else "",
            ],
            prefix="episode_",
        )
        linkage_rows.append(
            {
                "event_row_id": event.event_row_id,
                "voyage_id": voyage_id,
                "linkage_method": method,
                "linkage_confidence": max(best_score, 0.0),
                "episode_fallback_key": fallback_episode,
                "top_candidates": json.dumps(top_scores, sort_keys=True),
                "captain_id": best_record.get("captain_id") if best_record and pd.notna(voyage_id) else pd.NA,
                "agent_id": best_record.get("agent_id") if best_record and pd.notna(voyage_id) else pd.NA,
                "vessel_id": best_record.get("vessel_id") if best_record and pd.notna(voyage_id) else pd.NA,
                "voyage_basin": best_record.get("basin") if best_record and pd.notna(voyage_id) else pd.NA,
                "voyage_theater": best_record.get("theater") if best_record and pd.notna(voyage_id) else pd.NA,
                "voyage_major_ground": best_record.get("major_ground") if best_record and pd.notna(voyage_id) else pd.NA,
            }
        )
    linkage_df = pd.DataFrame(linkage_rows)
    linkage_df["entity_link_uncertain"] = linkage_df["linkage_confidence"] < config.linkage_threshold
    return linkage_df


def attach_voyage_linkage(events_df: pd.DataFrame, linkage_df: pd.DataFrame, config: WSLReliabilityConfig) -> pd.DataFrame:
    linkage_columns = [column for column in linkage_df.columns if column != "event_row_id"]
    merged = events_df.merge(linkage_df, on="event_row_id", how="left", suffixes=("", "__linkage"))
    for column in linkage_columns:
        linkage_column = f"{column}__linkage"
        if linkage_column not in merged.columns:
            continue
        if column in events_df.columns:
            merged[column] = merged[column].combine_first(merged[linkage_column])
            merged = merged.drop(columns=[linkage_column])
        else:
            merged = merged.rename(columns={linkage_column: column})
    merged["episode_id"] = merged["voyage_id"].fillna(merged["episode_fallback_key"])
    if "row_weight_linkage_penalty_applied" not in merged.columns:
        merged["row_weight_linkage_penalty_applied"] = False
    penalty_mask = ~merged["row_weight_linkage_penalty_applied"].fillna(False)
    if penalty_mask.any():
        weights = pd.to_numeric(merged.loc[penalty_mask, "row_weight"], errors="coerce").fillna(0.0)
        penalties = np.where(
            merged.loc[penalty_mask, "entity_link_uncertain"].fillna(False),
            float(config.flag_penalties["entity_link_uncertain"]),
            1.0,
        )
        merged.loc[penalty_mask, "row_weight"] = weights * penalties
        merged.loc[penalty_mask, "row_weight_linkage_penalty_applied"] = True
    return merged


def _serialize_parquet_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, set):
        value = sorted(value)
    return json.dumps(value, sort_keys=True, default=_json_default)


def _prepare_dataframe_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    for column in prepared.select_dtypes(include=["object", "string"]).columns:
        non_missing = prepared[column].dropna()
        if non_missing.empty:
            continue
        if non_missing.map(lambda value: isinstance(value, str)).all():
            continue
        prepared[column] = prepared[column].map(_serialize_parquet_value)
    return prepared


def save_dataframe(df: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        _prepare_dataframe_for_parquet(df).to_parquet(path, index=index)
    elif path.suffix == ".csv":
        df.to_csv(path, index=index)
    else:
        raise ValueError(f"Unsupported output format: {path}")


def build_manifest_payload(
    config: WSLReliabilityConfig,
    output_paths: Sequence[Path],
    *,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "title": "wsl_reliability_ml",
        "generated_at_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config_hash": compute_config_hash(config),
        "code_version_hint": "local_workspace",
        "inputs": {
            "cleaned_events_path": str(config.cleaned_events_path),
            "cleaned_events_hash": compute_file_hash(config.cleaned_events_path),
            "issue_index_path": str(config.issue_index_path),
            "issue_index_hash": compute_file_hash(config.issue_index_path),
            "voyage_panel_path": str(config.voyage_panel_path),
            "voyage_panel_hash": compute_file_hash(config.voyage_panel_path),
        },
        "outputs": [
            {
                "path": str(path),
                "sha256": compute_file_hash(path) if path.exists() else None,
            }
            for path in output_paths
        ],
    }
    if extra:
        payload.update(extra)
    return payload


def summarize_overlap(propensity: pd.Series, clip_low: float, clip_high: float) -> dict[str, float]:
    clean = pd.to_numeric(propensity, errors="coerce").dropna()
    if clean.empty:
        return {
            "n": 0,
            "mean": math.nan,
            "std": math.nan,
            "share_outside_clip": math.nan,
            "clip_low": clip_low,
            "clip_high": clip_high,
        }
    return {
        "n": int(clean.shape[0]),
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=0)),
        "min": float(clean.min()),
        "max": float(clean.max()),
        "share_outside_clip": float(((clean < clip_low) | (clean > clip_high)).mean()),
        "clip_low": clip_low,
        "clip_high": clip_high,
    }
