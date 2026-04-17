from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder

from .utils import (
    PRIMARY_CLASSES,
    SECONDARY_TAGS,
    REMARK_TEXT_HINT,
    COORDINATE_HINT,
    MULTI_PORT_HINT,
    WSLReliabilityConfig,
    compute_config_hash,
    save_dataframe,
    stable_hash,
    write_json,
)

logger = logging.getLogger(__name__)

PRODUCTIVE_TAGS = {"whales_sighted", "good_catch", "full_or_nearly_full"}
WEAK_PRODUCTIVE_TAGS = {"small_catch", "no_whales_or_clean"}
DISTRESS_TAGS = {
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
}
REPAIR_TAGS = {"in_port", "repairing", "wintering_or_delayed"}
HOMEBOUND_TAGS = {"bound_home", "ordered_home_or_recalled"}
TERMINAL_TAGS = {"wrecked_or_condemned", "abandoned_or_lost"}
ASSISTANCE_TAGS = {
    "assisted_or_towed",
    "transferred_oil_or_bone",
    "transferred_crew_or_boats",
    "received_orders_or_intelligence",
}
COMMERCIAL_TAGS = {"sold_or_withdrawn", "non_whaling_trade", "legal_or_customs"}

KEYWORD_RULES = {
    "coordinates_only": [r"\blat\b", r"\blon\b", r"\d+\s*[ns]\b", r"\d+\s*[ew]\b"],
    "spoken_or_seen": [r"\bspoke\b", r"\bspoken\b", r"\bspk\b", r"\bseen\b"],
    "received_report": [r"\breported\b", r"\breceived report\b", r"\bheard from\b"],
    "whales_sighted": [r"\bwhales sighted\b", r"\bwhales seen\b", r"\bwhales\b"],
    "good_catch": [r"\bgood catch\b", r"\bdoing well\b", r"\bcatching\b"],
    "full_or_nearly_full": [r"\bfull cargo\b", r"\bfull\b", r"\bnearly full\b", r"\bfilled\b"],
    "small_catch": [r"\bsmall catch\b", r"\bbut little\b", r"\blittle oil\b"],
    "no_whales_or_clean": [r"\bno whales\b", r"\bclean\b", r"\bcleaned out\b", r"\bnothing\b"],
    "leaking_or_damaged": [r"\bleak", r"\bdamaged\b", r"\bstove\b", r"\bsprung\b"],
    "dismasted_or_disabled": [r"\bdismast", r"\bdisabled\b", r"\bdisabled\b", r"\blost mast\b"],
    "ashore_or_aground": [r"\bashore\b", r"\baground\b", r"\bstranded\b", r"\breef\b"],
    "storm_or_weather": [r"\bstorm\b", r"\bgale\b", r"\bhurricane\b", r"\bbad weather\b"],
    "ice_or_frozen": [r"\bice\b", r"\bfrozen\b", r"\bblocked by ice\b"],
    "fire_or_collision": [r"\bfire\b", r"\bburnt\b", r"\bcollision\b", r"\brun into\b"],
    "sick_or_injured": [r"\bsick\b", r"\binjured\b", r"\bill\b"],
    "death_or_mortality": [r"\bdied\b", r"\bdead\b", r"\bkilled\b", r"\bmortality\b"],
    "desertion_or_crew_shortage": [r"\bdesert", r"\bshort handed\b", r"\bcrew short\b"],
    "mutiny_or_discipline": [r"\bmutiny\b", r"\bmutinous\b", r"\bdiscipline\b"],
    "short_provisions_or_water": [r"\bshort of water\b", r"\bshort provisions\b", r"\bwant of water\b"],
    "in_port": [r"\bin port\b", r"\bat [a-z .'-]+ repairing\b", r"\bput into\b"],
    "repairing": [r"\brepair", r"\brefitting\b", r"\bfit'g\b", r"\bfitting out\b"],
    "wintering_or_delayed": [r"\bwinter", r"\bdetained\b", r"\bdelayed\b", r"\bstopped\b"],
    "bound_home": [r"\bbound home\b", r"\bhomeward bound\b", r"\bhome\b"],
    "ordered_home_or_recalled": [r"\bordered home\b", r"\brecalled\b", r"\bordered to return\b"],
    "wrecked_or_condemned": [r"\bwreck", r"\bcondemn", r"\btotal loss\b"],
    "abandoned_or_lost": [r"\babandon", r"\blost\b", r"\bsunk\b", r"\bfoundered\b"],
    "assisted_or_towed": [r"\bassisted\b", r"\btowed\b", r"\bin tow\b"],
    "transferred_oil_or_bone": [r"\btransferred oil\b", r"\boil transferred\b", r"\bbone transferred\b"],
    "transferred_crew_or_boats": [r"\bcrew transferred\b", r"\bboats transferred\b", r"\btransferred crew\b"],
    "received_orders_or_intelligence": [r"\border", r"\bintelligence\b", r"\binstructions\b"],
    "sold_or_withdrawn": [r"\bsold\b", r"\bwithdrawn\b", r"\bcondemned and sold\b"],
    "non_whaling_trade": [r"\bon freight\b", r"\bmerchant\b", r"\btrading\b"],
    "legal_or_customs": [r"\bcustoms\b", r"\blegal\b", r"\bseized\b", r"\blibel\b"],
    "ocr_or_layout_noise": [r"\bcrew list of vessels sailed\b", r"\bwhalemen'?s shipping list\b"],
}


@dataclass
class RemarksFeatureBundle:
    word_vectorizer: TfidfVectorizer
    char_vectorizer: TfidfVectorizer
    encoder: OneHotEncoder
    numeric_columns: list[str]
    categorical_columns: list[str]


def _normalize_ocr_punctuation(text: str) -> str:
    normalized = text.replace("—", "-").replace("–", "-").replace("’", "'")
    normalized = re.sub(r"[|]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip(" ;,")


def _overflow_fragments(row: pd.Series) -> list[str]:
    fragments: list[str] = []
    flags = set(row.get("_flags") or [])
    for field in ["destination", "agent", "reported_by"]:
        value = row.get(field)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        text = str(value).strip()
        if not text:
            continue
        if field == "agent" and len(text.split()) <= 4 and "&" in text and not REMARK_TEXT_HINT.search(text):
            continue
        if field == "reported_by" and len(text.split()) <= 3 and not REMARK_TEXT_HINT.search(text):
            continue
        if REMARK_TEXT_HINT.search(text) or COORDINATE_HINT.search(text) or MULTI_PORT_HINT.search(text):
            fragments.append(text)
    raw_payload = row.get("_raw") or {}
    if isinstance(raw_payload, dict):
        for field in ["date", "p", "port", "agent", "reported_by", "dest"]:
            raw_value = raw_payload.get(field)
            if raw_value is None:
                continue
            text = str(raw_value).strip()
            if not text:
                continue
            if REMARK_TEXT_HINT.search(text) or COORDINATE_HINT.search(text):
                if text not in fragments:
                    fragments.append(text)
    if any(
        flag.startswith("vessel_is_port_name")
        or flag.startswith("captain_is_port_name")
        or flag == "status_in_port_field"
        for flag in flags
    ):
        for field in ["port", "date"]:
            value = row.get(field)
            if value is not None and not pd.isna(value):
                text = str(value).strip()
                if text and text not in fragments:
                    fragments.append(text)
    return fragments


def build_remarks_canonical_text(events_df: pd.DataFrame) -> pd.DataFrame:
    df = events_df.copy()
    canonical_text: list[str] = []
    overflow_only: list[bool] = []
    pollution_present: list[bool] = []
    empty_flags: list[bool] = []

    for row in df.to_dict(orient="records"):
        remarks_raw = row.get("remarks")
        remarks_text = "" if remarks_raw is None or pd.isna(remarks_raw) else str(remarks_raw).strip()
        overflow = _overflow_fragments(pd.Series(row))
        pollution = bool(overflow)
        pieces = [piece for piece in [remarks_text, *overflow] if piece]
        combined = "; ".join(dict.fromkeys(pieces))
        combined = _normalize_ocr_punctuation(combined)
        canonical_text.append(combined)
        overflow_only.append(bool(not remarks_text and overflow))
        pollution_present.append(pollution)
        empty_flags.append(combined == "")

    df["remarks_canonical_text"] = canonical_text
    df["remarks_canonical_normalized"] = (
        df["remarks_canonical_text"].fillna("").astype(str).str.lower().map(_normalize_ocr_punctuation)
    )
    df["remarks_missing_or_empty"] = empty_flags
    df["remarks_from_overflow_only"] = overflow_only
    df["structured_field_pollution_present"] = pollution_present
    df["remarks_length"] = df["remarks_canonical_text"].fillna("").astype(str).str.len()
    return df


def _match_any(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def _derive_rule_secondary_tags(row: pd.Series) -> list[str]:
    text = str(row.get("remarks_canonical_normalized") or "")
    tags: set[str] = set()
    event_type = str(row.get("event_type") or "").lower()
    if COORDINATE_HINT.search(text) and not REMARK_TEXT_HINT.search(text.replace("lat", "").replace("lon", "")):
        tags.add("coordinates_only")
    if MULTI_PORT_HINT.search(text):
        tags.add("multi_port_list")
    if event_type == "spk":
        tags.add("spoken_or_seen")
    if event_type == "rpt":
        tags.add("received_report")
    if event_type == "inp":
        tags.add("in_port")
    if event_type == "wrk":
        tags.add("wrecked_or_condemned")
    for tag, patterns in KEYWORD_RULES.items():
        if _match_any(text, patterns):
            tags.add(tag)
    if "good_catch" in tags and "full_or_nearly_full" in tags:
        tags.discard("small_catch")
    if row.get("structured_field_pollution_present") and row.get("remarks_missing_or_empty"):
        tags.add("ocr_or_layout_noise")
    return [tag for tag in SECONDARY_TAGS if tag in tags]


def _rule_primary_class(tags: set[str], row: pd.Series) -> str:
    if tags & TERMINAL_TAGS or str(row.get("event_type") or "").lower() == "wrk":
        return "terminal_loss"
    if tags & DISTRESS_TAGS:
        return "distress_hazard"
    if tags & REPAIR_TAGS:
        return "interruption_repair"
    if tags & HOMEBOUND_TAGS:
        return "homebound_or_termination"
    if tags & WEAK_PRODUCTIVE_TAGS:
        return "weak_or_empty_productivity"
    if tags & PRODUCTIVE_TAGS:
        return "positive_productivity"
    if tags & ASSISTANCE_TAGS:
        return "assistance_transfer_coordination"
    if tags & COMMERCIAL_TAGS:
        return "commercial_admin_status"
    if row.get("remarks_missing_or_empty") and row.get("structured_field_pollution_present"):
        return "extraction_noise_or_uncertain"
    text = str(row.get("remarks_canonical_normalized") or "")
    if not text:
        return "routine_info"
    if not REMARK_TEXT_HINT.search(text):
        return "routine_info"
    if "ocr_or_layout_noise" in tags:
        return "extraction_noise_or_uncertain"
    return "routine_info"


def _distress_score(tags: set[str], primary_class: str) -> int:
    if primary_class == "terminal_loss":
        return 4
    if tags & {"ashore_or_aground", "fire_or_collision", "dismasted_or_disabled"}:
        return 3
    if primary_class == "distress_hazard":
        return 2
    if tags & {"repairing", "wintering_or_delayed", "sick_or_injured"}:
        return 1
    return 0


def _productivity_score(tags: set[str], primary_class: str) -> int:
    if "full_or_nearly_full" in tags:
        return 2
    if primary_class == "positive_productivity":
        return 1
    if "small_catch" in tags:
        return -1
    if primary_class == "weak_or_empty_productivity":
        return -2
    return 0


def _actionability_score(tags: set[str], primary_class: str) -> int:
    if primary_class in {"terminal_loss", "distress_hazard"}:
        return 3
    if primary_class in {"interruption_repair", "homebound_or_termination", "assistance_transfer_coordination"}:
        return 2
    if tags & {"received_report", "spoken_or_seen", "good_catch", "whales_sighted"}:
        return 1
    return 0


def _contamination_score(row: pd.Series) -> int:
    flags = set(row.get("_flags") or [])
    count = 0
    if row.get("structured_field_pollution_present"):
        count += 1
    if any(
        flag.startswith("vessel_is_port_name")
        or flag.startswith("captain_is_port_name")
        or flag == "vessel_captain_echo"
        or flag == "status_in_port_field"
        for flag in flags
    ):
        count += 1
    if row.get("validation_status") in {"invalid", "suspicious"}:
        count += 1
    return int(np.clip(count, 0, 3))


def _source_type(row: pd.Series, tags: set[str]) -> str:
    if row.get("page_type") == "fleet_registry_stock":
        return "cumulative_registry_summary"
    if row.get("event_type") in {"spk", "rpt"} or tags & {"spoken_or_seen", "received_report"}:
        return "third_party_report"
    if row.get("remarks_canonical_text"):
        return "direct_status_note"
    return "unknown"


def _apply_rule_taxonomy(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    tags_list: list[list[str]] = []
    primary_classes: list[str] = []
    distress_scores: list[int] = []
    productivity_scores: list[int] = []
    actionability_scores: list[int] = []
    contamination_scores: list[int] = []
    source_types: list[str] = []
    reasons: list[str] = []
    rule_confidences: list[float] = []

    for row in work.to_dict(orient="records"):
        row_series = pd.Series(row)
        tags = _derive_rule_secondary_tags(row_series)
        tags_set = set(tags)
        primary_class = _rule_primary_class(tags_set, row_series)
        distress = _distress_score(tags_set, primary_class)
        productivity = _productivity_score(tags_set, primary_class)
        actionability = _actionability_score(tags_set, primary_class)
        contamination = _contamination_score(row_series)
        source_type = _source_type(row_series, tags_set)
        confidence = 0.55
        if primary_class == "terminal_loss":
            confidence = 0.95
        elif primary_class == "distress_hazard":
            confidence = 0.85
        elif primary_class in {"interruption_repair", "homebound_or_termination"}:
            confidence = 0.80
        elif tags_set:
            confidence = 0.70
        if contamination >= 2:
            confidence *= 0.8
        tags_list.append(tags)
        primary_classes.append(primary_class)
        distress_scores.append(distress)
        productivity_scores.append(productivity)
        actionability_scores.append(actionability)
        contamination_scores.append(contamination)
        source_types.append(source_type)
        reasons.append("; ".join(tags or [primary_class]))
        rule_confidences.append(float(np.clip(confidence, 0.1, 0.99)))

    work["rule_secondary_tags"] = tags_list
    work["rule_primary_class"] = primary_classes
    work["rule_distress_severity_0_4"] = distress_scores
    work["rule_productivity_polarity_m2_p2"] = productivity_scores
    work["rule_actionability_0_3"] = actionability_scores
    work["rule_contamination_score_0_3"] = contamination_scores
    work["rule_source_type"] = source_types
    work["rule_reason_codes"] = reasons
    work["rule_confidence"] = rule_confidences
    return work


def _build_feature_bundle(train_df: pd.DataFrame) -> RemarksFeatureBundle:
    word_vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        min_df=2,
        max_features=20000,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 6),
        min_df=2,
        max_features=15000,
    )
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    categorical_columns = ["event_type", "page_type"]
    numeric_columns = [
        "structured_field_pollution_present",
        "_confidence",
        "row_weight",
        "remarks_length",
        "page_type_confidence",
    ]
    word_vectorizer.fit(train_df["remarks_canonical_normalized"].fillna(""))
    char_vectorizer.fit(train_df["remarks_canonical_normalized"].fillna(""))
    encoder.fit(train_df[categorical_columns].fillna("UNK"))
    return RemarksFeatureBundle(
        word_vectorizer=word_vectorizer,
        char_vectorizer=char_vectorizer,
        encoder=encoder,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )


def _transform_features(bundle: RemarksFeatureBundle, frame: pd.DataFrame) -> sparse.csr_matrix:
    text = frame["remarks_canonical_normalized"].fillna("")
    X_word = bundle.word_vectorizer.transform(text)
    X_char = bundle.char_vectorizer.transform(text)
    X_cat = bundle.encoder.transform(frame[bundle.categorical_columns].fillna("UNK"))
    numeric = frame[bundle.numeric_columns].copy()
    for column in numeric.columns:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce").fillna(0.0)
    X_num = sparse.csr_matrix(numeric.to_numpy(dtype=float))
    return sparse.hstack([X_word, X_char, X_cat, X_num], format="csr")


def _select_label_source(frame: pd.DataFrame) -> str:
    if "primary_class" in frame.columns and frame["primary_class"].notna().sum() >= 100:
        return "manual"
    return "weak_rules"


def sample_remarks_goldset(events_df: pd.DataFrame, config: WSLReliabilityConfig) -> pd.DataFrame:
    df = build_remarks_canonical_text(events_df)
    df = _apply_rule_taxonomy(df)
    rng = np.random.default_rng(config.random_seed)

    confidence_bin = pd.cut(
        pd.to_numeric(df["_confidence"], errors="coerce").fillna(0),
        bins=[-0.01, 0.6, 0.8, 1.0],
        labels=["low", "mid", "high"],
    ).astype(str)
    length_bin = pd.cut(
        df["remarks_length"].fillna(0),
        bins=[-0.01, 1, 20, 80, np.inf],
        labels=["empty", "short", "medium", "long"],
    ).astype(str)
    oversample_weight = (
        1.0
        + 3.0 * df["rule_primary_class"].isin(
            [
                "terminal_loss",
                "distress_hazard",
                "weak_or_empty_productivity",
                "interruption_repair",
                "commercial_admin_status",
            ]
        ).astype(float)
        + 1.5 * df["structured_field_pollution_present"].astype(float)
        + 1.0 * confidence_bin.eq("low").astype(float)
    )
    sample_frame = df.copy()
    sample_frame["_sample_weight"] = oversample_weight
    sample_frame["_stratum"] = (
        sample_frame["event_type"].fillna("unk").astype(str)
        + "|"
        + sample_frame["page_type"].fillna("unk").astype(str)
        + "|"
        + sample_frame["decade"].fillna(-1).astype(int).astype(str)
        + "|"
        + confidence_bin
        + "|"
        + length_bin
        + "|"
        + sample_frame["structured_field_pollution_present"].astype(int).astype(str)
    )

    pieces = []
    for _, group in sample_frame.groupby("_stratum", dropna=False):
        take = max(1, int(round(config.gold_sample_size_target * len(group) / max(len(sample_frame), 1))))
        take = min(take, len(group))
        probabilities = group["_sample_weight"] / group["_sample_weight"].sum()
        chosen = rng.choice(group.index.to_numpy(), size=take, replace=False, p=probabilities.to_numpy())
        pieces.append(group.loc[chosen])
    sampled = pd.concat(pieces).drop_duplicates("event_row_id")
    if len(sampled) > config.gold_sample_size_target:
        probabilities = sampled["_sample_weight"] / sampled["_sample_weight"].sum()
        chosen = rng.choice(
            sampled.index.to_numpy(),
            size=config.gold_sample_size_target,
            replace=False,
            p=probabilities.to_numpy(),
        )
        sampled = sampled.loc[chosen].copy()
    sampled = sampled.sort_values(["issue_date", "page_key", "event_row_id"]).reset_index(drop=True)
    sampled["gold_sample_row_id"] = [
        stable_hash([row.event_row_id, i], prefix="remarks_gold_")
        for i, row in enumerate(sampled.itertuples(index=False))
    ]
    for manual_column in [
        "primary_class",
        "secondary_tags",
        "distress_severity_0_4",
        "productivity_polarity_m2_p2",
        "actionability_0_3",
        "contamination_score_0_3",
        "source_type",
        "free_text_notes",
    ]:
        if manual_column not in sampled.columns:
            sampled[manual_column] = pd.NA
    return sampled


def _train_supported_tagger(
    X_train: sparse.csr_matrix,
    X_test: sparse.csr_matrix,
    tag_lists_train: list[list[str]],
    tag_lists_test: list[list[str]],
    config: WSLReliabilityConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    mlb = MultiLabelBinarizer(classes=SECONDARY_TAGS)
    Y_train_full = mlb.fit_transform(tag_lists_train)
    Y_test_full = mlb.transform(tag_lists_test)
    support = Y_train_full.sum(axis=0)
    supported_tags = [tag for tag, count in zip(mlb.classes_, support) if count >= config.remarks_min_tag_support]
    if not supported_tags:
        return (
            {
                "supported_tags": [],
                "mlb": mlb,
                "model": None,
                "thresholds": {},
            },
            [],
        )

    supported_indices = [list(mlb.classes_).index(tag) for tag in supported_tags]
    Y_train = Y_train_full[:, supported_indices]
    Y_test = Y_test_full[:, supported_indices]
    model = OneVsRestClassifier(
        LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            solver="liblinear",
        )
    )
    model.fit(X_train, Y_train)
    probas = model.predict_proba(X_test)
    thresholds: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    for j, tag in enumerate(supported_tags):
        best_threshold = 0.5
        best_f1 = -1.0
        y_true = Y_test[:, j]
        for threshold in [0.2, 0.3, 0.4, 0.5, 0.6]:
            y_pred = (probas[:, j] >= threshold).astype(int)
            score = f1_score(y_true, y_pred, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = threshold
        thresholds[tag] = best_threshold
        y_pred = (probas[:, j] >= best_threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        rows.append(
            {
                "tag": tag,
                "support_train": int(Y_train[:, j].sum()),
                "support_test": int(Y_test[:, j].sum()),
                "threshold": best_threshold,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )
    return {
        "supported_tags": supported_tags,
        "mlb": mlb,
        "model": model,
        "thresholds": thresholds,
    }, rows


def train_remarks_models(labeled_df: pd.DataFrame, config: WSLReliabilityConfig) -> dict[str, Any]:
    df = build_remarks_canonical_text(labeled_df)
    df = _apply_rule_taxonomy(df)
    label_source = _select_label_source(df)
    if label_source == "manual":
        df["train_primary_class"] = df["primary_class"].fillna(df["rule_primary_class"])
        df["train_secondary_tags"] = df["secondary_tags"].fillna("").map(
            lambda value: value if isinstance(value, list) else [tag.strip() for tag in str(value).split(";") if tag.strip()]
        )
        df["train_distress"] = pd.to_numeric(df["distress_severity_0_4"], errors="coerce").fillna(df["rule_distress_severity_0_4"])
        df["train_actionability"] = pd.to_numeric(df["actionability_0_3"], errors="coerce").fillna(df["rule_actionability_0_3"])
        df["train_contamination"] = pd.to_numeric(df["contamination_score_0_3"], errors="coerce").fillna(df["rule_contamination_score_0_3"])
    else:
        df["train_primary_class"] = df["rule_primary_class"]
        df["train_secondary_tags"] = df["rule_secondary_tags"]
        df["train_distress"] = df["rule_distress_severity_0_4"]
        df["train_actionability"] = df["rule_actionability_0_3"]
        df["train_contamination"] = df["rule_contamination_score_0_3"]

    trainable = df[df["train_primary_class"].notna()].copy()
    if len(trainable) > config.remarks_max_train_rows:
        sampled_groups: list[pd.DataFrame] = []
        total_rows = len(trainable)
        for _label, group in trainable.groupby("train_primary_class", sort=False):
            sample_n = min(
                len(group),
                max(50, int(round(config.remarks_max_train_rows * len(group) / total_rows))),
            )
            sampled_groups.append(group.sample(sample_n, random_state=config.random_seed))
        trainable = pd.concat(sampled_groups, ignore_index=True)

    class_counts = trainable["train_primary_class"].value_counts()
    rare_classes = set(class_counts[class_counts < 2].index.tolist())
    rare_train = trainable[trainable["train_primary_class"].isin(rare_classes)].copy()
    split_pool = trainable[~trainable["train_primary_class"].isin(rare_classes)].copy()
    if split_pool.empty:
        train_df = trainable.copy()
        test_df = trainable.copy()
    else:
        stratify = None
        pool_counts = split_pool["train_primary_class"].value_counts()
        if len(pool_counts) > 1 and int(pool_counts.min()) >= 2:
            stratify = split_pool["train_primary_class"]
        train_df, test_df = train_test_split(
            split_pool,
            test_size=0.2,
            random_state=config.random_seed,
            stratify=stratify,
        )
        if not rare_train.empty:
            train_df = pd.concat([train_df, rare_train], ignore_index=True)

    bundle = _build_feature_bundle(train_df)
    X_train = _transform_features(bundle, train_df)
    X_test = _transform_features(bundle, test_df)

    primary_estimator = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )
    train_class_counts = train_df["train_primary_class"].value_counts()
    min_train_class_count = int(train_class_counts.min()) if not train_class_counts.empty else 0
    if len(train_class_counts) > 1 and min_train_class_count >= 3:
        primary_model: Any = CalibratedClassifierCV(
            estimator=primary_estimator,
            method="sigmoid",
            cv=min(3, min_train_class_count),
        )
    else:
        primary_model = primary_estimator
    primary_model.fit(X_train, train_df["train_primary_class"])
    y_pred = primary_model.predict(X_test)
    y_prob = primary_model.predict_proba(X_test)
    class_order = list(primary_model.classes_)
    y_true = test_df["train_primary_class"].to_numpy()
    distress_mask = np.isin(y_true, ["distress_hazard", "terminal_loss"])
    distress_pred_mask = np.isin(y_pred, ["distress_hazard", "terminal_loss"])
    terminal_true = y_true == "terminal_loss"
    terminal_pred = y_pred == "terminal_loss"
    contamination_auc = np.nan

    tag_bundle, rare_tag_rows = _train_supported_tagger(
        X_train,
        X_test,
        train_df["train_secondary_tags"].tolist(),
        test_df["train_secondary_tags"].tolist(),
        config,
    )

    def _fit_ordinal(column: str) -> tuple[Any, dict[str, Any]]:
        y_train = train_df[column].astype(int)
        observed_classes = sorted(pd.Series(y_train).dropna().astype(int).unique().tolist())
        if len(observed_classes) < 2:
            constant = observed_classes[0] if observed_classes else 0
            model = DummyClassifier(strategy="constant", constant=constant)
            model.fit(X_train, np.repeat(constant, X_train.shape[0]))
        else:
            model = LogisticRegression(max_iter=1500, class_weight="balanced", solver="lbfgs")
            model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return model, {
            "macro_f1": float(f1_score(test_df[column].astype(int), preds, average="macro", zero_division=0)),
            "classes": observed_classes,
        }

    distress_model, distress_metrics = _fit_ordinal("train_distress")
    actionability_model, actionability_metrics = _fit_ordinal("train_actionability")
    contamination_model, contamination_metrics = _fit_ordinal("train_contamination")
    if len(np.unique(test_df["train_contamination"])) > 1:
        contamination_binary = (test_df["train_contamination"].astype(int) >= 2).astype(int)
        contamination_score = contamination_model.predict_proba(X_test)[:, -1]
        contamination_auc = float(roc_auc_score(contamination_binary, contamination_score))

    metrics = {
        "label_source": label_source,
        "config_hash": compute_config_hash(config),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "primary_calibrated": bool(isinstance(primary_model, CalibratedClassifierCV)),
        "primary_class_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "distress_recall": float(recall_score(distress_mask, distress_pred_mask, zero_division=0)),
        "terminal_loss_precision": float(precision_score(terminal_true, terminal_pred, zero_division=0)),
        "contamination_auroc": contamination_auc,
        "primary_classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        "primary_confusion_matrix": confusion_matrix(y_true, y_pred, labels=class_order).tolist(),
        "primary_confusion_labels": class_order,
        "rare_tag_metrics": rare_tag_rows,
        "distress_metrics": distress_metrics,
        "actionability_metrics": actionability_metrics,
        "contamination_metrics": contamination_metrics,
    }

    return {
        "feature_bundle": bundle,
        "primary_model": primary_model,
        "tag_bundle": tag_bundle,
        "distress_model": distress_model,
        "actionability_model": actionability_model,
        "contamination_model": contamination_model,
        "metrics": metrics,
        "class_order": class_order,
    }


def predict_remarks_annotations(events_df: pd.DataFrame, models: dict[str, Any], config: WSLReliabilityConfig) -> pd.DataFrame:
    df = build_remarks_canonical_text(events_df)
    df = _apply_rule_taxonomy(df)
    feature_bundle: RemarksFeatureBundle = models["feature_bundle"]
    chunk_size = 50000
    predicted_frames: list[pd.DataFrame] = []
    for start in range(0, len(df), chunk_size):
        stop = min(len(df), start + chunk_size)
        chunk = df.iloc[start:stop].copy()
        X_chunk = _transform_features(feature_bundle, chunk)
        primary_model = models["primary_model"]
        primary_probs = primary_model.predict_proba(X_chunk)
        primary_pred = primary_model.predict(X_chunk)
        probability_lookup = {
            cls: primary_probs[:, idx]
            for idx, cls in enumerate(primary_model.classes_)
        }
        chunk["model_primary_class"] = primary_pred
        chunk["model_primary_class_probability"] = primary_probs.max(axis=1)
        for cls, values in probability_lookup.items():
            chunk[f"p_primary__{cls}"] = values

        final_primary: list[str] = []
        for row in chunk.itertuples(index=False):
            if row.rule_primary_class in {"terminal_loss", "distress_hazard"} and row.rule_confidence >= 0.85:
                final_primary.append(row.rule_primary_class)
            elif row.rule_primary_class in {"interruption_repair", "homebound_or_termination"} and row.rule_confidence >= 0.80:
                final_primary.append(row.rule_primary_class)
            elif row.model_primary_class_probability < 0.45 and row.rule_confidence >= 0.70:
                final_primary.append(row.rule_primary_class)
            else:
                final_primary.append(row.model_primary_class)
        chunk["primary_class"] = final_primary

        tag_bundle = models["tag_bundle"]
        if tag_bundle["model"] is not None:
            tag_probs = tag_bundle["model"].predict_proba(X_chunk)
            tag_lists: list[list[str]] = []
            for row_index, row in enumerate(chunk.itertuples(index=False)):
                predicted_tags = {
                    tag
                    for j, tag in enumerate(tag_bundle["supported_tags"])
                    if tag_probs[row_index, j] >= tag_bundle["thresholds"][tag]
                }
                combined = sorted(set(row.rule_secondary_tags) | predicted_tags, key=SECONDARY_TAGS.index)
                tag_lists.append(combined)
                for j, tag in enumerate(tag_bundle["supported_tags"]):
                    chunk.loc[chunk.index[row_index], f"p_tag__{tag}"] = tag_probs[row_index, j]
            chunk["secondary_tags"] = tag_lists
        else:
            chunk["secondary_tags"] = chunk["rule_secondary_tags"]

        distress_probs = models["distress_model"].predict_proba(X_chunk)
        actionability_probs = models["actionability_model"].predict_proba(X_chunk)
        contamination_probs = models["contamination_model"].predict_proba(X_chunk)
        distress_expected = (distress_probs * models["distress_model"].classes_).sum(axis=1)
        actionability_expected = (actionability_probs * models["actionability_model"].classes_).sum(axis=1)
        contamination_expected = (contamination_probs * models["contamination_model"].classes_).sum(axis=1)
        chunk["distress_severity_0_4"] = np.maximum(
            np.rint(distress_expected).astype(int),
            chunk["rule_distress_severity_0_4"].astype(int),
        )
        chunk["actionability_0_3"] = np.maximum(
            np.rint(actionability_expected).astype(int),
            chunk["rule_actionability_0_3"].astype(int),
        )
        chunk["contamination_score_0_3"] = np.maximum(
            np.rint(contamination_expected).astype(int),
            chunk["rule_contamination_score_0_3"].astype(int),
        )
        chunk["productivity_polarity_m2_p2"] = chunk["rule_productivity_polarity_m2_p2"].astype(int)
        chunk["source_type"] = chunk["rule_source_type"]
        chunk["reason_codes"] = chunk["rule_reason_codes"]
        chunk["remarks_annotation_confidence"] = np.maximum(
            chunk["model_primary_class_probability"],
            chunk["rule_confidence"],
        )
        predicted_frames.append(chunk)
    return pd.concat(predicted_frames).reset_index(drop=True)


def export_remarks_taxonomy_outputs(
    output_dir: Path,
    gold_sample_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    models: dict[str, Any],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "gold_sample": output_dir / "remarks_gold_sample.parquet",
        "labelbook": output_dir / "remarks_labelbook.json",
        "metrics": output_dir / "remarks_model_metrics.json",
        "predictions": output_dir / "remarks_predictions.parquet",
        "error_audit": output_dir / "remarks_error_audit.csv",
        "confusion_matrix": output_dir / "remarks_primary_class_confusion_matrix.csv",
        "rare_tags": output_dir / "remarks_rare_tag_precision_recall.csv",
    }
    save_dataframe(gold_sample_df, paths["gold_sample"])
    save_dataframe(predictions_df, paths["predictions"])
    labelbook_payload = {
        "primary_class": PRIMARY_CLASSES,
        "secondary_tags": SECONDARY_TAGS,
        "ordinal_scores": {
            "distress_severity_0_4": {"0": "none", "1": "mild", "2": "clear adverse", "3": "serious danger", "4": "terminal"},
            "productivity_polarity_m2_p2": {"-2": "strong negative", "-1": "weak negative", "0": "neutral", "1": "positive", "2": "very strong"},
            "actionability_0_3": {"0": "descriptive only", "1": "background", "2": "decision relevant", "3": "immediately decision relevant"},
            "contamination_score_0_3": {"0": "clean", "1": "minor", "2": "substantial", "3": "largely unusable"},
        },
    }
    write_json(paths["labelbook"], labelbook_payload)
    write_json(paths["metrics"], models["metrics"])

    confusion = pd.DataFrame(
        models["metrics"]["primary_confusion_matrix"],
        index=models["metrics"]["primary_confusion_labels"],
        columns=models["metrics"]["primary_confusion_labels"],
    )
    confusion.to_csv(paths["confusion_matrix"])
    pd.DataFrame(models["metrics"]["rare_tag_metrics"]).to_csv(paths["rare_tags"], index=False)

    audit = predictions_df[
        [
            "event_row_id",
            "issue_date",
            "page_key",
            "event_type",
            "remarks_canonical_text",
            "primary_class",
            "rule_primary_class",
            "model_primary_class",
            "remarks_annotation_confidence",
            "secondary_tags",
            "contamination_score_0_3",
        ]
    ].copy()
    audit["secondary_tags"] = audit["secondary_tags"].map(lambda tags: "; ".join(tags) if isinstance(tags, list) else "")
    audit = pd.concat(
        [
            audit.sort_values(["contamination_score_0_3", "remarks_annotation_confidence"], ascending=[False, False]).head(200),
            audit[audit["primary_class"].isin(["distress_hazard", "terminal_loss"])].head(200),
            audit[audit["primary_class"] == "routine_info"].sample(min(200, int((audit["primary_class"] == "routine_info").sum())), random_state=42),
        ]
    ).drop_duplicates("event_row_id")
    audit.to_csv(paths["error_audit"], index=False)
    return paths
