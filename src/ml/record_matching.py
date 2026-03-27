"""
Shared ML helpers for probabilistic record matching.

These utilities are intentionally lightweight so the rule-heavy ETL and
crosswalk modules can add learned ranking without fully rewriting their
domain-specific blocking logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from config import ML_SHIFT_CONFIG

try:
    from sklearn.ensemble import RandomForestClassifier
except ImportError:  # pragma: no cover - dependency handled by runtime env
    RandomForestClassifier = None

try:
    from parsing.string_normalizer import jaro_winkler_similarity
except ImportError:  # pragma: no cover
    def jaro_winkler_similarity(s1: str, s2: str) -> float:
        if s1 == s2:
            return 1.0
        return 0.0


@dataclass
class MatchModelBundle:
    """Container for a fitted probabilistic match model."""

    trained: bool
    feature_names: list[str]
    model: Optional[object] = None
    training_rows: int = 0
    positive_rows: int = 0
    negative_rows: int = 0
    notes: str = ""


def _safe_text(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def _token_set(text: str) -> set[str]:
    return {tok for tok in _safe_text(text).upper().split() if tok}


def compute_text_pair_features(left, right, prefix: str = "") -> dict[str, float]:
    """Compute a compact set of text-similarity features."""
    left_text = _safe_text(left).upper()
    right_text = _safe_text(right).upper()
    left_tokens = _token_set(left_text)
    right_tokens = _token_set(right_text)
    token_union = left_tokens | right_tokens
    shared = left_tokens & right_tokens

    if left_text and right_text:
        jw = float(jaro_winkler_similarity(left_text, right_text))
        contains = float(left_text in right_text or right_text in left_text)
        prefix_match = float(left_text[:3] == right_text[:3]) if len(left_text) >= 3 and len(right_text) >= 3 else 0.0
        length_ratio = min(len(left_text), len(right_text)) / max(len(left_text), len(right_text))
    else:
        jw = 0.0
        contains = 0.0
        prefix_match = 0.0
        length_ratio = 0.0

    return {
        f"{prefix}jw": jw,
        f"{prefix}exact": float(left_text == right_text and bool(left_text)),
        f"{prefix}contains": contains,
        f"{prefix}prefix3_match": prefix_match,
        f"{prefix}length_ratio": float(length_ratio),
        f"{prefix}token_jaccard": float(len(shared) / len(token_union)) if token_union else 0.0,
        f"{prefix}shared_tokens": float(len(shared)),
    }


def compute_numeric_distance_features(
    left,
    right,
    *,
    scale: float = 1.0,
    prefix: str = "",
) -> dict[str, float]:
    """Distance-based numeric similarity features with graceful NA handling."""
    if left is None or right is None or pd.isna(left) or pd.isna(right):
        return {
            f"{prefix}missing": 1.0,
            f"{prefix}distance": float(scale),
            f"{prefix}similarity": 0.5,
        }

    distance = abs(float(left) - float(right))
    denom = max(float(scale), 1.0)
    similarity = max(0.0, 1.0 - (distance / denom))
    return {
        f"{prefix}missing": 0.0,
        f"{prefix}distance": float(distance),
        f"{prefix}similarity": float(similarity),
    }


def fit_match_probability_model(
    features_df: pd.DataFrame,
    positive_mask: Iterable[bool],
    negative_mask: Iterable[bool],
    *,
    random_state: Optional[int] = None,
    min_training_rows: Optional[int] = None,
) -> MatchModelBundle:
    """
    Fit a RandomForest matcher from pseudo-labeled positives and negatives.

    Returns an untrained bundle if dependencies or class balance are insufficient.
    """
    min_rows = min_training_rows or ML_SHIFT_CONFIG.min_training_rows
    seed = ML_SHIFT_CONFIG.random_state if random_state is None else random_state

    if RandomForestClassifier is None or len(features_df) == 0:
        return MatchModelBundle(
            trained=False,
            feature_names=list(features_df.columns),
            notes="sklearn_unavailable_or_empty",
        )

    clean = features_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    pos = pd.Series(list(positive_mask), index=clean.index).fillna(False).astype(bool)
    neg = pd.Series(list(negative_mask), index=clean.index).fillna(False).astype(bool)
    labeled = clean[pos | neg].copy()

    if len(labeled) < min_rows or pos.sum() == 0 or neg.sum() == 0:
        return MatchModelBundle(
            trained=False,
            feature_names=list(clean.columns),
            training_rows=int(len(labeled)),
            positive_rows=int(pos.sum()),
            negative_rows=int(neg.sum()),
            notes="insufficient_pseudo_labels",
        )

    y = pos.loc[labeled.index].astype(int).to_numpy()
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=4,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(labeled, y)

    return MatchModelBundle(
        trained=True,
        model=model,
        feature_names=list(clean.columns),
        training_rows=int(len(labeled)),
        positive_rows=int(pos.sum()),
        negative_rows=int(neg.sum()),
        notes="trained",
    )


def score_match_probability(
    bundle: MatchModelBundle,
    features_df: pd.DataFrame,
    *,
    fallback_scores: Optional[Iterable[float]] = None,
) -> np.ndarray:
    """Score candidate pairs with the fitted model or fall back gracefully."""
    if len(features_df) == 0:
        return np.array([], dtype=float)

    if bundle.trained and bundle.model is not None:
        clean = (
            features_df[bundle.feature_names]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        return bundle.model.predict_proba(clean)[:, 1]

    if fallback_scores is None:
        return np.full(len(features_df), 0.5, dtype=float)

    scores = np.asarray(list(fallback_scores), dtype=float)
    return np.clip(scores, 0.0, 1.0)


class _UnionFind:
    def __init__(self, items: Iterable[str]):
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        root = self.parent.setdefault(item, item)
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            nxt = self.parent[item]
            self.parent[item] = root
            item = nxt
        return root

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        canonical = min(left_root, right_root)
        other = right_root if canonical == left_root else left_root
        self.parent[other] = canonical


def cluster_match_pairs(
    pairs_df: pd.DataFrame,
    *,
    left_col: str,
    right_col: str,
    probability_col: str,
    threshold: float,
) -> dict[str, str]:
    """Cluster ids connected by pairwise probabilities above threshold."""
    if len(pairs_df) == 0:
        return {}

    items = set(pairs_df[left_col].astype(str)) | set(pairs_df[right_col].astype(str))
    uf = _UnionFind(items)

    keep = pairs_df[pairs_df[probability_col] >= threshold]
    for _, row in keep.iterrows():
        uf.union(str(row[left_col]), str(row[right_col]))

    return {item: uf.find(item) for item in items}
