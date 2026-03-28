"""
Shared lightweight text classification helpers.

Used to shift regex/keyword-heavy ETL modules toward ML without requiring
hand-labeled corpora up front.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from config import ML_SHIFT_CONFIG

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
except ImportError:  # pragma: no cover
    TfidfVectorizer = None
    LogisticRegression = None
    Pipeline = None


@dataclass
class TextClassifierBundle:
    trained: bool
    classes_: list[str]
    model: Optional[object] = None
    training_rows: int = 0
    notes: str = ""


def normalize_texts(texts: Iterable[object]) -> list[str]:
    return [("" if pd.isna(text) else str(text).strip()) for text in texts]


def fit_text_classifier(
    texts: Iterable[object],
    labels: Iterable[object],
    *,
    min_training_rows: Optional[int] = None,
    random_state: Optional[int] = None,
) -> TextClassifierBundle:
    min_rows = min_training_rows or ML_SHIFT_CONFIG.min_text_training_rows
    seed = ML_SHIFT_CONFIG.random_state if random_state is None else random_state

    if Pipeline is None or TfidfVectorizer is None or LogisticRegression is None:
        return TextClassifierBundle(False, [], notes="sklearn_unavailable")

    text_values = normalize_texts(texts)
    label_values = pd.Series(list(labels))
    keep = label_values.notna()
    if keep.sum() < min_rows or label_values[keep].nunique() < 2:
        return TextClassifierBundle(
            False,
            sorted(label_values[keep].astype(str).unique().tolist()),
            training_rows=int(keep.sum()),
            notes="insufficient_training_rows",
        )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000)),
        ("clf", LogisticRegression(max_iter=1000, random_state=seed, n_jobs=-1)),
    ])
    model.fit(pd.Series(text_values)[keep], label_values[keep].astype(str))

    classes = list(getattr(model.named_steps["clf"], "classes_", []))
    return TextClassifierBundle(
        True,
        classes,
        model=model,
        training_rows=int(keep.sum()),
        notes="trained",
    )


def predict_text_probabilities(
    bundle: TextClassifierBundle,
    texts: Iterable[object],
) -> pd.DataFrame:
    text_values = normalize_texts(texts)
    if not bundle.trained or bundle.model is None or len(text_values) == 0:
        return pd.DataFrame(index=np.arange(len(text_values)))

    probs = bundle.model.predict_proba(text_values)
    return pd.DataFrame(
        probs,
        columns=[str(cls) for cls in bundle.classes_],
    )
