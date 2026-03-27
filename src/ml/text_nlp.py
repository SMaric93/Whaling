"""
ML Layer — Appendix ML-9: Text / NLP Analysis.

Analyze instruction text, logbook remarks, and other textual data.

Only runs if text data exists (conditional on build_text_dataset).

Methods:
- TF-IDF vectorization
- Topic modeling (LDA / NMF)
- Simple keyword extraction
- Link topics to organizational capability (psi)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.ml.config import ML_CFG, ML_TABLES_DIR, ML_FIGURES_DIR

logger = logging.getLogger(__name__)


def analyze_text(
    *,
    n_topics: int = 5,
    max_features: int = 1000,
    save_outputs: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Analyze text data from logbook remarks / instructions.

    Returns None if no text data is available.
    """
    t0 = time.time()
    logger.info("Attempting text/NLP analysis...")

    from src.ml.build_text_dataset import build_text_dataset
    text_df = build_text_dataset()

    if text_df is None or len(text_df) == 0:
        logger.info("No text data available; skipping NLP analysis")
        return None

    logger.info("Text dataset: %d documents", len(text_df))

    # ── TF-IDF ──────────────────────────────────────────────────────
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = text_df["cleaned_text"].fillna("").values
    tfidf = TfidfVectorizer(
        max_features=max_features,
        min_df=5,
        max_df=0.95,
        stop_words="english",
    )
    tfidf_matrix = tfidf.fit_transform(texts)
    vocab = tfidf.get_feature_names_out()

    logger.info("TF-IDF: %d documents × %d features", *tfidf_matrix.shape)

    # ── Topic modeling ──────────────────────────────────────────────
    try:
        from sklearn.decomposition import NMF
        nmf = NMF(n_components=n_topics, random_state=ML_CFG.random_seed, max_iter=300)
        topic_weights = nmf.fit_transform(tfidf_matrix)
        topic_term_matrix = nmf.components_

        # Extract top words per topic
        topics = {}
        for i in range(n_topics):
            top_idx = topic_term_matrix[i].argsort()[-10:][::-1]
            top_words = [vocab[j] for j in top_idx]
            topics[f"topic_{i}"] = top_words

        text_df["dominant_topic"] = topic_weights.argmax(axis=1)
        for i in range(n_topics):
            text_df[f"topic_{i}_weight"] = topic_weights[:, i]
    except Exception as e:
        logger.warning("Topic modeling failed: %s", e)
        topics = {}

    # ── Link to psi ─────────────────────────────────────────────────
    topic_psi = None
    if "agent_id" in text_df.columns and "dominant_topic" in text_df.columns:
        from src.ml.build_outcome_ml_dataset import build_outcome_ml_dataset
        outcomes = build_outcome_ml_dataset()

        if "psi_hat_holdout" in outcomes.columns:
            agent_psi = outcomes.groupby("agent_id")["psi_hat_holdout"].mean()
            text_df = text_df.merge(
                agent_psi.rename("psi_hat_holdout"),
                on="agent_id", how="left",
            )

            if "psi_hat_holdout" in text_df.columns:
                topic_psi = text_df.groupby("dominant_topic")["psi_hat_holdout"].agg(
                    ["mean", "std", "count"]
                )
                logger.info("Topic-psi linkage:\n%s", topic_psi)

    # ── Keyword extraction ──────────────────────────────────────────
    # Most distinctive words by psi group
    keywords_by_psi = {}
    if "psi_hat_holdout" in text_df.columns:
        psi_med = text_df["psi_hat_holdout"].median()
        for label, mask in [("high_psi", text_df["psi_hat_holdout"] > psi_med),
                           ("low_psi", text_df["psi_hat_holdout"] <= psi_med)]:
            sub_texts = text_df.loc[mask, "cleaned_text"]
            if len(sub_texts) > 10:
                sub_tfidf = tfidf.transform(sub_texts)
                mean_tfidf = np.asarray(sub_tfidf.mean(axis=0)).flatten()
                top_idx = mean_tfidf.argsort()[-20:][::-1]
                keywords_by_psi[label] = [vocab[j] for j in top_idx]

    # ── Save ────────────────────────────────────────────────────────
    if save_outputs:
        if topics:
            pd.DataFrame(topics).to_csv(ML_TABLES_DIR / "text_topics.csv", index=False)
        if topic_psi is not None:
            topic_psi.to_csv(ML_TABLES_DIR / "topic_psi_linkage.csv")
        if keywords_by_psi:
            pd.DataFrame(dict([(k, pd.Series(v)) for k, v in keywords_by_psi.items()])).to_csv(
                ML_TABLES_DIR / "keywords_by_psi.csv", index=False
            )

    elapsed = time.time() - t0
    logger.info("Text analysis complete in %.1fs", elapsed)

    return {
        "n_documents": len(text_df),
        "n_topics": n_topics,
        "topics": topics,
        "topic_psi": topic_psi,
        "keywords_by_psi": keywords_by_psi,
    }
