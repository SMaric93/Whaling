"""
ML Layer — Text Dataset Builder.

Only runs if text data (logbook remarks, instructions, letters) are available.

Scans for text columns in the logbook data and constructs a text dataset
linked to voyages, captains, and agents.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Optional

import numpy as np
import pandas as pd

from src.ml.config import ML_CFG, ML_DATA_DIR

logger = logging.getLogger(__name__)

OUTPUT_PATH = ML_DATA_DIR / "text_dataset.parquet"


def build_text_dataset(
    *,
    force_rebuild: bool = False,
    save: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Build text dataset from logbook remarks and any other text sources.

    Only creates output if text data actually exists.

    Returns
    -------
    pd.DataFrame or None
        Text dataset with document ID, cleaned text, and metadata links.
    """
    if OUTPUT_PATH.exists() and not force_rebuild:
        logger.info("Loading cached text dataset from %s", OUTPUT_PATH)
        return pd.read_parquet(OUTPUT_PATH)

    t0 = time.time()
    logger.info("Attempting to build text dataset...")

    from src.reinforcement.data_builder import load_logbook_positions

    try:
        positions = load_logbook_positions()
    except Exception as e:
        logger.warning("Cannot load logbook positions: %s", e)
        return None

    # ── Find text columns ───────────────────────────────────────────
    text_cols = []
    for col in ["remarks", "place", "text", "notes", "instructions", "log_entry"]:
        if col in positions.columns:
            n_nonempty = positions[col].dropna().str.strip().str.len().gt(0).sum()
            if n_nonempty > 100:
                text_cols.append(col)
                logger.info("Found text column '%s' with %d non-empty entries", col, n_nonempty)

    if not text_cols:
        logger.info("No substantive text data found; skipping text dataset")
        return None

    # ── Build text records ──────────────────────────────────────────
    records = []
    for col in text_cols:
        mask = positions[col].notna() & (positions[col].str.strip().str.len() > 0)
        df_text = positions.loc[mask].copy()

        id_cols = [c for c in ["voyage_id", "captain_id", "agent_id", "obs_date", "year"]
                   if c in df_text.columns]

        for _, row in df_text.iterrows():
            raw = str(row[col]).strip()
            if len(raw) < 3:
                continue
            record = {
                "document_id": f"{row.get('voyage_id', 'unk')}_{row.get('obs_date', 'unk')}_{col}",
                "source_type": col,
                "raw_text": raw,
                "cleaned_text": _clean_text(raw),
                "date": row.get("obs_date"),
            }
            for ic in id_cols:
                record[ic] = row.get(ic)
            records.append(record)

    if not records:
        logger.info("No usable text records after cleaning; skipping")
        return None

    result = pd.DataFrame(records)
    result = result.drop_duplicates(subset=["document_id"]).reset_index(drop=True)

    elapsed = time.time() - t0
    logger.info(
        "Text dataset built: %d documents from %d sources, %.1fs",
        len(result), len(text_cols), elapsed,
    )

    if save:
        result.to_parquet(OUTPUT_PATH, index=False)
        logger.info("Saved to %s", OUTPUT_PATH)

    return result


def _clean_text(text: str) -> str:
    """Basic text cleaning for logbook entries."""
    # Lowercase
    text = text.lower()
    # Remove excess whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove very short tokens
    tokens = text.split()
    tokens = [t for t in tokens if len(t) > 1]
    return " ".join(tokens)
