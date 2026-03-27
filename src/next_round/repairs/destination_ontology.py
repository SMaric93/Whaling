"""
Repair 3: Destination Ontology Builder.

Canonicalizes the 500+ ground_or_route labels into a hierarchical ontology:
  basin → theater → major_ground → local_ground

Handles spelling variants, punctuation, hybrids, singletons.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.next_round.config import DATA_FINAL, DATA_DERIVED, DOCS_DIR

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Basin / Theater Classification Rules
# ═══════════════════════════════════════════════════════════════════════════

BASIN_RULES = [
    # (pattern, basin, theater)
    (r"\bpacific\b", "Pacific", "Pacific"),
    (r"\bn\s*pacific\b", "Pacific", "North Pacific"),
    (r"\bs\s*pacific\b", "Pacific", "South Pacific"),
    (r"\bindian\b", "Indian", "Indian Ocean"),
    (r"\batlantic\b", "Atlantic", "Atlantic"),
    (r"\bn\s*atlantic\b", "Atlantic", "North Atlantic"),
    (r"\bs\s*atlantic\b", "Atlantic", "South Atlantic"),
    (r"\barctic\b", "Arctic", "Arctic"),
    (r"\bbering\b|behring", "Arctic", "Bering Sea"),
    (r"\bjapan\b", "Pacific", "Japan Grounds"),
    (r"\bnw\s*coast\b|northwest\s*coast", "Pacific", "NW Coast"),
    (r"\bkodiak\b|kamchatka\b", "Pacific", "North Pacific"),
    (r"\bhudson\b", "Arctic", "Hudson Bay"),
    (r"\bgulf\s*of\s*mexico\b", "Atlantic", "Gulf of Mexico"),
    (r"\bcarib\b|caribbean\b|w\s*ind", "Atlantic", "Caribbean/W Indies"),
    (r"\bcape\s*verde\b|de\s*verde", "Atlantic", "Cape Verde"),
    (r"\bwestern\s*islands?\b|azores\b|fayal\b", "Atlantic", "Western Islands/Azores"),
    (r"\bdesolation\b", "Indian", "Desolation/Kerguelen"),
    (r"\bnew\s*zealand\b|n\s*z\b", "Pacific", "New Zealand"),
    (r"\bafrica\b", "Indian", "Africa/Madagascar"),
    (r"\bmadagascar\b", "Indian", "Africa/Madagascar"),
    (r"\bbrazil\b", "Atlantic", "Brazil"),
    (r"\bhawaii\b|sandwich\s*islands?\b|honolulu\b", "Pacific", "Hawaii"),
    (r"\bchile\b|peru\b", "Pacific", "South America Pacific"),
    (r"\bpatagonia\b|falkland\b", "Atlantic", "Patagonia/Falklands"),
    (r"\bochotsk\b|okhotsk\b", "Pacific", "Sea of Okhotsk"),
    (r"\bbaja\b|california\b", "Pacific", "Baja/California"),
]

MAJOR_GROUND_PATTERNS = [
    (r"wh\s*gr\s*90|whale\s*ground\s*90", "Whale Ground 90"),
    (r"\bline\b", "On the Line (Equatorial)"),
    (r"\boff\s*shore\b", "Offshore Ground"),
    (r"\bcoast\s*of\s*(?:japan|nippon)", "Coast of Japan"),
    (r"\bsperm\b", "Sperm Whale Ground"),
    (r"\bright\b", "Right Whale Ground"),
    (r"\bbowhead\b", "Bowhead Ground"),
]


def build_destination_ontology(*, save: bool = True) -> pd.DataFrame:
    """
    Build a hierarchical destination ontology from raw ground_or_route labels.

    Returns DataFrame with columns:
        ground_or_route, ground_clean, basin, theater, major_ground, local_ground
    """
    logger.info("=" * 60)
    logger.info("Repair 3: Building Destination Ontology")
    logger.info("=" * 60)

    # Load voyage data
    df = pd.read_parquet(DATA_FINAL / "analysis_voyage_augmented.parquet")
    raw_labels = df["ground_or_route"].dropna().unique()
    logger.info("Found %d unique ground_or_route labels", len(raw_labels))

    rows = []
    for label in sorted(raw_labels):
        clean = _clean_label(label)
        basin, theater = _classify_basin_theater(clean)
        major = _classify_major_ground(clean)
        local = clean if clean != major else clean

        rows.append({
            "ground_or_route": label,
            "ground_clean": clean,
            "basin": basin,
            "theater": theater,
            "major_ground": major,
            "local_ground": local,
        })

    ontology = pd.DataFrame(rows)

    # ── Handle singletons ─────────────────────────────────────────────
    # Count voyages per label
    voyage_counts = df["ground_or_route"].value_counts()
    ontology = ontology.merge(
        voyage_counts.rename("n_voyages").reset_index().rename(columns={"index": "ground_or_route"}),
        on="ground_or_route", how="left",
    )
    ontology["n_voyages"] = ontology["n_voyages"].fillna(0).astype(int)

    # Mark singletons
    ontology["is_singleton"] = ontology["n_voyages"] <= 2

    # Collapse rare labels to theater-level
    ontology["ground_for_model"] = np.where(
        ontology["is_singleton"],
        ontology["theater"],
        ontology["ground_clean"],
    )

    # ── Summary statistics ────────────────────────────────────────────
    n_basins = ontology["basin"].nunique()
    n_theaters = ontology["theater"].nunique()
    n_majors = ontology["major_ground"].nunique()
    n_singletons = ontology["is_singleton"].sum()

    logger.info("Ontology: %d basins, %d theaters, %d major grounds, %d singletons",
                n_basins, n_theaters, n_majors, n_singletons)

    if save:
        out_path = DATA_DERIVED / "destination_ontology.parquet"
        ontology.to_parquet(out_path, index=False)
        logger.info("Saved %s", out_path)

        # Also save CSV for human review
        ontology.to_csv(DATA_DERIVED / "destination_ontology.csv", index=False)

        _save_memo(ontology)

    return ontology


def _clean_label(label: str) -> str:
    """Normalize a ground_or_route label."""
    s = str(label).strip().upper()
    # Normalize whitespace
    s = re.sub(r"\s+", " ", s)
    # Remove trailing/leading punctuation
    s = s.strip(" ,-.")
    # Normalize common abbreviations
    s = re.sub(r"\bO\b$", "OCEAN", s)
    s = re.sub(r"\bO\b(?=\s*,)", "OCEAN", s)
    s = re.sub(r"\bS\b(?=\s)", "SEA", s)
    s = re.sub(r"\bGR\b", "GROUND", s)
    s = re.sub(r"\bIND\b", "INDIAN", s)
    s = re.sub(r"\bATL\b", "ATLANTIC", s)
    s = re.sub(r"\bPAC\b", "PACIFIC", s)
    return s


def _classify_basin_theater(label: str) -> Tuple[str, str]:
    """Classify a cleaned label into basin and theater."""
    label_lower = label.lower()
    matches = []
    for pattern, basin, theater in BASIN_RULES:
        if re.search(pattern, label_lower):
            matches.append((basin, theater))

    if not matches:
        return ("Unknown", "Unknown")

    # If multiple basins, mark as multi-ocean
    basins = set(m[0] for m in matches)
    if len(basins) > 1:
        return ("Multi-Ocean", " + ".join(sorted(set(m[1] for m in matches))))

    return matches[0]


def _classify_major_ground(label: str) -> str:
    """Classify into a major ground if pattern matches."""
    label_lower = label.lower()
    for pattern, ground in MAJOR_GROUND_PATTERNS:
        if re.search(pattern, label_lower):
            return ground
    return label


def _save_memo(ontology: pd.DataFrame):
    """Save the ontology documentation."""
    memo_path = DOCS_DIR / "destination_ontology.md"

    basin_summary = (
        ontology.groupby("basin")
        .agg(n_labels=("ground_or_route", "count"),
             n_voyages=("n_voyages", "sum"),
             n_theaters=("theater", "nunique"))
        .sort_values("n_voyages", ascending=False)
    )

    theater_summary = (
        ontology.groupby(["basin", "theater"])
        .agg(n_labels=("ground_or_route", "count"),
             n_voyages=("n_voyages", "sum"))
        .sort_values("n_voyages", ascending=False)
        .head(20)
    )

    lines = [
        "# Destination Ontology",
        "",
        "## Overview",
        f"- **{len(ontology)}** unique ground/route labels",
        f"- **{ontology['basin'].nunique()}** basins",
        f"- **{ontology['theater'].nunique()}** theaters",
        f"- **{ontology['is_singleton'].sum()}** singletons (≤2 voyages) collapsed to theater level",
        "",
        "## Basin Summary",
        "",
        basin_summary.to_markdown(),
        "",
        "## Top 20 Theaters",
        "",
        theater_summary.to_markdown(),
        "",
        "## Hierarchy",
        "```",
        "basin → theater → major_ground → local_ground",
        "```",
        "",
        "Singletons are collapsed: `ground_for_model` = theater for rare labels.",
    ]

    memo_path.write_text("\n".join(lines))
    logger.info("Saved %s", memo_path)
