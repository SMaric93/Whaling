"""
Standardised tidy coefficient table builder (Step 9).

Every regression result flows through ``build_tidy_row`` to produce the
schema required by the revision.  Helper ``save_result_table`` writes
CSV + metadata sidecar.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import TABLES_DIR


# ── Required columns ────────────────────────────────────────────────────
TIDY_COLUMNS = [
    "outcome",
    "sample_name",
    "spec_name",
    "term",
    "estimate",
    "std_error",
    "test_stat",
    "p_value",
    "ci_low",
    "ci_high",
    "n_obs",
    "n_captains",
    "n_agents",
    "fixed_effects",
    "cluster_scheme",
    "effect_object_used",
    "notes",
]


def build_tidy_row(
    *,
    outcome: str,
    sample_name: str,
    spec_name: str,
    term: str,
    estimate: float,
    std_error: float,
    n_obs: int,
    n_captains: int = 0,
    n_agents: int = 0,
    fixed_effects: str = "",
    cluster_scheme: str = "captain_id",
    effect_object_used: str = "",
    notes: str = "",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Return one row of the tidy coefficient schema."""
    z = 1.96 if alpha == 0.05 else abs(float(__import__("scipy").stats.norm.ppf(alpha / 2)))
    t_stat = estimate / std_error if std_error > 0 else np.nan
    p_val = float(2 * (1 - __import__("scipy").stats.norm.cdf(abs(t_stat)))) if not np.isnan(t_stat) else np.nan
    return {
        "outcome": outcome,
        "sample_name": sample_name,
        "spec_name": spec_name,
        "term": term,
        "estimate": estimate,
        "std_error": std_error,
        "test_stat": t_stat,
        "p_value": p_val,
        "ci_low": estimate - z * std_error,
        "ci_high": estimate + z * std_error,
        "n_obs": n_obs,
        "n_captains": n_captains,
        "n_agents": n_agents,
        "fixed_effects": fixed_effects,
        "cluster_scheme": cluster_scheme,
        "effect_object_used": effect_object_used,
        "notes": notes,
    }


def rows_to_tidy_df(rows: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """Convert a list of tidy-row dicts to a DataFrame with canonical column order."""
    df = pd.DataFrame(rows)
    for col in TIDY_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[TIDY_COLUMNS]


def save_result_table(
    df: pd.DataFrame,
    name: str,
    *,
    output_dir: Optional[Path] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save a tidy result table to CSV and a companion _meta.json."""
    import json

    if output_dir is None:
        output_dir = TABLES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{name}.csv"
    df.to_csv(csv_path, index=False)

    if metadata:
        meta_path = output_dir / f"{name}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    return csv_path


def save_markdown_table(df: pd.DataFrame, name: str, *, output_dir: Optional[Path] = None) -> Path:
    """Save a DataFrame as a markdown pipe-table."""
    if output_dir is None:
        output_dir = TABLES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / f"{name}.md"

    cols = list(df.columns)
    lines = [
        "| " + " | ".join(str(c) for c in cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")

    md_path.write_text("\n".join(lines) + "\n")
    return md_path
