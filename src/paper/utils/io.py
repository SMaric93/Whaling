from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable
import csv

import pandas as pd


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def _rows_from_input(rows: Iterable[dict] | pd.DataFrame) -> list[dict]:
    if isinstance(rows, pd.DataFrame):
        if rows.empty:
            return []
        return rows.to_dict(orient="records")
    return list(rows)


def _format_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return "; ".join(_format_value(item) for item in value if _format_value(item))
    if isinstance(value, dict):
        return "; ".join(f"{key}={_format_value(val)}" for key, val in value.items() if _format_value(val))
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    except ValueError:
        pass

    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        if value.is_integer() and abs(value) >= 1000:
            return f"{value:,.0f}"
        if abs(value) >= 100:
            return f"{value:,.2f}"
        if abs(value) >= 1:
            return f"{value:.3f}"
        return f"{value:.4f}"
    return str(value)


def write_csv(path: Path, rows: Iterable[dict] | pd.DataFrame) -> None:
    rows = _rows_from_input(rows)
    if not rows:
        rows = [{"note": "no rows generated"}]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_md_preview(path: Path, rows: list[dict] | pd.DataFrame, title: str) -> None:
    rows = _rows_from_input(rows)
    if not rows:
        rows = [{"note": "no rows generated"}]
    cols = list(rows[0].keys())
    lines = [f"# {title}", "", "| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows[:30]:
        lines.append("| " + " | ".join(_format_value(r.get(c, "")) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
