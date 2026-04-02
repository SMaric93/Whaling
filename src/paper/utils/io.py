from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable

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


def _is_blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _format_collection(items: Iterable[object]) -> str:
    rendered_items = [
        rendered
        for item in items
        if (rendered := _format_value(item))
    ]
    return "; ".join(rendered_items)


def _escape_markdown_cell(value: str) -> str:
    return value.replace("|", r"\|").replace("\n", "<br>")


def _format_value(value: object) -> str:
    if _is_blank(value):
        return ""
    if isinstance(value, (list, tuple, set)):
        return _format_collection(value)
    if isinstance(value, dict):
        return _format_collection(
            f"{key}={rendered}"
            for key, val in value.items()
            if (rendered := _format_value(val))
        )

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
    ensure_dirs(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_md_preview(path: Path, rows: list[dict] | pd.DataFrame, title: str) -> None:
    rows = _rows_from_input(rows)
    if not rows:
        rows = [{"note": "no rows generated"}]
    cols = list(rows[0].keys())
    lines = [
        f"# {title}",
        "",
        "| " + " | ".join(_escape_markdown_cell(col) for col in cols) + " |",
        "|" + "|".join([" --- "] * len(cols)) + "|",
    ]
    for r in rows[:30]:
        lines.append(
            "| "
            + " | ".join(
                _escape_markdown_cell(_format_value(r.get(c, "")))
                for c in cols
            )
            + " |"
        )
    ensure_dirs(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
