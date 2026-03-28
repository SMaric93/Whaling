from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


def _rows_from_input(rows: list[dict] | pd.DataFrame) -> list[dict]:
    if isinstance(rows, pd.DataFrame):
        if rows.empty:
            return []
        return rows.to_dict(orient="records")
    return list(rows)


def _escape_latex(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
    }
    for src, dest in replacements.items():
        text = text.replace(src, dest)
    return text


def _is_blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    try:
        return bool(pd.isna(value))
    except TypeError:
        pass
    except ValueError:
        return False
    if isinstance(value, str):
        return not value.strip()
    return False


def _format_value(value: object) -> str:
    if _is_blank(value):
        return ""
    if isinstance(value, (list, tuple, set)):
        return "; ".join(_format_value(item) for item in value if not _is_blank(item))
    if isinstance(value, dict):
        return "; ".join(f"{key}={_format_value(val)}" for key, val in value.items() if not _is_blank(val))
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        if value.is_integer():
            return f"{int(value):,}"
        magnitude = abs(value)
        if magnitude >= 1000:
            return f"{value:,.2f}"
        if magnitude >= 1:
            return f"{value:.3f}"
        if magnitude >= 0.001:
            return f"{value:.4f}"
        return f"{value:.2e}"
    return str(value)


def _column_alignment(df: pd.DataFrame, columns: list[str]) -> str:
    align = []
    for column in columns:
        if column == "row_label":
            align.append("l")
        elif pd.api.types.is_numeric_dtype(df[column]):
            align.append("r")
        else:
            align.append("l")
    return "".join(align) or "l"


def _render_tabular(df: pd.DataFrame, *, columns: list[str]) -> list[str]:
    align = _column_alignment(df, columns)
    lines = [
        f"\\begin{{tabular}}{{{align}}}",
        r"\hline",
        " & ".join(_escape_latex(column) for column in columns) + r" \\",
        r"\hline",
    ]
    for _, row in df[columns].head(100).iterrows():
        rendered = [_escape_latex(_format_value(row[column])) for column in columns]
        lines.append(" & ".join(rendered) + r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    return lines


def write_simple_table_tex(path: Path, title: str, rows: list[dict] | pd.DataFrame) -> None:
    frame = pd.DataFrame(_rows_from_input(rows))
    if frame.empty:
        frame = pd.DataFrame([{"note": "no rows generated"}])

    out = [
        r"\documentclass{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\begin{document}",
        f"\\section*{{{title}}}",
    ]

    if "panel" in frame.columns and frame["panel"].nunique(dropna=True) > 1:
        for panel in frame["panel"].dropna().unique():
            subset = frame[frame["panel"] == panel].copy()
            columns = [
                column
                for column in subset.columns
                if column not in {"panel", "note"} and subset[column].apply(lambda value: not _is_blank(value)).any()
            ]
            if not columns:
                columns = ["row_label"] if "row_label" in subset.columns else list(subset.columns[:1])
            out.append(f"\\subsection*{{{_escape_latex(panel)}}}")
            out.extend(_render_tabular(subset, columns=columns))
            out.append("")
    else:
        columns = [
            column
            for column in frame.columns
            if column != "note" and frame[column].apply(lambda value: not _is_blank(value)).any()
        ]
        if not columns:
            columns = list(frame.columns[:1])
        out.extend(_render_tabular(frame, columns=columns))

    out += [r"\end{document}"]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
