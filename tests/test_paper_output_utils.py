from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paper.utils.io import write_md_preview
from src.paper.utils.latex import write_simple_table_tex


def test_write_md_preview_escapes_markdown_cells(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "preview.md"
    rows = pd.DataFrame(
        [
            {
                "label|name": "Alpha|Beta",
                "notes": "line one\nline two",
                "metadata": {"route": "North|Pacific", "season": "Spring"},
            }
        ]
    )

    write_md_preview(path, rows, "Preview")

    preview = path.read_text(encoding="utf-8")
    assert r"label\|name" in preview
    assert r"Alpha\|Beta" in preview
    assert "line one<br>line two" in preview
    assert r"route=North\|Pacific; season=Spring" in preview


def test_write_simple_table_tex_escapes_latex_sensitive_characters(tmp_path: Path) -> None:
    path = tmp_path / "latex" / "table.tex"
    rows = [
        {
            "row_label": "Cost_1",
            "summary": "50% & rising #1",
            "formula": "x_{t} + y^2 ~ $100",
        }
    ]

    write_simple_table_tex(path, "Budget {Draft}", rows)

    tex = path.read_text(encoding="utf-8")
    assert r"\section*{Budget \{Draft\}}" in tex
    assert r"Cost\_1" in tex
    assert r"50\% \& rising \#1" in tex
    assert r"x\_\{t\} + y\textasciicircum{}2 \textasciitilde{} \$100" in tex
