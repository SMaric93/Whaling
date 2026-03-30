from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import BuildContext
from .utils.footnotes import standard_footnote
from .utils.io import write_csv, write_md_preview
from .utils.latex import write_simple_table_tex


def save_table_outputs(
    *,
    name: str,
    frame: pd.DataFrame,
    out_dir: Path,
    context: BuildContext,
    memo: str,
    title: str | None = None,
) -> dict[str, Any]:
    csv_path = out_dir / f"{name}.csv"
    tex_path = context.outputs / "latex" / f"{name}.tex"
    md_path = out_dir / f"{name}.md"
    memo_path = context.outputs / "memos" / f"{name}.md"

    write_csv(csv_path, frame)
    write_simple_table_tex(tex_path, title or name, frame)
    write_md_preview(md_path, frame, title or name)
    memo_path.write_text(memo.rstrip() + "\n", encoding="utf-8")

    return {
        "name": name,
        "csv": str(csv_path),
        "tex": str(tex_path),
        "md": str(md_path),
        "memo": str(memo_path),
    }


def build_named_table(
    name: str,
    out_dir: Path,
    context: BuildContext,
    supporting_modules: list[str],
    sample: str,
    supports: str = "main",
) -> dict[str, Any]:
    if supports == "appendix":
        from .appendix.real_builders import build_appendix_table

        return build_appendix_table(name, context)

    rows = []
    for idx, module in enumerate(supporting_modules, start=1):
        rows.append(
            {
                "row": idx,
                "module": module,
                "status": "registered",
                "note": f"Mapped from existing repository module {module}",
                "notation": "theta_hat / psi_hat",
                "supports": supports,
            }
        )
    if not rows:
        rows = [
            {
                "row": 1,
                "module": "n/a",
                "status": "pending",
                "note": "No legacy mapping found",
                "notation": "theta_hat / psi_hat",
                "supports": supports,
            }
        ]

    memo = standard_footnote(
        sample=sample,
        unit="voyage or voyage-day depending on mapped module",
        types_note="theta_hat/psi_hat names standardized from alpha_hat/gamma_hat where applicable",
        fe="captain FE default unless module-specific",
        cluster="captain default with agent-cluster sensitivity",
        controls="environment and lagged quality controls per module",
        interpretation="Table consolidates evidence from mapped modules.",
        caution="Rows are integration-layer summaries; inspect source modules for full estimating equations.",
    )
    return save_table_outputs(
        name=name,
        frame=pd.DataFrame(rows),
        out_dir=out_dir,
        context=context,
        memo=f"# {name}\n\n{memo}",
        title=name,
    )
