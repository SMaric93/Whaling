"""
I/O helpers: write tables to csv / tex / md, create output directories.
"""

from pathlib import Path
from typing import Optional
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "minor_revision"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
MEMOS_DIR = OUTPUT_DIR / "memos"
MANIFESTS_DIR = OUTPUT_DIR / "manifests"
DOCS_DIR = PROJECT_ROOT / "docs" / "minor_revision"
DATA_DIR = PROJECT_ROOT / "data" / "final"
STAGING_DIR = PROJECT_ROOT / "data" / "staging"


def ensure_dirs() -> None:
    """Create all output directories."""
    for d in [TABLES_DIR, FIGURES_DIR, MEMOS_DIR, MANIFESTS_DIR, DOCS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def _escape_latex(text: str) -> str:
    """Escape special LaTeX characters."""
    if not isinstance(text, str):
        text = str(text)
    for old, new in [
        ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
        ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
        ("×", r"$\\times$"), ("±", r"$\\pm$"),
        ("σ", r"$\\sigma$"), ("θ", r"$\\theta$"), ("ψ", r"$\\psi$"),
        ("μ", r"$\\mu$"), ("Δ", r"$\\Delta$"),
    ]:
        text = text.replace(old, new)
    return text


def df_to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a pipe-delimited markdown table."""
    cols = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in df.columns) + " |")
    return "\n".join(lines)


def df_to_latex(
    df: pd.DataFrame,
    caption: str,
    label: str,
    notes: str = "",
) -> str:
    """Render a DataFrame as a LaTeX table."""
    cols = df.columns.tolist()
    ncols = len(cols)
    col_spec = "l" + "c" * (ncols - 1)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{_escape_latex(caption)}}}",
        rf"\label{{tab:{label}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(_escape_latex(c) for c in cols) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            " & ".join(_escape_latex(str(row[c])) for c in cols) + r" \\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
    ])
    if notes:
        lines.extend([
            r"\begin{tablenotes}",
            rf"\small \item {_escape_latex(notes)}",
            r"\end{tablenotes}",
        ])
    lines.append(r"\end{table}")
    return "\n".join(lines)


def write_table(
    df: pd.DataFrame,
    stem: str,
    caption: str = "",
    label: Optional[str] = None,
    notes: str = "",
) -> None:
    """Write a table to csv, tex, and md under TABLES_DIR."""
    ensure_dirs()
    if label is None:
        label = stem

    df.to_csv(TABLES_DIR / f"{stem}.csv", index=False)

    md = f"## {caption}\n\n" + df_to_markdown(df)
    if notes:
        md += f"\n\n*{notes}*\n"
    (TABLES_DIR / f"{stem}.md").write_text(md, encoding="utf-8")

    tex = df_to_latex(df, caption, label, notes)
    (TABLES_DIR / f"{stem}.tex").write_text(tex, encoding="utf-8")

    print(f"  → Wrote {stem}.{{csv,md,tex}}")


def write_doc(filename: str, content: str) -> None:
    """Write a documentation markdown file to docs/minor_revision/."""
    ensure_dirs()
    path = DOCS_DIR / filename
    path.write_text(content, encoding="utf-8")
    print(f"  → Wrote docs/minor_revision/{filename}")


def write_memo(filename: str, content: str) -> None:
    """Write a memo to outputs/minor_revision/memos/."""
    ensure_dirs()
    path = MEMOS_DIR / filename
    path.write_text(content, encoding="utf-8")
    print(f"  → Wrote memo {filename}")
