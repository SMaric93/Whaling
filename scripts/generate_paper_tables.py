#!/usr/bin/env python3
"""
Generate consolidated paper tables in Markdown and LaTeX formats.

Reads all CSV output tables from the econometric and ML pipelines
and writes them to:
  - output/paper_tables.md
  - output/paper_tables.tex
"""

from pathlib import Path
import pandas as pd
import numpy as np
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"
ML_TABLES = ROOT / "outputs" / "tables" / "ml"
ECON_TABLES = ROOT / "output" / "tables"

# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def fmt(x, decimals=4):
    """Format a number for display."""
    if pd.isna(x):
        return "—"
    if isinstance(x, str):
        return x
    if isinstance(x, (int, np.integer)):
        return f"{x:,}"
    if abs(x) > 1000:
        return f"{x:,.0f}"
    if abs(x) < 0.001 and x != 0:
        return f"{x:.2e}"
    return f"{x:.{decimals}f}"


def csv_to_md_table(df, caption="", float_fmt=4):
    """Convert a DataFrame to a Markdown table string."""
    lines = []
    if caption:
        lines.append(f"**{caption}**\n")

    # Format all values
    formatted = df.copy()
    for col in formatted.columns:
        formatted[col] = formatted[col].apply(lambda x: fmt(x, float_fmt))

    # Header
    headers = list(formatted.columns)
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    # Rows
    for _, row in formatted.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")

    lines.append("")
    return "\n".join(lines)


def csv_to_latex_table(df, caption="", label="", float_fmt=4):
    """Convert a DataFrame to a LaTeX table string."""
    lines = []
    n_cols = len(df.columns)
    col_spec = "l" + "r" * (n_cols - 1)

    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    if caption:
        safe_caption = caption.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
        lines.append(r"\caption{" + safe_caption + "}")
    if label:
        lines.append(r"\label{" + label + "}")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Header
    safe_headers = []
    for h in df.columns:
        h_safe = str(h).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
        safe_headers.append(h_safe)
    lines.append(" & ".join(safe_headers) + r" \\")
    lines.append(r"\midrule")

    # Rows
    formatted = df.copy()
    for col in formatted.columns:
        formatted[col] = formatted[col].apply(lambda x: fmt(x, float_fmt))

    for _, row in formatted.iterrows():
        cells = []
        for v in row:
            s = str(v).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
            # Handle special chars
            for ch in ["α", "γ", "ε", "θ", "ψ", "φ", "η", "×", "²"]:
                if ch in s:
                    s = s.replace(ch, {"α": r"$\alpha$", "γ": r"$\gamma$",
                                       "ε": r"$\varepsilon$", "θ": r"$\theta$",
                                       "ψ": r"$\psi$", "φ": r"$\varphi$",
                                       "η": r"$\eta$", "×": r"$\times$",
                                       "²": r"$^2$"}.get(ch, ch))
            cells.append(s)
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    return "\n".join(lines)


def safe_read(path):
    """Read CSV, return None if missing."""
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as e:
            print(f"  Warning: could not read {path}: {e}")
            return None
    return None


# ═══════════════════════════════════════════════════════════════════════
# Table definitions — each is (title, csv_path, label, columns_to_show)
# ═══════════════════════════════════════════════════════════════════════

TABLES = []

# ────────────────────────────────────────────────
# PART I: ECONOMETRIC TABLES
# ────────────────────────────────────────────────

def build_table_1():
    """Table 1: Variance Decomposition (R1)."""
    df = safe_read(ECON_TABLES / "table_variance_decomposition.csv")
    if df is None:
        return None
    show = df[["Component", "Plugin_Share_Pct", "KSS_Share_Pct"]].copy()
    show.columns = ["Component", "Plugin Share (%)", "KSS Share (%)"]
    return show

def build_table_2():
    """Table 2: Main Regression Summary."""
    df = safe_read(ECON_TABLES / "table_main_regressions.csv")
    if df is None:
        return None
    show = df[["Panel", "Specification", "Key_Coefficient", "Value", "R2", "N"]].copy()
    show.columns = ["Panel", "Specification", "Key Coefficient", "Value", "R²", "N"]
    return show

def build_table_3():
    """Table 3: Portability and Persistence (R2, R4)."""
    df = safe_read(ECON_TABLES / "r2_r4_portability_summary.csv")
    if df is None:
        return None
    return df

def build_table_4():
    """Table 4: Event Study Coefficients (R3)."""
    df = safe_read(ECON_TABLES / "r3_event_study_coefficients.csv")
    if df is None:
        return None
    show = df[["event_time", "coefficient", "se", "ci_lower", "ci_upper"]].copy()
    show.columns = ["Event Time", "Coefficient", "SE", "CI Lower", "CI Upper"]
    return show

def build_table_5():
    """Table 5: Complementarity and Resilience (R5, R6)."""
    df = safe_read(ECON_TABLES / "r5_r6_complementarity_resilience.csv")
    if df is None:
        return None
    return df

def build_table_6():
    """Table 6: Strategic Behavior (R10–R12)."""
    df = safe_read(ECON_TABLES / "r10_r11_r12_strategy.csv")
    if df is None:
        return None
    return df

def build_table_7():
    """Table 7: Labor Market Sorting and Switching (R13–R15)."""
    df = safe_read(ECON_TABLES / "r13_r14_r15_labor_market.csv")
    if df is None:
        return None
    return df

def build_table_8():
    """Table 8: Crew-Level Mechanism Analysis."""
    df = safe_read(ECON_TABLES / "mechanism_analysis.csv")
    if df is None:
        return None
    show = df[["test"] + [c for c in ["n", "beta", "se", "t_stat", "p_value", "r2_captain", "r2_full", "incremental_r2"] if c in df.columns]].copy()
    return show

def build_table_9():
    """Table 9: Compass Regressions (C1–C6)."""
    df = safe_read(OUTPUT_DIR / "compass" / "compass_regressions_summary.csv")
    if df is None:
        return None
    show = df[["spec", "variable", "coef", "se", "t", "r2", "n"]].copy()
    show.columns = ["Spec", "Variable", "Coef", "SE", "t", "R²", "N"]
    return show

# ────────────────────────────────────────────────
# PART II: ML TABLES
# ────────────────────────────────────────────────

def build_ml_table_1():
    """ML Table 1: Production Surface Benchmark."""
    df = safe_read(ML_TABLES / "production_surface_benchmark.csv")
    if df is None:
        return None
    show = df[["model", "val_r_squared", "test_r_squared", "val_rmse", "test_rmse"]].copy()
    show.columns = ["Model", "Val R²", "Test R²", "Val RMSE", "Test RMSE"]
    return show

def build_ml_table_2():
    """ML Table 2: Assignment Welfare Counterfactuals."""
    df = safe_read(ML_TABLES / "assignment_welfare.csv")
    if df is None:
        return None
    return df

def build_ml_table_3():
    """ML Table 3: Heterogeneity by Mover Status."""
    df = safe_read(ML_TABLES / "heterogeneity_mover.csv")
    if df is None:
        return None
    show = df[["group", "r_squared", "rmse", "n"]].copy()
    show.columns = ["Group", "R²", "RMSE", "N"]
    return show

def build_ml_table_4():
    """ML Table 4: Heterogeneity by Scarcity."""
    df = safe_read(ML_TABLES / "heterogeneity_scarcity.csv")
    if df is None:
        return None
    show = df[["group", "r_squared", "rmse", "n"]].copy()
    show.columns = ["Group", "R²", "RMSE", "N"]
    return show

def build_ml_table_5():
    """ML Table 5: Off-Policy Evaluation."""
    df = safe_read(ML_TABLES / "off_policy_evaluation.csv")
    if df is None:
        return None
    show = df[["method", "ate", "se"]].copy()
    show.columns = ["Method", "ATE", "SE"]
    return show

def build_ml_table_6():
    """ML Table 6: Network Imprinting."""
    df = safe_read(ML_TABLES / "network_imprinting_results.csv")
    if df is None:
        return None
    return df

def build_ml_table_7():
    """ML Table 7: Conformal Prediction Intervals."""
    df = safe_read(ML_TABLES / "conformal_results.csv")
    if df is None:
        return None
    show = df[["method", "coverage", "avg_width", "test_coverage"]].copy()
    show.columns = ["Method", "Calibration Coverage", "Avg Width", "Test Coverage"]
    return show

def build_ml_table_8():
    """ML Table 8: Latent Search States."""
    df = safe_read(ML_TABLES / "state_summary.csv")
    if df is None:
        return None
    show = df[["state_id", "label", "count", "share", "avg_speed", "revisit_rate"]].copy()
    show.columns = ["State", "Label", "Count", "Share", "Avg Speed", "Revisit Rate"]
    return show

def build_ml_table_9():
    """ML Table 9: Policy Learning Benchmarks (Map)."""
    df = safe_read(ML_TABLES / "policy_map_benchmark.csv")
    if df is None:
        return None
    return df

def build_ml_table_10():
    """ML Table 10: Policy Learning Benchmarks (Compass)."""
    df = safe_read(ML_TABLES / "policy_compass_benchmark.csv")
    if df is None:
        return None
    return df

def build_ml_table_11():
    """ML Table 11: Spatial Quality Index."""
    df = safe_read(ML_TABLES / "spatial_quality_index.csv")
    if df is None:
        return None
    if len(df) > 20:
        df = df.head(20)  # Top 20 for display
    return df

def build_ml_table_12():
    """ML Table 12: Text NLP Topics."""
    df = safe_read(ML_TABLES / "text_topics.csv")
    if df is None:
        return None
    return df

def build_ml_table_13():
    """ML Table 13: Heterogeneity by Agent Capability."""
    df = safe_read(ML_TABLES / "heterogeneity_psi_group.csv")
    if df is None:
        return None
    show = df[["group", "r_squared", "rmse", "n"]].copy()
    show.columns = ["Group", "R²", "RMSE", "N"]
    return show

def build_ml_table_14():
    """ML Table 14: Trajectory Motif Summary."""
    df = safe_read(ML_TABLES / "trajectory_motif_summary.csv")
    if df is None:
        return None
    return df

def build_ml_table_15():
    """ML Table 15: Survival / Exit Benchmark."""
    df = safe_read(ML_TABLES / "exit_policy_benchmark.csv")
    if df is None:
        return None
    return df


# ═══════════════════════════════════════════════════════════════════════
# Master table list
# ═══════════════════════════════════════════════════════════════════════

ALL_TABLES = [
    # Econometric tables
    ("Table 1: Variance Decomposition (AKM with KSS Correction)", build_table_1, "tab:variance_decomposition"),
    ("Table 2: Main Regression Summary", build_table_2, "tab:main_regressions"),
    ("Table 3: Portability and Persistence (R2, R4)", build_table_3, "tab:portability"),
    ("Table 4: Event Study — Agent Switch Effects (R3)", build_table_4, "tab:event_study"),
    ("Table 5: Complementarity and Resilience (R5, R6)", build_table_5, "tab:complementarity"),
    ("Table 6: Strategic Behavior (R10–R12)", build_table_6, "tab:strategy"),
    ("Table 7: Labor Market Sorting and Switching (R13–R15)", build_table_7, "tab:labor_market"),
    ("Table 8: Crew-Level Mechanism Analysis", build_table_8, "tab:mechanisms"),
    ("Table 9: Compass Regressions (C1–C6)", build_table_9, "tab:compass"),
    # ML tables
    ("Table A1: Production Surface Model Comparison", build_ml_table_1, "tab:ml_production_surface"),
    ("Table A2: Assignment Welfare Counterfactuals", build_ml_table_2, "tab:ml_assignment"),
    ("Table A3: Heterogeneity by Mover Status", build_ml_table_3, "tab:ml_het_mover"),
    ("Table A4: Heterogeneity by Scarcity Regime", build_ml_table_4, "tab:ml_het_scarcity"),
    ("Table A5: Off-Policy Evaluation of Agent Assignment", build_ml_table_5, "tab:ml_ope"),
    ("Table A6: Network Imprinting Results", build_ml_table_6, "tab:ml_imprinting"),
    ("Table A7: Conformal Prediction Intervals", build_ml_table_7, "tab:ml_conformal"),
    ("Table A8: Latent Search State Summary", build_ml_table_8, "tab:ml_states"),
    ("Table A9: Policy Learning — Map Model Benchmark", build_ml_table_9, "tab:ml_policy_map"),
    ("Table A10: Policy Learning — Compass Model Benchmark", build_ml_table_10, "tab:ml_policy_compass"),
    ("Table A11: Spatial Quality Index (Top 20)", build_ml_table_11, "tab:ml_spatial"),
    ("Table A12: Text NLP Topics", build_ml_table_12, "tab:ml_topics"),
    ("Table A13: Heterogeneity by Agent Capability Group", build_ml_table_13, "tab:ml_het_psi"),
    ("Table A14: Trajectory Motif Summary", build_ml_table_14, "tab:ml_motifs"),
    ("Table A15: Survival / Exit Policy Benchmark", build_ml_table_15, "tab:ml_survival"),
]


# ═══════════════════════════════════════════════════════════════════════
# Generate files
# ═══════════════════════════════════════════════════════════════════════

def generate():
    md_lines = [
        "# Whaling Productivity Analysis: Complete Paper Tables",
        "",
        "Generated from pipeline output.",
        "",
        "---",
        "",
    ]

    tex_lines = [
        r"% Whaling Productivity Analysis: Complete Paper Tables",
        r"% Auto-generated from pipeline output",
        r"\documentclass[11pt]{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{booktabs}",
        r"\usepackage{geometry}",
        r"\geometry{margin=1in}",
        r"\usepackage{longtable}",
        r"\usepackage{caption}",
        r"\usepackage{amsmath}",
        r"\begin{document}",
        r"\title{Whaling Productivity Analysis: Complete Paper Tables}",
        r"\maketitle",
        r"\tableofcontents",
        r"\clearpage",
        "",
    ]

    # Section headers
    md_lines.append("## Part I: Econometric Regression Tables\n")
    tex_lines.append(r"\section{Econometric Regression Tables}")
    tex_lines.append("")

    n_total = 0
    n_generated = 0

    for i, (title, builder, label) in enumerate(ALL_TABLES):
        # Section break between econ and ML
        if title.startswith("Table A1"):
            md_lines.append("\n---\n")
            md_lines.append("## Part II: Machine Learning Appendix Tables\n")
            tex_lines.append(r"\clearpage")
            tex_lines.append(r"\section{Machine Learning Appendix Tables}")
            tex_lines.append("")

        n_total += 1
        df = builder()

        if df is None:
            print(f"  SKIP: {title} (data not found)")
            md_lines.append(f"### {title}\n")
            md_lines.append("*Data not available.*\n")
            continue

        n_generated += 1
        print(f"  OK: {title} ({len(df)} rows)")

        # Markdown
        md_lines.append(f"### {title}\n")
        md_lines.append(csv_to_md_table(df, float_fmt=4))

        # LaTeX
        tex_lines.append(csv_to_latex_table(df, caption=title, label=label, float_fmt=4))

    # Close LaTeX
    tex_lines.append(r"\end{document}")
    tex_lines.append("")

    # Write files
    md_path = OUTPUT_DIR / "paper_tables.md"
    tex_path = OUTPUT_DIR / "paper_tables.tex"

    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    tex_path.write_text("\n".join(tex_lines), encoding="utf-8")

    print(f"\nGenerated {n_generated}/{n_total} tables")
    print(f"  Markdown: {md_path}")
    print(f"  LaTeX:    {tex_path}")


if __name__ == "__main__":
    generate()
