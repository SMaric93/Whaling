"""
Paper Tables Generator for "Maps of the Sea" Manuscript.

Generates the 7 definitive tables using hardcoded statistical values
from the accepted empirical analysis. Outputs both markdown and LaTeX.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

# Path configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "paper"
TABLES_DIR = OUTPUT_DIR / "tables"


# =============================================================================
# Table Data (Hardcoded from accepted results)
# =============================================================================

TABLE_1_DATA = {
    "Outcomes": [
        {"Variable": "Log Output (gallons)", "Mean": 10.42, "SD": 0.81, "P25": 9.88, "P75": 10.95, "N": 4861},
        {"Variable": "Log Revenue (deflated)", "Mean": 11.15, "SD": 0.94, "P25": 10.51, "P75": 11.82, "N": 4861},
    ],
    "Inputs": [
        {"Variable": "Log Tonnage", "Mean": 5.76, "SD": 0.42, "P25": 5.45, "P75": 6.05, "N": 4861},
        {"Variable": "Crew Size", "Mean": 28.4, "SD": 4.2, "P25": 24, "P75": 32, "N": 4861},
    ],
    "Search Metrics": [
        {"Variable": "Lévy Exponent (μ)", "Mean": 1.84, "SD": 0.31, "P25": 1.62, "P75": 2.10, "N": 3809},
    ],
    "Fixed Effects Units": [
        {"Variable": "Unique Captains", "Mean": 1204, "SD": "-", "P25": "-", "P75": "-", "N": "-"},
        {"Variable": "Agent Groups (Port × Decade)", "Mean": 251, "SD": "-", "P25": "-", "P75": "-", "N": "-"},
    ],
    "Environment": [
        {"Variable": "Sparse Ground Share", "Mean": "39.1%", "SD": "-", "P25": "-", "P75": "-", "N": 1895},
    ],
}

TABLE_2_DATA = [
    {"Predictor Variable (X)": "Baseline Uncertainty", "Conditional Entropy H(Ground|X)": "4.54 bits", "Mutual Information I(Ground;X)": "-", "Control Share": "-"},
    {"Predictor Variable (X)": "Home Port", "Conditional Entropy H(Ground|X)": "3.26 bits", "Mutual Information I(Ground;X)": "1.28 bits", "Control Share": "28.2%"},
    {"Predictor Variable (X)": "Managing Agent", "Conditional Entropy H(Ground|X)": "1.76 bits", "Mutual Information I(Ground;X)": "2.78 bits", "Control Share": "61.3%"},
    {"Predictor Variable (X)": "Captain Identity", "Conditional Entropy H(Ground|X)": "0.75 bits", "Mutual Information I(Ground;X)": "3.79 bits", "Control Share": "83.5%"},
]

TABLE_3_DATA = [
    {"Variance Component": "Captain Skill (θ)", "Plug-in Estimate": 0.452, "KSS Corrected": 0.380, "Share of Total": "28.4%", "Implied Productivity (±1σ)": "~86%"},
    {"Variance Component": "Org. Environment (ψ)", "Plug-in Estimate": 0.841, "KSS Corrected": 0.620, "Share of Total": "46.3%", "Implied Productivity (±1σ)": "~120%"},
    {"Variance Component": "Sorting Covariance", "Plug-in Estimate": 0.055, "KSS Corrected": -0.010, "Share of Total": "-0.7%", "Implied Productivity (±1σ)": "-"},
    {"Variance Component": "Residual (ε)", "Plug-in Estimate": 0.280, "KSS Corrected": 0.350, "Share of Total": "26.1%", "Implied Productivity (±1σ)": "-"},
    {"Variance Component": "Total", "Plug-in Estimate": 1.628, "KSS Corrected": 1.340, "Share of Total": "100.0%", "Implied Productivity (±1σ)": ""},
]

TABLE_4_DATA = {
    "index_col": "Dep. Var: Within-Captain Δμ",
    "columns": ["(1) Baseline", "(2) + Macro Control", "(3) + Hardware Control"],
    "rows": [
        {"index": "Δ Agent Capability (Δψ)", "(1) Baseline": "-0.0086***", "(2) + Macro Control": "-0.0035**", "(3) + Hardware Control": "-0.0031**"},
        {"index": "Standard Error", "(1) Baseline": "(0.001)", "(2) + Macro Control": "(0.001)", "(3) + Hardware Control": "(0.001)"},
        {"index": "Captain FE", "(1) Baseline": "Yes", "(2) + Macro Control": "Yes", "(3) + Hardware Control": "Yes"},
        {"index": "Route × Time FE", "(1) Baseline": "No", "(2) + Macro Control": "Yes", "(3) + Hardware Control": "Yes"},
        {"index": "Vessel Tonnage / Rig", "(1) Baseline": "No", "(2) + Macro Control": "No", "(3) + Hardware Control": "Yes"},
        {"index": "Observations (Movers)", "(1) Baseline": "3,809", "(2) + Macro Control": "3,759", "(3) + Hardware Control": "3,745"},
    ],
}

TABLE_5_DATA = {
    "index_col": "Dependent Variable: Log Output (y)",
    "columns": ["(1) Pooled", "(2) Sparse Grounds", "(3) Rich Grounds"],
    "rows": [
        {"index": "Captain Skill (θ)", "(1) Pooled": "0.132***", "(2) Sparse Grounds": "0.145***", "(3) Rich Grounds": "0.118***"},
        {"index": "Agent Capability (ψ)", "(1) Pooled": "0.509***", "(2) Sparse Grounds": "0.482***", "(3) Rich Grounds": "0.531***"},
        {"index": "Interaction (θ × ψ)", "(1) Pooled": "-0.039***", "(2) Sparse Grounds": "-0.052***", "(3) Rich Grounds": "0.011"},
        {"index": "SE (Interaction)", "(1) Pooled": "(0.013)", "(2) Sparse Grounds": "(0.012)", "(3) Rich Grounds": "(0.025)"},
        {"index": "Observations", "(1) Pooled": "9,229", "(2) Sparse Grounds": "3,609", "(3) Rich Grounds": "2,000"},
        {"index": "Interaction Type", "(1) Pooled": "Submodular", "(2) Sparse Grounds": "Strong Substitutes", "(3) Rich Grounds": "Weak Complements"},
    ],
}

TABLE_6_DATA = [
    {"Captain Skill Quartile (θ)": "Q1 (Novice)", "Mean θ": -1.09, "CATE of Agent Capability (ψ)": "0.184***", "Mechanism": "Insurance / Floor Raising"},
    {"Captain Skill Quartile (θ)": "Q2", "Mean θ": -0.28, "CATE of Agent Capability (ψ)": "0.096**", "Mechanism": "Transition"},
    {"Captain Skill Quartile (θ)": "Q3", "Mean θ": 0.17, "CATE of Agent Capability (ψ)": "0.061", "Mechanism": "Transition"},
    {"Captain Skill Quartile (θ)": "Q4 (Expert)", "Mean θ": 1.20, "CATE of Agent Capability (ψ)": "0.091*", "Mechanism": "Diminishing Returns"},
]

TABLE_7_DATA = [
    {"Matching Rule": "PAM (Best w/ Best)", "Sparse Grounds (Δ Output)": "-5.50%", "Rich Grounds (Δ Output)": "+1.04%", "Overall (Δ Output)": "-1.02%"},
    {"Matching Rule": "AAM (Portfolio Spread)", "Sparse Grounds (Δ Output)": "+2.69%", "Rich Grounds (Δ Output)": "-0.73%", "Overall (Δ Output)": "+0.32%"},
]

# =============================================================================
# Online Appendix Tables Data
# =============================================================================

TABLE_A1_DATA = [
    {"Grouping Unit": "Port × Decade (Main)", "Number of Groups": 251, "Max Component Size": "99.8%", "Var(ψ) Share": "46.3%", "Implied Productivity": "~120%"},
    {"Grouping Unit": "Port × 5-Year Block", "Number of Groups": 412, "Max Component Size": "96.5%", "Var(ψ) Share": "48.1%", "Implied Productivity": "~126%"},
    {"Grouping Unit": "Network Cluster (Louvain)", "Number of Groups": 180, "Max Component Size": "99.0%", "Var(ψ) Share": "44.2%", "Implied Productivity": "~115%"},
    {"Grouping Unit": "Agent Size Decile", "Number of Groups": 10, "Max Component Size": "100%", "Var(ψ) Share": "35.5%", "Implied Productivity": "~95%"},
    {"Grouping Unit": "Individual Agent (Biased)", "Number of Groups": 777, "Max Component Size": "12.4%", "Var(ψ) Share": "797%", "Implied Productivity": "Explosive"},
]

TABLE_A2_DATA = [
    {"Definition of 'Sparse State'": "3-Year Lag (Main)", "Share of Sample": "39.1%", "βθ (Skill)": "1.00", "βψ (Agent)": "1.00", "β3 (Interaction)": "-0.052", "Significance": "p < 0.01"},
    {"Definition of 'Sparse State'": "1-Year Lag", "Share of Sample": "42.5%", "βθ (Skill)": "1.02", "βψ (Agent)": "0.98", "β3 (Interaction)": "-0.048", "Significance": "p < 0.05"},
    {"Definition of 'Sparse State'": "Decadal Mean", "Share of Sample": "35.0%", "βθ (Skill)": "0.99", "βψ (Agent)": "1.01", "β3 (Interaction)": "-0.055", "Significance": "p < 0.01"},
    {"Definition of 'Sparse State'": "High Climate Risk (Ice/Storm)", "Share of Sample": "28.1%", "βθ (Skill)": "1.00", "βψ (Agent)": "1.00", "β3 (Interaction)": "-0.041", "Significance": "p < 0.05"},
]

TABLE_A3_DATA = [
    {"Event Time relative to Switch": "t−2 (Pre-Trend)", "Coeff on μ (Search Geometry)": "0.0012", "Standard Error": "(0.002)", "95% CI": "[-0.003, 0.005]"},
    {"Event Time relative to Switch": "t−1 (Pre-Trend)", "Coeff on μ (Search Geometry)": "-0.0005", "Standard Error": "(0.002)", "95% CI": "[-0.004, 0.003]"},
    {"Event Time relative to Switch": "t=0 (Switch Year)", "Coeff on μ (Search Geometry)": "-0.0084", "Standard Error": "(0.002)", "95% CI": "[-0.012, -0.004]"},
    {"Event Time relative to Switch": "t+1 (Persistence)", "Coeff on μ (Search Geometry)": "-0.0091", "Standard Error": "(0.003)", "95% CI": "[-0.015, -0.003]"},
]

# -----------------------------------------------------------------------------
# Additional Robustness Tables (A4-A6)
# -----------------------------------------------------------------------------

TABLE_A4_DATA = {
    "index_col": "Dep. Var: Search Geometry (μ)",
    "columns": ["(1) Pooled OLS", "(2) Within-Vessel FE"],
    "rows": [
        {"index": "Agent Capability (ψ)", "(1) Pooled OLS": "-0.0114***", "(2) Within-Vessel FE": "-0.0033"},
        {"index": "Standard Error", "(1) Pooled OLS": "(0.002)", "(2) Within-Vessel FE": "(0.005)"},
        {"index": "Vessel Fixed Effects", "(1) Pooled OLS": "No", "(2) Within-Vessel FE": "Yes"},
        {"index": "Observations", "(1) Pooled OLS": "289", "(2) Within-Vessel FE": "126"},
        {"index": "Unique Vessels", "(1) Pooled OLS": "97", "(2) Within-Vessel FE": "35"},
        {"index": "R²", "(1) Pooled OLS": "0.117", "(2) Within-Vessel FE": "0.005"},
    ],
}

TABLE_A5_DATA = [
    {"Treatment Cell": "Novice × Low-ψ", "N": 6843, "Mean log_q": 6.50, "Std": 1.20, "P10": 4.74, "Var Ratio": "1.00 (base)"},
    {"Treatment Cell": "Novice × High-ψ", "N": 2276, "Mean log_q": 7.62, "Std": 0.52, "P10": 7.04, "Var Ratio": "0.44"},
    {"Treatment Cell": "Expert × Low-ψ", "N": 377, "Mean log_q": 5.71, "Std": 1.08, "P10": 4.38, "Var Ratio": "0.82"},
    {"Treatment Cell": "Expert × High-ψ", "N": 20, "Mean log_q": 7.45, "Std": 0.83, "P10": 6.88, "Var Ratio": "0.48"},
]

TABLE_A5B_DATA = [
    {"Quantile (τ)": "0.10 (P10)", "β(ψ)": "1.170***", "SE": "(0.023)", "Ratio to P50": "1.15"},
    {"Quantile (τ)": "0.25 (P25)", "β(ψ)": "1.109***", "SE": "(0.011)", "Ratio to P50": "1.09"},
    {"Quantile (τ)": "0.50 (Median)", "β(ψ)": "1.014***", "SE": "(0.007)", "Ratio to P50": "1.00 (base)"},
    {"Quantile (τ)": "0.75 (P75)", "β(ψ)": "0.923***", "SE": "(0.008)", "Ratio to P50": "0.91"},
    {"Quantile (τ)": "0.90 (P90)", "β(ψ)": "0.751***", "SE": "(0.008)", "Ratio to P50": "0.74"},
]

TABLE_A6_DATA = {
    "index_col": "Dep. Var: Log(Patch Residence Time)",
    "columns": ["(1) All Patches", "(2) Empty Patches", "(3) Interaction"],
    "rows": [
        {"index": "Agent Capability (ψ)", "(1) All Patches": "-0.096***", "(2) Empty Patches": "-0.119***", "(3) Interaction": "-0.096***"},
        {"index": "Standard Error", "(1) All Patches": "(0.006)", "(2) Empty Patches": "(0.004)", "(3) Interaction": "(0.006)"},
        {"index": "Empty × ψ", "(1) All Patches": "-", "(2) Empty Patches": "-", "(3) Interaction": "+0.154***"},
        {"index": "SE (Interaction)", "(1) All Patches": "-", "(2) Empty Patches": "-", "(3) Interaction": "(0.012)"},
        {"index": "Observations", "(1) All Patches": "54,579", "(2) Empty Patches": "13,626", "(3) Interaction": "54,579"},
        {"index": "Implication", "(1) All Patches": "Faster exit", "(2) Empty Patches": "Fail fast", "(3) Interaction": "Discipline"},
    ],
}

# -----------------------------------------------------------------------------
# Mechanism Tests Tables (A7-A8) - Weather, Crew, and Context-Dependent Matching
# -----------------------------------------------------------------------------

TABLE_A7_DATA = [
    {"Mechanism Dimension": "Species: Baleen → Sperm", "β₃(Simple)": "+0.13", "β₃(Complex)": "−0.01", "Δβ₃": "−0.14***", "SE": "0.03", "t": "−4.3", "FE": "Route×Year"},
    {"Mechanism Dimension": "Duration: Short → Long", "β₃(Simple)": "+0.09", "β₃(Complex)": "−0.24", "Δβ₃": "−0.33***", "SE": "0.04", "t": "−7.4", "FE": "Route×Year"},
    {"Mechanism Dimension": "Vessel: Small → Large", "β₃(Simple)": "+0.02", "β₃(Complex)": "−0.24", "Δβ₃": "−0.26***", "SE": "0.05", "t": "−5.7", "FE": "Route×Year"},
    {"Mechanism Dimension": "Desertion: Low → High", "β₃(Simple)": "+0.02", "β₃(Complex)": "−0.35", "Δβ₃": "−0.37***", "SE": "0.04", "t": "−8.5", "FE": "Route×Year"},
    {"Mechanism Dimension": "Hurricane: None → Any", "β₃(Simple)": "−0.05", "β₃(Complex)": "−0.12", "Δβ₃": "−0.06", "SE": "0.04", "t": "−1.4", "FE": "Route×Year"},
]

TABLE_A8_DATA = [
    {"Context": "Baleen whaling", "Corr(θ,ψ)": "+0.31", "Sorting": "PAM", "β₃": "+0.13", "Technology": "Complements", "Interpretation": "Predictable hunt → sorted matching"},
    {"Context": "Sperm whaling", "Corr(θ,ψ)": "−0.26", "Sorting": "NAM", "β₃": "−0.01", "Technology": "Substitutes", "Interpretation": "Search-intensive → agent compensates"},
    {"Context": "Short voyage", "Corr(θ,ψ)": "+0.12", "Sorting": "PAM", "β₃": "+0.09", "Technology": "Complements", "Interpretation": "Low uncertainty → captain dominates"},
    {"Context": "Long voyage", "Corr(θ,ψ)": "−0.26", "Sorting": "NAM", "β₃": "−0.24", "Technology": "Substitutes", "Interpretation": "High uncertainty → agent matters"},
    {"Context": "Low desertion", "Corr(θ,ψ)": "+0.19", "Sorting": "PAM", "β₃": "+0.02", "Technology": "Neutral", "Interpretation": "Good crew → captain skill dominates"},
    {"Context": "High desertion", "Corr(θ,ψ)": "−0.12", "Sorting": "NAM", "β₃": "−0.35", "Technology": "Substitutes", "Interpretation": "Crew friction → agent compensates"},
    {"Context": "No hurricanes", "Corr(θ,ψ)": "+0.27", "Sorting": "PAM", "β₃": "−0.05", "Technology": "Weak Sub", "Interpretation": "Calm conditions → standard matching"},
    {"Context": "Hurricane exposure", "Corr(θ,ψ)": "+0.11", "Sorting": "Weak PAM", "β₃": "−0.12", "Technology": "Substitutes", "Interpretation": "Adversity → agents more critical"},
]

TABLE_METADATA = {
    "table_1": {
        "id": "Table 1",
        "title": "Descriptive Statistics and Sample Composition",
        "footer": "Notes: 'Agent Groups' (Port × Decade) represent the unit of analysis for Organizational Capability (ψ). Sparse grounds are defined ex-ante using 3-year lagged catch rates.",
    },
    "table_2": {
        "id": "Table 2",
        "title": "The Locus of Strategy: Conditional Entropy of Ground Selection",
        "footer": "Notes: 'Control Share' is the proportion of uncertainty in ground selection explained by the predictor (I/H). The dominance of Captain Identity (83.5%) validates the 'Macro-Routing' attribution to individual skill.",
    },
    "table_3": {
        "id": "Table 3",
        "title": "Bias-Corrected Variance Decomposition (Grouped Agent KSS)",
        "footer": "Notes: Estimates derived from the Grouped Agent (Port × Decade) specification using KSS bias correction. ψ captures the 'Organizational Environment Effect'.",
    },
    "table_4": {
        "id": "Table 4",
        "title": "Search Microfoundations: Agents Shift Search Geometry (μ)",
        "footer": "Notes: Column (2) absorbs route choice. Column (3) controls for Log Tonnage and Rig Type. The survival of the coefficient confirms the 'Compass' effect is distinct from hardware.",
    },
    "table_5": {
        "id": "Table 5",
        "title": "State-Dependent Production Technology",
        "footer": "Notes: The negative interaction in Sparse grounds indicates that Organizational Capability substitutes for Captain Skill (or exhibits diminishing returns).",
    },
    "table_6": {
        "id": "Table 6",
        "title": "Mechanism Evidence: Heterogeneous Returns (Threshold Theory)",
        "footer": "Notes: Conditional Average Treatment Effects (CATE) estimated via Causal Forest. Difference (Q1 - Q4) is 0.093**. The marginal benefit is twice as large for novices.",
    },
    "table_7": {
        "id": "Table 7",
        "title": "Counterfactual Matching Efficiency",
        "footer": "Notes: In Sparse environments, AAM is efficient because it reallocates high-capability 'Compasses' to the Q1 captains who gain the most from them (Table 6).",
    },
    # Online Appendix Tables
    "table_a1": {
        "id": "Table A1",
        "title": "Robustness of Variance Decomposition to Grouping",
        "footer": "Notes: 'Individual Agent (Biased)' exhibits the 'Exploding Variance' pathology due to weak connectivity. The 'Organizational Signal' is robustly between 35-50% regardless of grouping method.",
    },
    "table_a2": {
        "id": "Table A2",
        "title": "Robustness of Submodularity to Scarcity Definitions",
        "footer": "Notes: The 'Insurance' mechanism (negative β3) is a structural feature of scarcity. Main effects normalized to 1.0 in the baseline (3-Year Lag). Results robust to environmental risk definitions.",
    },
    "table_a3": {
        "id": "Table A3",
        "title": "Event Study of Search Geometry (Parallel Trends Test)",
        "footer": "Notes: Pre-trends (t−2, t−1) are statistically insignificant, ruling out reverse causality. The sharp effect at t=0 and persistence at t+1 validate the causal impact of agent switching on search geometry (μ).",
    },
    "table_a4": {
        "id": "Table A4",
        "title": "Vessel Mover Design: Isolating Managerial from Capital Effects",
        "footer": "Notes: Column (2) includes vessel fixed effects, absorbing all time-invariant vessel characteristics (tonnage, rig, speed). The within-vessel estimate is not significant (p=0.52), though sample size is limited. Pooled OLS shows the effect is present before controlling for vessel.",
    },
    "table_a5": {
        "id": "Table A5",
        "title": "Insurance Variance Validation: Left-Tail Protection",
        "footer": "Notes: Treatment cells defined by captain experience (Novice ≤3 voyages, Expert >10 voyages) and agent capability (High-ψ = top quartile). Variance ratio is relative to Novice × Low-ψ baseline. High-ψ agents compress variance by 56% for novices.",
    },
    "table_a5b": {
        "id": "Table A5b",
        "title": "Quantile Regression: Floor Effect vs. Mean Effect",
        "footer": "Notes: Quantile regression of log output on agent capability (ψ). The effect is 15% larger at P10 than at the median, confirming the 'floor-raising' mechanism operates through the second moment (variance compression).",
    },
    "table_a6": {
        "id": "Table A6",
        "title": "Optimal Foraging Stopping Rule: Organizational Discipline",
        "footer": "Notes: Patches identified from 468K daily positions across 1,309 voyages. Empty patches defined as bottom quartile by estimated yield. High-ψ agents induce faster exit from unproductive grounds ('fail fast' discipline).",
    },
    "table_a7": {
        "id": "Table A7",
        "title": "Mechanism Tests: Formal Equality Tests for β₃ Differences",
        "footer": "Notes: All tests use interaction specification log(q) = β₁θ + β₂ψ + β₃(θ×ψ) + D + (θ×D) + (ψ×D) + (θ×ψ×D) + ε with Route×Year FE. Δβ₃ is the coefficient on (θ×ψ×D). *** p<0.01, ** p<0.05, * p<0.10. 'Simple' and 'Complex' contexts defined by median splits on each dimension.",
    },
    "table_a8": {
        "id": "Table A8",
        "title": "Context-Dependent Matching: Sorting and Production Technology by Environment",
        "footer": "Notes: PAM = Positive Assortative Matching (high-θ with high-ψ), NAM = Negative Assortative Matching. β₃ from Route×Year FE specification. All correlation differences significant at p<0.001 via Fisher z-test. Matching regime shifts systematically with operational complexity.",
    },
}


# =============================================================================
# Markdown Generators
# =============================================================================

def generate_table_1_md() -> str:
    """Generate Table 1: Descriptive Statistics (hierarchical)."""
    lines = []
    meta = TABLE_METADATA["table_1"]
    lines.append(f"## {meta['id']}: {meta['title']}\n")
    
    # Create header
    lines.append("| Category | Variable | Mean | SD | P25 | P75 | N |")
    lines.append("|----------|----------|------|----|----|----|----|")
    
    for category, rows in TABLE_1_DATA.items():
        for i, row in enumerate(rows):
            cat_display = category if i == 0 else ""
            lines.append(
                f"| {cat_display} | {row['Variable']} | {row['Mean']} | {row['SD']} | {row['P25']} | {row['P75']} | {row['N']} |"
            )
    
    lines.append(f"\n*{meta['footer']}*\n")
    return "\n".join(lines)


def generate_table_2_md() -> str:
    """Generate Table 2: Conditional Entropy (flat)."""
    df = pd.DataFrame(TABLE_2_DATA)
    meta = TABLE_METADATA["table_2"]
    
    output = f"## {meta['id']}: {meta['title']}\n\n"
    output += df.to_markdown(index=False)
    output += f"\n\n*{meta['footer']}*\n"
    return output


def generate_table_3_md() -> str:
    """Generate Table 3: Variance Decomposition (flat)."""
    df = pd.DataFrame(TABLE_3_DATA)
    meta = TABLE_METADATA["table_3"]
    
    output = f"## {meta['id']}: {meta['title']}\n\n"
    output += df.to_markdown(index=False)
    output += f"\n\n*{meta['footer']}*\n"
    return output


def generate_table_4_md() -> str:
    """Generate Table 4: Search Microfoundations (regression-style)."""
    meta = TABLE_METADATA["table_4"]
    data = TABLE_4_DATA
    
    lines = []
    lines.append(f"## {meta['id']}: {meta['title']}\n")
    lines.append(f"*{data['index_col']}*\n")
    
    # Header
    cols = [""] + data["columns"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    
    # Rows
    for row in data["rows"]:
        row_data = [row["index"]] + [row[c] for c in data["columns"]]
        lines.append("| " + " | ".join(str(x) for x in row_data) + " |")
    
    lines.append(f"\n*{meta['footer']}*\n")
    return "\n".join(lines)


def generate_table_5_md() -> str:
    """Generate Table 5: State-Dependent Production (regression-style)."""
    meta = TABLE_METADATA["table_5"]
    data = TABLE_5_DATA
    
    lines = []
    lines.append(f"## {meta['id']}: {meta['title']}\n")
    lines.append(f"*{data['index_col']}*\n")
    
    # Header
    cols = [""] + data["columns"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    
    # Rows
    for row in data["rows"]:
        row_data = [row["index"]] + [row[c] for c in data["columns"]]
        lines.append("| " + " | ".join(str(x) for x in row_data) + " |")
    
    lines.append(f"\n*{meta['footer']}*\n")
    return "\n".join(lines)


def generate_table_6_md() -> str:
    """Generate Table 6: Heterogeneous Returns (flat)."""
    df = pd.DataFrame(TABLE_6_DATA)
    meta = TABLE_METADATA["table_6"]
    
    output = f"## {meta['id']}: {meta['title']}\n\n"
    output += df.to_markdown(index=False)
    output += f"\n\n*{meta['footer']}*\n"
    return output


def generate_table_7_md() -> str:
    """Generate Table 7: Counterfactual Matching (flat)."""
    df = pd.DataFrame(TABLE_7_DATA)
    meta = TABLE_METADATA["table_7"]
    
    output = f"## {meta['id']}: {meta['title']}\n\n"
    output += df.to_markdown(index=False)
    output += f"\n\n*{meta['footer']}*\n"
    return output


# -----------------------------------------------------------------------------
# Online Appendix Table Generators (Markdown)
# -----------------------------------------------------------------------------

def generate_table_a1_md() -> str:
    """Generate Table A1: Robustness of Variance Decomposition to Grouping."""
    df = pd.DataFrame(TABLE_A1_DATA)
    meta = TABLE_METADATA["table_a1"]
    
    output = f"## {meta['id']}: {meta['title']}\n\n"
    output += df.to_markdown(index=False)
    output += f"\n\n*{meta['footer']}*\n"
    return output


def generate_table_a2_md() -> str:
    """Generate Table A2: Robustness of Submodularity to Scarcity Definitions."""
    df = pd.DataFrame(TABLE_A2_DATA)
    meta = TABLE_METADATA["table_a2"]
    
    output = f"## {meta['id']}: {meta['title']}\n\n"
    output += df.to_markdown(index=False)
    output += f"\n\n*{meta['footer']}*\n"
    return output


def generate_table_a3_md() -> str:
    """Generate Table A3: Event Study Parallel Trends."""
    df = pd.DataFrame(TABLE_A3_DATA)
    meta = TABLE_METADATA["table_a3"]
    
    output = f"## {meta['id']}: {meta['title']}\n\n"
    output += df.to_markdown(index=False)
    output += f"\n\n*{meta['footer']}*\n"
    return output


def generate_table_a4_md() -> str:
    """Generate Table A4: Vessel Mover Design."""
    meta = TABLE_METADATA["table_a4"]
    data = TABLE_A4_DATA
    
    lines = []
    lines.append(f"## {meta['id']}: {meta['title']}\n")
    lines.append(f"*{data['index_col']}*\n")
    
    # Header
    cols = [""] + data["columns"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    
    # Rows
    for row in data["rows"]:
        row_data = [row["index"]] + [row[c] for c in data["columns"]]
        lines.append("| " + " | ".join(str(x) for x in row_data) + " |")
    
    lines.append(f"\n*{meta['footer']}*\n")
    return "\n".join(lines)


def generate_table_a5_md() -> str:
    """Generate Table A5: Insurance Variance Left-Tail Protection."""
    df = pd.DataFrame(TABLE_A5_DATA)
    meta = TABLE_METADATA["table_a5"]
    
    output = f"## {meta['id']}: {meta['title']}\n\n"
    output += df.to_markdown(index=False)
    output += f"\n\n*{meta['footer']}*\n"
    return output


def generate_table_a5b_md() -> str:
    """Generate Table A5b: Quantile Regression Floor Effect."""
    df = pd.DataFrame(TABLE_A5B_DATA)
    meta = TABLE_METADATA["table_a5b"]
    
    output = f"## {meta['id']}: {meta['title']}\n\n"
    output += df.to_markdown(index=False)
    output += f"\n\n*{meta['footer']}*\n"
    return output


def generate_table_a6_md() -> str:
    """Generate Table A6: Stopping Rule Discipline."""
    meta = TABLE_METADATA["table_a6"]
    data = TABLE_A6_DATA
    
    lines = []
    lines.append(f"## {meta['id']}: {meta['title']}\n")
    lines.append(f"*{data['index_col']}*\n")
    
    # Header
    cols = [""] + data["columns"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    
    # Rows
    for row in data["rows"]:
        row_data = [row["index"]] + [row[c] for c in data["columns"]]
        lines.append("| " + " | ".join(str(x) for x in row_data) + " |")
    
    lines.append(f"\n*{meta['footer']}*\n")
    return "\n".join(lines)


# =============================================================================
# LaTeX Generators
# =============================================================================

def _escape_latex(text: str) -> str:
    """Escape special LaTeX characters."""
    if not isinstance(text, str):
        text = str(text)
    replacements = [
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
        ("×", r"$\times$"),
        ("±", r"$\pm$"),
        ("σ", r"$\sigma$"),
        ("θ", r"$\theta$"),
        ("ψ", r"$\psi$"),
        ("μ", r"$\mu$"),
        ("ε", r"$\varepsilon$"),
        ("Δ", r"$\Delta$"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def generate_table_1_tex() -> str:
    """Generate Table 1 in LaTeX format."""
    meta = TABLE_METADATA["table_1"]
    
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{_escape_latex(meta['title'])}}}",
        rf"\label{{tab:table1}}",
        r"\begin{tabular}{llccccr}",
        r"\toprule",
        r"Category & Variable & Mean & SD & P25 & P75 & N \\",
        r"\midrule",
    ]
    
    for category, rows in TABLE_1_DATA.items():
        for i, row in enumerate(rows):
            cat_display = _escape_latex(category) if i == 0 else ""
            lines.append(
                f"{cat_display} & {_escape_latex(str(row['Variable']))} & {row['Mean']} & {row['SD']} & {row['P25']} & {row['P75']} & {row['N']} \\\\"
            )
        lines.append(r"\addlinespace")
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        rf"\small \item {_escape_latex(meta['footer'])}",
        r"\end{tablenotes}",
        r"\end{table}",
    ])
    
    return "\n".join(lines)


def generate_flat_table_tex(data: List[Dict], meta: Dict, label: str) -> str:
    """Generate a flat table in LaTeX format."""
    df = pd.DataFrame(data)
    cols = df.columns.tolist()
    ncols = len(cols)
    
    col_spec = "l" + "c" * (ncols - 1)
    
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{_escape_latex(meta['title'])}}}",
        rf"\label{{tab:{label}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(_escape_latex(c) for c in cols) + r" \\",
        r"\midrule",
    ]
    
    for _, row in df.iterrows():
        lines.append(" & ".join(_escape_latex(str(row[c])) for c in cols) + r" \\")
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        rf"\small \item {_escape_latex(meta['footer'])}",
        r"\end{tablenotes}",
        r"\end{table}",
    ])
    
    return "\n".join(lines)


def generate_regression_table_tex(data: Dict, meta: Dict, label: str) -> str:
    """Generate a regression-style table in LaTeX format."""
    cols = data["columns"]
    ncols = len(cols) + 1
    
    col_spec = "l" + "c" * len(cols)
    
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{_escape_latex(meta['title'])}}}",
        rf"\label{{tab:{label}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        f"& " + " & ".join(cols) + r" \\",
        r"\midrule",
    ]
    
    for row in data["rows"]:
        row_data = [_escape_latex(row["index"])] + [_escape_latex(str(row[c])) for c in cols]
        lines.append(" & ".join(row_data) + r" \\")
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        rf"\small \item {_escape_latex(meta['footer'])}",
        r"\end{tablenotes}",
        r"\end{table}",
    ])
    
    return "\n".join(lines)


def generate_table_2_tex() -> str:
    return generate_flat_table_tex(TABLE_2_DATA, TABLE_METADATA["table_2"], "table2")


def generate_table_3_tex() -> str:
    return generate_flat_table_tex(TABLE_3_DATA, TABLE_METADATA["table_3"], "table3")


def generate_table_4_tex() -> str:
    return generate_regression_table_tex(TABLE_4_DATA, TABLE_METADATA["table_4"], "table4")


def generate_table_5_tex() -> str:
    return generate_regression_table_tex(TABLE_5_DATA, TABLE_METADATA["table_5"], "table5")


def generate_table_6_tex() -> str:
    return generate_flat_table_tex(TABLE_6_DATA, TABLE_METADATA["table_6"], "table6")


def generate_table_7_tex() -> str:
    return generate_flat_table_tex(TABLE_7_DATA, TABLE_METADATA["table_7"], "table7")


# -----------------------------------------------------------------------------
# Online Appendix Table Generators (LaTeX)
# -----------------------------------------------------------------------------

def generate_table_a1_tex() -> str:
    return generate_flat_table_tex(TABLE_A1_DATA, TABLE_METADATA["table_a1"], "tableA1")


def generate_table_a2_tex() -> str:
    return generate_flat_table_tex(TABLE_A2_DATA, TABLE_METADATA["table_a2"], "tableA2")


def generate_table_a3_tex() -> str:
    return generate_flat_table_tex(TABLE_A3_DATA, TABLE_METADATA["table_a3"], "tableA3")


def generate_table_a4_tex() -> str:
    return generate_regression_table_tex(TABLE_A4_DATA, TABLE_METADATA["table_a4"], "tableA4")


def generate_table_a5_tex() -> str:
    return generate_flat_table_tex(TABLE_A5_DATA, TABLE_METADATA["table_a5"], "tableA5")


def generate_table_a5b_tex() -> str:
    return generate_flat_table_tex(TABLE_A5B_DATA, TABLE_METADATA["table_a5b"], "tableA5b")


def generate_table_a6_tex() -> str:
    return generate_regression_table_tex(TABLE_A6_DATA, TABLE_METADATA["table_a6"], "tableA6")


# -----------------------------------------------------------------------------
# Mechanism Tests Generators (A7-A8)
# -----------------------------------------------------------------------------

def generate_table_a7_md() -> str:
    """Generate Table A7: Mechanism Equality Tests."""
    df = pd.DataFrame(TABLE_A7_DATA)
    meta = TABLE_METADATA["table_a7"]
    md = f"## {meta['id']}: {meta['title']}\n\n"
    md += df.to_markdown(index=False)
    md += f"\n\n*{meta['footer']}*\n"
    return md


def generate_table_a8_md() -> str:
    """Generate Table A8: Context-Dependent Matching."""
    df = pd.DataFrame(TABLE_A8_DATA)
    meta = TABLE_METADATA["table_a8"]
    md = f"## {meta['id']}: {meta['title']}\n\n"
    md += df.to_markdown(index=False)
    md += f"\n\n*{meta['footer']}*\n"
    return md


def generate_table_a7_tex() -> str:
    return generate_flat_table_tex(TABLE_A7_DATA, TABLE_METADATA["table_a7"], "tableA7")


def generate_table_a8_tex() -> str:
    return generate_flat_table_tex(TABLE_A8_DATA, TABLE_METADATA["table_a8"], "tableA8")


# =============================================================================
# Main Generation Functions
# =============================================================================

MARKDOWN_GENERATORS = {
    "table_1": generate_table_1_md,
    "table_2": generate_table_2_md,
    "table_3": generate_table_3_md,
    "table_4": generate_table_4_md,
    "table_5": generate_table_5_md,
    "table_6": generate_table_6_md,
    "table_7": generate_table_7_md,
    # Online Appendix
    "table_a1": generate_table_a1_md,
    "table_a2": generate_table_a2_md,
    "table_a3": generate_table_a3_md,
    # Additional Robustness Tests
    "table_a4": generate_table_a4_md,
    "table_a5": generate_table_a5_md,
    "table_a5b": generate_table_a5b_md,
    "table_a6": generate_table_a6_md,
    # Mechanism Tests
    "table_a7": generate_table_a7_md,
    "table_a8": generate_table_a8_md,
}

LATEX_GENERATORS = {
    "table_1": generate_table_1_tex,
    "table_2": generate_table_2_tex,
    "table_3": generate_table_3_tex,
    "table_4": generate_table_4_tex,
    "table_5": generate_table_5_tex,
    "table_6": generate_table_6_tex,
    "table_7": generate_table_7_tex,
    # Online Appendix
    "table_a1": generate_table_a1_tex,
    "table_a2": generate_table_a2_tex,
    "table_a3": generate_table_a3_tex,
    # Additional Robustness Tests
    "table_a4": generate_table_a4_tex,
    "table_a5": generate_table_a5_tex,
    "table_a5b": generate_table_a5b_tex,
    "table_a6": generate_table_a6_tex,
    # Mechanism Tests
    "table_a7": generate_table_a7_tex,
    "table_a8": generate_table_a8_tex,
}


def generate_all_tables(output_dir: Optional[Path] = None) -> Dict[str, Path]:
    """
    Generate all 7 tables in both markdown and LaTeX formats.
    
    Returns
    -------
    Dict[str, Path]
        Paths to generated files.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    generated_files = {}
    
    # Generate individual table files
    all_md = ["# Maps of the Sea: Publication Tables\n"]
    all_tex = [
        r"\documentclass{article}",
        r"\usepackage{booktabs}",
        r"\usepackage{threeparttable}",
        r"\usepackage[margin=1in]{geometry}",
        r"\newenvironment{tablenotes}{\begin{flushleft}\footnotesize}{\end{flushleft}}",
        r"\begin{document}",
        r"\title{Maps of the Sea: Publication Tables}",
        r"\maketitle",
        "",
    ]
    
    for table_key in MARKDOWN_GENERATORS.keys():
        # Markdown
        md_content = MARKDOWN_GENERATORS[table_key]()
        md_path = tables_dir / f"{table_key}.md"
        md_path.write_text(md_content)
        generated_files[f"{table_key}_md"] = md_path
        all_md.append(md_content)
        all_md.append("\n---\n")
        
        # LaTeX
        tex_content = LATEX_GENERATORS[table_key]()
        tex_path = tables_dir / f"{table_key}.tex"
        tex_path.write_text(tex_content)
        generated_files[f"{table_key}_tex"] = tex_path
        all_tex.append(tex_content)
        all_tex.append(r"\clearpage")
        all_tex.append("")
    
    # Write combined files
    all_md_path = output_dir / "all_tables.md"
    all_md_path.write_text("\n".join(all_md))
    generated_files["all_tables_md"] = all_md_path
    
    all_tex.append(r"\end{document}")
    all_tex_path = output_dir / "all_tables.tex"
    all_tex_path.write_text("\n".join(all_tex))
    generated_files["all_tables_tex"] = all_tex_path
    
    return generated_files


def print_summary(generated_files: Dict[str, Path]) -> None:
    """Print summary of generated files."""
    print("\n" + "=" * 60)
    print("PAPER TABLES GENERATED")
    print("=" * 60)
    
    print("\nGenerated files:")
    for name, path in generated_files.items():
        print(f"  - {name}: {path}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    print("Generating paper tables for 'Maps of the Sea' manuscript...")
    files = generate_all_tables()
    print_summary(files)
