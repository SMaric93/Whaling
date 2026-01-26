"""
Output generation for regression exhibits.

Produces tables, figures, and summary documents for the whaling
empirical analysis module.
"""

from pathlib import Path
from typing import Dict, List, Optional
import warnings

import numpy as np
import pandas as pd

from .config import (
    TABLES_DIR,
    FIGURES_DIR,
    DIAGNOSTICS_DIR,
    SUMMARY_DIR,
    REGRESSIONS,
    DEFAULT_EXHIBITS,
)

warnings.filterwarnings("ignore", category=FutureWarning)


def format_coefficient(value: float, se: Optional[float] = None, stars: str = "") -> str:
    """Format coefficient for table display."""
    if pd.isna(value):
        return ""
    formatted = f"{value:.4f}{stars}"
    if se is not None and not pd.isna(se):
        formatted += f"\n({se:.4f})"
    return formatted


def get_significance_stars(pvalue: float) -> str:
    """Get significance stars based on p-value."""
    if pd.isna(pvalue):
        return ""
    if pvalue < 0.01:
        return "***"
    elif pvalue < 0.05:
        return "**"
    elif pvalue < 0.10:
        return "*"
    return ""


def generate_variance_decomposition_table(
    r1_results: Dict,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Generate variance decomposition table (R1 main exhibit).
    
    Parameters
    ----------
    r1_results : Dict
        Results from baseline_production.run_r1.
    output_path : Path, optional
        Path to save table.
        
    Returns
    -------
    pd.DataFrame
        Formatted variance decomposition table.
    """
    if output_path is None:
        output_path = TABLES_DIR / "table_variance_decomposition.csv"
    
    decomp = r1_results.get("variance_decomposition")
    if decomp is None:
        print("Variance decomposition not found in results")
        return None
    
    # Format for presentation
    table = decomp.copy()
    
    # Plugin_Share and KSS_Share now refer to shares WITHIN captain+agent variance
    # which is the correct AKM interpretation (shares sum to ~100% for first 3 rows)
    if "Plugin_Share_Pct" not in table.columns:
        table["Plugin_Share_Pct"] = (table["Plugin_Share"] * 100).round(1)
    else:
        table["Plugin_Share_Pct"] = table["Plugin_Share_Pct"].round(1)
        
    if "KSS_Share_Pct" not in table.columns:
        table["KSS_Share_Pct"] = (table["KSS_Share"] * 100).round(1)
    else:
        table["KSS_Share_Pct"] = table["KSS_Share_Pct"].round(1)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    
    print(f"Variance decomposition table saved to {output_path}")
    return table


def generate_main_regression_table(
    all_results: Dict,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Generate main regression results table.
    
    Parameters
    ----------
    all_results : Dict
        All regression results.
    output_path : Path, optional
        Path to save table.
        
    Returns
    -------
    pd.DataFrame
        Formatted regression table.
    """
    if output_path is None:
        output_path = TABLES_DIR / "table_main_regressions.csv"
    
    rows = []
    
    # R1: Baseline
    if "r1" in all_results:
        r1 = all_results["r1"]
        rows.append({
            "Panel": "A. Baseline Production Function",
            "Specification": "R1",
            "Key_Coefficient": "Var(α)/Var(y)",
            "Value": r1.get("kss", {}).get("var_alpha_kss", np.nan) / np.var(r1.get("y", [1])) if r1.get("y") is not None else np.nan,
            "R2": r1.get("r2", np.nan),
            "N": r1.get("n", np.nan),
        })
    
    # R2: Portability
    if "r2" in all_results:
        r2 = all_results["r2"]
        rows.append({
            "Panel": "B. Portability Validation",
            "Specification": "R2",
            "Key_Coefficient": "b (α̂_train)",
            "Value": r2.get("b_alpha_full", np.nan),
            "R2": r2.get("r2_full", np.nan),
            "N": r2.get("n_test_with_alpha", np.nan),
        })
    
    # R6: Resilience
    if "r6" in all_results:
        r6 = all_results["r6"]
        rows.append({
            "Panel": "C. Resilience",
            "Specification": "R6",
            "Key_Coefficient": "φ (Z × HighCap)",
            "Value": r6.get("phi", np.nan),
            "R2": r6.get("r2", np.nan),
            "N": r6.get("n", np.nan),
        })
    
    # R9: Pass-through
    if "r9" in all_results:
        r9 = all_results["r9"]
        rows.append({
            "Panel": "D. Shock Pass-Through",
            "Specification": "R9",
            "Key_Coefficient": "φ (Ice × HighCap)",
            "Value": r9.get("phi", np.nan),
            "R2": r9.get("r2", np.nan),
            "N": r9.get("n", np.nan),
        })
    
    # R13: Sorting
    if "r13" in all_results:
        r13 = all_results["r13"]
        rows.append({
            "Panel": "E. Labor Market",
            "Specification": "R13",
            "Key_Coefficient": "Sorting (b)",
            "Value": r13.get("b_sorting", np.nan),
            "R2": r13.get("r2", np.nan),
            "N": r13.get("n", np.nan),
        })
    
    # R14: Switching
    if "r14" in all_results:
        r14 = all_results["r14"]
        rows.append({
            "Panel": "E. Labor Market",
            "Specification": "R14",
            "Key_Coefficient": "γ̂ → Switch",
            "Value": r14.get("b_gamma", np.nan),
            "R2": r14.get("r2", np.nan),
            "N": r14.get("n", np.nan),
        })
    
    table = pd.DataFrame(rows)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    
    print(f"Main regression table saved to {output_path}")
    return table


def generate_connected_set_diagnostics(
    diagnostics: Dict,
    output_path: Optional[Path] = None,
) -> None:
    """
    Generate connected set diagnostics report.
    
    Parameters
    ----------
    diagnostics : Dict
        Diagnostics from connected_set.full_connected_set_analysis.
    output_path : Path, optional
        Path to save report.
    """
    if output_path is None:
        output_path = DIAGNOSTICS_DIR / "connected_set_diagnostics.md"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write("# Connected Set Diagnostics\n\n")
        
        # Standard connected set
        if "connected_set" in diagnostics:
            cc = diagnostics["connected_set"]
            f.write("## Standard Connected Set\n\n")
            f.write(f"- **Components**: {cc.get('n_components', 'N/A')}\n")
            f.write(f"- **Captains**: {cc.get('largest_component_captains', 'N/A'):,} ({100*cc.get('coverage_captains', 0):.1f}%)\n")
            f.write(f"- **Agents**: {cc.get('largest_component_agents', 'N/A'):,} ({100*cc.get('coverage_agents', 0):.1f}%)\n")
            f.write(f"- **Voyages**: {cc.get('voyages_in_connected_set', 'N/A'):,} ({100*cc.get('coverage_voyages', 0):.1f}%)\n\n")
        
        # LOO connected set
        if "loo_set" in diagnostics:
            loo = diagnostics["loo_set"]
            f.write("## Leave-One-Out Connected Set (KSS)\n\n")
            f.write(f"- **Voyages**: {loo.get('loo_voyages', 'N/A'):,} ({100*loo.get('coverage', 0):.1f}%)\n")
            f.write(f"- **Captains**: {loo.get('loo_captains', 'N/A'):,}\n")
            f.write(f"- **Agents**: {loo.get('loo_agents', 'N/A'):,}\n")
            f.write(f"- **Articulation edges**: {loo.get('articulation_edges', 'N/A'):,} ({100*loo.get('articulation_rate', 0):.1f}%)\n\n")
        
        # Mobility
        if "mobility" in diagnostics:
            mob = diagnostics["mobility"]
            f.write("## Mobility Diagnostics\n\n")
            f.write(f"- **Multi-agent captains**: {mob.get('multi_agent_captains', 'N/A'):,} ({100*mob.get('multi_agent_captain_rate', 0):.1f}%)\n")
            f.write(f"- **Multi-captain agents**: {mob.get('multi_captain_agents', 'N/A'):,} ({100*mob.get('multi_captain_agent_rate', 0):.1f}%)\n")
            f.write(f"- **Repeat partnerships**: {100*mob.get('repeat_pair_rate', 0):.1f}%\n")
    
    print(f"Connected set diagnostics saved to {output_path}")


def generate_executive_summary(
    all_results: Dict,
    output_path: Optional[Path] = None,
) -> None:
    """
    Generate executive summary of findings.
    
    Parameters
    ----------
    all_results : Dict
        All regression results.
    output_path : Path, optional
        Path to save summary.
    """
    if output_path is None:
        output_path = SUMMARY_DIR / "executive_summary.md"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write("# Whaling Empirical Analysis: Executive Summary\n\n")
        
        f.write("## Key Findings\n\n")
        
        # R1: Variance decomposition
        if "r1" in all_results:
            r1 = all_results["r1"]
            kss = r1.get("kss", {})
            
            # Get the captain+agent variance total and shares
            captain_agent_total = r1.get("captain_agent_var_kss", None)
            captain_agent_share_of_y = r1.get("captain_agent_share_of_var_y", 0) * 100
            
            # Compute shares within captain+agent variance
            var_alpha_kss = kss.get("var_alpha_kss", 0)
            var_gamma_kss = kss.get("var_gamma_kss", 0)
            cov_kss = kss.get("cov_kss", 0)
            
            if captain_agent_total and captain_agent_total > 0:
                captain_share = var_alpha_kss / captain_agent_total * 100
                agent_share = var_gamma_kss / captain_agent_total * 100
                sorting_share = 2 * cov_kss / captain_agent_total * 100
            else:
                # Fallback calculation
                total = var_alpha_kss + var_gamma_kss + 2 * cov_kss
                captain_share = var_alpha_kss / total * 100 if total > 0 else 0
                agent_share = var_gamma_kss / total * 100 if total > 0 else 0
                sorting_share = 2 * cov_kss / total * 100 if total > 0 else 0
            
            f.write("### 1. Variance Decomposition (R1)\n\n")
            
            # Handle case where captain+agent total exceeds Var(y)
            # This happens when other FEs are negatively correlated with α+γ
            if captain_agent_share_of_y > 100:
                f.write("- **Labor market components** (captain + agent) variance is large relative to outcome variance\n")
                f.write("- This reflects negative covariance with vessel×period and route×time FEs\n\n")
            
            f.write("**Within captain+agent variance** (KSS-corrected):\n")
            f.write(f"  - **Captain skill (α)**: {captain_share:.1f}%\n")
            f.write(f"  - **Agent capability (γ)**: {agent_share:.1f}%\n")
            f.write(f"  - **Sorting (2×Cov)**: {sorting_share:.1f}%\n")
            f.write(f"- Model R² = {r1.get('r2', 0):.3f}\n")
            f.write(f"- Corr(α, γ) = {r1.get('corr_alpha_gamma_kss', 0):.3f} (negative = substitutes)\n\n")
        
        # R2: Portability
        if "r2" in all_results:
            r2 = all_results["r2"]
            f.write("### 2. Portability Validation (R2)\n\n")
            f.write(f"- OOS prediction coefficient b = {r2.get('b_alpha_full', 0):.3f}\n")
            f.write(f"- Spearman rank correlation = {r2.get('spearman_r', 0):.3f}\n\n")
        
        # R6: Resilience
        if "r6" in all_results:
            r6 = all_results["r6"]
            f.write("### 3. Resilience Under Adversity (R6)\n\n")
            f.write(f"- Interaction coefficient φ = {r6.get('phi', 0):.4f}\n")
            if r6.get("phi", 0) > 0:
                f.write("- High-capability agents **attenuate** output losses under adversity\n\n")
            else:
                f.write("- No evidence of resilience advantage\n\n")
        
        # R13: Sorting
        if "r13" in all_results:
            r13 = all_results["r13"]
            f.write("### 4. Labor Market Sorting (R13)\n\n")
            f.write(f"- Sorting coefficient = {r13.get('b_sorting', 0):.4f}\n")
            f.write(f"- Raw correlation(α̂, γ̂) = {r13.get('raw_corr', 0):.4f}\n")
            f.write(f"- Diagonal share = {r13.get('diagonal_share', 0):.1f}% (random = 20%)\n\n")
        
        f.write("## Sample Information\n\n")
        if "r1" in all_results:
            f.write(f"- Analysis sample: {all_results['r1'].get('n', 0):,} voyages\n")
        
        f.write("\n---\n*Generated by Whaling Analyses Module*\n")
    
    print(f"Executive summary saved to {output_path}")


def generate_all_outputs(
    all_results: Dict,
    diagnostics: Optional[Dict] = None,
) -> None:
    """
    Generate all output exhibits.
    
    Parameters
    ----------
    all_results : Dict
        All regression results.
    diagnostics : Dict, optional
        Connected set diagnostics.
    """
    print("\n" + "=" * 60)
    print("GENERATING OUTPUT EXHIBITS")
    print("=" * 60)
    
    # Ensure output directories exist
    for dir_path in [TABLES_DIR, FIGURES_DIR, DIAGNOSTICS_DIR, SUMMARY_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Main regression table
    generate_main_regression_table(all_results)
    
    # Variance decomposition
    if "r1" in all_results:
        generate_variance_decomposition_table(all_results["r1"])
    
    # Connected set diagnostics
    if diagnostics:
        generate_connected_set_diagnostics(diagnostics)
    
    # Executive summary
    generate_executive_summary(all_results)
    
    print("\n" + "=" * 60)
    print("OUTPUT GENERATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    print("Run via run_all.py to generate outputs")
