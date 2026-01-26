"""
QA report generation for data pipeline.

Produces qa_report.md with merge rates, diagnostics, and data quality metrics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FINAL_DIR, STAGING_DIR, DOCS_DIR
from qa.validators import (
    validate_analysis_voyage,
    validate_analysis_captain_year,
    ValidationResult,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_missingness(df: pd.DataFrame) -> Dict[str, float]:
    """Compute missingness rate for each column."""
    return {
        col: df[col].isna().mean()
        for col in df.columns
    }


def compute_merge_rates(
    voyage_df: pd.DataFrame,
    captain_df: pd.DataFrame,
) -> Dict[str, float]:
    """Compute merge/coverage rates."""
    rates = {}
    
    # Voyage-level coverage
    if "has_labor_data" in voyage_df.columns:
        rates["voyage_labor_coverage"] = voyage_df["has_labor_data"].mean()
    
    if "has_route_data" in voyage_df.columns:
        rates["voyage_route_coverage"] = voyage_df["has_route_data"].mean()
    
    if "has_vqi_data" in voyage_df.columns:
        rates["voyage_vqi_coverage"] = voyage_df["has_vqi_data"].mean()
    
    # Captain-level coverage
    if "has_census_link" in captain_df.columns:
        rates["captain_census_link_rate"] = captain_df["has_census_link"].mean()
    
    if "has_wealth_data" in captain_df.columns:
        rates["captain_wealth_coverage"] = captain_df["has_wealth_data"].mean()
    
    return rates


def generate_qa_report(
    output_path: Optional[Path] = None,
) -> str:
    """
    Generate comprehensive QA report.
    
    Returns:
        Markdown report string
    """
    if output_path is None:
        output_path = DOCS_DIR / "qa_report.md"
    
    sections = []
    
    # Header
    sections.append(f"""# Data Quality Assurance Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

This report summarizes data quality metrics for the Whaling Data Pipeline outputs.
""")
    
    # Load files
    voyage_path = FINAL_DIR / "analysis_voyage.parquet"
    captain_path = FINAL_DIR / "analysis_captain_year.parquet"
    
    voyage_df = None
    captain_df = None
    
    if voyage_path.exists():
        voyage_df = pd.read_parquet(voyage_path)
        sections.append(f"""
## Analysis Voyage Summary

- **Total voyages**: {len(voyage_df):,}
- **Unique vessels**: {voyage_df['vessel_id'].nunique() if 'vessel_id' in voyage_df.columns else 'N/A':,}
- **Unique captains**: {voyage_df['captain_id'].nunique() if 'captain_id' in voyage_df.columns else 'N/A':,}
- **Year range**: {voyage_df['year_out'].min()} - {voyage_df['year_out'].max()}
""")
    
    if captain_path.exists():
        captain_df = pd.read_parquet(captain_path)
        sections.append(f"""
## Analysis Captain-Year Summary

- **Total captain-year observations**: {len(captain_df):,}
- **Unique captains**: {captain_df['captain_id'].nunique():,}
- **Census years covered**: {sorted(captain_df['census_year'].unique().tolist())}
""")
    
    # Merge rates
    if voyage_df is not None and captain_df is not None:
        rates = compute_merge_rates(voyage_df, captain_df)
        
        sections.append("""
## Merge Coverage Rates

| Metric | Rate |
|--------|------|""")
        
        for metric, rate in rates.items():
            sections.append(f"| {metric} | {rate:.1%} |")
    
    # Validation results
    sections.append("""
## Validation Checks
""")
    
    if voyage_df is not None:
        voyage_validations = validate_analysis_voyage(voyage_df)
        
        sections.append("""
### Voyage Validations

| Check | Status | Details |
|-------|--------|---------|""")
        
        for v in voyage_validations:
            status = "✅ Pass" if v.passed else "❌ Fail"
            sections.append(f"| {v.check_name} | {status} | {v.message} |")
    
    if captain_df is not None:
        captain_validations = validate_analysis_captain_year(captain_df)
        
        sections.append("""
### Captain-Year Validations

| Check | Status | Details |
|-------|--------|---------|""")
        
        for v in captain_validations:
            status = "✅ Pass" if v.passed else "❌ Fail"
            sections.append(f"| {v.check_name} | {status} | {v.message} |")
    
    # Missingness summary
    if voyage_df is not None:
        voyage_missing = compute_missingness(voyage_df)
        
        # Top missing columns
        top_missing = sorted(
            [(k, v) for k, v in voyage_missing.items() if v > 0.01],
            key=lambda x: x[1],
            reverse=True
        )[:15]
        
        if top_missing:
            sections.append("""
## Missingness Summary (Voyage)

| Column | Missing Rate |
|--------|--------------|""")
            
            for col, rate in top_missing:
                sections.append(f"| {col} | {rate:.1%} |")
    
    # Key distributions
    if voyage_df is not None:
        sections.append("""
## Key Variable Distributions

### Oil Output (q_oil_bbl)
""")
        if "q_oil_bbl" in voyage_df.columns:
            q = voyage_df["q_oil_bbl"].dropna()
            sections.append(f"""
- Count: {len(q):,}
- Mean: {q.mean():,.0f}
- Median: {q.median():,.0f}
- Max: {q.max():,.0f}
""")
    
    # Linkage diagnostics
    linkage_path = STAGING_DIR / "captain_to_ipums.parquet"
    if linkage_path.exists():
        linkage_df = pd.read_parquet(linkage_path)
        
        sections.append("""
## Captain-Census Linkage Diagnostics
""")
        
        best = linkage_df[linkage_df["match_rank"] == 1]
        
        sections.append(f"""
- **Total link candidates**: {len(linkage_df):,}
- **Captains with matches**: {best['captain_id'].nunique():,}
- **Mean best match score**: {best['match_score'].mean():.3f}
- **Median best match score**: {best['match_score'].median():.3f}

### Score Distribution

| Percentile | Score |
|------------|-------|
| 10th | {best['match_score'].quantile(0.1):.3f} |
| 25th | {best['match_score'].quantile(0.25):.3f} |
| 50th | {best['match_score'].quantile(0.5):.3f} |
| 75th | {best['match_score'].quantile(0.75):.3f} |
| 90th | {best['match_score'].quantile(0.9):.3f} |

### Match Methods

| Method | Count |
|--------|-------|""")
        
        for method, count in best["match_method"].value_counts().items():
            sections.append(f"| {method} | {count:,} |")
    
    # Recommendations
    sections.append("""
## Recommendations

1. Review any failed validation checks above
2. Consider thresholds for captain-census linkage based on your analysis needs
3. Document any data quality issues discovered during analysis
""")
    
    # Compile report
    report = "\n".join(sections)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    
    logger.info(f"Generated QA report: {output_path}")
    
    return report


if __name__ == "__main__":
    report = generate_qa_report()
    print(report)
