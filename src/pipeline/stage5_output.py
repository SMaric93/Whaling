"""
Stage 5: Write Output

Generates all publication-ready outputs in both MD and TEX formats.

Outputs:
    - Main text tables (Tables 1-6)
    - Appendix tables (Tables A1-A8)
    - All tables combined (all_tables.md, all_tables.tex)
    - Figures (PNG format)
    - Summary documents
"""

import logging
from pathlib import Path
import shutil

logger = logging.getLogger(__name__)

# Output directories
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'output'
PAPER_DIR = OUTPUT_DIR / 'paper'
TABLES_DIR = PAPER_DIR / 'tables'
FIGURES_DIR = OUTPUT_DIR / 'figures'


def ensure_output_dirs() -> None:
    """Ensure all output directories exist."""
    for d in [PAPER_DIR, TABLES_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def generate_main_tables_md() -> list:
    """Generate main text tables as markdown."""
    from src.analyses.paper_tables import (
        generate_table_1, generate_table_2, generate_table_3,
        generate_table_4, generate_table_5, generate_table_6,
    )
    
    logger.info("Generating main text tables (MD)...")
    
    tables_generated = []
    
    table_generators = [
        ('table_1.md', generate_table_1, "Summary Statistics"),
        ('table_2.md', generate_table_2, "Information Control Through Ground Selection"),
        ('table_3.md', generate_table_3, "Variance Decomposition"),
        ('table_4.md', generate_table_4, "Search Geometry Movers Design"),
        ('table_5.md', generate_table_5, "Complementarity by Ground Type"),
        ('table_6.md', generate_table_6, "CATE of Agent Capability"),
    ]
    
    for filename, generator, title in table_generators:
        try:
            content = generator()
            output_path = TABLES_DIR / filename
            output_path.write_text(content)
            tables_generated.append(filename)
            logger.info(f"  - {title} → {filename}")
        except Exception as e:
            logger.warning(f"  - {title} failed: {e}")
    
    return tables_generated


def generate_appendix_tables_md() -> list:
    """Generate appendix tables as markdown."""
    from src.analyses.paper_tables import (
        generate_table_a1, generate_table_a2, generate_table_a3,
        generate_table_a4, generate_table_a5, generate_table_a6,
        generate_table_a7, generate_table_a8,
    )
    
    logger.info("Generating appendix tables (MD)...")
    
    tables_generated = []
    
    table_generators = [
        ('table_a1.md', generate_table_a1, "Robustness of Variance Decomposition"),
        ('table_a2.md', generate_table_a2, "Robustness to Scarcity Definitions"),
        ('table_a3.md', generate_table_a3, "Event Study of Search Geometry"),
        ('table_a4.md', generate_table_a4, "Variance Ratios by Treatment Cell"),
        ('table_a5.md', generate_table_a5, "Quantile Regression Results"),
        ('table_a6.md', generate_table_a6, "Optimal Foraging Stopping Rule"),
        ('table_a7.md', generate_table_a7, "Weather and Mechanism Tests"),
        ('table_a8.md', generate_table_a8, "Context-Dependent Matching"),
    ]
    
    for filename, generator, title in table_generators:
        try:
            content = generator()
            output_path = TABLES_DIR / filename
            output_path.write_text(content)
            tables_generated.append(filename)
            logger.info(f"  - {title} → {filename}")
        except Exception as e:
            logger.warning(f"  - {title} failed: {e}")
    
    return tables_generated


def generate_all_tables_tex() -> str:
    """Generate combined LaTeX file with all tables."""
    from src.analyses.paper_tables import generate_all_latex_tables
    
    logger.info("Generating combined LaTeX tables...")
    
    try:
        content = generate_all_latex_tables()
        output_path = PAPER_DIR / 'all_tables.tex'
        output_path.write_text(content)
        logger.info(f"  → all_tables.tex ({len(content):,} bytes)")
        return str(output_path)
    except Exception as e:
        logger.error(f"LaTeX generation failed: {e}")
        return None


def generate_all_tables_md() -> str:
    """Generate combined markdown file with all tables."""
    from src.analyses.paper_tables import generate_all_markdown_tables
    
    logger.info("Generating combined markdown tables...")
    
    try:
        content = generate_all_markdown_tables()
        output_path = PAPER_DIR / 'all_tables.md'
        output_path.write_text(content)
        logger.info(f"  → all_tables.md ({len(content):,} bytes)")
        return str(output_path)
    except Exception as e:
        logger.error(f"Combined markdown generation failed: {e}")
        return None


def copy_figures_to_paper() -> int:
    """Copy all figures to paper directory."""
    logger.info("Copying figures to paper directory...")
    
    paper_figures_dir = PAPER_DIR / 'figures'
    paper_figures_dir.mkdir(parents=True, exist_ok=True)
    
    copied_count = 0
    
    # Copy from main figures directory
    if FIGURES_DIR.exists():
        for fig in FIGURES_DIR.glob('*.png'):
            shutil.copy2(fig, paper_figures_dir / fig.name)
            copied_count += 1
    
    # Copy from baseline_loo_eb figures
    baseline_figs = OUTPUT_DIR / 'baseline_loo_eb' / 'figures'
    if baseline_figs.exists():
        for fig in baseline_figs.glob('*.png'):
            shutil.copy2(fig, paper_figures_dir / fig.name)
            copied_count += 1
    
    logger.info(f"  → Copied {copied_count} figures")
    return copied_count


def generate_robustness_summary() -> None:
    """Generate robustness results summary."""
    logger.info("Generating robustness summary...")
    
    # Collect robustness results
    summary_content = """# Robustness Results Summary

## Three Key Robustness Tests

### 1. Vessel Mover Design
Tests whether agent effects persist when controlling for vessel quality.

### 2. Optimal Foraging Stopping Rule
Validates that high-ψ agents implement efficient patch-leaving.

### 3. Insurance Variance Validation
Confirms that high-ψ agents compress novice captain variance (floor-raising).

See `output/paper/robustness_three_tests.md` for full results.
"""
    
    output_path = PAPER_DIR / 'robustness_results.md'
    output_path.write_text(summary_content)
    logger.info(f"  → robustness_results.md")


def generate_mechanism_summary() -> None:
    """Generate mechanism analysis summary."""
    logger.info("Generating mechanism summary...")
    
    summary_content = """# Mechanism Analysis Results

## Weather Allocation
High-ψ agents allocate captains more efficiently to weather conditions.

## Crew Mechanism
Agent capability affects hiring quality, retention, and discipline.

## Context-Dependent Matching
Sorting patterns switch between PAM and NAM based on operational context.

See `output/baseline_loo_eb/mechanism_tests/` for full results.
"""
    
    output_path = PAPER_DIR / 'mechanism_analysis_results.md'
    output_path.write_text(summary_content)
    logger.info(f"  → mechanism_analysis_results.md")


def run_output(include_figures: bool = True) -> dict:
    """
    Run the complete output generation stage.
    
    Args:
        include_figures: Also copy figures to paper directory
    
    Returns:
        dict: Summary of output generation
    """
    logger.info("=" * 60)
    logger.info("STAGE 5: WRITE OUTPUT (MD + TEX)")
    logger.info("=" * 60)
    
    ensure_output_dirs()
    
    results = {
        'main_tables_md': [],
        'appendix_tables_md': [],
        'all_tables_tex': None,
        'all_tables_md': None,
        'figures_copied': 0,
    }
    
    # Generate individual tables (MD)
    try:
        results['main_tables_md'] = generate_main_tables_md()
    except Exception as e:
        logger.error(f"Main tables MD failed: {e}")
    
    try:
        results['appendix_tables_md'] = generate_appendix_tables_md()
    except Exception as e:
        logger.error(f"Appendix tables MD failed: {e}")
    
    # Generate combined files
    try:
        results['all_tables_tex'] = generate_all_tables_tex()
    except Exception as e:
        logger.error(f"Combined TEX failed: {e}")
    
    try:
        results['all_tables_md'] = generate_all_tables_md()
    except Exception as e:
        logger.error(f"Combined MD failed: {e}")
    
    # Copy figures
    if include_figures:
        try:
            results['figures_copied'] = copy_figures_to_paper()
        except Exception as e:
            logger.warning(f"Figure copying failed: {e}")
    
    # Generate summaries
    try:
        generate_robustness_summary()
        generate_mechanism_summary()
    except Exception as e:
        logger.warning(f"Summary generation failed: {e}")
    
    # Summary
    total_tables = len(results['main_tables_md']) + len(results['appendix_tables_md'])
    logger.info(f"Stage 5 complete: {total_tables} tables generated")
    logger.info(f"  Output directory: {PAPER_DIR}")
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_output()
