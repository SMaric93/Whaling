"""
Stage 5: Write Output

Generates all publication-ready outputs in both MD and TEX formats
using the data-driven ``src.paper`` build layer.

Outputs:
    - Main text tables (Tables 1-10)
    - Appendix tables (Tables A1-A18)
    - LaTeX renderings
    - Figures (PNG format)
    - Summary documents
"""

import logging
from pathlib import Path
import shutil

from src.pipeline._runner import StepSpec, run_steps

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


def generate_paper_outputs() -> dict:
    """Generate all paper outputs using the data-driven paper layer."""
    from src.paper.build_all import build_all

    return build_all()


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
        'paper_outputs': None,
        'figures_copied': 0,
    }

    run_steps(
        results,
        [
            StepSpec('paper_outputs', generate_paper_outputs, "Paper output generation failed", failure_level="error"),
        ],
        logger=logger,
    )

    # Copy figures
    if include_figures:
        run_steps(
            results,
            [StepSpec('figures_copied', copy_figures_to_paper, "Figure copying failed")],
            logger=logger,
        )

    # Generate summaries
    try:
        generate_robustness_summary()
        generate_mechanism_summary()
    except Exception as e:
        logger.warning(f"Summary generation failed: {e}")

    logger.info("Stage 5 complete")
    logger.info(f"  Output directory: {PAPER_DIR}")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_output()
