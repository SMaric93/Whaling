#!/usr/bin/env python3
"""
Whaling Data Pipeline - Main Runner

A 5-stage pipeline for the "Venture Capital of the Sea" project:

    Stage 1: PULL     - Download all raw data sources
    Stage 2: CLEAN    - Parse and standardize data  
    Stage 3: MERGE    - Assemble and link datasets
    Stage 4: ANALYZE  - Run full analysis suite
    Stage 5: OUTPUT   - Generate MD and TEX outputs

Usage:
    python run_pipeline.py --help
    python run_pipeline.py pull       # Stage 1: Download data
    python run_pipeline.py clean      # Stage 2: Parse and standardize
    python run_pipeline.py merge      # Stage 3: Assemble and link
    python run_pipeline.py analyze    # Stage 4: Run all analyses
    python run_pipeline.py output     # Stage 5: Generate MD/TEX
    python run_pipeline.py all        # Run complete pipeline

Legacy Commands (for backward compatibility):
    python run_pipeline.py download          # Alias for 'pull'
    python run_pipeline.py parse             # Alias for 'clean'
    python run_pipeline.py assemble-voyages  # Part of 'merge'
    python run_pipeline.py augment-all       # Part of 'merge'
"""

import argparse
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))


# =============================================================================
# NEW 5-STAGE PIPELINE COMMANDS
# =============================================================================

def cmd_pull(args):
    """Stage 1: Download all data sources."""
    from src.pipeline import run_pull
    run_pull(force=args.force)


def cmd_clean(args):
    """Stage 2: Parse and standardize data."""
    from src.pipeline import run_clean
    run_clean()


def cmd_merge(args):
    """Stage 3: Assemble and link datasets."""
    from src.pipeline import run_merge
    run_merge()


def cmd_analyze(args):
    """Stage 4: Run full analysis suite."""
    from src.pipeline import run_analyze
    run_analyze(quick=args.quick if hasattr(args, 'quick') else False)


def cmd_output(args):
    """Stage 5: Generate MD and TEX outputs."""
    from src.pipeline import run_output
    run_output()


def cmd_all(args):
    """Run complete 5-stage pipeline."""
    from src.pipeline import run_full_pipeline
    run_full_pipeline(
        force_pull=args.force if hasattr(args, 'force') else False,
        quick_analyze=args.quick if hasattr(args, 'quick') else False,
    )


# =============================================================================
# LEGACY COMMANDS (Backward Compatibility)
# =============================================================================

def cmd_download_legacy(args):
    """Legacy: Stage 1 download (alias for 'pull')."""
    cmd_pull(args)


def cmd_parse_legacy(args):
    """Legacy: Stage 2 parse (alias for 'clean')."""
    cmd_clean(args)


def cmd_aggregate_legacy(args):
    """Legacy: Aggregate metrics (now part of 'merge')."""
    logger.info("Note: 'aggregate' is now part of the 'merge' stage")
    from src.pipeline.stage3_merge import compute_labor_metrics, compute_route_exposure
    compute_labor_metrics()
    compute_route_exposure()


def cmd_assemble_voyages_legacy(args):
    """Legacy: Assemble voyages (now part of 'merge')."""
    logger.info("Note: 'assemble-voyages' is now part of the 'merge' stage")
    from src.pipeline.stage3_merge import assemble_voyages
    assemble_voyages()


def cmd_link_captains_legacy(args):
    """Legacy: Link captains (now part of 'merge')."""
    logger.info("Note: 'link-captains' is now part of the 'merge' stage")
    from src.pipeline.stage3_merge import link_captains
    link_captains()


def cmd_assemble_captains_legacy(args):
    """Legacy: Assemble captains (now part of 'merge')."""
    logger.info("Note: 'assemble-captains' is now part of the 'merge' stage")
    from src.pipeline.stage3_merge import assemble_captains
    assemble_captains()


def cmd_qa_legacy(args):
    """Legacy: QA report (now part of 'output')."""
    logger.info("Note: 'qa' is now part of the 'output' stage")
    from src.qa import generate_qa_report
    from src.config import DOCS_DIR
    report = generate_qa_report()
    logger.info(f"QA report saved to {DOCS_DIR / 'qa_report.md'}")


def cmd_augment_all_legacy(args):
    """Legacy: Run augmentation pipeline (now part of 'merge')."""
    logger.info("Note: 'augment-all' is now part of the 'merge' stage")
    from src.pipeline.stage3_merge import augment_voyages, merge_climate_data
    augment_voyages()
    merge_climate_data()


def cmd_all_legacy(args):
    """Legacy: Run full pipeline using old stages."""
    logger.info("Running legacy full pipeline...")
    logger.info("Consider using 'python run_pipeline.py all' for the new 5-stage pipeline")
    
    # Import legacy functions
    from config import FINAL_DIR
    
    cmd_download_legacy(args)
    cmd_parse_legacy(args)
    cmd_aggregate_legacy(args)
    cmd_assemble_voyages_legacy(args)
    cmd_link_captains_legacy(args)
    cmd_assemble_captains_legacy(args)
    cmd_qa_legacy(args)
    
    logger.info("Pipeline complete!")
    logger.info(f"Output files in: {FINAL_DIR}")


# =============================================================================
# MAIN CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Whaling Data Pipeline - Venture Capital of the Sea",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run the complete 5-stage pipeline
    python run_pipeline.py all
    
    # Run individual stages
    python run_pipeline.py pull       # Download data
    python run_pipeline.py clean      # Parse and standardize
    python run_pipeline.py merge      # Assemble and link
    python run_pipeline.py analyze    # Run analyses
    python run_pipeline.py output     # Generate MD/TEX
    
    # Quick analysis (main regressions only)
    python run_pipeline.py analyze --quick
"""
    )
    
    parser.add_argument(
        "stage",
        choices=[
            # New 5-stage commands
            "pull", "clean", "merge", "analyze", "output", "all",
            # Legacy commands (for backward compatibility)
            "download", "parse", "aggregate",
            "assemble-voyages", "link-captains", "assemble-captains",
            "qa", "augment-all",
            # Augmentation sub-stages (legacy)
            "download-online", "extract-wsl", "crosswalk-wsl",
            "starbuck", "maury", "augment",
        ],
        help="Pipeline stage to run"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download/reprocess even if files exist"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick analysis (main regressions only, skip extended analyses)"
    )
    
    args = parser.parse_args()
    
    # New 5-stage commands
    stage_handlers = {
        "pull": cmd_pull,
        "clean": cmd_clean,
        "merge": cmd_merge,
        "analyze": cmd_analyze,
        "output": cmd_output,
        "all": cmd_all,
        # Legacy mappings
        "download": cmd_download_legacy,
        "parse": cmd_parse_legacy,
        "aggregate": cmd_aggregate_legacy,
        "assemble-voyages": cmd_assemble_voyages_legacy,
        "link-captains": cmd_link_captains_legacy,
        "assemble-captains": cmd_assemble_captains_legacy,
        "qa": cmd_qa_legacy,
        "augment-all": cmd_augment_all_legacy,
    }
    
    # Handle augmentation sub-stages (legacy)
    augment_stages = ["download-online", "extract-wsl", "crosswalk-wsl", 
                      "starbuck", "maury", "augment"]
    if args.stage in augment_stages:
        logger.info(f"Note: '{args.stage}' is now part of the 'merge' stage")
        logger.info("Running legacy augmentation stage...")
        # Import and run from old module
        try:
            if args.stage == "download-online":
                from src.download.online_sources_downloader import download_all_online_sources
                download_all_online_sources(force=args.force)
            elif args.stage == "extract-wsl":
                from src.parsing.wsl_event_extractor import extract_all_wsl_events
                from src.config import RAW_DIR, STAGING_DIR
                wsl_dir = RAW_DIR / 'wsl_pdfs'
                if wsl_dir.exists():
                    df = extract_all_wsl_events(wsl_dir)
                    df.to_parquet(STAGING_DIR / 'wsl_events.parquet', index=False)
            # Add other legacy handlers as needed
        except Exception as e:
            logger.error(f"Legacy stage failed: {e}")
        return
    
    # Run the selected stage
    handler = stage_handlers.get(args.stage)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
