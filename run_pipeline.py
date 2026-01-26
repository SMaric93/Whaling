#!/usr/bin/env python3
"""
Whaling Data Pipeline - Main Runner

Usage:
    python run_pipeline.py --help
    python run_pipeline.py download          # Download raw data
    python run_pipeline.py parse             # Parse and standardize
    python run_pipeline.py aggregate         # Compute metrics
    python run_pipeline.py assemble-voyages  # Build analysis_voyage
    python run_pipeline.py link-captains     # Run captain-census linkage
    python run_pipeline.py assemble-captains # Build analysis_captain_year
    python run_pipeline.py qa                # Generate QA report
    python run_pipeline.py all               # Run full pipeline
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import (
    PROJECT_ROOT, RAW_DIR, STAGING_DIR, FINAL_DIR, DOCS_DIR,
    LINKAGE_CONFIG,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def stage_download(force: bool = False):
    """Stage 1: Download raw data."""
    logger.info("=== Stage 1: Download Raw Data ===")
    
    from download import download_aowv_data, download_archive_data, ManifestManager
    
    manifest = ManifestManager()
    
    # Download AOWV datasets
    logger.info("Downloading AOWV datasets...")
    voyages, crews, logbooks = download_aowv_data(manifest, force=force)
    
    logger.info(f"  Voyages: {len(voyages)} files")
    logger.info(f"  Crew lists: {len(crews)} files")
    logger.info(f"  Logbooks: {len(logbooks)} files")
    
    # Download Archive.org register
    logger.info("Downloading vessel register...")
    register_files = download_archive_data(manifest, force=force)
    logger.info(f"  Register: {len(register_files)} files")
    
    logger.info(f"Manifest contains {len(manifest)} entries")


def stage_parse():
    """Stage 2: Parse and standardize data."""
    logger.info("=== Stage 2: Parse and Standardize ===")
    
    from parsing import VoyageParser, CrewParser, LogbookParser
    from parsing.register_parser import RegisterParser
    from entities import EntityResolver
    
    # Parse voyages
    logger.info("Parsing voyages...")
    voyage_parser = VoyageParser()
    voyages = voyage_parser.parse()
    
    # Add entity IDs
    logger.info("Resolving entities...")
    resolver = EntityResolver()
    voyages = resolver.resolve_voyages_df(voyages)
    resolver.save_registries()
    
    # Save voyages
    voyage_parser._parsed_df = voyages
    voyage_parser.save()
    
    summary = voyage_parser.get_summary()
    logger.info(f"  Voyages: {summary['total_voyages']}")
    logger.info(f"  Vessels: {summary['unique_vessels']}")
    logger.info(f"  Captains: {summary['unique_captains']}")
    
    # Parse crew lists
    logger.info("Parsing crew lists...")
    crew_parser = CrewParser()
    try:
        crew_parser.parse()
        crew_parser.save()
        crew_summary = crew_parser.get_summary()
        logger.info(f"  Crew records: {crew_summary['total_crew_records']}")
    except Exception as e:
        logger.warning(f"  Crew parsing failed: {e}")
    
    # Parse logbooks
    logger.info("Parsing logbooks...")
    logbook_parser = LogbookParser()
    try:
        logbook_parser.parse()
        logbook_parser.save()
        log_summary = logbook_parser.get_summary()
        logger.info(f"  Logbook observations: {log_summary['total_observations']}")
    except Exception as e:
        logger.warning(f"  Logbook parsing failed: {e}")
    
    # Parse vessel register
    logger.info("Parsing vessel register...")
    register_parser = RegisterParser()
    try:
        register_parser.parse()
        register_parser.save()
        reg_summary = register_parser.get_summary()
        logger.info(f"  Register entries: {reg_summary['total_entries']}")
    except Exception as e:
        logger.warning(f"  Register parsing failed: {e}")


def stage_aggregate():
    """Stage 3-5: Compute aggregated metrics."""
    logger.info("=== Stages 3-5: Aggregate Metrics ===")
    
    import pandas as pd
    from aggregation import compute_voyage_labor_metrics, compute_route_exposure
    from entities import CrosswalkBuilder
    
    # Load parsed data
    voyages_path = STAGING_DIR / "voyages_master.parquet"
    crew_path = STAGING_DIR / "crew_roster.parquet"
    logbook_path = STAGING_DIR / "logbook_positions.parquet"
    
    if voyages_path.exists():
        voyages = pd.read_parquet(voyages_path)
    else:
        logger.error("voyages_master not found, run parse stage first")
        return
    
    # Build crosswalks if needed
    crosswalk_builder = CrosswalkBuilder()
    
    # Aggregate crew metrics
    if crew_path.exists():
        logger.info("Computing labor metrics...")
        crew = pd.read_parquet(crew_path)
        
        # Build crosswalk if missing voyage_ids
        if crew["voyage_id"].isna().any():
            logger.info("  Building crew-voyage crosswalk...")
            crosswalk = crosswalk_builder.build_crew_to_voyage_crosswalk(
                crew, voyages,
                output_path=STAGING_DIR / "crosswalk_crew_voyage.csv"
            )
            crew = crosswalk_builder.apply_crosswalk(
                crew, crosswalk, source_index_col="crew_index"
            )
        
        labor_metrics = compute_voyage_labor_metrics(
            crew, output_path=STAGING_DIR / "voyage_labor_metrics.parquet"
        )
        logger.info(f"  Labor metrics: {len(labor_metrics)} voyages")
    else:
        logger.warning("Crew roster not found, skipping labor metrics")
    
    # Aggregate route metrics
    if logbook_path.exists():
        logger.info("Computing route exposure...")
        logbook = pd.read_parquet(logbook_path)
        
        # Build crosswalk if missing voyage_ids
        if logbook["voyage_id"].isna().any():
            logger.info("  Building logbook-voyage crosswalk...")
            crosswalk = crosswalk_builder.build_logbook_to_voyage_crosswalk(
                logbook, voyages,
                output_path=STAGING_DIR / "crosswalk_logbook_voyage.csv"
            )
            logbook = crosswalk_builder.apply_crosswalk(
                logbook, crosswalk, source_index_col="logbook_index"
            )
        
        route_metrics = compute_route_exposure(
            logbook, output_path=STAGING_DIR / "voyage_routes.parquet"
        )
        logger.info(f"  Route metrics: {len(route_metrics)} voyages")
    else:
        logger.warning("Logbook positions not found, skipping route metrics")


def stage_assemble_voyages():
    """Stage 6: Build analysis_voyage."""
    logger.info("=== Stage 6: Assemble Voyage Analysis ===")
    
    from assembly import VoyageAssembler
    
    assembler = VoyageAssembler()
    
    try:
        df = assembler.assemble()
        assembler.save()
        
        summary = assembler.get_summary()
        logger.info(f"  Total voyages: {summary['total_voyages']}")
        logger.info(f"  Labor coverage: {summary['labor_data_coverage']:.1%}")
        logger.info(f"  Route coverage: {summary['route_data_coverage']:.1%}")
        logger.info(f"  VQI coverage: {summary['vqi_data_coverage']:.1%}")
        
    except FileNotFoundError as e:
        logger.error(f"Required files missing: {e}")


def stage_link_captains():
    """Stage 7-8: IPUMS loading and captain linkage."""
    logger.info("=== Stages 7-8: Captain-Census Linkage ===")
    
    from linkage import IPUMSLoader, CaptainProfiler, RecordLinker
    import pandas as pd
    
    # Load IPUMS data
    logger.info("Loading IPUMS data...")
    ipums_loader = IPUMSLoader()
    
    try:
        ipums = ipums_loader.parse()
        if len(ipums) == 0:
            logger.warning("No IPUMS data found. See docs/ipums_extract_instructions.md")
            logger.warning("Skipping captain linkage stage")
            return
        
        ipums_loader.save()
        ipums_summary = ipums_loader.get_summary()
        logger.info(f"  IPUMS records: {ipums_summary['total_person_years']}")
    except Exception as e:
        logger.warning(f"IPUMS loading failed: {e}")
        logger.warning("See docs/ipums_extract_instructions.md for setup instructions")
        return
    
    # Build captain profiles
    logger.info("Building captain profiles...")
    profiler = CaptainProfiler()
    
    try:
        profiles = profiler.build_profiles()
        profiler.save()
        profile_summary = profiler.get_summary()
        logger.info(f"  Captain profiles: {profile_summary['total_captains']}")
    except FileNotFoundError as e:
        logger.error(f"Voyage data not found: {e}")
        return
    
    # Run linkage
    logger.info("Running captain-census linkage...")
    linker = RecordLinker()
    
    all_linkages = []
    
    for year in LINKAGE_CONFIG.target_years:
        candidates = profiler.get_linkage_candidates(year)
        
        if len(candidates) == 0:
            logger.info(f"  {year}: no candidates")
            continue
        
        linkage = linker.link_captains_to_census(
            candidates, ipums, year, top_k=3
        )
        
        if len(linkage) > 0:
            all_linkages.append(linkage)
            linked = linkage[linkage["match_rank"] == 1]["captain_id"].nunique()
            logger.info(f"  {year}: linked {linked} captains")
    
    if all_linkages:
        combined = pd.concat(all_linkages, ignore_index=True)
        linker.save_linkage(combined)
        
        summary = linker.get_linkage_summary(combined)
        logger.info(f"  Total captains with matches: {summary['total_captains_with_matches']}")
        logger.info(f"  Mean match score: {summary['mean_best_match_score']:.3f}")


def stage_assemble_captains():
    """Stage 9: Build analysis_captain_year."""
    logger.info("=== Stage 9: Assemble Captain-Year Analysis ===")
    
    from assembly import CaptainAssembler
    
    assembler = CaptainAssembler()
    
    try:
        df = assembler.assemble()
        assembler.save(save_variants=True)
        
        summary = assembler.get_summary()
        logger.info(f"  Captain-year observations: {summary['total_captain_years']}")
        logger.info(f"  Census link rate: {summary['census_link_rate']:.1%}")
        logger.info(f"  Wealth data rate: {summary['wealth_data_rate']:.1%}")
        
    except FileNotFoundError as e:
        logger.error(f"Required files missing: {e}")


def stage_qa():
    """Generate QA report."""
    logger.info("=== QA Report ===")
    
    from qa import generate_qa_report
    
    report = generate_qa_report()
    
    qa_path = DOCS_DIR / "qa_report.md"
    logger.info(f"QA report saved to {qa_path}")


def run_all(force_download: bool = False):
    """Run full pipeline."""
    logger.info("Running full pipeline...")
    
    stage_download(force=force_download)
    stage_parse()
    stage_aggregate()
    stage_assemble_voyages()
    stage_link_captains()
    stage_assemble_captains()
    stage_qa()
    
    logger.info("Pipeline complete!")
    logger.info(f"Output files in: {FINAL_DIR}")


def main():
    parser = argparse.ArgumentParser(
        description="Whaling Data Pipeline - Venture Capital of the Sea"
    )
    
    parser.add_argument(
        "stage",
        choices=[
            "download", "parse", "aggregate",
            "assemble-voyages", "link-captains", "assemble-captains",
            "qa", "all"
        ],
        help="Pipeline stage to run"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download/reprocess even if files exist"
    )
    
    args = parser.parse_args()
    
    if args.stage == "download":
        stage_download(force=args.force)
    elif args.stage == "parse":
        stage_parse()
    elif args.stage == "aggregate":
        stage_aggregate()
    elif args.stage == "assemble-voyages":
        stage_assemble_voyages()
    elif args.stage == "link-captains":
        stage_link_captains()
    elif args.stage == "assemble-captains":
        stage_assemble_captains()
    elif args.stage == "qa":
        stage_qa()
    elif args.stage == "all":
        run_all(force_download=args.force)


if __name__ == "__main__":
    main()
