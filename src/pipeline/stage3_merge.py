"""
Stage 3: Data Merging

Assembles and links all parsed datasets into analysis-ready files.

Operations:
    - Compute labor metrics (crew count, desertion rate, etc.)
    - Compute route exposure (Arctic exposure, mean lat/lon)
    - Resolve entities (vessels, captains, agents) with string-based matching
    - Assemble voyage dataset
    - Link captains to census records
    - Assemble captain-year panel
    - Augment voyages with supplementary sources

String-Based Entity Matching:
    - Uses Jaro-Winkler similarity for fuzzy name matching
    - Applies Soundex for phonetic matching
    - Normalizes names (abbreviations, case, punctuation)
    - Supports configurable similarity thresholds
"""

import logging

from src.pipeline._runner import StepSpec, run_steps, summarize_step_results

logger = logging.getLogger(__name__)


def _write_output_variants(df, primary_path, *alias_paths) -> None:
    """Persist canonical pipeline outputs alongside legacy compatibility names."""
    primary_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(primary_path, index=False)
    df.to_csv(primary_path.with_suffix(".csv"), index=False)
    for alias_path in alias_paths:
        df.to_parquet(alias_path, index=False)
        df.to_csv(alias_path.with_suffix(".csv"), index=False)


def compute_labor_metrics() -> bool:
    """Compute crew count, desertion rate, and role composition."""
    import pandas as pd
    from src.aggregation.labor_metrics import compute_voyage_labor_metrics
    from src.config import STAGING_DIR
    
    logger.info("Computing labor metrics...")
    
    crew_candidates = [
        STAGING_DIR / 'crew_roster.parquet',
        STAGING_DIR / 'crew_parsed.parquet',
    ]
    crew_path = next((path for path in crew_candidates if path.exists()), None)
    if crew_path is None:
        logger.warning("Crew file not found - skipping labor metrics")
        return False

    crew_df = pd.read_parquet(crew_path)
    df = compute_voyage_labor_metrics(crew_df)
    output_path = STAGING_DIR / 'voyage_labor_metrics.parquet'
    _write_output_variants(df, output_path, STAGING_DIR / 'labor_metrics.parquet')
    logger.info(f"Computed labor metrics for {len(df):,} voyages → {output_path}")
    return output_path.exists()


def compute_route_exposure() -> bool:
    """Compute Arctic exposure, mean lat/lon, and voyage duration."""
    import pandas as pd
    from src.aggregation.route_exposure import (
        compute_route_exposure as compute_voyage_routes,
        compute_whaling_ground_exposure,
    )
    from src.config import STAGING_DIR
    
    logger.info("Computing route exposure metrics...")
    
    logbook_candidates = [
        STAGING_DIR / 'logbook_positions.parquet',
        STAGING_DIR / 'logbooks_parsed.parquet',
    ]
    logbooks_path = next((path for path in logbook_candidates if path.exists()), None)
    if logbooks_path is None:
        logger.warning("Logbooks file not found - skipping route exposure")
        return False

    logbook_df = pd.read_parquet(logbooks_path)
    df = compute_voyage_routes(logbook_df)
    ground_df = compute_whaling_ground_exposure(logbook_df)
    if len(ground_df) > 0:
        df = df.merge(ground_df, on='voyage_id', how='left')

    output_path = STAGING_DIR / 'voyage_routes.parquet'
    _write_output_variants(df, output_path, STAGING_DIR / 'route_metrics.parquet')
    logger.info(f"Computed route metrics for {len(df):,} voyages → {output_path}")
    return output_path.exists()


def resolve_entities() -> bool:
    """
    Resolve vessel, captain, and agent entities using string-based matching.
    
    Uses the EntityResolver with:
        - Jaro-Winkler similarity for fuzzy name matching
        - Soundex for phonetic matching
        - Configurable similarity thresholds (default 0.85)
    """
    from src.entities.entity_resolver import EntityResolver
    from src.config import CROSSWALKS_DIR, STAGING_DIR
    import pandas as pd
    
    logger.info("Resolving entities with string-based matching...")
    
    voyage_candidates = [
        STAGING_DIR / 'voyages_master.parquet',
        STAGING_DIR / 'voyages_parsed.parquet',
    ]
    voyages_path = next((path for path in voyage_candidates if path.exists()), None)
    if voyages_path is None:
        logger.warning("Voyages file not found - skipping entity resolution")
        return False
    
    voyages = pd.read_parquet(voyages_path)
    resolver = EntityResolver()
    resolved_voyages = resolver.resolve_voyages_df(voyages, ml_refine=True)
    _write_output_variants(
        resolved_voyages,
        STAGING_DIR / 'voyages_master.parquet',
        STAGING_DIR / 'voyages_parsed.parquet',
    )
    
    # Resolve vessels
    logger.info("  - Resolving vessel entities...")
    CROSSWALKS_DIR.mkdir(parents=True, exist_ok=True)
    vessel_crosswalk = resolver.resolve_vessels(resolved_voyages)
    vessel_crosswalk.to_parquet(CROSSWALKS_DIR / 'vessel_crosswalk.parquet', index=False)
    logger.info(f"    → {len(vessel_crosswalk):,} unique vessels")
    
    # Resolve captains with string normalization
    logger.info("  - Resolving captain entities (string-based matching)...")
    captain_crosswalk = resolver.resolve_captains(resolved_voyages)
    captain_crosswalk.to_parquet(CROSSWALKS_DIR / 'captain_crosswalk.parquet', index=False)
    logger.info(f"    → {len(captain_crosswalk):,} unique captains")
    
    # Resolve agents with string normalization
    logger.info("  - Resolving agent entities (string-based matching)...")
    agent_crosswalk = resolver.resolve_agents(resolved_voyages)
    agent_crosswalk.to_parquet(CROSSWALKS_DIR / 'agent_crosswalk.parquet', index=False)
    logger.info(f"    → {len(agent_crosswalk):,} unique agents")
    
    logger.info("Entity resolution complete.")
    return True


def build_entity_crosswalks() -> bool:
    """Build crosswalks linking entities across different data sources."""
    import pandas as pd
    from src.config import CROSSWALKS_DIR, STAGING_DIR
    from src.entities.crosswalk_builder import CrosswalkBuilder
    
    logger.info("Building entity crosswalks across sources...")

    voyages_path = STAGING_DIR / 'voyages_master.parquet'
    if not voyages_path.exists():
        logger.warning("Resolved voyage file not found - skipping crosswalks")
        return False

    builder = CrosswalkBuilder()
    voyages_df = pd.read_parquet(voyages_path)
    built_any = False

    crew_path = STAGING_DIR / 'crew_roster.parquet'
    if crew_path.exists():
        crew_df = pd.read_parquet(crew_path)
        if crew_df['voyage_id'].isna().any():
            crosswalk = builder.build_crew_to_voyage_crosswalk(
                crew_df,
                voyages_df,
                output_path=CROSSWALKS_DIR / 'crew_to_voyage_crosswalk.csv',
            )
            updated = builder.apply_crosswalk(crew_df, crosswalk, source_index_col='crew_index')
            _write_output_variants(updated, crew_path, STAGING_DIR / 'crew_parsed.parquet')
            built_any = built_any or len(crosswalk) > 0
        else:
            logger.info("  - Crew roster already contains voyage_id")
            built_any = True

    logbook_path = STAGING_DIR / 'logbook_positions.parquet'
    if logbook_path.exists():
        logbook_df = pd.read_parquet(logbook_path)
        if logbook_df['voyage_id'].isna().any():
            crosswalk = builder.build_logbook_to_voyage_crosswalk(
                logbook_df,
                voyages_df,
                output_path=CROSSWALKS_DIR / 'logbook_to_voyage_crosswalk.csv',
            )
            updated = builder.apply_crosswalk(logbook_df, crosswalk, source_index_col='logbook_index')
            _write_output_variants(updated, logbook_path, STAGING_DIR / 'logbooks_parsed.parquet')
            built_any = built_any or len(crosswalk) > 0
        else:
            logger.info("  - Logbook positions already contain voyage_id")
            built_any = True

    return built_any


def assemble_voyages() -> bool:
    """Assemble the main voyage analysis dataset."""
    from src.assembly.voyage_assembly import VoyageAssembler
    from src.config import FINAL_DIR
    
    logger.info("Assembling voyage dataset...")
    
    assembler = VoyageAssembler()
    df = assembler.assemble(force_reload=True)
    output_path = assembler.save(FINAL_DIR / 'analysis_voyage.parquet')
    logger.info(f"Assembled {len(df):,} voyages → {output_path}")
    return output_path.exists()


def link_captains() -> bool:
    """
    Link captains to census records using probabilistic matching.
    
    Matching algorithm:
        - Blocking: Geography (state) + age band (±5 years)
        - Scoring: Name similarity (40%) + Age (25%) + Geography (20%) + 
                   Occupation (10%) + Spouse (5%)
        - String matching uses Jaro-Winkler similarity (threshold 0.85)
    """
    import pandas as pd
    from src.config import LINKAGE_CONFIG, RAW_IPUMS, STAGING_DIR
    from src.linkage.record_linker import RecordLinker
    from src.linkage.captain_profiler import CaptainProfiler
    from src.linkage.ipums_loader import IPUMSLoader
    
    logger.info("Linking captains to census records...")

    profiler = CaptainProfiler(STAGING_DIR / 'voyages_master.parquet')
    profiles = profiler.build_profiles(force_reload=True)
    if len(profiles) == 0:
        logger.warning("No captain profiles available - skipping linkage")
        return None
    profiler.save()

    ipums_path = STAGING_DIR / "ipums_person_year.parquet"
    if ipums_path.exists():
        census_data = pd.read_parquet(ipums_path)
        logger.info("Loaded staged IPUMS data: %s rows", len(census_data))
    else:
        if not RAW_IPUMS.exists():
            logger.warning(
                "IPUMS data not found - add an extract under %s or save %s first",
                RAW_IPUMS,
                ipums_path,
            )
            return None

        loader = IPUMSLoader(RAW_IPUMS)
        census_data = loader.parse(force_reload=True)
        if len(census_data) == 0:
            logger.warning("No IPUMS records loaded - skipping linkage")
            return None
        loader.save(ipums_path)

    linker = RecordLinker()
    yearly_linkages = []
    for target_year in LINKAGE_CONFIG.target_years:
        candidates = profiler.get_linkage_candidates(target_year)
        if len(candidates) == 0:
            continue
        logger.info(f"  - Linking to {target_year} census...")
        linkage = linker.link_captains_to_census(candidates, census_data, target_year)
        if len(linkage) > 0:
            yearly_linkages.append(linkage)

    if not yearly_linkages:
        logger.warning("No captain-census matches found")
        return None

    linkage_df = pd.concat(yearly_linkages, ignore_index=True)
    output_path = linker.save_linkage(linkage_df)
    logger.info("Captain-census linkage complete.")
    return output_path.exists()


def assemble_captains() -> bool:
    """Assemble the captain-year panel with census wealth data."""
    from src.assembly.captain_assembly import CaptainAssembler
    from src.config import FINAL_DIR
    
    logger.info("Assembling captain-year panel...")
    
    assembler = CaptainAssembler()
    df = assembler.assemble(force_reload=True)
    if len(df) == 0:
        logger.warning("Captain panel is empty")
        return False
    output_path = assembler.save(FINAL_DIR / 'analysis_captain_year.parquet')
    logger.info(f"Assembled {len(df):,} captain-years → {output_path}")
    return output_path.exists()


def augment_voyages() -> bool:
    """Augment voyages with supplementary sources (Starbuck, Maury, WSL)."""
    from src.assembly.voyage_augmentor import run_voyage_augmentation
    from src.config import FINAL_DIR
    
    logger.info("Augmenting voyage dataset...")
    
    base_path = FINAL_DIR / 'analysis_voyage.parquet'
    if not base_path.exists():
        logger.warning("Base voyage file not found - skipping augmentation")
        return False
    
    df = run_voyage_augmentation(base_path)
    output_path = FINAL_DIR / 'analysis_voyage_augmented.parquet'
    logger.info(f"Augmented {len(df):,} voyages → {output_path}")
    return output_path.exists()


def merge_climate_data() -> bool:
    """Merge climate/weather data with voyage dataset."""
    from src.config import FINAL_DIR
    from src.download.weather_downloader import download_and_integrate_weather
    import pandas as pd
    
    logger.info("Merging climate data...")
    
    voyage_path = FINAL_DIR / 'analysis_voyage_augmented.parquet'
    if not voyage_path.exists():
        voyage_path = FINAL_DIR / 'analysis_voyage.parquet'
    
    if not voyage_path.exists():
        logger.warning("Voyage file not found - skipping climate merge")
        return False
    
    voyages = pd.read_parquet(voyage_path)
    voyage_weather_path = FINAL_DIR / 'voyage_weather.parquet'

    if not voyage_weather_path.exists():
        try:
            _, voyage_weather = download_and_integrate_weather(voyages_path=voyage_path, save_raw=True)
        except Exception as exc:
            logger.warning(f"Weather integration skipped: {exc}")
            return None
    else:
        voyage_weather = pd.read_parquet(voyage_weather_path)

    if len(voyage_weather) == 0:
        logger.warning("No voyage weather data available")
        return None

    climate_df = voyage_weather.drop(columns=['year_out'], errors='ignore')
    voyages = voyages.merge(climate_df, on='voyage_id', how='left')
    logger.info("  - Merged voyage_weather controls")
    
    output_path = FINAL_DIR / 'analysis_voyage_with_climate.parquet'
    voyages.to_parquet(output_path, index=False)
    voyages.to_csv(output_path.with_suffix('.csv'), index=False)
    logger.info(f"Climate merge complete → {output_path}")
    return output_path.exists()


def run_merge() -> dict:
    """
    Run the complete data merging stage.
    
    Returns:
        dict: Summary of merging operations
    """
    logger.info("=" * 60)
    logger.info("STAGE 3: DATA MERGING")
    logger.info("=" * 60)
    
    results = {
        'labor_metrics': False,
        'route_exposure': False,
        'entities': False,
        'crosswalks': False,
        'voyage_assembly': False,
        'captain_linkage': False,
        'captain_assembly': False,
        'augmentation': False,
        'climate_merge': False,
    }

    run_steps(
        results,
        [
            StepSpec('labor_metrics', compute_labor_metrics, "Labor metrics failed", failure_level="error"),
            StepSpec('route_exposure', compute_route_exposure, "Route exposure failed", failure_level="error"),
            StepSpec('entities', resolve_entities, "Entity resolution failed", failure_level="error"),
            StepSpec('crosswalks', build_entity_crosswalks, "Crosswalk building skipped"),
            StepSpec('voyage_assembly', assemble_voyages, "Voyage assembly failed", failure_level="error"),
            StepSpec('captain_linkage', link_captains, "Captain linkage skipped"),
            StepSpec('captain_assembly', assemble_captains, "Captain assembly skipped"),
            StepSpec('augmentation', augment_voyages, "Voyage augmentation skipped"),
            StepSpec('climate_merge', merge_climate_data, "Climate merge skipped"),
        ],
        logger=logger,
    )

    # Summary
    success_count, skipped_count, failed_count = summarize_step_results(results)
    logger.info(
        "Stage 3 complete: %s successful, %s skipped, %s failed",
        success_count,
        skipped_count,
        failed_count,
    )
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_merge()
