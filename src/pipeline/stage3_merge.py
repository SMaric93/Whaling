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
from pathlib import Path

logger = logging.getLogger(__name__)


def compute_labor_metrics() -> None:
    """Compute crew count, desertion rate, and role composition."""
    from src.aggregation.labor_metrics import compute_all_labor_metrics
    from src.config import STAGING_DIR
    
    logger.info("Computing labor metrics...")
    
    crew_path = STAGING_DIR / 'crew_parsed.parquet'
    if crew_path.exists():
        df = compute_all_labor_metrics(crew_path)
        output_path = STAGING_DIR / 'labor_metrics.parquet'
        df.to_parquet(output_path, index=False)
        logger.info(f"Computed labor metrics for {len(df):,} voyages → {output_path}")
    else:
        logger.warning(f"Crew file not found: {crew_path}")


def compute_route_exposure() -> None:
    """Compute Arctic exposure, mean lat/lon, and voyage duration."""
    from src.aggregation.route_exposure import compute_route_metrics
    from src.config import STAGING_DIR
    
    logger.info("Computing route exposure metrics...")
    
    logbooks_path = STAGING_DIR / 'logbooks_parsed.parquet'
    if logbooks_path.exists():
        df = compute_route_metrics(logbooks_path)
        output_path = STAGING_DIR / 'route_metrics.parquet'
        df.to_parquet(output_path, index=False)
        logger.info(f"Computed route metrics for {len(df):,} voyages → {output_path}")
    else:
        logger.warning(f"Logbooks file not found: {logbooks_path}")


def resolve_entities() -> None:
    """
    Resolve vessel, captain, and agent entities using string-based matching.
    
    Uses the EntityResolver with:
        - Jaro-Winkler similarity for fuzzy name matching
        - Soundex for phonetic matching
        - Configurable similarity thresholds (default 0.85)
    """
    from src.entities.entity_resolver import EntityResolver
    from src.parsing.string_normalizer import normalize_name, jaro_winkler_similarity
    from src.config import STAGING_DIR, CROSSWALK_DIR
    import pandas as pd
    
    logger.info("Resolving entities with string-based matching...")
    
    voyages_path = STAGING_DIR / 'voyages_parsed.parquet'
    if not voyages_path.exists():
        logger.warning(f"Voyages file not found: {voyages_path}")
        return
    
    voyages = pd.read_parquet(voyages_path)
    resolver = EntityResolver()
    
    # Resolve vessels
    logger.info("  - Resolving vessel entities...")
    vessel_crosswalk = resolver.resolve_vessels(voyages)
    vessel_crosswalk.to_parquet(CROSSWALK_DIR / 'vessel_crosswalk.parquet', index=False)
    logger.info(f"    → {len(vessel_crosswalk):,} unique vessels")
    
    # Resolve captains with string normalization
    logger.info("  - Resolving captain entities (string-based matching)...")
    captain_crosswalk = resolver.resolve_captains(voyages)
    captain_crosswalk.to_parquet(CROSSWALK_DIR / 'captain_crosswalk.parquet', index=False)
    logger.info(f"    → {len(captain_crosswalk):,} unique captains")
    
    # Resolve agents with string normalization
    logger.info("  - Resolving agent entities (string-based matching)...")
    agent_crosswalk = resolver.resolve_agents(voyages)
    agent_crosswalk.to_parquet(CROSSWALK_DIR / 'agent_crosswalk.parquet', index=False)
    logger.info(f"    → {len(agent_crosswalk):,} unique agents")
    
    logger.info("Entity resolution complete.")


def build_entity_crosswalks() -> None:
    """Build crosswalks linking entities across different data sources."""
    from src.entities.crosswalk_builder import build_all_crosswalks
    from src.config import STAGING_DIR, CROSSWALK_DIR
    
    logger.info("Building entity crosswalks across sources...")
    
    try:
        build_all_crosswalks(STAGING_DIR, CROSSWALK_DIR)
        logger.info("Entity crosswalks built successfully.")
    except Exception as e:
        logger.warning(f"Crosswalk building skipped: {e}")


def assemble_voyages() -> None:
    """Assemble the main voyage analysis dataset."""
    from src.assembly.voyage_assembly import assemble_voyage_dataset
    from src.config import STAGING_DIR, FINAL_DIR
    
    logger.info("Assembling voyage dataset...")
    
    df = assemble_voyage_dataset(STAGING_DIR)
    output_path = FINAL_DIR / 'analysis_voyage.parquet'
    df.to_parquet(output_path, index=False)
    logger.info(f"Assembled {len(df):,} voyages → {output_path}")


def link_captains() -> None:
    """
    Link captains to census records using probabilistic matching.
    
    Matching algorithm:
        - Blocking: Geography (state) + age band (±5 years)
        - Scoring: Name similarity (40%) + Age (25%) + Geography (20%) + 
                   Occupation (10%) + Spouse (5%)
        - String matching uses Jaro-Winkler similarity (threshold 0.85)
    """
    from src.linkage.record_linker import RecordLinker
    from src.linkage.captain_profiler import build_captain_profiles
    from src.linkage.ipums_loader import load_ipums_extract
    from src.config import STAGING_DIR, CROSSWALK_DIR, RAW_DIR
    
    logger.info("Linking captains to census records...")
    
    # Build captain profiles
    captain_crosswalk = CROSSWALK_DIR / 'captain_crosswalk.parquet'
    if not captain_crosswalk.exists():
        logger.warning("Captain crosswalk not found - skipping linkage")
        return
    
    profiles = build_captain_profiles(captain_crosswalk, STAGING_DIR / 'voyages_parsed.parquet')
    
    # Load IPUMS census data
    ipums_path = RAW_DIR / 'ipums'
    if not ipums_path.exists():
        logger.warning("IPUMS data not found - skipping linkage")
        return
    
    # Link for each census year
    linker = RecordLinker()
    for target_year in [1850, 1860, 1870, 1880]:
        census_path = ipums_path / f'census_{target_year}.parquet'
        if census_path.exists():
            logger.info(f"  - Linking to {target_year} census...")
            census_data = load_ipums_extract(census_path)
            linkage = linker.link_captains_to_census(profiles, census_data, target_year)
            
            output_path = CROSSWALK_DIR / f'captain_census_{target_year}.parquet'
            linkage.to_parquet(output_path, index=False)
            logger.info(f"    → {len(linkage):,} links → {output_path}")
    
    logger.info("Captain-census linkage complete.")


def assemble_captains() -> None:
    """Assemble the captain-year panel with census wealth data."""
    from src.assembly.captain_assembly import assemble_captain_panel
    from src.config import STAGING_DIR, CROSSWALK_DIR, FINAL_DIR
    
    logger.info("Assembling captain-year panel...")
    
    df = assemble_captain_panel(STAGING_DIR, CROSSWALK_DIR)
    output_path = FINAL_DIR / 'analysis_captain_year.parquet'
    df.to_parquet(output_path, index=False)
    logger.info(f"Assembled {len(df):,} captain-years → {output_path}")


def augment_voyages() -> None:
    """Augment voyages with supplementary sources (Starbuck, Maury, WSL)."""
    from src.assembly.voyage_augmentor import augment_voyage_dataset
    from src.config import STAGING_DIR, FINAL_DIR
    
    logger.info("Augmenting voyage dataset...")
    
    base_path = FINAL_DIR / 'analysis_voyage.parquet'
    if not base_path.exists():
        logger.warning("Base voyage file not found - skipping augmentation")
        return
    
    df = augment_voyage_dataset(base_path, STAGING_DIR)
    output_path = FINAL_DIR / 'analysis_voyage_augmented.parquet'
    df.to_parquet(output_path, index=False)
    logger.info(f"Augmented {len(df):,} voyages → {output_path}")


def merge_climate_data() -> None:
    """Merge climate/weather data with voyage dataset."""
    from src.config import STAGING_DIR, FINAL_DIR
    import pandas as pd
    
    logger.info("Merging climate data...")
    
    voyage_path = FINAL_DIR / 'analysis_voyage_augmented.parquet'
    if not voyage_path.exists():
        voyage_path = FINAL_DIR / 'analysis_voyage.parquet'
    
    if not voyage_path.exists():
        logger.warning("Voyage file not found - skipping climate merge")
        return
    
    voyages = pd.read_parquet(voyage_path)
    
    # Merge climate sources
    climate_files = [
        ('weather_annual.parquet', 'year'),
        ('hurricane_annual.parquet', 'year'),
        ('sea_ice_annual.parquet', 'year'),
    ]
    
    for filename, merge_key in climate_files:
        climate_path = STAGING_DIR / filename
        if climate_path.exists():
            climate_df = pd.read_parquet(climate_path)
            # Determine appropriate year column
            if 'sail_year' in voyages.columns:
                voyages = voyages.merge(
                    climate_df, 
                    left_on='sail_year', 
                    right_on=merge_key,
                    how='left',
                    suffixes=('', f'_{filename.split(".")[0]}')
                )
                logger.info(f"  - Merged {filename}")
    
    output_path = FINAL_DIR / 'analysis_voyage_with_climate.parquet'
    voyages.to_parquet(output_path, index=False)
    logger.info(f"Climate merge complete → {output_path}")


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
    
    # Aggregation
    try:
        compute_labor_metrics()
        results['labor_metrics'] = True
    except Exception as e:
        logger.error(f"Labor metrics failed: {e}")
    
    try:
        compute_route_exposure()
        results['route_exposure'] = True
    except Exception as e:
        logger.error(f"Route exposure failed: {e}")
    
    # Entity resolution (with string-based matching)
    try:
        resolve_entities()
        results['entities'] = True
    except Exception as e:
        logger.error(f"Entity resolution failed: {e}")
    
    try:
        build_entity_crosswalks()
        results['crosswalks'] = True
    except Exception as e:
        logger.warning(f"Crosswalk building skipped: {e}")
    
    # Assembly
    try:
        assemble_voyages()
        results['voyage_assembly'] = True
    except Exception as e:
        logger.error(f"Voyage assembly failed: {e}")
    
    try:
        link_captains()
        results['captain_linkage'] = True
    except Exception as e:
        logger.warning(f"Captain linkage skipped: {e}")
    
    try:
        assemble_captains()
        results['captain_assembly'] = True
    except Exception as e:
        logger.warning(f"Captain assembly skipped: {e}")
    
    try:
        augment_voyages()
        results['augmentation'] = True
    except Exception as e:
        logger.warning(f"Voyage augmentation skipped: {e}")
    
    try:
        merge_climate_data()
        results['climate_merge'] = True
    except Exception as e:
        logger.warning(f"Climate merge skipped: {e}")
    
    # Summary
    success_count = sum(results.values())
    total_count = len(results)
    logger.info(f"Stage 3 complete: {success_count}/{total_count} operations successful")
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_merge()
