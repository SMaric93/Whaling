"""
Stage 2: Data Cleaning

Parses and standardizes all raw data sources.

Operations:
    - Parse voyage records (AOWV)
    - Parse crew lists with role extraction
    - Parse logbook entries with coordinate extraction
    - Parse vessel registry records
    - Parse Starbuck historical data
    - Parse Maury logbook coordinates
    - Extract events from WSL PDFs
    
String Normalization:
    - All names normalized using Jaro-Winkler similarity and Soundex
    - Handles abbreviations (WM → WILLIAM, CHAS → CHARLES)
    - Preserves suffixes (JR, SR)
"""

import logging

from src.pipeline._runner import StepSpec, run_steps

logger = logging.getLogger(__name__)


def _save_output_variants(df, primary_path, *alias_paths) -> None:
    """Write a dataframe to the canonical output plus compatibility aliases."""
    primary_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(primary_path, index=False)
    df.to_csv(primary_path.with_suffix(".csv"), index=False)
    for alias_path in alias_paths:
        df.to_parquet(alias_path, index=False)
        df.to_csv(alias_path.with_suffix(".csv"), index=False)


def clean_voyages() -> bool:
    """Parse and standardize AOWV voyage records."""
    from src.parsing.voyage_parser import VoyageParser
    from src.config import RAW_AOWV, STAGING_DIR
    
    logger.info("Cleaning voyage records...")
    
    if not RAW_AOWV.exists():
        logger.warning(f"Voyage directory not found: {RAW_AOWV}")
        return False

    df = VoyageParser(raw_dir=RAW_AOWV).parse()
    output_path = STAGING_DIR / 'voyages_master.parquet'
    _save_output_variants(df, output_path, STAGING_DIR / 'voyages_parsed.parquet')
    logger.info(f"Parsed {len(df):,} voyages → {output_path}")
    return output_path.exists()


def clean_crew() -> bool:
    """Parse crew lists with role extraction."""
    from src.parsing.crew_parser import parse_crew_lists
    from src.config import RAW_CREWLIST, STAGING_DIR
    
    logger.info("Cleaning crew lists...")
    
    if not RAW_CREWLIST.exists():
        logger.warning(f"Crew directory not found: {RAW_CREWLIST}")
        return False

    df = parse_crew_lists(RAW_CREWLIST)
    output_path = STAGING_DIR / 'crew_roster.parquet'
    _save_output_variants(df, output_path, STAGING_DIR / 'crew_parsed.parquet')
    logger.info(f"Parsed {len(df):,} crew records → {output_path}")
    return output_path.exists()


def clean_logbooks() -> bool:
    """Parse logbook entries with coordinate extraction."""
    from src.parsing.logbook_parser import LogbookParser
    from src.config import RAW_LOGBOOKS, STAGING_DIR
    
    logger.info("Cleaning logbook entries...")
    
    if not RAW_LOGBOOKS.exists():
        logger.warning(f"Logbooks directory not found: {RAW_LOGBOOKS}")
        return False

    df = LogbookParser(raw_dir=RAW_LOGBOOKS).parse()
    output_path = STAGING_DIR / 'logbook_positions.parquet'
    _save_output_variants(df, output_path, STAGING_DIR / 'logbooks_parsed.parquet')
    logger.info(f"Parsed {len(df):,} logbook entries → {output_path}")
    return output_path.exists()


def clean_registers() -> bool:
    """Parse vessel registry and insurance records."""
    from src.parsing.register_parser import parse_registers
    from src.config import RAW_INSURANCE, STAGING_DIR
    
    logger.info("Cleaning vessel registers...")
    
    if not RAW_INSURANCE.exists():
        logger.info("No vessel register source found - skipping")
        return False

    df = parse_registers(RAW_INSURANCE)
    output_path = STAGING_DIR / 'vessel_register_year.parquet'
    _save_output_variants(df, output_path, STAGING_DIR / 'registers_parsed.parquet')
    logger.info(f"Parsed {len(df):,} register records → {output_path}")
    return output_path.exists()


def clean_starbuck() -> bool:
    """Parse Starbuck (1878) historical data."""
    from src.parsing.starbuck_parser import parse_starbuck
    from src.config import RAW_STARBUCK, STAGING_DIR
    
    logger.info("Cleaning Starbuck data...")
    
    if not RAW_STARBUCK.exists():
        logger.info("No Starbuck data found - skipping")
        return False

    df = parse_starbuck(RAW_STARBUCK)
    output_path = STAGING_DIR / 'starbuck_voyage_list.parquet'
    _save_output_variants(df, output_path, STAGING_DIR / 'starbuck_parsed.parquet')
    logger.info(f"Parsed {len(df):,} Starbuck records → {output_path}")
    return output_path.exists()


def clean_maury() -> bool:
    """Parse Maury logbook coordinates."""
    from src.parsing.maury_parser import run_maury_parser
    from src.config import RAW_MAURY, STAGING_DIR
    
    logger.info("Cleaning Maury logbooks...")
    
    if not RAW_MAURY.exists():
        logger.info("No Maury data found - skipping")
        return False

    df = run_maury_parser(RAW_MAURY)
    if len(df) == 0:
        return False

    output_path = STAGING_DIR / 'maury_positions.parquet'
    _save_output_variants(df, output_path, STAGING_DIR / 'maury_parsed.parquet')
    logger.info(f"Parsed {len(df):,} Maury entries → {output_path}")
    return output_path.exists()


def clean_wsl() -> bool:
    """Extract events from WSL PDFs."""
    from src.parsing.wsl_event_extractor import extract_all_wsl_events
    from src.config import RAW_WSL, STAGING_DIR
    
    logger.info("Extracting WSL events...")
    
    wsl_dir = RAW_WSL
    if wsl_dir.exists() and any(wsl_dir.glob('*.pdf')):
        df = extract_all_wsl_events(wsl_dir)
        output_path = STAGING_DIR / 'wsl_events.parquet'
        _save_output_variants(df, output_path)
        logger.info(f"Extracted {len(df):,} WSL events → {output_path}")
        return output_path.exists()

    logger.info("No WSL PDFs found - skipping")
    return False


def run_clean() -> dict:
    """
    Run the complete data cleaning stage.
    
    Returns:
        dict: Summary of cleaning operations
    """
    logger.info("=" * 60)
    logger.info("STAGE 2: DATA CLEANING")
    logger.info("=" * 60)
    
    results = {
        'voyages': False,
        'crew': False,
        'logbooks': False,
        'registers': False,
        'starbuck': False,
        'maury': False,
        'wsl': False,
    }

    run_steps(
        results,
        [
            StepSpec('voyages', clean_voyages, "Voyage cleaning failed", failure_level="error"),
            StepSpec('crew', clean_crew, "Crew cleaning failed", failure_level="error"),
            StepSpec('logbooks', clean_logbooks, "Logbook cleaning failed", failure_level="error"),
            StepSpec('registers', clean_registers, "Register cleaning skipped"),
            StepSpec('starbuck', clean_starbuck, "Starbuck cleaning skipped"),
            StepSpec('maury', clean_maury, "Maury cleaning skipped"),
            StepSpec('wsl', clean_wsl, "WSL cleaning skipped"),
        ],
        logger=logger,
    )

    # Summary
    success_count = sum(results.values())
    total_count = len(results)
    logger.info(f"Stage 2 complete: {success_count}/{total_count} sources cleaned successfully")
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_clean()
