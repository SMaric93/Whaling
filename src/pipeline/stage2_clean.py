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
from pathlib import Path

logger = logging.getLogger(__name__)


def clean_voyages() -> None:
    """Parse and standardize AOWV voyage records."""
    from src.parsing.voyage_parser import parse_voyages
    from src.config import RAW_DIR, STAGING_DIR
    
    logger.info("Cleaning voyage records...")
    
    voyages_raw = RAW_DIR / 'aowv_voyages.csv'
    if voyages_raw.exists():
        df = parse_voyages(voyages_raw)
        output_path = STAGING_DIR / 'voyages_parsed.parquet'
        df.to_parquet(output_path, index=False)
        logger.info(f"Parsed {len(df):,} voyages → {output_path}")
    else:
        logger.warning(f"Voyage file not found: {voyages_raw}")


def clean_crew() -> None:
    """Parse crew lists with role extraction."""
    from src.parsing.crew_parser import parse_crew_lists
    from src.config import RAW_DIR, STAGING_DIR
    
    logger.info("Cleaning crew lists...")
    
    crew_raw = RAW_DIR / 'aowv_crew.csv'
    if crew_raw.exists():
        df = parse_crew_lists(crew_raw)
        output_path = STAGING_DIR / 'crew_parsed.parquet'
        df.to_parquet(output_path, index=False)
        logger.info(f"Parsed {len(df):,} crew records → {output_path}")
    else:
        logger.warning(f"Crew file not found: {crew_raw}")


def clean_logbooks() -> None:
    """Parse logbook entries with coordinate extraction."""
    from src.parsing.logbook_parser import parse_logbooks
    from src.config import RAW_DIR, STAGING_DIR
    
    logger.info("Cleaning logbook entries...")
    
    logbooks_raw = RAW_DIR / 'aowv_logbooks.csv'
    if logbooks_raw.exists():
        df = parse_logbooks(logbooks_raw)
        output_path = STAGING_DIR / 'logbooks_parsed.parquet'
        df.to_parquet(output_path, index=False)
        logger.info(f"Parsed {len(df):,} logbook entries → {output_path}")
    else:
        logger.warning(f"Logbooks file not found: {logbooks_raw}")


def clean_registers() -> None:
    """Parse vessel registry and insurance records."""
    from src.parsing.register_parser import parse_registers
    from src.config import RAW_DIR, STAGING_DIR
    
    logger.info("Cleaning vessel registers...")
    
    register_raw = RAW_DIR / 'marine_register.csv'
    if register_raw.exists():
        df = parse_registers(register_raw)
        output_path = STAGING_DIR / 'registers_parsed.parquet'
        df.to_parquet(output_path, index=False)
        logger.info(f"Parsed {len(df):,} register records → {output_path}")
    else:
        logger.info("No vessel register file found - skipping")


def clean_starbuck() -> None:
    """Parse Starbuck (1878) historical data."""
    from src.parsing.starbuck_parser import parse_starbuck
    from src.config import RAW_DIR, STAGING_DIR
    
    logger.info("Cleaning Starbuck data...")
    
    starbuck_raw = RAW_DIR / 'starbuck'
    if starbuck_raw.exists():
        df = parse_starbuck(starbuck_raw)
        output_path = STAGING_DIR / 'starbuck_parsed.parquet'
        df.to_parquet(output_path, index=False)
        logger.info(f"Parsed {len(df):,} Starbuck records → {output_path}")
    else:
        logger.info("No Starbuck data found - skipping")


def clean_maury() -> None:
    """Parse Maury logbook coordinates."""
    from src.parsing.maury_parser import parse_maury
    from src.config import RAW_DIR, STAGING_DIR
    
    logger.info("Cleaning Maury logbooks...")
    
    maury_raw = RAW_DIR / 'maury'
    if maury_raw.exists():
        df = parse_maury(maury_raw)
        output_path = STAGING_DIR / 'maury_parsed.parquet'
        df.to_parquet(output_path, index=False)
        logger.info(f"Parsed {len(df):,} Maury entries → {output_path}")
    else:
        logger.info("No Maury data found - skipping")


def clean_wsl() -> None:
    """Extract events from WSL PDFs."""
    from src.parsing.wsl_event_extractor import extract_all_wsl_events
    from src.config import RAW_DIR, STAGING_DIR
    
    logger.info("Extracting WSL events...")
    
    wsl_dir = RAW_DIR / 'wsl_pdfs'
    if wsl_dir.exists() and any(wsl_dir.glob('*.pdf')):
        df = extract_all_wsl_events(wsl_dir)
        output_path = STAGING_DIR / 'wsl_events.parquet'
        df.to_parquet(output_path, index=False)
        logger.info(f"Extracted {len(df):,} WSL events → {output_path}")
    else:
        logger.info("No WSL PDFs found - skipping")


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
    
    # Core data
    try:
        clean_voyages()
        results['voyages'] = True
    except Exception as e:
        logger.error(f"Voyage cleaning failed: {e}")
    
    try:
        clean_crew()
        results['crew'] = True
    except Exception as e:
        logger.error(f"Crew cleaning failed: {e}")
    
    try:
        clean_logbooks()
        results['logbooks'] = True
    except Exception as e:
        logger.error(f"Logbook cleaning failed: {e}")
    
    # Supplementary data
    try:
        clean_registers()
        results['registers'] = True
    except Exception as e:
        logger.warning(f"Register cleaning skipped: {e}")
    
    try:
        clean_starbuck()
        results['starbuck'] = True
    except Exception as e:
        logger.warning(f"Starbuck cleaning skipped: {e}")
    
    try:
        clean_maury()
        results['maury'] = True
    except Exception as e:
        logger.warning(f"Maury cleaning skipped: {e}")
    
    try:
        clean_wsl()
        results['wsl'] = True
    except Exception as e:
        logger.warning(f"WSL cleaning skipped: {e}")
    
    # Summary
    success_count = sum(results.values())
    total_count = len(results)
    logger.info(f"Stage 2 complete: {success_count}/{total_count} sources cleaned successfully")
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_clean()
