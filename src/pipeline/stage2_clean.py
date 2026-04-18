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

from src.config import PROJECT_ROOT, STAGING_DIR
from src.pipeline._runner import StepSpec, run_steps, summarize_step_results

logger = logging.getLogger(__name__)


def _price_extracted_dir():
    """Return the VLM-extraction directory. Hook for tests to override."""
    return PROJECT_ROOT / "data" / "extracted"


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
        return None

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
        return None

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
        return None

    df = run_maury_parser(RAW_MAURY)
    if len(df) == 0:
        logger.info("No valid Maury positions parsed - skipping output")
        return None

    output_path = STAGING_DIR / 'maury_positions.parquet'
    _save_output_variants(df, output_path, STAGING_DIR / 'maury_parsed.parquet')
    logger.info(f"Parsed {len(df):,} Maury entries → {output_path}")
    return output_path.exists()


def ingest_wsl_vlm() -> bool:
    """
    Ingest VLM-extracted WSL events from HPCC outputs.

    Looks for per-year JSONL files in ``data/extracted/`` (produced by
    ``scripts/hpcc_extract_v4.sb`` on MSU ICER), runs the V4 post-processor
    to consolidate them into ``data/staging/wsl_events_v4.parquet``, validates
    the result against the extraction gate, and logs a run record.

    Returns True on successful validated ingest, None if no JSONL outputs
    are present (fall back to rule-based extractor), or raises on validation
    failure so the pipeline halts before bad data propagates.
    """
    from src.config import PROJECT_ROOT, STAGING_DIR
    from src.parsing.wsl_v4_postprocess import EXTRACTOR_VERSION, run_postprocess
    from src.utils.extraction_validators import (
        ExtractionValidationError,
        validate_parquet,
    )
    from src.utils.run_registry import log_extraction_run

    extracted_dir = PROJECT_ROOT / "data" / "extracted"
    jsonl_files = sorted(extracted_dir.glob("wsl_events_*.jsonl")) if extracted_dir.exists() else []
    if not jsonl_files:
        logger.info("No VLM extraction JSONL files in %s — skipping ingest", extracted_dir)
        return None

    logger.info("Ingesting %d VLM extraction JSONL files from %s",
                len(jsonl_files), extracted_dir)
    summary = run_postprocess(extracted_dir=extracted_dir, staging_dir=STAGING_DIR)
    if summary.get("status") != "success":
        logger.error("VLM post-processing failed: %s", summary.get("message"))
        return False

    events_path = STAGING_DIR / "wsl_events_v4.parquet"
    try:
        result, _ = validate_parquet(events_path, strict=True)
    except ExtractionValidationError:
        log_extraction_run(
            stage="wsl_vlm_ingest",
            config={"extractor_version": EXTRACTOR_VERSION,
                    "n_jsonl_files": len(jsonl_files)},
            metrics={"status": "validation_failed"},
            artifacts={"events_parquet": str(events_path)},
            tags={"outcome": "failed"},
        )
        raise

    log_extraction_run(
        stage="wsl_vlm_ingest",
        config={
            "extractor_version": EXTRACTOR_VERSION,
            "n_jsonl_files": len(jsonl_files),
            "years_range": summary.get("years_range"),
        },
        metrics={
            "n_events": summary.get("n_events", 0),
            "n_pages": summary.get("n_pages", 0),
            "n_years": summary.get("n_years", 0),
            **{f"val_{k}": v for k, v in result.metrics.items()},
        },
        artifacts={
            "events_parquet": str(events_path),
            "pages_parquet": str(STAGING_DIR / "wsl_pages_v4.parquet"),
            "manifest": str(STAGING_DIR / "wsl_v4_postprocess_manifest.json"),
        },
        tags={"outcome": "passed" if result.ok else "warnings"},
    )
    return events_path.exists()


def ingest_wsl_prices() -> bool:
    """
    Ingest VLM-extracted commodity prices from HPCC outputs.

    Reads per-year JSONL files from ``data/extracted/``, flattens their
    ``prices[]`` arrays into ``data/staging/wsl_prices_v4.parquet``, builds
    a monthly (year_month × commodity) panel, validates against the price
    gate, and logs a run record. Returns None if no JSONL files are present
    (the HPCC price backfill hasn't run yet); raises PriceValidationError
    on validation failure.
    """
    from src.parsing.wsl_price_postprocess import run_price_postprocess
    from src.parsing.wsl_price_panel import build_monthly_panel
    from src.utils.price_validators import (
        PriceValidationError, validate_wsl_prices,
    )
    from src.utils.run_registry import log_extraction_run
    import pandas as pd

    extracted_dir = _price_extracted_dir()
    jsonl_files = sorted(extracted_dir.glob("wsl_events_*.jsonl")) if extracted_dir.exists() else []
    if not jsonl_files:
        logger.info("No VLM extraction JSONL files in %s — skipping price ingest",
                    extracted_dir)
        return None

    logger.info("Ingesting prices from %d JSONL files", len(jsonl_files))
    summary = run_price_postprocess(extracted_dir=extracted_dir, staging_dir=STAGING_DIR)
    if summary.get("status") != "success" or summary.get("n_prices", 0) == 0:
        logger.info("Price post-process produced no rows — skipping")
        return None

    prices_path = STAGING_DIR / "wsl_prices_v4.parquet"
    prices_df = pd.read_parquet(prices_path)

    try:
        result = validate_wsl_prices(prices_df, strict=True)
    except PriceValidationError:
        log_extraction_run(
            stage="wsl_price_ingest",
            config={"n_jsonl_files": len(jsonl_files)},
            metrics={"status": "validation_failed",
                     "n_prices": len(prices_df)},
            artifacts={"prices_parquet": str(prices_path)},
            tags={"outcome": "failed"},
        )
        raise

    panel = build_monthly_panel(prices_df, gap_fill=True)
    panel_path = STAGING_DIR / "wsl_price_panel_monthly.parquet"
    panel.to_parquet(panel_path, index=False)

    log_extraction_run(
        stage="wsl_price_ingest",
        config={"n_jsonl_files": len(jsonl_files)},
        metrics={
            "n_prices": len(prices_df),
            "n_panel_cells": len(panel),
            **{f"val_{k}": v for k, v in result.metrics.items()},
        },
        artifacts={
            "prices_parquet": str(prices_path),
            "panel_parquet": str(panel_path),
        },
        tags={"outcome": "passed"},
    )
    return True


def clean_wsl() -> bool:
    """Extract events from WSL PDFs (rule-based fallback when VLM output absent)."""
    from src.parsing.wsl_event_extractor import extract_all_wsl_events
    from src.config import RAW_WSL, STAGING_DIR

    # Prefer VLM-extracted parquet if ingest already produced it.
    v4_path = STAGING_DIR / 'wsl_events_v4.parquet'
    if v4_path.exists():
        logger.info("Using VLM-extracted events at %s — skipping rule-based extractor",
                    v4_path)
        output_path = STAGING_DIR / 'wsl_events.parquet'
        compatibility_path = STAGING_DIR / 'wsl_extracted_events.parquet'
        if not output_path.exists():
            import pandas as pd
            _save_output_variants(pd.read_parquet(v4_path), output_path, compatibility_path)
    else:
        logger.info("Extracting WSL events (rule-based)...")
        wsl_dir = RAW_WSL
        if not (wsl_dir.exists() and any(wsl_dir.rglob('*.pdf'))):
            logger.info("No WSL PDFs found - skipping")
            return None
        df = extract_all_wsl_events(wsl_dir)
        output_path = STAGING_DIR / 'wsl_events.parquet'
        compatibility_path = STAGING_DIR / 'wsl_extracted_events.parquet'
        _save_output_variants(df, output_path, compatibility_path)
        logger.info(f"Extracted {len(df):,} WSL events → {output_path}")

    voyages_path = STAGING_DIR / 'voyages_master.parquet'
    if output_path.exists() and voyages_path.exists():
        from src.entities.wsl_voyage_matcher import run_wsl_crosswalk

        try:
            crosswalk_df, panel_df = run_wsl_crosswalk(
                events_path=output_path,
                voyages_path=voyages_path,
            )
            logger.info(
                "WSL crosswalk complete: %s matched events, %s voyage-panel rows",
                crosswalk_df['voyage_id'].notna().sum(),
                len(panel_df),
            )
        except Exception as exc:
            logger.warning("WSL crosswalk skipped or failed: %s", exc)
    return output_path.exists()


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
        'wsl_vlm_ingest': False,
        'wsl_price_ingest': False,
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
            StepSpec('wsl_vlm_ingest', ingest_wsl_vlm, "VLM ingest failed", failure_level="error"),
            StepSpec('wsl_price_ingest', ingest_wsl_prices, "Price ingest failed", failure_level="error"),
            StepSpec('wsl', clean_wsl, "WSL cleaning skipped"),
        ],
        logger=logger,
    )

    # Summary
    success_count, skipped_count, failed_count = summarize_step_results(results)
    logger.info(
        "Stage 2 complete: %s successful, %s skipped, %s failed",
        success_count,
        skipped_count,
        failed_count,
    )
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_clean()
