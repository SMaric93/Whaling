"""
Stage 1: Data Pull

Downloads all raw data sources required for the whaling analysis pipeline.

Data Sources:
    - AOWV (voyages, crew lists, logbooks)
    - Online sources (Starbuck, Maury, Townsend, COML)
    - WSL PDFs (Whalemen's Shipping List)
    - Weather/Climate data (NAO, PDO, AMO, HURDAT, Ice, etc.)
    - Economic data (petroleum prices, WSL market prices)
"""

import logging

logger = logging.getLogger(__name__)


def pull_aowv(force: bool = False) -> bool:
    """Download AOWV voyages, crew lists, and logbooks."""
    from src.download.aowv_downloader import download_all_aowv_data
    from src.config import RAW_AOWV, RAW_CREWLIST, RAW_LOGBOOKS
    
    logger.info("Pulling AOWV data...")
    download_all_aowv_data(force=force)
    logger.info("AOWV data pull complete.")
    return all(path.exists() and any(path.iterdir()) for path in (RAW_AOWV, RAW_CREWLIST, RAW_LOGBOOKS))


def pull_online_sources(force: bool = False) -> bool:
    """Download Starbuck, Maury, Townsend, and COML sources."""
    from src.download.online_sources_downloader import download_all_online_sources
    from src.config import RAW_CONSOLIDATED, RAW_MAURY, RAW_STARBUCK
    
    logger.info("Pulling online sources (Starbuck, Maury, etc.)...")
    download_all_online_sources(force=force)
    logger.info("Online sources pull complete.")
    required_paths = (RAW_STARBUCK, RAW_MAURY, RAW_CONSOLIDATED)
    return all(path.exists() and any(path.iterdir()) for path in required_paths)


def pull_wsl_pdfs(force: bool = False) -> bool:
    """Download Whalemen's Shipping List PDFs."""
    from src.download.wsl_pdf_downloader import download_wsl_pdfs
    from src.config import RAW_WSL
    
    logger.info("Pulling WSL PDFs...")
    try:
        # Keep the pipeline responsive when the NMDL site falls back to
        # speculative PDF URLs; a small sample is enough to validate access.
        download_wsl_pdfs(force=force, max_issues=5)
        logger.info("WSL PDF pull complete.")
        return RAW_WSL.exists() and any(RAW_WSL.glob("*.pdf"))
    except Exception as e:
        logger.warning(f"WSL PDF download skipped or failed: {e}")
        return False


def pull_weather(force: bool = False) -> bool:
    """Download weather and climate data (NAO, PDO, AMO, HURDAT, Ice, etc.)."""
    from src.download.weather_downloader import download_and_integrate_weather
    from src.download.weather_downloader import WEATHER_RAW_DIR
    
    logger.info("Pulling weather/climate data...")
    annual_weather, _ = download_and_integrate_weather(save_raw=True)
    logger.info("Weather data pull complete.")
    return len(annual_weather) > 0 or (WEATHER_RAW_DIR / "weather_annual_combined.csv").exists()


def pull_economic(force: bool = False) -> bool:
    """Download economic data (petroleum prices)."""
    from src.download.economic_downloader import download_and_integrate_economic
    from src.download.economic_downloader import ECONOMIC_RAW_DIR
    
    logger.info("Pulling economic data...")
    try:
        economic_df = download_and_integrate_economic(save_raw=True)
        logger.info("Economic data pull complete.")
        return len(economic_df) > 0 or (ECONOMIC_RAW_DIR / "economic_annual_combined.csv").exists()
    except Exception as e:
        logger.warning(f"Economic data download skipped or failed: {e}")
        return False


def run_pull(force: bool = False, skip_optional: bool = False) -> dict:
    """
    Run the complete data pull stage.
    
    Args:
        force: Re-download even if files exist
        skip_optional: Skip optional sources (WSL PDFs, economic data)
    
    Returns:
        dict: Summary of pull operations
    """
    logger.info("=" * 60)
    logger.info("STAGE 1: DATA PULL")
    logger.info("=" * 60)
    
    results = {
        'aowv': False,
        'online_sources': False,
        'wsl_pdfs': False,
        'weather': False,
        'economic': False,
    }
    
    # Core data sources
    try:
        results['aowv'] = pull_aowv(force=force)
    except Exception as e:
        logger.error(f"AOWV pull failed: {e}")
    
    try:
        results['online_sources'] = pull_online_sources(force=force)
    except Exception as e:
        logger.error(f"Online sources pull failed: {e}")
    
    try:
        results['weather'] = pull_weather(force=force)
    except Exception as e:
        logger.error(f"Weather pull failed: {e}")
    
    # Optional data sources
    if not skip_optional:
        try:
            results['wsl_pdfs'] = pull_wsl_pdfs(force=force)
        except Exception as e:
            logger.warning(f"WSL PDF pull skipped: {e}")
        
        try:
            results['economic'] = pull_economic(force=force)
        except Exception as e:
            logger.warning(f"Economic pull skipped: {e}")
    
    # Summary
    success_count = sum(results.values())
    total_count = len(results)
    logger.info(f"Stage 1 complete: {success_count}/{total_count} sources pulled successfully")
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_pull()
