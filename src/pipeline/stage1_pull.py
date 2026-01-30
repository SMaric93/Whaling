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
from pathlib import Path

logger = logging.getLogger(__name__)


def pull_aowv(force: bool = False) -> None:
    """Download AOWV voyages, crew lists, and logbooks."""
    from src.download.aowv_downloader import download_all_aowv_data
    
    logger.info("Pulling AOWV data...")
    download_all_aowv_data(force=force)
    logger.info("AOWV data pull complete.")


def pull_online_sources(force: bool = False) -> None:
    """Download Starbuck, Maury, Townsend, and COML sources."""
    from src.download.online_sources_downloader import download_all_online_sources
    
    logger.info("Pulling online sources (Starbuck, Maury, etc.)...")
    download_all_online_sources(force=force)
    logger.info("Online sources pull complete.")


def pull_wsl_pdfs(force: bool = False) -> None:
    """Download Whalemen's Shipping List PDFs."""
    from src.download.wsl_pdf_downloader import download_wsl_pdfs
    
    logger.info("Pulling WSL PDFs...")
    try:
        download_wsl_pdfs(force=force)
        logger.info("WSL PDF pull complete.")
    except Exception as e:
        logger.warning(f"WSL PDF download skipped or failed: {e}")


def pull_weather(force: bool = False) -> None:
    """Download weather and climate data (NAO, PDO, AMO, HURDAT, Ice, etc.)."""
    from src.download.weather_downloader import download_all_weather
    
    logger.info("Pulling weather/climate data...")
    download_all_weather(force=force)
    logger.info("Weather data pull complete.")


def pull_economic(force: bool = False) -> None:
    """Download economic data (petroleum prices)."""
    from src.download.economic_downloader import download_petroleum_prices
    
    logger.info("Pulling economic data...")
    try:
        download_petroleum_prices(force=force)
        logger.info("Economic data pull complete.")
    except Exception as e:
        logger.warning(f"Economic data download skipped or failed: {e}")


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
        pull_aowv(force=force)
        results['aowv'] = True
    except Exception as e:
        logger.error(f"AOWV pull failed: {e}")
    
    try:
        pull_online_sources(force=force)
        results['online_sources'] = True
    except Exception as e:
        logger.error(f"Online sources pull failed: {e}")
    
    try:
        pull_weather(force=force)
        results['weather'] = True
    except Exception as e:
        logger.error(f"Weather pull failed: {e}")
    
    # Optional data sources
    if not skip_optional:
        try:
            pull_wsl_pdfs(force=force)
            results['wsl_pdfs'] = True
        except Exception as e:
            logger.warning(f"WSL PDF pull skipped: {e}")
        
        try:
            pull_economic(force=force)
            results['economic'] = True
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
