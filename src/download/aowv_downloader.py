"""
Downloader for American Offshore Whaling Voyages (AOWV) data from WhalingHistory.org.

Downloads and extracts:
- AOWV Voyages dataset
- AOWV Crew Lists dataset
- AOWV Logbook data
"""

import requests
import zipfile
import io
from pathlib import Path
from typing import Optional, List, Tuple
import logging

from ..config import AOWV_URLS, RAW_AOWV, RAW_CREWLIST, RAW_LOGBOOKS, LICENSE_NOTES
from .manifest import ManifestManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_and_extract_zip(
    url: str,
    extract_dir: Path,
    timeout: int = 120,
) -> List[Path]:
    """
    Download a ZIP file and extract its contents.
    
    Args:
        url: URL to download from
        extract_dir: Directory to extract contents into
        timeout: Request timeout in seconds
        
    Returns:
        List of paths to extracted files
    """
    logger.info(f"Downloading from {url}")
    
    response = requests.get(url, timeout=timeout, stream=True)
    response.raise_for_status()
    
    # Extract ZIP contents
    extract_dir.mkdir(parents=True, exist_ok=True)
    extracted_files = []
    
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        for member in zf.namelist():
            # Skip directories and hidden files
            if member.endswith('/') or member.startswith('__MACOSX'):
                continue
            
            # Extract file
            target_path = extract_dir / Path(member).name
            with zf.open(member) as source, open(target_path, 'wb') as target:
                target.write(source.read())
            
            extracted_files.append(target_path)
            logger.info(f"  Extracted: {target_path.name}")
    
    return extracted_files


def download_aowv_voyages(
    manifest: Optional[ManifestManager] = None,
    force: bool = False,
) -> List[Path]:
    """
    Download AOWV Voyages dataset.
    
    Args:
        manifest: ManifestManager to record download (creates new if None)
        force: If True, download even if files exist
        
    Returns:
        List of paths to downloaded files
    """
    if manifest is None:
        manifest = ManifestManager()
    
    url = AOWV_URLS["voyages"]
    extract_dir = RAW_AOWV
    
    # Check if already downloaded
    existing_files = list(extract_dir.glob("*.csv")) + list(extract_dir.glob("*.txt"))
    if existing_files and not force:
        logger.info(f"AOWV Voyages already downloaded: {len(existing_files)} files")
        return existing_files
    
    # Download and extract
    extracted = download_and_extract_zip(url, extract_dir)
    
    # Record in manifest
    for filepath in extracted:
        manifest.add_entry(
            source_name="AOWV_Voyages",
            download_url=url,
            local_path=filepath,
            license_key="aowv",
        )
    
    logger.info(f"Downloaded AOWV Voyages: {len(extracted)} files")
    return extracted


def download_aowv_crewlists(
    manifest: Optional[ManifestManager] = None,
    force: bool = False,
) -> List[Path]:
    """
    Download AOWV Crew Lists dataset.
    
    Args:
        manifest: ManifestManager to record download (creates new if None)
        force: If True, download even if files exist
        
    Returns:
        List of paths to downloaded files
    """
    if manifest is None:
        manifest = ManifestManager()
    
    url = AOWV_URLS["crewlists"]
    extract_dir = RAW_CREWLIST
    
    # Check if already downloaded
    existing_files = list(extract_dir.glob("*.csv")) + list(extract_dir.glob("*.txt"))
    if existing_files and not force:
        logger.info(f"AOWV Crew Lists already downloaded: {len(existing_files)} files")
        return existing_files
    
    # Download and extract
    extracted = download_and_extract_zip(url, extract_dir)
    
    # Record in manifest
    for filepath in extracted:
        manifest.add_entry(
            source_name="AOWV_CrewLists",
            download_url=url,
            local_path=filepath,
            license_key="aowv",
        )
    
    logger.info(f"Downloaded AOWV Crew Lists: {len(extracted)} files")
    return extracted


def download_aowv_logbooks(
    manifest: Optional[ManifestManager] = None,
    force: bool = False,
) -> List[Path]:
    """
    Download AOWV Logbook data.
    
    Args:
        manifest: ManifestManager to record download (creates new if None)
        force: If True, download even if files exist
        
    Returns:
        List of paths to downloaded files
    """
    if manifest is None:
        manifest = ManifestManager()
    
    url = AOWV_URLS["logbooks"]
    extract_dir = RAW_LOGBOOKS
    
    # Check if already downloaded
    existing_files = list(extract_dir.glob("*.csv")) + list(extract_dir.glob("*.txt"))
    if existing_files and not force:
        logger.info(f"AOWV Logbooks already downloaded: {len(existing_files)} files")
        return existing_files
    
    # Download and extract
    extracted = download_and_extract_zip(url, extract_dir)
    
    # Record in manifest
    for filepath in extracted:
        manifest.add_entry(
            source_name="AOWV_LogbookData",
            download_url=url,
            local_path=filepath,
            license_key="aowv",
        )
    
    logger.info(f"Downloaded AOWV Logbooks: {len(extracted)} files")
    return extracted


def download_aowv_data(
    manifest: Optional[ManifestManager] = None,
    force: bool = False,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Download all AOWV datasets.
    
    Args:
        manifest: ManifestManager to record downloads (creates new if None)
        force: If True, download even if files exist
        
    Returns:
        Tuple of (voyage_files, crew_files, logbook_files)
    """
    if manifest is None:
        manifest = ManifestManager()
    
    voyages = download_aowv_voyages(manifest, force)
    crews = download_aowv_crewlists(manifest, force)
    logbooks = download_aowv_logbooks(manifest, force)
    
    return voyages, crews, logbooks


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download AOWV data from WhalingHistory.org")
    parser.add_argument("--force", action="store_true", help="Force re-download even if files exist")
    parser.add_argument("--voyages-only", action="store_true", help="Download only voyages dataset")
    parser.add_argument("--crew-only", action="store_true", help="Download only crew lists dataset")
    parser.add_argument("--logbooks-only", action="store_true", help="Download only logbooks dataset")
    
    args = parser.parse_args()
    
    manifest = ManifestManager()
    
    if args.voyages_only:
        download_aowv_voyages(manifest, args.force)
    elif args.crew_only:
        download_aowv_crewlists(manifest, args.force)
    elif args.logbooks_only:
        download_aowv_logbooks(manifest, args.force)
    else:
        download_aowv_data(manifest, args.force)
    
    print(f"\nManifest now contains {len(manifest)} entries")
