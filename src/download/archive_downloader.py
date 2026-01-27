"""
Downloader for Mutual Marine Insurance Register from Archive.org.

Downloads:
- PDF version of the register (1843-1862)
- OCR text version for parsing
"""

import requests
from pathlib import Path
from typing import Optional, List
import logging

from ..config import ARCHIVE_ORG_URLS, RAW_INSURANCE
from .manifest import ManifestManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_file(
    url: str,
    target_path: Path,
    timeout: int = 120,
) -> Path:
    """
    Download a single file.
    
    Args:
        url: URL to download from
        target_path: Path to save file to
        timeout: Request timeout in seconds
        
    Returns:
        Path to downloaded file
    """
    logger.info(f"Downloading from {url}")
    
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    response = requests.get(url, timeout=timeout, stream=True)
    response.raise_for_status()
    
    with open(target_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    logger.info(f"  Saved to: {target_path}")
    return target_path


def download_register_pdf(
    manifest: Optional[ManifestManager] = None,
    force: bool = False,
) -> Path:
    """
    Download the Mutual Marine Register PDF.
    
    Args:
        manifest: ManifestManager to record download (creates new if None)
        force: If True, download even if file exists
        
    Returns:
        Path to downloaded file
    """
    if manifest is None:
        manifest = ManifestManager()
    
    url = ARCHIVE_ORG_URLS["register_pdf"]
    target_path = RAW_INSURANCE / "mutual_marine_register_1843_1862.pdf"
    
    # Check if already downloaded
    if target_path.exists() and not force:
        logger.info(f"Register PDF already downloaded: {target_path}")
        return target_path
    
    # Download
    download_file(url, target_path)
    
    # Record in manifest
    manifest.add_entry(
        source_name="Mutual_Marine_Register_PDF",
        download_url=url,
        local_path=target_path,
        license_key="archive_org",
    )
    
    return target_path


def download_register_ocr(
    manifest: Optional[ManifestManager] = None,
    force: bool = False,
) -> Path:
    """
    Download the Mutual Marine Register OCR text.
    
    Args:
        manifest: ManifestManager to record download (creates new if None)
        force: If True, download even if file exists
        
    Returns:
        Path to downloaded file
    """
    if manifest is None:
        manifest = ManifestManager()
    
    url = ARCHIVE_ORG_URLS["register_ocr"]
    target_path = RAW_INSURANCE / "mutual_marine_register_1843_1862_ocr.txt"
    
    # Check if already downloaded
    if target_path.exists() and not force:
        logger.info(f"Register OCR already downloaded: {target_path}")
        return target_path
    
    # Download
    download_file(url, target_path)
    
    # Record in manifest
    manifest.add_entry(
        source_name="Mutual_Marine_Register_OCR",
        download_url=url,
        local_path=target_path,
        license_key="archive_org",
    )
    
    return target_path


def download_archive_data(
    manifest: Optional[ManifestManager] = None,
    force: bool = False,
    pdf: bool = True,
    ocr: bool = True,
) -> List[Path]:
    """
    Download Mutual Marine Register files from Archive.org.
    
    Args:
        manifest: ManifestManager to record downloads (creates new if None)
        force: If True, download even if files exist
        pdf: If True, download PDF version
        ocr: If True, download OCR text version
        
    Returns:
        List of downloaded file paths
    """
    if manifest is None:
        manifest = ManifestManager()
    
    downloaded = []
    
    if pdf:
        downloaded.append(download_register_pdf(manifest, force))
    
    if ocr:
        downloaded.append(download_register_ocr(manifest, force))
    
    return downloaded


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Mutual Marine Register from Archive.org")
    parser.add_argument("--force", action="store_true", help="Force re-download even if files exist")
    parser.add_argument("--pdf-only", action="store_true", help="Download only PDF")
    parser.add_argument("--ocr-only", action="store_true", help="Download only OCR text")
    
    args = parser.parse_args()
    
    manifest = ManifestManager()
    
    pdf = not args.ocr_only
    ocr = not args.pdf_only
    
    download_archive_data(manifest, args.force, pdf=pdf, ocr=ocr)
    
    print(f"\nManifest now contains {len(manifest)} entries")
