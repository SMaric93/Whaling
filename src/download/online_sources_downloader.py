"""
Downloader for Online Voyage Augmentation Pack sources.

Downloads:
- Starbuck (1878) PDF and OCR text from Archive.org
- Maury Logbook Data from WhalingHistory.org
- Townsend Logbook Data from WhalingHistory.org
- Census of Marine Life (CoML) Logbook Data from WhalingHistory.org
"""

import requests
import zipfile
import io
from pathlib import Path
from typing import Optional, List, Dict
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ONLINE_SOURCE_URLS, RAW_STARBUCK, RAW_MAURY, RAW_CONSOLIDATED,
    LICENSE_NOTES_ONLINE,
)
from download.manifest import ManifestManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_file(url: str, target_path: Path, timeout: int = 120) -> Path:
    """
    Download a single file with progress logging.
    
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
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    with open(target_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = (downloaded / total_size) * 100
                if downloaded % (1024 * 1024) < 8192:  # Log every ~1MB
                    logger.info(f"  Progress: {pct:.1f}% ({downloaded:,} / {total_size:,} bytes)")
    
    logger.info(f"  Saved to: {target_path} ({downloaded:,} bytes)")
    return target_path


def download_and_extract_zip(
    url: str,
    target_dir: Path,
    source_name: str,
    timeout: int = 300,
) -> List[Path]:
    """
    Download a ZIP file and extract its contents.
    
    Args:
        url: URL to download from
        target_dir: Directory to extract files to
        source_name: Name for logging
        timeout: Request timeout in seconds
        
    Returns:
        List of extracted file paths
    """
    logger.info(f"Downloading {source_name} ZIP from {url}")
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    
    extracted_files = []
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        for member in zf.namelist():
            # Skip directories and hidden files
            if member.endswith('/') or member.startswith('__MACOSX'):
                continue
            
            # Extract to target directory
            target_path = target_dir / Path(member).name
            with zf.open(member) as src, open(target_path, 'wb') as dst:
                dst.write(src.read())
            
            extracted_files.append(target_path)
            logger.info(f"  Extracted: {target_path.name}")
    
    logger.info(f"  Extracted {len(extracted_files)} files to {target_dir}")
    return extracted_files


def download_starbuck(
    manifest: Optional[ManifestManager] = None,
    force: bool = False,
) -> Dict[str, Path]:
    """
    Download Starbuck (1878) PDF and OCR text from Archive.org.
    
    Args:
        manifest: ManifestManager to record downloads
        force: If True, download even if files exist
        
    Returns:
        Dict with 'pdf' and 'ocr' keys mapping to file paths
    """
    if manifest is None:
        manifest = ManifestManager()
    
    downloaded = {}
    
    # Download PDF
    pdf_url = ONLINE_SOURCE_URLS["starbuck_pdf"]
    pdf_path = RAW_STARBUCK / "starbuck_1878.pdf"
    
    if pdf_path.exists() and not force:
        logger.info(f"Starbuck PDF already exists: {pdf_path}")
        downloaded["pdf"] = pdf_path
    else:
        download_file(pdf_url, pdf_path, timeout=300)
        manifest.add_entry(
            source_name="Starbuck_1878_PDF",
            download_url=pdf_url,
            local_path=pdf_path,
            license_note=LICENSE_NOTES_ONLINE["starbuck"],
        )
        downloaded["pdf"] = pdf_path
    
    # Download OCR text
    ocr_url = ONLINE_SOURCE_URLS["starbuck_ocr"]
    ocr_path = RAW_STARBUCK / "starbuck_1878_ocr.txt"
    
    if ocr_path.exists() and not force:
        logger.info(f"Starbuck OCR already exists: {ocr_path}")
        downloaded["ocr"] = ocr_path
    else:
        download_file(ocr_url, ocr_path, timeout=120)
        manifest.add_entry(
            source_name="Starbuck_1878_OCR",
            download_url=ocr_url,
            local_path=ocr_path,
            license_note=LICENSE_NOTES_ONLINE["starbuck"],
        )
        downloaded["ocr"] = ocr_path
    
    return downloaded


def download_maury_logbooks(
    manifest: Optional[ManifestManager] = None,
    force: bool = False,
) -> List[Path]:
    """
    Download Maury Logbook Data from WhalingHistory.org.
    
    Args:
        manifest: ManifestManager to record downloads
        force: If True, download even if files exist
        
    Returns:
        List of extracted file paths
    """
    if manifest is None:
        manifest = ManifestManager()
    
    url = ONLINE_SOURCE_URLS["maury_logbooks"]
    
    # Check if already downloaded
    if RAW_MAURY.exists() and list(RAW_MAURY.glob("*.txt")) and not force:
        existing = list(RAW_MAURY.glob("*"))
        logger.info(f"Maury data already exists: {len(existing)} files in {RAW_MAURY}")
        return existing
    
    extracted = download_and_extract_zip(url, RAW_MAURY, "Maury Logbooks")
    
    # Record in manifest
    manifest.add_entry(
        source_name="Maury_Logbook_Data",
        download_url=url,
        local_path=RAW_MAURY,
        license_note=LICENSE_NOTES_ONLINE["maury"],
    )
    
    return extracted


def download_townsend_logbooks(
    manifest: Optional[ManifestManager] = None,
    force: bool = False,
) -> List[Path]:
    """
    Download Townsend Logbook Data from WhalingHistory.org.
    
    Args:
        manifest: ManifestManager to record downloads
        force: If True, download even if files exist
        
    Returns:
        List of extracted file paths
    """
    if manifest is None:
        manifest = ManifestManager()
    
    url = ONLINE_SOURCE_URLS["townsend_logbooks"]
    target_dir = RAW_CONSOLIDATED / "townsend"
    
    # Check if already downloaded
    if target_dir.exists() and list(target_dir.glob("*")) and not force:
        existing = list(target_dir.glob("*"))
        logger.info(f"Townsend data already exists: {len(existing)} files in {target_dir}")
        return existing
    
    extracted = download_and_extract_zip(url, target_dir, "Townsend Logbooks")
    
    manifest.add_entry(
        source_name="Townsend_Logbook_Data",
        download_url=url,
        local_path=target_dir,
        license_note=LICENSE_NOTES_ONLINE["townsend"],
    )
    
    return extracted


def download_coml_logbooks(
    manifest: Optional[ManifestManager] = None,
    force: bool = False,
) -> List[Path]:
    """
    Download Census of Marine Life Logbook Data from WhalingHistory.org.
    
    Args:
        manifest: ManifestManager to record downloads
        force: If True, download even if files exist
        
    Returns:
        List of extracted file paths
    """
    if manifest is None:
        manifest = ManifestManager()
    
    url = ONLINE_SOURCE_URLS["coml_logbooks"]
    target_dir = RAW_CONSOLIDATED / "coml"
    
    # Check if already downloaded
    if target_dir.exists() and list(target_dir.glob("*")) and not force:
        existing = list(target_dir.glob("*"))
        logger.info(f"CoML data already exists: {len(existing)} files in {target_dir}")
        return existing
    
    extracted = download_and_extract_zip(url, target_dir, "CoML Logbooks")
    
    manifest.add_entry(
        source_name="CoML_Logbook_Data",
        download_url=url,
        local_path=target_dir,
        license_note=LICENSE_NOTES_ONLINE["coml"],
    )
    
    return extracted


def download_all_online_sources(
    manifest: Optional[ManifestManager] = None,
    force: bool = False,
    skip_wsl: bool = True,  # WSL has separate downloader
) -> Dict[str, any]:
    """
    Download all online sources for the Voyage Augmentation Pack.
    
    Args:
        manifest: ManifestManager to record downloads
        force: If True, download even if files exist
        skip_wsl: If True, skip WSL PDFs (handled by wsl_pdf_downloader)
        
    Returns:
        Dict mapping source names to downloaded paths/files
    """
    if manifest is None:
        manifest = ManifestManager()
    
    results = {}
    
    logger.info("=" * 60)
    logger.info("Downloading Online Voyage Augmentation Pack sources")
    logger.info("=" * 60)
    
    # Starbuck
    logger.info("\n[1/4] Downloading Starbuck (1878)...")
    results["starbuck"] = download_starbuck(manifest, force)
    
    # Maury
    logger.info("\n[2/4] Downloading Maury Logbook Data...")
    results["maury"] = download_maury_logbooks(manifest, force)
    
    # Townsend
    logger.info("\n[3/4] Downloading Townsend Logbook Data...")
    results["townsend"] = download_townsend_logbooks(manifest, force)
    
    # CoML
    logger.info("\n[4/4] Downloading Census of Marine Life Data...")
    results["coml"] = download_coml_logbooks(manifest, force)
    
    logger.info("\n" + "=" * 60)
    logger.info("Download complete!")
    logger.info(f"Manifest now contains {len(manifest)} entries")
    logger.info("=" * 60)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download Online Voyage Augmentation Pack sources"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-download even if files exist"
    )
    parser.add_argument(
        "--source", choices=["starbuck", "maury", "townsend", "coml", "all"],
        default="all",
        help="Which source to download (default: all)"
    )
    
    args = parser.parse_args()
    manifest = ManifestManager()
    
    if args.source == "all":
        download_all_online_sources(manifest, args.force)
    elif args.source == "starbuck":
        download_starbuck(manifest, args.force)
    elif args.source == "maury":
        download_maury_logbooks(manifest, args.force)
    elif args.source == "townsend":
        download_townsend_logbooks(manifest, args.force)
    elif args.source == "coml":
        download_coml_logbooks(manifest, args.force)
