"""
Downloader for Whalemen's Shipping List (WSL) PDFs.

Enumerates PDF links from the NMDL project page and downloads
PDFs for years overlapping with AOWV voyage data.

WSL Coverage: 1843-1914
Target: Years overlapping with AOWV data (approx. 1784-1928)
"""

import requests
from bs4 import BeautifulSoup
import re
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from datetime import datetime
import logging
import hashlib
import json
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ONLINE_SOURCE_URLS, RAW_WSL, STAGING_DIR, LICENSE_NOTES_ONLINE,
)
from download.manifest import ManifestManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# WSL target years - align with AOWV coverage
WSL_TARGET_YEARS = list(range(1843, 1915))  # Full coverage: 1843-1914


def scrape_wsl_pdf_links(
    project_url: Optional[str] = None,
    timeout: int = 60,
) -> List[Dict[str, str]]:
    """
    Scrape the NMDL WSL project page to enumerate available PDF links.
    
    The page uses a dynamic table, so we look for PDF link patterns.
    
    Args:
        project_url: URL of WSL project page (defaults to config)
        timeout: Request timeout
        
    Returns:
        List of dicts with 'url', 'year', 'month', 'day', 'issue_id' keys
    """
    if project_url is None:
        project_url = ONLINE_SOURCE_URLS["wsl_project_page"]
    
    logger.info(f"Fetching WSL project page: {project_url}")
    
    response = requests.get(project_url, timeout=timeout)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    pdf_links = []
    
    # Look for PDF links in the page
    # WSL PDFs are typically named like: wsl_1843_03_21.pdf or similar
    for link in soup.find_all('a', href=True):
        href = link['href']
        
        # Match PDF links
        if '.pdf' in href.lower():
            pdf_info = parse_wsl_pdf_url(href)
            if pdf_info:
                pdf_links.append(pdf_info)
    
    # Also check for any table rows with PDF data
    for row in soup.find_all('tr'):
        cells = row.find_all('td')
        for cell in cells:
            link = cell.find('a', href=True)
            if link and '.pdf' in link['href'].lower():
                pdf_info = parse_wsl_pdf_url(link['href'])
                if pdf_info:
                    # Dedup by URL
                    if pdf_info not in pdf_links:
                        pdf_links.append(pdf_info)
    
    logger.info(f"Found {len(pdf_links)} PDF links on project page")
    
    # If no PDFs found directly, the page might use JavaScript
    # In that case, we'll generate expected URLs based on known patterns
    if len(pdf_links) == 0:
        logger.warning("No PDFs found via scraping. Page may use JavaScript.")
        logger.info("Generating expected PDF URLs based on known patterns...")
        pdf_links = generate_expected_wsl_urls()
    
    return pdf_links


def parse_wsl_pdf_url(url: str) -> Optional[Dict[str, str]]:
    """
    Parse a WSL PDF URL to extract date information.
    
    Expected patterns:
    - wsl_YYYY_MM_DD.pdf
    - wsl-YYYY-MM-DD.pdf
    - YYYY-MM-DD_wsl.pdf
    - Various numeric patterns
    
    Args:
        url: PDF URL or filename
        
    Returns:
        Dict with url, year, month, day, issue_id or None
    """
    # Ensure full URL
    if not url.startswith('http'):
        if url.startswith('/'):
            url = f"https://nmdl.org{url}"
        else:
            url = f"https://nmdl.org/projects/wsl/{url}"
    
    # Extract filename
    filename = url.split('/')[-1].lower()
    
    # Pattern 1: wsl_YYYY_MM_DD.pdf or wsl-YYYY-MM-DD.pdf
    match = re.search(r'wsl[_-](\d{4})[_-](\d{1,2})[_-](\d{1,2})', filename)
    if match:
        year, month, day = match.groups()
        return {
            'url': url,
            'year': int(year),
            'month': int(month),
            'day': int(day),
            'issue_id': f"wsl_{year}_{month.zfill(2)}_{day.zfill(2)}",
        }
    
    # Pattern 2: YYYYMMDD in filename
    match = re.search(r'(\d{4})(\d{2})(\d{2})', filename)
    if match:
        year, month, day = match.groups()
        if 1843 <= int(year) <= 1914:
            return {
                'url': url,
                'year': int(year),
                'month': int(month),
                'day': int(day),
                'issue_id': f"wsl_{year}_{month}_{day}",
            }
    
    # Pattern 3: Just year in filename
    match = re.search(r'(\d{4})', filename)
    if match:
        year = int(match.group(1))
        if 1843 <= year <= 1914:
            return {
                'url': url,
                'year': year,
                'month': None,
                'day': None,
                'issue_id': f"wsl_{year}_unknown",
            }
    
    return None


def generate_expected_wsl_urls(
    years: Optional[List[int]] = None,
    sample_size: int = 10,
) -> List[Dict[str, str]]:
    """
    Generate expected WSL PDF URLs for a sample of issues.
    
    For validation purposes, we generate a small sample of expected URLs
    based on known NMDL hosting patterns.
    
    Args:
        years: List of years to generate URLs for (defaults to sample)
        sample_size: Number of issues to sample per decade
        
    Returns:
        List of expected PDF info dicts
    """
    if years is None:
        # Sample years across the coverage period for validation
        years = [1845, 1850, 1855, 1860, 1865, 1870, 1875, 1880, 1885, 1890]
    
    expected_urls = []
    
    # Common WSL publication dates (weekly on Tuesdays, mostly)
    sample_dates = [
        (1, 7), (2, 14), (3, 21), (4, 4), (5, 2), (6, 13),
        (7, 4), (8, 8), (9, 12), (10, 3), (11, 14), (12, 5),
    ]
    
    base_url = "https://nmdl.org/projects/wsl/pdfs"
    
    for year in years:
        for month, day in sample_dates[:sample_size]:
            issue_id = f"wsl_{year}_{month:02d}_{day:02d}"
            url = f"{base_url}/{year}/{issue_id}.pdf"
            expected_urls.append({
                'url': url,
                'year': year,
                'month': month,
                'day': day,
                'issue_id': issue_id,
            })
    
    logger.info(f"Generated {len(expected_urls)} expected WSL URLs for validation")
    return expected_urls


def download_wsl_pdf(
    pdf_info: Dict[str, str],
    target_dir: Path,
    timeout: int = 120,
) -> Optional[Path]:
    """
    Download a single WSL PDF issue.
    
    Args:
        pdf_info: Dict with 'url', 'year', 'issue_id' keys
        target_dir: Base directory for downloads
        timeout: Request timeout
        
    Returns:
        Path to downloaded file or None if failed
    """
    url = pdf_info['url']
    year = pdf_info['year']
    issue_id = pdf_info['issue_id']
    
    # Create year subdirectory
    year_dir = target_dir / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)
    
    target_path = year_dir / f"{issue_id}.pdf"
    
    # Skip if already exists
    if target_path.exists():
        logger.debug(f"Already exists: {target_path}")
        return target_path
    
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        
        with open(target_path, 'wb') as f:
            f.write(response.content)
        
        logger.info(f"Downloaded: {target_path.name}")
        return target_path
        
    except requests.exceptions.RequestException as e:
        logger.warning(f"Failed to download {url}: {e}")
        return None


def compute_file_hash(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def download_wsl_pdfs(
    target_years: Optional[List[int]] = None,
    manifest: Optional[ManifestManager] = None,
    force: bool = False,
    max_issues: Optional[int] = None,
) -> Tuple[List[Path], pd.DataFrame]:
    """
    Download WSL PDFs for target years and build issue index.
    
    Args:
        target_years: Years to download (defaults to overlapping with AOWV)
        manifest: ManifestManager to record downloads
        force: If True, re-download even if files exist
        max_issues: Maximum number of issues to download (for testing)
        
    Returns:
        Tuple of (list of downloaded paths, issue index DataFrame)
    """
    if manifest is None:
        manifest = ManifestManager()
    
    if target_years is None:
        target_years = WSL_TARGET_YEARS
    
    logger.info("=" * 60)
    logger.info("Downloading Whalemen's Shipping List PDFs")
    logger.info(f"Target years: {min(target_years)}-{max(target_years)}")
    logger.info("=" * 60)
    
    # Get PDF links
    pdf_links = scrape_wsl_pdf_links()
    
    # Filter to target years
    target_pdfs = [p for p in pdf_links if p['year'] in target_years]
    logger.info(f"Found {len(target_pdfs)} issues in target year range")
    
    if max_issues:
        target_pdfs = target_pdfs[:max_issues]
        logger.info(f"Limited to {max_issues} issues for testing")
    
    # Download PDFs
    downloaded = []
    index_rows = []
    
    for i, pdf_info in enumerate(target_pdfs):
        logger.info(f"[{i+1}/{len(target_pdfs)}] {pdf_info['issue_id']}")
        
        path = download_wsl_pdf(pdf_info, RAW_WSL)
        
        if path:
            downloaded.append(path)
            
            # Build index row
            index_rows.append({
                'wsl_issue_id': pdf_info['issue_id'],
                'year': pdf_info['year'],
                'month': pdf_info.get('month'),
                'day': pdf_info.get('day'),
                'source_url': pdf_info['url'],
                'local_path': str(path),
                'sha256': compute_file_hash(path),
            })
    
    # Build issue index DataFrame
    issue_index = pd.DataFrame(index_rows)
    
    logger.info(f"\nDownloaded {len(downloaded)} WSL PDFs")
    
    # Record in manifest (single entry for the collection)
    if downloaded:
        manifest.add_entry(
            source_name="Whalemens_Shipping_List_PDFs",
            download_url=ONLINE_SOURCE_URLS["wsl_project_page"],
            local_path=RAW_WSL,
            license_note=LICENSE_NOTES_ONLINE["wsl"],
        )
    
    return downloaded, issue_index


def build_wsl_issue_index(
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Build or reload WSL issue index from downloaded PDFs.
    
    Args:
        output_path: Path to save index (defaults to staging)
        
    Returns:
        Issue index DataFrame
    """
    if output_path is None:
        output_path = STAGING_DIR / "wsl_issue_index.parquet"
    
    if not RAW_WSL.exists():
        logger.warning(f"WSL directory not found: {RAW_WSL}")
        return pd.DataFrame()
    
    index_rows = []
    
    for year_dir in sorted(RAW_WSL.iterdir()):
        if not year_dir.is_dir():
            continue
        
        for pdf_path in sorted(year_dir.glob("*.pdf")):
            # Parse issue_id from filename
            issue_id = pdf_path.stem
            
            # Try to extract date from issue_id
            match = re.search(r'wsl_(\d{4})_(\d{2})_(\d{2})', issue_id)
            if match:
                year, month, day = map(int, match.groups())
            else:
                year = int(year_dir.name) if year_dir.name.isdigit() else None
                month, day = None, None
            
            index_rows.append({
                'wsl_issue_id': issue_id,
                'year': year,
                'month': month,
                'day': day,
                'source_url': None,  # Would need to reconstruct
                'local_path': str(pdf_path),
                'sha256': compute_file_hash(pdf_path),
            })
    
    issue_index = pd.DataFrame(index_rows)
    
    if len(issue_index) > 0:
        # Save to staging
        output_path.parent.mkdir(parents=True, exist_ok=True)
        issue_index.to_parquet(output_path)
        issue_index.to_csv(output_path.with_suffix('.csv'), index=False)
        logger.info(f"Saved WSL issue index: {len(issue_index)} issues to {output_path}")
    
    return issue_index


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download Whalemen's Shipping List PDFs"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-download even if files exist"
    )
    parser.add_argument(
        "--years", type=str, default=None,
        help="Comma-separated list of years to download (e.g., '1850,1860,1870')"
    )
    parser.add_argument(
        "--max-issues", type=int, default=None,
        help="Maximum number of issues to download (for testing)"
    )
    parser.add_argument(
        "--build-index-only", action="store_true",
        help="Only build index from existing downloads, don't download"
    )
    
    args = parser.parse_args()
    
    if args.build_index_only:
        index = build_wsl_issue_index()
        print(f"Built index with {len(index)} issues")
    else:
        years = None
        if args.years:
            years = [int(y.strip()) for y in args.years.split(',')]
        
        manifest = ManifestManager()
        downloaded, index = download_wsl_pdfs(
            target_years=years,
            manifest=manifest,
            force=args.force,
            max_issues=args.max_issues,
        )
        
        # Save index
        if len(index) > 0:
            index_path = STAGING_DIR / "wsl_issue_index.parquet"
            index_path.parent.mkdir(parents=True, exist_ok=True)
            index.to_parquet(index_path)
            index.to_csv(index_path.with_suffix('.csv'), index=False)
            print(f"\nSaved index: {index_path}")
