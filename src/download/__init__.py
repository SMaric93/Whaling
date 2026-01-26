"""Data download and manifest management."""

from .aowv_downloader import download_aowv_data
from .archive_downloader import download_archive_data
from .manifest import ManifestManager
from .online_sources_downloader import (
    download_starbuck,
    download_maury_logbooks,
    download_townsend_logbooks,
    download_coml_logbooks,
    download_all_online_sources,
)
from .wsl_pdf_downloader import (
    download_wsl_pdfs,
    build_wsl_issue_index,
    scrape_wsl_pdf_links,
)

__all__ = [
    "download_aowv_data",
    "download_archive_data",
    "ManifestManager",
    "download_starbuck",
    "download_maury_logbooks",
    "download_townsend_logbooks",
    "download_coml_logbooks",
    "download_all_online_sources",
    "download_wsl_pdfs",
    "build_wsl_issue_index",
    "scrape_wsl_pdf_links",
]

