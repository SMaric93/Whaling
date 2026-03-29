"""Shared public API exports for the ``src`` package and compatibility shim."""

from __future__ import annotations

from .config import (
    CROSSWALK_CONFIG,
    FINAL_DIR,
    LINKAGE_CONFIG,
    ML_SHIFT_CONFIG,
    PROJECT_ROOT,
    RAW_DIR,
    STAGING_DIR,
    VALIDATION_CONFIG,
)
from .download import ManifestManager, download_aowv_data
from .entities import EntityResolver
from .parsing import jaro_winkler_similarity, normalize_name

__version__ = "0.626"


def run_all_analyses(*args, **kwargs):
    """Run all regression analyses. See ``src.analyses.run_all`` for details."""
    from .analyses.run_all import run_all_analyses as _run

    return _run(*args, **kwargs)


def prepare_analysis_sample(*args, **kwargs):
    """Prepare the analysis sample. See ``src.analyses.data_loader`` for details."""
    from .analyses.data_loader import prepare_analysis_sample as _prepare

    return _prepare(*args, **kwargs)


PUBLIC_API = {
    "PROJECT_ROOT": PROJECT_ROOT,
    "RAW_DIR": RAW_DIR,
    "STAGING_DIR": STAGING_DIR,
    "FINAL_DIR": FINAL_DIR,
    "LINKAGE_CONFIG": LINKAGE_CONFIG,
    "CROSSWALK_CONFIG": CROSSWALK_CONFIG,
    "VALIDATION_CONFIG": VALIDATION_CONFIG,
    "ML_SHIFT_CONFIG": ML_SHIFT_CONFIG,
    "download_aowv_data": download_aowv_data,
    "ManifestManager": ManifestManager,
    "normalize_name": normalize_name,
    "jaro_winkler_similarity": jaro_winkler_similarity,
    "EntityResolver": EntityResolver,
    "run_all_analyses": run_all_analyses,
    "prepare_analysis_sample": prepare_analysis_sample,
}

__all__ = ["__version__", *PUBLIC_API.keys()]
