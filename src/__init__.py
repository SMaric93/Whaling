"""
Whaling Data Pipeline - Venture Capital of the Sea.

A comprehensive data pipeline for assembling, analyzing, and linking
American whaling voyage data with census records for captain wealth analysis.

Public API
----------
Core configuration:
    PROJECT_ROOT, RAW_DIR, STAGING_DIR, FINAL_DIR
    LINKAGE_CONFIG, CROSSWALK_CONFIG, VALIDATION_CONFIG

Download utilities:
    download_aowv_data, ManifestManager

Parsing utilities:
    normalize_name, jaro_winkler_similarity

Entity resolution:
    EntityResolver

Analysis:
    run_all_analyses, prepare_analysis_sample
"""

__version__ = "0.23"


# Re-export core configuration
from .config import (
    PROJECT_ROOT,
    RAW_DIR,
    STAGING_DIR,
    FINAL_DIR,
    LINKAGE_CONFIG,
    CROSSWALK_CONFIG,
    VALIDATION_CONFIG,
)

# Re-export key download utilities
from .download import (
    download_aowv_data,
    ManifestManager,
)

# Re-export key parsing utilities
from .parsing import (
    normalize_name,
    jaro_winkler_similarity,
)

# Re-export entity resolution
from .entities import EntityResolver

# Re-export analysis utilities (lazy imports for performance)
def run_all_analyses(*args, **kwargs):
    """Run all regression analyses. See src.analyses.run_all for details."""
    from .analyses.run_all import run_all_analyses as _run
    return _run(*args, **kwargs)


def prepare_analysis_sample(*args, **kwargs):
    """Prepare analysis sample. See src.analyses.data_loader for details."""
    from .analyses.data_loader import prepare_analysis_sample as _prepare
    return _prepare(*args, **kwargs)


__all__ = [
    # Version
    "__version__",
    # Configuration
    "PROJECT_ROOT",
    "RAW_DIR", 
    "STAGING_DIR",
    "FINAL_DIR",
    "LINKAGE_CONFIG",
    "CROSSWALK_CONFIG",
    "VALIDATION_CONFIG",
    # Download
    "download_aowv_data",
    "ManifestManager",
    # Parsing
    "normalize_name",
    "jaro_winkler_similarity",
    # Entities
    "EntityResolver",
    # Analysis
    "run_all_analyses",
    "prepare_analysis_sample",
]
