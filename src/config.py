"""
Global configuration for the Whaling Data Pipeline.

Implements project conventions for:
- Date formats and time zones
- Standard units for quantities
- String normalization rules
- File paths and constants
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Set
import re

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
STAGING_DIR = DATA_DIR / "staging"
FINAL_DIR = DATA_DIR / "final"
CROSSWALKS_DIR = DATA_DIR / "crosswalks"
DOCS_DIR = PROJECT_ROOT / "docs"

# Raw data subdirectories
RAW_AOWV = RAW_DIR / "aowv"
RAW_CREWLIST = RAW_DIR / "crewlist"
RAW_LOGBOOKS = RAW_DIR / "logbooks"
RAW_INSURANCE = RAW_DIR / "insurance_registers"
RAW_IPUMS = RAW_DIR / "ipums"

# =============================================================================
# DATA SOURCE URLS
# =============================================================================

AOWV_URLS = {
    "voyages": "https://whalinghistory.org/aowv/AmericanOffshoreWhalingVoyages.zip",
    "crewlists": "https://whalinghistory.org/crew/AmericanOffshoreWhalingCrewlists.zip",
    "logbooks": "https://whalinghistory.org/aowl/AmericanOffshoreWhalingLogbookData.zip",
}

ARCHIVE_ORG_URLS = {
    "register_pdf": "https://archive.org/download/registerofwhalin1843mutu/registerofwhalin1843mutu.pdf",
    "register_ocr": "https://archive.org/stream/registerofwhalin1843mutu/registerofwhalin1843mutu_djvu.txt",
}

# Online Voyage Augmentation Pack - New Sources
ONLINE_SOURCE_URLS = {
    # Starbuck (1878) - History of American Whale Fishery
    "starbuck_pdf": "https://archive.org/download/historyofamerica00star/historyofamerica00star.pdf",
    "starbuck_ocr": "https://archive.org/stream/historyofamerica00star/historyofamerica00star_djvu.txt",
    # Logbook datasets from WhalingHistory.org
    "maury_logbooks": "https://whalinghistory.org/maury/MauryLogbookData.zip",
    "townsend_logbooks": "https://whalinghistory.org/townsend/TownsendLogbookData.zip",
    "coml_logbooks": "https://whalinghistory.org/coml/CoMLLogbookData.zip",
    # Whalemen's Shipping List project page (for PDF enumeration)
    "wsl_project_page": "https://nmdl.org/projects/wsl/",
}

# Weather and climate index URLs
WEATHER_DATA_URLS = {
    # Atlantic Oscillation
    "nao_index": "https://climatedataguide.ucar.edu/sites/default/files/2022-03/nao_station_annual.txt",
    "hurdat2": "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2023-051124.txt",
    # Pacific climate indices
    "pdo_index": "https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat",
    "enso_nino34": "https://psl.noaa.gov/data/correlation/nina34.anom.data",
}

# Economic data URLs
ECONOMIC_DATA_URLS = {
    # EIA historical petroleum prices (interactive, may need hardcoded fallback)
    "eia_crude_oil": "https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=pet&s=f000000__3&f=a",
}

# Raw data subdirectories for new sources
RAW_WSL = RAW_DIR / "wsl_pdfs"
RAW_STARBUCK = RAW_DIR / "starbuck"
RAW_MAURY = RAW_DIR / "maury"
RAW_CONSOLIDATED = RAW_DIR / "consolidated_logbooks"
RAW_ICOADS = RAW_DIR / "icoads"
RAW_WEATHER = RAW_DIR / "weather"
RAW_ECONOMIC = RAW_DIR / "economic"

# =============================================================================
# WSL EVENT CONFIGURATION
# =============================================================================

WSL_EVENT_TYPES = [
    "DEPARTURE",
    "ARRIVAL",
    "SPOKEN_WITH",
    "REPORTED_AT",
    "WRECK",
    "LOSS",
    "CAPTURED",
    "DAMAGED",
    "RETURNED_HOME",
    "OTHER",
]

# Match confidence thresholds for crosswalks
MATCH_CONFIDENCE_THRESHOLDS = {
    "high": 0.9,
    "medium": 0.7,
    "low": 0.5,
}

# License notes for new sources
LICENSE_NOTES_ONLINE = {
    "starbuck": "Public Domain - Starbuck (1878) via Archive.org",
    "maury": "CC BY 4.0 - WhalingHistory.org Maury Logbook Data",
    "townsend": "CC BY 4.0 - WhalingHistory.org Townsend Logbook Data",
    "coml": "CC BY 4.0 - WhalingHistory.org Census of Marine Life Data",
    "wsl": "Public Domain - Whalemen's Shipping List via NMDL",
    "icoads": "Public Domain - NOAA ICOADS",
}

# =============================================================================
# DATE AND TIME CONVENTIONS
# =============================================================================

TIMEZONE = "UTC"
DATE_FORMATS = ["YYYY-MM-DD", "YYYY-MM"]
DATE_PARSE_FORMATS = [
    "%Y-%m-%d",
    "%Y-%m",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%Y",
    "%B %d, %Y",
    "%b %d, %Y",
]

# =============================================================================
# STANDARD UNITS
# =============================================================================

@dataclass
class Units:
    oil_barrels: str = "bbl"
    bone_weight: str = "lbs"
    currency: str = "nominal_usd"


UNITS = Units()

# =============================================================================
# STRING NORMALIZATION CONFIGURATION
# =============================================================================

@dataclass
class StringNormConfig:
    """Configuration for name normalization."""
    
    # Convert to uppercase
    case: str = "UPPER"
    
    # Strip punctuation (keep alphanumeric and spaces)
    strip_punctuation: bool = True
    
    # Collapse multiple whitespace to single space
    collapse_whitespace: bool = True
    
    # Titles to remove from names
    remove_titles: Set[str] = field(default_factory=lambda: {
        "CAPT", "CAPTAIN", "MASTER", "MR", "MRS", "DR", "HON", "REV",
        "CAPT.", "MR.", "MRS.", "DR.", "HON.", "REV."
    })
    
    # Suffixes to keep
    keep_suffixes: Set[str] = field(default_factory=lambda: {
        "JR", "SR", "II", "III", "IV", "2ND", "3RD"
    })
    
    # Common name abbreviation expansions
    expand_abbreviations: Dict[str, str] = field(default_factory=lambda: {
        "WM": "WILLIAM",
        "JNO": "JOHN",
        "CHAS": "CHARLES",
        "JAS": "JAMES",
        "GEO": "GEORGE",
        "BENJ": "BENJAMIN",
        "THOS": "THOMAS",
        "SAML": "SAMUEL",
        "DANL": "DANIEL",
        "NATHL": "NATHANIEL",
        "ROBT": "ROBERT",
        "EDWD": "EDWARD",
        "ANDW": "ANDREW",
        "RICHD": "RICHARD",
        "MICHL": "MICHAEL",
        "JONATHN": "JONATHAN",
        "ALEXR": "ALEXANDER",
        "FREDK": "FREDERICK",
    })


STRING_NORM_CONFIG = StringNormConfig()

# =============================================================================
# GEOGRAPHIC CONFIGURATION
# =============================================================================

# Arctic polygon (simplified: latitude > 66.5°N)
ARCTIC_LAT_THRESHOLD = 66.5

# Bering Sea approximate bounding box
BERING_SEA_BOUNDS = {
    "lat_min": 51.0,
    "lat_max": 66.0,
    "lon_min": -180.0,
    "lon_max": -157.0,
}

# Whaling port counties for census matching
WHALING_PORT_COUNTIES = {
    "MA": ["Bristol", "Nantucket", "Dukes", "Suffolk", "Barnstable"],
    "CT": ["New London"],
    "NY": ["Suffolk"],  # Sag Harbor
    "RI": ["Newport", "Providence"],
}

# FIPS codes for whaling states
WHALING_STATE_FIPS = {
    "MA": 25,
    "CT": 9,
    "NY": 36,
    "RI": 44,
}

# =============================================================================
# CROSSWALK MATCHING CONFIGURATION
# =============================================================================

@dataclass
class CrosswalkConfig:
    """Configuration for crosswalk/fuzzy matching."""
    
    # Date tolerance for voyage matching
    out_date_tolerance_days: int = 30
    in_date_tolerance_days: int = 60
    
    # Minimum Jaro-Winkler similarity for name matches
    min_name_similarity: float = 0.85
    
    # Age tolerance for census matching (years)
    age_tolerance: int = 5
    

CROSSWALK_CONFIG = CrosswalkConfig()

# =============================================================================
# LINKAGE CONFIGURATION
# =============================================================================

@dataclass
class LinkageConfig:
    """Configuration for captain-to-census record linkage."""
    
    # Target census years
    target_years: List[int] = field(default_factory=lambda: [1850, 1860, 1870, 1880])
    
    # Match probability thresholds for sensitivity variants
    strict_threshold: float = 0.90
    medium_threshold: float = 0.70
    
    # Weights for probabilistic matching components
    name_weight: float = 0.40
    age_weight: float = 0.20
    geography_weight: float = 0.20
    occupation_weight: float = 0.15
    spouse_weight: float = 0.05
    
    # Occupation codes associated with whaling/maritime
    maritime_occupations: Set[str] = field(default_factory=lambda: {
        "MARINER", "SAILOR", "SEAMAN", "MASTER", "CAPTAIN", "WHALER",
        "SHIPMASTER", "SHIP MASTER", "NAVIGATOR", "MATE", "FIRST MATE",
        "SECOND MATE", "BOATSWAIN", "HARPOONER", "COOPER"
    })


LINKAGE_CONFIG = LinkageConfig()

# =============================================================================
# MANIFEST CONFIGURATION
# =============================================================================

MANIFEST_FILE = PROJECT_ROOT / "manifest.jsonl"

MANIFEST_FIELDS = [
    "source_name",
    "retrieval_date_utc",
    "download_url",
    "local_path",
    "file_hash_sha256",
    "license_or_terms_note",
]

# License notes for data sources
LICENSE_NOTES = {
    "aowv": "CC BY 4.0 - American Offshore Whaling Voyages by NBWM, Mystic Seaport, NHA",
    "archive_org": "CC BY-NC-SA 4.0 - Mutual Marine Insurance Company Records",
    "ipums": "IPUMS USA Terms of Use - Restricted to registered users",
}

# =============================================================================
# VALIDATION CONFIGURATION
# =============================================================================

@dataclass
class ValidationConfig:
    """Configuration for data validation checks."""
    
    # Quantity ranges
    max_oil_barrels: int = 10000
    max_bone_lbs: int = 100000
    min_voyage_days: int = 30
    max_voyage_days: int = 2000
    
    # Rate bounds
    min_rate: float = 0.0
    max_rate: float = 1.0


VALIDATION_CONFIG = ValidationConfig()
