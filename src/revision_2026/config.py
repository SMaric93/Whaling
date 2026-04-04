"""
Shared configuration for revision 2026 tests.

Loads from revision_2026_config.yaml if present, else uses defaults.
"""

from __future__ import annotations
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FINAL = PROJECT_ROOT / "data" / "final"
VOYAGE_PARQUET = DATA_FINAL / "analysis_voyage.parquet"

OUTPUT_BASE = PROJECT_ROOT / "outputs" / "revision_2026"
TABLES_DIR = OUTPUT_BASE / "tables"
FIGURES_DIR = OUTPUT_BASE / "figures"
INTERMEDIATES_DIR = OUTPUT_BASE / "intermediates"
LOGS_DIR = OUTPUT_BASE / "logs"

# Post-connectivity manifests (authoritative connected set + types)
POST_CONN_DIR = PROJECT_ROOT / "outputs" / "post_connectivity"
CANONICAL_CONNECTED_SET = POST_CONN_DIR / "manifests" / "canonical_connected_set.parquet"
AUTHORITATIVE_TYPES = POST_CONN_DIR / "manifests" / "type_file_authoritative.parquet"

# Ensure output directories exist
for d in [TABLES_DIR, FIGURES_DIR, INTERMEDIATES_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Threading safety ─────────────────────────────────────────────────────
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


@dataclass
class RevisionConfig:
    """Configuration for all revision 2026 analyses."""

    # Year bounds
    min_year: int = 1780
    max_year: int = 1930

    # Minimum voyage requirements
    min_captain_voyages: int = 1
    min_agent_voyages: int = 1

    # OOS / pre-period cutoff
    oos_cutoff_year: int = 1870

    # Trimming
    output_trim_lower_pct: float = 0.5
    output_trim_upper_pct: float = 99.5

    # Near-zero threshold (percentile of q_total_index)
    near_zero_percentile: float = 5.0

    # Quantile regression taus
    quantile_taus: List[float] = field(
        default_factory=lambda: [0.10, 0.25, 0.50, 0.75, 0.90]
    )

    # Skill bins
    n_skill_bins: int = 4
    low_skill_fraction: float = 0.25

    # Event study
    event_window: int = 3
    event_min_pre: int = 2
    event_min_post: int = 2

    # LOO construction
    use_exact_loo: bool = True
    n_loo_workers: int = 8

    # Clustering
    primary_cluster: str = "captain_id"
    secondary_cluster: str = "agent_id"

    # Reproducibility
    random_seed: int = 42

    # Figures
    figure_dpi: int = 300
    figure_format: str = "png"


def load_config() -> RevisionConfig:
    """Load configuration from YAML if available, else use defaults."""
    cfg_path = PROJECT_ROOT / "revision_2026_config.yaml"
    cfg = RevisionConfig()
    if cfg_path.exists():
        try:
            import yaml
            with open(cfg_path) as f:
                data = yaml.safe_load(f) or {}
            for key, val in data.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, val)
        except ImportError:
            pass  # pyyaml not installed, use defaults
    return cfg


CFG = load_config()
