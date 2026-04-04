"""
Phase 2 shared configuration and paths.

Reuses Phase 1 output_schema and the core package loaders.
"""
from __future__ import annotations
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FINAL   = PROJECT_ROOT / "data" / "final"
VOYAGE_PARQUET = DATA_FINAL / "analysis_voyage.parquet"

# Phase 1 intermediates (effect objects we can reuse)
P1_INTERMEDIATES = PROJECT_ROOT / "outputs" / "revision_2026" / "intermediates"

# Phase 2 output tree
OUTPUT_BASE      = PROJECT_ROOT / "outputs" / "revision_2026_phase2"
TABLES_DIR       = OUTPUT_BASE / "tables"
FIGURES_DIR      = OUTPUT_BASE / "figures"
INTERMEDIATES_DIR= OUTPUT_BASE / "intermediates"
LOGS_DIR         = OUTPUT_BASE / "logs"

for d in [TABLES_DIR, FIGURES_DIR, INTERMEDIATES_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

@dataclass
class P2Config:
    near_zero_percentile: float = 5.0
    n_skill_bins: int = 4
    low_skill_fraction: float = 0.25
    quantile_taus: List[float] = field(default_factory=lambda: [0.10, 0.25, 0.50, 0.75, 0.90])
    n_bootstrap: int = 499
    n_loo_workers: int = 8
    event_windows: List[int] = field(default_factory=lambda: [2, 3])
    trim_pcts: List[float] = field(default_factory=lambda: [0.0, 0.5, 1.0])
    random_seed: int = 42
    figure_dpi: int = 300
    figure_format: str = "png"

CFG = P2Config()
