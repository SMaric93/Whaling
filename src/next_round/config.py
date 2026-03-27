"""
Shared configuration for next-round tests.

Reuses existing ML config and reinforcement data_builder.
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FINAL = PROJECT_ROOT / "data" / "final"
DATA_DERIVED = PROJECT_ROOT / "data" / "derived"
OUTPUTS_TABLES = PROJECT_ROOT / "outputs" / "tables" / "next_round"
OUTPUTS_FIGURES = PROJECT_ROOT / "outputs" / "figures" / "next_round"
DEBUG_TABLES = PROJECT_ROOT / "outputs" / "tables" / "debug"
DEBUG_DOCS = PROJECT_ROOT / "docs" / "debug"
DOCS_DIR = PROJECT_ROOT / "docs"

# Ensure output dirs exist
for d in [DATA_DERIVED, OUTPUTS_TABLES, OUTPUTS_FIGURES, DEBUG_TABLES, DEBUG_DOCS, DOCS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Threading safety (must be set before numpy/sklearn imports) ──────────
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

# ── Shared constants ─────────────────────────────────────────────────────
RANDOM_SEED = 42
PSI_COL = "psi"
THETA_COL = "theta"
# ML-specific datasets use psi_hat_holdout / theta_hat_holdout.
# The voyage panel from data_builder uses psi / theta.
PSI_COL_ML = "psi_hat_holdout"
THETA_COL_ML = "theta_hat_holdout"
N_PSI_QUARTILES = 4
FIGURE_DPI = 150
FIGURE_FORMAT = "png"
