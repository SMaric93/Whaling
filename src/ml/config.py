"""
ML Layer — Central Configuration.

Hyperparameters, output paths, random seeds, GPU settings.
All ML modules import from here instead of hardcoding values.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from torch_device import resolve_torch_device

# ── Path constants ──────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "final"
STAGING_DIR = PROJECT_ROOT / "data" / "staging"

# ML-specific output directories
ML_OUTPUT_DIR = PROJECT_ROOT / "outputs"
ML_TABLES_DIR = ML_OUTPUT_DIR / "tables" / "ml"
ML_FIGURES_DIR = ML_OUTPUT_DIR / "figures" / "ml"
ML_MODELS_DIR = ML_OUTPUT_DIR / "models" / "ml"
ML_DATA_DIR = ML_OUTPUT_DIR / "datasets" / "ml"

for _d in (ML_TABLES_DIR, ML_FIGURES_DIR, ML_MODELS_DIR, ML_DATA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Also ensure notebooks dir exists
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


# ── GPU / device detection ──────────────────────────────────────────────────

DEVICE = resolve_torch_device()


# ── ML Configuration dataclass ──────────────────────────────────────────────

@dataclass
class MLConfig:
    """All tuneable ML parameters."""

    # ── random seeds ────────────────────────────────────────────────────
    random_seed: int = 42
    n_bootstrap: int = 100

    # ── split ratios (Split A: rolling time) ────────────────────────────
    train_frac: float = 0.60
    val_frac: float = 0.20
    test_frac: float = 0.20

    # ── split B/C: group holdout ────────────────────────────────────────
    group_holdout_test_frac: float = 0.20
    group_holdout_val_frac: float = 0.10

    # ── tree model defaults ─────────────────────────────────────────────
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    min_samples_leaf: int = 20
    subsample: float = 0.8
    tree_method: str = "hist"  # good default for Apple Silicon

    # ── random forest defaults ──────────────────────────────────────────
    rf_n_estimators: int = 500
    rf_max_depth: int = 12
    rf_min_samples_leaf: int = 10

    # ── HMM defaults ───────────────────────────────────────────────────
    hmm_n_states_range: List[int] = field(
        default_factory=lambda: [3, 4, 5]
    )
    hmm_n_iter: int = 200
    hmm_tol: float = 1e-4

    # ── survival model defaults ─────────────────────────────────────────
    cox_penalizer: float = 0.01

    # ── quantile regression ─────────────────────────────────────────────
    quantiles: List[float] = field(
        default_factory=lambda: [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    )

    # ── interpretation ──────────────────────────────────────────────────
    shap_max_samples: int = 1000
    pdp_grid_resolution: int = 50
    n_permutation_repeats: int = 10

    # ── output ──────────────────────────────────────────────────────────
    figure_dpi: int = 300
    figure_format: str = "png"
    save_fitted_models: bool = True
    torch_device: str = DEVICE

    # ── policy learning ─────────────────────────────────────────────────
    action_classes: List[str] = field(
        default_factory=lambda: [
            "stay_in_patch",
            "move_local",
            "move_long_within_ground",
            "exit_patch",
            "switch_ground",
        ]
    )

    # ── assignment optimizer ────────────────────────────────────────────
    assignment_rules: List[str] = field(
        default_factory=lambda: [
            "observed",
            "pam",
            "aam",
            "constrained_optimal",
            "robust_optimal",
        ]
    )

    # ── subgroup definitions for reporting ──────────────────────────────
    experience_bins: Dict[str, int] = field(
        default_factory=lambda: {"novice_max": 3, "expert_min": 8}
    )
    n_psi_quartiles: int = 4
    n_theta_quartiles: int = 4
    n_scarcity_bins: int = 3


# ── singleton ───────────────────────────────────────────────────────────────

ML_CFG = MLConfig()
