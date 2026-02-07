"""
Compass Pipeline — Configuration.

Centralises all tuneable thresholds, model hyper-parameters, and path
conventions.  The canonical source is a JSON file loaded at runtime;
every field has a sensible default so that the pipeline works out of the
box.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# ── defaults ────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OUTPUTS = _PROJECT_ROOT / "outputs" / "compass"


@dataclass
class CompassConfig:
    """All compass-pipeline parameters in one place."""

    # ── Step 1: data validation ─────────────────────────────────────────
    minimum_points_per_voyage: int = 20
    max_speed_mps_plausible: float = 25.0
    gap_threshold_hours: float = 48.0

    # ── Step 2: projection ──────────────────────────────────────────────
    smoothing_enabled: bool = False
    smoothing_window: int = 5

    # ── Step 3: step construction ───────────────────────────────────────
    time_resample_hours: List[float] = field(default_factory=lambda: [6.0, 12.0])
    interp_max_gap_hours: float = 12.0
    distance_thin_meters: List[float] = field(default_factory=lambda: [5000.0, 10000.0])

    # ── Step 4: HMM regime segmentation ─────────────────────────────────
    num_regimes_candidates: List[int] = field(default_factory=lambda: [3, 4])
    min_steps_for_hmm: int = 50
    hmm_n_iter: int = 200
    hmm_tol: float = 1e-4
    hmm_covariance_type: str = "diag"
    hmm_self_transition_init: float = 0.85
    hmm_random_state: int = 42

    # ── Step 5: compass features ────────────────────────────────────────
    hill_tail_k_frac: float = 0.10
    grid_cell_size_m: float = 10_000.0
    loiter_speed_mps_threshold: float = 0.5
    min_search_steps_for_features: int = 30
    bootstrap_reps: int = 0          # 0 ⇒ skip bootstrap SEs

    # ── Step 6: compass index ───────────────────────────────────────────
    pca_n_components: int = 3
    standardize_group_col: Optional[str] = "state_time_cell_id"

    # ── Step 7: early-window ────────────────────────────────────────────
    early_window_search_steps: List[int] = field(default_factory=lambda: [50, 100])
    early_window_days: List[int] = field(default_factory=lambda: [10, 20])

    # ── Step 8: self-supervised embedding ───────────────────────────────
    embedding_enabled: bool = False
    segment_length_steps: int = 128
    embedding_dim: int = 32
    embedding_epochs: int = 20
    embedding_batch_size: int = 256
    embedding_lr: float = 1e-3
    embedding_random_state: int = 42

    # ── Step 9: robustness ──────────────────────────────────────────────
    robustness_enabled: bool = True

    # ── I/O paths ───────────────────────────────────────────────────────
    trajectory_points_path: str = ""
    voyage_metadata_path: str = ""
    outputs_dir: str = str(_DEFAULT_OUTPUTS)

    # ── misc ────────────────────────────────────────────────────────────
    random_seed: int = 42
    verbose: bool = True

    # ── helpers ──────────────────────────────────────────────────────────

    @property
    def output_path(self) -> Path:
        p = Path(self.outputs_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


def load_config(path: Optional[str | Path] = None) -> CompassConfig:
    """Load config from JSON, falling back to defaults for missing keys."""
    if path is None:
        logger.info("No config path supplied — using all defaults.")
        return CompassConfig()
    path = Path(path)
    if not path.exists():
        logger.warning("Config file %s not found — using defaults.", path)
        return CompassConfig()
    with open(path) as f:
        data = json.load(f)
    # Only pass known fields
    known = {f.name for f in CompassConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in known}
    cfg = CompassConfig(**filtered)
    logger.info("Loaded config from %s (%d overrides).", path, len(filtered))
    return cfg


def save_config(cfg: CompassConfig, path: str | Path) -> None:
    """Serialise the current config to JSON for reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(cfg), f, indent=2, default=str)
    logger.info("Saved config to %s.", path)
