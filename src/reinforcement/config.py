"""
Reinforcement Test Suite — Central Configuration.

All thresholds, column mappings, and output paths live here.
No module should hardcode column names or magic numbers.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# ── path constants ──────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "final"
STAGING_DIR = PROJECT_ROOT / "data" / "staging"
OUTPUT_DIR = PROJECT_ROOT / "output" / "reinforcement"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
MEMOS_DIR = OUTPUT_DIR / "memos"

for _d in (OUTPUT_DIR, TABLES_DIR, FIGURES_DIR, MEMOS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ── column name mappings (abstract → actual) ───────────────────────────────

@dataclass(frozen=True)
class ColumnMap:
    """Maps abstract variable names to actual column names in data."""

    # Identifiers
    voyage_id: str = "voyage_id"
    captain_id: str = "captain_id"
    agent_id: str = "agent_id"
    vessel_id: str = "vessel_id"

    # Dates
    year_out: str = "year_out"
    year_in: str = "year_in"
    date_out: str = "date_out"
    date_in: str = "date_in"
    obs_date: str = "obs_date"

    # Output
    q_oil_bbl: str = "q_oil_bbl"
    q_total_index: str = "q_total_index"
    log_q: str = "log_q"

    # Geography
    ground_or_route: str = "ground_or_route"
    route_time: str = "route_year_cell"
    home_port: str = "home_port"
    lat: str = "lat"
    lon: str = "lon"

    # Encounter data (from expanded logbook parser)
    encounter: str = "encounter"
    species: str = "species"
    n_struck: str = "n_struck"
    n_tried: str = "n_tried"

    # Vessel/crew controls
    tonnage: str = "tonnage"
    rig: str = "rig"
    crew_count: str = "crew_count"
    desertion_rate: str = "desertion_rate"

    # AKM effects
    theta: str = "theta"
    psi: str = "psi"
    theta_heldout: str = "theta_heldout"
    psi_heldout: str = "psi_heldout"

    # Switch indicators
    switch_agent: str = "switch_agent"
    switch_vessel: str = "switch_vessel"

    # Experience
    captain_experience: str = "captain_experience"
    captain_voyage_num: str = "captain_voyage_num"


COLS = ColumnMap()


# ── analysis parameters ────────────────────────────────────────────────────

@dataclass
class ReinforcementConfig:
    """All tuneable parameters for the reinforcement test suite."""

    # ── sample filters ──────────────────────────────────────────────────
    min_year: int = 1780
    max_year: int = 1930
    min_captain_voyages: int = 2
    min_agent_voyages: int = 2
    output_trim_lower_pct: float = 0.5
    output_trim_upper_pct: float = 99.5

    # ── ground spell construction ───────────────────────────────────────
    # Whaling ground bounding boxes (reused from logbook_features.py)
    min_ground_spell_days: int = 3
    max_ground_transit_gap_days: int = 2

    # ── patch spell construction ────────────────────────────────────────
    patch_radii_nm: List[float] = field(
        default_factory=lambda: [25.0, 50.0, 100.0]
    )
    default_patch_radius_nm: float = 50.0
    min_patch_duration_days: int = 2
    patch_revisit_window_days: int = 7

    # ── experience bins ─────────────────────────────────────────────────
    novice_max_voyages: int = 3
    expert_min_voyages: int = 8
    experience_tercile_method: str = "voyage_count"  # or "years"

    # ── type estimation ─────────────────────────────────────────────────
    cross_fit_n_folds: int = 5
    cross_fit_method: str = "time_split"  # "time_split" or "kfold"
    cross_fit_seed: int = 42

    # ── test 1: map vs compass ──────────────────────────────────────────
    min_movers_per_ground: int = 5
    event_study_window: int = 5
    event_study_omit_period: int = -1

    # ── test 3: stopping rule ───────────────────────────────────────────
    empty_streak_threshold: int = 3  # consecutive NoEnc days
    season_length_days: int = 180

    # ── test 5: submodularity ───────────────────────────────────────────
    n_scarcity_bins: int = 3
    n_skill_bins: int = 4
    n_capability_bins: int = 4
    spline_df: int = 4  # degrees of freedom for flexible f(theta), g(psi)

    # ── inference ───────────────────────────────────────────────────────
    default_cluster_var: str = "captain_id"
    alt_cluster_vars: List[str] = field(
        default_factory=lambda: ["agent_id", "captain_agent"]
    )
    wild_bootstrap_reps: int = 999
    wild_bootstrap_seed: int = 42
    significance_levels: List[float] = field(
        default_factory=lambda: [0.10, 0.05, 0.01]
    )

    # ── output ──────────────────────────────────────────────────────────
    figure_dpi: int = 300
    figure_format: str = "png"
    random_seed: int = 42


# ── singleton config ────────────────────────────────────────────────────────

CFG = ReinforcementConfig()
