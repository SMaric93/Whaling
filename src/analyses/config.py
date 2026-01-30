"""
Configuration for regression specifications and analysis parameters.

Contains all global settings, sample filters, fixed effect definitions,
and exhibit configuration for the whaling empirical analysis module.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum

# =============================================================================
# Path Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "final"
STAGING_DIR = PROJECT_ROOT / "data" / "staging"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Output subdirectories
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
DIAGNOSTICS_DIR = OUTPUT_DIR / "diagnostics"
SUMMARY_DIR = OUTPUT_DIR / "summary"


# =============================================================================
# Sample Filters
# =============================================================================

@dataclass
class SampleConfig:
    """Configuration for sample construction."""
    
    # Year bounds
    min_year: int = 1780
    max_year: int = 1930
    
    # Minimum voyage requirements
    min_captain_voyages: int = 1
    min_agent_voyages: int = 1
    
    # Train/test split for OOS validation (R2, R4)
    oos_cutoff_year: int = 1870
    
    # Period bin size for vessel depreciation controls
    period_bin_years: int = 5
    
    # Minimum observations for event study pre/post
    event_study_min_pre: int = 2
    event_study_min_post: int = 2
    
    # Trimming rules for outliers
    output_trim_lower_pct: float = 0.5
    output_trim_upper_pct: float = 99.5


DEFAULT_SAMPLE = SampleConfig()


@dataclass
class TFPConfig:
    """Configuration for TFP analysis."""
    
    # Regime cutoff
    regime_cutoff_year: int = 1870
    alternative_cutoffs: List[int] = field(default_factory=lambda: [1865, 1875])
    
    # Winsorization
    winsorize_tfp_pct: float = 1.0  # Winsorize at 1st/99th percentile
    
    # Analysis options
    run_chow_test: bool = True
    use_common_support: bool = False  # Restrict to overlapping tonnage range
    use_regime_specific_beta: bool = True  # If False, use pooled beta
    
    # Clustering
    primary_cluster: str = "agent_id"
    secondary_cluster: str = "captain_id"


DEFAULT_TFP_CONFIG = TFPConfig()


# =============================================================================
# Fixed Effect Structures
# =============================================================================

class FixedEffectType(Enum):
    """Types of fixed effects used in regressions."""
    CAPTAIN = "captain_id"
    AGENT = "agent_id"
    VESSEL = "vessel_id"
    ROUTE = "route_or_ground"
    PORT = "home_port"
    TIME = "year_out"
    DECADE = "decade"
    VESSEL_PERIOD = "vessel_period"
    ROUTE_TIME = "route_time"
    PORT_TIME = "port_time"


@dataclass
class RegressionSpec:
    """Specification for a single regression."""
    
    id: str
    name: str
    dependent_var: str
    fixed_effects: List[FixedEffectType]
    controls: List[str] = field(default_factory=list)
    cluster_var: str = "captain_id"
    sample_restriction: Optional[str] = None
    is_main_text: bool = False
    description: str = ""


# =============================================================================
# Regression Specifications (R1-R17)
# =============================================================================

REGRESSIONS: Dict[str, RegressionSpec] = {
    # Main text regressions
    "R1": RegressionSpec(
        id="R1",
        name="Baseline production function with captain and agent effects",
        dependent_var="log_q",
        fixed_effects=[
            FixedEffectType.CAPTAIN,
            FixedEffectType.AGENT,
            FixedEffectType.VESSEL_PERIOD,
            FixedEffectType.ROUTE_TIME,
        ],
        controls=["log_duration", "log_tonnage"],
        cluster_var="captain_id",
        is_main_text=True,
        description="AKM-style decomposition of productivity into captain and agent effects"
    ),
    
    "R2": RegressionSpec(
        id="R2",
        name="OOS prediction of output using pre-period captain effects",
        dependent_var="log_q",
        fixed_effects=[
            FixedEffectType.VESSEL_PERIOD,
            FixedEffectType.ROUTE_TIME,
        ],
        controls=["alpha_hat_train", "log_duration", "log_tonnage"],
        sample_restriction="test_period",
        is_main_text=True,
        description="Portability validation: do captain effects predict out-of-sample?"
    ),
    
    "R3": RegressionSpec(
        id="R3",
        name="Event-time effects around switching to a new agent",
        dependent_var="log_q",
        fixed_effects=[
            FixedEffectType.CAPTAIN,
            FixedEffectType.VESSEL_PERIOD,
            FixedEffectType.ROUTE_TIME,
        ],
        controls=["event_time_dummies"],
        sample_restriction="switchers_with_pre_post",
        is_main_text=True,
        description="Within-captain event study around agent switches"
    ),
    
    "R4": RegressionSpec(
        id="R4",
        name="OOS prediction using pre-period agent effects",
        dependent_var="log_q",
        fixed_effects=[
            FixedEffectType.CAPTAIN,
            FixedEffectType.VESSEL_PERIOD,
            FixedEffectType.ROUTE_TIME,
        ],
        controls=["gamma_hat_train", "log_duration", "log_tonnage"],
        sample_restriction="test_period",
        is_main_text=False,
        description="Agent capability persistence validation"
    ),
    
    "R5": RegressionSpec(
        id="R5",
        name="Skill-capability complementarity",
        dependent_var="log_q",
        fixed_effects=[
            FixedEffectType.CAPTAIN,
            FixedEffectType.AGENT,
            FixedEffectType.VESSEL_PERIOD,
            FixedEffectType.ROUTE_TIME,
        ],
        controls=["alpha_hat_x_gamma_hat"],
        is_main_text=False,
        description="Tests for complementarity between captain and agent quality"
    ),
    
    "R6": RegressionSpec(
        id="R6",
        name="Resilience: adverse conditions × agent capability",
        dependent_var="log_q",
        fixed_effects=[
            FixedEffectType.CAPTAIN,
            FixedEffectType.AGENT,
            FixedEffectType.VESSEL_PERIOD,
            FixedEffectType.ROUTE_TIME,
        ],
        controls=["adverse_Z", "adverse_Z_x_high_cap_agent"],
        is_main_text=True,
        description="High-capability agents may attenuate output losses under adversity"
    ),
    
    "R7": RegressionSpec(
        id="R7",
        name="First stage: ice affects access/feasibility",
        dependent_var="access_indicator",
        fixed_effects=[
            FixedEffectType.VESSEL_PERIOD,
            FixedEffectType.ROUTE_TIME,
        ],
        controls=["ice_anomaly", "log_duration", "log_tonnage"],
        sample_restriction="arctic_subsample",
        is_main_text=False,
        description="Confirms ice materially shifts feasibility"
    ),
    
    "R8": RegressionSpec(
        id="R8",
        name="Reduced form: ice affects output",
        dependent_var="log_q",
        fixed_effects=[
            FixedEffectType.CAPTAIN,
            FixedEffectType.AGENT,
            FixedEffectType.VESSEL_PERIOD,
            FixedEffectType.ROUTE_TIME,
        ],
        controls=["ice_anomaly"],
        sample_restriction="arctic_subsample",
        is_main_text=False,
        description="Separates environmental luck from execution"
    ),
    
    "R9": RegressionSpec(
        id="R9",
        name="Heterogeneous pass-through: ice × agent capability",
        dependent_var="log_q",
        fixed_effects=[
            FixedEffectType.CAPTAIN,
            FixedEffectType.AGENT,
            FixedEffectType.VESSEL_PERIOD,
            FixedEffectType.ROUTE_TIME,
        ],
        controls=["ice_anomaly", "ice_x_high_cap_agent"],
        sample_restriction="arctic_subsample",
        is_main_text=True,
        description="Organizational intermediation insulates from shocks"
    ),
    
    "R10": RegressionSpec(
        id="R10",
        name="Route choice as function of skill and capability",
        dependent_var="arctic_route",
        fixed_effects=[
            FixedEffectType.PORT_TIME,
            FixedEffectType.VESSEL_PERIOD,
        ],
        controls=["alpha_hat", "gamma_hat"],
        is_main_text=False,
        description="Tests strategic risk-taking by talent and organization"
    ),
    
    "R11": RegressionSpec(
        id="R11",
        name="Downside risk / failure outcomes",
        dependent_var="failure_indicator",
        fixed_effects=[
            FixedEffectType.ROUTE_TIME,
            FixedEffectType.VESSEL_PERIOD,
        ],
        controls=["alpha_hat", "gamma_hat"],
        is_main_text=False,
        description="Separates mean performance from tail risk"
    ),
    
    "R12": RegressionSpec(
        id="R12",
        name="Learning-by-doing: route experience",
        dependent_var="log_q",
        fixed_effects=[
            FixedEffectType.CAPTAIN,
            FixedEffectType.AGENT,
            FixedEffectType.VESSEL_PERIOD,
            FixedEffectType.ROUTE_TIME,
        ],
        controls=["route_experience"],
        is_main_text=False,
        description="Captures learning with potential organization-enabled interactions"
    ),
    
    "R13": RegressionSpec(
        id="R13",
        name="Assortative matching: do high-skill captains work with high-capability agents?",
        dependent_var="alpha_hat",
        fixed_effects=[
            FixedEffectType.PORT_TIME,
        ],
        controls=["gamma_hat_assigned"],
        is_main_text=True,
        description="Quantifies sorting; addresses selection concerns"
    ),
    
    "R14": RegressionSpec(
        id="R14",
        name="Switching hazard: what predicts captain moves between agents?",
        dependent_var="switch_next",
        fixed_effects=[
            FixedEffectType.PORT_TIME,
        ],
        controls=["alpha_hat", "gamma_hat_current", "shock_indicator"],
        is_main_text=True,
        description="Measures mobility frictions and retention power"
    ),
    
    "R15": RegressionSpec(
        id="R15",
        name="Talent acquisition advantage",
        dependent_var="next_hire_quality",
        fixed_effects=[
            FixedEffectType.PORT_TIME,
        ],
        controls=["gamma_hat"],
        is_main_text=False,
        description="Capability may operate through recruiting/screening"
    ),
    
    "R16": RegressionSpec(
        id="R16",
        name="Revenue decomposition: log revenue as outcome",
        dependent_var="log_revenue",
        fixed_effects=[
            FixedEffectType.CAPTAIN,
            FixedEffectType.AGENT,
            FixedEffectType.VESSEL_PERIOD,
            FixedEffectType.ROUTE_TIME,
        ],
        controls=["log_duration", "log_tonnage"],
        is_main_text=False,
        description="Economic significance (requires price data)"
    ),
    
    "R17": RegressionSpec(
        id="R17",
        name="Settlement netting intensity by agent",
        dependent_var="net_to_gross",
        fixed_effects=[
            FixedEffectType.AGENT,
            FixedEffectType.ROUTE_TIME,
            FixedEffectType.VESSEL_PERIOD,
        ],
        controls=["gross_proceeds_controls"],
        is_main_text=False,
        description="Governance/extraction channel (requires settlement data)"
    ),
}


# =============================================================================
# Exhibit Configuration
# =============================================================================

@dataclass
class ExhibitConfig:
    """Configuration for output exhibits."""
    
    # Main text exhibits
    main_text_regressions: List[str] = field(
        default_factory=lambda: ["R1", "R2", "R3", "R6", "R9", "R13", "R14"]
    )
    
    # Appendix regressions
    appendix_regressions: List[str] = field(
        default_factory=lambda: ["R4", "R5", "R7", "R8", "R10", "R11", "R12", "R15", "R16", "R17"]
    )
    
    # Main figures
    main_figures: List[str] = field(
        default_factory=lambda: [
            "captain_portability_oos",      # R2
            "variance_decomposition",        # R1
            "ice_passthrough_heterogeneity", # R9
            "sorting_mobility",              # R13, R14
        ]
    )
    
    # Figure settings
    figure_dpi: int = 300
    figure_format: str = "png"
    
    # Table settings
    table_format: str = "latex"  # or "csv"
    include_se: bool = True
    include_r2: bool = True
    star_significance: List[float] = field(
        default_factory=lambda: [0.10, 0.05, 0.01]
    )


DEFAULT_EXHIBITS = ExhibitConfig()


# =============================================================================
# Control Variable Definitions
# =============================================================================

CONTROL_VARIABLES = {
    "log_duration": {
        "source_col": "duration_days",
        "transform": "log",
        "fill_na": "median",
        "description": "Log voyage duration in days"
    },
    "log_tonnage": {
        "source_col": "tonnage",
        "transform": "log", 
        "fill_na": "median",
        "description": "Log vessel tonnage"
    },
    "departure_month": {
        "source_col": "departure_month",
        "transform": "categorical",
        "description": "Departure month seasonality"
    },
    "crew_size": {
        "source_col": "crew_size",
        "transform": "none",
        "fill_na": "median",
        "description": "Crew size (if available)"
    },
    "arctic_exposure": {
        "source_col": "arctic_exposure",
        "transform": "binary",
        "description": "Arctic route indicator"
    },
}


# =============================================================================
# Clustering Options
# =============================================================================

class ClusterLevel(Enum):
    """Clustering levels for standard errors."""
    CAPTAIN = "captain_id"
    VESSEL = "vessel_id"
    AGENT = "agent_id"
    ROUTE_TIME = "route_time"
    PORT = "home_port"
    TWO_WAY = "two_way"  # captain + route_time


DEFAULT_CLUSTER = ClusterLevel.CAPTAIN


# =============================================================================
# Reporting Configuration
# =============================================================================

REPORTING_REQUIREMENTS = {
    "always_report": [
        "sample_size_total",
        "sample_size_connected",
        "n_unique_captains",
        "n_unique_agents", 
        "n_unique_vessels",
        "n_unique_routes",
        "connected_set_size",
        "mobility_rates",
        "fe_absorption_strategy",
        "cluster_justification",
    ],
    "portability_section": [
        "train_test_split_year",
        "switch_only_sample_results",
        "rank_correlation_evidence",
    ],
    "selection_and_sorting": [
        "sorting_regression_R13",
        "within_captain_designs",
    ],
}
