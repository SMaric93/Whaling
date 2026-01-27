"""
Logbook Feature Engineering Package.

Provides voyage-level features computed from daily logbook position data:
- Route metrics (efficiency, distance, dwell time)
- Storm exposure (HURDAT2 intersection)
- Information networks (Maury SpokenVessels)
- Agent strategy metrics
"""

# Core route features (always available)
from .logbook_features import (
    compute_all_route_features,
    compute_route_features,
    haversine_distance,
    haversine_vectorized,
    load_logbook_positions,
    save_route_features,
    get_feature_summary,
    VoyageRouteFeatures,
)

__all__ = [
    # Route features
    "compute_all_route_features",
    "compute_route_features",
    "haversine_distance",
    "haversine_vectorized",
    "load_logbook_positions",
    "save_route_features",
    "get_feature_summary",
    "VoyageRouteFeatures",
]

# Optional: Storm exposure (requires weather_downloader dependencies)
try:
    from .storm_exposure import (
        compute_all_storm_exposure,
        compute_voyage_storm_exposure,
        VoyageStormExposure,
    )
    __all__.extend([
        "compute_all_storm_exposure",
        "compute_voyage_storm_exposure",
        "VoyageStormExposure",
    ])
except ImportError:
    pass

# Optional: Information network
try:
    from .information_network import (
        compute_voyage_network_features,
        load_maury_data,
        parse_spoken_vessels,
        VoyageNetworkFeatures,
    )
    __all__.extend([
        "compute_voyage_network_features",
        "load_maury_data",
        "parse_spoken_vessels",
        "VoyageNetworkFeatures",
    ])
except ImportError:
    pass

# Optional: Agent strategy
try:
    from .agent_strategy import (
        compute_agent_portfolio_metrics,
        compute_route_assignment_quality,
        AgentStrategyMetrics,
    )
    __all__.extend([
        "compute_agent_portfolio_metrics",
        "compute_route_assignment_quality",
        "AgentStrategyMetrics",
    ])
except ImportError:
    pass
