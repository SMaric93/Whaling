"""
Logbook Feature Engineering Package.

Provides voyage-level features computed from daily logbook position data:
- Route metrics (efficiency, distance, dwell time)
- Storm exposure (HURDAT2 intersection)
- Information networks (Maury SpokenVessels)
- Agent strategy metrics
"""

from importlib import import_module

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


def _export_optional(module_name: str, names: tuple[str, ...]) -> None:
    try:
        module = import_module(f".{module_name}", __name__)
    except ImportError:
        return

    globals().update({name: getattr(module, name) for name in names})
    __all__.extend(names)


_export_optional(
    "storm_exposure",
    (
        "compute_all_storm_exposure",
        "compute_voyage_storm_exposure",
        "VoyageStormExposure",
    ),
)
_export_optional(
    "information_network",
    (
        "compute_voyage_network_features",
        "load_maury_data",
        "parse_spoken_vessels",
        "VoyageNetworkFeatures",
    ),
)
_export_optional(
    "agent_strategy",
    (
        "compute_agent_portfolio_metrics",
        "compute_route_assignment_quality",
        "AgentStrategyMetrics",
    ),
)
