"""
Compass Pipeline — Micro-Routing / Search Policy Measurement.

Converts raw vessel trajectories into regime-labeled step datasets,
computes interpretable search-policy features, constructs PCA-based
compass indices, and exports panel-ready datasets for causal/movers
econometric designs.
"""

from compass.config import CompassConfig, load_config
from compass.cli import run_compass_pipeline

__all__ = [
    "CompassConfig",
    "load_config",
    "run_compass_pipeline",
]
