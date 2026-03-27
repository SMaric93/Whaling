"""
ML Layer for Whaling Project.

Extends the econometric backbone with interpretable machine learning:
- Policy learning (map vs compass)
- Latent search states
- Nonlinear survival models
- Heterogeneous / distributional effects
- Production surface estimation
- Assignment optimization
- Appendix extensions (embeddings, changepoints, NLP, etc.)

All modules import lazily to avoid heavy startup costs.
"""

from importlib import import_module as _im

__all__ = [
    # Infrastructure
    "config",
    "splits",
    "metrics",
    "interpret",
    "calibration",
    "baselines",
    "record_matching",
    "text_models",
    "anomaly_detection",
    # Dataset builders
    "build_action_dataset",
    "build_state_dataset",
    "build_survival_dataset",
    "build_outcome_ml_dataset",
    "build_network_dataset",
    "build_text_dataset",
    # Core ML modules
    "policy_learning",
    "state_models",
    "survival_ml",
    "heterogeneity_ml",
    "production_surface_ml",
    "assignment_optimizer",
    # Appendix modules
    "trajectory_embeddings",
    "changepoints",
    "network_imprinting",
    "text_nlp",
    "spatial_quality",
    "conformal_risk",
    "off_policy_eval",
]


def __getattr__(name: str):
    """Lazy-load submodules on first access."""
    if name in __all__:
        return _im(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
