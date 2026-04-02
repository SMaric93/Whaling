from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_features_package_exports_core_route_api() -> None:
    from src import features

    for name in [
        "compute_all_route_features",
        "compute_route_features",
        "haversine_distance",
        "haversine_vectorized",
        "load_logbook_positions",
        "save_route_features",
        "get_feature_summary",
        "VoyageRouteFeatures",
    ]:
        assert hasattr(features, name)
        assert name in features.__all__
