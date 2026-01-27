"""
Tests for the Whaling data pipeline.

This package contains unit tests, integration tests,
and validation checks for all pipeline components.
"""

import pytest


def test_package_imports():
    """Test that the main package can be imported."""
    import src
    assert hasattr(src, "__version__")
    assert src.__version__ >= "0.22"


def test_config_imports():
    """Test that configuration is accessible."""
    from src import PROJECT_ROOT, RAW_DIR, STAGING_DIR, FINAL_DIR
    assert PROJECT_ROOT.exists()


def test_parsing_imports():
    """Test that parsing utilities are accessible."""
    from src import normalize_name, jaro_winkler_similarity
    
    # Test normalize_name
    result = normalize_name("Capt. Wm. Smith Jr.")
    assert "SMITH" in result
    
    # Test similarity
    score = jaro_winkler_similarity("SMITH", "SMYTH")
    assert 0 < score <= 1


def test_entity_resolver_import():
    """Test that EntityResolver is accessible."""
    from src import EntityResolver
    assert EntityResolver is not None


def test_utils_imports():
    """Test that utils module is accessible."""
    from src.utils import cluster_robust_se, build_fe_design_matrix
    from src.utils import ensure_dir, save_parquet, load_parquet
    from src.utils import cached_dataframe, clear_cache
    
    # All functions should be callable
    assert callable(cluster_robust_se)
    assert callable(build_fe_design_matrix)
    assert callable(ensure_dir)
