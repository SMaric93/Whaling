"""
Data caching utilities for repeated DataFrame loads.

Provides LRU caching for expensive data loading operations
to avoid redundant disk I/O and parsing.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional, Callable, Any
import hashlib
import pandas as pd


# Module-level cache storage
_dataframe_cache: dict = {}


def _hash_config(config: Any) -> str:
    """Create a hash of configuration object for cache key."""
    if config is None:
        return "none"
    # Use repr for simple hashable representation
    return hashlib.md5(repr(config).encode()).hexdigest()[:8]


def cached_dataframe(
    path: Path,
    loader_func: Callable[..., pd.DataFrame],
    config: Optional[Any] = None,
    force_reload: bool = False,
) -> pd.DataFrame:
    """
    Load a DataFrame with caching.
    
    Parameters
    ----------
    path : Path
        Path to the data file (used as cache key).
    loader_func : Callable
        Function that loads and returns the DataFrame.
    config : Any, optional
        Configuration object (included in cache key hash).
    force_reload : bool
        If True, bypass cache and reload.
        
    Returns
    -------
    pd.DataFrame
        Loaded (possibly cached) DataFrame.
    """
    cache_key = f"{path}:{_hash_config(config)}"
    
    if force_reload or cache_key not in _dataframe_cache:
        df = loader_func()
        _dataframe_cache[cache_key] = df
    
    return _dataframe_cache[cache_key].copy()


def clear_cache() -> None:
    """Clear all cached DataFrames."""
    global _dataframe_cache
    _dataframe_cache.clear()


@lru_cache(maxsize=8)
def _load_parquet_cached(path_str: str) -> pd.DataFrame:
    """LRU-cached parquet loader (internal)."""
    return pd.read_parquet(path_str)


def load_parquet_cached(path: Path) -> pd.DataFrame:
    """
    Load a parquet file with LRU caching.
    
    Parameters
    ----------
    path : Path
        Path to parquet file.
        
    Returns
    -------
    pd.DataFrame
        Loaded DataFrame (copy to prevent mutation).
    """
    return _load_parquet_cached(str(path)).copy()
