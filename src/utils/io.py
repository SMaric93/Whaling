"""
File I/O utilities.

Helper functions for common file operations with consistent
error handling and logging.
"""

from pathlib import Path
from typing import Optional
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def ensure_dir(path: Path) -> Path:
    """
    Ensure directory exists, creating if necessary.
    
    Parameters
    ----------
    path : Path
        Directory path to ensure exists.
        
    Returns
    -------
    Path
        The input path (for chaining).
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_parquet(
    df: pd.DataFrame,
    path: Path,
    compression: str = "snappy",
) -> Path:
    """
    Save DataFrame to parquet with logging.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    path : Path
        Output path.
    compression : str
        Compression algorithm.
        
    Returns
    -------
    Path
        The output path.
    """
    ensure_dir(path.parent)
    df.to_parquet(path, compression=compression, index=False)
    logger.info(f"Saved {len(df):,} rows to {path.name}")
    return path


def load_parquet(
    path: Path,
    columns: Optional[list] = None,
) -> pd.DataFrame:
    """
    Load parquet file with optional column selection.
    
    Parameters
    ----------
    path : Path
        Path to parquet file.
    columns : list, optional
        Columns to load (loads all if None).
        
    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.
        
    Raises
    ------
    FileNotFoundError
        If file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    
    df = pd.read_parquet(path, columns=columns)
    logger.debug(f"Loaded {len(df):,} rows from {path.name}")
    return df
