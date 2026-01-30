"""
Whaling Data Pipeline - 5-Stage Architecture

The pipeline is organized into 5 sequential stages:
    1. PULL   - Download all raw data sources
    2. CLEAN  - Parse and standardize data
    3. MERGE  - Assemble and link datasets
    4. ANALYZE - Run full analysis suite
    5. OUTPUT - Generate MD and TEX outputs

Usage:
    from src.pipeline import run_full_pipeline
    run_full_pipeline()

Or run individual stages:
    from src.pipeline import run_pull, run_clean, run_merge, run_analyze, run_output
    run_pull()
    run_clean()
    run_merge()
    run_analyze()
    run_output()
"""

from .stage1_pull import run_pull
from .stage2_clean import run_clean
from .stage3_merge import run_merge
from .stage4_analyze import run_analyze
from .stage5_output import run_output
from .runner import run_full_pipeline

__all__ = [
    'run_pull',
    'run_clean',
    'run_merge',
    'run_analyze',
    'run_output',
    'run_full_pipeline',
]
