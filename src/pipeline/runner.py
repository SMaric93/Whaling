"""
Pipeline Runner

Orchestrates the complete 5-stage data pipeline.

Stages:
    1. PULL   - Download all raw data sources
    2. CLEAN  - Parse and standardize data
    3. MERGE  - Assemble and link datasets
    4. ANALYZE - Run full analysis suite
    5. OUTPUT - Generate MD and TEX outputs
"""

import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def run_full_pipeline(
    skip_pull: bool = False,
    skip_clean: bool = False,
    skip_merge: bool = False,
    skip_analyze: bool = False,
    skip_output: bool = False,
    force_pull: bool = False,
    quick_analyze: bool = False,
) -> dict:
    """
    Run the complete 5-stage pipeline.
    
    Args:
        skip_pull: Skip Stage 1 (data download)
        skip_clean: Skip Stage 2 (parsing)
        skip_merge: Skip Stage 3 (assembly)
        skip_analyze: Skip Stage 4 (analysis)
        skip_output: Skip Stage 5 (output generation)
        force_pull: Force re-download of all data
        quick_analyze: Run only main regressions
    
    Returns:
        dict: Results from all stages
    """
    start_time = time.time()
    
    logger.info("=" * 70)
    logger.info("WHALING DATA PIPELINE")
    logger.info("Venture Capital of the Sea — Full Analysis Suite")
    logger.info("=" * 70)
    
    results = {
        'stage1_pull': None,
        'stage2_clean': None,
        'stage3_merge': None,
        'stage4_analyze': None,
        'stage5_output': None,
    }
    
    # Stage 1: Data Pull
    if not skip_pull:
        from .stage1_pull import run_pull
        logger.info("\n" + "=" * 70)
        results['stage1_pull'] = run_pull(force=force_pull)
    else:
        logger.info("Skipping Stage 1: Data Pull")
    
    # Stage 2: Data Cleaning
    if not skip_clean:
        from .stage2_clean import run_clean
        logger.info("\n" + "=" * 70)
        results['stage2_clean'] = run_clean()
    else:
        logger.info("Skipping Stage 2: Data Cleaning")
    
    # Stage 3: Data Merging
    if not skip_merge:
        from .stage3_merge import run_merge
        logger.info("\n" + "=" * 70)
        results['stage3_merge'] = run_merge()
    else:
        logger.info("Skipping Stage 3: Data Merging")
    
    # Stage 4: Full Analyses
    if not skip_analyze:
        from .stage4_analyze import run_analyze
        logger.info("\n" + "=" * 70)
        results['stage4_analyze'] = run_analyze(quick=quick_analyze)
    else:
        logger.info("Skipping Stage 4: Full Analyses")
    
    # Stage 5: Write Output
    if not skip_output:
        from .stage5_output import run_output
        logger.info("\n" + "=" * 70)
        results['stage5_output'] = run_output()
    else:
        logger.info("Skipping Stage 5: Write Output")
    
    # Final summary
    elapsed = time.time() - start_time
    
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    
    # Count successes
    stages_run = sum(1 for v in results.values() if v is not None)
    logger.info(f"Stages completed: {stages_run}/5")
    
    return results


def run_from_stage(start_stage: int, **kwargs) -> dict:
    """
    Run pipeline starting from a specific stage.
    
    Args:
        start_stage: Stage number to start from (1-5)
        **kwargs: Additional arguments passed to run_full_pipeline
    
    Returns:
        dict: Results from executed stages
    """
    skip_flags = {
        'skip_pull': start_stage > 1,
        'skip_clean': start_stage > 2,
        'skip_merge': start_stage > 3,
        'skip_analyze': start_stage > 4,
        'skip_output': start_stage > 5,
    }
    
    return run_full_pipeline(**skip_flags, **kwargs)


def run_single_stage(stage: int, **kwargs) -> dict:
    """
    Run only a single stage.
    
    Args:
        stage: Stage number to run (1-5)
        **kwargs: Stage-specific arguments
    
    Returns:
        dict: Results from the stage
    """
    logger.info(f"Running single stage: {stage}")
    
    if stage == 1:
        from .stage1_pull import run_pull
        return run_pull(**kwargs)
    elif stage == 2:
        from .stage2_clean import run_clean
        return run_clean()
    elif stage == 3:
        from .stage3_merge import run_merge
        return run_merge()
    elif stage == 4:
        from .stage4_analyze import run_analyze
        return run_analyze(**kwargs)
    elif stage == 5:
        from .stage5_output import run_output
        return run_output(**kwargs)
    else:
        raise ValueError(f"Invalid stage number: {stage}. Must be 1-5.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    run_full_pipeline()
