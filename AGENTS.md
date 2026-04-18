# AGENTS.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Structure

This is a 5-stage empirical research pipeline for 19th-century American whaling productivity analysis. The codebase implements an AKM/KSS variance decomposition, econometric regression specifications, and an ML layer for policy learning and heterogeneous effects estimation.

- `src/` - All application code
  - `src/pipeline/` - 5-stage ETL/analysis orchestration
  - `src/analyses/` - Econometric regression modules (R1-R17)
  - `src/reinforcement/` - Reinforcement test suite (stopping rules, data builder)
  - `src/ml/` - Machine learning layer (policy, states, survival, heterogeneity)
  - `src/paper/` - Data-driven manuscript table, figure, and appendix builders
  - `src/utils/` - Shared utilities (caching, IO, regression helpers)
  - `src/download/` - AOWV data downloader
  - `src/config.py` - Global configuration
- `tests/` - Test files using pytest
- `data/` - Raw, staging, derived, and final parquet datasets
- `output/` - Pipeline outputs (tables, figures, memos)
- `scripts/` - Standalone utility scripts

## Common Commands

### Running the Pipeline

```bash
# Run all 5 stages
python -m src.pipeline.runner

# Run a specific stage (1=download, 2=parse, 3=merge, 4=analyze, 5=output)
python -m src.pipeline.runner --stage 4
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run a specific test file
pytest tests/test_main.py

# Run with verbose output
pytest tests/ -v
```

### Installing Dependencies

```bash
pip install -e .                    # Core dependencies
pip install -e ".[dev]"             # + pytest
pip install -e ".[wsl]"            # + WSL document processing
pip install -e ".[all]"            # Everything
```

## Code Architecture

- The pipeline flows: Download → Parse → Merge (entity resolution, AKM) → Analyze → Output
- AKM captain/agent effects (theta/psi) are merged in `src/reinforcement/data_builder.py`
- The `src/paper/` layer generates all manuscript tables dynamically from data
- Fixed effects absorption uses sparse LSQR via `src/reinforcement/utils.py`
- Clustered standard errors use the sandwich estimator with HC1 correction
- Apple Silicon MPS support is handled via `src/torch_device.py`
