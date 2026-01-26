# AGENTS.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Structure

This is a simple Python greeting application with a clean separation between source code and tests:

- `src/` - Application code
- `tests/` - Test files using pytest

The test files manually add `src/` to the Python path since there's no package setup (no `setup.py` or `pyproject.toml`).

## Common Commands

### Running the Application
```bash
python src/main.py
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
pip install -r requirements.txt
```

## Code Architecture

The application has a simple modular design:
- Pure functions (like `greet()`) are separated from I/O operations (like `main()`)
- All application logic is testable without running the CLI interface
- Tests import functions directly by manipulating `sys.path` rather than using package imports
