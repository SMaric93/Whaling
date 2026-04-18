"""Smoke tests for the repo setup script."""

from __future__ import annotations

import subprocess
from pathlib import Path


def test_setup_script_help():
    """The setup script should expose a readable help message."""
    repo_root = Path(__file__).resolve().parent.parent
    script = repo_root / "setup.sh"

    result = subprocess.run(
        ["sh", str(script), "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Creates ./venv" in result.stdout
    assert ".[all]" in result.stdout
