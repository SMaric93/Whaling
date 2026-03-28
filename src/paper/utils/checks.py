from __future__ import annotations

from pathlib import Path


def assert_outputs_exist(paths: list[Path]) -> list[str]:
    missing = [str(p) for p in paths if not p.exists()]
    return missing
