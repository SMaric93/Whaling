"""Paper package orchestration layer."""

from __future__ import annotations

import os
import tempfile
import warnings
from pathlib import Path


os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "whaling-mpl"))
warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores for the following reason:",
    module=r"joblib\\.externals\\.loky\\.backend\\.context",
)
