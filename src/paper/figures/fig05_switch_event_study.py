from __future__ import annotations

from ..config import BuildContext
from .real_builders import build_figure


def build(context: BuildContext):
    return build_figure("fig05_switch_event_study", context)
