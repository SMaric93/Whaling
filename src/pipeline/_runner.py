from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Callable, Iterable, MutableMapping, Optional


@dataclass(frozen=True)
class StepSpec:
    """Describe a pipeline step and how orchestration should report failures."""

    key: str
    func: Callable[[], Any]
    failure_message: str
    failure_level: str = "warning"
    success_message: Optional[str] = None
    exc_info: bool = False


def run_step(
    results: MutableMapping[str, Any],
    spec: StepSpec,
    *,
    logger: logging.Logger,
) -> bool:
    """Execute one step, log failures uniformly, and store successful results."""
    try:
        value = spec.func()
    except Exception as exc:
        log = getattr(logger, spec.failure_level)
        log("%s: %s", spec.failure_message, exc, exc_info=spec.exc_info)
        return False

    results[spec.key] = value
    if spec.success_message:
        logger.info(spec.success_message)
    return True


def run_steps(
    results: MutableMapping[str, Any],
    specs: Iterable[StepSpec],
    *,
    logger: logging.Logger,
) -> None:
    """Execute multiple steps in sequence, preserving pre-seeded defaults."""
    for spec in specs:
        run_step(results, spec, logger=logger)
