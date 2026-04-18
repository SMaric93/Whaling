"""Explicit state space, transition grammar, and duration priors for the HSMM.

Keeps the existing 9-state design from ``utils.py`` but adds the structured
metadata needed by the explicit-duration HSMM: duration families, support
windows, prior parameters, and a boolean transition mask.

Usage::

    from .state_space import STATE_DEFS, build_transition_mask, build_duration_prior
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch


# ---------------------------------------------------------------------------
# State definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StateDef:
    """Metadata for one voyage state."""

    id: str
    name: str
    description: str
    absorbing: bool
    duration_family: str  # "shifted_poisson", "discrete_lognormal", "absorbing"
    support_weeks: tuple[int, int]  # (min, max) inclusive
    prior_mean_weeks: float | None  # None for absorbing states


STATE_DEFS: list[StateDef] = [
    StateDef(
        id="O",
        name="outbound_initial_transit",
        description="Recently departed; outbound to initial theater.",
        absorbing=False,
        duration_family="shifted_poisson",
        support_weeks=(1, 26),
        prior_mean_weeks=4.0,
    ),
    StateDef(
        id="N",
        name="active_search_neutral",
        description="Ordinary search; neither clearly productive nor barren.",
        absorbing=False,
        duration_family="discrete_lognormal",
        support_weeks=(1, 104),
        prior_mean_weeks=16.0,
    ),
    StateDef(
        id="S",
        name="productive_search",
        description="Productive search with positive catch signals.",
        absorbing=False,
        duration_family="discrete_lognormal",
        support_weeks=(1, 104),
        prior_mean_weeks=20.0,
    ),
    StateDef(
        id="L",
        name="low_yield_or_stalled_search",
        description="Search continues but looks barren or stagnant.",
        absorbing=False,
        duration_family="discrete_lognormal",
        support_weeks=(1, 78),
        prior_mean_weeks=12.0,
    ),
    StateDef(
        id="D",
        name="distress_at_sea",
        description="Serious hazard, damage, illness, or high-risk disruption.",
        absorbing=False,
        duration_family="discrete_lognormal",
        support_weeks=(1, 26),
        prior_mean_weeks=4.0,
    ),
    StateDef(
        id="I",
        name="in_port_interruption_or_repair",
        description="In port, repairing, provisioning, or otherwise interrupted.",
        absorbing=False,
        duration_family="discrete_lognormal",
        support_weeks=(1, 52),
        prior_mean_weeks=6.0,
    ),
    StateDef(
        id="H",
        name="homebound_or_terminated",
        description="Homebound or on a premature termination path.",
        absorbing=False,
        duration_family="discrete_lognormal",
        support_weeks=(1, 52),
        prior_mean_weeks=8.0,
    ),
    StateDef(
        id="C",
        name="completed_arrival",
        description="Voyage completed; successful or ordinary completion.",
        absorbing=True,
        duration_family="absorbing",
        support_weeks=(1, 260),
        prior_mean_weeks=None,
    ),
    StateDef(
        id="F",
        name="terminal_loss",
        description="Wrecked, condemned, lost, or other terminal failure.",
        absorbing=True,
        duration_family="absorbing",
        support_weeks=(1, 260),
        prior_mean_weeks=None,
    ),
]

# Ordered state names (matches existing STATE_SPACE in utils.py)
STATE_NAMES: list[str] = [s.name for s in STATE_DEFS]
STATE_IDS: list[str] = [s.id for s in STATE_DEFS]
NUM_STATES: int = len(STATE_DEFS)
STATE_INDEX: dict[str, int] = {s.name: i for i, s in enumerate(STATE_DEFS)}
STATE_ID_INDEX: dict[str, int] = {s.id: i for i, s in enumerate(STATE_DEFS)}

# Allowed transitions — mirrors existing DEFAULT_ALLOWED_TRANSITIONS in utils.py
ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "outbound_initial_transit": {
        "outbound_initial_transit",
        "active_search_neutral",
        "productive_search",
        "low_yield_or_stalled_search",
        "in_port_interruption_or_repair",
        "terminal_loss",
    },
    "active_search_neutral": {
        "active_search_neutral",
        "productive_search",
        "low_yield_or_stalled_search",
        "distress_at_sea",
        "homebound_or_terminated",
        "completed_arrival",
    },
    "productive_search": {
        "productive_search",
        "active_search_neutral",
        "low_yield_or_stalled_search",
        "distress_at_sea",
        "homebound_or_terminated",
        "completed_arrival",
    },
    "low_yield_or_stalled_search": {
        "low_yield_or_stalled_search",
        "active_search_neutral",
        "productive_search",
        "distress_at_sea",
        "in_port_interruption_or_repair",
        "homebound_or_terminated",
    },
    "distress_at_sea": {
        "distress_at_sea",
        "in_port_interruption_or_repair",
        "homebound_or_terminated",
        "terminal_loss",
        "active_search_neutral",
    },
    "in_port_interruption_or_repair": {
        "in_port_interruption_or_repair",
        "active_search_neutral",
        "low_yield_or_stalled_search",
        "homebound_or_terminated",
        "distress_at_sea",
        "completed_arrival",
        "terminal_loss",
    },
    "homebound_or_terminated": {
        "homebound_or_terminated",
        "completed_arrival",
        "terminal_loss",
    },
    "completed_arrival": {"completed_arrival"},
    "terminal_loss": {"terminal_loss"},
}

# Initial state priors
INITIAL_STATE_PRIORS: np.ndarray = np.array(
    [0.45, 0.18, 0.10, 0.10, 0.05, 0.05, 0.04, 0.015, 0.015],
    dtype=np.float64,
)
INITIAL_STATE_PRIORS /= INITIAL_STATE_PRIORS.sum()

BAD_STATES: frozenset[str] = frozenset({
    "distress_at_sea",
    "in_port_interruption_or_repair",
    "terminal_loss",
})

BAD_STATE_INDICES: frozenset[int] = frozenset(
    STATE_INDEX[s] for s in BAD_STATES
)


# ---------------------------------------------------------------------------
# Transition mask
# ---------------------------------------------------------------------------

def build_transition_mask() -> torch.Tensor:
    """Return a boolean ``[K, K]`` mask where ``True`` = transition allowed."""
    mask = torch.zeros(NUM_STATES, NUM_STATES, dtype=torch.bool)
    for source, targets in ALLOWED_TRANSITIONS.items():
        si = STATE_INDEX[source]
        for target in targets:
            ti = STATE_INDEX[target]
            mask[si, ti] = True
    return mask


# ---------------------------------------------------------------------------
# Duration priors
# ---------------------------------------------------------------------------

def _discrete_lognormal_pmf(support: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Compute a discrete lognormal PMF over integer support values."""
    log_x = np.log(support.astype(np.float64))
    log_pdf = -(log_x - mu) ** 2 / (2 * sigma ** 2) - log_x - 0.5 * np.log(2 * np.pi * sigma ** 2)
    pdf = np.exp(log_pdf)
    pdf /= pdf.sum() + 1e-30
    return pdf


def _shifted_poisson_pmf(support: np.ndarray, mean: float) -> np.ndarray:
    """Shifted Poisson PMF: P(X=k) ∝ λ^(k-1) * exp(-λ) / (k-1)! for k≥1."""
    lam = max(mean - 1.0, 0.5)
    shifted = support - 1
    shifted = np.clip(shifted, 0, None)
    log_pmf = shifted * np.log(lam) - lam - np.array(
        [sum(np.log(np.arange(1, k + 1))) for k in shifted], dtype=np.float64
    )
    pmf = np.exp(log_pmf - log_pmf.max())
    pmf /= pmf.sum() + 1e-30
    return pmf


def build_duration_prior(state_def: StateDef, max_duration: int | None = None) -> torch.Tensor:
    """Build a duration prior PMF as a 1-D tensor of length ``U_max``.

    For absorbing states, returns a flat (uniform) distribution over the
    support — the state persists indefinitely once entered.

    Parameters
    ----------
    state_def : StateDef
        State metadata including family and prior parameters.
    max_duration : int, optional
        Override the state's ``support_weeks[1]`` cap.
    """
    lo, hi = state_def.support_weeks
    if max_duration is not None:
        hi = min(hi, max_duration)
    support = np.arange(lo, hi + 1)

    if state_def.duration_family == "absorbing":
        pmf = np.ones(len(support), dtype=np.float64)
        pmf /= pmf.sum()
    elif state_def.duration_family == "shifted_poisson":
        pmf = _shifted_poisson_pmf(support, state_def.prior_mean_weeks)
    elif state_def.duration_family == "discrete_lognormal":
        mu = np.log(max(state_def.prior_mean_weeks, 1.0))
        sigma = 0.8  # moderate dispersion
        pmf = _discrete_lognormal_pmf(support, mu, sigma)
    else:
        raise ValueError(f"Unknown duration family: {state_def.duration_family}")

    return torch.tensor(pmf, dtype=torch.float64)


def build_all_duration_priors(max_duration: int | None = None) -> list[torch.Tensor]:
    """Build duration priors for all states, returning a list indexed by state."""
    return [build_duration_prior(sd, max_duration=max_duration) for sd in STATE_DEFS]


def get_max_duration() -> int:
    """Return the global maximum duration across all states."""
    return max(sd.support_weeks[1] for sd in STATE_DEFS)
