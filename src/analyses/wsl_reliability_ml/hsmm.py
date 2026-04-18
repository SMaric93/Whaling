"""Explicit-duration Hidden Semi-Markov Model (HSMM) in PyTorch.

Implements log-space forward/backward recursions and Viterbi decoding
with state-specific duration distributions, masked transitions, and
quality-weight downweighting.

The recursion follows the spec:
    segment_score  E_s(t,u) = sum_{k=t-u+1}^{t} B[k,s]
    forward_log    α_t(s)   = logsumexp_{u,r} [α_{t-u}(r) + log A_{r,s} + log D_s(u) + E_s(t,u)]
    viterbi        Replace logsumexp with max and store backpointers.

Usage::

    from .hsmm import ExplicitDurationHSMM
    model = ExplicitDurationHSMM(...)
    model.fit(sequences)
    posteriors = model.posterior_decode(sequence)
    path = model.viterbi_decode(sequence)
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .state_space import (
    ALLOWED_TRANSITIONS,
    INITIAL_STATE_PRIORS,
    NUM_STATES,
    STATE_DEFS,
    STATE_INDEX,
    STATE_NAMES,
    build_all_duration_priors,
    build_transition_mask,
    get_max_duration,
)

logger = logging.getLogger(__name__)

_NEG_INF = -1e18


class ExplicitDurationHSMM(nn.Module):
    """Explicit-duration HSMM with masked transitions, log-space DP."""

    def __init__(
        self,
        *,
        max_duration: int | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.K = NUM_STATES
        self.max_duration = max_duration or min(get_max_duration(), 104)  # cap for memory
        self.device = torch.device(device)

        # Transition mask (boolean)
        self.register_buffer("transition_mask", build_transition_mask().to(self.device))

        # Log transition probs (initialized uniform over allowed transitions)
        init_trans = torch.zeros(self.K, self.K, dtype=torch.float64, device=self.device)
        mask = self.transition_mask
        for i in range(self.K):
            n_allowed = mask[i].sum().item()
            if n_allowed > 0:
                init_trans[i, mask[i]] = 1.0 / n_allowed
        self.log_A = nn.Parameter(
            torch.log(init_trans.clamp(min=1e-30)),
            requires_grad=False,
        )

        # Duration log-probs [K, U_max]
        duration_priors = build_all_duration_priors(max_duration=self.max_duration)
        # Pad all to same length
        dur_matrix = torch.full(
            (self.K, self.max_duration),
            _NEG_INF,
            dtype=torch.float64,
            device=self.device,
        )
        for s, prior in enumerate(duration_priors):
            lo = STATE_DEFS[s].support_weeks[0]
            length = min(len(prior), self.max_duration - lo + 1)
            if length > 0:
                probs = prior[:length].to(torch.float64)
                probs = probs / probs.sum().clamp(min=1e-30)
                dur_matrix[s, lo - 1 : lo - 1 + length] = torch.log(probs.clamp(min=1e-30))
        self.log_D = nn.Parameter(dur_matrix, requires_grad=False)

        # Initial state log-probs
        init_probs = torch.tensor(INITIAL_STATE_PRIORS, dtype=torch.float64, device=self.device)
        self.log_pi = nn.Parameter(
            torch.log(init_probs.clamp(min=1e-30)),
            requires_grad=False,
        )

    # -----------------------------------------------------------------
    # Segment scores (cached cumulative emission sums)
    # -----------------------------------------------------------------

    def _compute_segment_scores(
        self,
        emission_logprobs: torch.Tensor,
        quality_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute segment scores E_s(t, u) for all (t, s, u).

        E_s(t, u) = sum_{k=t-u+1}^{t} w[k] * B[k, s]

        Returns tensor of shape [T, K, U_max].
        """
        T, K = emission_logprobs.shape
        U = self.max_duration

        # Weight emissions
        if quality_weights is not None:
            weighted = emission_logprobs * quality_weights.unsqueeze(1)
        else:
            weighted = emission_logprobs

        # Cumulative sum for efficient segment computation
        # prepend a zero row for offset indexing
        cum = torch.zeros(T + 1, K, dtype=torch.float64, device=self.device)
        cum[1:] = torch.cumsum(weighted, dim=0)

        # E_s(t, u) = cum[t+1, s] - cum[t+1-u, s]   for u = 1..min(t+1, U)
        seg = torch.full((T, K, U), _NEG_INF, dtype=torch.float64, device=self.device)
        for t in range(T):
            max_u = min(t + 1, U)
            for u_idx in range(max_u):
                u = u_idx + 1  # duration 1-indexed
                start = t + 1 - u
                seg[t, :, u_idx] = cum[t + 1] - cum[start]

        return seg

    # -----------------------------------------------------------------
    # Forward algorithm
    # -----------------------------------------------------------------

    def forward_logprob(
        self,
        emission_logprobs: torch.Tensor,
        quality_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass in log space.

        Parameters
        ----------
        emission_logprobs : Tensor [T, K]
            Per-state log emission scores for each time step.
        quality_weights : Tensor [T], optional
            Per-step quality weights (0–1 scale, multiplicative on emissions).

        Returns
        -------
        alpha : Tensor [T, K]
            Forward log-probabilities.
        """
        T, K = emission_logprobs.shape
        U = self.max_duration
        seg = self._compute_segment_scores(emission_logprobs, quality_weights)

        alpha = torch.full((T, K), _NEG_INF, dtype=torch.float64, device=self.device)

        # Initialization: α_0(s) = log π(s) + log D_s(1) + B[0,s]
        for s in range(K):
            alpha[0, s] = self.log_pi[s] + self.log_D[s, 0] + emission_logprobs[0, s]

        # Recursion
        for t in range(1, T):
            max_u = min(t + 1, U)
            for s in range(K):
                candidates = []
                for u_idx in range(max_u):
                    u = u_idx + 1
                    if self.log_D[s, u_idx] <= _NEG_INF + 1:
                        continue  # duration not supported
                    entry_time = t - u
                    if entry_time < 0:
                        # Segment starts before sequence — use initial prob
                        init_score = (
                            self.log_pi[s]
                            + self.log_D[s, u_idx]
                            + seg[t, s, u_idx]
                        )
                        candidates.append(init_score)
                    else:
                        # Transition from all allowed predecessors
                        for r in range(K):
                            if not self.transition_mask[r, s]:
                                continue
                            score = (
                                alpha[entry_time, r]
                                + self.log_A[r, s]
                                + self.log_D[s, u_idx]
                                + seg[t, s, u_idx]
                            )
                            candidates.append(score)

                if candidates:
                    alpha[t, s] = torch.logsumexp(
                        torch.stack(candidates), dim=0
                    )

        return alpha

    # -----------------------------------------------------------------
    # Backward algorithm
    # -----------------------------------------------------------------

    def backward_logprob(
        self,
        emission_logprobs: torch.Tensor,
        quality_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Backward pass in log space.

        Returns
        -------
        beta : Tensor [T, K]
            Backward log-probabilities.
        """
        T, K = emission_logprobs.shape
        U = self.max_duration
        seg = self._compute_segment_scores(emission_logprobs, quality_weights)

        beta = torch.full((T, K), _NEG_INF, dtype=torch.float64, device=self.device)
        beta[T - 1, :] = 0.0  # log(1)

        for t in range(T - 2, -1, -1):
            for r in range(K):
                candidates = []
                max_u = min(T - t - 1, U)
                for s in range(K):
                    if not self.transition_mask[r, s]:
                        continue
                    for u_idx in range(max_u):
                        u = u_idx + 1
                        end_time = t + u
                        if end_time >= T:
                            break
                        if self.log_D[s, u_idx] <= _NEG_INF + 1:
                            continue
                        score = (
                            self.log_A[r, s]
                            + self.log_D[s, u_idx]
                            + seg[end_time, s, u_idx]
                            + beta[end_time, s]
                        )
                        candidates.append(score)

                if candidates:
                    beta[t, r] = torch.logsumexp(
                        torch.stack(candidates), dim=0
                    )

        return beta

    # -----------------------------------------------------------------
    # Posterior decoding
    # -----------------------------------------------------------------

    def posterior_decode(
        self,
        emission_logprobs: torch.Tensor,
        quality_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute posterior state probabilities via forward-backward.

        Returns
        -------
        posterior : Tensor [T, K]
            Normalized posterior probabilities.
        """
        alpha = self.forward_logprob(emission_logprobs, quality_weights)
        beta = self.backward_logprob(emission_logprobs, quality_weights)

        log_gamma = alpha + beta
        # Normalize per time step
        log_norm = torch.logsumexp(log_gamma, dim=1, keepdim=True)
        posterior = torch.exp(log_gamma - log_norm)

        return posterior

    # -----------------------------------------------------------------
    # Viterbi decoding
    # -----------------------------------------------------------------

    def viterbi_decode(
        self,
        emission_logprobs: torch.Tensor,
        quality_weights: torch.Tensor | None = None,
    ) -> np.ndarray:
        """Viterbi decoding: find the most likely state sequence.

        Returns
        -------
        path : ndarray of int, shape [T]
            Most likely state index at each time step.
        """
        T, K = emission_logprobs.shape
        U = self.max_duration
        seg = self._compute_segment_scores(emission_logprobs, quality_weights)

        delta = torch.full((T, K), _NEG_INF, dtype=torch.float64, device=self.device)
        # Backpointers: (prev_state, duration)
        bp_state = torch.zeros(T, K, dtype=torch.long, device=self.device)
        bp_dur = torch.ones(T, K, dtype=torch.long, device=self.device)

        # Initialization
        for s in range(K):
            delta[0, s] = self.log_pi[s] + self.log_D[s, 0] + emission_logprobs[0, s]

        # Recursion
        for t in range(1, T):
            max_u = min(t + 1, U)
            for s in range(K):
                best_score = _NEG_INF
                best_r = 0
                best_u = 1
                for u_idx in range(max_u):
                    u = u_idx + 1
                    if self.log_D[s, u_idx] <= _NEG_INF + 1:
                        continue
                    entry_time = t - u
                    if entry_time < 0:
                        score = (
                            self.log_pi[s]
                            + self.log_D[s, u_idx]
                            + seg[t, s, u_idx]
                        )
                        if score > best_score:
                            best_score = score.item()
                            best_r = s
                            best_u = u
                    else:
                        for r in range(K):
                            if not self.transition_mask[r, s]:
                                continue
                            score = (
                                delta[entry_time, r]
                                + self.log_A[r, s]
                                + self.log_D[s, u_idx]
                                + seg[t, s, u_idx]
                            )
                            if score > best_score:
                                best_score = score.item()
                                best_r = r
                                best_u = u

                delta[t, s] = best_score
                bp_state[t, s] = best_r
                bp_dur[t, s] = best_u

        # Backtrace
        path = np.zeros(T, dtype=np.int64)
        # Best final state
        path[T - 1] = int(delta[T - 1].argmax().item())

        t = T - 1
        while t > 0:
            s = int(path[t])
            u = int(bp_dur[t, s].item())
            r = int(bp_state[t, s].item())
            # Fill segment with state s
            start = max(0, t - u + 1)
            path[start : t + 1] = s
            # Jump back
            t = start - 1
            if t >= 0:
                path[t] = r

        return path

    # -----------------------------------------------------------------
    # M-step updates
    # -----------------------------------------------------------------

    def update_transitions(
        self,
        expected_transitions: torch.Tensor,
        smoothing: float = 1.0,
    ) -> None:
        """Update transition parameters from expected transition counts.

        Parameters
        ----------
        expected_transitions : Tensor [K, K]
            Expected number of transitions between state pairs.
        smoothing : float
            Additive smoothing for allowed transitions.
        """
        counts = expected_transitions.clone()
        # Add smoothing only to allowed transitions
        counts += smoothing * self.transition_mask.float()
        # Zero out forbidden transitions
        counts *= self.transition_mask.float()
        # Normalize
        row_sums = counts.sum(dim=1, keepdim=True).clamp(min=1e-30)
        trans = counts / row_sums
        self.log_A.data = torch.log(trans.clamp(min=1e-30))

    def update_durations(
        self,
        expected_durations: list[torch.Tensor],
        smoothing: float = 0.01,
    ) -> None:
        """Update duration parameters from expected duration counts.

        Parameters
        ----------
        expected_durations : list of Tensor, one per state
            Expected duration count histograms.
        smoothing : float
            Additive smoothing.
        """
        for s in range(self.K):
            counts = expected_durations[s].clone()
            # Only update within supported range
            lo = STATE_DEFS[s].support_weeks[0] - 1
            hi = min(STATE_DEFS[s].support_weeks[1], self.max_duration)
            # Smooth
            counts[lo:hi] += smoothing
            # Zero outside support
            counts[:lo] = 0.0
            if hi < self.max_duration:
                counts[hi:] = 0.0
            total = counts.sum().clamp(min=1e-30)
            self.log_D.data[s] = torch.log((counts / total).clamp(min=1e-30))

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save model parameters to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "max_duration": self.max_duration,
                "K": self.K,
            },
            path,
        )
        logger.info("[hsmm] Saved model to %s", path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "ExplicitDurationHSMM":
        """Load model parameters from disk."""
        path = Path(path)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls(max_duration=checkpoint["max_duration"], device=device)
        model.load_state_dict(checkpoint["state_dict"])
        logger.info("[hsmm] Loaded model from %s", path)
        return model


def prepare_sequence(
    emission_probs: np.ndarray,
    quality_weights: np.ndarray | None = None,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Convert numpy arrays to the tensor format expected by the HSMM.

    Parameters
    ----------
    emission_probs : ndarray [T, K]
        State emission probabilities (not log).
    quality_weights : ndarray [T], optional
        Per-step quality weights.
    device : str
        PyTorch device.

    Returns
    -------
    dict with 'emission_logprobs' and optionally 'quality_weights'.
    """
    emission_probs = np.clip(emission_probs, 1e-12, 1.0)
    result = {
        "emission_logprobs": torch.tensor(
            np.log(emission_probs), dtype=torch.float64, device=device
        ),
    }
    if quality_weights is not None:
        result["quality_weights"] = torch.tensor(
            quality_weights, dtype=torch.float64, device=device
        )
    return result
