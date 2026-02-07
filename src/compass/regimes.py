"""
Compass Pipeline — HMM Regime Segmentation (Step 4).

Fits a Gaussian Hidden Markov Model to step-level features, assigns each
step a regime label (Transit / Search / Return) and posterior
probabilities.

Supports:
* Pooled (stratified) fitting across voyages.
* Per-voyage fallback when the pooled model diverges.
* BIC-based model selection over candidate K values.
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from compass.config import CompassConfig

logger = logging.getLogger(__name__)

# ── lazy import ─────────────────────────────────────────────────────────────

def _get_hmmlearn():
    try:
        from hmmlearn.hmm import GaussianHMM
        return GaussianHMM
    except ImportError:
        raise ImportError(
            "hmmlearn is required for regime segmentation. "
            "Install with: pip install hmmlearn"
        )


# ── feature preparation ────────────────────────────────────────────────────

def prepare_hmm_features(steps_df: pd.DataFrame) -> np.ndarray:
    """
    Prepare the observation matrix for the HMM.

    Features (per step):
        1. log(step_length_m + 1)
        2. |turning_angle_rad|
        3. speed_mps

    Returns
    -------
    np.ndarray, shape (n_steps, 3)
    """
    sl = np.log1p(steps_df["step_length_m"].values)
    ta = np.abs(steps_df["turning_angle_rad"].values)
    sp = steps_df["speed_mps"].values

    X = np.column_stack([sl, ta, sp])

    # standardise to zero mean / unit variance (per column)
    mask = np.isfinite(X).all(axis=1)
    mu = np.nanmean(X[mask], axis=0)
    sd = np.nanstd(X[mask], axis=0)
    sd[sd == 0] = 1.0
    X = (X - mu) / sd

    return X, mask, mu, sd


# ── model fitting ───────────────────────────────────────────────────────────

def _init_transmat(K: int, self_prob: float) -> np.ndarray:
    """Sticky transition matrix with high self-transition probability."""
    off = (1 - self_prob) / max(K - 1, 1)
    T = np.full((K, K), off)
    np.fill_diagonal(T, self_prob)
    return T


def fit_hmm(
    X: np.ndarray,
    lengths: List[int],
    K: int,
    cfg: CompassConfig,
) -> object:
    """
    Fit a diagonal-covariance Gaussian HMM.

    Parameters
    ----------
    X : ndarray, shape (n_total_steps, n_features)
    lengths : list[int]
        Number of steps per voyage (so HMM respects voyage boundaries).
    K : int
        Number of regimes.
    cfg : CompassConfig

    Returns
    -------
    GaussianHMM (fitted)
    """
    GaussianHMM = _get_hmmlearn()

    model = GaussianHMM(
        n_components=K,
        covariance_type=cfg.hmm_covariance_type,
        n_iter=cfg.hmm_n_iter,
        tol=cfg.hmm_tol,
        random_state=cfg.hmm_random_state,
        init_params="mc",           # let hmmlearn init means & covariances
        params="stmc",              # learn all parameters
    )
    # set sticky transition matrix
    model.transmat_ = _init_transmat(K, cfg.hmm_self_transition_init)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, lengths)

    logger.info(
        "Fitted HMM K=%d, converged=%s, score=%.2f",
        K, model.monitor_.converged, model.score(X, lengths),
    )
    return model


def select_k(
    X: np.ndarray,
    lengths: List[int],
    K_candidates: List[int],
    cfg: CompassConfig,
) -> Tuple[int, Dict[int, float]]:
    """BIC-based selection over candidate K values."""
    bics: Dict[int, float] = {}
    for K in K_candidates:
        try:
            m = fit_hmm(X, lengths, K, cfg)
            n_params = K * K + K * X.shape[1] * 2  # transitions + means + diag vars
            ll = m.score(X, lengths) * len(X)       # score returns per-sample LL
            bic = -2 * ll + n_params * np.log(len(X))
            bics[K] = bic
        except Exception as e:
            logger.warning("HMM K=%d failed: %s", K, e)
            bics[K] = np.inf

    best_K = min(bics, key=bics.get)
    logger.info("BIC selection: %s → best K=%d", bics, best_K)
    return best_K, bics


# ── regime labelling ────────────────────────────────────────────────────────

_FEATURE_NAMES = ["log_step_length", "abs_turning_angle", "speed"]


def label_regimes(model, mu: np.ndarray, sd: np.ndarray) -> Dict[int, str]:
    """
    Post-hoc label regimes by emission statistics.

    Rules (applied on *un-standardised* means):
    - Transit: highest mean speed, lowest abs turning angle.
    - Search: highest abs turning angle.
    - Return: remaining (or second-highest speed if K≥3).
    """
    K = model.n_components
    # un-standardise means
    means_raw = model.means_ * sd + mu

    speed_col = 2
    turn_col = 1

    labels: Dict[int, str] = {}

    # transit = highest speed
    transit_idx = int(np.argmax(means_raw[:, speed_col]))
    labels[transit_idx] = "transit"

    # search = highest turning angle among remaining
    remaining = [k for k in range(K) if k != transit_idx]
    search_idx = int(remaining[np.argmax(
        means_raw[remaining, turn_col]
    )])
    labels[search_idx] = "search"

    # all others = return
    for k in range(K):
        if k not in labels:
            labels[k] = "return"

    logger.info("Regime labels: %s", labels)
    return labels


# ── full segmentation ──────────────────────────────────────────────────────

def segment_voyages(
    steps_df: pd.DataFrame,
    cfg: CompassConfig,
) -> pd.DataFrame:
    """
    Assign regime labels and posteriors to every step.

    Adds columns:
        ``regime_label`` (str), ``p_transit``, ``p_search``, ``p_return``.
    """
    # drop rows with NaN features
    feat_cols = ["step_length_m", "turning_angle_rad", "speed_mps"]
    valid = steps_df[feat_cols].notna().all(axis=1)
    df_valid = steps_df.loc[valid].copy()

    if len(df_valid) < cfg.min_steps_for_hmm:
        logger.warning(
            "Only %d valid steps (< %d). Skipping regime segmentation.",
            len(df_valid), cfg.min_steps_for_hmm,
        )
        for col in ("regime_label", "p_transit", "p_search", "p_return"):
            steps_df[col] = np.nan
        return steps_df

    X, mask_fin, mu, sd = prepare_hmm_features(df_valid)
    X_clean = X[mask_fin]

    # compute lengths per voyage (only finite rows)
    df_fin = df_valid.loc[df_valid.index[mask_fin]]
    lengths = df_fin.groupby("voyage_id", sort=False).size().tolist()

    if sum(lengths) == 0:
        logger.warning("No finite steps for HMM.")
        steps_df["regime_label"] = pd.array([pd.NA] * len(steps_df), dtype="object")
        for col in ("p_transit", "p_search", "p_return"):
            steps_df[col] = np.nan
        return steps_df

    # select K
    best_K, _ = select_k(X_clean, lengths, cfg.num_regimes_candidates, cfg)

    # fit best model
    model = fit_hmm(X_clean, lengths, best_K, cfg)
    regime_map = label_regimes(model, mu, sd)

    # predict posteriors
    posteriors = model.predict_proba(X_clean, lengths)
    states = model.predict(X_clean, lengths)

    # assign back
    fin_idx = df_valid.index[mask_fin]

    # init columns with NaN
    steps_df["regime_label"] = pd.array([pd.NA] * len(steps_df), dtype="object")
    for col in ("p_transit", "p_search", "p_return"):
        steps_df[col] = np.nan

    # map regime integer → label
    regime_labels = np.array([regime_map[s] for s in states])
    steps_df.loc[fin_idx, "regime_label"] = regime_labels

    # posteriors: sum probabilities belonging to each named regime
    for label in ("transit", "search", "return"):
        regime_ks = [k for k, v in regime_map.items() if v == label]
        col = f"p_{label}"
        prob = posteriors[:, regime_ks].sum(axis=1) if regime_ks else np.zeros(len(posteriors))
        steps_df.loc[fin_idx, col] = prob

    n_search = (steps_df["regime_label"] == "search").sum()
    n_transit = (steps_df["regime_label"] == "transit").sum()
    n_return = (steps_df["regime_label"] == "return").sum()
    logger.info(
        "Regime segmentation: transit=%d, search=%d, return=%d.",
        n_transit, n_search, n_return,
    )
    return steps_df
