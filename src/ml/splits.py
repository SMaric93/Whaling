"""
ML Layer — Train / Validation / Test Split Utilities.

Five split strategies, all respecting unit integrity
(whole voyages stay together, no leakage).

Split A: Rolling time split (60/20/20 by year)
Split B: Group holdout by captain
Split C: Group holdout by agent
Split D: Switch-event split
Split E: Spatial holdout by ground(-year)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.ml.config import ML_CFG

logger = logging.getLogger(__name__)

# Return type for all split functions
SplitResult = Tuple[np.ndarray, np.ndarray, np.ndarray]  # train, val, test


# ═══════════════════════════════════════════════════════════════════════════
# Split A: Rolling Time
# ═══════════════════════════════════════════════════════════════════════════

def split_rolling_time(
    df: pd.DataFrame,
    *,
    year_col: str = None,
    voyage_col: str = "voyage_id",
    train_frac: float = None,
    val_frac: float = None,
) -> SplitResult:
    """
    Rolling time split preserving whole voyages.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain year_col and voyage_col.
    year_col : str
        Column with the year to sort on. Auto-detected if None.
    voyage_col : str
        Column identifying voyages (kept together).
    train_frac, val_frac : float
        Override default fractions from config.

    Returns
    -------
    (train_idx, val_idx, test_idx) : arrays of integer index positions.
    """
    train_frac = train_frac or ML_CFG.train_frac
    val_frac = val_frac or ML_CFG.val_frac

    # Auto-detect year column
    if year_col is None:
        for candidate in ["year_out", "year", "start_year"]:
            if candidate in df.columns:
                year_col = candidate
                break
        if year_col is None:
            # Fall back to extracting year from a date column
            for date_candidate in ["obs_date", "date", "departure_date"]:
                if date_candidate in df.columns:
                    df = df.copy()
                    df["_year_auto"] = pd.to_datetime(df[date_candidate]).dt.year
                    year_col = "_year_auto"
                    break
        if year_col is None:
            raise KeyError("No year column found. Tried: year_out, year, start_year, obs_date, date")

    # Get unique voyages sorted by year
    voyage_years = (
        df.groupby(voyage_col)[year_col]
        .min()
        .sort_values()
        .reset_index()
    )
    n = len(voyage_years)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_voyages = set(voyage_years[voyage_col].iloc[:n_train])
    val_voyages = set(voyage_years[voyage_col].iloc[n_train:n_train + n_val])
    test_voyages = set(voyage_years[voyage_col].iloc[n_train + n_val:])

    train_idx = np.where(df[voyage_col].isin(train_voyages))[0]
    val_idx = np.where(df[voyage_col].isin(val_voyages))[0]
    test_idx = np.where(df[voyage_col].isin(test_voyages))[0]

    _log_split("A (rolling time)", len(train_idx), len(val_idx), len(test_idx))
    _check_no_leakage(df, train_idx, val_idx, test_idx, voyage_col)
    return train_idx, val_idx, test_idx


# ═══════════════════════════════════════════════════════════════════════════
# Split B: Group Holdout by Captain
# ═══════════════════════════════════════════════════════════════════════════

def split_group_captain(
    df: pd.DataFrame,
    *,
    captain_col: str = "captain_id",
    voyage_col: str = "voyage_id",
    seed: int = None,
) -> SplitResult:
    """Hold out entire captains for validation and test."""
    return _split_group(
        df,
        group_col=captain_col,
        voyage_col=voyage_col,
        seed=seed or ML_CFG.random_seed,
        label="B (captain holdout)",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Split C: Group Holdout by Agent
# ═══════════════════════════════════════════════════════════════════════════

def split_group_agent(
    df: pd.DataFrame,
    *,
    agent_col: str = "agent_id",
    voyage_col: str = "voyage_id",
    seed: int = None,
) -> SplitResult:
    """Hold out entire agents for validation and test."""
    return _split_group(
        df,
        group_col=agent_col,
        voyage_col=voyage_col,
        seed=seed or ML_CFG.random_seed,
        label="C (agent holdout)",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Split D: Switch-Event Split
# ═══════════════════════════════════════════════════════════════════════════

def split_switch_event(
    df: pd.DataFrame,
    *,
    switch_col: str = "switch_agent",
    captain_col: str = "captain_id",
    voyage_col: str = "voyage_id",
    window_before: int = 3,
    window_after: int = 3,
    seed: int = None,
) -> SplitResult:
    """
    Split around agent-switch events.

    Groups voyages into switch episodes (window_before + window_after around
    each switch). Entire episodes stay together. Non-switch voyages are
    assigned to train.
    """
    seed = seed or ML_CFG.random_seed
    rng = np.random.RandomState(seed)

    # Identify switch events
    df = df.copy()
    df["_vnum"] = df.groupby(captain_col).cumcount()
    switch_mask = df[switch_col] == 1
    switch_captains = df.loc[switch_mask, captain_col].unique()

    episode_voyages: Dict[int, set] = {}
    ep_id = 0
    for cap in switch_captains:
        cap_df = df[df[captain_col] == cap].sort_values("_vnum")
        switch_vnums = cap_df.loc[cap_df[switch_col] == 1, "_vnum"].values
        for sv in switch_vnums:
            lo = max(0, sv - window_before)
            hi = sv + window_after
            ep_voys = set(
                cap_df.loc[
                    (cap_df["_vnum"] >= lo) & (cap_df["_vnum"] <= hi),
                    voyage_col,
                ]
            )
            episode_voyages[ep_id] = ep_voys
            ep_id += 1

    # Shuffle episodes
    ep_ids = list(episode_voyages.keys())
    rng.shuffle(ep_ids)
    n_ep = len(ep_ids)
    n_train_ep = max(1, int(n_ep * ML_CFG.train_frac))
    n_val_ep = max(1, int(n_ep * ML_CFG.val_frac))

    train_ep = ep_ids[:n_train_ep]
    val_ep = ep_ids[n_train_ep:n_train_ep + n_val_ep]
    test_ep = ep_ids[n_train_ep + n_val_ep:]

    def _gather(eps):
        voys = set()
        for e in eps:
            voys |= episode_voyages[e]
        return voys

    train_voys = _gather(train_ep)
    val_voys = _gather(val_ep)
    test_voys = _gather(test_ep)

    # Non-episode voyages go to train
    all_ep_voys = train_voys | val_voys | test_voys
    non_ep_voys = set(df[voyage_col].unique()) - all_ep_voys
    train_voys |= non_ep_voys

    train_idx = np.where(df[voyage_col].isin(train_voys))[0]
    val_idx = np.where(df[voyage_col].isin(val_voys))[0]
    test_idx = np.where(df[voyage_col].isin(test_voys))[0]

    df.drop(columns=["_vnum"], inplace=True, errors="ignore")
    _log_split("D (switch event)", len(train_idx), len(val_idx), len(test_idx))
    return train_idx, val_idx, test_idx


# ═══════════════════════════════════════════════════════════════════════════
# Split E: Spatial Holdout
# ═══════════════════════════════════════════════════════════════════════════

def split_spatial_holdout(
    df: pd.DataFrame,
    *,
    ground_col: str = "ground_id",
    year_col: str = "year_out",
    voyage_col: str = "voyage_id",
    seed: int = None,
) -> SplitResult:
    """
    Spatial holdout by ground or ground-year.

    Hold out entire grounds (or ground-years) for test/val.
    """
    seed = seed or ML_CFG.random_seed
    rng = np.random.RandomState(seed)

    if ground_col not in df.columns:
        logger.warning("No %s column; falling back to rolling time split", ground_col)
        return split_rolling_time(df, year_col=year_col, voyage_col=voyage_col)

    grounds = df[ground_col].dropna().unique()
    rng.shuffle(grounds)
    n = len(grounds)
    n_train = max(1, int(n * ML_CFG.train_frac))
    n_val = max(1, int(n * ML_CFG.val_frac))

    train_grounds = set(grounds[:n_train])
    val_grounds = set(grounds[n_train:n_train + n_val])
    test_grounds = set(grounds[n_train + n_val:])

    train_idx = np.where(df[ground_col].isin(train_grounds))[0]
    val_idx = np.where(df[ground_col].isin(val_grounds))[0]
    test_idx = np.where(df[ground_col].isin(test_grounds))[0]

    _log_split("E (spatial holdout)", len(train_idx), len(val_idx), len(test_idx))
    return train_idx, val_idx, test_idx


# ═══════════════════════════════════════════════════════════════════════════
# Convenience dispatcher
# ═══════════════════════════════════════════════════════════════════════════

SPLIT_REGISTRY = {
    "A": split_rolling_time,
    "B": split_group_captain,
    "C": split_group_agent,
    "D": split_switch_event,
    "E": split_spatial_holdout,
}


def get_split(
    df: pd.DataFrame,
    split_name: str,
    **kwargs,
) -> SplitResult:
    """Get a named split. split_name one of A, B, C, D, E."""
    fn = SPLIT_REGISTRY.get(split_name.upper())
    if fn is None:
        raise ValueError(f"Unknown split '{split_name}'. Options: {list(SPLIT_REGISTRY)}")
    return fn(df, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _split_group(
    df: pd.DataFrame,
    group_col: str,
    voyage_col: str,
    seed: int,
    label: str,
) -> SplitResult:
    """Generic group-holdout split."""
    rng = np.random.RandomState(seed)
    groups = df[group_col].dropna().unique()
    rng.shuffle(groups)
    n = len(groups)
    n_test = max(1, int(n * ML_CFG.group_holdout_test_frac))
    n_val = max(1, int(n * ML_CFG.group_holdout_val_frac))

    test_groups = set(groups[:n_test])
    val_groups = set(groups[n_test:n_test + n_val])
    train_groups = set(groups[n_test + n_val:])

    train_idx = np.where(df[group_col].isin(train_groups))[0]
    val_idx = np.where(df[group_col].isin(val_groups))[0]
    test_idx = np.where(df[group_col].isin(test_groups))[0]

    _log_split(label, len(train_idx), len(val_idx), len(test_idx))
    _check_no_leakage(df, train_idx, val_idx, test_idx, voyage_col)
    return train_idx, val_idx, test_idx


def _log_split(label: str, n_train: int, n_val: int, n_test: int) -> None:
    total = n_train + n_val + n_test
    logger.info(
        "Split %s: train=%d (%.1f%%), val=%d (%.1f%%), test=%d (%.1f%%)",
        label,
        n_train, 100 * n_train / max(total, 1),
        n_val, 100 * n_val / max(total, 1),
        n_test, 100 * n_test / max(total, 1),
    )


def _check_no_leakage(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    voyage_col: str,
) -> None:
    """Assert no voyage appears in more than one partition."""
    train_voys = set(df.iloc[train_idx][voyage_col])
    val_voys = set(df.iloc[val_idx][voyage_col])
    test_voys = set(df.iloc[test_idx][voyage_col])

    tv = train_voys & val_voys
    tt = train_voys & test_voys
    vt = val_voys & test_voys

    if tv or tt or vt:
        raise ValueError(
            f"Leakage detected! train∩val={len(tv)}, train∩test={len(tt)}, val∩test={len(vt)}"
        )
