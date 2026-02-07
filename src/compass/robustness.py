"""
Compass Pipeline — Reliability & Robustness Checks (Step 9).

Implements five diagnostic batteries:

1. **Split-half reliability** — ICC of compass indices from two halves
   of each voyage's search steps.
2. **Step-definition robustness** — rank-correlation of indices across
   time-resampled vs distance-thinned variants.
3. **Regime placebo** — compass indices computed on *transit* steps
   (expect null organisational effects).
4. **HMM K sensitivity** — compare indices with K = 3 vs K = 4.
5. **Missingness sensitivity** — compare with / without gap-flagged
   voyages.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from compass.config import CompassConfig

logger = logging.getLogger(__name__)


# ── 1. split-half reliability ───────────────────────────────────────────────

def split_half_reliability(
    steps_df: pd.DataFrame,
    cfg: CompassConfig,
) -> Dict[str, float]:
    """
    Split early search steps into odd/even halves, compute CompassIndex1
    on each, and report ICC (intra-class correlation).

    Returns
    -------
    dict with ``icc``, ``pearson_r``, ``spearman_r``, ``n_voyages``.
    """
    from compass.features import compute_compass_features
    from compass.compass_index import compute_compass_index

    search = steps_df.loc[steps_df["regime_label"] == "search"].copy()

    half_a_parts, half_b_parts = [], []
    for _vid, sub in search.groupby("voyage_id", sort=False):
        sub = sub.sort_values("timestamp_utc").reset_index(drop=True)
        half_a_parts.append(sub.iloc[::2])
        half_b_parts.append(sub.iloc[1::2])

    half_a = pd.concat(half_a_parts, ignore_index=True) if half_a_parts else pd.DataFrame()
    half_b = pd.concat(half_b_parts, ignore_index=True) if half_b_parts else pd.DataFrame()

    feat_a = compute_compass_features(half_a, cfg)
    feat_b = compute_compass_features(half_b, cfg)

    if feat_a.empty or feat_b.empty:
        return {"icc": np.nan, "pearson_r": np.nan, "spearman_r": np.nan, "n_voyages": 0}

    idx_a, _ = compute_compass_index(feat_a, cfg)
    idx_b, _ = compute_compass_index(feat_b, cfg)

    merged = idx_a[["voyage_id", "CompassIndex1"]].merge(
        idx_b[["voyage_id", "CompassIndex1"]],
        on="voyage_id",
        suffixes=("_a", "_b"),
    ).dropna()

    if len(merged) < 3:
        return {"icc": np.nan, "pearson_r": np.nan, "spearman_r": np.nan, "n_voyages": len(merged)}

    a = merged["CompassIndex1_a"].values
    b = merged["CompassIndex1_b"].values

    # ICC(3,1) — two-way mixed, single measures
    n = len(a)
    mean_ab = (a + b) / 2
    ssb = n * np.var(mean_ab, ddof=1)
    sse = np.sum((a - b) ** 2) / 2
    msb = ssb / max(n - 1, 1)
    mse = sse / max(n, 1)
    icc = (msb - mse) / (msb + mse) if (msb + mse) > 0 else np.nan

    pearson_r = float(np.corrcoef(a, b)[0, 1])
    spearman_r = float(spearmanr(a, b).statistic)

    result = {
        "icc": float(icc),
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
        "n_voyages": int(n),
    }
    logger.info("Split-half reliability: %s", result)
    return result


# ── 2. step-definition robustness ───────────────────────────────────────────

def step_definition_robustness(
    variants: Dict[str, pd.DataFrame],
    cfg: CompassConfig,
) -> Dict[str, float]:
    """
    Rank-correlate CompassIndex1 across step-definition variants.

    Parameters
    ----------
    variants : dict
        Keyed by variant name, values are step DataFrames with regime labels.

    Returns
    -------
    dict of pairwise Spearman correlations.
    """
    from compass.features import compute_compass_features
    from compass.compass_index import compute_compass_index

    indices: Dict[str, pd.DataFrame] = {}
    for name, sdf in variants.items():
        feat = compute_compass_features(sdf, cfg)
        if feat.empty:
            continue
        idx_df, _ = compute_compass_index(feat, cfg)
        indices[name] = idx_df[["voyage_id", "CompassIndex1"]].dropna()

    results: Dict[str, float] = {}
    names = list(indices.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            merged = indices[names[i]].merge(
                indices[names[j]],
                on="voyage_id",
                suffixes=(f"_{names[i]}", f"_{names[j]}"),
            )
            if len(merged) < 3:
                continue
            a = merged.iloc[:, 1].values
            b = merged.iloc[:, 2].values
            r = float(spearmanr(a, b).statistic)
            key = f"{names[i]}_vs_{names[j]}"
            results[key] = r

    logger.info("Step-definition robustness: %s", results)
    return results


# ── 3. regime placebo ───────────────────────────────────────────────────────

def regime_placebo(
    steps_df: pd.DataFrame,
    cfg: CompassConfig,
) -> Dict[str, float]:
    """
    Compute compass features on *transit* regime steps.

    The expectation is that Compass indices computed from transit should
    show much smaller organisational/captain effects than search indices.

    Returns feature-level summary stats for comparison.
    """
    from compass.features import compute_compass_features

    transit = steps_df.copy()
    transit["regime_label"] = np.where(
        transit["regime_label"] == "transit", "search", transit["regime_label"],
    )
    feat = compute_compass_features(transit, cfg)

    if feat.empty:
        return {"n_voyages": 0}

    summary = {
        "n_voyages": int(len(feat)),
    }
    for col in feat.select_dtypes(include="number").columns:
        if col == "n_search_steps":
            continue
        summary[f"{col}_mean"] = float(feat[col].mean())
        summary[f"{col}_std"] = float(feat[col].std())

    logger.info("Regime placebo (transit): %d voyages.", len(feat))
    return summary


# ── 4. HMM K sensitivity ───────────────────────────────────────────────────

def hmm_k_sensitivity(
    steps_raw: pd.DataFrame,
    cfg: CompassConfig,
) -> Dict[str, float]:
    """
    Re-run regime segmentation with each K candidate, compare
    CompassIndex1 rank-correlations.
    """
    from compass.regimes import segment_voyages
    from compass.features import compute_compass_features
    from compass.compass_index import compute_compass_index
    from copy import deepcopy

    indices: Dict[int, pd.DataFrame] = {}
    for K in cfg.num_regimes_candidates:
        cfg_k = deepcopy(cfg)
        cfg_k.num_regimes_candidates = [K]
        segmented = segment_voyages(steps_raw.copy(), cfg_k)
        feat = compute_compass_features(segmented, cfg_k)
        if feat.empty:
            continue
        idx_df, _ = compute_compass_index(feat, cfg_k)
        indices[K] = idx_df[["voyage_id", "CompassIndex1"]].dropna()

    results: Dict[str, float] = {}
    ks = sorted(indices.keys())
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            merged = indices[ks[i]].merge(
                indices[ks[j]],
                on="voyage_id",
                suffixes=(f"_K{ks[i]}", f"_K{ks[j]}"),
            )
            if len(merged) < 3:
                continue
            a = merged.iloc[:, 1].values
            b = merged.iloc[:, 2].values
            r = float(spearmanr(a, b).statistic)
            results[f"K{ks[i]}_vs_K{ks[j]}"] = r

    logger.info("HMM K sensitivity: %s", results)
    return results


# ── 5. missingness sensitivity ──────────────────────────────────────────────

def missingness_sensitivity(
    steps_df: pd.DataFrame,
    cfg: CompassConfig,
) -> Dict[str, float]:
    """
    Re-run excluding voyages that had any large gap flags.

    Returns Spearman correlation of CompassIndex1 between full and
    reduced samples.
    """
    from compass.features import compute_compass_features
    from compass.compass_index import compute_compass_index

    # full sample
    feat_full = compute_compass_features(steps_df, cfg)
    if feat_full.empty:
        return {"spearman_r": np.nan, "n_full": 0, "n_reduced": 0}
    idx_full, _ = compute_compass_index(feat_full, cfg)

    # reduced: drop voyages that had any gap_flag
    if "gap_flag" in steps_df.columns:
        gap_vids = steps_df.loc[steps_df["gap_flag"] == True, "voyage_id"].unique()
        reduced = steps_df.loc[~steps_df["voyage_id"].isin(gap_vids)]
    else:
        reduced = steps_df

    feat_red = compute_compass_features(reduced, cfg)
    if feat_red.empty:
        return {"spearman_r": np.nan, "n_full": len(feat_full), "n_reduced": 0}
    idx_red, _ = compute_compass_index(feat_red, cfg)

    merged = idx_full[["voyage_id", "CompassIndex1"]].merge(
        idx_red[["voyage_id", "CompassIndex1"]],
        on="voyage_id",
        suffixes=("_full", "_reduced"),
    ).dropna()

    r = float(spearmanr(
        merged["CompassIndex1_full"], merged["CompassIndex1_reduced"],
    ).statistic) if len(merged) >= 3 else np.nan

    result = {
        "spearman_r": r,
        "n_full": int(len(feat_full)),
        "n_reduced": int(len(feat_red)),
    }
    logger.info("Missingness sensitivity: %s", result)
    return result


# ── aggregate report ────────────────────────────────────────────────────────

def run_all_robustness(
    steps_df: pd.DataFrame,
    variants: Optional[Dict[str, pd.DataFrame]],
    steps_raw: Optional[pd.DataFrame],
    cfg: CompassConfig,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    Run all robustness checks and save report.

    Parameters
    ----------
    steps_df : pd.DataFrame
        Primary step variant with regime labels.
    variants : dict, optional
        All step-definition variants (for step-definition robustness).
    steps_raw : pd.DataFrame, optional
        Raw steps *before* regime labelling (for K sensitivity).
    cfg : CompassConfig
    output_dir : Path, optional

    Returns
    -------
    dict — full report.
    """
    report: Dict = {}

    report["split_half"] = split_half_reliability(steps_df, cfg)
    logger.info("✓ Split-half reliability done.")

    if variants:
        report["step_definition"] = step_definition_robustness(variants, cfg)
        logger.info("✓ Step-definition robustness done.")

    report["regime_placebo"] = regime_placebo(steps_df, cfg)
    logger.info("✓ Regime placebo done.")

    if steps_raw is not None:
        report["hmm_k_sensitivity"] = hmm_k_sensitivity(steps_raw, cfg)
        logger.info("✓ HMM K sensitivity done.")

    report["missingness"] = missingness_sensitivity(steps_df, cfg)
    logger.info("✓ Missingness sensitivity done.")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "robustness_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Saved robustness_report.json.")

    return report
