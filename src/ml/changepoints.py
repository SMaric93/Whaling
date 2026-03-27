"""
ML Layer — Appendix ML-7: Change-Point Detection.

Detect shifts in search behavior within a voyage.

Methods:
1. PELT/pruned exact linear time (via ruptures)
2. Bayesian online change-point detection (BOCPD) fallback
3. Simple CUSUM baseline
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.ml.config import ML_CFG, ML_TABLES_DIR, ML_FIGURES_DIR

logger = logging.getLogger(__name__)


def detect_changepoints(
    df: pd.DataFrame = None,
    *,
    signal_cols: List[str] = None,
    method: str = "pelt",
    min_segment_length: int = 5,
    save_outputs: bool = True,
) -> Dict[str, Any]:
    """
    Detect changepoints in search behavior within voyages.

    Parameters
    ----------
    signal_cols : list
        Columns to monitor for change. Default: speed, turn_angle.
    method : str
        'pelt', 'binseg', or 'cusum'
    """
    t0 = time.time()
    logger.info("Detecting changepoints (method=%s)...", method)

    if df is None:
        from src.ml.build_action_dataset import build_action_dataset
        df = build_action_dataset()

    if "active_search_flag" in df.columns:
        df = df[df["active_search_flag"] == 1].copy()

    signal_cols = signal_cols or ["speed", "move_length", "turn_angle"]
    signal_cols = [c for c in signal_cols if c in df.columns]

    if not signal_cols:
        return {"error": "no_signal_columns"}

    # ── Detect changepoints per voyage ──────────────────────────────
    all_changepoints = []

    try:
        import ruptures as rpt
        ruptures_available = True
    except ImportError:
        ruptures_available = False
        logger.warning("ruptures not installed; using CUSUM fallback")

    for vid, grp in df.groupby("voyage_id"):
        if len(grp) < min_segment_length * 2:
            continue

        signal = grp[signal_cols].fillna(0).values

        if ruptures_available and method in ("pelt", "binseg"):
            try:
                if method == "pelt":
                    algo = rpt.Pelt(model="rbf", min_size=min_segment_length).fit(signal)
                    bkps = algo.predict(pen=np.log(len(signal)) * signal.shape[1])
                else:
                    algo = rpt.Binseg(model="l2", min_size=min_segment_length).fit(signal)
                    bkps = algo.predict(n_bkps=min(5, len(grp) // min_segment_length))
            except Exception as e:
                logger.debug("ruptures failed for voyage %s: %s", vid, e)
                bkps = _cusum_detect(signal, min_segment_length)
        else:
            bkps = _cusum_detect(signal, min_segment_length)

        # Convert to records
        dates = grp["obs_date"].values if "obs_date" in grp.columns else np.arange(len(grp))
        for bp in bkps:
            if bp < len(grp):  # ruptures includes endpoint
                all_changepoints.append({
                    "voyage_id": vid,
                    "changepoint_idx": bp,
                    "voyage_day": grp.iloc[bp]["voyage_day"] if "voyage_day" in grp.columns else bp,
                    "date": dates[bp] if bp < len(dates) else None,
                    "n_changepoints": len(bkps) - 1,  # -1 for endpoint
                })

    cp_df = pd.DataFrame(all_changepoints)

    # ── Summary statistics ──────────────────────────────────────────
    if len(cp_df) > 0:
        # Merge with voyage-level data for grouping
        cp_df["has_changepoint"] = 1
        by_voyage = cp_df.groupby("voyage_id").agg(
            n_changepoints=("changepoint_idx", "count"),
            first_cp_day=("voyage_day", "min"),
            last_cp_day=("voyage_day", "max"),
        )

        # Link to psi
        if "psi_hat_holdout" in df.columns:
            psi_by_voyage = df.groupby("voyage_id")["psi_hat_holdout"].first()
            by_voyage = by_voyage.merge(psi_by_voyage, on="voyage_id", how="left")

        summary = by_voyage.describe()
    else:
        by_voyage = pd.DataFrame()
        summary = pd.DataFrame()

    if save_outputs and len(cp_df) > 0:
        cp_df.to_csv(ML_TABLES_DIR / "changepoints.csv", index=False)
        if len(by_voyage) > 0:
            by_voyage.to_csv(ML_TABLES_DIR / "changepoints_by_voyage.csv")

    elapsed = time.time() - t0
    logger.info("Changepoint detection: %d voyages, %d total changepoints, %.1fs",
                cp_df["voyage_id"].nunique() if len(cp_df) > 0 else 0,
                len(cp_df), elapsed)

    return {
        "changepoints": cp_df,
        "by_voyage": by_voyage,
        "summary": summary,
    }


def _cusum_detect(signal: np.ndarray, min_len: int) -> List[int]:
    """Simple CUSUM-based changepoint detection fallback."""
    if signal.ndim > 1:
        signal = signal.mean(axis=1)

    n = len(signal)
    mean = signal.mean()
    std = signal.std() + 1e-8
    threshold = 4.0  # ~4 sigma

    S_plus = np.zeros(n)
    S_minus = np.zeros(n)
    changepoints = []
    last_cp = 0

    for i in range(1, n):
        z = (signal[i] - mean) / std
        S_plus[i] = max(0, S_plus[i - 1] + z - 0.5)
        S_minus[i] = max(0, S_minus[i - 1] - z - 0.5)

        if (S_plus[i] > threshold or S_minus[i] > threshold) and (i - last_cp) >= min_len:
            changepoints.append(i)
            S_plus[i] = 0
            S_minus[i] = 0
            last_cp = i

    changepoints.append(n)  # endpoint
    return changepoints
