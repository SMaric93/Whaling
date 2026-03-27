"""
ML Layer — Appendix ML-10: Spatial / Latent Patch Quality.

Estimate latent quality of whaling grounds from observed yields,
separate from captain/vessel effects wrt AKM framework.

Methods:
- Ground fixed-effect residualization
- Spatial kernel smoothing of yields
- Time-varying quality index
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.ml.config import ML_CFG, ML_TABLES_DIR, ML_FIGURES_DIR

logger = logging.getLogger(__name__)


def estimate_spatial_quality(
    *,
    save_outputs: bool = True,
) -> Dict[str, Any]:
    """
    Estimate latent patch/ground quality.

    Returns ground-year quality indices and spatial smoothed maps.
    """
    t0 = time.time()
    logger.info("Estimating spatial/ground quality...")

    from src.ml.build_outcome_ml_dataset import build_outcome_ml_dataset

    df = build_outcome_ml_dataset()
    target_col = "log_q"

    if target_col not in df.columns:
        return {"error": "no_outcome_column"}

    # ── Ground fixed-effect residualization ──────────────────────────
    ground_col = None
    for col in ["ground_or_route", "ground_id", "ground"]:
        if col in df.columns:
            ground_col = col
            break

    if ground_col is None:
        return {"error": "no_ground_column"}

    year_col = "year_out" if "year_out" in df.columns else "year"

    # Residualize outcome for captain/vessel effects
    df_valid = df.dropna(subset=[target_col, ground_col]).copy()

    # Simple approach: demean within captain
    if "theta_hat_holdout" in df_valid.columns:
        X_controls = []
        control_names = []
        for c in ["theta_hat_holdout", "tonnage"]:
            if c in df_valid.columns:
                X_controls.append(df_valid[c].fillna(0).values)
                control_names.append(c)

        if X_controls:
            from sklearn.linear_model import LinearRegression
            X = np.column_stack(X_controls)
            y = df_valid[target_col].values
            lr = LinearRegression().fit(X, y)
            residuals = y - lr.predict(X)
            df_valid["_residual"] = residuals
        else:
            df_valid["_residual"] = df_valid[target_col]
    else:
        df_valid["_residual"] = df_valid[target_col]

    # ── Ground-year quality index ───────────────────────────────────
    if year_col in df_valid.columns:
        quality_index = df_valid.groupby([ground_col, year_col]).agg(
            quality_mean=("_residual", "mean"),
            quality_std=("_residual", "std"),
            n_voyages=("_residual", "count"),
            raw_output_mean=(target_col, "mean"),
        ).reset_index()
    else:
        quality_index = df_valid.groupby(ground_col).agg(
            quality_mean=("_residual", "mean"),
            quality_std=("_residual", "std"),
            n_voyages=("_residual", "count"),
            raw_output_mean=(target_col, "mean"),
        ).reset_index()

    # ── Quality volatility (risk measure) ───────────────────────────
    ground_volatility = df_valid.groupby(ground_col)["_residual"].agg(
        ["mean", "std", "count"]
    ).reset_index()
    ground_volatility.columns = [ground_col, "quality_mean", "quality_vol", "n_voyages"]

    # ── Time-varying quality ────────────────────────────────────────
    if year_col in df_valid.columns:
        # Rolling 5-year average quality
        yearly_quality = df_valid.groupby([ground_col, year_col])["_residual"].mean().reset_index()
        yearly_quality.columns = [ground_col, year_col, "quality"]

        yearly_quality["quality_5yr_avg"] = yearly_quality.groupby(ground_col)["quality"].transform(
            lambda s: s.rolling(5, min_periods=2).mean()
        )
    else:
        yearly_quality = pd.DataFrame()

    # ── Save ────────────────────────────────────────────────────────
    if save_outputs:
        quality_index.to_csv(ML_TABLES_DIR / "spatial_quality_index.csv", index=False)
        ground_volatility.to_csv(ML_TABLES_DIR / "ground_volatility.csv", index=False)
        if len(yearly_quality) > 0:
            yearly_quality.to_csv(ML_TABLES_DIR / "time_varying_quality.csv", index=False)

    elapsed = time.time() - t0
    logger.info(
        "Spatial quality: %d grounds, %.1fs",
        quality_index[ground_col].nunique(), elapsed,
    )

    return {
        "quality_index": quality_index,
        "ground_volatility": ground_volatility,
        "yearly_quality": yearly_quality,
    }
