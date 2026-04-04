"""
Held-out Route Prediction (Step 8 — optional).

If ground_or_route is well-populated: multinomial prediction of route
choice using psi_hat, theta_hat, and controls.  Evaluates whether
organizational environment (psi) carries predictive power for route
choice beyond captain identity and hardware.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from .config import CFG, VOYAGE_PARQUET, INTERMEDIATES_DIR, TABLES_DIR
from .output_schema import save_result_table

logger = logging.getLogger(__name__)


def run_route_prediction() -> pd.DataFrame:
    """Held-out route prediction.

    Fits a Random Forest classifier predicting ground_or_route using
    psi_hat, theta_hat, and controls, and evaluates accuracy in a
    stratified train/test split.
    """
    logger.info("=" * 60)
    logger.info("STEP 8: HELD-OUT ROUTE PREDICTION (OPTIONAL)")
    logger.info("=" * 60)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_parquet(VOYAGE_PARQUET)
    df = df.dropna(subset=["captain_id", "agent_id", "year_out"]).copy()

    # Check if route column is usable
    route_col = "ground_or_route" if "ground_or_route" in df.columns else None
    if route_col is None:
        logger.info("  No ground_or_route column found; skipping route prediction")
        return pd.DataFrame()

    # Need at least 5 routes with ≥20 observations each
    route_counts = df[route_col].value_counts()
    valid_routes = route_counts[route_counts >= 20].index
    if len(valid_routes) < 5:
        logger.info("  Insufficient route diversity (%d routes); skipping", len(valid_routes))
        return pd.DataFrame()

    df = df[df[route_col].isin(valid_routes)].copy()
    logger.info("  Valid routes: %d, observations: %d", len(valid_routes), len(df))

    # Merge effects
    psi_path = INTERMEDIATES_DIR / "psi_hat_leave_one_captain_out.parquet"
    if psi_path.exists():
        psi_loo = pd.read_parquet(psi_path)
        df = df.merge(psi_loo[["voyage_id", "psi_hat_loo"]], on="voyage_id", how="left")
    else:
        df["psi_hat_loo"] = np.nan

    theta_path = INTERMEDIATES_DIR / "theta_hat_separate_sample.parquet"
    if theta_path.exists():
        theta = pd.read_parquet(theta_path)
        df = df.merge(theta[["captain_id", "theta_hat_sep"]], on="captain_id", how="left")
    else:
        df["theta_hat_sep"] = np.nan

    # Build features
    feature_cols = []
    for col in ["psi_hat_loo", "theta_hat_sep", "tonnage", "year_out"]:
        if col in df.columns:
            feature_cols.append(col)

    if "tonnage" in df.columns:
        df["log_tonnage"] = np.log(df["tonnage"].clip(lower=1))
        feature_cols.append("log_tonnage")
        if "tonnage" in feature_cols:
            feature_cols.remove("tonnage")

    df_valid = df.dropna(subset=feature_cols + [route_col]).copy()
    if len(df_valid) < 100:
        logger.info("  Insufficient valid observations (%d); skipping", len(df_valid))
        return pd.DataFrame()

    le = LabelEncoder()
    y = le.fit_transform(df_valid[route_col])
    X = df_valid[feature_cols].values

    # Cross-validation
    results_rows = []
    feature_sets = {
        "theta_only": [c for c in feature_cols if "theta" in c],
        "psi_only": [c for c in feature_cols if "psi" in c],
        "controls_only": [c for c in feature_cols if c not in ["psi_hat_loo", "theta_hat_sep"]],
        "theta_psi": [c for c in feature_cols if "theta" in c or "psi" in c],
        "all_features": feature_cols,
    }

    for set_name, feat_list in feature_sets.items():
        valid_feats = [c for c in feat_list if c in feature_cols]
        if not valid_feats:
            continue

        X_sub = df_valid[valid_feats].values
        try:
            rf = RandomForestClassifier(
                n_estimators=200, max_depth=8, random_state=CFG.random_seed, n_jobs=-1
            )
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=CFG.random_seed)
            scores = cross_val_score(rf, X_sub, y, cv=cv, scoring="accuracy")
            results_rows.append({
                "feature_set": set_name,
                "features": ", ".join(valid_feats),
                "n_features": len(valid_feats),
                "cv_accuracy_mean": float(scores.mean()),
                "cv_accuracy_std": float(scores.std()),
                "n_obs": len(df_valid),
                "n_routes": len(valid_routes),
                "baseline_accuracy": float(1.0 / len(valid_routes)),
            })
            logger.info("    %s: CV accuracy = %.3f (±%.3f)",
                        set_name, scores.mean(), scores.std())
        except Exception as e:
            logger.warning("    Route prediction failed for %s: %s", set_name, e)

    result = pd.DataFrame(results_rows)
    if not result.empty:
        save_result_table(result, "table_route_heldout_prediction", metadata={
            "description": "Route prediction accuracy with different feature sets",
        })

    logger.info("  Route prediction complete.")
    return result
