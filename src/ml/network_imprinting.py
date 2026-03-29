"""
ML Layer — Appendix ML-8: Network Imprinting.

Test whether co-working exposure to a capable agent changes a captain's
subsequent behavior. Graph-based diffusion of organizational capability.

Uses: network_dataset from build_network_dataset.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.ml.config import ML_CFG, ML_TABLES_DIR, ML_FIGURES_DIR

logger = logging.getLogger(__name__)


def analyze_network_imprinting(
    *,
    save_outputs: bool = True,
) -> Dict[str, Any]:
    """
    Test for network-mediated capability transfer.

    1. Compute exposure-weighted psi for each captain
    2. Test whether early-career exposure predicts later performance
    3. Compare pre/post exposure outcomes
    """
    t0 = time.time()
    logger.info("Analyzing network imprinting...")

    from src.ml.build_network_dataset import build_network_dataset
    from src.ml.build_outcome_ml_dataset import build_outcome_ml_dataset

    network = build_network_dataset()
    outcomes = build_outcome_ml_dataset()

    if len(network) == 0:
        return {"error": "no_network_data"}

    # ── Exposure-weighted psi ───────────────────────────────────────
    ca_edges = network[network["relationship_type"] == "captain_agent"].copy()

    if len(ca_edges) == 0 or "psi_hat_holdout" not in outcomes.columns:
        return {"error": "insufficient_data"}

    # Get agent-level psi
    agent_psi = outcomes.groupby("agent_id")["psi_hat_holdout"].mean()

    # Merge agent psi into edges
    ca_edges = ca_edges.merge(
        agent_psi.rename("agent_psi"),
        left_on="person_id_2", right_index=True, how="left",
    )

    def _weighted_psi(g):
        mask = g["agent_psi"].notna()
        if not mask.any():
            return np.nan
        return np.average(g.loc[mask, "agent_psi"], weights=g.loc[mask, "exposure_count"])

    cap_exposure = ca_edges.groupby("person_id_1").apply(
        _weighted_psi
    ).rename("exposure_psi")

    # Early-career exposure
    early_edges = ca_edges[ca_edges["early_career_flag"] == 1]
    cap_early_exposure = early_edges.groupby("person_id_1").apply(
        _weighted_psi
    ).rename("early_exposure_psi")

    # ── Merge with outcomes ─────────────────────────────────────────
    # Later-career outcomes
    if "captain_id" in outcomes.columns and "captain_voyage_num" in outcomes.columns:
        late = outcomes[outcomes["captain_voyage_num"] > ML_CFG.experience_bins["novice_max"]].copy()

        late = late.merge(cap_exposure, left_on="captain_id", right_index=True, how="left")
        late = late.merge(cap_early_exposure, left_on="captain_id", right_index=True, how="left")

        # ── Simple regression test ──────────────────────────────────
        target_col = "log_q"
        if target_col in late.columns:
            valid = late.dropna(subset=[target_col, "early_exposure_psi"])

            if len(valid) > 30:
                from sklearn.linear_model import LinearRegression
                X_exp = valid[["early_exposure_psi"]].values
                y = valid[target_col].values

                lr = LinearRegression().fit(X_exp, y)
                coef = lr.coef_[0]
                r2 = lr.score(X_exp, y)

                logger.info("Early exposure → later output: coef=%.4f, R²=%.4f", coef, r2)

                imprinting_result = {
                    "early_exposure_coef": float(coef),
                    "r_squared": float(r2),
                    "n_captains": int(valid["captain_id"].nunique()),
                    "n_observations": len(valid),
                }

                # With controls
                control_cols = [c for c in ["theta_hat_holdout", "tonnage", "scarcity"]
                               if c in valid.columns]
                if control_cols:
                    X_ctrl = valid[["early_exposure_psi"] + control_cols].fillna(0).values
                    lr_ctrl = LinearRegression().fit(X_ctrl, y)
                    imprinting_result["controlled_coef"] = float(lr_ctrl.coef_[0])
                    imprinting_result["controlled_r2"] = float(lr_ctrl.score(X_ctrl, y))
            else:
                imprinting_result = {"error": "insufficient_late_career_data"}
        else:
            imprinting_result = {"error": "no_outcome_column"}
    else:
        imprinting_result = {"error": "no_captain_data"}

    # ── Network summary statistics ──────────────────────────────────
    network_stats = {
        "n_edges": len(network),
        "n_captain_agent_edges": len(ca_edges),
        "n_captains_with_exposure": len(cap_exposure.dropna()),
        "n_captains_with_early_exposure": len(cap_early_exposure.dropna()),
        "mean_exposure_psi": float(cap_exposure.mean()) if len(cap_exposure) > 0 else np.nan,
    }

    if save_outputs:
        pd.DataFrame([imprinting_result]).to_csv(
            ML_TABLES_DIR / "network_imprinting_results.csv", index=False
        )
        pd.DataFrame([network_stats]).to_csv(
            ML_TABLES_DIR / "network_summary_stats.csv", index=False
        )

    elapsed = time.time() - t0
    logger.info("Network imprinting analysis complete in %.1fs", elapsed)

    return {
        "imprinting": imprinting_result,
        "network_stats": network_stats,
        "exposure_psi": cap_exposure,
        "early_exposure_psi": cap_early_exposure,
    }
