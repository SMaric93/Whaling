"""
Test 3: State-Contingent Stopping-Rule Hazard Model.

Tests whether organizational capability (ψ) affects patch/ground
departure decisions in response to negative signals.

Key prediction: High-ψ organizations enforce patience rules —
they wait longer before abandoning a patch when encountering
consecutive empty days, NOT when encountering positive signals.

Uses daily encounter data (Encounter: NoEnc/Sight/Strike) for
precise measurement of success/failure signals.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import CFG, COLS, TABLES_DIR, FIGURES_DIR
from .utils import cluster_se, make_table, make_figure, save_figure, write_memo

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Core Hazard Model
# ═══════════════════════════════════════════════════════════════════════════

def run_test3(
    patch_days: pd.DataFrame,
    voyages: pd.DataFrame,
    *,
    use_heldout: bool = True,
    save_outputs: bool = True,
) -> Dict:
    """
    Run Test 3: State-contingent stopping-rule hazard model.

    Parameters
    ----------
    patch_days : pd.DataFrame
        Patch-day panel from expand_to_patch_days.
    voyages : pd.DataFrame
        Voyage panel with theta, psi.
    use_heldout : bool
        Use held-out psi.
    save_outputs : bool
        Save tables and figures.

    Returns
    -------
    dict with hazard regression results.
    """
    psi_col = "psi_heldout" if (use_heldout and "psi_heldout" in voyages.columns) else "psi"

    # Merge voyage-level data
    merge_cols = [COLS.voyage_id, COLS.captain_id, COLS.agent_id,
                  psi_col, COLS.year_out, "novice", "expert"]
    if "theta" in voyages.columns:
        merge_cols.append("theta")
    if "theta_heldout" in voyages.columns:
        merge_cols.append("theta_heldout")
    merge_cols = [c for c in merge_cols if c in voyages.columns]

    df = patch_days.merge(
        voyages[list(set(merge_cols))],
        on=COLS.voyage_id,
        how="left",
    )
    df = df.dropna(subset=[psi_col, "exit_tomorrow"])

    logger.info(
        "Test 3: %d patch-day observations, %d voyages",
        len(df), df[COLS.voyage_id].nunique(),
    )

    if len(df) < 100:
        return {"status": "insufficient_data"}

    # ── 1. Main hazard regressions ─────────────────────────────────────
    results = []

    # Spec 1: Baseline logit
    res = _logit_hazard(df, psi_col, spec="baseline")
    if res:
        results.append(res)

    # Spec 2: With negative-signal interaction
    res = _logit_hazard(df, psi_col, spec="interaction")
    if res:
        results.append(res)

    # Spec 3: With ψ × novice interaction
    res = _logit_hazard(df, psi_col, spec="novice_interaction")
    if res:
        results.append(res)

    # ── 2. Placebo: ψ after positive signals ───────────────────────────
    placebo = _logit_hazard(df, psi_col, spec="placebo_positive")
    if placebo:
        results.append(placebo)

    # ── 3. Survival curves ─────────────────────────────────────────────
    survival = _compute_survival_curves(df, psi_col)

    # ── 4. Save ────────────────────────────────────────────────────────
    if save_outputs:
        _save_test3_outputs(results, survival, df, psi_col)

    return {
        "results": results,
        "survival": survival,
        "status": "complete",
    }


def _logit_hazard(
    df: pd.DataFrame,
    psi_col: str,
    spec: str = "baseline",
) -> Optional[Dict]:
    """
    Estimate discrete-time logit hazard model.

    DV: exit_tomorrow (0/1)
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        logger.warning("statsmodels not installed, using manual logit")
        return _manual_logit(df, psi_col, spec)

    # Build regressors
    regressors = {psi_col: df[psi_col].values}
    regressors["consecutive_empty_days"] = df["consecutive_empty_days"].values
    regressors["day_in_patch"] = df["day_in_patch"].values
    regressors["log_day_in_patch"] = np.log(df["day_in_patch"].clip(lower=1))

    if spec == "interaction":
        # Key test: ψ × negative signal
        neg_signal = (df["consecutive_empty_days"] >= CFG.empty_streak_threshold).astype(float)
        regressors["neg_signal"] = neg_signal.values
        regressors[f"{psi_col}_x_neg_signal"] = (
            df[psi_col].values * neg_signal.values
        )
    elif spec == "novice_interaction" and "novice" in df.columns:
        regressors["novice"] = df["novice"].values
        regressors[f"{psi_col}_x_novice"] = (
            df[psi_col].values * df["novice"].values
        )
    elif spec == "placebo_positive":
        pos_signal = (df["encounter_today"] == 1).astype(float)
        regressors["pos_signal"] = pos_signal.values
        regressors[f"{psi_col}_x_pos_signal"] = (
            df[psi_col].values * pos_signal.values
        )

    X = pd.DataFrame(regressors).dropna()
    y = df.loc[X.index, "exit_tomorrow"].values
    X_vals = sm.add_constant(X.values)
    var_names = ["const"] + list(X.columns)

    try:
        model = sm.Logit(y, X_vals)
        result = model.fit(disp=0, maxiter=100)

        coefs = dict(zip(var_names, result.params))
        ses = dict(zip(var_names, result.bse))
        pvals = dict(zip(var_names, result.pvalues))

        # Remove const from output
        for d in [coefs, ses, pvals]:
            d.pop("const", None)

        return {
            "name": f"Logit Hazard ({spec})",
            "coefficients": coefs,
            "se": ses,
            "pvalues": pvals,
            "n_obs": len(y),
            "r_squared": result.prsquared,  # pseudo R²
            "n_clusters": df[COLS.voyage_id].nunique(),
            "cluster_var": COLS.voyage_id,
            "fe_structure": "none",
            "spec": spec,
            "aic": result.aic,
            "bic": result.bic,
        }
    except Exception as e:
        logger.warning("Logit failed for spec %s: %s", spec, e)
        return None


def _manual_logit(df, psi_col, spec):
    """Fallback linear probability model when statsmodels unavailable."""
    from .utils import absorb_fixed_effects

    X_cols = [psi_col, "consecutive_empty_days", "day_in_patch"]
    if spec == "interaction":
        neg_signal = (df["consecutive_empty_days"] >= CFG.empty_streak_threshold).astype(float)
        df = df.copy()
        df["neg_signal"] = neg_signal
        df[f"{psi_col}_x_neg_signal"] = df[psi_col] * neg_signal
        X_cols.extend(["neg_signal", f"{psi_col}_x_neg_signal"])

    clean = df.dropna(subset=X_cols + ["exit_tomorrow"])
    y = clean["exit_tomorrow"].values.astype(float)
    X = clean[X_cols].values.astype(float)

    result = absorb_fixed_effects(y, X, [], return_residuals=True)
    if result["n_obs"] < 50:
        return None

    mask = result["_mask"]
    clusters = clean[COLS.voyage_id].values[mask]
    se = cluster_se(X[mask], result["residuals"], clusters)

    coefs = dict(zip(X_cols, result["coefficients"]))
    ses = dict(zip(X_cols, se))

    return {
        "name": f"LPM Hazard ({spec})",
        "coefficients": coefs,
        "se": ses,
        "pvalues": {k: np.nan for k in X_cols},  # crude
        "n_obs": result["n_obs"],
        "r_squared": result["r_squared"],
        "n_clusters": len(np.unique(clusters)),
        "fe_structure": "none",
        "spec": spec,
    }


def _compute_survival_curves(df, psi_col):
    """Compute Kaplan-Meier-style survival by psi group."""
    df = df.copy()
    median_psi = df[psi_col].median()
    df["psi_group"] = np.where(df[psi_col] >= median_psi, "High ψ", "Low ψ")

    survival = {}
    for group, gdf in df.groupby("psi_group"):
        max_day = int(gdf["day_in_patch"].max())
        surv = []
        for d in range(1, min(max_day + 1, 61)):
            at_risk = (gdf["day_in_patch"] >= d).sum()
            exits = ((gdf["day_in_patch"] == d) & (gdf["exit_tomorrow"] == 1)).sum()
            surv.append({
                "day": d,
                "at_risk": at_risk,
                "exits": exits,
                "hazard": exits / max(at_risk, 1),
            })
        sdf = pd.DataFrame(surv)
        sdf["survival"] = (1 - sdf["hazard"]).cumprod()
        survival[group] = sdf

    return survival


def _save_test3_outputs(results, survival, df, psi_col):
    """Save Test 3 outputs."""
    if results:
        make_table(results, "Exit Hazard", "test3_stopping_rule")

    # Survival curves
    if survival:
        try:
            fig, ax = make_figure("test3", "survival_curves")
            colors = {"High ψ": "#2196F3", "Low ψ": "#FF5722"}
            for group, sdf in survival.items():
                ax.step(sdf["day"], sdf["survival"],
                        label=group, color=colors.get(group, "gray"), linewidth=2)
            ax.set_xlabel("Days in Patch")
            ax.set_ylabel("Survival Probability")
            ax.set_title("Patch Survival by Organizational Capability")
            ax.legend()
            ax.set_xlim(1, 60)
            ax.set_ylim(0, 1.05)
            save_figure(fig, "test3", "survival_curves")
        except Exception as e:
            logger.warning("Failed to save survival figure: %s", e)

    write_memo(
        "test3_stopping_rule",
        "## State-Contingent Stopping Rule\n\n"
        "Tests whether high-ψ organizations enforce patience rules — waiting "
        "longer after consecutive NoEnc days (negative signals) before "
        "abandoning a patch.\n\n"
        f"- Key interaction: ψ × consecutive_empty_days ≥ {CFG.empty_streak_threshold}\n"
        "- Placebo: ψ should NOT predict faster exit after positive signals\n"
        "- Uses real daily encounter data (NoEnc/Sight/Strike)\n",
        threats=[
            "Patch boundaries depend on radius choice (robustness at 25/50/100nm)",
            "Multiple encounters per day may bias daily success measure",
            "Season effects (approaching winter) confound stopping decision",
        ],
    )
    logger.info("Test 3 outputs saved")
