"""
Test 7: Tail Submodularity.

Tests whether theta and psi are substitutes in downside-risk production
even if mean-output submodularity is mixed.

Key question: Does the marginal value of psi rise as theta falls
in the left tail, especially under scarcity?
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

from src.next_round.config import (
    OUTPUTS_TABLES, OUTPUTS_FIGURES, PSI_COL, THETA_COL,
    RANDOM_SEED, FIGURE_DPI, FIGURE_FORMAT,
)

logger = logging.getLogger(__name__)


def run_tail_submodularity(*, save_outputs: bool = True) -> Dict:
    """
    Test theta-psi substitution patterns in the left tail.
    """
    logger.info("=" * 60)
    logger.info("Test 7: Tail Submodularity")
    logger.info("=" * 60)

    from src.ml.build_outcome_ml_dataset import build_outcome_ml_dataset

    df = build_outcome_ml_dataset(force_rebuild=False, save=False)

    outcome = "q_total_index" if "q_total_index" in df.columns else "log_q"

    # Build tail indicators
    y = df[outcome]
    df["bottom_decile"] = (y <= y.quantile(0.10)).astype(int)
    df["bottom_5pct"] = (y <= y.quantile(0.05)).astype(int)

    results = {}

    # ── 1. Linear interaction model ───────────────────────────────────
    results["linear"] = _linear_interaction(df, outcome)

    # ── 2. Spline interaction model ───────────────────────────────────
    results["spline"] = _spline_interaction(df, outcome)

    # ── 3. Quantile regression interaction ────────────────────────────
    results["quantile"] = _quantile_interaction(df, outcome)

    # ── 4. Flexible response surface ──────────────────────────────────
    results["surface"] = _flexible_surface(df, outcome)

    if save_outputs:
        _save_outputs(results, df, outcome)

    return results


def _linear_interaction(df: pd.DataFrame, outcome: str) -> Dict:
    """Linear interaction: theta × psi."""
    try:
        import statsmodels.formula.api as smf

        df_reg = df[[outcome, PSI_COL, THETA_COL, "captain_id"]].dropna()
        df_reg["theta_x_psi"] = df_reg[THETA_COL] * df_reg[PSI_COL]

        formula = f"{outcome} ~ {PSI_COL} + {THETA_COL} + theta_x_psi"
        model = smf.ols(formula, data=df_reg).fit(
            cov_type="cluster", cov_kwds={"groups": df_reg["captain_id"]})

        return {
            "interaction_coef": model.params.get("theta_x_psi", np.nan),
            "interaction_se": model.bse.get("theta_x_psi", np.nan),
            "interaction_pval": model.pvalues.get("theta_x_psi", np.nan),
            "psi_coef": model.params.get(PSI_COL, np.nan),
            "theta_coef": model.params.get(THETA_COL, np.nan),
            "r_squared": model.rsquared,
            "n_obs": int(model.nobs),
            "interpretation": "negative = substitutes, positive = complements",
        }
    except Exception as e:
        return {"error": str(e)}


def _spline_interaction(df: pd.DataFrame, outcome: str) -> Dict:
    """Spline interaction using basis expansion."""
    try:
        from sklearn.preprocessing import SplineTransformer
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score

        df_clean = df[[outcome, PSI_COL, THETA_COL]].dropna()
        X_psi = df_clean[[PSI_COL]].values
        X_theta = df_clean[[THETA_COL]].values
        y = df_clean[outcome].values

        n = len(y)
        train_n = int(0.7 * n)

        # Spline basis
        spline = SplineTransformer(n_knots=5, degree=3)
        X_psi_s = spline.fit_transform(X_psi)
        X_theta_s = SplineTransformer(n_knots=5, degree=3).fit_transform(X_theta)

        # Interaction basis (outer product of first few terms)
        X_interact = X_psi_s[:, :3] * X_theta_s[:, :3]

        X_full = np.hstack([X_psi_s, X_theta_s, X_interact])

        lr = LinearRegression()
        lr.fit(X_full[:train_n], y[:train_n])
        y_pred = lr.predict(X_full[train_n:])

        return {
            "test_r_squared": r2_score(y[train_n:], y_pred),
            "n_features": X_full.shape[1],
            "n_obs": n,
        }
    except Exception as e:
        return {"error": str(e)}


def _quantile_interaction(df: pd.DataFrame, outcome: str) -> Dict:
    """Quantile regression at the 10th percentile."""
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor

        df_clean = df[[outcome, PSI_COL, THETA_COL, "scarcity"]].dropna()
        available = [PSI_COL, THETA_COL]
        if "scarcity" in df_clean.columns:
            available.append("scarcity")

        X = df_clean[available].values
        y = df_clean[outcome].values

        n = len(y)
        train_n = int(0.7 * n)

        # Quantile loss at 10th percentile
        hgb_q10 = HistGradientBoostingRegressor(
            loss="quantile", quantile=0.10,
            max_iter=200, max_depth=4, random_state=RANDOM_SEED)
        hgb_q10.fit(X[:train_n], y[:train_n])
        q10_pred = hgb_q10.predict(X[train_n:])

        # Marginal effect of psi by theta tertile
        theta_vals = df_clean[THETA_COL].values
        theta_p33 = np.percentile(theta_vals, 33)
        theta_p67 = np.percentile(theta_vals, 67)

        results = {
            "median_q10_pred": np.median(q10_pred),
            "n_obs": n,
        }

        for theta_group, mask in [
            ("low_theta", theta_vals[train_n:] <= theta_p33),
            ("mid_theta", (theta_vals[train_n:] > theta_p33) & (theta_vals[train_n:] <= theta_p67)),
            ("high_theta", theta_vals[train_n:] > theta_p67),
        ]:
            if mask.sum() > 10:
                psi_vals = df_clean[PSI_COL].values[train_n:][mask]
                q10_sub = q10_pred[mask]
                results[f"psi_q10_slope_{theta_group}"] = np.corrcoef(psi_vals, q10_sub)[0, 1]

        return results
    except Exception as e:
        return {"error": str(e)}


def _flexible_surface(df: pd.DataFrame, outcome: str) -> Dict:
    """Flexible response surface over theta, psi, scarcity."""
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.metrics import r2_score

        features = [PSI_COL, THETA_COL]
        if "scarcity" in df.columns:
            features.append("scarcity")

        df_clean = df[features + [outcome]].dropna()
        X = df_clean[features].values
        y = df_clean[outcome].values

        n = len(y)
        train_n = int(0.7 * n)

        hgb = HistGradientBoostingRegressor(
            max_iter=200, max_depth=4, random_state=RANDOM_SEED)
        hgb.fit(X[:train_n], y[:train_n])

        return {
            "test_r_squared": r2_score(y[train_n:], hgb.predict(X[train_n:])),
            "n_obs": n,
        }
    except Exception as e:
        return {"error": str(e)}


def _save_outputs(results: Dict, df: pd.DataFrame, outcome: str):
    """Save results."""
    rows = []
    for method, info in results.items():
        if isinstance(info, dict):
            info["method"] = method
            rows.append(info)

    if rows:
        pd.DataFrame(rows).to_csv(
            OUTPUTS_TABLES / "tail_submodularity.csv", index=False)

    # Figure: marginal return to psi by theta group
    try:
        import matplotlib.pyplot as plt

        quant = results.get("quantile", {})
        groups = ["low_theta", "mid_theta", "high_theta"]
        slopes = [quant.get(f"psi_q10_slope_{g}", np.nan) for g in groups]

        if not all(np.isnan(s) for s in slopes):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(range(len(groups)), slopes,
                   color=["#F44336", "#FFC107", "#4CAF50"])
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels(["Low θ", "Mid θ", "High θ"])
            ax.set_ylabel("Correlation: ψ ↔ Q10 Pred")
            ax.set_title("Marginal Return to ψ by θ Group (10th Percentile)")
            ax.axhline(0, color="black", lw=0.5)
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(OUTPUTS_FIGURES / f"tail_submodularity.{FIGURE_FORMAT}",
                       dpi=FIGURE_DPI, bbox_inches="tight")
            plt.close(fig)
    except ImportError:
        pass

    # Memo
    memo = [
        "# Test 7: Tail Submodularity — Memo",
        "",
        "## What this identifies",
        "Whether θ (captain skill) and ψ (org capability) are substitutes",
        "specifically in downside-risk production (left tail).",
        "",
        "## What this does NOT identify",
        "- Cannot distinguish skill substitution from selection",
        "- Quantile regression is model-dependent",
    ]
    (OUTPUTS_TABLES / "tail_submodularity_memo.md").write_text("\n".join(memo))
