"""
ML Layer — Phase ML-3: Nonlinear Survival Models for Patch Exit.

Models patch exit as a nonlinear policy response to negative information.

Models:
1. Baseline discrete-time logit hazard
2. Gradient boosted classifier on person-period data
3. Random forest classifier
4. Cox PH benchmark (lifelines)

Key interactions: psi × consecutive_empty_days, psi × scarcity.
Required placebos: transit, homebound, shuffled-agent.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.ml.config import ML_CFG, ML_TABLES_DIR, ML_FIGURES_DIR, ML_MODELS_DIR
from src.ml.metrics import classification_metrics, concordance_index
from src.ml.splits import split_rolling_time
from src.ml.baselines import fit_logistic_baseline, fit_cox_baseline

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Feature Definitions
# ═══════════════════════════════════════════════════════════════════════════

CORE_PREDICTORS = [
    "psi_hat_holdout",
    "theta_hat_holdout",
    "consecutive_empty_days",
    "days_since_last_success",
    "duration_day",
    "season_remaining",
    "scarcity",
]

INTERACTION_FEATURES = [
    "psi_x_empty_days",
    "psi_x_days_since_success",
    "psi_x_scarcity",
]


def _add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Add explicit interaction features."""
    df = df.copy()
    psi = df.get("psi_hat_holdout", pd.Series(0, index=df.index))

    for empty_col in ["consecutive_empty_days", "max_empty_streak"]:
        if empty_col in df.columns:
            df["psi_x_empty_days"] = psi * df[empty_col]
            break
    else:
        df["psi_x_empty_days"] = 0

    if "days_since_last_success" in df.columns:
        df["psi_x_days_since_success"] = psi * df["days_since_last_success"]
    else:
        df["psi_x_days_since_success"] = 0

    if "scarcity" in df.columns:
        df["psi_x_scarcity"] = psi * df["scarcity"]
    else:
        df["psi_x_scarcity"] = 0

    return df


# ═══════════════════════════════════════════════════════════════════════════
# Main Survival Model Suite
# ═══════════════════════════════════════════════════════════════════════════

def run_survival_models(
    df: pd.DataFrame = None,
    *,
    save_outputs: bool = True,
) -> Dict[str, Any]:
    """
    Fit all survival models on the patch-spell-day data.

    Returns benchmark table, hazard figures, and placebo results.
    """
    t0 = time.time()
    logger.info("Running survival models...")

    if df is None:
        from src.ml.build_survival_dataset import build_survival_dataset
        df = build_survival_dataset()

    # ── Determine target column ─────────────────────────────────────
    target_col = None
    for col in ["event_exit", "exit_tomorrow", "exit_patch_next"]:
        if col in df.columns:
            target_col = col
            break

    if target_col is None:
        logger.error("No exit event column found in survival dataset")
        return {"error": "no_exit_column"}

    logger.info("Using target column: %s", target_col)

    # ── Add interactions ────────────────────────────────────────────
    df = _add_interactions(df)

    # ── Feature selection ───────────────────────────────────────────
    all_features = CORE_PREDICTORS + INTERACTION_FEATURES
    features = [f for f in all_features if f in df.columns]
    logger.info("Features available: %s", features)

    # Drop rows with missing target
    df = df.dropna(subset=[target_col]).copy()
    df[target_col] = df[target_col].astype(int)

    # ── Split ───────────────────────────────────────────────────────
    train_idx, val_idx, test_idx = split_rolling_time(df)

    X = df[features].fillna(0).values
    y = df[target_col].values

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # ── Fit models ──────────────────────────────────────────────────
    results = {}

    # 1. Discrete-time logit baseline
    try:
        logit = fit_logistic_baseline(X_train, y_train)
        results["discrete_logit"] = _evaluate_clf(logit, X_val, y_val, X_test, y_test, "discrete_logit")
        results["discrete_logit_model"] = logit
    except Exception as e:
        logger.warning("Logit baseline failed: %s", e)

    # 2. HistGradientBoosting
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        hgb = HistGradientBoostingClassifier(
            max_iter=ML_CFG.n_estimators,
            max_depth=ML_CFG.max_depth,
            learning_rate=ML_CFG.learning_rate,
            min_samples_leaf=ML_CFG.min_samples_leaf,
            random_state=ML_CFG.random_seed,
        )
        hgb.fit(X_train, y_train)
        results["hist_gbt"] = _evaluate_clf(hgb, X_val, y_val, X_test, y_test, "hist_gbt")
        results["hist_gbt_model"] = hgb
    except Exception as e:
        logger.warning("HistGBT failed: %s", e)

    # 3. Random forest
    try:
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(
            n_estimators=ML_CFG.rf_n_estimators,
            max_depth=ML_CFG.rf_max_depth,
            min_samples_leaf=ML_CFG.rf_min_samples_leaf,
            random_state=ML_CFG.random_seed,
            n_jobs=1,
        )
        rf.fit(X_train, y_train)
        results["random_forest"] = _evaluate_clf(rf, X_val, y_val, X_test, y_test, "random_forest")
        results["random_forest_model"] = rf
    except Exception as e:
        logger.warning("RF failed: %s", e)

    # 4. Cox PH
    try:
        duration_col = "duration_day" if "duration_day" in df.columns else None
        if duration_col:
            cox = fit_cox_baseline(
                df.iloc[train_idx],
                duration_col=duration_col,
                event_col=target_col,
                covariate_cols=[f for f in features if f != duration_col],
            )
            if cox is not None:
                cox_ci = cox.concordance_index_
                results["cox_ph"] = {
                    "concordance_index": cox_ci,
                    "model": "cox_ph",
                }
                results["cox_ph_model"] = cox
    except Exception as e:
        logger.warning("Cox PH failed: %s", e)

    # ── Build benchmark table ───────────────────────────────────────
    benchmark = _build_survival_benchmark(results)

    # ── Partial dependence on negative signals ──────────────────────
    pdp_results = {}
    best_model = results.get("hist_gbt_model") or results.get("random_forest_model")
    if best_model is not None:
        pdp_results = _compute_exit_pdp(best_model, df, features, save=save_outputs)

    # ── Placebos ────────────────────────────────────────────────────
    placebo_results = _run_placebos(df, features, target_col, save=save_outputs)

    if save_outputs:
        benchmark.to_csv(ML_TABLES_DIR / "exit_policy_benchmark.csv", index=False)

    elapsed = time.time() - t0
    logger.info("Survival models complete in %.1fs", elapsed)

    return {
        "benchmark": benchmark,
        "model_results": results,
        "pdp_results": pdp_results,
        "placebo_results": placebo_results,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation Helper
# ═══════════════════════════════════════════════════════════════════════════

def _evaluate_clf(model, X_val, y_val, X_test, y_test, name: str) -> Dict:
    """Evaluate a classifier on val and test sets."""
    result = {"model": name}
    for split_name, X_eval, y_eval in [("val", X_val, y_val), ("test", X_test, y_test)]:
        y_pred = model.predict(X_eval)
        y_proba = model.predict_proba(X_eval) if hasattr(model, "predict_proba") else None
        if y_proba is not None:
            metrics = classification_metrics(y_eval, y_proba, y_pred)
        else:
            metrics = {"accuracy": float(np.mean(y_pred == y_eval))}
        for k, v in metrics.items():
            result[f"{split_name}_{k}"] = v
    return result


def _build_survival_benchmark(results: Dict) -> pd.DataFrame:
    """Build benchmark table from results."""
    rows = []
    for key, val in results.items():
        if isinstance(val, dict) and "model" in val and not key.endswith("_model"):
            rows.append(val)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════
# Partial Dependence on Negative Signals
# ═══════════════════════════════════════════════════════════════════════════

def _compute_exit_pdp(
    model,
    df: pd.DataFrame,
    features: List[str],
    *,
    save: bool = False,
) -> Dict:
    """
    Compute partial dependence of exit probability on negative signal accumulation.

    Shows how predicted exit hazard changes as empty days / days since success increase.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return {}

    results = {}

    for neg_signal in ["consecutive_empty_days", "days_since_last_success"]:
        if neg_signal not in features:
            continue

        col_idx = features.index(neg_signal)
        grid = np.linspace(0, df[neg_signal].quantile(0.95), 30)

        # Split by psi quartile
        psi_col = "psi_hat_holdout"
        if psi_col in df.columns:
            psi_q = pd.qcut(df[psi_col].rank(method="first"), q=4, labels=["Q1", "Q2", "Q3", "Q4"])
        else:
            psi_q = pd.Series("All", index=df.index)

        fig, ax = plt.subplots(figsize=(9, 6))

        for q_label in psi_q.unique():
            mask = psi_q == q_label
            X_sub = df.loc[mask, features].fillna(0).values

            if len(X_sub) < 50:
                continue

            # Sample for speed
            if len(X_sub) > 1000:
                rng = np.random.RandomState(ML_CFG.random_seed)
                idx = rng.choice(len(X_sub), 1000, replace=False)
                X_sub = X_sub[idx]

            pdp_vals = []
            for g in grid:
                X_mod = X_sub.copy()
                X_mod[:, col_idx] = g
                if hasattr(model, "predict_proba"):
                    pred = model.predict_proba(X_mod)[:, 1].mean()
                else:
                    pred = model.predict(X_mod).mean()
                pdp_vals.append(pred)

            ax.plot(grid, pdp_vals, lw=2, label=f"ψ {q_label}")

        ax.set_xlabel(neg_signal.replace("_", " ").title())
        ax.set_ylabel("Predicted Exit Probability")
        ax.set_title(f"Exit Hazard vs {neg_signal.replace('_', ' ').title()}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        results[neg_signal] = {"grid": grid}

        if save:
            path = ML_FIGURES_DIR / f"exit_hazard_{neg_signal}.{ML_CFG.figure_format}"
            fig.savefig(path, dpi=ML_CFG.figure_dpi, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.close(fig)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Placebos
# ═══════════════════════════════════════════════════════════════════════════

def _run_placebos(
    df: pd.DataFrame,
    features: List[str],
    target_col: str,
    *,
    save: bool = False,
) -> Dict[str, Any]:
    """
    Run placebo tests:
    1. Transit-only sample
    2. Homebound sample
    3. Shuffled agent labels
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    results = {}

    # 1. Transit placebo
    if "transit_flag" in df.columns:
        transit = df[df["transit_flag"] == 1].copy()
        if len(transit) > 100 and transit[target_col].nunique() > 1:
            results["transit"] = _fit_placebo(transit, features, target_col, "transit")
        else:
            results["transit"] = {"auc": np.nan, "note": "insufficient_transit_data"}

    # 2. Homebound placebo
    if "homebound_flag" in df.columns:
        homebound = df[df["homebound_flag"] == 1].copy()
        if len(homebound) > 100 and homebound[target_col].nunique() > 1:
            results["homebound"] = _fit_placebo(homebound, features, target_col, "homebound")
        else:
            results["homebound"] = {"auc": np.nan, "note": "insufficient_homebound_data"}

    # 3. Shuffled agent labels
    df_shuffled = df.copy()
    if "psi_hat_holdout" in df_shuffled.columns:
        rng = np.random.RandomState(ML_CFG.random_seed)
        # Shuffle within ground-year
        if "ground_id" in df_shuffled.columns and "year" in df_shuffled.columns:
            for _, grp in df_shuffled.groupby(["ground_id", "year"]):
                idx = grp.index
                df_shuffled.loc[idx, "psi_hat_holdout"] = rng.permutation(
                    df_shuffled.loc[idx, "psi_hat_holdout"].values
                )
        else:
            df_shuffled["psi_hat_holdout"] = rng.permutation(
                df_shuffled["psi_hat_holdout"].values
            )
        results["shuffled_agent"] = _fit_placebo(
            df_shuffled, features, target_col, "shuffled_agent"
        )

    if save and results:
        placebo_df = pd.DataFrame([
            {"placebo": k, **{kk: vv for kk, vv in v.items() if not isinstance(vv, str)}}
            for k, v in results.items()
        ])
        placebo_df.to_csv(ML_TABLES_DIR / "survival_placebos.csv", index=False)

    return results


def _fit_placebo(df, features, target_col, label):
    """Fit a quick model on placebo data and return AUC."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    avail = [f for f in features if f in df.columns]
    X = df[avail].fillna(0).values
    y = df[target_col].astype(int).values

    if len(np.unique(y)) < 2:
        return {"auc": np.nan, "note": "single_class"}

    split = int(0.7 * len(X))
    X_tr, y_tr = X[:split], y[:split]
    X_te, y_te = X[split:], y[split:]

    try:
        model = HistGradientBoostingClassifier(
            max_iter=100, max_depth=4, random_state=ML_CFG.random_seed,
        )
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)
        metrics = classification_metrics(y_te, proba)
        return {**metrics, "label": label, "n": len(df)}
    except Exception as e:
        return {"auc": np.nan, "note": str(e)}


# ═══════════════════════════════════════════════════════════════════════════
# Main Entry
# ═══════════════════════════════════════════════════════════════════════════

def run_survival_ml(*, save_outputs: bool = True) -> Dict[str, Any]:
    """Run the full survival ML pipeline."""
    logger.info("=" * 60)
    logger.info("Phase ML-3: Nonlinear Survival Models")
    logger.info("=" * 60)
    return run_survival_models(save_outputs=save_outputs)
