"""
ML Layer — Interpretation Utilities.

Feature importance, permutation importance, PDP/ICE, SHAP wrapper,
and subgroup performance tables.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.ml.config import ML_CFG, ML_FIGURES_DIR, ML_TABLES_DIR

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Feature Importance
# ═══════════════════════════════════════════════════════════════════════════

def feature_importance(
    model: Any,
    feature_names: List[str],
    *,
    importance_type: str = "auto",
) -> pd.DataFrame:
    """
    Extract feature importance from a fitted model.

    Supports sklearn tree models, linear models with coef_.
    """
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_).flatten()
        if len(imp) != len(feature_names):
            imp = np.abs(model.coef_).mean(axis=0)
    else:
        logger.warning("Model has no feature_importances_ or coef_")
        imp = np.zeros(len(feature_names))

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": imp,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    df["importance_pct"] = 100 * df["importance"] / max(df["importance"].sum(), 1e-12)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Permutation Importance
# ═══════════════════════════════════════════════════════════════════════════

def permutation_importance(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    *,
    scoring: str = "neg_mean_squared_error",
    n_repeats: int = None,
    seed: int = None,
) -> pd.DataFrame:
    """
    Permutation importance using sklearn.

    Returns DataFrame with feature, importance_mean, importance_std.
    """
    from sklearn.inspection import permutation_importance as _perm_imp

    n_repeats = n_repeats or ML_CFG.n_permutation_repeats
    seed = seed or ML_CFG.random_seed

    result = _perm_imp(
        model, X, y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=-1,
    )

    df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    return df


# ═══════════════════════════════════════════════════════════════════════════
# Partial Dependence / ICE
# ═══════════════════════════════════════════════════════════════════════════

def partial_dependence_plot(
    model: Any,
    X: pd.DataFrame,
    feature: str,
    *,
    kind: str = "both",  # "average", "individual", "both"
    grid_resolution: int = None,
    save_path: str = None,
) -> Dict:
    """
    Compute and optionally plot partial dependence.

    Returns dict with grid_values, pd_values.
    """
    grid_resolution = grid_resolution or ML_CFG.pdp_grid_resolution

    try:
        from sklearn.inspection import partial_dependence
        result = partial_dependence(
            model, X, features=[feature],
            grid_resolution=grid_resolution,
            kind="average",
        )
        grid_values = result["grid_values"][0]  # type: ignore
        pd_values = result["average"][0]  # type: ignore
    except Exception as e:
        logger.warning("PDP computation failed: %s", e)
        return {"grid_values": np.array([]), "pd_values": np.array([])}

    output = {
        "feature": feature,
        "grid_values": grid_values,
        "pd_values": pd_values,
    }

    if save_path:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(grid_values, pd_values, lw=2)
            ax.set_xlabel(feature)
            ax.set_ylabel("Partial Dependence")
            ax.set_title(f"PDP: {feature}")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(save_path, dpi=ML_CFG.figure_dpi, bbox_inches="tight")
            plt.close(fig)
            logger.info("PDP saved to %s", save_path)
        except ImportError:
            logger.warning("matplotlib not available for PDP plot")

    return output


# ═══════════════════════════════════════════════════════════════════════════
# SHAP Wrapper
# ═══════════════════════════════════════════════════════════════════════════

def compute_shap_values(
    model: Any,
    X: pd.DataFrame,
    *,
    max_samples: int = None,
    save_prefix: str = None,
) -> Optional[Dict]:
    """
    Compute SHAP values (tree or kernel explainer).

    Returns dict with shap_values, feature_names, expected_value.
    Returns None if shap is not available.
    """
    max_samples = max_samples or ML_CFG.shap_max_samples

    try:
        import shap
    except ImportError:
        logger.warning("shap not installed; skipping SHAP computation")
        return None

    X_sample = X.sample(min(max_samples, len(X)), random_state=ML_CFG.random_seed)

    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        try:
            explainer = shap.KernelExplainer(
                model.predict, X_sample.iloc[:100]
            )
        except Exception as e:
            logger.warning("SHAP explainer creation failed: %s", e)
            return None

    shap_values = explainer.shap_values(X_sample)

    result = {
        "shap_values": shap_values,
        "feature_names": list(X.columns),
        "expected_value": explainer.expected_value,
        "X_sample": X_sample,
    }

    if save_prefix:
        try:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(10, 8))
            if isinstance(shap_values, list):
                shap.summary_plot(shap_values[0], X_sample, show=False)
            else:
                shap.summary_plot(shap_values, X_sample, show=False)
            plt.tight_layout()
            plt.savefig(
                f"{save_prefix}_shap_summary.{ML_CFG.figure_format}",
                dpi=ML_CFG.figure_dpi,
                bbox_inches="tight",
            )
            plt.close()
        except Exception as e:
            logger.warning("SHAP plot failed: %s", e)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Subgroup Performance Tables
# ═══════════════════════════════════════════════════════════════════════════

def subgroup_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: pd.Series,
    *,
    task: str = "regression",
    y_pred_proba: np.ndarray = None,
) -> pd.DataFrame:
    """
    Compute performance metrics by subgroup.

    Parameters
    ----------
    groups : pd.Series
        Group labels (e.g., novice/expert, high/low psi).

    Returns
    -------
    DataFrame with one row per group, metrics as columns.
    """
    from src.ml.metrics import evaluate_model

    # Ensure groups has a 0-based index so .groups returns positional indices
    groups = pd.Series(groups).reset_index(drop=True)

    rows = []
    for group_name, mask in groups.groupby(groups).groups.items():
        mask_arr = np.array(list(mask))
        yt = np.asarray(y_true)[mask_arr]
        yp = np.asarray(y_pred)[mask_arr]

        kwargs = {"task": task}
        if task == "classification" and y_pred_proba is not None:
            kwargs["y_pred_proba"] = np.asarray(y_pred_proba)[mask_arr]

        try:
            metrics = evaluate_model(yt, yp, **kwargs)
        except Exception:
            metrics = {}

        metrics["group"] = group_name
        metrics["n"] = len(yt)
        rows.append(metrics)

    return pd.DataFrame(rows).set_index("group")


def standard_subgroup_labels(
    df: pd.DataFrame,
    *,
    psi_col: str = "psi_hat_holdout",
    theta_col: str = "theta_hat_holdout",
    novice_col: str = "novice",
    scarcity_col: str = "scarcity",
) -> Dict[str, pd.Series]:
    """
    Create standard subgroup label Series for reporting.

    Returns dict mapping label name to pd.Series of group labels.
    """
    labels = {}

    # Novice vs expert
    if novice_col in df.columns:
        labels["experience"] = df[novice_col].map({True: "novice", False: "expert", 1: "novice", 0: "expert"})
    elif "captain_experience" in df.columns:
        exp = df["captain_experience"]
        labels["experience"] = pd.cut(
            exp,
            bins=[0, ML_CFG.experience_bins["novice_max"], ML_CFG.experience_bins["expert_min"], exp.max() + 1],
            labels=["novice", "mid", "expert"],
        )

    # High vs low psi
    if psi_col in df.columns:
        labels["psi_group"] = pd.qcut(
            df[psi_col].rank(method="first"),
            q=ML_CFG.n_psi_quartiles,
            labels=[f"psi_Q{i+1}" for i in range(ML_CFG.n_psi_quartiles)],
        )

    # Sparse vs rich
    if scarcity_col in df.columns:
        if df[scarcity_col].dtype in ("object", "category"):
            labels["scarcity"] = df[scarcity_col]
        else:
            labels["scarcity"] = pd.qcut(
                df[scarcity_col].rank(method="first"),
                q=ML_CFG.n_scarcity_bins,
                labels=["sparse", "medium", "rich"],
            )

    # Movers vs stayers
    if "switch_agent" in df.columns:
        labels["mover"] = df["switch_agent"].map({1: "mover", 0: "stayer"}).fillna("stayer")

    return labels
