"""
ML Layer — Phase ML-1: Policy Learning (Map vs Compass).

Part A: Map model — ground-choice prediction (destination selection).
Part B: Compass model — within-ground next-action prediction (search policy).

Ablation ladder:
  1. Environment only
  2. Environment + captain
  3. Environment + agent
  4. Environment + captain + agent

Models: multinomial logit baseline, RF, HistGradientBoosting.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.ml.config import ML_CFG, ML_TABLES_DIR, ML_FIGURES_DIR, ML_MODELS_DIR
from src.ml.metrics import classification_metrics, regression_metrics
from src.ml.splits import split_rolling_time
from src.ml.baselines import fit_logistic_baseline, MajorityClassBaseline

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Feature Set Definitions
# ═══════════════════════════════════════════════════════════════════════════

ENVIRONMENT_FEATURES = [
    "year", "season_remaining", "voyage_day",
    "tonnage", "scarcity",
]

CAPTAIN_FEATURES = [
    "theta_hat_holdout", "captain_voyage_num",
]

AGENT_FEATURES = [
    "psi_hat_holdout",
]

STATE_FEATURES = [
    "speed", "move_length", "turn_angle", "net_displacement",
    "revisit_indicator", "days_since_last_success",
    "consecutive_empty_days", "days_in_ground",
]

ABLATION_LADDER = {
    "env_only": ENVIRONMENT_FEATURES,
    "env_captain": ENVIRONMENT_FEATURES + CAPTAIN_FEATURES,
    "env_agent": ENVIRONMENT_FEATURES + AGENT_FEATURES,
    "env_captain_agent": ENVIRONMENT_FEATURES + CAPTAIN_FEATURES + AGENT_FEATURES,
}


def _available_features(df: pd.DataFrame, feature_list: List[str]) -> List[str]:
    """Return only features that exist in the DataFrame."""
    return [f for f in feature_list if f in df.columns]


def _encode_categoricals(df: pd.DataFrame, features: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Label-encode any object/category columns in place."""
    from sklearn.preprocessing import LabelEncoder
    encoded_features = list(features)
    for f in features:
        if f in df.columns and df[f].dtype in ("object", "category"):
            le = LabelEncoder()
            df[f"_{f}_enc"] = le.fit_transform(df[f].astype(str))
            encoded_features = [f"_{f}_enc" if x == f else x for x in encoded_features]
    return df, encoded_features


# ═══════════════════════════════════════════════════════════════════════════
# Part A: Map Model (Ground Choice)
# ═══════════════════════════════════════════════════════════════════════════

def run_map_model(
    df: pd.DataFrame = None,
    *,
    save_outputs: bool = True,
) -> Dict[str, Any]:
    """
    Learn destination choice (ground selection) at the voyage/segment level.

    Target: ground_id chosen at voyage start or next ground after switch.

    Returns
    -------
    Dict with benchmark table and ablation results.
    """
    t0 = time.time()
    logger.info("Running Map model (ground-choice prediction)...")

    if df is None:
        from src.ml.build_action_dataset import build_action_dataset
        df = build_action_dataset()

    # ── Build ground-choice observations ────────────────────────────
    # Take first observation per voyage-ground spell as ground-choice event
    if "ground_id" not in df.columns or df["ground_id"].nunique() < 3:
        logger.warning("Insufficient ground data for map model")
        return {"error": "insufficient_ground_data"}

    # Ground choice = first day in each new ground spell per voyage
    df_sorted = df.sort_values(["voyage_id", "obs_date"])
    df_sorted["_prev_ground"] = df_sorted.groupby("voyage_id")["ground_id"].shift(1)
    ground_choices = df_sorted[
        (df_sorted["ground_id"] != df_sorted["_prev_ground"]) |
        df_sorted["_prev_ground"].isna()
    ].copy()
    ground_choices = ground_choices.dropna(subset=["ground_id"]).reset_index(drop=True)

    # Encode target
    from sklearn.preprocessing import LabelEncoder
    le_ground = LabelEncoder()
    ground_choices["_target"] = le_ground.fit_transform(ground_choices["ground_id"].astype(str))
    n_classes = len(le_ground.classes_)
    logger.info("Map model: %d ground-choice events, %d ground classes", len(ground_choices), n_classes)

    if n_classes < 2:
        return {"error": "too_few_ground_classes"}

    # ── Split ───────────────────────────────────────────────────────
    train_idx, val_idx, test_idx = split_rolling_time(ground_choices)

    # ── Run ablation ladder ─────────────────────────────────────────
    results = {}
    for ablation_name, feature_set in ABLATION_LADDER.items():
        features = _available_features(ground_choices, feature_set)
        if not features:
            continue

        gc = ground_choices.copy()
        gc, enc_features = _encode_categoricals(gc, features)

        X = gc[enc_features].fillna(0).values
        y = gc["_target"].values

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        ablation_results = _fit_model_suite(
            X_train, y_train, X_val, y_val, X_test, y_test,
            task="classification",
            n_classes=n_classes,
            label=f"map_{ablation_name}",
        )
        ablation_results["features"] = enc_features
        ablation_results["n_features"] = len(enc_features)
        results[ablation_name] = ablation_results

    # ── Build benchmark table ───────────────────────────────────────
    benchmark = _build_benchmark_table(results, "map")

    if save_outputs:
        benchmark.to_csv(ML_TABLES_DIR / "policy_map_benchmark.csv", index=False)
        _plot_contribution(results, "map", save=True)

    elapsed = time.time() - t0
    logger.info("Map model complete in %.1fs", elapsed)

    return {
        "benchmark": benchmark,
        "ablation_results": results,
        "n_ground_classes": n_classes,
        "n_observations": len(ground_choices),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Part B: Compass Model (Within-Ground Actions)
# ═══════════════════════════════════════════════════════════════════════════

def run_compass_model(
    df: pd.DataFrame = None,
    *,
    save_outputs: bool = True,
) -> Dict[str, Any]:
    """
    Learn within-ground next-action policy at the day level.

    Targets: next_action_class (5-way), exit_patch_next, next_move_length,
    next_turn_angle.

    Restricts to active_search_flag == 1.

    Returns
    -------
    Dict with benchmark tables and ablation results per target.
    """
    t0 = time.time()
    logger.info("Running Compass model (within-ground action prediction)...")

    if df is None:
        from src.ml.build_action_dataset import build_action_dataset
        df = build_action_dataset()

    # Restrict to active search
    if "active_search_flag" in df.columns:
        df = df[df["active_search_flag"] == 1].copy()

    # ── Classification targets ──────────────────────────────────────
    targets_cls = {}
    if "next_action_class" in df.columns:
        valid = df["next_action_class"].notna()
        targets_cls["next_action_class"] = df.loc[valid].reset_index(drop=True).copy()
    if "exit_patch_next" in df.columns:
        valid = df["exit_patch_next"].notna()
        targets_cls["exit_patch_next"] = df.loc[valid].reset_index(drop=True).copy()

    # ── Regression targets ──────────────────────────────────────────
    targets_reg = {}
    for col in ["next_move_length", "next_turn_angle"]:
        if col in df.columns:
            valid = df[col].notna()
            targets_reg[col] = df.loc[valid].reset_index(drop=True).copy()

    compass_features = ENVIRONMENT_FEATURES + STATE_FEATURES

    COMPASS_ABLATION = {
        "env_state": compass_features,
        "env_state_captain": compass_features + CAPTAIN_FEATURES,
        "env_state_agent": compass_features + AGENT_FEATURES,
        "env_state_captain_agent": compass_features + CAPTAIN_FEATURES + AGENT_FEATURES,
        "env_state_captain_agent_types": compass_features + CAPTAIN_FEATURES + AGENT_FEATURES + ["theta_hat_holdout", "psi_hat_holdout"],
    }

    all_results = {}

    # ── Classification targets ──────────────────────────────────────
    for target_name, target_df in targets_cls.items():
        logger.info("Compass model: target=%s, n=%d", target_name, len(target_df))

        target_df["_target"] = target_df[target_name].astype(int)
        n_classes = int(target_df["_target"].nunique())

        train_idx, val_idx, test_idx = split_rolling_time(target_df)

        target_results = {}
        for abl_name, feature_set in COMPASS_ABLATION.items():
            features = _available_features(target_df, feature_set)
            # Deduplicate
            features = list(dict.fromkeys(features))
            if not features:
                continue

            tdf = target_df.copy()
            tdf, enc_features = _encode_categoricals(tdf, features)

            X = tdf[enc_features].fillna(0).values
            y = tdf["_target"].values

            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            abl_res = _fit_model_suite(
                X_train, y_train, X_val, y_val, X_test, y_test,
                task="classification",
                n_classes=n_classes,
                label=f"compass_{target_name}_{abl_name}",
            )
            abl_res["features"] = enc_features
            target_results[abl_name] = abl_res

        all_results[target_name] = target_results

    # ── Regression targets ──────────────────────────────────────────
    for target_name, target_df in targets_reg.items():
        logger.info("Compass model (regression): target=%s, n=%d", target_name, len(target_df))

        train_idx, val_idx, test_idx = split_rolling_time(target_df)

        target_results = {}
        for abl_name, feature_set in COMPASS_ABLATION.items():
            features = _available_features(target_df, feature_set)
            features = list(dict.fromkeys(features))
            if not features:
                continue

            tdf = target_df.copy()
            tdf, enc_features = _encode_categoricals(tdf, features)

            X = tdf[enc_features].fillna(0).values
            y = tdf[target_name].values

            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            abl_res = _fit_model_suite(
                X_train, y_train, X_val, y_val, X_test, y_test,
                task="regression",
                label=f"compass_{target_name}_{abl_name}",
            )
            abl_res["features"] = enc_features
            target_results[abl_name] = abl_res

        all_results[target_name] = target_results

    # ── Build benchmark table ───────────────────────────────────────
    benchmarks = {}
    for target_name, target_results in all_results.items():
        benchmark = _build_benchmark_table(target_results, f"compass_{target_name}")
        benchmarks[target_name] = benchmark

    if save_outputs:
        # Save combined benchmark
        all_benchmarks = pd.concat(
            [b.assign(target=t) for t, b in benchmarks.items()],
            ignore_index=True,
        )
        all_benchmarks.to_csv(ML_TABLES_DIR / "policy_compass_benchmark.csv", index=False)

        _plot_contribution(
            all_results.get("next_action_class", {}),
            "compass",
            save=True,
        )

    elapsed = time.time() - t0
    logger.info("Compass model complete in %.1fs", elapsed)

    return {
        "benchmarks": benchmarks,
        "all_results": all_results,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Counterfactual Identity Swap
# ═══════════════════════════════════════════════════════════════════════════

def counterfactual_identity_swap(
    df: pd.DataFrame,
    model: Any,
    feature_names: List[str],
    *,
    n_samples: int = 1000,
    seed: int = None,
) -> Dict[str, Any]:
    """
    Counterfactual identity-swap diagnostic.

    Hold state fixed, swap captain or agent identifiers, compare predictions.

    Returns
    -------
    Dict with swap effects for captain and agent.
    """
    seed = seed or ML_CFG.random_seed
    rng = np.random.RandomState(seed)

    df_sample = df.sample(min(n_samples, len(df)), random_state=seed)

    results = {}

    # Captain swap: replace captain features with random other captain
    for swap_type, swap_cols in [
        ("captain", ["theta_hat_holdout", "captain_voyage_num"]),
        ("agent", ["psi_hat_holdout"]),
    ]:
        available_swap = [c for c in swap_cols if c in feature_names and c in df.columns]
        if not available_swap:
            continue

        X_original = df_sample[feature_names].fillna(0).values

        # Shuffle swap columns
        X_swapped = X_original.copy()
        for col in available_swap:
            col_idx = feature_names.index(col)
            X_swapped[:, col_idx] = rng.permutation(X_swapped[:, col_idx])

        # Get predictions
        if hasattr(model, "predict_proba"):
            pred_orig = model.predict_proba(X_original)
            pred_swap = model.predict_proba(X_swapped)
            # KL divergence approximation
            eps = 1e-10
            kl = np.mean(np.sum(
                pred_orig * np.log((pred_orig + eps) / (pred_swap + eps)),
                axis=1,
            ))
            # Total variation
            tv = np.mean(np.sum(np.abs(pred_orig - pred_swap), axis=1)) / 2
        else:
            pred_orig = model.predict(X_original)
            pred_swap = model.predict(X_swapped)
            kl = np.nan
            tv = np.mean(np.abs(pred_orig - pred_swap))

        results[swap_type] = {
            "kl_divergence": float(kl),
            "total_variation": float(tv),
            "n_samples": len(df_sample),
        }

    logger.info("Identity swap: captain TV=%.4f, agent TV=%.4f",
                results.get("captain", {}).get("total_variation", 0),
                results.get("agent", {}).get("total_variation", 0))

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Internal: Model Suite
# ═══════════════════════════════════════════════════════════════════════════

def _fit_model_suite(
    X_train, y_train, X_val, y_val, X_test, y_test,
    *,
    task: str = "classification",
    n_classes: int = 2,
    label: str = "",
) -> Dict[str, Any]:
    """Fit baseline + HistGBT and evaluate."""
    from sklearn.ensemble import (
        HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    )

    results = {}

    models_to_fit = {}

    if task == "classification":
        # Baseline
        models_to_fit["majority_class"] = MajorityClassBaseline()
        try:
            models_to_fit["logistic"] = fit_logistic_baseline(X_train, y_train)
        except Exception as e:
            logger.warning("Logistic baseline failed for %s: %s", label, e)

        models_to_fit["hist_gradient_boosting"] = HistGradientBoostingClassifier(
            max_iter=ML_CFG.n_estimators,
            max_depth=ML_CFG.max_depth,
            learning_rate=ML_CFG.learning_rate,
            min_samples_leaf=ML_CFG.min_samples_leaf,
            random_state=ML_CFG.random_seed,
        )
    else:
        from src.ml.baselines import MeanBaseline, fit_linear_baseline
        models_to_fit["mean"] = MeanBaseline()
        try:
            models_to_fit["linear"] = fit_linear_baseline(X_train, y_train)
        except Exception as e:
            logger.warning("Linear baseline failed for %s: %s", label, e)

        models_to_fit["hist_gradient_boosting"] = HistGradientBoostingRegressor(
            max_iter=ML_CFG.n_estimators,
            max_depth=ML_CFG.max_depth,
            learning_rate=ML_CFG.learning_rate,
            min_samples_leaf=ML_CFG.min_samples_leaf,
            random_state=ML_CFG.random_seed,
        )

    for model_name, model in models_to_fit.items():
        t0 = time.time()

        if isinstance(model, type):
            model = model()

        try:
            if not hasattr(model, "is_fitted_") or not model.is_fitted_ if hasattr(model, "is_fitted_") else True:
                if hasattr(model, "fit"):
                    model.fit(X_train, y_train)
        except Exception as e:
            logger.warning("Model %s failed for %s: %s", model_name, label, e)
            continue

        elapsed = time.time() - t0

        # Evaluate on validation and test
        for split_name, X_eval, y_eval in [("val", X_val, y_val), ("test", X_test, y_test)]:
            try:
                if task == "classification":
                    y_pred = model.predict(X_eval)
                    y_proba = model.predict_proba(X_eval) if hasattr(model, "predict_proba") else None
                    if y_proba is not None:
                        metrics = classification_metrics(y_eval, y_proba, y_pred)
                    else:
                        metrics = {"accuracy": float(np.mean(y_pred == y_eval))}
                else:
                    y_pred = model.predict(X_eval)
                    metrics = regression_metrics(y_eval, y_pred)

                results[f"{model_name}_{split_name}"] = {
                    **metrics,
                    "model": model_name,
                    "split": split_name,
                    "elapsed_sec": elapsed,
                }
            except Exception as e:
                logger.warning("Eval failed for %s/%s: %s", model_name, split_name, e)

        # Store fitted model reference
        results[f"{model_name}_model"] = model

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Internal: Benchmark Table
# ═══════════════════════════════════════════════════════════════════════════

def _build_benchmark_table(
    ablation_results: Dict[str, Dict],
    prefix: str,
) -> pd.DataFrame:
    """Build a comparison table from ablation results."""
    rows = []
    for abl_name, abl_res in ablation_results.items():
        for key, val in abl_res.items():
            if isinstance(val, dict) and "model" in val and "split" in val:
                row = {"ablation": abl_name, **val}
                rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["task"] = prefix
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Internal: Contribution Plots
# ═══════════════════════════════════════════════════════════════════════════

def _plot_contribution(
    results: Dict[str, Dict],
    model_type: str,
    *,
    save: bool = False,
):
    """
    Plot captain vs agent predictive contribution.

    Compares metrics across ablation steps to isolate each contributor.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # Extract best ML model's val metric across ablations
    metric_key = "log_loss" if model_type == "map" else "r_squared"
    higher_is_better = metric_key == "r_squared"

    abl_order = list(ABLATION_LADDER.keys())
    vals = []
    for abl in abl_order:
        if abl in results:
            # Find hist_gradient_boosting val metric
            for key, val in results[abl].items():
                if "hist_gradient_boosting_val" in key and isinstance(val, dict):
                    vals.append(val.get(metric_key, np.nan))
                    break
            else:
                vals.append(np.nan)
        else:
            vals.append(np.nan)

    if all(np.isnan(v) for v in vals):
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(abl_order))
    labels = ["Env Only", "+ Captain", "+ Agent", "+ Captain + Agent"][:len(abl_order)]

    ax.bar(x, vals, color=["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"][:len(vals)], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel(metric_key.replace("_", " ").title())
    ax.set_title(f"{model_type.title()} Model: Predictive Contribution")
    ax.grid(True, alpha=0.3, axis="y")

    for i, v in enumerate(vals):
        if not np.isnan(v):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()

    if save:
        path = ML_FIGURES_DIR / f"{model_type}_contribution.{ML_CFG.figure_format}"
        fig.savefig(path, dpi=ML_CFG.figure_dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Contribution plot saved to %s", path)
    else:
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def run_policy_learning(
    *,
    save_outputs: bool = True,
) -> Dict[str, Any]:
    """Run the full policy learning pipeline (Map + Compass)."""
    logger.info("=" * 60)
    logger.info("Phase ML-1: Policy Learning (Map vs Compass)")
    logger.info("=" * 60)

    from src.ml.build_action_dataset import build_action_dataset
    df = build_action_dataset()

    map_results = run_map_model(df, save_outputs=save_outputs)
    compass_results = run_compass_model(df, save_outputs=save_outputs)

    # Run identity swap on best compass model
    swap_results = {}
    if "all_results" in compass_results:
        nac_results = compass_results["all_results"].get("next_action_class", {})
        full_ablation = nac_results.get("env_state_captain_agent", {})
        best_model = full_ablation.get("hist_gradient_boosting_model")
        features = full_ablation.get("features", [])

        if best_model is not None and features:
            active_df = df[df.get("active_search_flag", pd.Series(1)) == 1].copy()
            swap_results = counterfactual_identity_swap(
                active_df, best_model, features,
            )

    return {
        "map": map_results,
        "compass": compass_results,
        "identity_swap": swap_results,
    }
