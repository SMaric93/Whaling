"""
ML Layer — Phase ML-2: Latent Search State Models.

Infer latent movement/search states and test whether organizations
change transition logic.

Models:
1. Gaussian Mixture Model on window features
2. Hidden Markov Model with 3-5 states
3. Optional hidden semi-Markov if tractable

State labels assigned post-hoc by observable statistics.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.ml.config import ML_CFG, ML_TABLES_DIR, ML_FIGURES_DIR, ML_MODELS_DIR

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Feature Selection for State Models
# ═══════════════════════════════════════════════════════════════════════════

STATE_FEATURES = [
    "avg_speed", "var_speed",
    "avg_move_length", "var_move_length",
    "avg_turn_angle", "var_turn_angle",
    "revisit_rate",
    "time_since_success",
    "patch_residence",
    "max_empty_streak_window",
]


def _prepare_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str], StandardScaler]:
    """Prepare and standardize features for state models."""
    available = [f for f in STATE_FEATURES if f in df.columns]
    if not available:
        raise ValueError("No state features available")

    X = df[available].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, available, scaler


# ═══════════════════════════════════════════════════════════════════════════
# GMM State Discovery
# ═══════════════════════════════════════════════════════════════════════════

def fit_gmm_states(
    df: pd.DataFrame,
    *,
    n_components_range: List[int] = None,
) -> Dict[str, Any]:
    """
    Fit Gaussian Mixture Model on trajectory window features.

    Selects best n_components by BIC.
    """
    from sklearn.mixture import GaussianMixture

    n_components_range = n_components_range or ML_CFG.hmm_n_states_range

    X, feature_names, scaler = _prepare_features(df)

    best_bic = np.inf
    best_model = None
    best_k = None
    bic_results = []

    for k in n_components_range:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            n_init=5,
            max_iter=300,
            random_state=ML_CFG.random_seed,
        )
        gmm.fit(X)
        bic = gmm.bic(X)
        bic_results.append({"n_components": k, "bic": bic, "log_likelihood": gmm.score(X) * len(X)})
        logger.info("GMM k=%d: BIC=%.1f", k, bic)

        if bic < best_bic:
            best_bic = bic
            best_model = gmm
            best_k = k

    labels = best_model.predict(X)
    probs = best_model.predict_proba(X)

    logger.info("Best GMM: k=%d, BIC=%.1f", best_k, best_bic)

    return {
        "model": best_model,
        "labels": labels,
        "probs": probs,
        "best_k": best_k,
        "bic_results": pd.DataFrame(bic_results),
        "feature_names": feature_names,
        "scaler": scaler,
    }


# ═══════════════════════════════════════════════════════════════════════════
# HMM State Discovery
# ═══════════════════════════════════════════════════════════════════════════

def fit_hmm_states(
    df: pd.DataFrame,
    *,
    n_states_range: List[int] = None,
    voyage_col: str = "voyage_id",
) -> Dict[str, Any]:
    """
    Fit Hidden Markov Model on trajectory sequences.

    Fits separate sequences per voyage, selects best n_states by BIC.
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        logger.warning("hmmlearn not installed; skipping HMM")
        return {"error": "hmmlearn_not_available"}

    n_states_range = n_states_range or ML_CFG.hmm_n_states_range

    X, feature_names, scaler = _prepare_features(df)

    # Build sequence lengths per voyage
    if voyage_col in df.columns:
        lengths = df.groupby(voyage_col).size().values
        # Ensure lengths sum to total
        assert lengths.sum() == len(X), "Length mismatch"
    else:
        lengths = [len(X)]

    best_bic = np.inf
    best_model = None
    best_k = None
    results_list = []

    for k in n_states_range:
        try:
            hmm = GaussianHMM(
                n_components=k,
                covariance_type="diag",
                n_iter=ML_CFG.hmm_n_iter,
                tol=ML_CFG.hmm_tol,
                random_state=ML_CFG.random_seed,
            )
            hmm.fit(X, lengths)

            ll = hmm.score(X, lengths)
            n_params = k * k + k * X.shape[1] * 2  # transition + means + vars
            bic = -2 * ll + n_params * np.log(len(X))
            results_list.append({"n_states": k, "bic": bic, "log_likelihood": ll})
            logger.info("HMM k=%d: BIC=%.1f, LL=%.1f", k, bic, ll)

            if bic < best_bic:
                best_bic = bic
                best_model = hmm
                best_k = k
        except Exception as e:
            logger.warning("HMM k=%d failed: %s", k, e)

    if best_model is None:
        return {"error": "all_hmm_fits_failed"}

    # Decode states
    labels = best_model.predict(X, lengths)
    posteriors = best_model.predict_proba(X, lengths)

    logger.info("Best HMM: k=%d, BIC=%.1f", best_k, best_bic)

    return {
        "model": best_model,
        "labels": labels,
        "posteriors": posteriors,
        "best_k": best_k,
        "bic_results": pd.DataFrame(results_list),
        "transition_matrix": best_model.transmat_,
        "feature_names": feature_names,
        "scaler": scaler,
    }


# ═══════════════════════════════════════════════════════════════════════════
# State Labeling
# ═══════════════════════════════════════════════════════════════════════════

def label_states(
    df: pd.DataFrame,
    labels: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Assign interpretable labels to discovered states.

    Uses observable statistics of each cluster to suggest names:
    - transit: high speed, low revisit
    - local_exploration: medium speed, high turn variance
    - exploitation: low speed, high revisit, productive
    - barren_search: low speed, high turn variance, low productivity
    - homebound: high net displacement, near end of season
    """
    df = df.copy()
    df["state_id"] = labels

    # Compute cluster centroids
    summary_cols = [f for f in feature_names if f in df.columns]
    centroids = df.groupby("state_id")[summary_cols].mean()

    # Rule-based labeling
    state_labels = {}
    for sid in centroids.index:
        c = centroids.loc[sid]

        speed = c.get("avg_speed", 0)
        revisit = c.get("revisit_rate", 0)
        turn_var = c.get("var_turn_angle", 0)
        empty = c.get("max_empty_streak_window", 0)

        speed_rank = centroids["avg_speed"].rank(pct=True).get(sid, 0.5) if "avg_speed" in centroids.columns else 0.5
        revisit_rank = centroids["revisit_rate"].rank(pct=True).get(sid, 0.5) if "revisit_rate" in centroids.columns else 0.5

        if speed_rank > 0.7 and revisit_rank < 0.3:
            state_labels[sid] = "transit"
        elif revisit_rank > 0.7 and empty < centroids["max_empty_streak_window"].median() if "max_empty_streak_window" in centroids.columns else True:
            state_labels[sid] = "exploitation"
        elif speed_rank < 0.4 and revisit_rank < 0.5:
            state_labels[sid] = "local_exploration"
        elif empty > centroids["max_empty_streak_window"].median() if "max_empty_streak_window" in centroids.columns else False:
            state_labels[sid] = "barren_search"
        else:
            state_labels[sid] = f"state_{sid}"

    df["state_label"] = df["state_id"].map(state_labels)

    state_summary = centroids.copy()
    state_summary["label"] = pd.Series(state_labels)
    state_summary["count"] = df.groupby("state_id").size()
    state_summary["share"] = state_summary["count"] / len(df)

    return df, state_summary


# ═══════════════════════════════════════════════════════════════════════════
# PSI-Conditional Analysis
# ═══════════════════════════════════════════════════════════════════════════

def analyze_psi_state_effects(
    df: pd.DataFrame,
    *,
    psi_col: str = "psi_hat_holdout",
    n_groups: int = 4,
) -> Dict[str, pd.DataFrame]:
    """
    Analyze how psi affects state occupancy and transitions.

    Returns
    -------
    Dict with:
    - occupancy_by_psi: state share by psi quartile
    - transition_by_psi: transition matrix by psi group
    """
    if psi_col not in df.columns or "state_id" not in df.columns:
        return {"error": "missing_psi_or_state_columns"}

    df = df.copy()
    df["psi_group"] = pd.qcut(
        df[psi_col].rank(method="first"),
        q=n_groups,
        labels=[f"Q{i+1}" for i in range(n_groups)],
    )

    # State occupancy by psi
    occupancy = df.groupby(["psi_group", "state_id"]).size().unstack(fill_value=0)
    occupancy_pct = occupancy.div(occupancy.sum(axis=1), axis=0)

    # Also by state_label if available
    if "state_label" in df.columns:
        occ_label = df.groupby(["psi_group", "state_label"]).size().unstack(fill_value=0)
        occ_label_pct = occ_label.div(occ_label.sum(axis=1), axis=0)
    else:
        occ_label_pct = occupancy_pct

    # Transition probabilities by psi group
    df["_next_state"] = df.groupby("voyage_id")["state_id"].shift(-1)
    transitions = {}
    for pg in df["psi_group"].unique():
        sub = df[df["psi_group"] == pg].dropna(subset=["_next_state"])
        trans = pd.crosstab(sub["state_id"], sub["_next_state"], normalize="index")
        transitions[pg] = trans

    return {
        "occupancy_by_psi": occ_label_pct,
        "transition_by_psi": transitions,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

def plot_state_transitions(
    transitions: Dict[str, pd.DataFrame],
    *,
    save_path: str = None,
):
    """Plot transition matrix heatmaps by psi group."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
    except ImportError:
        return

    n_groups = len(transitions)
    fig, axes = plt.subplots(1, n_groups, figsize=(5 * n_groups, 4), squeeze=False)

    for idx, (group_name, trans_mat) in enumerate(transitions.items()):
        ax = axes[0, idx]
        im = ax.imshow(trans_mat.values, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
        ax.set_title(f"Psi {group_name}")
        ax.set_xlabel("To State")
        ax.set_ylabel("From State")
        ax.set_xticks(range(len(trans_mat.columns)))
        ax.set_yticks(range(len(trans_mat.index)))
        ax.set_xticklabels(trans_mat.columns, fontsize=8)
        ax.set_yticklabels(trans_mat.index, fontsize=8)

        # Annotate cells
        for i in range(len(trans_mat.index)):
            for j in range(len(trans_mat.columns)):
                val = trans_mat.values[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                       fontsize=7, color="white" if val > 0.5 else "black")

    fig.suptitle("State Transition Matrices by Agent Capability (ψ)", fontsize=12)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=ML_CFG.figure_dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def run_state_models(
    *,
    save_outputs: bool = True,
) -> Dict[str, Any]:
    """Run the full latent state model pipeline."""
    logger.info("=" * 60)
    logger.info("Phase ML-2: Latent Search State Models")
    logger.info("=" * 60)

    from src.ml.build_state_dataset import build_state_dataset
    df = build_state_dataset()

    # ── Fit GMM ─────────────────────────────────────────────────────
    gmm_results = fit_gmm_states(df)

    # ── Fit HMM ─────────────────────────────────────────────────────
    hmm_results = fit_hmm_states(df)

    # ── Label states (prefer HMM if available) ──────────────────────
    best_results = hmm_results if "labels" in hmm_results else gmm_results
    if "labels" in best_results:
        X_scaled, feat_names, _ = _prepare_features(df)
        df_labeled, state_summary = label_states(df, best_results["labels"], feat_names)

        # ── PSI effects ─────────────────────────────────────────────
        psi_effects = analyze_psi_state_effects(df_labeled)

        if save_outputs:
            state_summary.to_csv(ML_TABLES_DIR / "state_summary.csv")

            if "transition_by_psi" in psi_effects:
                plot_state_transitions(
                    psi_effects["transition_by_psi"],
                    save_path=str(ML_FIGURES_DIR / f"state_transitions_by_psi.{ML_CFG.figure_format}"),
                )

            if "occupancy_by_psi" in psi_effects:
                psi_effects["occupancy_by_psi"].to_csv(
                    ML_TABLES_DIR / "state_occupancy_by_psi.csv"
                )
    else:
        df_labeled = df
        state_summary = pd.DataFrame()
        psi_effects = {}

    return {
        "gmm": gmm_results,
        "hmm": hmm_results,
        "state_summary": state_summary,
        "psi_effects": psi_effects,
    }
