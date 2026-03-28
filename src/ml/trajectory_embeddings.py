"""
ML Layer — Appendix ML-6: Trajectory Embeddings & Motif Discovery.

Embed sequences of search actions using:
- PCA / UMAP of trajectory windows
- Optional autoencoder (if PyTorch available)
- K-means or HDBSCAN clustering of motifs

Links motifs to organizational capability for interpretive visualization.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.ml.config import ML_CFG, ML_TABLES_DIR, ML_FIGURES_DIR

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Feature Extraction
# ═══════════════════════════════════════════════════════════════════════════

TRAJECTORY_FEATURES = [
    "avg_speed", "var_speed",
    "avg_move_length", "var_move_length",
    "avg_turn_angle", "var_turn_angle",
    "revisit_rate",
    "time_since_success",
    "patch_residence",
    "local_loop_ratio",
]


def build_embeddings(
    df: pd.DataFrame = None,
    *,
    method: str = "pca",
    n_components: int = 3,
    save_outputs: bool = True,
) -> Dict[str, Any]:
    """
    Build trajectory embeddings from state dataset.

    Parameters
    ----------
    method : str
        'pca' or 'umap'
    """
    t0 = time.time()
    logger.info("Building trajectory embeddings (method=%s)...", method)

    if df is None:
        from src.ml.build_state_dataset import build_state_dataset
        df = build_state_dataset()

    features = [f for f in TRAJECTORY_FEATURES if f in df.columns]
    if len(features) < 3:
        return {"error": "insufficient_features"}

    X = df[features].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if method == "pca":
        model = PCA(n_components=n_components, random_state=ML_CFG.random_seed)
        embeddings = model.fit_transform(X_scaled)
        explained_var = model.explained_variance_ratio_
        logger.info("PCA explained variance: %s", explained_var)
    elif method == "umap":
        try:
            import umap
            model = umap.UMAP(
                n_components=n_components,
                random_state=ML_CFG.random_seed,
                n_neighbors=15,
                min_dist=0.1,
            )
            embeddings = model.fit_transform(X_scaled)
            explained_var = None
        except ImportError:
            logger.warning("umap-learn not installed; falling back to PCA")
            model = PCA(n_components=n_components, random_state=ML_CFG.random_seed)
            embeddings = model.fit_transform(X_scaled)
            explained_var = model.explained_variance_ratio_
            method = "pca"
    else:
        raise ValueError(f"Unknown method: {method}")

    # ── Cluster motifs ──────────────────────────────────────────────
    from sklearn.cluster import KMeans

    # Use inertia elbow (O(n)) instead of silhouette (O(n²)) to pick k.
    k_range = list(range(3, 8))
    inertias = []
    km_models = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=ML_CFG.random_seed, n_init=10)
        km.fit(embeddings)
        inertias.append(km.inertia_)
        km_models[k] = km

    # Elbow = k with largest inertia drop (second derivative)
    if len(inertias) >= 3:
        drops = [inertias[i] - inertias[i + 1] for i in range(len(inertias) - 1)]
        second_diff = [drops[i] - drops[i + 1] for i in range(len(drops) - 1)]
        best_k = k_range[np.argmax(second_diff) + 1]
    else:
        best_k = k_range[0]
    logger.info("Elbow selection: best k=%d (inertias: %s)", best_k,
                [f"{x:.0f}" for x in inertias])

    motif_labels = km_models[best_k].predict(embeddings)

    df_result = df.copy()
    for i in range(n_components):
        df_result[f"embed_{i}"] = embeddings[:, i]
    df_result["motif_id"] = motif_labels

    # ── Motif summary ───────────────────────────────────────────────
    motif_summary = df_result.groupby("motif_id")[features].mean()
    motif_summary["count"] = df_result.groupby("motif_id").size()
    motif_summary["share"] = motif_summary["count"] / len(df_result)

    if "psi_hat_holdout" in df_result.columns:
        motif_summary["mean_psi"] = df_result.groupby("motif_id")["psi_hat_holdout"].mean()

    if save_outputs:
        motif_summary.to_csv(ML_TABLES_DIR / "trajectory_motif_summary.csv")
        _plot_embeddings(embeddings, motif_labels, df_result, method, save=True)

    elapsed = time.time() - t0
    logger.info(
        "Trajectory embeddings: %d observations, %d motifs, %.1fs",
        len(embeddings), best_k, elapsed,
    )

    return {
        "embeddings": embeddings,
        "motif_labels": motif_labels,
        "motif_summary": motif_summary,
        "method": method,
        "explained_variance": explained_var,
        "n_motifs": best_k,
    }


def _plot_embeddings(embeddings, labels, df, method, *, save=False):
    """Plot 2D embedding scatter colored by motif."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Color by motif
    scatter = axes[0].scatter(
        embeddings[:, 0], embeddings[:, 1],
        c=labels, cmap="Set1", alpha=0.3, s=5,
    )
    axes[0].set_xlabel(f"{method.upper()} 1")
    axes[0].set_ylabel(f"{method.upper()} 2")
    axes[0].set_title("Trajectory Embeddings by Motif")
    fig.colorbar(scatter, ax=axes[0], label="Motif ID")

    # Color by psi
    if "psi_hat_holdout" in df.columns:
        psi = df["psi_hat_holdout"].fillna(0).values
        scatter2 = axes[1].scatter(
            embeddings[:, 0], embeddings[:, 1],
            c=psi, cmap="coolwarm", alpha=0.3, s=5,
        )
        axes[1].set_xlabel(f"{method.upper()} 1")
        axes[1].set_ylabel(f"{method.upper()} 2")
        axes[1].set_title("Trajectory Embeddings by Agent Capability (ψ)")
        fig.colorbar(scatter2, ax=axes[1], label="ψ")
    else:
        axes[1].text(0.5, 0.5, "No ψ data", ha="center", va="center",
                    transform=axes[1].transAxes)

    fig.tight_layout()

    if save:
        path = ML_FIGURES_DIR / f"trajectory_embeddings.{ML_CFG.figure_format}"
        fig.savefig(path, dpi=ML_CFG.figure_dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.close(fig)
