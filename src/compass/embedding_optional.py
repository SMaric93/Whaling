"""
Compass Pipeline — Self-Supervised Embedding (Step 8, Optional).

Trains a lightweight 1-D CNN encoder on fixed-length segments of
search-regime steps using masked-feature prediction.  Produces a
per-voyage embedding vector ``z`` and a scalar ``DLCompassScore``
by probing with CompassIndex1.

**Gated on torch availability** — the rest of the pipeline works
without it.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from compass.config import CompassConfig

logger = logging.getLogger(__name__)


def _torch_available() -> bool:
    try:
        import torch
        return True
    except ImportError:
        return False


# ── feature columns for the encoder ────────────────────────────────────────

_STEP_FEATURES = [
    "step_length_m", "speed_mps", "heading_rad", "turning_angle_rad",
]


# ── dataset ─────────────────────────────────────────────────────────────────

def _build_segments(
    steps_df: pd.DataFrame,
    segment_length: int,
) -> Tuple[np.ndarray, list[str]]:
    """
    Build (n_segments, segment_length, n_features) array.

    Pads short voyages and truncates long ones.
    Returns segment array and list of voyage_ids.
    """
    search = steps_df.loc[steps_df["regime_label"] == "search"]
    segments: list[np.ndarray] = []
    vids: list[str] = []

    for vid, sub in search.groupby("voyage_id", sort=False):
        X = sub[_STEP_FEATURES].values.astype("float32")
        X = np.nan_to_num(X, nan=0.0)

        # split into chunks
        for start in range(0, max(len(X), 1), segment_length):
            chunk = X[start: start + segment_length]
            if len(chunk) < segment_length:
                pad = np.zeros((segment_length - len(chunk), X.shape[1]), dtype="float32")
                chunk = np.concatenate([chunk, pad], axis=0)
            segments.append(chunk)
            vids.append(vid)

    if not segments:
        return np.empty((0, segment_length, len(_STEP_FEATURES)), dtype="float32"), []

    return np.stack(segments), vids


# ── model ───────────────────────────────────────────────────────────────────

def _build_encoder(n_features: int, embed_dim: int):
    """1D-CNN encoder → fixed-length embedding."""
    import torch
    import torch.nn as nn

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(n_features, 32, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.fc = nn.Linear(64, embed_dim)

        def forward(self, x):
            # x: (B, T, F) → (B, F, T) for Conv1d
            x = x.permute(0, 2, 1)
            x = self.conv(x).squeeze(-1)
            return self.fc(x)

    return Encoder()


def _build_decoder(embed_dim: int, n_features: int, seq_len: int):
    """Simple linear decoder for masked prediction."""
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(embed_dim, 128),
        nn.ReLU(),
        nn.Linear(128, seq_len * n_features),
    )


# ── training ────────────────────────────────────────────────────────────────

def train_embedding(
    steps_df: pd.DataFrame,
    cfg: CompassConfig,
) -> Tuple[object, np.ndarray, list[str]]:
    """
    Train self-supervised embedding and return (model, embeddings, vids).

    The model uses masked-feature prediction:
    - Randomly mask 15% of time-steps in each segment.
    - Encoder produces embedding z.
    - Decoder reconstructs masked features from z.
    - Loss = MSE on masked positions.

    Returns
    -------
    (encoder, embeddings_array, voyage_ids)
    """
    if not _torch_available():
        raise ImportError("torch is required for embedding training.")

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(cfg.embedding_random_state)
    np.random.seed(cfg.embedding_random_state)

    seg_arr, vids = _build_segments(steps_df, cfg.segment_length_steps)
    if len(seg_arr) == 0:
        logger.warning("No segments built — skipping embedding.")
        return None, np.empty((0, cfg.embedding_dim)), []

    n_features = seg_arr.shape[2]
    seg_len = seg_arr.shape[1]

    segments_t = torch.tensor(seg_arr, dtype=torch.float32)
    dataset = TensorDataset(segments_t)
    loader = DataLoader(
        dataset, batch_size=cfg.embedding_batch_size, shuffle=True,
    )

    encoder = _build_encoder(n_features, cfg.embedding_dim)
    decoder = _build_decoder(cfg.embedding_dim, n_features, seg_len)
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.embedding_lr)
    loss_fn = nn.MSELoss()

    encoder.train()
    decoder.train()

    for epoch in range(cfg.embedding_epochs):
        total_loss = 0.0
        for (batch,) in loader:
            # mask 15% of timesteps
            mask = torch.rand(batch.shape[0], batch.shape[1]) < 0.15
            masked_input = batch.clone()
            masked_input[mask] = 0.0

            z = encoder(masked_input)
            recon = decoder(z).view(batch.shape)

            # loss only on masked positions
            loss = loss_fn(recon[mask], batch[mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.shape[0]

        if cfg.verbose and (epoch + 1) % 5 == 0:
            logger.info(
                "Embedding epoch %d/%d, loss=%.4f",
                epoch + 1, cfg.embedding_epochs, total_loss / len(segments_t),
            )

    # extract embeddings
    encoder.eval()
    with torch.no_grad():
        all_z = encoder(segments_t).numpy()

    # mean-pool per voyage
    vid_arr = np.array(vids)
    unique_vids = list(dict.fromkeys(vids))  # preserves order
    pooled = np.zeros((len(unique_vids), cfg.embedding_dim), dtype="float32")
    for i, v in enumerate(unique_vids):
        mask = vid_arr == v
        pooled[i] = all_z[mask].mean(axis=0)

    logger.info(
        "Trained embedding: %d segments → %d voyage embeddings (dim=%d).",
        len(seg_arr), len(unique_vids), cfg.embedding_dim,
    )
    return encoder, pooled, unique_vids


# ── probe ───────────────────────────────────────────────────────────────────

def probe_dl_score(
    embeddings: np.ndarray,
    voyage_ids: list[str],
    compass_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Linear probe: regress CompassIndex1 on embedding z → DLCompassScore.

    Returns a DataFrame with ``voyage_id``, ``DLCompassScore``, and the
    embedding columns ``z_0 … z_d``.
    """
    from sklearn.linear_model import Ridge

    emb_df = pd.DataFrame(
        embeddings,
        columns=[f"z_{i}" for i in range(embeddings.shape[1])],
    )
    emb_df["voyage_id"] = voyage_ids

    merged = emb_df.merge(
        compass_df[["voyage_id", "CompassIndex1"]].dropna(),
        on="voyage_id",
        how="inner",
    )

    z_cols = [c for c in emb_df.columns if c.startswith("z_")]
    X = merged[z_cols].values
    y = merged["CompassIndex1"].values

    model = Ridge(alpha=1.0)
    model.fit(X, y)
    preds = model.predict(emb_df[z_cols].values)

    emb_df["DLCompassScore"] = preds
    r2 = model.score(X, y)
    logger.info("Probe R² = %.3f.", r2)

    return emb_df
