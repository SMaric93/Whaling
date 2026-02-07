"""
Run the Full Compass Pipeline from Raw AOWL Logbook Data.

This script:
  1. Parses 468K daily AOWL observations (lat, lon, date, encounter, species, strikes)
  2. Converts to the compass pipeline's expected format (voyage_id, timestamp_utc, lat, lon)
  3. Enriches with encounter features (catch_event_flag, species channels)
  4. Runs the full 10-step compass pipeline: validation → projection → steps →
     HMM regime segmentation → compass features → PCA index → early window →
     self-supervised 1D-CNN embedding → robustness → econometric export
  5. Produces panel_voyage_compass.parquet for downstream regressions

Usage:
    python -m src.compass.run_compass_from_aowl
    python -m src.compass.run_compass_from_aowl --embedding --epochs 30
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── project paths ───────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_LOGBOOKS = DATA_DIR / "raw" / "logbooks"
STAGING_DIR = DATA_DIR / "staging"
FINAL_DIR = DATA_DIR / "final"
OUTPUT_DIR = PROJECT_ROOT / "output" / "compass"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 1: Enhanced AOWL Parser
# ═════════════════════════════════════════════════════════════════════════════

# Species groupings for feature channels
SPECIES_GROUPS = {
    "sperm": ["Sperm"],
    "right": ["Right"],
    "bowhead": ["Bowhead"],
    "humpback": ["Humpback"],
    "gray": ["Gray"],
    "other_baleen": ["Finback", "Blue", "Whale"],  # Whale = unspecified baleen
    "small_cetacean": ["Pilot", "Grampus", "Blackfish", "Porpoise", "Dolphin",
                       "Killer", "Killers"],
}


def _map_species_group(species: str) -> str:
    """Map raw species string to canonical group."""
    if pd.isna(species) or species in ("NULL", ""):
        return "none"
    for group, members in SPECIES_GROUPS.items():
        if species in members:
            return group
    return "other"


def parse_aowl_raw(raw_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Parse the raw AOWL tab-delimited file into a rich daily position dataset.

    Returns DataFrame with columns:
        voyage_id, timestamp_utc, lat, lon,
        encounter_type, species_raw, species_group,
        n_struck, n_tried, place_name, source, remarks,
        catch_event_flag, year, month, day
    """
    if raw_path is None:
        candidates = sorted(RAW_LOGBOOKS.glob("aowl_*.txt"))
        if not candidates:
            raise FileNotFoundError(f"No AOWL files in {RAW_LOGBOOKS}")
        raw_path = candidates[-1]  # latest version

    logger.info("Parsing AOWL raw file: %s", raw_path)
    df = pd.read_csv(
        raw_path, sep="\t", low_memory=False, encoding="utf-8",
        dtype={"VoyageID": str, "Lat": float, "Lon": float,
               "Day": "Int64", "Month": "Int64", "Year": "Int64"},
    )

    # Rename to standard schema
    col_map = {
        "sequence": "seq_id",
        "VoyageID": "voyage_id",
        "Lat": "lat",
        "Lon": "lon",
        "Day": "day",
        "Month": "month",
        "Year": "year",
        "Encounter": "encounter_type",
        "Species": "species_raw",
        "NStruck": "n_struck",
        "NTried": "n_tried",
        "Place": "place_name",
        "Source": "source",
        "Remarks": "remarks",
    }
    df = df.rename(columns=col_map)
    logger.info("  Raw rows: %d", len(df))

    # ── Clean coordinates ──
    valid_coords = (
        df["lat"].notna() & df["lon"].notna()
        & df["lat"].between(-90, 90)
        & df["lon"].between(-180, 180)
    )
    n_bad = (~valid_coords).sum()
    if n_bad:
        logger.warning("  Dropping %d rows with invalid coordinates", n_bad)
    df = df.loc[valid_coords].copy()

    # ── Build timestamp_utc from day/month/year ──
    # AOWL provides day-level resolution; set time to noon UTC
    def _make_ts(row):
        try:
            return pd.Timestamp(
                year=int(row["year"]), month=int(row["month"]),
                day=int(row["day"]), hour=12, tz="UTC",
            )
        except (ValueError, TypeError):
            return pd.NaT

    df["timestamp_utc"] = df.apply(_make_ts, axis=1)
    n_nat = df["timestamp_utc"].isna().sum()
    if n_nat:
        logger.warning("  Dropping %d rows with invalid dates", n_nat)
    df = df.dropna(subset=["timestamp_utc"]).copy()

    # ── Encounter enrichment ──
    df["encounter_type"] = df["encounter_type"].fillna("NoEnc")
    df["species_group"] = df["species_raw"].apply(_map_species_group)

    # Numeric strike/try columns
    for col in ("n_struck", "n_tried"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Binary flag: was there a productive encounter at this position?
    df["catch_event_flag"] = (
        df["encounter_type"].isin(["Strike"])
        & (df["n_struck"] > 0)
    ).astype(int)

    # Sight flag (saw whales but didn't catch)
    df["sight_event_flag"] = (
        df["encounter_type"].isin(["Sight"])
    ).astype(int)

    # Sort chronologically within each voyage
    sort_cols = ["voyage_id", "timestamp_utc"]
    if "seq_id" in df.columns:
        sort_cols.append("seq_id")
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # ── De-duplicate: same voyage+date ──
    # Keep the most informative row (strikes > sights > no encounter)
    enc_priority = {"Strike": 0, "Sight": 1, "Spoke": 2, "NoEnc": 3}
    df["_enc_priority"] = df["encounter_type"].map(enc_priority).fillna(4)
    df = df.sort_values(["voyage_id", "timestamp_utc", "_enc_priority"])

    # For the trajectory: keep one position per day per voyage
    # But accumulate encounter info
    df["_date"] = df["timestamp_utc"].dt.date

    agg_funcs = {
        "lat": "first",
        "lon": "first",
        "timestamp_utc": "first",
        "encounter_type": "first",  # best encounter for that day
        "species_raw": "first",
        "species_group": "first",
        "n_struck": "sum",           # total whales struck that day
        "n_tried": "sum",            # total whales tried that day
        "catch_event_flag": "max",   # any catch that day?
        "sight_event_flag": "max",   # any sighting that day?
        "place_name": "first",
        "year": "first",
        "month": "first",
        "day": "first",
    }
    df = df.groupby(["voyage_id", "_date"], sort=False).agg(
        agg_funcs
    ).reset_index()
    df = df.drop(columns=["_date", "_enc_priority"], errors="ignore")

    logger.info(
        "  Parsed: %d daily positions, %d voyages, years %d–%d",
        len(df), df["voyage_id"].nunique(),
        df["year"].min(), df["year"].max(),
    )
    logger.info(
        "  Encounters: %d strike-days, %d sight-days",
        df["catch_event_flag"].sum(), df["sight_event_flag"].sum(),
    )

    return df


# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 2: Encounter-Enriched Step Features
# ═════════════════════════════════════════════════════════════════════════════

def compute_encounter_step_features(steps_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add encounter-derived features to step-level data for richer signal.

    New columns:
      - days_since_last_catch: temporal distance from last strike
      - days_since_last_sight: temporal distance from last sighting
      - cumulative_catch: running total of whales struck
      - catch_rate: cumulative catch / days elapsed
      - heading_change_post_catch: heading delta after a catch event
      - is_sperm_search, is_right_search, etc: species-specific encounter flags
    """
    df = steps_df.copy()

    # ── Per-voyage running features ──
    for vid, idx in df.groupby("voyage_id", sort=False).groups.items():
        sub = df.loc[idx].copy()
        n = len(sub)

        # Days since last catch
        catch_mask = sub["catch_event_flag"].values == 1 if "catch_event_flag" in sub.columns else np.zeros(n, bool)
        days_since_catch = np.full(n, np.nan)
        last_catch = -999
        for i in range(n):
            if catch_mask[i]:
                last_catch = i
            if last_catch >= 0:
                days_since_catch[i] = i - last_catch
        df.loc[idx, "days_since_catch"] = days_since_catch

        # Days since last sight
        sight_mask = sub["sight_event_flag"].values == 1 if "sight_event_flag" in sub.columns else np.zeros(n, bool)
        days_since_sight = np.full(n, np.nan)
        last_sight = -999
        for i in range(n):
            if sight_mask[i]:
                last_sight = i
            if last_sight >= 0:
                days_since_sight[i] = i - last_sight
        df.loc[idx, "days_since_sight"] = days_since_sight

        # Cumulative catch
        if "n_struck" in sub.columns:
            df.loc[idx, "cumulative_catch"] = sub["n_struck"].cumsum().values

        # Catch rate (cumulative catch / step index)
        if "n_struck" in sub.columns:
            cum = sub["n_struck"].cumsum().values
            step_idx = np.arange(1, n + 1)
            df.loc[idx, "catch_rate"] = cum / step_idx

    # ── Species-specific encounter channels ──
    if "species_group" in df.columns:
        for species in ["sperm", "right", "bowhead", "humpback"]:
            col = f"enc_{species}"
            df[col] = (df["species_group"] == species).astype(float)

    return df


# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 3: Enhanced 1D-CNN with Encounter Channels
# ═════════════════════════════════════════════════════════════════════════════

# The step features for the enriched encoder:
ENRICHED_STEP_FEATURES = [
    # Original movement features
    "step_length_m", "speed_mps", "heading_rad", "turning_angle_rad",
    # Encounter features
    "days_since_catch", "days_since_sight", "catch_rate",
    "catch_event_flag", "sight_event_flag",
    # Species channels
    "enc_sperm", "enc_right", "enc_bowhead", "enc_humpback",
]


def build_enriched_encoder(n_features: int, embed_dim: int):
    """
    Enhanced 1D-CNN encoder with attention pooling and deeper architecture.

    Uses:
    - 3-layer 1D convolutions with residual connections
    - Multi-head self-attention pooling
    - Produces embed_dim-sized voyage representation
    """
    import torch
    import torch.nn as nn

    class ResBlock(nn.Module):
        def __init__(self, channels, kernel_size=5):
            super().__init__()
            self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
            self.bn1 = nn.BatchNorm1d(channels)
            self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
            self.bn2 = nn.BatchNorm1d(channels)
            self.relu = nn.ReLU()

        def forward(self, x):
            residual = x
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.bn2(self.conv2(x))
            return self.relu(x + residual)

    class AttentionPool(nn.Module):
        """Weighted attention pooling over the time dimension."""
        def __init__(self, d_model):
            super().__init__()
            self.query = nn.Linear(d_model, 1)

        def forward(self, x):
            # x: (B, C, T) → (B, T, C)
            x_t = x.permute(0, 2, 1)
            weights = torch.softmax(self.query(x_t), dim=1)  # (B, T, 1)
            return (x_t * weights).sum(dim=1)  # (B, C)

    class EnrichedEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Conv1d(n_features, 64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(),
            )
            self.res1 = ResBlock(64, kernel_size=5)
            self.res2 = ResBlock(64, kernel_size=3)

            self.expand = nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
            )
            self.res3 = ResBlock(128, kernel_size=3)

            self.pool = AttentionPool(128)
            self.fc = nn.Sequential(
                nn.Linear(128, embed_dim),
                nn.LayerNorm(embed_dim),
            )

        def forward(self, x):
            # x: (B, T, F) → (B, F, T) for Conv1d
            x = x.permute(0, 2, 1)
            x = self.input_proj(x)
            x = self.res1(x)
            x = self.res2(x)
            x = self.expand(x)
            x = self.res3(x)
            x = self.pool(x)
            return self.fc(x)

    return EnrichedEncoder()


def build_enriched_decoder(embed_dim: int, n_features: int, seq_len: int):
    """Deeper decoder for masked-feature prediction."""
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(embed_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, seq_len * n_features),
    )


def build_enriched_segments(
    steps_df: pd.DataFrame,
    segment_length: int,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build segments from search-regime steps with enriched features.

    Returns (n_segments, segment_length, n_features) array and voyage_ids.
    """
    fcols = feature_cols or ENRICHED_STEP_FEATURES
    # Use only columns that exist
    fcols = [c for c in fcols if c in steps_df.columns]

    search = steps_df.loc[steps_df["regime_label"] == "search"] if "regime_label" in steps_df.columns else steps_df
    segments: List[np.ndarray] = []
    vids: List[str] = []

    for vid, sub in search.groupby("voyage_id", sort=False):
        X = sub[fcols].values.astype("float32")
        X = np.nan_to_num(X, nan=0.0)

        # Split into overlapping chunks (stride = segment_length // 2)
        stride = max(segment_length // 2, 1)
        for start in range(0, max(len(X), 1), stride):
            chunk = X[start: start + segment_length]
            if len(chunk) < segment_length // 4:
                continue  # Skip tiny trailing chunks
            if len(chunk) < segment_length:
                pad = np.zeros((segment_length - len(chunk), len(fcols)), dtype="float32")
                chunk = np.concatenate([chunk, pad], axis=0)
            segments.append(chunk)
            vids.append(vid)

    if not segments:
        return np.empty((0, segment_length, len(fcols)), dtype="float32"), []
    return np.stack(segments), vids


def train_enriched_embedding(
    steps_df: pd.DataFrame,
    segment_length: int = 128,
    embed_dim: int = 64,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 5e-4,
    mask_ratio: float = 0.20,
    seed: int = 42,
) -> Tuple[object, np.ndarray, List[str]]:
    """
    Train enriched self-supervised embedding with masked-feature prediction.

    Enhanced over the basic version:
    - Deeper encoder with residual blocks and attention pooling
    - More features (movement + encounter channels)
    - Higher mask ratio (20% vs 15%)
    - Cosine annealing LR schedule
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build feature columns that actually exist
    fcols = [c for c in ENRICHED_STEP_FEATURES if c in steps_df.columns]
    logger.info("Embedding features (%d): %s", len(fcols), fcols)

    seg_arr, vids = build_enriched_segments(steps_df, segment_length, fcols)
    if len(seg_arr) == 0:
        logger.warning("No segments — skipping embedding.")
        return None, np.empty((0, embed_dim)), []

    n_features = seg_arr.shape[2]
    seg_len = seg_arr.shape[1]
    logger.info("Built %d segments (shape %s) from %d voyages.",
                len(seg_arr), seg_arr.shape, len(set(vids)))

    # Normalize per-feature
    flat = seg_arr.reshape(-1, n_features)
    means = np.nanmean(flat, axis=0)
    stds = np.nanstd(flat, axis=0)
    stds[stds == 0] = 1.0
    seg_arr = (seg_arr - means) / stds

    segments_t = torch.tensor(seg_arr, dtype=torch.float32)
    dataset = TensorDataset(segments_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    encoder = build_enriched_encoder(n_features, embed_dim)
    decoder = build_enriched_decoder(embed_dim, n_features, seg_len)
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()

    encoder.train()
    decoder.train()

    logger.info("Training enriched encoder (%d params)...",
                sum(p.numel() for p in encoder.parameters()))

    for epoch in range(epochs):
        total_loss = 0.0
        n_samples = 0
        for (batch,) in loader:
            # Mask random timesteps
            mask = torch.rand(batch.shape[0], batch.shape[1]) < mask_ratio
            masked_input = batch.clone()
            masked_input[mask] = 0.0

            z = encoder(masked_input)
            recon = decoder(z).view(batch.shape)

            # Loss only on masked positions
            loss = loss_fn(recon[mask], batch[mask])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * batch.shape[0]
            n_samples += batch.shape[0]

        scheduler.step()
        avg_loss = total_loss / max(n_samples, 1)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                "  Epoch %d/%d, loss=%.6f, lr=%.2e",
                epoch + 1, epochs, avg_loss, scheduler.get_last_lr()[0],
            )

    # Extract embeddings
    encoder.eval()
    with torch.no_grad():
        # Process in batches to avoid OOM
        all_z = []
        for i in range(0, len(segments_t), batch_size):
            batch = segments_t[i:i + batch_size]
            all_z.append(encoder(batch).numpy())
        all_z = np.concatenate(all_z, axis=0)

    # Mean-pool per voyage
    vid_arr = np.array(vids)
    unique_vids = list(dict.fromkeys(vids))
    pooled = np.zeros((len(unique_vids), embed_dim), dtype="float32")
    for i, v in enumerate(unique_vids):
        mask_v = vid_arr == v
        pooled[i] = all_z[mask_v].mean(axis=0)

    logger.info(
        "Trained embedding: %d segments → %d voyage embeddings (dim=%d).",
        len(seg_arr), len(unique_vids), embed_dim,
    )
    return encoder, pooled, unique_vids


# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 4: Full Pipeline Orchestrator
# ═════════════════════════════════════════════════════════════════════════════

def run_full_compass(
    embedding_enabled: bool = True,
    embedding_epochs: int = 30,
    embedding_dim: int = 64,
) -> Dict:
    """
    End-to-end: AOWL raw → compass features → PCA index → DL embedding.

    Produces:
        output/compass/panel_voyage_compass.parquet
        output/compass/steps_with_regimes.parquet
        output/compass/voyage_compass_embeddings.parquet
        output/compass/compass_pipeline_summary.json
    """
    from compass.preprocess import project_all_voyages, smooth_positions
    from compass.steps import compute_raw_steps
    from compass.regimes import segment_voyages
    from compass.features import compute_compass_features
    from compass.compass_index import (
        compute_compass_index, compute_early_window, save_loadings,
    )
    from compass.config import CompassConfig

    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: Dict = {}

    cfg = CompassConfig(
        minimum_points_per_voyage=20,
        smoothing_enabled=True,
        smoothing_window=5,
        time_resample_hours=[],       # skip resampling (daily data is fine)
        distance_thin_meters=[],      # skip thinning
        num_regimes_candidates=[3, 4],
        min_steps_for_hmm=30,
        hmm_n_iter=300,
        embedding_enabled=embedding_enabled,
        segment_length_steps=128,
        embedding_dim=embedding_dim,
        embedding_epochs=embedding_epochs,
        pca_n_components=3,
        early_window_search_steps=[30, 60],
        early_window_days=[10, 20],
        min_search_steps_for_features=20,
        verbose=True,
    )

    # ── Step 1: Parse AOWL data ──
    print("\n" + "═" * 70)
    print("STEP 1: PARSING AOWL LOGBOOK DATA")
    print("═" * 70)
    df_raw = parse_aowl_raw()
    results["n_raw_observations"] = len(df_raw)
    results["n_voyages"] = df_raw["voyage_id"].nunique()

    # Save enriched positions
    df_raw.to_parquet(OUTPUT_DIR / "enriched_positions.parquet", index=False)

    # ── Step 2: Validate & Project ──
    print("\n" + "═" * 70)
    print("STEP 2: COORDINATE PROJECTION (UTM)")
    print("═" * 70)

    # Validate
    from compass.data_io import validate_trajectories
    traj = validate_trajectories(df_raw, cfg)
    traj = traj.reset_index(drop=True)

    # Project to UTM
    traj = project_all_voyages(traj, cfg)
    traj = smooth_positions(traj, cfg)
    traj.to_parquet(OUTPUT_DIR / "projected_points.parquet", index=False)
    print(f"  Projected {len(traj):,} points, {traj['voyage_id'].nunique()} voyages")

    # ── Step 3: Step Construction ──
    print("\n" + "═" * 70)
    print("STEP 3: STEP CONSTRUCTION")
    print("═" * 70)
    steps = compute_raw_steps(traj)
    print(f"  Raw steps: {len(steps):,}")

    # ── Step 3b: Encounter enrichment ──
    print("\n  Enriching with encounter features...")
    steps = compute_encounter_step_features(steps)
    enc_cols = [c for c in steps.columns if c.startswith("days_since") or
                c.startswith("enc_") or c in ("catch_rate", "cumulative_catch",
                "catch_event_flag", "sight_event_flag")]
    print(f"  Added {len(enc_cols)} encounter features: {enc_cols}")

    # ── Step 4: HMM Regime Segmentation ──
    print("\n" + "═" * 70)
    print("STEP 4: HMM REGIME SEGMENTATION")
    print("═" * 70)
    steps_with_regimes = segment_voyages(steps, cfg)
    results["steps_with_regimes"] = steps_with_regimes

    n_search = (steps_with_regimes["regime_label"] == "search").sum()
    n_transit = (steps_with_regimes["regime_label"] == "transit").sum()
    n_return = (steps_with_regimes["regime_label"] == "return").sum()
    print(f"  Transit: {n_transit:,}, Search: {n_search:,}, Return: {n_return:,}")

    steps_with_regimes.to_parquet(OUTPUT_DIR / "steps_with_regimes.parquet", index=False)

    # ── Step 5: Compass Feature Suite ──
    print("\n" + "═" * 70)
    print("STEP 5: COMPASS FEATURES (14 features per voyage)")
    print("═" * 70)
    features = compute_compass_features(steps_with_regimes, cfg)
    print(f"  Features computed for {len(features)} voyages")
    print(f"  Feature columns: {[c for c in features.columns if c not in ('voyage_id', 'n_search_steps')]}")
    features.to_parquet(OUTPUT_DIR / "voyage_compass_features.parquet", index=False)

    # ── Step 6: PCA Compass Index ──
    print("\n" + "═" * 70)
    print("STEP 6: PCA COMPASS INDEX")
    print("═" * 70)
    index_df, loadings = compute_compass_index(features, cfg)
    results["compass_index"] = index_df

    evr = [loadings.get(f"CompassIndex{i+1}_explained_var", 0) for i in range(3)]
    print(f"  PCA explained variance: {[round(v, 3) for v in evr]}")
    print(f"  Total: {sum(evr):.3f}")

    index_df.to_parquet(OUTPUT_DIR / "voyage_compass_index.parquet", index=False)
    save_loadings(loadings, OUTPUT_DIR / "pca_loadings.json")

    # ── Step 7: Early-Window Compass ──
    print("\n" + "═" * 70)
    print("STEP 7: EARLY-WINDOW COMPASS (reverse causality)")
    print("═" * 70)
    try:
        ew = compute_early_window(steps_with_regimes, None, cfg)
        results["early_window"] = ew
        print(f"  Early-window features: {len(ew)} (voyage × window) rows")
        ew.to_parquet(OUTPUT_DIR / "voyage_compass_early_window.parquet", index=False)
    except Exception as e:
        logger.warning("Early-window failed: %s", e)
        ew = pd.DataFrame()

    # ── Step 8: Self-Supervised DL Embedding ──
    if embedding_enabled:
        print("\n" + "═" * 70)
        print("STEP 8: SELF-SUPERVISED 1D-CNN EMBEDDING")
        print("═" * 70)
        try:
            encoder, embeddings, vids = train_enriched_embedding(
                steps_with_regimes,
                segment_length=cfg.segment_length_steps,
                embed_dim=embedding_dim,
                epochs=embedding_epochs,
                batch_size=cfg.embedding_batch_size,
            )

            if encoder is not None and len(embeddings) > 0:
                # Build embedding DataFrame
                emb_df = pd.DataFrame(
                    embeddings,
                    columns=[f"z_{i}" for i in range(embeddings.shape[1])],
                )
                emb_df["voyage_id"] = vids

                # Probe: regress CompassIndex1 on z → DLCompassScore
                from sklearn.linear_model import Ridge
                merged = emb_df.merge(
                    index_df[["voyage_id", "CompassIndex1"]].dropna(),
                    on="voyage_id", how="inner",
                )
                z_cols = [c for c in emb_df.columns if c.startswith("z_")]
                if len(merged) > 10:
                    model = Ridge(alpha=1.0)
                    X_probe = merged[z_cols].values
                    y_probe = merged["CompassIndex1"].values
                    model.fit(X_probe, y_probe)
                    emb_df["DLCompassScore"] = model.predict(emb_df[z_cols].values)
                    r2 = model.score(X_probe, y_probe)
                    print(f"  DL Probe R² = {r2:.4f}")
                    results["dl_probe_r2"] = r2

                emb_df.to_parquet(
                    OUTPUT_DIR / "voyage_compass_embeddings.parquet", index=False,
                )
                results["embeddings"] = emb_df
        except Exception as e:
            logger.error("Embedding training failed: %s", e, exc_info=True)

    # ── Step 9: Build panel_voyage_compass.parquet ──
    print("\n" + "═" * 70)
    print("STEP 9: ASSEMBLING FINAL PANEL")
    print("═" * 70)

    panel = index_df.copy()

    # Merge early-window features
    if not ew.empty:
        for n in cfg.early_window_search_steps:
            ew_n = ew.loc[ew["early_window_n"] == n]
            ci_cols = [c for c in ew_n.columns if c.startswith("CompassIndex")]
            for c in ci_cols:
                ew_n = ew_n.rename(columns={c: f"Early{n}_{c}"})
            merge_cols = ["voyage_id"] + [c for c in ew_n.columns if c.startswith(f"Early{n}_")]
            merge_cols = [c for c in merge_cols if c in ew_n.columns]
            if len(merge_cols) > 1:
                panel = panel.merge(ew_n[merge_cols], on="voyage_id", how="left")

    # Merge DL embedding columns
    if embedding_enabled and "embeddings" in results:
        emb_df = results["embeddings"]
        emb_merge = emb_df[["voyage_id", "DLCompassScore"] +
                           [c for c in emb_df.columns if c.startswith("z_")]].copy()
        panel = panel.merge(emb_merge, on="voyage_id", how="left")

    # Merge encounter summary stats per voyage
    enc_agg = df_raw.groupby("voyage_id").agg(
        total_struck=("n_struck", "sum"),
        total_tried=("n_tried", "sum"),
        n_catch_days=("catch_event_flag", "sum"),
        n_sight_days=("sight_event_flag", "sum"),
        n_distinct_species=("species_group", "nunique"),
        primary_species=("species_group", lambda x: x.value_counts().index[0] if len(x) > 0 else "none"),
    ).reset_index()
    panel = panel.merge(enc_agg, on="voyage_id", how="left")

    panel.to_parquet(OUTPUT_DIR / "panel_voyage_compass.parquet", index=False)
    panel.to_csv(OUTPUT_DIR / "panel_voyage_compass.csv", index=False)
    print(f"  Final panel: {len(panel)} voyages × {len(panel.columns)} columns")
    print(f"  Columns: {panel.columns.tolist()}")

    # ── Summary ──
    elapsed = time.time() - t0
    print("\n" + "═" * 70)
    print(f"COMPASS PIPELINE COMPLETE ({elapsed:.1f}s)")
    print("═" * 70)
    print(f"  Voyages processed: {df_raw['voyage_id'].nunique()}")
    print(f"  Daily observations: {len(df_raw):,}")
    print(f"  Steps with regimes: {len(steps_with_regimes):,}")
    print(f"  Features computed: {len(features)}")
    print(f"  Panel columns: {len(panel.columns)}")
    print(f"  Output: {OUTPUT_DIR / 'panel_voyage_compass.parquet'}")

    summary = {
        "n_observations": len(df_raw),
        "n_voyages": int(df_raw["voyage_id"].nunique()),
        "n_steps": len(steps_with_regimes),
        "n_search_steps": int(n_search),
        "n_transit_steps": int(n_transit),
        "n_return_steps": int(n_return),
        "n_features_computed": len(features),
        "pca_explained_variance": evr,
        "n_panel_columns": len(panel.columns),
        "embedding_enabled": embedding_enabled,
        "elapsed_seconds": round(elapsed, 1),
    }
    if "dl_probe_r2" in results:
        summary["dl_probe_r2"] = round(results["dl_probe_r2"], 4)

    with open(OUTPUT_DIR / "compass_pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    results["panel"] = panel
    results["summary"] = summary
    return results


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Run full compass pipeline from raw AOWL logbook data.",
    )
    parser.add_argument(
        "--embedding", action="store_true", default=True,
        help="Enable DL embedding (default: True)",
    )
    parser.add_argument(
        "--no-embedding", action="store_true",
        help="Disable DL embedding",
    )
    parser.add_argument(
        "--epochs", type=int, default=30,
        help="Embedding training epochs (default: 30)",
    )
    parser.add_argument(
        "--embed-dim", type=int, default=64,
        help="Embedding dimensionality (default: 64)",
    )
    args = parser.parse_args()

    embedding = not args.no_embedding
    run_full_compass(
        embedding_enabled=embedding,
        embedding_epochs=args.epochs,
        embedding_dim=args.embed_dim,
    )


if __name__ == "__main__":
    main()
