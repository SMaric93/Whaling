"""
Compass regressions (C1–C6).

Uses compass-pipeline features (Hill tail index, MRL, CompassIndex PCA
scores, etc.) as RHS or LHS variables in sparse FE + lsqr regressions.

Specifications
--------------
C1 — Compass → Output:       log_q = f(CompassIndex1-3, captain FE, agent FE)
C2 — Skill → Search:         CompassIndex1 = f(α̂, γ̂, route×time FE)
C3 — Interaction:            log_q = f(CI1 × α̂, CI1 × γ̂, FEs)
C4 — Adversity × Search:     log_q = f(CI1 × arctic, FEs)
C5 — Tail → Failure:         failure = f(hill_tail, loiter_frac, FEs)
C6 — Early Window (reverse): log_q = f(EarlyCI1, FEs)
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

from .config import TABLES_DIR, OUTPUT_DIR
from .baseline_production import estimate_r1

warnings.filterwarnings("ignore", category=FutureWarning)

# ── compass data helpers ────────────────────────────────────────────────────

COMPASS_OUTPUT_DIR = OUTPUT_DIR / "compass"

COMPASS_FEATURE_COLS = [
    "hill_tail_index",
    "mean_resultant_length",
    "heading_autocorr_lag1",
    "net_to_gross_ratio",
    "grid_cells_visited",
    "recurrence_rate",
    "loiter_fraction",
    "median_speed_mps",
    "share_top_decile",
    "heading_run_length_mean",
]

COMPASS_INDEX_COLS = ["CompassIndex1", "CompassIndex2", "CompassIndex3"]


def load_compass_features(
    compass_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load the panel_voyage_compass.parquet produced by the compass pipeline.

    Falls back to individual feature columns if CompassIndex* are missing.
    """
    d = compass_dir or COMPASS_OUTPUT_DIR
    path = d / "panel_voyage_compass.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Compass output not found at {path}. "
            "Run the compass pipeline first: python -m compass.cli --config compass_config.json"
        )
    df = pd.read_parquet(path)
    return df


def _build_proxy_compass_features() -> pd.DataFrame:
    """
    Construct compass-like proxy features from existing logbook and spatial data.

    Uses voyage_logbook_features.parquet and spatial columns from
    analysis_voyage.parquet when the full compass pipeline has not been run.
    Applies PCA to produce CompassIndex1-3.
    """
    from sklearn.decomposition import PCA
    from .config import DATA_DIR

    logbook_path = DATA_DIR / "voyage_logbook_features.parquet"
    voyage_path = DATA_DIR / "analysis_voyage.parquet"

    if not logbook_path.exists():
        raise FileNotFoundError(f"No logbook features at {logbook_path}")

    lf = pd.read_parquet(logbook_path)
    print(f"  Loaded logbook features: {len(lf):,} voyages")

    # Merge spatial spread from analysis_voyage if available
    if voyage_path.exists():
        av = pd.read_parquet(voyage_path, columns=[
            "voyage_id", "std_lat", "std_lon", "lat_range", "lon_range",
        ])
        lf = lf.merge(av, on="voyage_id", how="left")

    # ── build proxy features ──
    proxy_cols = []

    # 1. Route efficiency ≈ net_to_gross_ratio
    if "route_efficiency" in lf.columns:
        lf["proxy_net_to_gross"] = lf["route_efficiency"].clip(0, 1)
        proxy_cols.append("proxy_net_to_gross")

    # 2. Ground dwell / transit ratio ≈ loiter fraction
    if "ground_dwell_days" in lf.columns and "transit_days" in lf.columns:
        total = lf["ground_dwell_days"] + lf["transit_days"]
        lf["proxy_dwell_fraction"] = np.where(
            total > 0, lf["ground_dwell_days"] / total, np.nan
        )
        proxy_cols.append("proxy_dwell_fraction")

    # 3. n_grounds_visited ≈ grid_cells_visited
    if "n_grounds_visited" in lf.columns:
        lf["proxy_grounds_visited"] = lf["n_grounds_visited"].astype(float)
        proxy_cols.append("proxy_grounds_visited")

    # 4. ground_switching_count ≈ recurrence_rate proxy
    if "ground_switching_count" in lf.columns:
        lf["proxy_ground_switches"] = lf["ground_switching_count"].astype(float)
        proxy_cols.append("proxy_ground_switches")

    # 5. avg_daily_distance ≈ median_speed proxy
    if "avg_daily_distance_nm" in lf.columns:
        lf["proxy_daily_distance"] = lf["avg_daily_distance_nm"]
        proxy_cols.append("proxy_daily_distance")

    # 6. furthest_from_start ≈ extent / exploration
    if "furthest_from_start_nm" in lf.columns:
        lf["proxy_max_extent"] = lf["furthest_from_start_nm"]
        proxy_cols.append("proxy_max_extent")

    # 7. Spatial spread from lat/lon
    if "std_lat" in lf.columns:
        lf["proxy_spatial_spread"] = np.sqrt(
            lf["std_lat"].fillna(0) ** 2 + lf["std_lon"].fillna(0) ** 2
        )
        proxy_cols.append("proxy_spatial_spread")

    if "lat_range" in lf.columns:
        lf["proxy_lat_range"] = lf["lat_range"]
        proxy_cols.append("proxy_lat_range")

    # 8. total_distance / beeline = gross/net (inverse of efficiency)
    if "total_distance_nm" in lf.columns and "beeline_distance_nm" in lf.columns:
        lf["proxy_tortuosity"] = np.where(
            lf["beeline_distance_nm"] > 0,
            lf["total_distance_nm"] / lf["beeline_distance_nm"],
            np.nan,
        )
        proxy_cols.append("proxy_tortuosity")

    if not proxy_cols:
        raise ValueError("No proxy features could be constructed")

    print(f"  Constructed {len(proxy_cols)} proxy compass features: {proxy_cols}")

    # ── PCA to produce CompassIndex1-3 ──
    valid = lf[proxy_cols].notna().all(axis=1)
    lf_valid = lf.loc[valid, proxy_cols].copy()
    print(f"  Valid for PCA: {len(lf_valid):,} voyages")

    # z-score
    for c in proxy_cols:
        mu, sd = lf_valid[c].mean(), lf_valid[c].std()
        if sd > 0:
            lf_valid[c] = (lf_valid[c] - mu) / sd
        else:
            lf_valid[c] = 0.0

    n_comp = min(3, len(proxy_cols), len(lf_valid) - 1)
    pca = PCA(n_components=n_comp)
    scores = pca.fit_transform(lf_valid.values)

    result = lf[["voyage_id"]].copy()
    for k in range(n_comp):
        col = f"CompassIndex{k + 1}"
        result[col] = np.nan
        result.loc[valid, col] = scores[:, k]

    # Also carry forward proxy features for C5
    for c in proxy_cols:
        result[c] = lf[c]

    # Map proxy names → canonical names for regression fallback
    renames = {
        "proxy_net_to_gross": "net_to_gross_ratio",
        "proxy_dwell_fraction": "loiter_fraction",
        "proxy_grounds_visited": "grid_cells_visited",
        "proxy_ground_switches": "recurrence_rate",
        "proxy_daily_distance": "median_speed_mps",
    }
    for old, new in renames.items():
        if old in result.columns:
            result[new] = result[old]

    evr = pca.explained_variance_ratio_
    print(f"  PCA explained variance: {[round(v, 3) for v in evr]}")
    print(f"  Total: {sum(evr):.3f}")

    return result


def merge_compass_to_analysis(
    analysis_df: pd.DataFrame,
    compass_df: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join compass features onto analysis sample by voyage_id."""
    # Only keep compass-specific columns + voyage_id
    keep = ["voyage_id"] + [
        c for c in compass_df.columns
        if c in COMPASS_FEATURE_COLS + COMPASS_INDEX_COLS
        or c.startswith("Compass") or c.startswith("Early")
        or c.startswith("proxy_")
        or c.startswith("z_")          # DL embedding dimensions
        or c.startswith("DL")          # DLCompassScore
        or c in ("n_search_steps", "total_struck", "total_tried",
                 "n_catch_days", "n_sight_days", "n_distinct_species",
                 "primary_species")
    ]
    keep = [c for c in keep if c in compass_df.columns]
    compass_slim = compass_df[keep].copy()

    merged = analysis_df.merge(compass_slim, on="voyage_id", how="left")
    n_matched = merged[COMPASS_INDEX_COLS[0]].notna().sum() if COMPASS_INDEX_COLS[0] in merged.columns else 0
    print(f"  Compass merge: {n_matched:,} / {len(merged):,} voyages matched")
    return merged


# ── sparse helpers ──────────────────────────────────────────────────────────

def _build_fe_block(df: pd.DataFrame, col: str, drop_first: bool = False) -> sp.csr_matrix:
    """Build a sparse dummy‐variable block for *col*."""
    n = len(df)
    ids = df[col].unique()
    id_map = {v: i for i, v in enumerate(ids)}
    idx = df[col].map(id_map).values
    X = sp.csr_matrix(
        (np.ones(n), (np.arange(n), idx)),
        shape=(n, len(ids)),
    )
    if drop_first:
        X = X[:, 1:]
    return X


def _ols_sparse(X: sp.spmatrix, y: np.ndarray, control_names: list[str],
                n_fe_cols: int) -> Dict:
    """Run lsqr, extract control coefficients, compute R² and approx SEs."""
    result = lsqr(X, y, iter_lim=15000, atol=1e-10, btol=1e-10)
    beta = result[0]
    y_hat = X @ beta
    resid = y - y_hat
    n, k = X.shape
    r2 = 1 - np.var(resid) / np.var(y) if np.var(y) > 0 else np.nan
    sigma2 = np.sum(resid ** 2) / max(n - k, 1)

    # Extract control coefficients (last len(control_names) entries)
    n_controls = len(control_names)
    control_betas = beta[-n_controls:]

    # Approximate SEs from diagonal of (X_ctrl'X_ctrl)^{-1}
    X_ctrl = X[:, -n_controls:].toarray() if n_controls > 0 else np.empty((n, 0))
    try:
        gram = X_ctrl.T @ X_ctrl
        gram_inv_diag = np.diag(np.linalg.inv(gram + 1e-12 * np.eye(n_controls)))
        se = np.sqrt(np.maximum(sigma2 * gram_inv_diag, 0))
    except np.linalg.LinAlgError:
        se = np.full(n_controls, np.nan)

    rows = []
    for name, b, s in zip(control_names, control_betas, se):
        t = b / s if s > 0 else np.nan
        rows.append({"variable": name, "coef": b, "se": s, "t": t})

    return {
        "coefficients": pd.DataFrame(rows),
        "r2": r2,
        "n": n,
        "k": k,
        "sigma2": sigma2,
        "beta_full": beta,
        "residuals": resid,
    }


# ── C1: Compass → Output ───────────────────────────────────────────────────

def run_c1_compass_to_output(df: pd.DataFrame) -> Dict:
    """
    C1: Do compass-index features predict output *within* captain×agent cells?

    log_q = α_c + γ_a + Σ β_k · CI_k + δ_{vp} + X·b + ε
    """
    print("\n" + "=" * 70)
    print("C1: COMPASS INDEX → OUTPUT (within captain×agent)")
    print("=" * 70)

    # Baseline FEs
    r1 = estimate_r1(df, use_loo_sample=True)
    df_est = r1["df"]

    # Check compass columns available
    ci_cols = [c for c in COMPASS_INDEX_COLS if c in df_est.columns]
    if not ci_cols:
        # fall back to raw features
        ci_cols = [c for c in COMPASS_FEATURE_COLS if c in df_est.columns]
    if not ci_cols:
        print("  No compass features found — skipping C1")
        return {"error": "no_compass_features"}

    # Drop rows with missing compass
    mask = df_est[ci_cols].notna().all(axis=1)
    df_c = df_est.loc[mask].copy()
    print(f"  Sample: {len(df_c):,} voyages with compass data")

    n = len(df_c)
    y = df_c["log_q"].values

    # FE blocks
    matrices = [
        _build_fe_block(df_c, "captain_id"),
        _build_fe_block(df_c, "agent_id", drop_first=True),
    ]
    n_fe = sum(m.shape[1] for m in matrices)

    if "vessel_period" in df_c.columns:
        vp = _build_fe_block(df_c, "vessel_period", drop_first=True)
        matrices.append(vp)
        n_fe += vp.shape[1]

    # Controls: log_duration, log_tonnage, + compass indices
    control_names = ["log_duration", "log_tonnage"] + ci_cols
    ctrl = np.column_stack([
        df_c["log_duration"].values,
        df_c["log_tonnage"].values,
    ] + [df_c[c].values for c in ci_cols])
    matrices.append(sp.csr_matrix(ctrl))

    X = sp.hstack(matrices)
    res = _ols_sparse(X, y, control_names, n_fe)

    print(f"\n--- C1 Results (R²={res['r2']:.4f}, N={res['n']:,}) ---")
    print(res["coefficients"].to_string(index=False, float_format="%.4f"))

    return {"spec": "C1", **res}


# ── C2: Skill → Search ─────────────────────────────────────────────────────

def run_c2_skill_to_search(df: pd.DataFrame) -> Dict:
    """
    C2: Does higher captain/agent quality produce different search geometry?

    CompassIndex1 = b1·α̂ + b2·γ̂ + θ_{route×time} + ε
    """
    print("\n" + "=" * 70)
    print("C2: SKILL / CAPABILITY → SEARCH GEOMETRY")
    print("=" * 70)

    r1 = estimate_r1(df, use_loo_sample=True)
    df_est = r1["df"]

    dv = "CompassIndex1"
    if dv not in df_est.columns:
        print(f"  {dv} missing — skipping C2")
        return {"error": "no_compass_index"}

    mask = df_est[dv].notna() & df_est["alpha_hat"].notna() & df_est["gamma_hat"].notna()
    df_c = df_est.loc[mask].copy()
    print(f"  Sample: {len(df_c):,} voyages")

    n = len(df_c)
    y = df_c[dv].values

    matrices = []
    n_fe = 0

    # Route×time FEs
    if "route_time" in df_c.columns:
        rt = _build_fe_block(df_c, "route_time", drop_first=True)
        matrices.append(rt)
        n_fe += rt.shape[1]

    # Controls
    control_names = ["alpha_hat", "gamma_hat", "log_tonnage"]
    ctrl = np.column_stack([
        df_c["alpha_hat"].values,
        df_c["gamma_hat"].values,
        df_c["log_tonnage"].values,
    ])
    matrices.append(sp.csr_matrix(ctrl))

    X = sp.hstack(matrices) if matrices else sp.csr_matrix(ctrl)
    res = _ols_sparse(X, y, control_names, n_fe)

    print(f"\n--- C2 Results (R²={res['r2']:.4f}, N={res['n']:,}) ---")
    print(res["coefficients"].to_string(index=False, float_format="%.4f"))

    # Quartile gradient
    df_c["alpha_q"] = pd.qcut(df_c["alpha_hat"], q=4, labels=[1, 2, 3, 4])
    df_c["gamma_q"] = pd.qcut(df_c["gamma_hat"], q=4, labels=[1, 2, 3, 4])
    print("\nMean CompassIndex1 by captain-skill quartile:")
    print(df_c.groupby("alpha_q")[dv].mean().round(3).to_string())
    print("\nMean CompassIndex1 by agent-capability quartile:")
    print(df_c.groupby("gamma_q")[dv].mean().round(3).to_string())

    return {"spec": "C2", "dv": dv, **res}


# ── C3: Search × Skill interaction ──────────────────────────────────────────

def run_c3_interaction(df: pd.DataFrame) -> Dict:
    """
    C3: Does search policy amplify or dampen skill returns?

    log_q = α_c + γ_a + b1·CI1 + b2·(CI1 × α̂) + b3·(CI1 × γ̂) + δ + ε
    """
    print("\n" + "=" * 70)
    print("C3: COMPASS × SKILL INTERACTION → OUTPUT")
    print("=" * 70)

    r1 = estimate_r1(df, use_loo_sample=True)
    df_est = r1["df"]

    ci = "CompassIndex1"
    if ci not in df_est.columns:
        print(f"  {ci} missing — skipping C3")
        return {"error": "no_compass_index"}

    mask = df_est[ci].notna()
    df_c = df_est.loc[mask].copy()
    print(f"  Sample: {len(df_c):,} voyages")

    # Interactions
    df_c["CI1_x_alpha"] = df_c[ci] * df_c["alpha_hat"]
    df_c["CI1_x_gamma"] = df_c[ci] * df_c["gamma_hat"]

    n = len(df_c)
    y = df_c["log_q"].values

    matrices = [
        _build_fe_block(df_c, "captain_id"),
        _build_fe_block(df_c, "agent_id", drop_first=True),
    ]
    n_fe = sum(m.shape[1] for m in matrices)

    if "vessel_period" in df_c.columns:
        vp = _build_fe_block(df_c, "vessel_period", drop_first=True)
        matrices.append(vp)
        n_fe += vp.shape[1]

    control_names = ["log_duration", "log_tonnage", ci, "CI1_x_alpha", "CI1_x_gamma"]
    ctrl = np.column_stack([
        df_c["log_duration"].values,
        df_c["log_tonnage"].values,
        df_c[ci].values,
        df_c["CI1_x_alpha"].values,
        df_c["CI1_x_gamma"].values,
    ])
    matrices.append(sp.csr_matrix(ctrl))

    X = sp.hstack(matrices)
    res = _ols_sparse(X, y, control_names, n_fe)

    print(f"\n--- C3 Results (R²={res['r2']:.4f}, N={res['n']:,}) ---")
    print(res["coefficients"].to_string(index=False, float_format="%.4f"))

    return {"spec": "C3", **res}


# ── C4: Adversity × Search ──────────────────────────────────────────────────

def run_c4_adversity(df: pd.DataFrame) -> Dict:
    """
    C4: Is area-restricted search more productive on risky/sparse grounds?

    log_q = α_c + γ_a + b1·CI1 + b2·Arctic + b3·(CI1 × Arctic) + δ + ε
    """
    print("\n" + "=" * 70)
    print("C4: COMPASS × ADVERSITY → OUTPUT")
    print("=" * 70)

    r1 = estimate_r1(df, use_loo_sample=True)
    df_est = r1["df"]

    ci = "CompassIndex1"
    if ci not in df_est.columns:
        print(f"  {ci} missing — skipping C4")
        return {"error": "no_compass_index"}

    # Build arctic indicator if needed
    if "arctic_route" not in df_est.columns:
        if "route_or_ground" in df_est.columns:
            kw = ["arctic", "bering", "hudson", "bowhead", "polar", "ice"]
            df_est["arctic_route"] = df_est["route_or_ground"].str.lower().str.contains(
                "|".join(kw), na=False
            ).astype(int)
        else:
            df_est["arctic_route"] = 0

    mask = df_est[ci].notna()
    df_c = df_est.loc[mask].copy()
    df_c["CI1_x_arctic"] = df_c[ci] * df_c["arctic_route"]

    print(f"  Sample: {len(df_c):,} voyages, Arctic: {df_c['arctic_route'].sum():,}")

    n = len(df_c)
    y = df_c["log_q"].values

    matrices = [
        _build_fe_block(df_c, "captain_id"),
        _build_fe_block(df_c, "agent_id", drop_first=True),
    ]
    n_fe = sum(m.shape[1] for m in matrices)

    if "vessel_period" in df_c.columns:
        vp = _build_fe_block(df_c, "vessel_period", drop_first=True)
        matrices.append(vp)
        n_fe += vp.shape[1]

    control_names = ["log_duration", "log_tonnage", ci, "arctic_route", "CI1_x_arctic"]
    ctrl = np.column_stack([
        df_c["log_duration"].values,
        df_c["log_tonnage"].values,
        df_c[ci].values,
        df_c["arctic_route"].values.astype(float),
        df_c["CI1_x_arctic"].values,
    ])
    matrices.append(sp.csr_matrix(ctrl))

    X = sp.hstack(matrices)
    res = _ols_sparse(X, y, control_names, n_fe)

    print(f"\n--- C4 Results (R²={res['r2']:.4f}, N={res['n']:,}) ---")
    print(res["coefficients"].to_string(index=False, float_format="%.4f"))

    # Marginal effect of CI1 by ground type
    coefs = res["coefficients"].set_index("variable")["coef"]
    main = coefs.get(ci, 0)
    inter = coefs.get("CI1_x_arctic", 0)
    print(f"\n  Marginal effect of CompassIndex1:")
    print(f"    Non-Arctic: {main:.4f}")
    print(f"    Arctic:     {main + inter:.4f}")

    return {"spec": "C4", **res}


# ── C5: Tail Index → Failure ────────────────────────────────────────────────

def run_c5_tail_failure(df: pd.DataFrame) -> Dict:
    """
    C5: Do Lévy-like search patterns (heavy tail) reduce failure risk?

    Failure = b1·hill_tail + b2·loiter_frac + α̂ + γ̂ + X + ε
    """
    print("\n" + "=" * 70)
    print("C5: SEARCH TAIL / LOITERING → FAILURE RISK")
    print("=" * 70)

    r1 = estimate_r1(df, use_loo_sample=True)
    df_est = r1["df"]

    # Build failure indicator
    if "failure_indicator" not in df_est.columns:
        if "voyage_outcome" in df_est.columns:
            df_est["failure_indicator"] = df_est["voyage_outcome"].isin(
                ["lost", "condemned", "wrecked", "missing"]
            ).astype(int)
        else:
            q5 = df_est["q_total_index"].quantile(0.05)
            df_est["failure_indicator"] = (df_est["q_total_index"] <= q5).astype(int)

    feats = ["hill_tail_index", "loiter_fraction"]
    avail = [f for f in feats if f in df_est.columns]
    if not avail:
        print("  No tail/loiter features — skipping C5")
        return {"error": "no_features"}

    mask = df_est[avail].notna().all(axis=1)
    df_c = df_est.loc[mask].copy()
    print(f"  Sample: {len(df_c):,} voyages, Failures: {df_c['failure_indicator'].sum():,}")

    n = len(df_c)
    y = df_c["failure_indicator"].values.astype(float)

    # LPM with FEs
    control_names = ["alpha_hat", "gamma_hat", "log_tonnage"] + avail
    ctrl_cols = [
        df_c["alpha_hat"].values,
        df_c["gamma_hat"].values,
        df_c["log_tonnage"].values,
    ] + [df_c[f].values for f in avail]
    ctrl = np.column_stack(ctrl_cols)

    matrices = []
    n_fe = 0
    if "route_time" in df_c.columns:
        rt = _build_fe_block(df_c, "route_time", drop_first=True)
        matrices.append(rt)
        n_fe += rt.shape[1]

    matrices.append(sp.csr_matrix(ctrl))
    X = sp.hstack(matrices)

    res = _ols_sparse(X, y, control_names, n_fe)

    print(f"\n--- C5 Results (LPM, R²={res['r2']:.4f}, N={res['n']:,}) ---")
    print(res["coefficients"].to_string(index=False, float_format="%.4f"))

    return {"spec": "C5", **res}


# ── C6: Early-Window Reduced Form ───────────────────────────────────────────

def run_c6_early_window(df: pd.DataFrame) -> Dict:
    """
    C6: Reverse-causality check — does *early* search geometry still predict output?

    log_q = α_c + γ_a + b·EarlyCI1 + δ + ε
    """
    print("\n" + "=" * 70)
    print("C6: EARLY-WINDOW COMPASS → OUTPUT (reverse-causality check)")
    print("=" * 70)

    r1 = estimate_r1(df, use_loo_sample=True)
    df_est = r1["df"]

    # Try to find early-window columns (handles both naming conventions)
    # Real pipeline: Early30_CompassIndex1, Early60_CompassIndex1
    # Alternative: EarlyCompassIndex1, early_compass_index1
    early_cols = [c for c in df_est.columns
                  if c.startswith("Early") and "CompassIndex" in c]
    if not early_cols:
        early_cols = [c for c in df_est.columns
                      if c.startswith("EarlyCompass") or c.startswith("early_")]
    if not early_cols:
        print("  No early-window features found — skipping C6")
        return {"error": "no_early_window"}

    ci = early_cols[0]
    mask = df_est[ci].notna()
    df_c = df_est.loc[mask].copy()
    print(f"  Sample: {len(df_c):,} voyages with {ci}")

    n = len(df_c)
    y = df_c["log_q"].values

    matrices = [
        _build_fe_block(df_c, "captain_id"),
        _build_fe_block(df_c, "agent_id", drop_first=True),
    ]
    n_fe = sum(m.shape[1] for m in matrices)

    if "vessel_period" in df_c.columns:
        vp = _build_fe_block(df_c, "vessel_period", drop_first=True)
        matrices.append(vp)
        n_fe += vp.shape[1]

    control_names = ["log_duration", "log_tonnage", ci]
    ctrl = np.column_stack([
        df_c["log_duration"].values,
        df_c["log_tonnage"].values,
        df_c[ci].values,
    ])
    matrices.append(sp.csr_matrix(ctrl))

    X = sp.hstack(matrices)
    res = _ols_sparse(X, y, control_names, n_fe)

    print(f"\n--- C6 Results (R²={res['r2']:.4f}, N={res['n']:,}) ---")
    print(res["coefficients"].to_string(index=False, float_format="%.4f"))

    return {"spec": "C6", **res}


# ── orchestrator ────────────────────────────────────────────────────────────

def run_compass_regressions(
    df: pd.DataFrame,
    compass_dir: Optional[Path] = None,
    save_outputs: bool = True,
) -> Dict:
    """
    Run all compass regressions (C1–C6).

    Parameters
    ----------
    df : pd.DataFrame
        Analysis-ready voyage data (from prepare_analysis_sample()).
    compass_dir : Path, optional
        Directory containing compass parquet files.
    save_outputs : bool
        Save summary CSVs to output/compass/.

    Returns
    -------
    Dict
        Keyed by spec id.
    """
    print("\n" + "=" * 70)
    print("COMPASS REGRESSIONS (C1–C6)")
    print("=" * 70)

    # ── load and merge compass features ──
    try:
        compass_df = load_compass_features(compass_dir)
        df = merge_compass_to_analysis(df, compass_df)
    except FileNotFoundError:
        print("\n  Full compass pipeline output not found.")
        print("  Building proxy compass features from logbook + spatial data...")
        try:
            compass_df = _build_proxy_compass_features()
            df = merge_compass_to_analysis(df, compass_df)
        except Exception as e2:
            print(f"  Proxy feature construction also failed: {e2}")
            print("  Will attempt regressions with whatever compass columns are already in df.")

    results: Dict[str, Dict] = {}

    # C1
    try:
        results["C1"] = run_c1_compass_to_output(df)
    except Exception as e:
        print(f"  C1 failed: {e}")
        results["C1"] = {"error": str(e)}

    # C2
    try:
        results["C2"] = run_c2_skill_to_search(df)
    except Exception as e:
        print(f"  C2 failed: {e}")
        results["C2"] = {"error": str(e)}

    # C3
    try:
        results["C3"] = run_c3_interaction(df)
    except Exception as e:
        print(f"  C3 failed: {e}")
        results["C3"] = {"error": str(e)}

    # C4
    try:
        results["C4"] = run_c4_adversity(df)
    except Exception as e:
        print(f"  C4 failed: {e}")
        results["C4"] = {"error": str(e)}

    # C5
    try:
        results["C5"] = run_c5_tail_failure(df)
    except Exception as e:
        print(f"  C5 failed: {e}")
        results["C5"] = {"error": str(e)}

    # C6
    try:
        results["C6"] = run_c6_early_window(df)
    except Exception as e:
        print(f"  C6 failed: {e}")
        results["C6"] = {"error": str(e)}

    # ── save summary ──
    if save_outputs:
        _save_summary(results)

    return results


def _save_summary(results: Dict) -> None:
    """Save a compact summary CSV + JSON of all compass regressions."""
    out_dir = COMPASS_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for spec_id, res in results.items():
        if "error" in res:
            rows.append({
                "spec": spec_id,
                "status": "error",
                "error": res["error"],
            })
            continue

        coefs = res.get("coefficients", pd.DataFrame())
        for _, row in coefs.iterrows():
            rows.append({
                "spec": spec_id,
                "variable": row["variable"],
                "coef": row["coef"],
                "se": row["se"],
                "t": row["t"],
                "r2": res["r2"],
                "n": res["n"],
            })

    summary = pd.DataFrame(rows)
    csv_path = out_dir / "compass_regressions_summary.csv"
    summary.to_csv(csv_path, index=False)
    print(f"\nSaved compass regression summary → {csv_path}")

    # JSON with key statistics
    json_out = {}
    for spec_id, res in results.items():
        if "error" in res:
            json_out[spec_id] = {"error": res["error"]}
            continue
        coefs = res.get("coefficients", pd.DataFrame())
        json_out[spec_id] = {
            "r2": round(res["r2"], 5),
            "n": res["n"],
            "coefficients": {
                row["variable"]: {
                    "coef": round(row["coef"], 6),
                    "se": round(row["se"], 6),
                    "t": round(row["t"], 3),
                }
                for _, row in coefs.iterrows()
            },
        }
    json_path = out_dir / "compass_regressions.json"
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"Saved compass regression JSON    → {json_path}")


# ── CLI entry ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from .data_loader import prepare_analysis_sample

    df = prepare_analysis_sample()
    results = run_compass_regressions(df)
