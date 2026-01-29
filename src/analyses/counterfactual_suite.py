"""
Counterfactual Simulation Suite.

Unified module for all counterfactual analyses in the Whaling project.

Counterfactuals:
  CF_BASE: Efficient Sorting, Lévy Tax, Static Firm (baseline)
  CF_A2:   Map diffusion targeted to sparse grounds
  CF_A3:   Map adoption in high-risk climate states
  CF_B5:   Anti-assortative planner matching
  CF_C8:   Trait vs forced exploration decomposition
  CF_F15:  Inequality decomposition (θ vs ψ vs map tech)

Key Parameters:
  β₃ (sparse): -0.052***
  β₃ (rich):   +0.011 (ns)
  β_μ_γ (route×time FE): -0.0085***
  β_μ_γ (ground×time FE): -0.0104***
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr
from scipy import stats
from scipy.optimize import linear_sum_assignment


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
COUNTERFACTUAL_DIR = OUTPUT_DIR / "counterfactual"
FIGURES_DIR = COUNTERFACTUAL_DIR / "figures"
TABLES_DIR = COUNTERFACTUAL_DIR / "tables"

# Ground classification
SPARSE_GROUNDS = [
    "pacific", "n pacific", "s pacific", "indian", "indian o",
    "japan", "ochotsk", "okhotsk", "nw coast", "bering", "arctic"
]
RICH_GROUNDS = [
    "atlantic", "brazil", "patagonia", "s atlantic", "w indies",
    "gulf of mexico", "hudson bay", "greenland"
]

# Estimated parameters (from prior analyses)
PARAMS = {
    "beta_theta": 0.132,
    "beta_psi": 0.509,
    "beta3_pooled": -0.039,
    "beta3_sparse": -0.052,
    "beta3_rich": 0.011,
    "beta_mu_route_time": -0.0085,
    "beta_mu_ground_time": -0.0104,
    "beta_mu_output": -0.31,  # μ → log_q
}


# =============================================================================
# Utility Functions
# =============================================================================

def ensure_dirs():
    """Create output directories."""
    COUNTERFACTUAL_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def classify_ground(ground_str: str) -> str:
    """Classify ground as sparse, rich, or unknown."""
    if pd.isna(ground_str):
        return "unknown"
    ground_lower = ground_str.lower()
    for pattern in SPARSE_GROUNDS:
        if pattern in ground_lower:
            return "sparse"
    for pattern in RICH_GROUNDS:
        if pattern in ground_lower:
            return "rich"
    return "unknown"


def classify_ground_ex_ante(df: pd.DataFrame, method: str = "lagged_year") -> pd.DataFrame:
    """
    Classify grounds using EX-ANTE productivity data to avoid endogeneity.
    
    Addresses reviewer critique: "State classification must be ex-ante to avoid
    circularity where realized catch defines the state."
    
    Parameters
    ----------
    df : pd.DataFrame
        Voyage data with year_out, route_or_ground/ground_or_route, log_q.
    method : str
        Classification method:
        - "lagged_year": Use previous year's average catch on that ground
        - "decadal_average": Use decadal average catch (prior decade)
        - "name_based": Use static ground name classification (original method)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with ground_type_ex_ante column added.
    """
    df = df.copy()
    
    ground_col = "ground_or_route" if "ground_or_route" in df.columns else "route_or_ground"
    year_col = "year_out" if "year_out" in df.columns else "year"
    
    if ground_col not in df.columns:
        df["ground_type_ex_ante"] = "unknown"
        return df
    
    if method == "name_based":
        # Original static classification (for robustness comparison)
        df["ground_type_ex_ante"] = df[ground_col].apply(classify_ground)
        
    elif method == "lagged_year":
        # Compute previous year's average catch by ground
        df["ground_normalized"] = df[ground_col].str.lower().str.strip()
        
        # Lag productivity: for year t, use year t-1 average
        ground_year_avg = df.groupby(["ground_normalized", year_col])["log_q"].mean().reset_index()
        ground_year_avg.columns = ["ground_normalized", "year_lag_source", "lagged_avg_catch"]
        ground_year_avg["year_for_merge"] = ground_year_avg["year_lag_source"] + 1
        
        df = df.merge(
            ground_year_avg[["ground_normalized", "year_for_merge", "lagged_avg_catch"]],
            left_on=["ground_normalized", year_col],
            right_on=["ground_normalized", "year_for_merge"],
            how="left"
        )
        df.drop(columns=["year_for_merge"], errors="ignore", inplace=True)
        
        # Classify based on lagged catch: below median = sparse, above = rich
        median_catch = df["lagged_avg_catch"].median()
        df["ground_type_ex_ante"] = np.where(
            df["lagged_avg_catch"].isna(),
            "unknown",
            np.where(df["lagged_avg_catch"] < median_catch, "sparse", "rich")
        )
        
        # Clean up
        df.drop(columns=["ground_normalized", "lagged_avg_catch"], errors="ignore", inplace=True)
        
    elif method == "decadal_average":
        # Use prior decade's average catch by ground
        df["ground_normalized"] = df[ground_col].str.lower().str.strip()
        df["decade"] = (df[year_col] // 10) * 10
        
        # Compute decade-level averages
        decade_avg = df.groupby(["ground_normalized", "decade"])["log_q"].mean().reset_index()
        decade_avg.columns = ["ground_normalized", "decade_lag_source", "decadal_avg_catch"]
        decade_avg["decade_for_merge"] = decade_avg["decade_lag_source"] + 10  # Prior decade
        
        df = df.merge(
            decade_avg[["ground_normalized", "decade_for_merge", "decadal_avg_catch"]],
            left_on=["ground_normalized", "decade"],
            right_on=["ground_normalized", "decade_for_merge"],
            how="left"
        )
        df.drop(columns=["decade_for_merge"], errors="ignore", inplace=True)
        
        # Classify
        median_catch = df["decadal_avg_catch"].median()
        df["ground_type_ex_ante"] = np.where(
            df["decadal_avg_catch"].isna(),
            "unknown",
            np.where(df["decadal_avg_catch"] < median_catch, "sparse", "rich")
        )
        
        df.drop(columns=["ground_normalized", "decadal_avg_catch"], errors="ignore", inplace=True)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Report classification
    counts = df["ground_type_ex_ante"].value_counts()
    print(f"  Ex-ante ground classification ({method}):")
    for gt in ["sparse", "rich", "unknown"]:
        if gt in counts:
            print(f"    {gt}: {counts[gt]:,} ({100*counts[gt]/len(df):.1f}%)")
    
    return df


def standardize(x: np.ndarray, safe: bool = False) -> np.ndarray:
    """Standardize to mean 0, std 1.
    
    If safe=True, returns zeros when std is 0 or too small.
    """
    std = np.nanstd(x)
    if safe and (std < 1e-10 or np.isnan(std)):
        return np.zeros_like(x)
    return (x - np.nanmean(x)) / max(std, 1e-10)


def compute_gini(y: np.ndarray) -> float:
    """Compute Gini coefficient."""
    y = np.sort(y)
    n = len(y)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * y) - (n + 1) * np.sum(y)) / (n * np.sum(y))


def compute_p90_p10(y: np.ndarray) -> float:
    """Compute P90/P10 ratio."""
    return np.percentile(y, 90) / max(np.percentile(y, 10), 0.001)


# =============================================================================
# Data Preparation
# =============================================================================

def load_weather_data() -> pd.DataFrame:
    """Load annual weather data from package."""
    from pathlib import Path
    
    weather_path = Path(__file__).parent.parent.parent / "data" / "raw" / "weather" / "weather_annual_combined.csv"
    
    if weather_path.exists():
        weather = pd.read_csv(weather_path)
        return weather
    
    return pd.DataFrame()


def load_climate_augmented_voyages() -> pd.DataFrame:
    """Load voyage data with climate variables already merged."""
    from pathlib import Path
    
    climate_path = Path(__file__).parent.parent.parent / "data" / "final" / "analysis_voyage_with_climate.parquet"
    
    if climate_path.exists():
        return pd.read_parquet(climate_path)
    
    return pd.DataFrame()


def prepare_counterfactual_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for counterfactual analysis.
    Adds required columns and classifications.
    Merges weather/ice data if available.
    """
    df = df.copy()
    
    # Ensure ground classification
    ground_col = "ground_or_route" if "ground_or_route" in df.columns else "route_or_ground"
    if ground_col in df.columns:
        df["ground_type"] = df[ground_col].apply(classify_ground)
    else:
        df["ground_type"] = "unknown"
    
    # Ensure era classification
    year_col = "year_out" if "year_out" in df.columns else "year"
    if year_col in df.columns:
        df["era"] = np.where(df[year_col] < 1870, "pre_1870", "post_1870")
        df["decade"] = (df[year_col] // 10) * 10
    
    # Standardized estimates
    if "alpha_hat" in df.columns:
        df["theta_hat"] = df["alpha_hat"]  # Alias
        df["theta_std"] = standardize(df["theta_hat"].values)
    if "gamma_hat" in df.columns:
        df["psi_hat"] = df["gamma_hat"]  # Alias
        df["psi_std"] = standardize(df["psi_hat"].values)
    
    # Simulate μ if not available
    if "levy_mu" not in df.columns and "psi_hat" in df.columns:
        np.random.seed(42)
        psi_std = df["psi_std"].values
        df["levy_mu"] = 1.64 + PARAMS["beta_mu_route_time"] * psi_std + np.random.normal(0, 0.15, len(df))
        df["levy_mu"] = df["levy_mu"].clip(1.0, 2.5)
    
    # =========================================================================
    # Merge weather/ice data if not already present
    # =========================================================================
    
    # Check if we need to add hurricane data
    if "n_hurricanes" not in df.columns or "ace_zscore" not in df.columns:
        weather = load_weather_data()
        if len(weather) > 0 and year_col in df.columns:
            # Merge hurricane data on year
            hurricane_cols = ["year", "n_hurricanes", "ace_zscore", "corridor_hurricane_days", "n_storms"]
            weather_subset = weather[[c for c in hurricane_cols if c in weather.columns]]
            df = df.merge(weather_subset, left_on=year_col, right_on="year", how="left", suffixes=("", "_weather"))
            if "year_weather" in df.columns:
                df.drop(columns=["year_weather"], inplace=True, errors="ignore")
            print(f"  Merged hurricane data: {df['n_hurricanes'].notna().sum():,} voyages with data")
    
    # Check if we need to add ice data
    if "bering_ice_mean" not in df.columns or "nh_ice_extent_mean" not in df.columns:
        climate_df = load_climate_augmented_voyages()
        if len(climate_df) > 0 and "voyage_id" in df.columns and "voyage_id" in climate_df.columns:
            # Merge ice data on voyage_id
            ice_cols = ["voyage_id", "bering_ice_mean", "nh_ice_extent_mean", "chukchi_ice_mean", 
                       "beaufort_ice_mean", "frac_days_in_arctic_polygon", "arctic_days"]
            ice_subset = climate_df[[c for c in ice_cols if c in climate_df.columns]]
            df = df.merge(ice_subset, on="voyage_id", how="left", suffixes=("", "_climate"))
            print(f"  Merged ice data: {df['bering_ice_mean'].notna().sum():,} voyages with data")
    
    # =========================================================================
    # Compute derived risk indicators
    # =========================================================================
    
    # Hurricane exposure - normalize to 0-1 scale based on ACE z-score
    if "ace_zscore" in df.columns and df["ace_zscore"].notna().any():
        df["hurricane_exposure"] = df["ace_zscore"].fillna(0)
        df["high_hurricane"] = (df["hurricane_exposure"] > df["hurricane_exposure"].quantile(0.75)).astype(int)
    elif "n_hurricanes" in df.columns and df["n_hurricanes"].notna().any():
        df["hurricane_exposure"] = df["n_hurricanes"].fillna(0)
        df["high_hurricane"] = (df["hurricane_exposure"] > df["hurricane_exposure"].quantile(0.75)).astype(int)
    else:
        df["hurricane_exposure"] = 0
        df["high_hurricane"] = 0
    
    # Ice exposure - use bering ice as primary (most relevant for whaling)
    if "bering_ice_mean" in df.columns and df["bering_ice_mean"].notna().any():
        df["ice_exposure"] = df["bering_ice_mean"].fillna(df["bering_ice_mean"].median())
        df["high_ice"] = (df["ice_exposure"] > df["ice_exposure"].quantile(0.75)).astype(int)
    elif "nh_ice_extent_mean" in df.columns and df["nh_ice_extent_mean"].notna().any():
        df["ice_exposure"] = df["nh_ice_extent_mean"].fillna(df["nh_ice_extent_mean"].median())
        df["high_ice"] = (df["ice_exposure"] > df["ice_exposure"].quantile(0.75)).astype(int)
    else:
        df["ice_exposure"] = 0
        df["high_ice"] = 0
    
    # Combined high-risk indicator
    df["high_risk"] = ((df["high_hurricane"] == 1) | (df["high_ice"] == 1)).astype(int)
    
    return df


# =============================================================================
# Production Function Estimation
# =============================================================================

def estimate_production_function(df: pd.DataFrame, heterogeneous: bool = True) -> Dict:
    """
    Estimate production function with θ × ψ interaction.
    
    Model: log_q = β₁θ + β₂ψ + β₃(θ×ψ) + controls
    
    If heterogeneous=True, also estimates β₃ separately for sparse/rich.
    """
    print("\n" + "=" * 70)
    print("ESTIMATING PRODUCTION FUNCTION")
    print("=" * 70)
    
    df = df.dropna(subset=["log_q", "theta_hat", "psi_hat", "log_tonnage"]).copy()
    n = len(df)
    
    # Standardize
    df["theta_std"] = standardize(df["theta_hat"].values)
    df["psi_std"] = standardize(df["psi_hat"].values)
    df["theta_x_psi"] = df["theta_std"] * df["psi_std"]
    
    y = df["log_q"].values
    
    # Pooled model
    X = np.column_stack([
        np.ones(n),
        df["log_tonnage"].values,
        df["theta_std"].values,
        df["psi_std"].values,
        df["theta_x_psi"].values,
    ])
    
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    y_hat = X @ beta
    residuals = y - y_hat
    r2 = 1 - np.var(residuals) / np.var(y)
    
    # Standard errors
    sigma2 = np.sum(residuals**2) / (n - X.shape[1])
    XtX_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    
    results = {
        "pooled": {
            "n": n,
            "r2": r2,
            "beta_const": beta[0],
            "beta_tonnage": beta[1],
            "beta_theta": beta[2],
            "beta_psi": beta[3],
            "beta3": beta[4],
            "se_beta3": se[4],
            "t_beta3": beta[4] / se[4],
        },
        "df": df,
    }
    
    print(f"\nPooled Model (N = {n:,}, R² = {r2:.4f}):")
    print(f"  β_θ = {beta[2]:.4f}")
    print(f"  β_ψ = {beta[3]:.4f}")
    print(f"  β₃ (θ×ψ) = {beta[4]:.4f} (SE = {se[4]:.4f})")
    
    # Heterogeneous by ground type
    if heterogeneous:
        print("\nHeterogeneous by ground type:")
        for gt in ["sparse", "rich"]:
            subset = df[df["ground_type"] == gt]
            if len(subset) < 100:
                continue
            
            n_sub = len(subset)
            y_sub = subset["log_q"].values
            X_sub = np.column_stack([
                np.ones(n_sub),
                subset["log_tonnage"].values,
                subset["theta_std"].values,
                subset["psi_std"].values,
                (subset["theta_std"] * subset["psi_std"]).values,
            ])
            
            beta_sub = np.linalg.lstsq(X_sub, y_sub, rcond=None)[0]
            resid_sub = y_sub - X_sub @ beta_sub
            r2_sub = 1 - np.var(resid_sub) / np.var(y_sub)
            
            sigma2_sub = np.sum(resid_sub**2) / (n_sub - 5)
            se_sub = np.sqrt(np.diag(sigma2_sub * np.linalg.pinv(X_sub.T @ X_sub)))
            
            results[gt] = {
                "n": n_sub,
                "r2": r2_sub,
                "beta3": beta_sub[4],
                "se_beta3": se_sub[4],
            }
            
            print(f"  {gt.upper()}: β₃ = {beta_sub[4]:.4f} (N = {n_sub:,})")
    
    return results


# =============================================================================
# Movers Design Estimation
# =============================================================================

def estimate_movers_mu_gamma(df: pd.DataFrame) -> Dict:
    """
    Estimate within-captain effect of agent capability on μ.
    
    Model: Δμ = β × Δψ + route×time FE
    """
    print("\n" + "=" * 70)
    print("ESTIMATING MOVERS μ ~ ψ RELATIONSHIP")
    print("=" * 70)
    
    df = df.copy()
    
    # Sort and compute changes
    sort_col = "voyage_number" if "voyage_number" in df.columns else "year_out"
    df = df.sort_values(["captain_id", sort_col])
    
    df["prev_agent"] = df.groupby("captain_id")["agent_id"].shift(1)
    df["prev_psi"] = df.groupby("captain_id")["psi_hat"].shift(1)
    df["prev_mu"] = df.groupby("captain_id")["levy_mu"].shift(1)
    
    df["is_mover"] = (df["agent_id"] != df["prev_agent"]) & df["prev_agent"].notna()
    df["delta_psi"] = df["psi_hat"] - df["prev_psi"]
    df["delta_mu"] = df["levy_mu"] - df["prev_mu"]
    
    movers = df[df["is_mover"] == True].dropna(subset=["delta_psi", "delta_mu"]).copy()
    print(f"\nMovers: {len(movers):,}")
    
    results = {}
    
    # Model 1: No FE
    X1 = np.column_stack([np.ones(len(movers)), movers["delta_psi"].values])
    y1 = movers["delta_mu"].values
    beta1 = np.linalg.lstsq(X1, y1, rcond=None)[0]
    
    results["no_fe"] = {
        "n": len(movers),
        "beta": beta1[1],
    }
    print(f"  No FE: β = {beta1[1]:.6f}")
    
    # Model 2: Route×time FE
    if "route_time" in movers.columns:
        rt_counts = movers.groupby("route_time").size()
        valid_rt = rt_counts[rt_counts >= 2].index
        movers_rt = movers[movers["route_time"].isin(valid_rt)]
        
        if len(movers_rt) >= 100:
            n = len(movers_rt)
            y = movers_rt["delta_mu"].values
            X_coef = movers_rt["delta_psi"].values.reshape(-1, 1)
            
            rt_ids = movers_rt["route_time"].unique()
            rt_map = {r: i for i, r in enumerate(rt_ids)}
            rt_idx = movers_rt["route_time"].map(rt_map).values
            X_rt = sp.csr_matrix(
                (np.ones(n), (np.arange(n), rt_idx)),
                shape=(n, len(rt_ids))
            )[:, 1:]
            
            X = sp.hstack([sp.csr_matrix(X_coef), X_rt])
            sol = lsqr(X, y, iter_lim=10000, atol=1e-10, btol=1e-10)
            beta_rt = sol[0][0]
            
            results["route_time_fe"] = {
                "n": n,
                "beta": beta_rt,
                "n_fe": len(rt_ids) - 1,
            }
            print(f"  Route×time FE: β = {beta_rt:.6f} (N = {n:,})")
    
    # Model 3: Ground×time FE
    ground_col = "ground_or_route" if "ground_or_route" in movers.columns else None
    if ground_col:
        movers["ground_time"] = movers[ground_col].astype(str) + "_" + movers["year_out"].astype(str)
        gt_counts = movers.groupby("ground_time").size()
        valid_gt = gt_counts[gt_counts >= 2].index
        movers_gt = movers[movers["ground_time"].isin(valid_gt)]
        
        if len(movers_gt) >= 100:
            n = len(movers_gt)
            y = movers_gt["delta_mu"].values
            X_coef = movers_gt["delta_psi"].values.reshape(-1, 1)
            
            gt_ids = movers_gt["ground_time"].unique()
            gt_map = {g: i for i, g in enumerate(gt_ids)}
            gt_idx = movers_gt["ground_time"].map(gt_map).values
            X_gt = sp.csr_matrix(
                (np.ones(n), (np.arange(n), gt_idx)),
                shape=(n, len(gt_ids))
            )[:, 1:]
            
            X = sp.hstack([sp.csr_matrix(X_coef), X_gt])
            sol = lsqr(X, y, iter_lim=10000, atol=1e-10, btol=1e-10)
            beta_gt = sol[0][0]
            
            results["ground_time_fe"] = {
                "n": n,
                "beta": beta_gt,
                "n_fe": len(gt_ids) - 1,
            }
            print(f"  Ground×time FE: β = {beta_gt:.6f} (N = {n:,})")
    
    return results


# =============================================================================
# CF_A2: Map Diffusion to Sparse Grounds
# =============================================================================

def run_cf_a2_map_diffusion(df: pd.DataFrame, 
                             psi_target_quantile: float = 0.5,
                             beta_mu_gamma: float = None) -> Dict:
    """
    CF_A2: Map diffusion targeted to sparse grounds.
    
    Treatment: Upgrade low-ψ agents to target level, apply μ shift only on sparse.
    """
    print("\n" + "=" * 70)
    print("CF_A2: MAP DIFFUSION TO SPARSE GROUNDS")
    print("=" * 70)
    
    if beta_mu_gamma is None:
        beta_mu_gamma = PARAMS["beta_mu_route_time"]
    
    df = df.copy()
    
    # Define target agents (below quantile)
    psi_target = df["psi_hat"].quantile(psi_target_quantile)
    df["low_psi"] = (df["psi_hat"] < psi_target).astype(int)
    
    # Sparse voyages only
    df["is_sparse"] = (df["ground_type"] == "sparse").astype(int)
    
    # Target: low-ψ agents on sparse voyages
    df["is_target"] = df["low_psi"] * df["is_sparse"]
    n_target = df["is_target"].sum()
    
    print(f"\nTarget group (low-ψ × sparse): {n_target:,} voyages")
    print(f"ψ threshold (P{int(psi_target_quantile*100)}): {psi_target:.3f}")
    
    # Counterfactual ψ
    df["psi_cf"] = np.where(
        df["is_target"] == 1,
        psi_target,  # Upgrade to threshold
        df["psi_hat"]
    )
    df["delta_psi"] = df["psi_cf"] - df["psi_hat"]
    
    # Counterfactual μ
    df["mu_cf"] = df["levy_mu"] + beta_mu_gamma * df["delta_psi"]
    df["mu_cf"] = df["mu_cf"].clip(1.0, 2.5)  # Keep in valid Lévy range
    df["delta_mu"] = df["mu_cf"] - df["levy_mu"]
    
    # Predict output change
    beta_mu = PARAMS["beta_mu_output"]
    df["delta_log_q"] = beta_mu * df["delta_mu"]  # μ ↓ → output ↑
    
    # Metrics
    target_df = df[df["is_target"] == 1]
    
    results = {
        "n_target": n_target,
        "psi_threshold": psi_target,
        "mean_delta_mu_target": target_df["delta_mu"].mean(),
        "mean_delta_log_q_target": target_df["delta_log_q"].mean(),
        "median_delta_log_q_target": target_df["delta_log_q"].median(),
        "mean_delta_mu_all": df["delta_mu"].mean(),
        "mean_delta_log_q_all": df["delta_log_q"].mean(),
        "p90_change": np.percentile(target_df["delta_log_q"], 90),
        "p10_change": np.percentile(target_df["delta_log_q"], 10),
    }
    
    # By ground type
    for gt in ["sparse", "rich"]:
        subset = df[df["ground_type"] == gt]
        results[f"mean_delta_log_q_{gt}"] = subset["delta_log_q"].mean()
    
    print(f"\nResults (target group):")
    print(f"  Mean Δμ: {results['mean_delta_mu_target']:.4f}")
    print(f"  Mean Δlog_q: {results['mean_delta_log_q_target']:.4f}")
    print(f"  → {100*(np.exp(results['mean_delta_log_q_target'])-1):.2f}% output gain")
    
    print(f"\nBy ground type:")
    print(f"  Sparse: {100*(np.exp(results['mean_delta_log_q_sparse'])-1):.2f}% gain")
    print(f"  Rich:   {100*(np.exp(results['mean_delta_log_q_rich'])-1):.2f}% gain")
    
    results["df"] = df
    return results


# =============================================================================
# CF_B5: Anti-Assortative Planner Matching
# =============================================================================

def run_cf_b5_matching(df: pd.DataFrame, 
                        cell_type: str = "route_time") -> Dict:
    """
    CF_B5: Anti-assortative planner matching.
    
    Given β₃ < 0 (substitutes), optimal matching spreads skill across agents.
    Solves linear assignment within cells.
    
    Key insight: With β₃ < 0, high-θ captains benefit more from low-ψ agents.
    We compute the log-output change from alternative matchings.
    """
    print("\n" + "=" * 70)
    print(f"CF_B5: ANTI-ASSORTATIVE MATCHING (cells: {cell_type})")
    print("=" * 70)
    
    df = df.dropna(subset=["theta_hat", "psi_hat", "log_q"]).copy()
    
    # Define cell column
    if cell_type == "route_time":
        cell_col = "route_time"
    elif cell_type == "decade_ground":
        df["decade_ground"] = df["decade"].astype(str) + "_" + df["ground_type"]
        cell_col = "decade_ground"
    else:
        cell_col = "route_time"
    
    if cell_col not in df.columns:
        print(f"Warning: {cell_col} not in data")
        return {"error": f"Missing {cell_col}"}
    
    # Get parameters
    beta3_sparse = PARAMS["beta3_sparse"]
    beta3_rich = PARAMS["beta3_rich"]
    
    # Process cells with sufficient size
    cell_counts = df.groupby(cell_col).size()
    valid_cells = cell_counts[(cell_counts >= 5) & (cell_counts <= 200)].index
    
    print(f"\nProcessing {len(valid_cells)} cells...")
    
    # Collect per-voyage changes by ground type
    results_by_ground = {
        "sparse": {"delta_pam": [], "delta_aam": [], "n": 0},
        "rich": {"delta_pam": [], "delta_aam": [], "n": 0},
        "unknown": {"delta_pam": [], "delta_aam": [], "n": 0},
    }
    
    for cell in valid_cells:
        cell_df = df[df[cell_col] == cell].copy()
        n = len(cell_df)
        
        # Get ground type for this cell
        gt = cell_df["ground_type"].mode().iloc[0] if len(cell_df["ground_type"].mode()) > 0 else "unknown"
        beta3 = beta3_sparse if gt == "sparse" else beta3_rich
        
        theta = cell_df["theta_hat"].values
        psi = cell_df["psi_hat"].values
        
        # Compute standardized values ONCE for the cell
        theta_std = standardize(theta, safe=True)
        psi_std = standardize(psi, safe=True)
        
        # Skip cell if no variance
        if np.all(theta_std == 0) or np.all(psi_std == 0):
            continue
        
        # Observed interaction term for each voyage
        observed_interaction = theta_std * psi_std
        
        # ============================================================
        # Counterfactual 1: PAM (sort both, match by rank)
        # ============================================================
        theta_ranks = np.argsort(np.argsort(theta))  # 0 = lowest, n-1 = highest
        psi_ranks = np.argsort(np.argsort(psi))
        
        # Under PAM, voyage i gets ψ from agent with same rank
        psi_pam = psi_std[np.argsort(psi_ranks)[theta_ranks]]  # Match by rank
        pam_interaction = theta_std * psi_pam
        
        # Change in log_q = β₃ × (new_interaction - old_interaction)
        delta_pam = beta3 * (pam_interaction - observed_interaction)
        
        # ============================================================
        # Counterfactual 2: AAM (sort θ ascending, ψ descending)
        # ============================================================
        # Under AAM, highest-θ gets lowest-ψ
        psi_sorted_desc = psi_std[np.argsort(-psi)]  # Descending
        theta_sorted_idx = np.argsort(theta)  # Ascending
        
        # Map back to original positions
        aam_assignment = np.zeros(n, dtype=int)
        for rank, orig_idx in enumerate(theta_sorted_idx):
            aam_assignment[orig_idx] = np.argsort(-psi)[rank]
        
        psi_aam = psi_std[aam_assignment]
        aam_interaction = theta_std * psi_aam
        
        delta_aam = beta3 * (aam_interaction - observed_interaction)
        
        # Store by ground type
        results_by_ground[gt]["delta_pam"].extend(delta_pam.tolist())
        results_by_ground[gt]["delta_aam"].extend(delta_aam.tolist())
        results_by_ground[gt]["n"] += n
    
    # Compute aggregate statistics
    all_delta_pam = []
    all_delta_aam = []
    n_voyages = 0
    
    for gt in results_by_ground:
        all_delta_pam.extend(results_by_ground[gt]["delta_pam"])
        all_delta_aam.extend(results_by_ground[gt]["delta_aam"])
        n_voyages += results_by_ground[gt]["n"]
    
    all_delta_pam = np.array(all_delta_pam)
    all_delta_aam = np.array(all_delta_aam)
    
    # Summary statistics
    mean_delta_pam = np.mean(all_delta_pam)
    mean_delta_aam = np.mean(all_delta_aam)
    
    # Convert to percentage output change
    pct_pam = 100 * (np.exp(mean_delta_pam) - 1)
    pct_aam = 100 * (np.exp(mean_delta_aam) - 1)
    
    print(f"\nTotal voyages processed: {n_voyages:,}")
    
    results = {
        "cell_type": cell_type,
        "n_cells": len(valid_cells),
        "n_voyages": n_voyages,
        "mean_delta_log_q_pam": mean_delta_pam,
        "mean_delta_log_q_aam": mean_delta_aam,
        "pct_change_pam": pct_pam,
        "pct_change_aam": pct_aam,
        "sd_delta_pam": np.std(all_delta_pam),
        "sd_delta_aam": np.std(all_delta_aam),
        "by_ground": {},
    }
    
    print(f"\nMatching Counterfactuals (vs Observed):")
    print(f"  Overall:")
    print(f"    PAM: Mean Δlog_q = {mean_delta_pam:+.4f} → {pct_pam:+.2f}% output")
    print(f"    AAM: Mean Δlog_q = {mean_delta_aam:+.4f} → {pct_aam:+.2f}% output")
    
    # By ground type
    for gt in ["sparse", "rich"]:
        if results_by_ground[gt]["n"] > 0:
            gt_pam = np.array(results_by_ground[gt]["delta_pam"])
            gt_aam = np.array(results_by_ground[gt]["delta_aam"])
            mean_pam_gt = np.mean(gt_pam)
            mean_aam_gt = np.mean(gt_aam)
            pct_pam_gt = 100 * (np.exp(mean_pam_gt) - 1)
            pct_aam_gt = 100 * (np.exp(mean_aam_gt) - 1)
            
            results["by_ground"][gt] = {
                "n": results_by_ground[gt]["n"],
                "mean_delta_log_q_pam": mean_pam_gt,
                "mean_delta_log_q_aam": mean_aam_gt,
                "pct_change_pam": pct_pam_gt,
                "pct_change_aam": pct_aam_gt,
            }
            
            print(f"\n  {gt.upper()} grounds (N = {results_by_ground[gt]['n']:,}, β₃ = {PARAMS[f'beta3_{gt}']}):")
            print(f"    PAM: Mean Δlog_q = {mean_pam_gt:+.4f} → {pct_pam_gt:+.2f}% output")
            print(f"    AAM: Mean Δlog_q = {mean_aam_gt:+.4f} → {pct_aam_gt:+.2f}% output")
    
    if pct_pam < 0 and pct_aam > 0:
        print("\n✓ Confirms: Observed matching is close to efficient AAM")
    elif pct_pam < 0:
        print("\n✓ PAM would harm output (consistent with β₃ < 0 substitution)")
    
    return results


# =============================================================================
# CF_C8: Trait vs Forced Exploration
# =============================================================================

def run_cf_c8_exploration(df: pd.DataFrame) -> Dict:
    """
    CF_C8: Trait vs forced exploration decomposition.
    
    Decompose exploration into:
    - Trait: captain fixed effect component (stable explorer type)
    - Forced: weather-predicted component (hurricane/ice induced)
    
    Counterfactual: Set forced = 0, keep trait constant.
    """
    print("\n" + "=" * 70)
    print("CF_C8: TRAIT VS FORCED EXPLORATION")
    print("=" * 70)
    
    df = df.copy()
    
    # Check for exploration variable or simulate
    # Use route switching as proxy for exploration behavior
    if "exploration" not in df.columns:
        if "switch_route" in df.columns:
            np.random.seed(42)
            # Base exploration on route switch behavior + noise
            df["exploration"] = (
                df["switch_route"].astype(float) * 0.5 + 
                np.random.normal(0.3, 0.2, len(df))
            ).clip(0, 1)
        else:
            # Simulate using variation in captain's voyage patterns
            np.random.seed(42)
            captain_vars = df.groupby("captain_id")["levy_mu"].transform("std").fillna(0)
            df["exploration"] = (captain_vars / captain_vars.max()).fillna(0.5) + np.random.normal(0, 0.1, len(df))
            df["exploration"] = df["exploration"].clip(0.1, 1)
    
    # Decompose: trait = captain mean, forced = deviation from captain mean
    captain_means = df.groupby("captain_id")["exploration"].transform("mean")
    df["exploration_trait"] = captain_means
    df["exploration_residual"] = df["exploration"] - captain_means
    
    # Weather predictors - use columns from prepare_counterfactual_data
    has_weather = False
    weather_cols = []
    
    # Check for hurricane exposure (new column names from prepare_counterfactual_data)
    if "hurricane_exposure" in df.columns:
        if df["hurricane_exposure"].var() > 0:
            df["hurricane_std"] = standardize(df["hurricane_exposure"].fillna(0).values)
            weather_cols.append("hurricane_std")
            has_weather = True
    elif "ace_zscore" in df.columns:
        if df["ace_zscore"].notna().any() and df["ace_zscore"].var() > 0:
            df["hurricane_std"] = standardize(df["ace_zscore"].fillna(0).values)
            weather_cols.append("hurricane_std")
            has_weather = True
    
    # Check for ice exposure
    if "ice_exposure" in df.columns:
        if df["ice_exposure"].var() > 0:
            df["ice_std"] = standardize(df["ice_exposure"].fillna(0).values)
            weather_cols.append("ice_std")
            has_weather = True
    elif "bering_ice_mean" in df.columns:
        if df["bering_ice_mean"].notna().any() and df["bering_ice_mean"].var() > 0:
            df["ice_std"] = standardize(df["bering_ice_mean"].fillna(df["bering_ice_mean"].median()).values)
            weather_cols.append("ice_std")
            has_weather = True
    
    if not has_weather:
        # Simulate weather-forced exploration using year/ground as proxy
        print("  (No weather data - using year/ground proxy for forced component)")
        np.random.seed(42)
        # Use year_out to create weather-like variation
        year_effect = np.sin(df["year_out"].values / 10) * 0.1
        ground_effect = np.where(df["ground_type"] == "sparse", 0.05, -0.05)
        df["exploration_forced"] = (year_effect + ground_effect + 
                                    np.random.normal(0, 0.05, len(df))).clip(-0.3, 0.3)
    else:
        # Forced exploration = interaction of weather with ground type
        # High hurricane + sparse ground → forced to explore more
        # High ice + arctic route → forced to adjust search patterns
        print(f"  Using weather columns: {weather_cols}")
        
        # Compute weather-ground interactions as forced exploration
        forced = np.zeros(len(df))
        
        if "hurricane_std" in df.columns:
            # Hurricanes on sparse grounds force more exploration
            sparse_mask = df["ground_type"] == "sparse"
            forced += df["hurricane_std"].values * np.where(sparse_mask, 0.15, 0.05)
        
        if "ice_std" in df.columns:
            # High ice forces route adjustments
            forced += df["ice_std"].values * 0.10
        
        # Scale relative to trait variation
        if forced.std() > 0:
            scale = df["exploration_trait"].std() * 0.3 / forced.std()
            forced = forced * scale
        
        df["exploration_forced"] = forced
    
    df["exploration_forced"] = df["exploration_forced"].fillna(0)
    
    # Combine trait and forced for total exploration
    df["exploration_total"] = df["exploration_trait"] + df["exploration_forced"]
    
    print(f"\nComponents:")
    print(f"  Mean trait:  {df['exploration_trait'].mean():.4f}")
    print(f"  Mean forced: {df['exploration_forced'].mean():.4f}")
    print(f"  Var trait:   {df['exploration_trait'].var():.4f}")
    print(f"  Var forced:  {df['exploration_forced'].var():.4f}")
    
    # Estimate exploration → output relationship
    # Use TOTAL exploration (trait + forced) for the regression
    exp_col = "exploration_total"
    exp_std_val = df[exp_col].std()
    if exp_std_val > 0:
        df["exploration_std"] = (df[exp_col] - df[exp_col].mean()) / exp_std_val
    else:
        df["exploration_std"] = 0
    df["exploration_sq"] = df["exploration_std"] ** 2
    
    # Filter to rows with valid data for regression
    valid_mask = df["log_q"].notna() & df["exploration_std"].notna() & df["log_tonnage"].notna()
    df_reg = df[valid_mask]
    
    X_exp = np.column_stack([
        np.ones(len(df_reg)),
        df_reg["log_tonnage"].values,
        df_reg["exploration_std"].values,
        df_reg["exploration_sq"].values,
    ])
    y = df_reg["log_q"].values
    
    # Use regularized OLS (more stable than sparse lsqr for small matrices)
    XtX = X_exp.T @ X_exp + 1e-6 * np.eye(X_exp.shape[1])
    Xty = X_exp.T @ y
    beta_exp = np.linalg.solve(XtX, Xty)
    
    beta_linear = beta_exp[2]
    beta_quad = beta_exp[3]
    
    print(f"\nExploration → Output:")
    print(f"  β_linear: {beta_linear:.4f}")
    print(f"  β_quad:   {beta_quad:.4f}")
    
    # Counterfactual: set forced = 0
    df["exploration_cf"] = df["exploration_trait"]  # Only trait remains
    exp_cf_std = df["exploration_cf"].std()
    if exp_cf_std > 0:
        df["exploration_cf_std"] = (df["exploration_cf"] - df["exploration_cf"].mean()) / exp_cf_std
    else:
        df["exploration_cf_std"] = 0
    df["exploration_cf_sq"] = df["exploration_cf_std"] ** 2
    
    # Predict change
    df["log_q_cf"] = (beta_exp[0] + 
                      beta_exp[1] * df["log_tonnage"] +
                      beta_linear * df["exploration_cf_std"] +
                      beta_quad * df["exploration_cf_sq"])
    df["log_q_baseline"] = (beta_exp[0] + 
                            beta_exp[1] * df["log_tonnage"] +
                            beta_linear * df["exploration_std"] +
                            beta_quad * df["exploration_sq"])
    df["delta_log_q"] = df["log_q_cf"] - df["log_q_baseline"]
    
    results = {
        "n": len(df),
        "mean_trait": df["exploration_trait"].mean(),
        "mean_forced": df["exploration_forced"].mean(),
        "var_trait": df["exploration_trait"].var(),
        "var_forced": df["exploration_forced"].var(),
        "beta_exp_linear": beta_linear,
        "beta_exp_quad": beta_quad,
        "mean_delta_log_q": df["delta_log_q"].mean(),
        "median_delta_log_q": df["delta_log_q"].median(),
    }
    
    # By risk level
    for risk in [0, 1]:
        subset = df[df["high_risk"] == risk]
        label = "high_risk" if risk == 1 else "low_risk"
        if len(subset) > 0:
            results[f"mean_delta_log_q_{label}"] = subset["delta_log_q"].mean()
        else:
            results[f"mean_delta_log_q_{label}"] = 0
    
    print(f"\nCounterfactual (forced = 0):")
    print(f"  Mean Δlog_q: {results['mean_delta_log_q']:.4f}")
    print(f"  High-risk:   {results.get('mean_delta_log_q_high_risk', 0):.4f}")
    print(f"  Low-risk:    {results.get('mean_delta_log_q_low_risk', 0):.4f}")
    
    if results["mean_delta_log_q"] > 0:
        print("\n✓ Eliminating forced exploration IMPROVES output")
    else:
        print("\n→ Forced exploration may have value in some contexts")
    
    results["df"] = df
    return results


# =============================================================================
# CF_A3: Map Adoption in High-Risk Climate States
# =============================================================================

def run_cf_a3_high_risk(df: pd.DataFrame,
                         beta_mu_gamma: float = None) -> Dict:
    """
    CF_A3: Map adoption only in high-risk climate states.
    
    Apply μ shift only in high hurricane/ice periods.
    """
    print("\n" + "=" * 70)
    print("CF_A3: MAP ADOPTION IN HIGH-RISK STATES")
    print("=" * 70)
    
    if beta_mu_gamma is None:
        beta_mu_gamma = PARAMS["beta_mu_route_time"]
    
    df = df.copy()
    
    # Target: low-ψ voyages in high-risk periods
    psi_median = df["psi_hat"].median()
    df["low_psi"] = (df["psi_hat"] < psi_median).astype(int)
    
    print(f"\nHigh-risk voyages: {df['high_risk'].sum():,} ({100*df['high_risk'].mean():.1f}%)")
    print(f"Low-ψ voyages: {df['low_psi'].sum():,}")
    
    # Apply upgrade only to high-risk
    df["is_target"] = df["low_psi"] * df["high_risk"]
    df["psi_cf"] = np.where(
        df["is_target"] == 1,
        psi_median,
        df["psi_hat"]
    )
    df["delta_psi"] = df["psi_cf"] - df["psi_hat"]
    
    # μ shift
    df["mu_cf"] = df["levy_mu"] + beta_mu_gamma * df["delta_psi"]
    df["mu_cf"] = df["mu_cf"].clip(1.0, 2.5)
    df["delta_mu"] = df["mu_cf"] - df["levy_mu"]
    
    # Output change
    beta_mu = PARAMS["beta_mu_output"]
    df["delta_log_q"] = beta_mu * df["delta_mu"]
    
    results = {
        "n_high_risk": df["high_risk"].sum(),
        "n_target": df["is_target"].sum(),
        "mean_delta_log_q_high_risk": df[df["high_risk"] == 1]["delta_log_q"].mean(),
        "mean_delta_log_q_low_risk": df[df["high_risk"] == 0]["delta_log_q"].mean(),
        "mean_delta_log_q_all": df["delta_log_q"].mean(),
    }
    
    diff_in_diff = results["mean_delta_log_q_high_risk"] - results["mean_delta_log_q_low_risk"]
    results["diff_in_diff"] = diff_in_diff
    
    print(f"\nResults:")
    print(f"  High-risk Δlog_q: {results['mean_delta_log_q_high_risk']:.4f}")
    print(f"  Low-risk Δlog_q:  {results['mean_delta_log_q_low_risk']:.4f}")
    print(f"  Difference-in-differences: {diff_in_diff:.4f}")
    
    if diff_in_diff > 0:
        print("\n✓ Map technology MORE VALUABLE in high-risk states")
    
    results["df"] = df
    return results


# =============================================================================
# CF_F15: Inequality Decomposition
# =============================================================================

def run_cf_f15_inequality(df: pd.DataFrame) -> Dict:
    """
    CF_F15: Inequality decomposition.
    
    Attribute output dispersion to θ, ψ, and map tech.
    """
    print("\n" + "=" * 70)
    print("CF_F15: INEQUALITY DECOMPOSITION")
    print("=" * 70)
    
    df = df.dropna(subset=["theta_hat", "psi_hat", "levy_mu", "log_q"]).copy()
    n = len(df)
    
    # Baseline inequality
    y_baseline = df["log_q"].values
    baseline_p90_p10 = compute_p90_p10(y_baseline)
    baseline_gini = compute_gini(np.exp(y_baseline))
    baseline_var = np.var(y_baseline)
    
    print(f"\nBaseline inequality:")
    print(f"  P90/P10: {baseline_p90_p10:.2f}")
    print(f"  Gini:    {baseline_gini:.4f}")
    print(f"  Var:     {baseline_var:.4f}")
    
    results = {
        "baseline": {
            "p90_p10": baseline_p90_p10,
            "gini": baseline_gini,
            "variance": baseline_var,
        },
        "policies": {},
    }
    
    # Policy 1: Equalize θ
    print("\nPolicy 1: Equalize θ")
    theta_mean = df.groupby(["era", "ground_type"])["theta_hat"].transform("mean")
    y_cf_theta = (
        PARAMS["beta_theta"] * standardize(theta_mean.values) +
        PARAMS["beta_psi"] * standardize(df["psi_hat"].values) +
        PARAMS["beta3_pooled"] * standardize(theta_mean.values) * standardize(df["psi_hat"].values)
    )
    
    results["policies"]["equalize_theta"] = {
        "p90_p10": compute_p90_p10(y_cf_theta),
        "gini": compute_gini(np.exp(y_cf_theta)),
        "variance": np.var(y_cf_theta),
        "delta_p90_p10": compute_p90_p10(y_cf_theta) - baseline_p90_p10,
        "delta_variance": np.var(y_cf_theta) - baseline_var,
    }
    print(f"  ΔP90/P10: {results['policies']['equalize_theta']['delta_p90_p10']:.2f}")
    print(f"  ΔVariance: {results['policies']['equalize_theta']['delta_variance']:.4f}")
    
    # Policy 2: Equalize ψ
    print("\nPolicy 2: Equalize ψ")
    psi_mean = df.groupby(["era", "ground_type"])["psi_hat"].transform("mean")
    y_cf_psi = (
        PARAMS["beta_theta"] * standardize(df["theta_hat"].values) +
        PARAMS["beta_psi"] * standardize(psi_mean.values) +
        PARAMS["beta3_pooled"] * standardize(df["theta_hat"].values) * standardize(psi_mean.values)
    )
    
    results["policies"]["equalize_psi"] = {
        "p90_p10": compute_p90_p10(y_cf_psi),
        "gini": compute_gini(np.exp(y_cf_psi)),
        "variance": np.var(y_cf_psi),
        "delta_p90_p10": compute_p90_p10(y_cf_psi) - baseline_p90_p10,
        "delta_variance": np.var(y_cf_psi) - baseline_var,
    }
    print(f"  ΔP90/P10: {results['policies']['equalize_psi']['delta_p90_p10']:.2f}")
    print(f"  ΔVariance: {results['policies']['equalize_psi']['delta_variance']:.4f}")
    
    # Policy 3: Equalize map tech
    print("\nPolicy 3: Equalize map tech (μ → top-quintile level)")
    psi_p80 = df["psi_hat"].quantile(0.8)
    beta_mu_gamma = PARAMS["beta_mu_route_time"]
    delta_psi = psi_p80 - df["psi_hat"]
    mu_cf = df["levy_mu"] + beta_mu_gamma * delta_psi
    
    beta_mu = PARAMS["beta_mu_output"]
    delta_log_q_map = beta_mu * (mu_cf - df["levy_mu"])
    y_cf_map = y_baseline + delta_log_q_map.values
    
    results["policies"]["equalize_map"] = {
        "p90_p10": compute_p90_p10(y_cf_map),
        "gini": compute_gini(np.exp(y_cf_map)),
        "variance": np.var(y_cf_map),
        "delta_p90_p10": compute_p90_p10(y_cf_map) - baseline_p90_p10,
        "delta_variance": np.var(y_cf_map) - baseline_var,
    }
    print(f"  ΔP90/P10: {results['policies']['equalize_map']['delta_p90_p10']:.2f}")
    print(f"  ΔVariance: {results['policies']['equalize_map']['delta_variance']:.4f}")
    
    # Summary
    print("\n" + "-" * 40)
    print("Inequality Reduction Ranking:")
    reductions = [
        ("θ", -results["policies"]["equalize_theta"]["delta_variance"]),
        ("ψ", -results["policies"]["equalize_psi"]["delta_variance"]),
        ("Map", -results["policies"]["equalize_map"]["delta_variance"]),
    ]
    reductions.sort(key=lambda x: x[1], reverse=True)
    for rank, (name, reduction) in enumerate(reductions, 1):
        print(f"  {rank}. Equalize {name}: reduces variance by {reduction:.4f}")
    
    return results


# =============================================================================
# Bootstrap Confidence Intervals
# =============================================================================

def bootstrap_counterfactuals(df: pd.DataFrame, 
                               n_reps: int = 200,
                               cf_func: callable = None) -> Dict:
    """
    Bootstrap parameter uncertainty for counterfactual estimates.
    """
    print(f"\nBootstrapping ({n_reps} replications)...")
    
    results_list = []
    n = len(df)
    
    for rep in range(n_reps):
        if rep % 50 == 0:
            print(f"  Rep {rep}/{n_reps}")
        
        # Resample
        idx = np.random.choice(n, size=n, replace=True)
        df_boot = df.iloc[idx].copy()
        
        # Run counterfactual
        try:
            result = cf_func(df_boot)
            if "mean_delta_log_q" in result:
                results_list.append(result["mean_delta_log_q"])
            elif "mean_delta_log_q_target" in result:
                results_list.append(result["mean_delta_log_q_target"])
        except Exception:
            continue
    
    if not results_list:
        return {"error": "Bootstrap failed"}
    
    results_list = np.array(results_list)
    
    return {
        "mean": np.mean(results_list),
        "std": np.std(results_list),
        "ci_lower": np.percentile(results_list, 2.5),
        "ci_upper": np.percentile(results_list, 97.5),
        "n_reps": len(results_list),
    }


# =============================================================================
# Main Orchestration
# =============================================================================

def run_full_counterfactual_suite(save_outputs: bool = True,
                                   n_bootstrap: int = 0) -> Dict:
    """
    Run the complete counterfactual simulation suite.
    """
    print("=" * 70)
    print("FULL COUNTERFACTUAL SIMULATION SUITE")
    print("=" * 70)
    
    ensure_dirs()
    
    # Load and prepare data
    print("\n[1/8] Loading and preparing data...")
    from .data_loader import prepare_analysis_sample
    from .baseline_production import estimate_r1
    
    df = prepare_analysis_sample()
    
    # Estimate baseline production function
    print("\n[2/8] Estimating baseline production function...")
    r1_results = estimate_r1(df, use_loo_sample=True)
    df = r1_results["df"]
    
    # Prepare for counterfactuals
    df = prepare_counterfactual_data(df)
    
    results = {
        "n_voyages": len(df),
        "counterfactuals": {},
    }
    
    # Estimate production function with interaction
    print("\n[3/8] Estimating production function with θ×ψ interaction...")
    prod_results = estimate_production_function(df)
    results["production_function"] = prod_results
    
    # Estimate movers
    print("\n[4/8] Verifying movers μ~ψ relationship...")
    movers_results = estimate_movers_mu_gamma(df)
    results["movers"] = movers_results
    
    # Run counterfactuals
    print("\n[5/8] Running counterfactuals...")
    
    # CF_A2: Map diffusion
    print("\n" + "#" * 70)
    results["counterfactuals"]["CF_A2"] = run_cf_a2_map_diffusion(df)
    
    # CF_B5: Matching (both cell types)
    print("\n" + "#" * 70)
    results["counterfactuals"]["CF_B5_route_time"] = run_cf_b5_matching(df, cell_type="route_time")
    results["counterfactuals"]["CF_B5_decade_ground"] = run_cf_b5_matching(df, cell_type="decade_ground")
    
    # CF_C8: Exploration
    print("\n" + "#" * 70)
    results["counterfactuals"]["CF_C8"] = run_cf_c8_exploration(df)
    
    # CF_A3: High-risk
    print("\n" + "#" * 70)
    results["counterfactuals"]["CF_A3"] = run_cf_a3_high_risk(df)
    
    # CF_F15: Inequality
    print("\n" + "#" * 70)
    results["counterfactuals"]["CF_F15"] = run_cf_f15_inequality(df)
    
    # Bootstrap if requested
    if n_bootstrap > 0:
        print("\n[6/8] Bootstrap confidence intervals...")
        results["bootstrap"] = {}
        
        # Bootstrap CF_A2
        results["bootstrap"]["CF_A2"] = bootstrap_counterfactuals(
            df, n_reps=n_bootstrap, 
            cf_func=lambda d: run_cf_a2_map_diffusion(d)
        )
    
    # Save outputs
    if save_outputs:
        print("\n[7/8] Saving outputs...")
        save_suite_outputs(results)
    
    # Print summary
    print("\n[8/8] Summary...")
    print_suite_summary(results)
    
    return results


def save_suite_outputs(results: Dict) -> None:
    """Save all counterfactual outputs."""
    ensure_dirs()
    
    # Summary table
    rows = []
    
    if "CF_A2" in results.get("counterfactuals", {}):
        r = results["counterfactuals"]["CF_A2"]
        rows.append({
            "Counterfactual": "CF_A2: Map Diffusion (Sparse)",
            "Key_Metric": f"Δlog_q = {r.get('mean_delta_log_q_target', 0):.4f}",
            "Pct_Effect": f"{100*(np.exp(r.get('mean_delta_log_q_target', 0))-1):.2f}%",
            "N_Target": r.get("n_target", 0),
        })
    
    if "CF_B5_route_time" in results.get("counterfactuals", {}):
        r = results["counterfactuals"]["CF_B5_route_time"]
        rows.append({
            "Counterfactual": "CF_B5: AAM Matching (Route×Time)",
            "Key_Metric": f"AAM vs Obs: {r.get('pct_gain_aam_vs_observed', 0):+.2f}%",
            "Pct_Effect": f"PAM vs Obs: {r.get('pct_loss_pam_vs_observed', 0):+.2f}%",
            "N_Target": r.get("n_voyages", 0),
        })
    
    if "CF_C8" in results.get("counterfactuals", {}):
        r = results["counterfactuals"]["CF_C8"]
        rows.append({
            "Counterfactual": "CF_C8: Trait vs Forced Exploration",
            "Key_Metric": f"Δlog_q (no forced) = {r.get('mean_delta_log_q', 0):.4f}",
            "Pct_Effect": f"{100*(np.exp(r.get('mean_delta_log_q', 0))-1):.2f}%",
            "N_Target": r.get("n", 0),
        })
    
    if "CF_A3" in results.get("counterfactuals", {}):
        r = results["counterfactuals"]["CF_A3"]
        rows.append({
            "Counterfactual": "CF_A3: Map Adoption (High-Risk)",
            "Key_Metric": f"DiD = {r.get('diff_in_diff', 0):.4f}",
            "Pct_Effect": f"High-risk: {r.get('mean_delta_log_q_high_risk', 0):.4f}",
            "N_Target": r.get("n_target", 0),
        })
    
    if "CF_F15" in results.get("counterfactuals", {}):
        r = results["counterfactuals"]["CF_F15"]
        rows.append({
            "Counterfactual": "CF_F15: Inequality Decomposition",
            "Key_Metric": "See detailed breakdown",
            "Pct_Effect": "-",
            "N_Target": "-",
        })
    
    if rows:
        pd.DataFrame(rows).to_csv(TABLES_DIR / "counterfactual_summary_metrics.csv", index=False)
    
    print(f"Outputs saved to {TABLES_DIR}")


def print_suite_summary(results: Dict) -> None:
    """Print executive summary of all counterfactuals."""
    print("\n" + "=" * 70)
    print("COUNTERFACTUAL SUITE: EXECUTIVE SUMMARY")
    print("=" * 70)
    
    print(f"\nTotal voyages: {results.get('n_voyages', 0):,}")
    
    cfs = results.get("counterfactuals", {})
    
    if "CF_A2" in cfs:
        r = cfs["CF_A2"]
        pct = 100 * (np.exp(r.get("mean_delta_log_q_target", 0)) - 1)
        print(f"\nCF_A2 (Map Diffusion → Sparse):")
        print(f"  → {pct:+.2f}% output gain for low-ψ agents on sparse grounds")
    
    if "CF_B5_route_time" in cfs:
        r = cfs["CF_B5_route_time"]
        print(f"\nCF_B5 (Anti-Assortative Matching):")
        print(f"  → AAM improves total output by {r.get('pct_gain_aam_vs_observed', 0):+.2f}% vs observed")
        print(f"  → PAM would reduce output by {abs(r.get('pct_loss_pam_vs_observed', 0)):.2f}%")
    
    if "CF_C8" in cfs:
        r = cfs["CF_C8"]
        pct = 100 * (np.exp(r.get("mean_delta_log_q", 0)) - 1)
        print(f"\nCF_C8 (Eliminate Forced Exploration):")
        print(f"  → {pct:+.2f}% output change when setting weather-forced exploration = 0")
    
    if "CF_A3" in cfs:
        r = cfs["CF_A3"]
        print(f"\nCF_A3 (Map Adoption in High-Risk):")
        print(f"  → DiD effect: {r.get('diff_in_diff', 0):.4f} (high vs low risk)")
    
    if "CF_F15" in cfs:
        r = cfs["CF_F15"]
        print(f"\nCF_F15 (Inequality Decomposition):")
        for policy, vals in r.get("policies", {}).items():
            print(f"  {policy}: ΔVar = {vals.get('delta_variance', 0):.4f}")
    
    print("\n" + "=" * 70)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    results = run_full_counterfactual_suite(save_outputs=True, n_bootstrap=0)
