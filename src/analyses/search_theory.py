"""
Search Theory Analysis Module.

Implements two analyses to prove captain search skill is a cognitive capability:

1. PLACEBO OCEAN SIMULATION (PO1-PO4)
   - Tests if captain trajectories are structurally superior ex-ante
   - Overlays trajectories on whale density maps from other years
   
2. LÉVY FLIGHT DETECTION (LF1-LF4)
   - Tests if skilled captains follow optimal foraging patterns
   - Lévy flights (μ ≈ 2) are optimal in sparse environments

Mathematical Foundation:
- Lévy Flight: P(d) ~ d^{-μ} where 1 < μ < 3 (optimal at μ ≈ 2)
- Brownian Motion: μ > 3 or exponential decay
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar

from .config import OUTPUT_DIR

# Import staging/final from the top-level config
from ..config import STAGING_DIR, FINAL_DIR


# =============================================================================
# Configuration
# =============================================================================

SEARCH_THEORY_DIR = OUTPUT_DIR / "search_theory"

# Grid resolution for density maps (degrees)
GRID_SIZE = 5.0

# Minimum observations for reliable estimates
MIN_POSITIONS_PER_VOYAGE = 10
MIN_VOYAGES_PER_CAPTAIN = 3
MIN_STEPS_FOR_LEVY = 20


# =============================================================================
# Utility Functions
# =============================================================================

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points (nautical miles).
    
    Parameters
    ----------
    lat1, lon1, lat2, lon2 : float
        Coordinates in decimal degrees.
        
    Returns
    -------
    float
        Distance in nautical miles.
    """
    R = 3440.065  # Earth radius in nautical miles
    
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c


def lat_lon_to_grid(lat: float, lon: float, grid_size: float = GRID_SIZE) -> Tuple[int, int]:
    """Convert lat/lon to grid cell indices."""
    lat_bin = int(np.floor((lat + 90) / grid_size))
    lon_bin = int(np.floor((lon + 180) / grid_size))
    return (lat_bin, lon_bin)


# =============================================================================
# PO1: Whale Density Heat Maps
# =============================================================================

def construct_density_maps(
    positions_df: pd.DataFrame,
    voyage_df: pd.DataFrame,
    grid_size: float = GRID_SIZE,
) -> Dict[int, Dict[Tuple[int, int], float]]:
    """
    PO1: Construct whale density heat maps by year.
    
    Uses voyage production (log_q) as proxy for whale density at location.
    
    Parameters
    ----------
    positions_df : pd.DataFrame
        Daily position observations with voyage_id, lat, lon, year.
    voyage_df : pd.DataFrame
        Voyage data with voyage_id, log_q (production).
    grid_size : float
        Grid cell size in degrees.
        
    Returns
    -------
    Dict[int, Dict[Tuple[int, int], float]]
        Nested dict: year → (lat_bin, lon_bin) → mean_production
    """
    print("\n" + "=" * 60)
    print("PO1: CONSTRUCTING WHALE DENSITY HEAT MAPS")
    print("=" * 60)
    
    # Merge production with positions
    merged = positions_df.merge(
        voyage_df[["voyage_id", "log_q"]],
        on="voyage_id",
        how="inner"
    )
    
    # Drop rows with missing coordinates or years
    merged = merged.dropna(subset=["lat", "lon", "year", "log_q"])
    
    # Add grid cells
    merged["lat_bin"] = ((merged["lat"] + 90) / grid_size).astype(int)
    merged["lon_bin"] = ((merged["lon"] + 180) / grid_size).astype(int)
    merged["year_int"] = merged["year"].astype(int)

    
    # Compute mean production per grid cell per year
    density_by_year = {}
    years = sorted(merged["year_int"].unique())
    
    for year in years:
        year_data = merged[merged["year_int"] == year]
        cell_means = year_data.groupby(["lat_bin", "lon_bin"])["log_q"].mean()
        density_by_year[year] = cell_means.to_dict()
    
    # Summary statistics
    total_cells = sum(len(d) for d in density_by_year.values())
    print(f"\nYears covered: {min(years)} - {max(years)}")
    print(f"Total year-cell observations: {total_cells:,}")
    print(f"Unique years: {len(years)}")
    print(f"Mean cells per year: {total_cells / len(years):.1f}")
    
    return density_by_year


def compute_expected_encounters(
    trajectory: List[Tuple[float, float]],
    density_map: Dict[Tuple[int, int], float],
    grid_size: float = GRID_SIZE,
) -> float:
    """
    Compute expected whale encounters for a trajectory on a density map.
    
    Parameters
    ----------
    trajectory : List[Tuple[float, float]]
        List of (lat, lon) positions.
    density_map : Dict[Tuple[int, int], float]
        Grid cell → mean production mapping.
        
    Returns
    -------
    float
        Sum of expected encounters (production) along trajectory.
    """
    total_encounters = 0.0
    
    for lat, lon in trajectory:
        cell = lat_lon_to_grid(lat, lon, grid_size)
        if cell in density_map:
            total_encounters += density_map[cell]
    
    return total_encounters


# =============================================================================
# PO2: Placebo Skill Scores
# =============================================================================

def compute_placebo_skill_scores(
    positions_df: pd.DataFrame,
    density_maps: Dict[int, Dict],
    voyage_df: pd.DataFrame,
    n_placebo_years: int = 5,
) -> pd.DataFrame:
    """
    PO2: Compute skill scores by comparing actual vs placebo encounters.
    
    Parameters
    ----------
    positions_df : pd.DataFrame
        Daily positions with voyage_id, lat, lon, year.
    density_maps : Dict[int, Dict]
        Year → density map from PO1.
    voyage_df : pd.DataFrame
        Voyage data with captain_id, alpha_hat.
    n_placebo_years : int
        Number of placebo years to average over.
        
    Returns
    -------
    pd.DataFrame
        Voyage-level skill scores.
    """
    print("\n" + "=" * 60)
    print("PO2: COMPUTING PLACEBO SKILL SCORES")
    print("=" * 60)
    
    # Get voyages with sufficient position data
    voyage_positions = positions_df.groupby("voyage_id").agg({
        "lat": list,
        "lon": list,
        "year": "first"
    }).reset_index()
    voyage_positions["n_positions"] = voyage_positions["lat"].apply(len)
    voyage_positions = voyage_positions[
        voyage_positions["n_positions"] >= MIN_POSITIONS_PER_VOYAGE
    ]
    
    print(f"Voyages with ≥{MIN_POSITIONS_PER_VOYAGE} positions: {len(voyage_positions):,}")
    
    available_years = sorted(density_maps.keys())
    results = []
    
    for _, row in voyage_positions.iterrows():
        voyage_id = row["voyage_id"]
        actual_year = int(row["year"])
        trajectory = list(zip(row["lat"], row["lon"]))
        
        # Skip if actual year not in maps
        if actual_year not in density_maps:
            continue
            
        # Actual encounters
        actual_encounters = compute_expected_encounters(
            trajectory, density_maps[actual_year]
        )
        
        # Placebo encounters (average over other years)
        placebo_years = [y for y in available_years 
                        if y != actual_year and abs(y - actual_year) <= 20]
        
        if len(placebo_years) < n_placebo_years:
            placebo_years = [y for y in available_years if y != actual_year]
        
        # Sample placebo years if too many
        if len(placebo_years) > n_placebo_years:
            np.random.seed(hash(voyage_id) % 2**32)
            placebo_years = np.random.choice(
                placebo_years, n_placebo_years, replace=False
            ).tolist()
        
        placebo_encounters = []
        for p_year in placebo_years:
            enc = compute_expected_encounters(trajectory, density_maps[p_year])
            placebo_encounters.append(enc)
        
        mean_placebo = np.mean(placebo_encounters) if placebo_encounters else 0
        
        # Skill score = actual / placebo (>1 means better than expected)
        if mean_placebo > 0:
            skill_score = actual_encounters / mean_placebo
        else:
            skill_score = np.nan
        
        results.append({
            "voyage_id": voyage_id,
            "year": actual_year,
            "n_positions": len(trajectory),
            "actual_encounters": actual_encounters,
            "mean_placebo_encounters": mean_placebo,
            "skill_score": skill_score,
            "n_placebo_years": len(placebo_years),
        })
    
    skill_df = pd.DataFrame(results)
    
    # Merge with captain info
    skill_df = skill_df.merge(
        voyage_df[["voyage_id", "captain_id", "alpha_hat"]],
        on="voyage_id",
        how="left"
    )
    
    valid_scores = skill_df["skill_score"].dropna()
    print(f"\nVoyages with valid skill scores: {len(valid_scores):,}")
    print(f"Mean skill score: {valid_scores.mean():.4f}")
    print(f"Median skill score: {valid_scores.median():.4f}")
    
    return skill_df


# =============================================================================
# PO3: Comparison Baselines
# =============================================================================

def generate_random_walk(
    start_pos: Tuple[float, float],
    n_steps: int,
    mean_step_size: float,
) -> List[Tuple[float, float]]:
    """Generate a random walk trajectory."""
    np.random.seed(42)  # For reproducibility
    trajectory = [start_pos]
    
    for _ in range(n_steps - 1):
        angle = np.random.uniform(0, 2 * np.pi)
        step_size = np.random.exponential(mean_step_size)
        
        # Convert step to lat/lon (simplified)
        dlat = step_size * np.cos(angle) / 60  # 1 degree ≈ 60 nm
        dlon = step_size * np.sin(angle) / (60 * np.cos(np.radians(trajectory[-1][0])))
        
        new_lat = np.clip(trajectory[-1][0] + dlat, -90, 90)
        new_lon = trajectory[-1][1] + dlon
        if new_lon > 180:
            new_lon -= 360
        if new_lon < -180:
            new_lon += 360
            
        trajectory.append((new_lat, new_lon))
    
    return trajectory


def generate_direct_route(
    start_pos: Tuple[float, float],
    end_pos: Tuple[float, float],
    n_steps: int,
) -> List[Tuple[float, float]]:
    """Generate a straight-line trajectory."""
    trajectory = []
    for i in range(n_steps):
        t = i / max(n_steps - 1, 1)
        lat = start_pos[0] + t * (end_pos[0] - start_pos[0])
        lon = start_pos[1] + t * (end_pos[1] - start_pos[1])
        trajectory.append((lat, lon))
    return trajectory


def compute_baseline_comparisons(
    skill_df: pd.DataFrame,
    positions_df: pd.DataFrame,
    density_maps: Dict[int, Dict],
) -> pd.DataFrame:
    """
    PO3: Compare captain trajectories to random walk and direct route baselines.
    """
    print("\n" + "=" * 60)
    print("PO3: COMPUTING BASELINE COMPARISONS")
    print("=" * 60)
    
    # Sample a subset for computational efficiency
    sample_voyages = skill_df.dropna(subset=["skill_score"]).head(500)
    
    results = []
    for _, row in sample_voyages.iterrows():
        voyage_id = row["voyage_id"]
        year = int(row["year"])
        
        if year not in density_maps:
            continue
            
        # Get actual trajectory
        voyage_pos = positions_df[positions_df["voyage_id"] == voyage_id]
        if len(voyage_pos) < MIN_POSITIONS_PER_VOYAGE:
            continue
            
        trajectory = list(zip(voyage_pos["lat"], voyage_pos["lon"]))
        
        # Compute step sizes for random walk
        step_sizes = []
        for i in range(len(trajectory) - 1):
            d = haversine_distance(
                trajectory[i][0], trajectory[i][1],
                trajectory[i+1][0], trajectory[i+1][1]
            )
            step_sizes.append(d)
        mean_step = np.mean(step_sizes) if step_sizes else 50
        
        # Generate baselines
        random_traj = generate_random_walk(trajectory[0], len(trajectory), mean_step)
        direct_traj = generate_direct_route(trajectory[0], trajectory[-1], len(trajectory))
        
        # Compute encounters
        actual_enc = compute_expected_encounters(trajectory, density_maps[year])
        random_enc = compute_expected_encounters(random_traj, density_maps[year])
        direct_enc = compute_expected_encounters(direct_traj, density_maps[year])
        
        results.append({
            "voyage_id": voyage_id,
            "actual_encounters": actual_enc,
            "random_walk_encounters": random_enc,
            "direct_route_encounters": direct_enc,
            "vs_random": actual_enc / random_enc if random_enc > 0 else np.nan,
            "vs_direct": actual_enc / direct_enc if direct_enc > 0 else np.nan,
        })
    
    baseline_df = pd.DataFrame(results)
    
    valid = baseline_df.dropna()
    print(f"\nVoyages with baseline comparisons: {len(valid):,}")
    print(f"Mean actual/random: {valid['vs_random'].mean():.3f}")
    print(f"Mean actual/direct: {valid['vs_direct'].mean():.3f}")
    
    return baseline_df


# =============================================================================
# LF1-LF2: Lévy Flight Detection
# =============================================================================

def compute_step_lengths(positions_df: pd.DataFrame) -> pd.DataFrame:
    """
    LF1: Compute step-length distribution for each voyage.
    
    Returns DataFrame with voyage_id and step lengths.
    """
    print("\n" + "=" * 60)
    print("LF1: COMPUTING STEP-LENGTH DISTRIBUTIONS")
    print("=" * 60)
    
    # Sort by voyage and date
    positions_df = positions_df.sort_values(["voyage_id", "obs_date"])
    
    all_steps = []
    
    for voyage_id, group in positions_df.groupby("voyage_id"):
        if len(group) < 2:
            continue
            
        lats = group["lat"].values
        lons = group["lon"].values
        
        for i in range(len(lats) - 1):
            step_length = haversine_distance(
                lats[i], lons[i], lats[i+1], lons[i+1]
            )
            if step_length > 0:  # Exclude zero-distance steps
                all_steps.append({
                    "voyage_id": voyage_id,
                    "step_length": step_length,
                })
    
    steps_df = pd.DataFrame(all_steps)
    
    print(f"\nTotal steps: {len(steps_df):,}")
    print(f"Voyages with steps: {steps_df['voyage_id'].nunique():,}")
    print(f"Mean step length: {steps_df['step_length'].mean():.2f} nm")
    print(f"Median step length: {steps_df['step_length'].median():.2f} nm")
    
    return steps_df


def fit_power_law_exponent(step_lengths: np.ndarray, x_min: float = 1.0) -> Tuple[float, float]:
    """
    LF2: Fit power-law exponent using maximum likelihood estimation.
    
    P(x) ∝ x^{-μ} for x ≥ x_min
    
    MLE estimator: μ = 1 + n / Σ ln(x_i / x_min)
    
    Parameters
    ----------
    step_lengths : np.ndarray
        Step length values.
    x_min : float
        Minimum value for power-law fit.
        
    Returns
    -------
    Tuple[float, float]
        (mu, standard_error)
    """
    x = step_lengths[step_lengths >= x_min]
    
    if len(x) < MIN_STEPS_FOR_LEVY:
        return np.nan, np.nan
    
    n = len(x)
    log_sum = np.sum(np.log(x / x_min))
    
    if log_sum <= 0:
        return np.nan, np.nan
    
    mu = 1 + n / log_sum
    se = (mu - 1) / np.sqrt(n)
    
    return mu, se


def compute_levy_metrics(
    steps_df: pd.DataFrame,
    voyage_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    LF2-LF3: Compute Lévy flight exponents for each captain.
    
    Returns captain-level μ estimates.
    """
    print("\n" + "=" * 60)
    print("LF2-LF3: COMPUTING LÉVY FLIGHT EXPONENTS")
    print("=" * 60)
    
    # Merge with captain info
    steps_with_captain = steps_df.merge(
        voyage_df[["voyage_id", "captain_id", "alpha_hat"]],
        on="voyage_id",
        how="left"
    )
    
    # Compute voyage-level μ
    voyage_levy = []
    for voyage_id, group in steps_with_captain.groupby("voyage_id"):
        steps = group["step_length"].values
        captain_id = group["captain_id"].iloc[0]
        alpha_hat = group["alpha_hat"].iloc[0]
        
        if len(steps) >= MIN_STEPS_FOR_LEVY:
            mu, se = fit_power_law_exponent(steps)
            voyage_levy.append({
                "voyage_id": voyage_id,
                "captain_id": captain_id,
                "alpha_hat": alpha_hat,
                "n_steps": len(steps),
                "mu": mu,
                "mu_se": se,
            })
    
    voyage_levy_df = pd.DataFrame(voyage_levy)
    
    # Aggregate to captain level
    captain_levy = voyage_levy_df.groupby("captain_id").agg({
        "alpha_hat": "first",
        "n_steps": "sum",
        "mu": "mean",
        "voyage_id": "count",
    }).reset_index()
    captain_levy.columns = ["captain_id", "alpha_hat", "total_steps", "mean_mu", "n_voyages"]
    
    # Filter to captains with sufficient data
    captain_levy = captain_levy[captain_levy["n_voyages"] >= MIN_VOYAGES_PER_CAPTAIN]
    
    # Categorize Lévy vs Brownian
    captain_levy["is_levy"] = (captain_levy["mean_mu"] > 1) & (captain_levy["mean_mu"] < 3)
    captain_levy["mu_deviation"] = (captain_levy["mean_mu"] - 2).abs()
    
    valid_mu = captain_levy["mean_mu"].dropna()
    print(f"\nCaptains with Lévy metrics: {len(valid_mu):,}")
    print(f"Mean μ: {valid_mu.mean():.3f}")
    print(f"Median μ: {valid_mu.median():.3f}")
    print(f"Lévy flights (1 < μ < 3): {captain_levy['is_levy'].sum():,} ({100*captain_levy['is_levy'].mean():.1f}%)")
    
    return captain_levy


# =============================================================================
# PO4 & LF4: Regression Tests
# =============================================================================

def run_skill_regressions(
    skill_df: pd.DataFrame,
    captain_levy: pd.DataFrame,
) -> Dict:
    """
    PO4 & LF4: Run regression tests.
    
    PO4: Skill_Score ~ alpha_hat (expect β > 0)
    LF4: alpha_hat ~ (μ - 2)² (expect β < 0)
    """
    print("\n" + "=" * 60)
    print("PO4 & LF4: SKILL REGRESSION TESTS")
    print("=" * 60)
    
    results = {}
    
    # PO4: Skill Score ~ Alpha
    print("\n--- PO4: Skill Score ~ Captain Skill ---")
    skill_valid = skill_df.dropna(subset=["skill_score", "alpha_hat"])
    
    if len(skill_valid) >= 30:
        y = skill_valid["skill_score"].values
        X = np.column_stack([np.ones(len(skill_valid)), skill_valid["alpha_hat"].values])
        
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_hat = X @ beta
        resid = y - y_hat
        
        n, k = X.shape
        sigma_sq = np.sum(resid**2) / (n - k)
        XtX_inv = np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(sigma_sq * XtX_inv))
        t_stat = beta[1] / se[1]
        p_val = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n-k))
        r2 = 1 - np.var(resid) / np.var(y)
        
        corr = skill_valid["skill_score"].corr(skill_valid["alpha_hat"])
        
        print(f"N = {n:,}")
        print(f"β₁ (alpha_hat) = {beta[1]:.4f}")
        print(f"SE = {se[1]:.4f}")
        print(f"t = {t_stat:.2f}")
        print(f"p = {p_val:.4f}")
        print(f"R² = {r2:.4f}")
        print(f"Corr(skill_score, alpha_hat) = {corr:.4f}")
        
        stars = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
        if beta[1] > 0 and p_val < 0.1:
            print(f"\n✓ High-α captains have EX-ANTE superior search (β = {beta[1]:.4f}{stars})")
        
        results["po4"] = {
            "n": n, "beta": beta[1], "se": se[1], "t": t_stat, "p": p_val, 
            "r2": r2, "corr": corr
        }
    else:
        print(f"Insufficient data (N = {len(skill_valid)})")
        results["po4"] = None
    
    # LF4: Alpha ~ (μ - 2)²
    print("\n--- LF4: Captain Skill ~ Lévy Optimality ---")
    levy_valid = captain_levy.dropna(subset=["mean_mu", "alpha_hat"])
    
    if len(levy_valid) >= 30:
        y = levy_valid["alpha_hat"].values
        mu_dev_sq = (levy_valid["mean_mu"] - 2).values ** 2
        X = np.column_stack([np.ones(len(levy_valid)), mu_dev_sq])
        
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_hat = X @ beta
        resid = y - y_hat
        
        n, k = X.shape
        sigma_sq = np.sum(resid**2) / (n - k)
        XtX_inv = np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(sigma_sq * XtX_inv))
        t_stat = beta[1] / se[1]
        p_val = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n-k))
        r2 = 1 - np.var(resid) / np.var(y)
        
        corr = levy_valid["alpha_hat"].corr(levy_valid["mu_deviation"])
        
        print(f"N = {n:,}")
        print(f"β₁ ((μ-2)²) = {beta[1]:.4f}")
        print(f"SE = {se[1]:.4f}")
        print(f"t = {t_stat:.2f}")
        print(f"p = {p_val:.4f}")
        print(f"R² = {r2:.4f}")
        print(f"Corr(alpha_hat, |μ-2|) = {corr:.4f}")
        
        stars = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
        if beta[1] < 0 and p_val < 0.1:
            print(f"\n✓ Captains with OPTIMAL Lévy (μ≈2) have higher skill (β = {beta[1]:.4f}{stars})")
        
        results["lf4"] = {
            "n": n, "beta": beta[1], "se": se[1], "t": t_stat, "p": p_val,
            "r2": r2, "corr": corr
        }
    else:
        print(f"Insufficient data (N = {len(levy_valid)})")
        results["lf4"] = None
    
    return results


# =============================================================================
# C3.1: Alternative Search Metrics
# =============================================================================

def compute_sinuosity_index(trajectory: List[Tuple[float, float]]) -> float:
    """
    Compute sinuosity index: path_length / straight_line_distance.
    
    Lower values = straighter paths (more ballistic).
    Sinuosity = 1 for perfect straight line.
    
    Parameters
    ----------
    trajectory : List[Tuple[float, float]]
        List of (lat, lon) positions.
        
    Returns
    -------
    float
        Sinuosity index.
    """
    if len(trajectory) < 2:
        return np.nan
    
    # Compute path length (sum of step distances)
    path_length = 0.0
    for i in range(len(trajectory) - 1):
        lat1, lon1 = trajectory[i]
        lat2, lon2 = trajectory[i + 1]
        path_length += haversine_distance(lat1, lon1, lat2, lon2)
    
    # Compute beeline distance
    beeline = haversine_distance(
        trajectory[0][0], trajectory[0][1],
        trajectory[-1][0], trajectory[-1][1]
    )
    
    if beeline <= 0:
        return np.nan
    
    return path_length / beeline


def compute_fractal_dimension(
    trajectory: List[Tuple[float, float]],
    box_sizes: List[float] = None,
) -> float:
    """
    Compute fractal dimension using box-counting algorithm.
    
    D closer to 1 = more linear movement.
    D closer to 2 = space-filling (more Brownian).
    
    Parameters
    ----------
    trajectory : List[Tuple[float, float]]
        List of (lat, lon) positions.
    box_sizes : List[float], optional
        Box sizes in degrees. Default: [0.5, 1, 2, 5, 10].
        
    Returns
    -------
    float
        Fractal dimension D.
    """
    if len(trajectory) < 10:
        return np.nan
    
    if box_sizes is None:
        box_sizes = [0.5, 1.0, 2.0, 5.0, 10.0]
    
    counts = []
    for eps in box_sizes:
        # Discretize to grid
        grid_cells = set()
        for lat, lon in trajectory:
            cell = (int(np.floor(lat / eps)), int(np.floor(lon / eps)))
            grid_cells.add(cell)
        counts.append(len(grid_cells))
    
    # Fit log-log slope
    log_eps = np.log(box_sizes)
    log_counts = np.log(counts)
    
    # Remove any invalid values
    valid = np.isfinite(log_counts) & np.isfinite(log_eps)
    if valid.sum() < 3:
        return np.nan
    
    slope, _ = np.polyfit(log_eps[valid], log_counts[valid], 1)
    
    # D = -slope (N ~ eps^{-D})
    return -slope


def compute_msd_exponent(trajectory: List[Tuple[float, float]]) -> float:
    """
    Compute Mean Squared Displacement (MSD) exponent.
    
    MSD(τ) = <|r(t+τ) - r(t)|²>
    log(MSD) ~ α log(τ)
    
    α closer to 2 = ballistic (superdiffusion)
    α = 1 = Brownian (normal diffusion)
    α < 1 = subdiffusion
    
    Parameters
    ----------
    trajectory : List[Tuple[float, float]]
        List of (lat, lon) positions.
        
    Returns
    -------
    float
        MSD exponent α.
    """
    if len(trajectory) < 20:
        return np.nan
    
    n = len(trajectory)
    tau_values = [1, 2, 5, 10, 20]
    tau_values = [t for t in tau_values if t < n // 2]
    
    if len(tau_values) < 3:
        return np.nan
    
    msd_values = []
    for tau in tau_values:
        displacements_sq = []
        for i in range(n - tau):
            d = haversine_distance(
                trajectory[i][0], trajectory[i][1],
                trajectory[i + tau][0], trajectory[i + tau][1]
            )
            displacements_sq.append(d ** 2)
        msd_values.append(np.mean(displacements_sq))
    
    # Fit log-log slope
    log_tau = np.log(tau_values)
    log_msd = np.log(msd_values)
    
    valid = np.isfinite(log_msd)
    if valid.sum() < 3:
        return np.nan
    
    alpha, _ = np.polyfit(log_tau[valid], log_msd[valid], 1)
    return alpha


def compute_alternative_metrics(positions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all alternative search metrics for each voyage.
    
    Returns DataFrame with voyage_id and metrics: sinuosity, fractal_D, msd_alpha.
    """
    print("\n" + "=" * 60)
    print("C3.1: COMPUTING ALTERNATIVE SEARCH METRICS")
    print("=" * 60)
    
    results = []
    
    for voyage_id, group in positions_df.groupby("voyage_id"):
        if len(group) < MIN_POSITIONS_PER_VOYAGE:
            continue
        
        # Sort by date
        group = group.sort_values("obs_date")
        trajectory = list(zip(group["lat"].values, group["lon"].values))
        
        sinuosity = compute_sinuosity_index(trajectory)
        fractal_D = compute_fractal_dimension(trajectory)
        msd_alpha = compute_msd_exponent(trajectory)
        
        results.append({
            "voyage_id": voyage_id,
            "n_positions": len(trajectory),
            "sinuosity": sinuosity,
            "fractal_D": fractal_D,
            "msd_alpha": msd_alpha,
        })
    
    metrics_df = pd.DataFrame(results)
    
    # Summary
    valid = metrics_df.dropna()
    print(f"\nVoyages with all metrics: {len(valid):,}")
    print(f"Mean sinuosity: {metrics_df['sinuosity'].mean():.3f}")
    print(f"Mean fractal D: {metrics_df['fractal_D'].mean():.3f}")
    print(f"Mean MSD α: {metrics_df['msd_alpha'].mean():.3f}")
    
    return metrics_df


# =============================================================================
# C3.2: Tail Sensitivity Analysis
# =============================================================================

def run_tail_sensitivity_analysis(
    steps_df: pd.DataFrame,
    voyage_df: pd.DataFrame,
    truncation_percentiles: List[float] = None,
) -> Dict:
    """
    C3.2: Re-estimate µ under different tail truncations.
    
    Tests robustness of Lévy exponent estimates to extreme step lengths.
    
    Parameters
    ----------
    steps_df : pd.DataFrame
        Step-length data from compute_step_lengths().
    voyage_df : pd.DataFrame
        Voyage data with organizational effects.
    truncation_percentiles : List[float], optional
        Percentiles to truncate. Default: [1, 5, 10].
        
    Returns
    -------
    Dict
        Results including stability metrics and pass/fail status.
    """
    print("\n" + "=" * 60)
    print("C3.2: TAIL SENSITIVITY ANALYSIS")
    print("=" * 60)
    
    if truncation_percentiles is None:
        truncation_percentiles = [1, 5, 10]
    
    # Baseline: no truncation
    baseline_levy = compute_levy_metrics(steps_df, voyage_df)
    baseline_mu = baseline_levy["mean_mu"].mean()
    
    print(f"\nBaseline mean µ: {baseline_mu:.4f}")
    print(f"Baseline N captains: {len(baseline_levy):,}")
    
    results = [{
        "truncation": "baseline",
        "truncation_pct": 0,
        "mean_mu": baseline_mu,
        "n_captains": len(baseline_levy),
        "deviation_from_baseline": 0.0,
    }]
    
    # Test upper tail truncation
    print("\n--- Upper Tail Truncation ---")
    for pct in truncation_percentiles:
        threshold = steps_df["step_length"].quantile(1 - pct / 100)
        truncated = steps_df[steps_df["step_length"] <= threshold].copy()
        
        n_excluded = len(steps_df) - len(truncated)
        
        levy_truncated = compute_levy_metrics(truncated, voyage_df)
        mu_truncated = levy_truncated["mean_mu"].mean()
        deviation = abs(mu_truncated - baseline_mu) / abs(baseline_mu) * 100
        
        print(f"Top {pct}% excluded: µ = {mu_truncated:.4f} (deviation: {deviation:.1f}%)")
        
        results.append({
            "truncation": f"top_{pct}pct",
            "truncation_pct": pct,
            "mean_mu": mu_truncated,
            "n_captains": len(levy_truncated),
            "deviation_from_baseline": deviation,
        })
    
    # Test lower tail truncation
    print("\n--- Lower Tail Truncation ---")
    for pct in [1, 5]:
        threshold = steps_df["step_length"].quantile(pct / 100)
        truncated = steps_df[steps_df["step_length"] >= threshold].copy()
        
        levy_truncated = compute_levy_metrics(truncated, voyage_df)
        mu_truncated = levy_truncated["mean_mu"].mean()
        deviation = abs(mu_truncated - baseline_mu) / abs(baseline_mu) * 100
        
        print(f"Bottom {pct}% excluded: µ = {mu_truncated:.4f} (deviation: {deviation:.1f}%)")
        
        results.append({
            "truncation": f"bottom_{pct}pct",
            "truncation_pct": -pct,  # Negative to indicate lower tail
            "mean_mu": mu_truncated,
            "n_captains": len(levy_truncated),
            "deviation_from_baseline": deviation,
        })
    
    results_df = pd.DataFrame(results)
    
    # Pass criterion: max deviation < 20%
    max_deviation = results_df["deviation_from_baseline"].max()
    passed = max_deviation < 20.0
    
    print(f"\n--- RESULT ---")
    print(f"Maximum deviation: {max_deviation:.1f}%")
    if passed:
        print("✓ PASS: µ estimates stable under tail truncation (max deviation < 20%)")
    else:
        print("✗ FAIL: µ estimates sensitive to tail truncation")
    
    return {
        "results_df": results_df,
        "baseline_mu": baseline_mu,
        "max_deviation": max_deviation,
        "passed": passed,
    }


# =============================================================================
# C3.3: Direct Efficiency Test
# =============================================================================

def test_search_efficiency(
    voyage_levy: pd.DataFrame,
    voyage_df: pd.DataFrame,
) -> Dict:
    """
    C3.3: Test whether µ predicts productivity (catch rate).
    
    catch_rate = δµ + θ_c + ψ_a + λ_{ground×time} + ε
    
    Parameters
    ----------
    voyage_levy : pd.DataFrame
        Voyage-level Lévy metrics with voyage_id and mu.
    voyage_df : pd.DataFrame
        Voyage data with productivity and fixed effects.
        
    Returns
    -------
    Dict
        Regression results for sparse and rich grounds separately.
    """
    print("\n" + "=" * 60)
    print("C3.3: DIRECT EFFICIENCY TEST (catch_rate ~ µ)")
    print("=" * 60)
    
    # Merge Lévy with voyage data
    df = voyage_df.merge(
        voyage_levy[["voyage_id", "mu"]],
        on="voyage_id",
        how="inner"
    )
    
    # Compute catch rate (barrels per day)
    df["catch_rate"] = (df["q_sperm_bbl"] + df["q_whale_bbl"]) / df["duration_days"]
    df = df[df["catch_rate"] > 0].dropna(subset=["mu", "catch_rate"])
    
    print(f"\nSample size: {len(df):,} voyages")
    print(f"Mean µ: {df['mu'].mean():.3f}")
    print(f"Mean catch rate: {df['catch_rate'].mean():.2f} bbl/day")
    
    # Simple OLS: catch_rate ~ mu
    y = df["catch_rate"].values
    X = np.column_stack([np.ones(len(df)), df["mu"].values])
    
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    y_hat = X @ beta
    resid = y - y_hat
    
    n, k = X.shape
    sigma_sq = np.sum(resid ** 2) / (n - k)
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(sigma_sq * XtX_inv))
    t_stat = beta[1] / se[1]
    p_val = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n - k))
    r2 = 1 - np.var(resid) / np.var(y)
    
    stars = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
    
    print(f"\n--- Regression: catch_rate ~ µ ---")
    print(f"δ (coefficient on µ) = {beta[1]:.4f}{stars}")
    print(f"SE = {se[1]:.4f}")
    print(f"t = {t_stat:.2f}, p = {p_val:.4f}")
    print(f"R² = {r2:.4f}")
    
    if beta[1] < 0 and p_val < 0.1:
        print("\n✓ Lower µ (more ballistic) → higher catch rate (efficiency claim supported)")
    elif beta[1] > 0 and p_val < 0.1:
        print("\n⚠ Higher µ (more Brownian) → higher catch rate (unexpected)")
    else:
        print("\nNo significant relationship between µ and catch rate")
        print("Consider reframing as 'organizational influence on search geometry'")
        print("without strong efficiency claims.")
    
    return {
        "n": n,
        "delta": beta[1],
        "se": se[1],
        "t_stat": t_stat,
        "p_value": p_val,
        "r2": r2,
        "efficiency_supported": beta[1] < 0 and p_val < 0.1,
    }


# =============================================================================
# Main Orchestration
# =============================================================================

def run_search_theory_analysis(
    save_outputs: bool = True,
) -> Dict:
    """
    Run full Search Theory analysis (PO1-PO4, LF1-LF4).
    """
    print("=" * 70)
    print("SEARCH THEORY ANALYSIS")
    print("Proving Ex-Ante Search Skill via Placebo Ocean & Lévy Flights")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    positions_df = pd.read_parquet(STAGING_DIR / "logbook_positions.parquet")
    
    # Use the data loader to get properly prepared voyage data with log_q
    from .data_loader import prepare_analysis_sample
    voyage_df = prepare_analysis_sample()
    
    print(f"Positions: {len(positions_df):,}")
    print(f"Voyages: {len(voyage_df):,}")
    
    # Add alpha_hat if not present
    if "alpha_hat" not in voyage_df.columns:
        from .baseline_production import estimate_r1
        print("\nRunning baseline estimation for alpha_hat...")
        r1_results = estimate_r1(voyage_df, use_loo_sample=True)
        voyage_df = r1_results["df"]

    
    # =========== PLACEBO OCEAN ===========
    
    # PO1: Construct density maps
    density_maps = construct_density_maps(positions_df, voyage_df)
    
    # PO2: Compute placebo skill scores
    skill_df = compute_placebo_skill_scores(positions_df, density_maps, voyage_df)
    
    # PO3: Baseline comparisons
    baseline_df = compute_baseline_comparisons(skill_df, positions_df, density_maps)
    
    # =========== LÉVY FLIGHTS ===========
    
    # LF1: Compute step lengths
    steps_df = compute_step_lengths(positions_df)
    
    # LF2-LF3: Compute Lévy metrics
    captain_levy = compute_levy_metrics(steps_df, voyage_df)
    
    # =========== REGRESSIONS ===========
    
    # PO4 & LF4: Run regressions
    regression_results = run_skill_regressions(skill_df, captain_levy)
    
    # Save outputs
    if save_outputs:
        save_search_theory_outputs(skill_df, baseline_df, captain_levy, regression_results)
    
    # Summary
    print_search_theory_summary(skill_df, captain_levy, regression_results)
    
    return {
        "skill_df": skill_df,
        "baseline_df": baseline_df,
        "captain_levy": captain_levy,
        "regression_results": regression_results,
    }


def save_search_theory_outputs(
    skill_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    captain_levy: pd.DataFrame,
    regression_results: Dict,
) -> None:
    """Save all outputs."""
    SEARCH_THEORY_DIR.mkdir(parents=True, exist_ok=True)
    
    skill_df.to_csv(SEARCH_THEORY_DIR / "placebo_skill_scores.csv", index=False)
    baseline_df.to_csv(SEARCH_THEORY_DIR / "baseline_comparisons.csv", index=False)
    captain_levy.to_csv(SEARCH_THEORY_DIR / "levy_exponents.csv", index=False)
    
    # Summary table
    summary_rows = []
    
    if regression_results.get("po4"):
        r = regression_results["po4"]
        summary_rows.extend([
            {"Metric": "PO4: N", "Value": r["n"]},
            {"Metric": "PO4: β₁ (skill→score)", "Value": r["beta"]},
            {"Metric": "PO4: p-value", "Value": r["p"]},
            {"Metric": "PO4: Corr(skill, score)", "Value": r["corr"]},
        ])
    
    if regression_results.get("lf4"):
        r = regression_results["lf4"]
        summary_rows.extend([
            {"Metric": "LF4: N", "Value": r["n"]},
            {"Metric": "LF4: β₁ ((μ-2)²→α)", "Value": r["beta"]},
            {"Metric": "LF4: p-value", "Value": r["p"]},
            {"Metric": "LF4: Corr(α, |μ-2|)", "Value": r["corr"]},
        ])
    
    # Add Lévy summary stats
    valid_mu = captain_levy["mean_mu"].dropna()
    summary_rows.extend([
        {"Metric": "Mean μ (Lévy exponent)", "Value": valid_mu.mean()},
        {"Metric": "% Lévy flights (1<μ<3)", "Value": 100 * captain_levy["is_levy"].mean()},
    ])
    
    pd.DataFrame(summary_rows).to_csv(
        SEARCH_THEORY_DIR / "search_theory_summary.csv", index=False
    )
    
    print(f"\nOutputs saved to {SEARCH_THEORY_DIR}")


def print_search_theory_summary(
    skill_df: pd.DataFrame,
    captain_levy: pd.DataFrame,
    regression_results: Dict,
) -> None:
    """Print final summary."""
    print("\n" + "=" * 70)
    print("SEARCH THEORY ANALYSIS: SUMMARY")
    print("=" * 70)
    
    # Placebo Ocean
    print("\n--- PLACEBO OCEAN SIMULATION ---")
    valid_skill = skill_df["skill_score"].dropna()
    print(f"Voyages analyzed: {len(valid_skill):,}")
    print(f"Mean skill score: {valid_skill.mean():.4f}")
    print(f"Skill score > 1 (better than placebo): {100*(valid_skill > 1).mean():.1f}%")
    
    if regression_results.get("po4"):
        r = regression_results["po4"]
        stars = "***" if r["p"] < 0.01 else "**" if r["p"] < 0.05 else "*" if r["p"] < 0.1 else ""
        print(f"\nPO4 Regression: Skill_Score ~ alpha_hat")
        print(f"  β₁ = {r['beta']:.4f}{stars} (p = {r['p']:.4f})")
        if r["beta"] > 0 and r["p"] < 0.1:
            print("  → ✓ HIGH-α CAPTAINS HAVE EX-ANTE SUPERIOR SEARCH")
    
    # Lévy Flights
    print("\n--- LÉVY FLIGHT DETECTION ---")
    valid_mu = captain_levy["mean_mu"].dropna()
    print(f"Captains analyzed: {len(valid_mu):,}")
    print(f"Mean μ: {valid_mu.mean():.3f}")
    print(f"Optimal Lévy (μ ≈ 2): {100*((valid_mu > 1.5) & (valid_mu < 2.5)).mean():.1f}%")
    
    if regression_results.get("lf4"):
        r = regression_results["lf4"]
        stars = "***" if r["p"] < 0.01 else "**" if r["p"] < 0.05 else "*" if r["p"] < 0.1 else ""
        print(f"\nLF4 Regression: alpha_hat ~ (μ-2)²")
        print(f"  β₁ = {r['beta']:.4f}{stars} (p = {r['p']:.4f})")
        if r["beta"] < 0 and r["p"] < 0.1:
            print("  → ✓ CAPTAINS WITH OPTIMAL LÉVY (μ≈2) HAVE HIGHER SKILL")
    
    # Final verdict
    print("\n" + "-" * 70)
    print("THEORETICAL CONTRIBUTION")
    print("-" * 70)
    
    po4_success = regression_results.get("po4") and regression_results["po4"]["beta"] > 0 and regression_results["po4"]["p"] < 0.1
    lf4_success = regression_results.get("lf4") and regression_results["lf4"]["beta"] < 0 and regression_results["lf4"]["p"] < 0.1
    
    if po4_success or lf4_success:
        print("""
SEARCH SKILL VALIDATED as a COGNITIVE CAPABILITY:
""")
        if po4_success:
            print("  ✓ Placebo Ocean: High-skill captains' trajectories generate more")
            print("    encounters even on OTHER YEARS' whale maps (ex-ante skill)")
        if lf4_success:
            print("  ✓ Lévy Flights: High-skill captains use mathematically optimal")
            print("    foraging patterns (μ ≈ 2 power-law)")
        print("""
This proves "Search" is not luck but a learnable, portable expertise.
""")
    else:
        print("Mixed evidence for search as cognitive capability.")
        print("Further investigation may be needed.")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    results = run_search_theory_analysis(save_outputs=True)
