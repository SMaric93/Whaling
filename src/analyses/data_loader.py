"""
Data loader for whaling empirical analysis.

Handles sample construction, variable engineering, switch indicators,
period bins, and train/test splits for the regression module.
"""

import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .config import (
    DATA_DIR,
    DEFAULT_SAMPLE,
    SampleConfig,
    CONTROL_VARIABLES,
)

warnings.filterwarnings("ignore", category=FutureWarning)


def load_voyage_data(
    config: Optional[SampleConfig] = None,
    use_climate_data: bool = False,
) -> pd.DataFrame:
    """
    Load and prepare voyage data for analysis.
    
    Parameters
    ----------
    config : SampleConfig, optional
        Sample configuration. Uses DEFAULT_SAMPLE if not provided.
    use_climate_data : bool
        If True, load analysis_voyage_with_climate.parquet for R7-R9.
        
    Returns
    -------
    pd.DataFrame
        Prepared voyage-level dataset.
    """
    if config is None:
        config = DEFAULT_SAMPLE
        
    print("=" * 60)
    print("LOADING AND PREPARING VOYAGE DATA")
    print("=" * 60)
    
    # Select data source
    if use_climate_data:
        data_path = DATA_DIR / "analysis_voyage_with_climate.parquet"
        if not data_path.exists():
            print(f"Climate data not found, falling back to standard voyage data")
            data_path = DATA_DIR / "analysis_voyage.parquet"
    else:
        data_path = DATA_DIR / "analysis_voyage.parquet"
    
    df = pd.read_parquet(data_path)
    print(f"Raw data: {len(df):,} voyages from {data_path.name}")
    
    # Apply year filter
    df = df[df["year_out"].notna()].copy()
    df["year_out"] = df["year_out"].astype(int)
    df = df[(df["year_out"] >= config.min_year) & (df["year_out"] <= config.max_year)]
    print(f"After year filter ({config.min_year}-{config.max_year}): {len(df):,} voyages")
    
    # Filter to valid IDs
    df = df[df["captain_id"].notna() & df["agent_id"].notna()]
    print(f"After ID filter: {len(df):,} voyages")
    
    # Filter to positive production
    df = df[df["q_total_index"] > 0]
    print(f"After positive production filter: {len(df):,} voyages")
    
    # Apply minimum voyage requirements
    captain_counts = df["captain_id"].value_counts()
    agent_counts = df["agent_id"].value_counts()
    valid_captains = captain_counts[captain_counts >= config.min_captain_voyages].index
    valid_agents = agent_counts[agent_counts >= config.min_agent_voyages].index
    df = df[df["captain_id"].isin(valid_captains) & df["agent_id"].isin(valid_agents)]
    print(f"After min voyages filter: {len(df):,} voyages")
    
    print(f"\nPrepared sample:")
    print(f"  Voyages: {len(df):,}")
    print(f"  Unique captains: {df['captain_id'].nunique():,}")
    print(f"  Unique agents: {df['agent_id'].nunique():,}")
    print(f"  Unique vessels: {df['vessel_id'].nunique():,}")
    
    return df


def construct_variables(
    df: pd.DataFrame,
    config: Optional[SampleConfig] = None,
) -> pd.DataFrame:
    """
    Construct all analysis variables from voyage data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw voyage data.
    config : SampleConfig, optional
        Sample configuration.
        
    Returns
    -------
    pd.DataFrame
        Voyage data with constructed variables.
    """
    if config is None:
        config = DEFAULT_SAMPLE
        
    print("\n" + "=" * 60)
    print("CONSTRUCTING ANALYSIS VARIABLES")
    print("=" * 60)
    
    df = df.copy()
    
    # =========================================================================
    # Core outcome variable
    # =========================================================================
    df["log_q"] = np.log(df["q_total_index"])
    
    # Trim extreme outliers
    lower = df["log_q"].quantile(config.output_trim_lower_pct / 100)
    upper = df["log_q"].quantile(config.output_trim_upper_pct / 100)
    n_before = len(df)
    df = df[(df["log_q"] >= lower) & (df["log_q"] <= upper)]
    print(f"Trimmed {n_before - len(df)} outliers ({config.output_trim_lower_pct}%-{config.output_trim_upper_pct}%)")
    
    # =========================================================================
    # Time indices and period bins
    # =========================================================================
    df["decade"] = (df["year_out"] // 10) * 10
    df["period_bin"] = (df["year_out"] // config.period_bin_years) * config.period_bin_years
    
    # Vessel × period fixed effect group
    df["vessel_period"] = df["vessel_id"].astype(str) + "_" + df["period_bin"].astype(str)
    
    # Route × time fixed effect group (use decade for tractability)
    if "route_or_ground" in df.columns:
        df["route_time"] = df["route_or_ground"].astype(str) + "_" + df["decade"].astype(str)
    else:
        # Fallback to home port if route not available
        df["route_time"] = df["home_port"].astype(str) + "_" + df["decade"].astype(str)
        
    # Port × time 
    df["port_time"] = df["home_port"].astype(str) + "_" + df["decade"].astype(str)
    
    print(f"Created period bins (size={config.period_bin_years} years)")
    print(f"  Unique vessel×period groups: {df['vessel_period'].nunique():,}")
    print(f"  Unique route×time groups: {df['route_time'].nunique():,}")
    
    # =========================================================================
    # Control variables
    # =========================================================================
    # Log transformations
    df["log_tonnage"] = np.log(df["tonnage"].clip(lower=1))
    df["log_duration"] = np.log(df["duration_days"].clip(lower=1))
    
    # Fill missing with median
    for col in ["log_tonnage", "log_duration"]:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    
    # Departure month (if available)
    if "departure_month" in df.columns:
        df["departure_month"] = df["departure_month"].fillna(6).astype(int)  # Default to June
    
    # Crew size (if available)
    if "crew_size" in df.columns:
        df["crew_size"] = df["crew_size"].fillna(df["crew_size"].median())
    
    # =========================================================================
    # Arctic/risk indicators
    # =========================================================================
    if "arctic_exposure" in df.columns:
        df["arctic_route"] = (df["arctic_exposure"] > 0).astype(int)
    elif "route_or_ground" in df.columns:
        # Infer from route name
        arctic_keywords = ["arctic", "bering", "hudson", "bowhead", "polar", "ice"]
        df["arctic_route"] = df["route_or_ground"].str.lower().str.contains(
            "|".join(arctic_keywords), na=False
        ).astype(int)
    else:
        df["arctic_route"] = 0
        
    print(f"Arctic routes: {df['arctic_route'].sum():,} ({100*df['arctic_route'].mean():.1f}%)")
    
    # Failure/loss indicators
    if "voyage_outcome" in df.columns:
        df["failure_indicator"] = df["voyage_outcome"].isin(["lost", "condemned", "wrecked"]).astype(int)
    else:
        # Use very low output as proxy
        df["failure_indicator"] = (df["q_total_index"] < df["q_total_index"].quantile(0.05)).astype(int)
    
    # =========================================================================
    # Train/test split for OOS validation
    # =========================================================================
    df["is_train"] = df["year_out"] < config.oos_cutoff_year
    df["is_test"] = df["year_out"] >= config.oos_cutoff_year
    
    n_train = df["is_train"].sum()
    n_test = df["is_test"].sum()
    print(f"\nTrain/test split (cutoff={config.oos_cutoff_year}):")
    print(f"  Train period: {n_train:,} voyages ({100*n_train/len(df):.1f}%)")
    print(f"  Test period: {n_test:,} voyages ({100*n_test/len(df):.1f}%)")
    
    return df


def compute_switch_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute switch indicators for each captain's voyage sequence.
    
    Creates:
    - switch_agent: captain changed agent vs previous voyage
    - switch_vessel: captain changed vessel
    - switch_route: captain changed route/ground
    
    Parameters
    ----------
    df : pd.DataFrame
        Voyage data with captain_id, agent_id, vessel_id, route info.
        
    Returns
    -------
    pd.DataFrame
        Data with switch indicators added.
    """
    print("\n" + "=" * 60)
    print("COMPUTING SWITCH INDICATORS")
    print("=" * 60)
    
    df = df.copy()
    
    # Sort by captain and departure time
    df = df.sort_values(["captain_id", "year_out"])
    
    # Previous voyage info within captain
    df["prev_agent"] = df.groupby("captain_id")["agent_id"].shift(1)
    df["prev_vessel"] = df.groupby("captain_id")["vessel_id"].shift(1)
    
    if "route_or_ground" in df.columns:
        df["prev_route"] = df.groupby("captain_id")["route_or_ground"].shift(1)
    else:
        df["prev_route"] = df.groupby("captain_id")["home_port"].shift(1)
    
    # Switch indicators (1 = switched, 0 = no switch, NaN = first voyage)
    df["switch_agent"] = (df["agent_id"] != df["prev_agent"]).astype(float)
    df["switch_vessel"] = (df["vessel_id"] != df["prev_vessel"]).astype(float)
    df["switch_route"] = (df.get("route_or_ground", df["home_port"]) != df["prev_route"]).astype(float)
    
    # First voyage for captain has no "switch"
    first_voyage_mask = df["prev_agent"].isna()
    df.loc[first_voyage_mask, ["switch_agent", "switch_vessel", "switch_route"]] = np.nan
    
    # Any switch indicator
    df["any_switch"] = ((df["switch_agent"] == 1) | 
                        (df["switch_vessel"] == 1) | 
                        (df["switch_route"] == 1)).astype(float)
    df.loc[first_voyage_mask, "any_switch"] = np.nan
    
    # Summary
    valid_switches = df[~first_voyage_mask]
    print(f"Switch rates (excluding first voyages):")
    print(f"  Agent switches: {valid_switches['switch_agent'].sum():,.0f} ({100*valid_switches['switch_agent'].mean():.1f}%)")
    print(f"  Vessel switches: {valid_switches['switch_vessel'].sum():,.0f} ({100*valid_switches['switch_vessel'].mean():.1f}%)")
    print(f"  Route switches: {valid_switches['switch_route'].sum():,.0f} ({100*valid_switches['switch_route'].mean():.1f}%)")
    print(f"  Any switch: {valid_switches['any_switch'].sum():,.0f} ({100*valid_switches['any_switch'].mean():.1f}%)")
    
    # Clean up temporary columns
    df = df.drop(columns=["prev_agent", "prev_vessel", "prev_route"])
    
    return df


def compute_route_experience(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cumulative route experience for each captain.
    
    RouteExperience_{c,r,t} = number of previous voyages by captain c 
    to route r before time t.
    
    Parameters
    ----------
    df : pd.DataFrame
        Voyage data.
        
    Returns
    -------
    pd.DataFrame
        Data with route_experience column added.
    """
    print("\nComputing route experience...")
    
    df = df.copy()
    df = df.sort_values(["captain_id", "year_out"])
    
    # Use route or port as route identifier
    route_col = "route_or_ground" if "route_or_ground" in df.columns else "home_port"
    
    # Cumulative count within captain-route groups (excluding current voyage)
    df["route_experience"] = df.groupby(["captain_id", route_col]).cumcount()
    
    print(f"  Mean route experience: {df['route_experience'].mean():.2f}")
    print(f"  Max route experience: {df['route_experience'].max()}")
    
    return df


def compute_next_voyage_switch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute whether captain switches agent on NEXT voyage.
    
    Used for R14 switching hazard models.
    
    Parameters
    ----------
    df : pd.DataFrame
        Voyage data with switch_agent computed.
        
    Returns
    -------
    pd.DataFrame
        Data with switch_next column added.
    """
    print("\nComputing next-voyage switch indicator...")
    
    df = df.copy()
    df = df.sort_values(["captain_id", "year_out"])
    
    # Lead the switch indicator
    df["switch_next"] = df.groupby("captain_id")["switch_agent"].shift(-1)
    
    valid = df["switch_next"].notna()
    print(f"  Valid observations: {valid.sum():,}")
    print(f"  Next-voyage switch rate: {df.loc[valid, 'switch_next'].mean()*100:.1f}%")
    
    return df


def prepare_analysis_sample(
    config: Optional[SampleConfig] = None,
    use_climate_data: bool = False,
) -> pd.DataFrame:
    """
    Full pipeline to prepare analysis-ready sample.
    
    Parameters
    ----------
    config : SampleConfig, optional
        Sample configuration.
    use_climate_data : bool
        Whether to load climate-augmented data.
        
    Returns
    -------
    pd.DataFrame
        Analysis-ready voyage panel.
    """
    # Load raw data
    df = load_voyage_data(config, use_climate_data)
    
    # Construct variables
    df = construct_variables(df, config)
    
    # Switch indicators
    df = compute_switch_indicators(df)
    
    # Route experience
    df = compute_route_experience(df)
    
    # Next-voyage switch
    df = compute_next_voyage_switch(df)
    
    print("\n" + "=" * 60)
    print("ANALYSIS SAMPLE READY")
    print("=" * 60)
    print(f"Final sample: {len(df):,} voyages")
    print(f"Columns: {len(df.columns)}")
    
    return df


def split_train_test(
    df: pd.DataFrame,
    cutoff_year: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test periods.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full voyage data.
    cutoff_year : int, optional
        Year to split on. Uses config default if not provided.
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df)
    """
    if cutoff_year is None:
        cutoff_year = DEFAULT_SAMPLE.oos_cutoff_year
        
    train = df[df["year_out"] < cutoff_year].copy()
    test = df[df["year_out"] >= cutoff_year].copy()
    
    print(f"Train/test split at {cutoff_year}:")
    print(f"  Train: {len(train):,} voyages, {train['captain_id'].nunique():,} captains")
    print(f"  Test: {len(test):,} voyages, {test['captain_id'].nunique():,} captains")
    
    # Captains in both periods
    common_captains = set(train["captain_id"]) & set(test["captain_id"])
    print(f"  Captains in both: {len(common_captains):,}")
    
    return train, test


if __name__ == "__main__":
    # Test the data loader
    df = prepare_analysis_sample()
    print(f"\nSample columns:\n{df.columns.tolist()}")
