"""
Mechanism Analysis: How Organizations Alter Search Geometry.

Uses crew-level data to test mechanism channels:
1. Mate FE: Do mates carry organizational protocols?
2. Crew experience: Do experienced crews → different μ?
3. Greenhand ratio: Less accumulated knowledge → higher μ?
4. Crew stability: Repeat crew members → cultural transmission?

Addresses editorial concern C2: Mechanism opacity.
"""

from typing import Dict, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

from .config import TABLES_DIR, FIGURES_DIR

warnings.filterwarnings("ignore", category=FutureWarning)


def load_crew_data() -> pd.DataFrame:
    """Load and parse crew roster data."""
    from pathlib import Path
    
    crew_path = Path("data/staging/crew_roster.parquet")
    if not crew_path.exists():
        raise FileNotFoundError("Crew roster not found")
    
    crew = pd.read_parquet(crew_path)
    print(f"Loaded {len(crew):,} crew entries")
    
    return crew


def compute_crew_features(crew: pd.DataFrame, voyage_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute voyage-level crew features for mechanism analysis.
    
    Features:
    - crew_size: Number of crew members
    - greenhand_ratio: Fraction of crew who are greenhands
    - experienced_crew_ratio: Fraction with prior voyages
    - mate_id: ID of 1st mate (for mate FE)
    - crew_diversity: Number of unique birthplaces
    - avg_crew_age: Average age of crew
    """
    print("\n--- Computing Crew Features ---")
    
    # Standardize voyage_id format
    crew = crew.copy()
    voyage_df = voyage_df.copy()
    
    # Clean up rank field
    if "rank" in crew.columns:
        crew["rank"] = crew["rank"].fillna("").str.upper().str.strip()
    
    # Compute per-voyage features
    voyage_features = []
    
    for voyage_id in voyage_df["voyage_id"].unique():
        v_crew = crew[crew["voyage_id"] == voyage_id]
        
        if len(v_crew) == 0:
            continue
        
        # Crew size
        crew_size = len(v_crew)
        
        # Greenhand ratio
        n_greenhand = v_crew["rank"].str.contains("GREEN", case=False, na=False).sum()
        greenhand_ratio = n_greenhand / crew_size if crew_size > 0 else np.nan
        
        # Get 1st mate
        mate_rows = v_crew[v_crew["rank"].isin(["1ST MATE", "1 MATE", "MATE"])]
        mate_id = mate_rows["crew_name_clean"].iloc[0] if len(mate_rows) > 0 else None
        
        # Crew diversity (unique birthplaces)
        if "birthplace" in v_crew.columns:
            birthplaces = v_crew["birthplace"].dropna().unique()
            crew_diversity = len(birthplaces)
        else:
            crew_diversity = np.nan
        
        # Average age
        if "age" in v_crew.columns:
            ages = pd.to_numeric(v_crew["age"], errors="coerce")
            avg_age = ages.mean()
        else:
            avg_age = np.nan
        
        # Desertion rate
        if "is_deserted" in v_crew.columns:
            desertion_rate = v_crew["is_deserted"].mean()
        else:
            desertion_rate = np.nan
        
        voyage_features.append({
            "voyage_id": voyage_id,
            "crew_size": crew_size,
            "greenhand_ratio": greenhand_ratio,
            "crew_diversity": crew_diversity,
            "avg_crew_age": avg_age,
            "desertion_rate": desertion_rate,
            "mate_id": mate_id,
        })
    
    features_df = pd.DataFrame(voyage_features)
    print(f"Computed features for {len(features_df):,} voyages")
    print(f"  Mean crew size: {features_df['crew_size'].mean():.1f}")
    print(f"  Mean greenhand ratio: {features_df['greenhand_ratio'].mean():.2%}")
    print(f"  Unique mates: {features_df['mate_id'].nunique():,}")
    
    return features_df


def track_crew_experience(crew: pd.DataFrame, voyage_df: pd.DataFrame) -> pd.DataFrame:
    """
    Track crew members across voyages to compute experience.
    
    For each voyage, compute:
    - avg_prior_voyages: Mean number of prior voyages by crew
    - repeat_crew_ratio: Fraction of crew who sailed on prior voyages
    """
    print("\n--- Tracking Crew Experience ---")
    
    # Need voyage dates to order voyages
    voyage_dates = voyage_df[["voyage_id", "year_out"]].drop_duplicates()
    
    # Merge dates to crew
    crew_with_dates = crew.merge(voyage_dates, on="voyage_id", how="left")
    crew_with_dates = crew_with_dates.dropna(subset=["year_out", "crew_name_clean"])
    
    # Count prior voyages for each crew member
    crew_with_dates = crew_with_dates.sort_values("year_out")
    
    # For each crew member, count cumulative voyages
    crew_with_dates["voyage_num"] = crew_with_dates.groupby("crew_name_clean").cumcount()
    
    # Aggregate to voyage level
    experience_agg = crew_with_dates.groupby("voyage_id").agg({
        "voyage_num": ["mean", "max"],
        "crew_name_clean": "count"
    }).reset_index()
    experience_agg.columns = ["voyage_id", "avg_prior_voyages", "max_prior_voyages", "n_crew"]
    
    # Repeat crew ratio: fraction with voyage_num > 0
    repeat_crew = crew_with_dates[crew_with_dates["voyage_num"] > 0].groupby("voyage_id").size()
    total_crew = crew_with_dates.groupby("voyage_id").size()
    repeat_ratio = (repeat_crew / total_crew).fillna(0).reset_index()
    repeat_ratio.columns = ["voyage_id", "repeat_crew_ratio"]
    
    experience_df = experience_agg.merge(repeat_ratio, on="voyage_id", how="left")
    
    print(f"Tracked experience for {len(experience_df):,} voyages")
    print(f"  Mean prior voyages: {experience_df['avg_prior_voyages'].mean():.2f}")
    print(f"  Mean repeat crew ratio: {experience_df['repeat_crew_ratio'].mean():.2%}")
    
    return experience_df


def run_mechanism_regressions(
    df: pd.DataFrame,
    crew_features: pd.DataFrame,
    experience_df: pd.DataFrame,
) -> Dict:
    """
    Run regressions to test mechanism channels.
    
    Dependent variable: mu (Lévy exponent) or log_q
    """
    print("\n" + "=" * 60)
    print("MECHANISM REGRESSIONS")
    print("=" * 60)
    
    # Merge crew features with voyage data
    merged = df.merge(crew_features, on="voyage_id", how="inner")
    merged = merged.merge(experience_df, on="voyage_id", how="left")
    
    print(f"Merged sample: {len(merged):,} voyages")
    
    # Check for mu column
    mu_col = None
    for col in ["mu", "levy_mu", "search_mu", "mu_ml"]:
        if col in merged.columns:
            mu_col = col
            break
    
    if mu_col is None:
        print("Warning: No μ column found, using log_q as outcome")
        outcome_col = "log_q"
    else:
        outcome_col = mu_col
        print(f"Using {outcome_col} as outcome variable")
    
    # Drop rows with missing outcome
    merged = merged.dropna(subset=[outcome_col])
    print(f"After dropping missing outcome: {len(merged):,}")
    
    results = {}
    
    # Test 1: Greenhand ratio → μ
    print("\n--- Test 1: Greenhand Ratio → Outcome ---")
    test_df = merged.dropna(subset=["greenhand_ratio"])
    if len(test_df) > 100:
        y = test_df[outcome_col].values
        X = np.column_stack([
            np.ones(len(test_df)),
            test_df["greenhand_ratio"].values,
        ])
        
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_hat = X @ beta
        resid = y - y_hat
        se = np.sqrt(np.diag(np.var(resid) * np.linalg.inv(X.T @ X)))
        t_stat = beta[1] / se[1]
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), len(test_df) - 2))
        
        print(f"  N = {len(test_df):,}")
        print(f"  β(greenhand_ratio) = {beta[1]:.4f} (SE: {se[1]:.4f})")
        print(f"  t = {t_stat:.3f}, p = {p_val:.4f}")
        print(f"  Interpretation: {'Higher greenhand → higher μ (less efficient search)' if beta[1] > 0 else 'No effect'}")
        
        results["greenhand_ratio"] = {
            "n": len(test_df),
            "beta": beta[1],
            "se": se[1],
            "t_stat": t_stat,
            "p_value": p_val,
        }
    
    # Test 2: Crew experience → μ
    print("\n--- Test 2: Crew Experience → Outcome ---")
    test_df = merged.dropna(subset=["avg_prior_voyages"])
    if len(test_df) > 100:
        y = test_df[outcome_col].values
        X = np.column_stack([
            np.ones(len(test_df)),
            test_df["avg_prior_voyages"].values,
        ])
        
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_hat = X @ beta
        resid = y - y_hat
        se = np.sqrt(np.diag(np.var(resid) * np.linalg.inv(X.T @ X)))
        t_stat = beta[1] / se[1]
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), len(test_df) - 2))
        
        print(f"  N = {len(test_df):,}")
        print(f"  β(avg_prior_voyages) = {beta[1]:.4f} (SE: {se[1]:.4f})")
        print(f"  t = {t_stat:.3f}, p = {p_val:.4f}")
        print(f"  Interpretation: {'More experienced crew → lower μ (more efficient)' if beta[1] < 0 else 'No protective effect'}")
        
        results["crew_experience"] = {
            "n": len(test_df),
            "beta": beta[1],
            "se": se[1],
            "t_stat": t_stat,
            "p_value": p_val,
        }
    
    # Test 3: Crew diversity → μ
    print("\n--- Test 3: Crew Diversity (Birthplaces) → Outcome ---")
    test_df = merged.dropna(subset=["crew_diversity"])
    if len(test_df) > 100:
        y = test_df[outcome_col].values
        X = np.column_stack([
            np.ones(len(test_df)),
            test_df["crew_diversity"].values,
        ])
        
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_hat = X @ beta
        resid = y - y_hat
        se = np.sqrt(np.diag(np.var(resid) * np.linalg.inv(X.T @ X)))
        t_stat = beta[1] / se[1]
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), len(test_df) - 2))
        
        print(f"  N = {len(test_df):,}")
        print(f"  β(crew_diversity) = {beta[1]:.4f} (SE: {se[1]:.4f})")
        print(f"  t = {t_stat:.3f}, p = {p_val:.4f}")
        
        results["crew_diversity"] = {
            "n": len(test_df),
            "beta": beta[1],
            "se": se[1],
            "t_stat": t_stat,
            "p_value": p_val,
        }
    
    # Test 4: Mate FE (variance decomposition)
    print("\n--- Test 4: Mate Fixed Effects ---")
    test_df = merged.dropna(subset=["mate_id"])
    unique_mates = test_df["mate_id"].nunique()
    
    if unique_mates > 50 and len(test_df) > 500:
        # Simple variance decomposition
        overall_var = test_df[outcome_col].var()
        
        # Between-mate variance
        mate_means = test_df.groupby("mate_id")[outcome_col].mean()
        between_var = mate_means.var()
        
        # Within-mate variance  
        test_df["mate_mean"] = test_df["mate_id"].map(mate_means)
        within_var = (test_df[outcome_col] - test_df["mate_mean"]).var()
        
        mate_share = between_var / (between_var + within_var)
        
        print(f"  N = {len(test_df):,}")
        print(f"  Unique mates: {unique_mates:,}")
        print(f"  Between-mate variance: {between_var:.4f}")
        print(f"  Within-mate variance: {within_var:.4f}")
        print(f"  Mate share of variance: {mate_share:.1%}")
        print(f"  Interpretation: {'Mates carry organizational culture' if mate_share > 0.05 else 'Mate effect is small'}")
        
        results["mate_fe"] = {
            "n": len(test_df),
            "unique_mates": unique_mates,
            "between_var": between_var,
            "within_var": within_var,
            "mate_share": mate_share,
        }
    else:
        print(f"  Insufficient mate variation (N={len(test_df)}, unique mates={unique_mates})")
    
    # Test 5: Repeat crew ratio → μ
    print("\n--- Test 5: Repeat Crew Ratio → Outcome ---")
    test_df = merged.dropna(subset=["repeat_crew_ratio"])
    if len(test_df) > 100:
        y = test_df[outcome_col].values
        X = np.column_stack([
            np.ones(len(test_df)),
            test_df["repeat_crew_ratio"].values,
        ])
        
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_hat = X @ beta
        resid = y - y_hat
        se = np.sqrt(np.diag(np.var(resid) * np.linalg.inv(X.T @ X)))
        t_stat = beta[1] / se[1]
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), len(test_df) - 2))
        
        print(f"  N = {len(test_df):,}")
        print(f"  β(repeat_crew_ratio) = {beta[1]:.4f} (SE: {se[1]:.4f})")
        print(f"  t = {t_stat:.3f}, p = {p_val:.4f}")
        print(f"  Interpretation: {'Repeat crews → cultural transmission' if abs(t_stat) > 1.96 else 'No significant effect'}")
        
        results["repeat_crew"] = {
            "n": len(test_df),
            "beta": beta[1],
            "se": se[1],
            "t_stat": t_stat,
            "p_value": p_val,
        }
    
    return results


def run_within_captain_variation(df: pd.DataFrame) -> Dict:
    """
    Test 6: Within-Captain, Across-Agent Variation.
    
    If same captain has different output with different agents,
    this rules out pure selection and points to organizational effects.
    """
    print("\n--- Test 6: Within-Captain, Across-Agent Variation ---")
    
    # Find captains who worked with multiple agents
    captain_agent_counts = df.groupby("captain_id")["agent_id"].nunique()
    multi_agent_captains = captain_agent_counts[captain_agent_counts > 1].index.tolist()
    
    if len(multi_agent_captains) < 100:
        print(f"  Insufficient multi-agent captains: {len(multi_agent_captains)}")
        return {}
    
    print(f"  Captains with 2+ agents: {len(multi_agent_captains):,}")
    
    # Subset to these captains
    multi_df = df[df["captain_id"].isin(multi_agent_captains)].copy()
    print(f"  Voyages from multi-agent captains: {len(multi_df):,}")
    
    y = multi_df["log_q"].values
    n = len(y)
    
    # Build sparse design matrix for captain FE
    captain_ids = multi_df["captain_id"].unique()
    captain_map = {c: i for i, c in enumerate(captain_ids)}
    captain_idx = multi_df["captain_id"].map(captain_map).values
    X_captain = sp.csr_matrix((np.ones(n), (np.arange(n), captain_idx)), 
                               shape=(n, len(captain_ids)))
    
    result = lsqr(X_captain, y, iter_lim=5000)
    resid_captain = y - X_captain @ result[0]
    r2_captain = 1 - np.var(resid_captain) / np.var(y)
    
    # Add agent FE
    agent_ids = multi_df["agent_id"].unique()
    agent_map = {a: i for i, a in enumerate(agent_ids)}
    agent_idx = multi_df["agent_id"].map(agent_map).values
    X_agent = sp.csr_matrix((np.ones(n), (np.arange(n), agent_idx)), 
                             shape=(n, len(agent_ids)))[:, 1:]
    
    X_full = sp.hstack([X_captain, X_agent])
    result_full = lsqr(X_full, y, iter_lim=5000)
    resid_full = y - X_full @ result_full[0]
    r2_full = 1 - np.var(resid_full) / np.var(y)
    
    incremental_r2 = r2_full - r2_captain
    
    # F-test for joint significance of agent FEs
    n_agents = len(agent_ids) - 1
    rss_r = np.sum(resid_captain**2)
    rss_u = np.sum(resid_full**2)
    f_stat = ((rss_r - rss_u) / n_agents) / (rss_u / (n - len(captain_ids) - n_agents))
    p_val = 1 - stats.f.cdf(f_stat, n_agents, n - len(captain_ids) - n_agents)
    
    print(f"  R² (captain FE only): {r2_captain:.4f}")
    print(f"  R² (captain + agent FE): {r2_full:.4f}")
    print(f"  Incremental R² from agent: {incremental_r2:.4f}")
    print(f"  F-test: F({n_agents}, {n - len(captain_ids) - n_agents}) = {f_stat:.2f}, p < {p_val:.4f}")
    print(f"  Conclusion: Agent effects are {'significant' if p_val < 0.05 else 'not significant'} WITHIN captains")
    
    return {
        "n": n,
        "n_captains": len(captain_ids),
        "n_agents": len(agent_ids),
        "r2_captain": r2_captain,
        "r2_full": r2_full,
        "incremental_r2": incremental_r2,
        "f_stat": f_stat,
        "p_value": p_val,
    }


def run_mate_to_captain_test(df: pd.DataFrame, crew: pd.DataFrame) -> Dict:
    """
    Test 7: Mate-to-Captain Career Paths.
    
    Do mates who become captains perform better with their training agent?
    """
    print("\n--- Test 7: Mate-to-Captain Career Paths ---")
    
    crew = crew.copy()
    crew["rank"] = crew["rank"].fillna("").str.upper().str.strip()
    
    # Find mates
    mates = crew[crew["rank"].isin(["1ST MATE", "1 MATE", "MATE", "2ND MATE", "2 MATE"])]
    mates = mates.dropna(subset=["crew_name_clean"])
    
    # Find captains
    captains = crew[crew["rank"] == "MASTER"]
    captains = captains.dropna(subset=["crew_name_clean"])
    
    print(f"  Unique mate names: {mates['crew_name_clean'].nunique():,}")
    print(f"  Unique captain names: {captains['crew_name_clean'].nunique():,}")
    
    # Find mates who became captains
    mate_names = set(mates["crew_name_clean"].unique())
    captain_names = set(captains["crew_name_clean"].unique())
    promoted = mate_names.intersection(captain_names)
    
    print(f"  Mates who became captains: {len(promoted):,}")
    
    if len(promoted) < 50:
        print("  Insufficient promoted mates for analysis")
        return {}
    
    # Get training agent (agent when first served as mate)
    mate_voyages = mates[mates["crew_name_clean"].isin(promoted)].merge(
        df[["voyage_id", "agent_id", "year_out"]], on="voyage_id", how="inner"
    )
    
    first_mate_voyage = mate_voyages.sort_values("year_out").groupby("crew_name_clean").first().reset_index()
    first_mate_voyage = first_mate_voyage[["crew_name_clean", "agent_id"]].rename(
        columns={"agent_id": "training_agent"}
    )
    
    # Get captain voyages
    captain_voyages = captains[captains["crew_name_clean"].isin(promoted)].merge(
        df[["voyage_id", "agent_id", "log_q", "year_out"]], on="voyage_id", how="inner"
    )
    
    captain_voyages = captain_voyages.merge(first_mate_voyage, on="crew_name_clean", how="inner")
    captain_voyages["same_agent"] = (captain_voyages["agent_id"] == captain_voyages["training_agent"]).astype(int)
    
    print(f"  Captain voyages with known training agent: {len(captain_voyages):,}")
    print(f"  - With training agent: {captain_voyages['same_agent'].sum():,}")
    print(f"  - With different agent: {(1 - captain_voyages['same_agent']).sum():,}")
    
    if len(captain_voyages) < 50:
        return {}
    
    # Regression
    X = np.column_stack([np.ones(len(captain_voyages)), captain_voyages["same_agent"].values])
    y = captain_voyages["log_q"].values
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    se = np.sqrt(np.var(resid) * np.linalg.inv(X.T @ X)[1, 1])
    t_stat = beta[1] / se
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), len(captain_voyages) - 2))
    
    print(f"  β(same_agent) = {beta[1]:.4f} (SE: {se:.4f}), t = {t_stat:.2f}")
    print(f"  Interpretation: Captains perform {'better' if beta[1] > 0 else 'worse'} when with training agent")
    
    return {
        "n": len(captain_voyages),
        "n_promoted": len(promoted),
        "n_with_training_agent": int(captain_voyages["same_agent"].sum()),
        "beta": beta[1],
        "se": se,
        "t_stat": t_stat,
        "p_value": p_val,
    }


def run_agent_network_effects(df: pd.DataFrame) -> Dict:
    """
    Test 8: Agent Network Effects.
    
    Do larger agents (more captains, more voyages) have different outcomes?
    """
    print("\n--- Test 8: Agent Network Effects ---")
    
    # Agent portfolio size
    agent_portfolio = df.groupby("agent_id").agg({
        "voyage_id": "count",
        "captain_id": "nunique",
        "log_q": "mean"
    }).reset_index()
    agent_portfolio.columns = ["agent_id", "n_voyages", "n_captains", "mean_output"]
    
    df_merged = df.merge(agent_portfolio[["agent_id", "n_voyages", "n_captains"]], on="agent_id")
    
    # Regression
    X = np.column_stack([
        np.ones(len(df_merged)),
        df_merged["n_voyages"].values,
        df_merged["n_captains"].values,
    ])
    y = df_merged["log_q"].values
    
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    mse = np.sum(resid**2) / (len(y) - 3)
    se = np.sqrt(np.diag(mse * np.linalg.inv(X.T @ X)))
    
    print(f"  N = {len(df_merged):,}")
    print(f"  β(n_voyages) = {beta[1]:.5f} (SE: {se[1]:.5f}), t = {beta[1]/se[1]:.2f}")
    print(f"  β(n_captains) = {beta[2]:.5f} (SE: {se[2]:.5f}), t = {beta[2]/se[2]:.2f}")
    
    return {
        "n": len(df_merged),
        "beta_n_voyages": beta[1],
        "se_n_voyages": se[1],
        "t_n_voyages": beta[1] / se[1],
        "beta_n_captains": beta[2],
        "se_n_captains": se[2],
        "t_n_captains": beta[2] / se[2],
    }


def run_full_mechanism_analysis(df: pd.DataFrame, save_outputs: bool = True) -> Dict:
    """
    Run complete mechanism analysis suite.
    """
    print("\n" + "=" * 60)
    print("MECHANISM ANALYSIS: HOW ORGANIZATIONS ALTER SEARCH")
    print("=" * 60)
    
    # Load crew data
    crew = load_crew_data()
    
    # Compute features
    crew_features = compute_crew_features(crew, df)
    experience_df = track_crew_experience(crew, df)
    
    # Run basic crew regressions (Tests 1-5)
    results = run_mechanism_regressions(df, crew_features, experience_df)
    
    # Run high-value tests (Tests 6-8)
    results["within_captain"] = run_within_captain_variation(df)
    results["mate_to_captain"] = run_mate_to_captain_test(df, crew)
    results["agent_network"] = run_agent_network_effects(df)
    
    # Summary
    print("\n" + "=" * 60)
    print("MECHANISM ANALYSIS SUMMARY")
    print("=" * 60)
    
    for test_name, test_results in results.items():
        if not test_results:
            continue
        if "p_value" in test_results:
            sig = "***" if test_results["p_value"] < 0.01 else "**" if test_results["p_value"] < 0.05 else "*" if test_results["p_value"] < 0.1 else ""
            if "beta" in test_results:
                print(f"{test_name}: β = {test_results['beta']:.4f} {sig}")
            elif "incremental_r2" in test_results:
                print(f"{test_name}: ΔR² = {test_results['incremental_r2']:.4f} {sig}")
        elif "mate_share" in test_results:
            print(f"{test_name}: Mate share = {test_results['mate_share']:.1%}")
    
    # Save results
    if save_outputs:
        output_path = TABLES_DIR / "mechanism_analysis.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        rows = []
        for test_name, test_results in results.items():
            if test_results:
                row = {"test": test_name, **test_results}
                rows.append(row)
        
        pd.DataFrame(rows).to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
    
    return results


if __name__ == "__main__":
    from .data_loader import prepare_analysis_sample
    
    df = prepare_analysis_sample()
    results = run_full_mechanism_analysis(df)

