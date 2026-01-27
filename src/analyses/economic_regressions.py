"""
Economic Regressions: Petroleum Shock and PDO Instruments.

Implements regression extensions E1-E5 using new economic and climate data:
- E1: Petroleum competition shock (demand-side control)
- E2: Pre/post-petroleum era structural break
- E3: PDO instrument for Pacific productivity
- E4: PDO × agent heterogeneity
- E5: Revenue valuation using WSL market prices
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr
from scipy import stats

# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "final"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "output"
ECONOMIC_DIR = OUTPUT_DIR / "economic"


# =============================================================================
# Data Loading and Merging
# =============================================================================

def load_petroleum_prices() -> pd.DataFrame:
    """Load historical petroleum prices from economic_downloader."""
    # Try to load from raw data
    petroleum_path = RAW_DIR / "economic" / "petroleum_prices.csv"
    
    if petroleum_path.exists():
        df = pd.read_csv(petroleum_path)
        print(f"Loaded petroleum prices: {len(df)} years from {df['year'].min()}-{df['year'].max()}")
        return df
    
    # If not saved yet, run the downloader
    print("Petroleum prices not found, running downloader...")
    try:
        from ..download.economic_downloader import download_petroleum_prices
        df = download_petroleum_prices(use_hardcoded=True)
        petroleum_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(petroleum_path, index=False)
        print(f"Downloaded and saved petroleum prices: {len(df)} years")
        return df
    except ImportError:
        print("WARNING: Could not import economic_downloader, using hardcoded fallback")
        return _get_hardcoded_petroleum_prices()


def _get_hardcoded_petroleum_prices() -> pd.DataFrame:
    """Hardcoded petroleum prices as fallback (EIA historical data)."""
    # Key years from EIA U.S. Field Production of Crude Oil data
    data = [
        (1860, 9.59), (1861, 0.49), (1862, 1.05), (1863, 3.15), (1864, 8.06),
        (1865, 6.59), (1866, 3.74), (1867, 2.41), (1868, 3.63), (1869, 5.64),
        (1870, 3.86), (1871, 4.34), (1872, 3.64), (1873, 1.83), (1874, 1.17),
        (1875, 1.35), (1876, 2.56), (1877, 2.42), (1878, 1.19), (1879, 0.86),
        (1880, 0.95), (1881, 0.86), (1882, 0.78), (1883, 1.00), (1884, 0.84),
        (1885, 0.88), (1886, 0.71), (1887, 0.67), (1888, 0.88), (1889, 0.94),
        (1890, 0.87), (1891, 0.67), (1892, 0.56), (1893, 0.64), (1894, 0.84),
        (1895, 1.36), (1896, 1.18), (1897, 0.79), (1898, 0.91), (1899, 1.29),
        (1900, 1.19), (1901, 0.96), (1902, 0.80), (1903, 0.94), (1904, 0.86),
        (1905, 0.62), (1906, 0.73), (1907, 0.72), (1908, 0.72), (1909, 0.70),
        (1910, 0.61), (1911, 0.61), (1912, 0.74), (1913, 0.95), (1914, 0.81),
        (1915, 0.64), (1916, 1.10), (1917, 1.56), (1918, 1.98), (1919, 2.01),
        (1920, 3.07),
    ]
    df = pd.DataFrame(data, columns=["year", "crude_oil_price_usd"])
    
    # Add derived metrics
    df["log_oil_price"] = np.log(df["crude_oil_price_usd"])
    df["oil_price_change"] = df["crude_oil_price_usd"].pct_change()
    df["oil_price_rel_1860"] = df["crude_oil_price_usd"] / 9.59  # Relative to 1860 peak
    df["petroleum_era"] = 1  # All years in this dataset are post-1859
    
    return df


def load_pdo_index() -> pd.DataFrame:
    """Load PDO climate index from weather data."""
    pdo_path = RAW_DIR / "weather" / "pdo_annual.csv"
    
    if pdo_path.exists():
        df = pd.read_csv(pdo_path)
        print(f"Loaded PDO index: {len(df)} years from {df['year'].min()}-{df['year'].max()}")
        return df
    
    # Try to run downloader
    print("PDO index not found, running downloader...")
    try:
        from ..download.weather_downloader import download_pdo_index
        df = download_pdo_index()
        pdo_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(pdo_path, index=False)
        print(f"Downloaded and saved PDO index: {len(df)} years")
        return df
    except Exception as e:
        print(f"WARNING: Could not load PDO index: {e}")
        return pd.DataFrame()


def merge_economic_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge voyage data with petroleum prices and PDO index.
    
    Parameters
    ----------
    df : pd.DataFrame
        Voyage data with year_out column.
        
    Returns
    -------
    pd.DataFrame
        Voyage data with economic/climate columns merged.
    """
    print("Merging economic and climate data...")
    
    df = df.copy()
    n_start = len(df)
    
    # Petroleum prices
    petroleum = load_petroleum_prices()
    if len(petroleum) > 0:
        df = df.merge(
            petroleum[["year", "crude_oil_price_usd", "log_oil_price", "oil_price_change"]],
            left_on="year_out",
            right_on="year",
            how="left"
        ).drop(columns=["year"], errors="ignore")
        df["petroleum_era"] = (df["year_out"] >= 1860).astype(int)
        print(f"  With petroleum data: {df['crude_oil_price_usd'].notna().sum():,} voyages")
    
    # PDO index
    pdo = load_pdo_index()
    if len(pdo) > 0:
        pdo_cols = ["year", "pdo_annual"]
        if "pdo_phase" in pdo.columns:
            pdo_cols.append("pdo_phase")
        df = df.merge(
            pdo[pdo_cols],
            left_on="year_out",
            right_on="year",
            how="left"
        ).drop(columns=["year"], errors="ignore")
        print(f"  With PDO data: {df['pdo_annual'].notna().sum():,} voyages")
    
    print(f"  Total voyages: {len(df):,} (unchanged)")
    
    return df


# =============================================================================
# Regression Helpers (from weather_regressions.py pattern)
# =============================================================================

def cluster_robust_se(residuals: np.ndarray, X: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    """Compute cluster-robust standard errors."""
    n, k = X.shape
    unique_clusters = np.unique(clusters)
    G = len(unique_clusters)
    
    XtX_inv = np.linalg.pinv(X.T @ X)
    
    meat = np.zeros((k, k))
    for c in unique_clusters:
        mask = clusters == c
        Xi = X[mask]
        ei = residuals[mask]
        score = Xi.T @ ei
        meat += np.outer(score, score)
    
    correction = (G / (G - 1)) * ((n - 1) / (n - k))
    vcov = correction * XtX_inv @ meat @ XtX_inv
    
    return np.sqrt(np.diag(vcov))


def run_ols_with_fe(
    df: pd.DataFrame,
    y_var: str,
    x_vars: list,
    fe_vars: list,
    cluster_var: str = "captain_id",
    label: str = "OLS"
) -> Dict:
    """
    Run OLS regression with fixed effects absorbed.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data.
    y_var : str
        Dependent variable.
    x_vars : list
        Control variables (will get coefficients).
    fe_vars : list
        Fixed effect variables (absorbed).
    cluster_var : str
        Clustering variable.
    label : str
        Label for output.
        
    Returns
    -------
    Dict
        Regression results.
    """
    # Drop missing
    all_vars = [y_var] + x_vars + fe_vars + [cluster_var]
    df_clean = df.dropna(subset=[v for v in all_vars if v in df.columns])
    
    n = len(df_clean)
    if n < 100:
        print(f"WARNING: Small sample size ({n}) for {label}")
    
    y = df_clean[y_var].values
    X_coef = df_clean[x_vars].values
    
    # Build FE design matrices
    fe_matrices = []
    for fe in fe_vars:
        if fe not in df_clean.columns:
            continue
        ids = df_clean[fe].unique()
        id_map = {v: i for i, v in enumerate(ids)}
        idx = df_clean[fe].map(id_map).values
        X_fe = sp.csr_matrix(
            (np.ones(n), (np.arange(n), idx)),
            shape=(n, len(ids))
        )
        fe_matrices.append(X_fe)
    
    X_fe_combined = sp.hstack(fe_matrices) if fe_matrices else None
    
    if X_fe_combined is not None:
        X_full = sp.hstack([sp.csr_matrix(X_coef), X_fe_combined])
    else:
        X_full = sp.csr_matrix(X_coef)
    
    # Solve
    sol = lsqr(X_full, y, iter_lim=10000, atol=1e-10, btol=1e-10)
    beta = sol[0]
    
    coefs = beta[:len(x_vars)]
    y_hat = X_full @ beta
    residuals = y - y_hat
    r2 = 1 - np.var(residuals) / np.var(y)
    
    # Cluster-robust SEs
    se = cluster_robust_se(residuals, X_coef, df_clean[cluster_var].values)
    t_stats = coefs / se
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - X_full.shape[1]))
    
    return {
        "label": label,
        "n": n,
        "r2": r2,
        "variables": x_vars,
        "coefficients": coefs,
        "std_errors": se,
        "t_stats": t_stats,
        "p_values": p_values,
    }


def format_results(results: Dict) -> str:
    """Format regression results for display."""
    lines = [
        f"\n{'=' * 70}",
        f"{results['label']}",
        f"{'=' * 70}",
        f"N = {results['n']:,}   R² = {results['r2']:.4f}",
        "-" * 70,
        f"{'Variable':<35} {'Coef':>10} {'SE':>10} {'t':>8} {'p':>8}",
        "-" * 70,
    ]
    
    for i, var in enumerate(results["variables"]):
        coef = results["coefficients"][i]
        se = results["std_errors"][i]
        t = results["t_stats"][i]
        p = results["p_values"][i]
        
        stars = ""
        if p < 0.01:
            stars = "***"
        elif p < 0.05:
            stars = "**"
        elif p < 0.10:
            stars = "*"
        
        lines.append(f"{var:<35} {coef:>10.4f} {se:>10.4f} {t:>8.2f} {p:>7.4f}{stars}")
    
    lines.append("-" * 70)
    return "\n".join(lines)


# =============================================================================
# Regression Specifications E1-E5
# =============================================================================

def run_e1_petroleum_shock(df: pd.DataFrame) -> Dict:
    """
    E1: Petroleum Competition Shock.
    
    logQ_v = α_c + γ_a + β₁·log_oil_price_t + controls + FEs + ε
    
    Tests whether petroleum competition reduced whale oil production.
    """
    print("\n" + "=" * 70)
    print("E1: PETROLEUM COMPETITION SHOCK")
    print("=" * 70)
    
    df = df.copy()
    
    # Require petroleum data
    df = df[df["log_oil_price"].notna()]
    print(f"Sample with petroleum prices: {len(df):,} voyages ({df['year_out'].min()}-{df['year_out'].max()})")
    
    result = run_ols_with_fe(
        df=df,
        y_var="log_q",
        x_vars=["log_tonnage", "log_oil_price"],
        fe_vars=["captain_id", "agent_id", "route_time"],
        cluster_var="captain_id",
        label="E1: Petroleum Competition (log_q ~ log_oil_price | captain + agent + route×time)"
    )
    
    print(format_results(result))
    
    # Interpretation
    coef_idx = result["variables"].index("log_oil_price")
    coef = result["coefficients"][coef_idx]
    p = result["p_values"][coef_idx]
    
    print(f"\nInterpretation:")
    print(f"  β(log_oil_price) = {coef:.4f}, p = {p:.4f}")
    if coef < 0 and p < 0.10:
        print("  → Higher oil prices REDUCE whale oil output (demand substitution)")
    elif coef > 0 and p < 0.10:
        print("  → Higher oil prices INCREASE whale oil output (supply response?)")
    else:
        print("  → No significant petroleum competition effect detected")
    
    return result


def run_e2_era_structural_break(df: pd.DataFrame) -> Dict:
    """
    E2: Pre/Post-Petroleum Era Structural Break.
    
    logQ_v = α_c + γ_a + β₁·petroleum_era + β₂·(petroleum_era × high_cap_agent) + ε
    
    Tests whether agent effects changed after petroleum disruption.
    """
    print("\n" + "=" * 70)
    print("E2: PRE/POST-PETROLEUM ERA STRUCTURAL BREAK")
    print("=" * 70)
    
    df = df.copy()
    
    # Create high-capability agent indicator
    agent_voyages = df.groupby("agent_id").size()
    df["agent_voyages"] = df["agent_id"].map(agent_voyages)
    df["high_cap_agent"] = (df["agent_voyages"] >= agent_voyages.median()).astype(int)
    
    # Interaction
    df["petroleum_x_highcap"] = df["petroleum_era"] * df["high_cap_agent"]
    
    # Pre/post comparison
    pre_1860 = df[df["petroleum_era"] == 0]
    post_1860 = df[df["petroleum_era"] == 1]
    print(f"Pre-petroleum era (< 1860): {len(pre_1860):,} voyages")
    print(f"Post-petroleum era (>= 1860): {len(post_1860):,} voyages")
    
    result = run_ols_with_fe(
        df=df,
        y_var="log_q",
        x_vars=["log_tonnage", "petroleum_era", "petroleum_x_highcap"],
        fe_vars=["captain_id", "route_time"],
        cluster_var="captain_id",
        label="E2: Era Structural Break (log_q ~ petroleum_era + petroleum_era×high_cap | captain + route×time)"
    )
    
    print(format_results(result))
    
    # Interpretation
    if "petroleum_x_highcap" in result["variables"]:
        coef_idx = result["variables"].index("petroleum_x_highcap")
        coef = result["coefficients"][coef_idx]
        p = result["p_values"][coef_idx]
        print(f"\nInterpretation:")
        print(f"  β(petroleum_era × high_cap_agent) = {coef:.4f}, p = {p:.4f}")
        if coef > 0 and p < 0.10:
            print("  → High-capability agents OUTPERFORMED after petroleum shock")
        elif coef < 0 and p < 0.10:
            print("  → High-capability agents UNDERPERFORMED after petroleum shock")
    
    return result


def run_e3_pdo_first_stage(df: pd.DataFrame) -> Dict:
    """
    E3: PDO as Pacific Productivity Instrument (First Stage).
    
    Pacific_choice_v = π·PDO_t + controls + FEs + ε
    
    Tests whether PDO predicts Pacific route choice (instrument relevance).
    """
    print("\n" + "=" * 70)
    print("E3: PDO INSTRUMENT (FIRST STAGE)")
    print("=" * 70)
    
    df = df.copy()
    
    # Require PDO data
    if "pdo_annual" not in df.columns or df["pdo_annual"].isna().all():
        print("WARNING: No PDO data available - skipping E3")
        return {"label": "E3: PDO First Stage (SKIPPED)", "n": 0}
    
    df = df[df["pdo_annual"].notna()]
    
    # Create Pacific route indicator
    pacific_keywords = ["pacific", "japan", "hawaii", "nw coast", "kodiak", "kamchatka", "ochotsk"]
    route_col = "ground_or_route" if "ground_or_route" in df.columns else "route_or_ground"
    if route_col in df.columns:
        df["pacific_route"] = df[route_col].str.lower().str.contains(
            "|".join(pacific_keywords), na=False
        ).astype(int)
    else:
        df["pacific_route"] = 0
    
    pacific_pct = df["pacific_route"].mean() * 100
    print(f"Sample with PDO: {len(df):,} voyages")
    print(f"Pacific routes: {df['pacific_route'].sum():,} ({pacific_pct:.1f}%)")
    
    if df["pacific_route"].sum() < 50:
        print("WARNING: Too few Pacific voyages for meaningful first stage")
    
    # Standardize PDO
    df["pdo_std"] = (df["pdo_annual"] - df["pdo_annual"].mean()) / df["pdo_annual"].std()
    
    result = run_ols_with_fe(
        df=df,
        y_var="pacific_route",
        x_vars=["log_tonnage", "pdo_std"],
        fe_vars=["port_time"],
        cluster_var="captain_id",
        label="E3: PDO First Stage (Pacific_route ~ PDO | port×time)"
    )
    
    print(format_results(result))
    
    # F-statistic for instrument strength
    if "pdo_std" in result["variables"]:
        pdo_idx = result["variables"].index("pdo_std")
        t_stat = result["t_stats"][pdo_idx]
        f_stat = t_stat ** 2
        print(f"\nInstrument Diagnostics:")
        print(f"  t-stat(PDO) = {t_stat:.2f}")
        print(f"  F-stat(PDO) = {f_stat:.2f}")
        if f_stat >= 10:
            print("  → STRONG instrument (F > 10)")
        else:
            print("  → WEAK instrument (F < 10)")
    
    return result


def run_e4_pdo_heterogeneity(df: pd.DataFrame) -> Dict:
    """
    E4: PDO × Agent Heterogeneity.
    
    logQ_v = β₁·PDO_t + β₂·(PDO_t × high_cap_agent) + α_c + γ_a + ε
    
    Tests whether high-capability agents exploit favorable PDO years.
    """
    print("\n" + "=" * 70)
    print("E4: PDO × AGENT HETEROGENEITY")
    print("=" * 70)
    
    df = df.copy()
    
    if "pdo_annual" not in df.columns or df["pdo_annual"].isna().all():
        print("WARNING: No PDO data available - skipping E4")
        return {"label": "E4: PDO Heterogeneity (SKIPPED)", "n": 0}
    
    df = df[df["pdo_annual"].notna()]
    
    # Standardize PDO
    df["pdo_std"] = (df["pdo_annual"] - df["pdo_annual"].mean()) / df["pdo_annual"].std()
    
    # High-capability agent
    agent_voyages = df.groupby("agent_id").size()
    df["agent_voyages"] = df["agent_id"].map(agent_voyages)
    df["high_cap_agent"] = (df["agent_voyages"] >= agent_voyages.median()).astype(int)
    
    # Interaction
    df["pdo_x_highcap"] = df["pdo_std"] * df["high_cap_agent"]
    
    result = run_ols_with_fe(
        df=df,
        y_var="log_q",
        x_vars=["log_tonnage", "pdo_std", "pdo_x_highcap"],
        fe_vars=["captain_id", "route_time"],
        cluster_var="captain_id",
        label="E4: PDO × Agent (log_q ~ PDO + PDO×high_cap | captain + route×time)"
    )
    
    print(format_results(result))
    
    # Interpretation
    if "pdo_x_highcap" in result["variables"]:
        coef_idx = result["variables"].index("pdo_x_highcap")
        coef = result["coefficients"][coef_idx]
        p = result["p_values"][coef_idx]
        print(f"\nInterpretation:")
        print(f"  β(PDO × high_cap_agent) = {coef:.4f}, p = {p:.4f}")
        if coef > 0 and p < 0.10:
            print("  → High-capability agents EXPLOIT favorable PDO conditions")
        elif coef < 0 and p < 0.10:
            print("  → High-capability agents INSULATED from PDO variation")
    
    return result


def run_e5_revenue_valuation(df: pd.DataFrame) -> Dict:
    """
    E5: Revenue Valuation Using WSL Market Prices.
    
    log_revenue_v = α_c + γ_a + controls + FEs + ε
    where: revenue = quantity × market_price
    
    Tests whether revenue decomposition matches production decomposition.
    """
    print("\n" + "=" * 70)
    print("E5: REVENUE VALUATION (WSL PRICES)")
    print("=" * 70)
    
    df = df.copy()
    
    # Check for market price data
    # For now, use a simple imputation based on historical average
    # (Full implementation would use wsl_market_parser.py output)
    
    # Historical average prices ($/barrel for oil, $/lb for bone)
    # Based on WSL averages circa 1850-1870
    SPERM_OIL_PRICE = 1.30  # $/gallon ≈ $54.60/barrel
    WHALE_OIL_PRICE = 0.60  # $/gallon ≈ $25.20/barrel
    WHALEBONE_PRICE = 0.90  # $/lb
    
    # Compute revenue proxy
    if "q_sperm_bbl" in df.columns and "q_whale_bbl" in df.columns:
        df["revenue_proxy"] = (
            df.get("q_sperm_bbl", 0).fillna(0) * 54.60 +
            df.get("q_whale_bbl", 0).fillna(0) * 25.20 +
            df.get("q_bone_lbs", 0).fillna(0) * WHALEBONE_PRICE
        )
    else:
        # Use total production with average price
        avg_price_per_barrel = 40.0  # Rough average
        df["revenue_proxy"] = df["q_total_index"] * avg_price_per_barrel
    
    df["log_revenue"] = np.log(df["revenue_proxy"].clip(lower=1))
    
    print(f"Sample: {len(df):,} voyages")
    print(f"Revenue range: ${df['revenue_proxy'].min():,.0f} - ${df['revenue_proxy'].max():,.0f}")
    print(f"Mean revenue: ${df['revenue_proxy'].mean():,.0f}")
    
    result = run_ols_with_fe(
        df=df,
        y_var="log_revenue",
        x_vars=["log_tonnage"],
        fe_vars=["captain_id", "agent_id", "route_time"],
        cluster_var="captain_id",
        label="E5: Revenue Decomposition (log_revenue ~ log_tonnage | captain + agent + route×time)"
    )
    
    print(format_results(result))
    
    # Compare with production decomposition
    production_result = run_ols_with_fe(
        df=df,
        y_var="log_q",
        x_vars=["log_tonnage"],
        fe_vars=["captain_id", "agent_id", "route_time"],
        cluster_var="captain_id",
        label="(Comparison) Production Decomposition"
    )
    
    print(f"\nComparison:")
    print(f"  R² (Revenue):   {result['r2']:.4f}")
    print(f"  R² (Production): {production_result['r2']:.4f}")
    
    return result


# =============================================================================
# Main Orchestration
# =============================================================================

def run_all_economic_regressions(
    save_outputs: bool = True,
    use_climate_data: bool = True,
) -> Dict[str, Dict]:
    """
    Run all economic regression specifications (E1-E5).
    
    Parameters
    ----------
    save_outputs : bool
        Whether to save output tables.
    use_climate_data : bool
        Whether to load climate-augmented data.
        
    Returns
    -------
    Dict[str, Dict]
        All regression results.
    """
    print("=" * 70)
    print("ECONOMIC REGRESSION SUITE (E1-E5)")
    print("=" * 70)
    
    # Load base data
    from .data_loader import prepare_analysis_sample
    df = prepare_analysis_sample(use_climate_data=use_climate_data)
    
    # Merge economic data
    df = merge_economic_data(df)
    
    results = {}
    
    # E1: Petroleum shock
    results["E1"] = run_e1_petroleum_shock(df)
    
    # E2: Era structural break
    results["E2"] = run_e2_era_structural_break(df)
    
    # E3: PDO first stage
    results["E3"] = run_e3_pdo_first_stage(df)
    
    # E4: PDO heterogeneity
    results["E4"] = run_e4_pdo_heterogeneity(df)
    
    # E5: Revenue valuation
    results["E5"] = run_e5_revenue_valuation(df)
    
    # Save results
    if save_outputs:
        save_results(results)
    
    # Summary
    print_summary(results)
    
    return results


def save_results(results: Dict[str, Dict]) -> None:
    """Save regression results to files."""
    ECONOMIC_DIR.mkdir(parents=True, exist_ok=True)
    
    rows = []
    for spec_id, res in results.items():
        if res.get("n", 0) == 0:
            continue
        for i, var in enumerate(res.get("variables", [])):
            rows.append({
                "specification": spec_id,
                "label": res["label"],
                "variable": var,
                "coefficient": res["coefficients"][i],
                "std_error": res["std_errors"][i],
                "t_stat": res["t_stats"][i],
                "p_value": res["p_values"][i],
                "n": res["n"],
                "r2": res["r2"],
            })
    
    if rows:
        df = pd.DataFrame(rows)
        output_path = ECONOMIC_DIR / "economic_regressions.csv"
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")


def print_summary(results: Dict[str, Dict]) -> None:
    """Print summary of key findings."""
    print("\n" + "=" * 70)
    print("SUMMARY: KEY FINDINGS FROM ECONOMIC REGRESSIONS")
    print("=" * 70)
    
    findings = []
    
    # E1: Petroleum shock
    if results.get("E1", {}).get("n", 0) > 0:
        e1 = results["E1"]
        if "log_oil_price" in e1["variables"]:
            idx = e1["variables"].index("log_oil_price")
            coef, p = e1["coefficients"][idx], e1["p_values"][idx]
            stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
            findings.append(f"E1: Petroleum Competition: β = {coef:.3f}{stars}")
    
    # E2: Era break
    if results.get("E2", {}).get("n", 0) > 0:
        e2 = results["E2"]
        if "petroleum_x_highcap" in e2["variables"]:
            idx = e2["variables"].index("petroleum_x_highcap")
            coef, p = e2["coefficients"][idx], e2["p_values"][idx]
            stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
            findings.append(f"E2: High-Cap × Petroleum Era: β = {coef:.3f}{stars}")
    
    # E3: PDO first stage
    if results.get("E3", {}).get("n", 0) > 0:
        e3 = results["E3"]
        if "pdo_std" in e3["variables"]:
            idx = e3["variables"].index("pdo_std")
            f_stat = e3["t_stats"][idx] ** 2
            findings.append(f"E3: PDO First Stage: F = {f_stat:.1f} ({'Strong' if f_stat >= 10 else 'Weak'})")
    
    # E4: PDO heterogeneity
    if results.get("E4", {}).get("n", 0) > 0:
        e4 = results["E4"]
        if "pdo_x_highcap" in e4["variables"]:
            idx = e4["variables"].index("pdo_x_highcap")
            coef, p = e4["coefficients"][idx], e4["p_values"][idx]
            stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
            findings.append(f"E4: PDO × High-Cap Agent: β = {coef:.3f}{stars}")
    
    # E5: Revenue
    if results.get("E5", {}).get("n", 0) > 0:
        e5 = results["E5"]
        findings.append(f"E5: Revenue Decomposition: R² = {e5['r2']:.3f}")
    
    for f in findings:
        print(f"  • {f}")
    
    print()


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    results = run_all_economic_regressions(save_outputs=True)
