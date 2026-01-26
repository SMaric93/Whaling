"""
AKM (Abowd-Kramarz-Margolis) Variance Decomposition for Whaling Captains and Agents.

Decomposes voyage productivity (log production) into:
- Captain fixed effects (θ_i) - captain skill/quality
- Agent fixed effects (ψ_j) - agent management ability
- Time-varying controls (Xβ) - voyage characteristics

References:
- Abowd, Kramarz, and Margolis (1999) "High Wage Workers and High Wage Firms"
- Kline, Saggio, and Sølvsten (2020) "Leave-Out Estimation of Variance Components"
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr
import networkx as nx

warnings.filterwarnings("ignore", category=FutureWarning)

# Paths
DATA_DIR = Path(__file__).parent.parent / "data" / "final"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "final"


def load_and_prepare_data(
    min_year: int = 1780,
    max_year: int = 1930,
    min_captain_voyages: int = 1,
    min_agent_voyages: int = 1,
) -> pd.DataFrame:
    """Load voyage data and prepare for AKM estimation."""
    print("=" * 60)
    print("LOADING AND PREPARING DATA")
    print("=" * 60)

    df = pd.read_parquet(DATA_DIR / "analysis_voyage.parquet")
    print(f"Raw data: {len(df):,} voyages")

    # Filter to year range
    df = df[df["year_out"].notna()].copy()
    df["year_out"] = df["year_out"].astype(int)
    df = df[(df["year_out"] >= min_year) & (df["year_out"] <= max_year)]
    print(f"After year filter ({min_year}-{max_year}): {len(df):,} voyages")

    # Filter to valid IDs
    df = df[df["captain_id"].notna() & df["agent_id"].notna()]
    print(f"After ID filter: {len(df):,} voyages")

    # Filter to positive production
    df = df[df["q_total_index"] > 0]
    print(f"After positive production filter: {len(df):,} voyages")

    # Create log production
    df["log_q"] = np.log(df["q_total_index"])

    # Filter to captains/agents with minimum voyages
    captain_counts = df["captain_id"].value_counts()
    agent_counts = df["agent_id"].value_counts()
    valid_captains = captain_counts[captain_counts >= min_captain_voyages].index
    valid_agents = agent_counts[agent_counts >= min_agent_voyages].index
    df = df[df["captain_id"].isin(valid_captains) & df["agent_id"].isin(valid_agents)]
    print(f"After min voyages filter (captain≥{min_captain_voyages}, agent≥{min_agent_voyages}): {len(df):,} voyages")

    # Prepare control variables
    df["log_tonnage"] = np.log(df["tonnage"].clip(lower=1))
    df["log_duration"] = np.log(df["duration_days"].clip(lower=1))

    # Fill missing controls with median
    for col in ["log_tonnage", "log_duration"]:
        df[col] = df[col].fillna(df[col].median())

    # Create decade for year effects
    df["decade"] = (df["year_out"] // 10) * 10

    print(f"\nFinal sample: {len(df):,} voyages")
    print(f"  Unique captains: {df['captain_id'].nunique():,}")
    print(f"  Unique agents: {df['agent_id'].nunique():,}")

    return df


def find_connected_set(df: pd.DataFrame) -> pd.DataFrame:
    """Find largest connected component in captain-agent bipartite graph."""
    print("\n" + "=" * 60)
    print("FINDING CONNECTED SET")
    print("=" * 60)

    # Build bipartite graph
    G = nx.Graph()
    for _, row in df[["captain_id", "agent_id"]].drop_duplicates().iterrows():
        G.add_edge(f"C_{row['captain_id']}", f"A_{row['agent_id']}")

    # Find largest connected component
    largest_cc = max(nx.connected_components(G), key=len)
    connected_captains = {n[2:] for n in largest_cc if n.startswith("C_")}
    connected_agents = {n[2:] for n in largest_cc if n.startswith("A_")}

    print(f"Connected set size:")
    print(f"  Captains: {len(connected_captains):,} / {df['captain_id'].nunique():,} ({100*len(connected_captains)/df['captain_id'].nunique():.1f}%)")
    print(f"  Agents: {len(connected_agents):,} / {df['agent_id'].nunique():,} ({100*len(connected_agents)/df['agent_id'].nunique():.1f}%)")

    df_cc = df[df["captain_id"].isin(connected_captains) & df["agent_id"].isin(connected_agents)].copy()
    print(f"  Voyages: {len(df_cc):,} / {len(df):,} ({100*len(df_cc)/len(df):.1f}%)")

    return df_cc


def find_leave_one_out_connected_set(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find leave-one-out connected set for KSS estimation.
    
    An observation is in the leave-one-out connected set if removing it
    does not disconnect the graph. This is equivalent to requiring that
    each edge (captain-agent pair) appears at least twice in the data.
    
    Following KSS (2020), we iteratively prune:
    1. Captains/agents with only 1 observation
    2. Captain-agent pairs that appear only once (would disconnect if removed)
    """
    print("\n" + "=" * 60)
    print("FINDING LEAVE-ONE-OUT CONNECTED SET (KSS)")
    print("=" * 60)
    
    df_loo = df.copy()
    initial_n = len(df_loo)
    
    # Iteratively prune until stable
    prev_n = 0
    iteration = 0
    while len(df_loo) != prev_n:
        prev_n = len(df_loo)
        iteration += 1
        
        # Remove observations where captain-agent pair appears only once
        # (these are "articulation edges" that would disconnect the graph)
        pair_counts = df_loo.groupby(["captain_id", "agent_id"]).size()
        df_loo["_pair"] = list(zip(df_loo["captain_id"], df_loo["agent_id"]))
        df_loo["_pair_count"] = df_loo["_pair"].map(pair_counts)
        
        # Keep observations where the pair appears more than once
        # OR where both captain and agent have multiple partners
        captain_n_agents = df_loo.groupby("captain_id")["agent_id"].transform("nunique")
        agent_n_captains = df_loo.groupby("agent_id")["captain_id"].transform("nunique")
        
        # An observation is "leave-out-able" if:
        # - The captain-agent pair appears > 1 time, OR
        # - The captain has > 1 agent AND the agent has > 1 captain
        keep_mask = (df_loo["_pair_count"] > 1) | ((captain_n_agents > 1) & (agent_n_captains > 1))
        df_loo = df_loo[keep_mask].copy()
        
        # Also ensure we're still in largest connected component
        if len(df_loo) > 0:
            G = nx.Graph()
            for _, row in df_loo[["captain_id", "agent_id"]].drop_duplicates().iterrows():
                G.add_edge(f"C_{row['captain_id']}", f"A_{row['agent_id']}")
            
            if len(G) > 0:
                largest_cc = max(nx.connected_components(G), key=len)
                connected_captains = {n[2:] for n in largest_cc if n.startswith("C_")}
                connected_agents = {n[2:] for n in largest_cc if n.startswith("A_")}
                df_loo = df_loo[
                    df_loo["captain_id"].isin(connected_captains) & 
                    df_loo["agent_id"].isin(connected_agents)
                ].copy()
    
    # Clean up temp columns
    df_loo = df_loo.drop(columns=["_pair", "_pair_count"], errors="ignore")
    
    print(f"  Iterations: {iteration}")
    print(f"  Leave-one-out set:")
    print(f"    Voyages: {len(df_loo):,} / {initial_n:,} ({100*len(df_loo)/initial_n:.1f}%)")
    print(f"    Captains: {df_loo['captain_id'].nunique():,}")
    print(f"    Agents: {df_loo['agent_id'].nunique():,}")
    
    return df_loo


def estimate_akm(df: pd.DataFrame) -> dict:
    """
    Estimate AKM model using sparse least squares.

    Model: log(q) = θ_captain + ψ_agent + β₁*log_tonnage + β₂*log_duration + γ_decade + ε
    """
    print("\n" + "=" * 60)
    print("ESTIMATING AKM MODEL")
    print("=" * 60)

    n = len(df)

    # Create captain indices
    captain_ids = df["captain_id"].unique()
    captain_map = {c: i for i, c in enumerate(captain_ids)}
    n_captains = len(captain_ids)
    captain_idx = df["captain_id"].map(captain_map).values

    # Create agent indices (drop one for identification)
    agent_ids = df["agent_id"].unique()
    agent_map = {a: i for i, a in enumerate(agent_ids)}
    n_agents = len(agent_ids)
    agent_idx = df["agent_id"].map(agent_map).values

    # Create decade indices
    decades = sorted(df["decade"].unique())
    decade_map = {d: i for i, d in enumerate(decades)}
    n_decades = len(decades)
    decade_idx = df["decade"].map(decade_map).values

    print(f"Fixed effects to estimate:")
    print(f"  {n_captains:,} captain effects")
    print(f"  {n_agents:,} agent effects")
    print(f"  {n_decades} decade effects")
    print(f"  2 continuous controls (log_tonnage, log_duration)")

    # Build sparse design matrix
    # [Captain FEs | Agent FEs | Decade FEs | log_tonnage | log_duration]
    # Drop one agent and one decade for identification

    # Captain dummies (all)
    row_c = np.arange(n)
    col_c = captain_idx
    data_c = np.ones(n)
    X_captain = sp.csr_matrix((data_c, (row_c, col_c)), shape=(n, n_captains))

    # Agent dummies (drop first)
    row_a = np.arange(n)
    col_a = agent_idx
    data_a = np.ones(n)
    X_agent_full = sp.csr_matrix((data_a, (row_a, col_a)), shape=(n, n_agents))
    X_agent = X_agent_full[:, 1:]  # Drop first agent

    # Decade dummies (drop first)
    row_d = np.arange(n)
    col_d = decade_idx
    data_d = np.ones(n)
    X_decade_full = sp.csr_matrix((data_d, (row_d, col_d)), shape=(n, n_decades))
    X_decade = X_decade_full[:, 1:]  # Drop first decade

    # Continuous controls
    X_controls = np.column_stack([
        df["log_tonnage"].values,
        df["log_duration"].values,
    ])
    X_controls_sparse = sp.csr_matrix(X_controls)

    # Stack all
    X = sp.hstack([X_captain, X_agent, X_decade, X_controls_sparse])
    y = df["log_q"].values

    print(f"\nDesign matrix shape: {X.shape}")
    print("Solving sparse least squares...")

    # Solve using LSQR (memory-efficient for sparse systems)
    result = lsqr(X, y, iter_lim=10000, atol=1e-10, btol=1e-10)
    beta = result[0]

    # Extract coefficients
    idx = 0
    theta = beta[idx:idx + n_captains]
    idx += n_captains

    psi_est = beta[idx:idx + n_agents - 1]
    psi = np.concatenate([[0], psi_est])  # First agent normalized to 0
    idx += n_agents - 1

    gamma_est = beta[idx:idx + n_decades - 1]
    gamma = np.concatenate([[0], gamma_est])  # First decade normalized to 0
    idx += n_decades - 1

    beta_tonnage = beta[idx]
    beta_duration = beta[idx + 1]

    # Fitted values and residuals
    y_hat = X @ beta
    residuals = y - y_hat
    r2 = 1 - np.var(residuals) / np.var(y)

    print(f"\nModel fit:")
    print(f"  R² = {r2:.4f}")
    print(f"  RMSE = {np.std(residuals):.4f}")

    # Create FE DataFrames
    captain_fe = pd.DataFrame({
        "captain_id": captain_ids,
        "theta": theta,
    })
    agent_fe = pd.DataFrame({
        "agent_id": agent_ids,
        "psi": psi,
    })

    # Merge FEs back to data
    df = df.merge(captain_fe, on="captain_id")
    df = df.merge(agent_fe, on="agent_id")

    # Compute decade effects for observations
    df["gamma"] = df["decade"].map(dict(zip(decades, gamma)))

    # Compute Xβ (controls only)
    df["Xbeta"] = beta_tonnage * df["log_tonnage"] + beta_duration * df["log_duration"] + df["gamma"]

    return {
        "df": df,
        "theta": theta,
        "psi": psi,
        "gamma": gamma,
        "beta_tonnage": beta_tonnage,
        "beta_duration": beta_duration,
        "captain_fe": captain_fe,
        "agent_fe": agent_fe,
        "residuals": residuals,
        "r2": r2,
        "captain_map": captain_map,
        "agent_map": agent_map,
        "decade_map": decade_map,
        "X": X,
        "y": y,
        "n_captains": n_captains,
        "n_agents": n_agents,
        "captain_idx": captain_idx,
        "agent_idx": agent_idx,
    }


def compute_kss_correction(results: dict) -> dict:
    """
    Compute KSS (Kline-Saggio-Sølvsten 2020) bias correction for variance components.
    
    The plug-in variance estimator is biased because:
        Var(θ̂_i) = Var(θ_i) + E[estimation error²]
    
    For the exact KSS estimator, we use the leave-out approach:
        Var_KSS(θ) = (1/n) Σᵢ θ̂ᵢ × θ̂₋ᵢ
    
    where θ̂₋ᵢ is the leave-one-out estimate. Since computing this exactly is 
    expensive, we use the approximation based on leverages:
    
        bias ≈ (1/n) Σᵢ Pᵢᵢ × σ²ᵢ
    
    where Pᵢᵢ is the leverage (diagonal of hat matrix) and σ²ᵢ is the 
    observation-specific residual variance.
    """
    print("\nComputing KSS bias correction...")
    
    df = results["df"]
    residuals = results["residuals"]
    n = len(df)
    
    theta_i = df["theta"].values
    psi_j = df["psi"].values
    
    # Compute leverages: for observation from captain with n_c obs, leverage ≈ 1/n_c
    captain_counts = df["captain_id"].value_counts()
    agent_counts = df["agent_id"].value_counts()
    
    h_captain = df["captain_id"].map(lambda c: 1.0 / captain_counts[c]).values
    h_agent = df["agent_id"].map(lambda a: 1.0 / agent_counts[a]).values
    
    # Residual variance estimate (heteroskedastic: use squared residuals)
    sigma_sq = residuals ** 2
    
    # Bias terms: E[Var(θ̂) - Var(θ)] ≈ mean(h × σ²)
    # For variance weighted by observation frequency
    bias_var_theta = np.sum(h_captain * sigma_sq) / n
    bias_var_psi = np.sum(h_agent * sigma_sq) / n
    
    # For covariance, the bias involves cross-leverage terms
    # Approximate: observations where captain i meets agent j contribute to Cov bias
    # This is small since captain and agent FEs are estimated from different parts of X
    bias_cov = 0.0  # Conservative approximation
    
    # Corrected estimates
    var_theta_plugin = np.var(theta_i)
    var_psi_plugin = np.var(psi_j)
    cov_plugin = np.cov(theta_i, psi_j)[0, 1]
    
    var_theta_kss = max(0, var_theta_plugin - bias_var_theta)
    var_psi_kss = max(0, var_psi_plugin - bias_var_psi)
    cov_kss = cov_plugin - bias_cov
    
    print(f"  Var(θ) plug-in: {var_theta_plugin:.4f}, bias: {bias_var_theta:.4f}, corrected: {var_theta_kss:.4f}")
    print(f"  Var(ψ) plug-in: {var_psi_plugin:.4f}, bias: {bias_var_psi:.4f}, corrected: {var_psi_kss:.4f}")
    
    return {
        "var_theta_plugin": var_theta_plugin,
        "var_psi_plugin": var_psi_plugin,
        "cov_plugin": cov_plugin,
        "var_theta_kss": var_theta_kss,
        "var_psi_kss": var_psi_kss,
        "cov_kss": cov_kss,
        "bias_var_theta": bias_var_theta,
        "bias_var_psi": bias_var_psi,
    }


def variance_decomposition(results: dict) -> pd.DataFrame:
    """
    Compute AKM variance decomposition with KSS correction.
    
    The decomposition:
        Var(y) = Var(θ) + Var(ψ) + 2Cov(θ,ψ) + Var(Xβ) + Var(ε) + 2Cov(θ,Xβ) + 2Cov(ψ,Xβ)
    
    For the corrected version, we adjust Var(θ) and Var(ψ) for estimation error,
    and attribute the bias to the residual term to ensure proper accounting.
    """
    print("\n" + "=" * 60)
    print("VARIANCE DECOMPOSITION")
    print("=" * 60)

    df = results["df"]

    # Components for each observation
    theta_i = df["theta"].values
    psi_j = df["psi"].values
    Xbeta = df["Xbeta"].values
    y = df["log_q"].values
    eps = results["residuals"]

    # Plug-in variances and covariances
    var_y = np.var(y)
    var_theta = np.var(theta_i)
    var_psi = np.var(psi_j)
    cov_theta_psi = np.cov(theta_i, psi_j)[0, 1]
    var_Xbeta = np.var(Xbeta)
    var_eps = np.var(eps)
    
    # Cross-covariances with controls
    cov_theta_Xbeta = np.cov(theta_i, Xbeta)[0, 1]
    cov_psi_Xbeta = np.cov(psi_j, Xbeta)[0, 1]

    # Full decomposition check
    decomp_sum = var_theta + var_psi + 2*cov_theta_psi + var_Xbeta + var_eps + 2*cov_theta_Xbeta + 2*cov_psi_Xbeta
    
    print("\n--- Plug-in Estimates ---")
    decomp_plugin = pd.DataFrame({
        "Component": [
            "Var(θ) - Captain",
            "Var(ψ) - Agent",
            "2×Cov(θ,ψ) - Sorting",
            "Var(Xβ) - Controls",
            "Var(ε) - Residual",
        ],
        "Variance": [
            var_theta,
            var_psi,
            2 * cov_theta_psi,
            var_Xbeta,
            var_eps,
        ],
    })
    decomp_plugin["Share"] = decomp_plugin["Variance"] / var_y
    print("\n" + decomp_plugin.to_string(index=False))
    print(f"\nTotal Var(y) = {var_y:.4f}")
    print(f"Sum of components = {decomp_sum:.4f}")
    
    # KSS correction
    kss = compute_kss_correction(results)
    
    # For KSS, the bias removed from FE variances goes to "unexplained"/residual
    # to maintain proper accounting: sum should still equal Var(y)
    total_bias = kss["bias_var_theta"] + kss["bias_var_psi"]
    var_eps_kss = var_eps + total_bias  # Attribute estimation error to residual
    
    print("\n--- KSS Corrected Estimates ---")
    decomp_kss = pd.DataFrame({
        "Component": [
            "Var(θ) - Captain",
            "Var(ψ) - Agent", 
            "2×Cov(θ,ψ) - Sorting",
            "Var(Xβ) - Controls",
            "Var(ε) - Residual",
        ],
        "Variance": [
            kss["var_theta_kss"],
            kss["var_psi_kss"],
            2 * kss["cov_kss"],
            var_Xbeta,
            var_eps_kss,
        ],
    })
    decomp_kss["Share"] = decomp_kss["Variance"] / var_y
    print("\n" + decomp_kss.to_string(index=False))
    
    kss_sum = kss["var_theta_kss"] + kss["var_psi_kss"] + 2*kss["cov_kss"] + var_Xbeta + var_eps_kss + 2*cov_theta_Xbeta + 2*cov_psi_Xbeta
    print(f"\nSum of KSS components = {kss_sum:.4f}")

    # Correlations
    corr_plugin = cov_theta_psi / (np.sqrt(var_theta) * np.sqrt(var_psi)) if var_theta > 0 and var_psi > 0 else 0
    corr_kss = kss["cov_kss"] / (np.sqrt(kss["var_theta_kss"]) * np.sqrt(kss["var_psi_kss"])) if kss["var_theta_kss"] > 0 and kss["var_psi_kss"] > 0 else 0
    
    print(f"\nCorr(θ, ψ) plug-in = {corr_plugin:.4f}")
    print(f"Corr(θ, ψ) KSS     = {corr_kss:.4f}")
    
    # Combine for output
    decomp_plugin["Type"] = "Plugin"
    decomp_kss["Type"] = "KSS"
    decomp = pd.concat([decomp_plugin, decomp_kss], ignore_index=True)

    return decomp


def analyze_matching_patterns(results: dict) -> None:
    """
    Detailed analysis of captain-agent matching/sorting patterns.
    """
    print("\n" + "=" * 60)
    print("CAPTAIN-AGENT MATCHING ANALYSIS")
    print("=" * 60)
    
    df = results["df"]
    captain_fe = results["captain_fe"]
    agent_fe = results["agent_fe"]
    
    # 1. Create quintiles for captains and agents
    captain_fe["theta_quintile"] = pd.qcut(captain_fe["theta"], q=5, labels=[1, 2, 3, 4, 5])
    agent_fe["psi_quintile"] = pd.qcut(agent_fe["psi"], q=5, labels=[1, 2, 3, 4, 5])
    
    # Merge quintiles to voyage data
    df = df.merge(captain_fe[["captain_id", "theta_quintile"]], on="captain_id", how="left")
    df = df.merge(agent_fe[["agent_id", "psi_quintile"]], on="agent_id", how="left")
    
    # 2. Sorting matrix: Captain quintile × Agent quintile
    print("\n--- SORTING MATRIX (% of matches) ---")
    print("Rows = Captain θ quintile (1=lowest, 5=highest)")
    print("Cols = Agent ψ quintile (1=lowest, 5=highest)")
    
    sorting_matrix = pd.crosstab(
        df["theta_quintile"], 
        df["psi_quintile"],
        normalize="all"
    ) * 100
    print("\n" + sorting_matrix.round(1).to_string())
    
    # Diagonal concentration (positive sorting = high diagonal)
    diagonal_share = sum(sorting_matrix.iloc[i, i] for i in range(5))
    print(f"\nDiagonal share: {diagonal_share:.1f}% (random = 20%, perfect sorting = 100%)")
    
    # 3. Mean agent quality by captain quintile
    print("\n--- MEAN AGENT ψ BY CAPTAIN θ QUINTILE ---")
    mean_psi_by_theta = df.groupby("theta_quintile")["psi"].agg(["mean", "std", "count"])
    print(mean_psi_by_theta.round(3).to_string())
    
    # 4. Mean captain quality by agent quintile  
    print("\n--- MEAN CAPTAIN θ BY AGENT ψ QUINTILE ---")
    mean_theta_by_psi = df.groupby("psi_quintile")["theta"].agg(["mean", "std", "count"])
    print(mean_theta_by_psi.round(3).to_string())
    
    # 5. Test for sorting gradient
    print("\n--- SORTING GRADIENT TEST ---")
    # Regress theta on psi quintile indicators
    from scipy import stats
    
    theta_by_psi_q = df.groupby("psi_quintile")["theta"].mean()
    slope, intercept, r, p, se = stats.linregress(range(1, 6), theta_by_psi_q.values)
    print(f"θ trend across ψ quintiles: slope = {slope:.4f}, p-value = {p:.4f}")
    
    psi_by_theta_q = df.groupby("theta_quintile")["psi"].mean()
    slope2, intercept2, r2, p2, se2 = stats.linregress(range(1, 6), psi_by_theta_q.values)
    print(f"ψ trend across θ quintiles: slope = {slope2:.4f}, p-value = {p2:.4f}")
    
    # 6. Time variation in sorting
    print("\n--- SORTING BY DECADE ---")
    df["decade"] = (df["year_out"] // 10) * 10
    decade_sorting = df.groupby("decade").apply(
        lambda x: x["theta"].corr(x["psi"]) if len(x) > 10 else np.nan
    )
    print("Corr(θ, ψ) by decade:")
    print(decade_sorting.round(4).to_string())
    
    # 7. Top matches (best captains × best agents)
    print("\n--- TOP 10 HIGHEST-QUALITY MATCHES ---")
    print("(Captain θ + Agent ψ)")
    df["match_quality"] = df["theta"] + df["psi"]
    top_matches = df.nlargest(10, "match_quality")[
        ["captain_id", "agent_id", "theta", "psi", "match_quality", "year_out"]
    ]
    print(top_matches.to_string(index=False))
    
    # 8. Repeat partnerships
    print("\n--- REPEAT PARTNERSHIPS ---")
    pair_counts = df.groupby(["captain_id", "agent_id"]).size().reset_index(name="voyages")
    repeat_pairs = pair_counts[pair_counts["voyages"] > 1]
    print(f"Total captain-agent pairs: {len(pair_counts):,}")
    print(f"Pairs with 2+ voyages: {len(repeat_pairs):,} ({100*len(repeat_pairs)/len(pair_counts):.1f}%)")
    
    # Does repeat partnership predict better sorting?
    repeat_pairs = repeat_pairs.merge(captain_fe[["captain_id", "theta"]], on="captain_id")
    repeat_pairs = repeat_pairs.merge(agent_fe[["agent_id", "psi"]], on="agent_id")
    
    single_pairs = pair_counts[pair_counts["voyages"] == 1]
    single_pairs = single_pairs.merge(captain_fe[["captain_id", "theta"]], on="captain_id")
    single_pairs = single_pairs.merge(agent_fe[["agent_id", "psi"]], on="agent_id")
    
    corr_repeat = repeat_pairs["theta"].corr(repeat_pairs["psi"])
    corr_single = single_pairs["theta"].corr(single_pairs["psi"])
    print(f"Corr(θ, ψ) for repeat pairs: {corr_repeat:.4f}")
    print(f"Corr(θ, ψ) for single pairs: {corr_single:.4f}")
    
    # 9. Home port variation
    print("\n--- SORTING BY HOME PORT (top 5 ports) ---")
    top_ports = df["home_port"].value_counts().head(5).index.tolist()
    for port in top_ports:
        port_df = df[df["home_port"] == port]
        if len(port_df) > 20:
            corr = port_df["theta"].corr(port_df["psi"])
            print(f"  {port}: n={len(port_df):,}, Corr(θ,ψ)={corr:.4f}")


def print_summary_stats(results: dict) -> None:
    """Print summary statistics for fixed effects."""
    print("\n" + "=" * 60)
    print("FIXED EFFECT DISTRIBUTIONS")
    print("=" * 60)

    captain_fe = results["captain_fe"]
    agent_fe = results["agent_fe"]

    print("\nCaptain Effects (θ):")
    print(captain_fe["theta"].describe().to_string())

    print("\nAgent Effects (ψ):")
    print(agent_fe["psi"].describe().to_string())

    # Top/bottom captains
    print("\nTop 5 Captains:")
    top5 = captain_fe.nlargest(5, "theta")
    print(top5.to_string(index=False))

    print("\nBottom 5 Captains:")
    bot5 = captain_fe.nsmallest(5, "theta")
    print(bot5.to_string(index=False))

    # Top/bottom agents
    print("\nTop 5 Agents:")
    top5_a = agent_fe.nlargest(5, "psi")
    print(top5_a.to_string(index=False))

    print("\nBottom 5 Agents:")
    bot5_a = agent_fe.nsmallest(5, "psi")
    print(bot5_a.to_string(index=False))


def save_results(results: dict, decomp: pd.DataFrame) -> None:
    """Save results to files."""
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    results["captain_fe"].to_csv(OUTPUT_DIR / "akm_captain_effects.csv", index=False)
    results["agent_fe"].to_csv(OUTPUT_DIR / "akm_agent_effects.csv", index=False)
    decomp.to_csv(OUTPUT_DIR / "akm_variance_decomposition.csv", index=False)

    print(f"Saved:")
    print(f"  - akm_captain_effects.csv ({len(results['captain_fe'])} captains)")
    print(f"  - akm_agent_effects.csv ({len(results['agent_fe'])} agents)")
    print(f"  - akm_variance_decomposition.csv")


def main():
    """Run full AKM analysis."""
    print("\n" + "=" * 60)
    print("AKM DECOMPOSITION: CAPTAINS AND AGENTS")
    print("=" * 60 + "\n")

    # Load and prepare (uses function defaults: 1780-1930)
    df = load_and_prepare_data()

    # Find standard connected set
    df_cc = find_connected_set(df)
    
    # Find leave-one-out connected set for KSS
    df_loo = find_leave_one_out_connected_set(df_cc)

    # Estimate AKM on leave-one-out sample
    results = estimate_akm(df_loo)

    # Variance decomposition with KSS correction
    decomp = variance_decomposition(results)
    
    # Detailed matching analysis
    analyze_matching_patterns(results)

    # Summary statistics
    print_summary_stats(results)

    # Save
    save_results(results, decomp)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

    return results, decomp


if __name__ == "__main__":
    main()
