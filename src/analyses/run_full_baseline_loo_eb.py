#!/usr/bin/env python
"""
Full Baseline Analysis Suite with LOO Connected Set + EB Variance Correction

This script runs ALL core analyses required for the paper using the updated
AKM specification:
- LOO (Leave-One-Out) connected set for proper identification
- EB (Empirical Bayes) variance correction for noise reduction

Analyses included:
1. R1: Baseline AKM Production Function with variance decomposition
2. R2-R3: Portability analysis (captain vs agent effects)
3. Event Study: Captain switching events
4. Complementarity: θ × ψ interactions by ground type
5. Heterogeneous Effects: CATE by captain skill quartile
6. Counterfactual Matching: PAM vs AAM efficiency

Outputs:
- All tables in markdown and CSV format
- Fixed effects estimates with EB shrinkage
- Variance decomposition with reliability diagnostics
"""

import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr
import networkx as nx
import warnings
from pathlib import Path
import json
from datetime import datetime
import statsmodels.api as sm
from scipy import stats

from src.analyses.parallel_akm import parallel_eb_shrinkage, get_n_workers

warnings.filterwarnings('ignore')

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'final'
OUTPUT_DIR = PROJECT_ROOT / 'output' / 'baseline_loo_eb'
TABLES_DIR = OUTPUT_DIR / 'tables'
DIAGNOSTICS_DIR = OUTPUT_DIR / 'diagnostics'

for d in [OUTPUT_DIR, TABLES_DIR, DIAGNOSTICS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_loo_connected_set(df):
    """Find Leave-One-Out connected set for proper AKM identification."""
    df_clean = df.dropna(subset=['log_q', 'captain_id', 'agent_id', 'year_out']).copy()
    
    # Standard connected set
    G = nx.Graph()
    pairs = df_clean[['captain_id', 'agent_id']].drop_duplicates()
    for _, row in pairs.iterrows():
        G.add_edge(f'C_{row["captain_id"]}', f'A_{row["agent_id"]}')
    
    components = list(nx.connected_components(G))
    largest_cc = max(components, key=len)
    connected_captains = {n[2:] for n in largest_cc if n.startswith('C_')}
    connected_agents = {n[2:] for n in largest_cc if n.startswith('A_')}
    
    df_cc = df_clean[
        df_clean['captain_id'].isin(connected_captains) & 
        df_clean['agent_id'].isin(connected_agents)
    ].copy()
    
    # LOO pruning
    df_loo = df_cc.copy()
    prev_n = 0
    while len(df_loo) != prev_n:
        prev_n = len(df_loo)
        pair_counts = df_loo.groupby(['captain_id', 'agent_id']).size()
        df_loo['_pair'] = list(zip(df_loo['captain_id'], df_loo['agent_id']))
        df_loo['_pair_count'] = df_loo['_pair'].map(pair_counts)
        captain_n_agents = df_loo.groupby('captain_id')['agent_id'].transform('nunique')
        agent_n_captains = df_loo.groupby('agent_id')['captain_id'].transform('nunique')
        keep_mask = (df_loo['_pair_count'] > 1) | ((captain_n_agents > 1) & (agent_n_captains > 1))
        df_loo = df_loo[keep_mask].copy()
        if len(df_loo) > 0:
            G = nx.Graph()
            for _, row in df_loo[['captain_id', 'agent_id']].drop_duplicates().iterrows():
                G.add_edge(f'C_{row["captain_id"]}', f'A_{row["agent_id"]}')
            if len(G) > 0:
                largest_cc = max(nx.connected_components(G), key=len)
                connected_captains = {n[2:] for n in largest_cc if n.startswith('C_')}
                connected_agents = {n[2:] for n in largest_cc if n.startswith('A_')}
                df_loo = df_loo[
                    df_loo['captain_id'].isin(connected_captains) & 
                    df_loo['agent_id'].isin(connected_agents)
                ].copy()
    
    df_loo = df_loo.drop(columns=['_pair', '_pair_count'], errors='ignore')
    
    return df_loo, {
        'raw': len(df),
        'clean': len(df_clean),
        'connected': len(df_cc),
        'loo': len(df_loo),
        'n_captains': df_loo['captain_id'].nunique(),
        'n_agents': df_loo['agent_id'].nunique(),
    }


def run_akm_with_eb(
    df_est, 
    control_cols=None, 
    outcome_col='log_q',
    use_parallel: bool = True,
    n_workers: int = None,
):
    """Run AKM with EB variance correction.
    
    Parameters
    ----------
    df_est : pd.DataFrame
        Estimation sample with captain_id, agent_id, and outcome.
    control_cols : list, optional
        Control variable column names.
    outcome_col : str
        Outcome column name.
    use_parallel : bool
        If True, use multithreaded EB shrinkage.
    n_workers : int, optional
        Number of worker threads.
    """
    if control_cols is None:
        control_cols = []
    
    n = len(df_est)
    matrices = []
    
    # Captain FEs
    captain_ids = df_est["captain_id"].unique()
    captain_map = {c: i for i, c in enumerate(captain_ids)}
    captain_idx = df_est["captain_id"].map(captain_map).values
    X_captain = sp.csr_matrix(
        (np.ones(n), (np.arange(n), captain_idx)),
        shape=(n, len(captain_ids))
    )
    matrices.append(X_captain)
    n_captains = len(captain_ids)
    
    # Agent FEs (drop first for identification)
    agent_ids = df_est["agent_id"].unique()
    agent_map = {a: i for i, a in enumerate(agent_ids)}
    agent_idx = df_est["agent_id"].map(agent_map).values
    X_agent_full = sp.csr_matrix(
        (np.ones(n), (np.arange(n), agent_idx)),
        shape=(n, len(agent_ids))
    )
    X_agent = X_agent_full[:, 1:]
    matrices.append(X_agent)
    n_agents = len(agent_ids)
    
    # Controls
    control_data = []
    control_names = []
    for col in control_cols:
        if col in df_est.columns:
            vals = df_est[col].fillna(df_est[col].median()).values.astype(float)
            if np.std(vals) > 0:
                control_data.append(vals)
                control_names.append(col)
    
    if control_data:
        X_controls = sp.csr_matrix(np.column_stack(control_data))
        matrices.append(X_controls)
    
    X = sp.hstack(matrices)
    y = df_est[outcome_col].values
    
    # Estimate via LSQR
    result = lsqr(X, y, iter_lim=10000, atol=1e-10, btol=1e-10)
    beta = result[0]
    
    theta = beta[:n_captains]
    psi_est = beta[n_captains:n_captains + n_agents - 1]
    psi = np.concatenate([[0], psi_est])
    
    y_hat = X @ beta
    residuals = y - y_hat
    r2 = 1 - np.var(residuals) / np.var(y)
    sigma2_eps = np.var(residuals)
    
    # Control coefficients
    control_betas = {}
    if control_names:
        control_start = n_captains + n_agents - 1
        for i, name in enumerate(control_names):
            control_betas[name] = float(beta[control_start + i])
    
    # Count observations per FE unit
    captain_counts = df_est.groupby('captain_id').size()
    agent_counts = df_est.groupby('agent_id').size()
    n_c = np.array([captain_counts.get(c, 1) for c in captain_ids])
    n_a = np.array([agent_counts.get(a, 1) for a in agent_ids])
    
    # EB correction
    if use_parallel:
        # Parallel EB shrinkage for captains
        theta_eb, lambda_captain, var_theta_signal = parallel_eb_shrinkage(
            theta, n_c, sigma2_eps, n_workers=n_workers
        )
        # Parallel EB shrinkage for agents
        psi_eb, lambda_agent, var_psi_signal = parallel_eb_shrinkage(
            psi, n_a, sigma2_eps, n_workers=n_workers
        )
    else:
        # Sequential EB shrinkage
        noise_captain = sigma2_eps * np.mean(1/n_c)
        noise_agent = sigma2_eps * np.mean(1/n_a)
        
        var_theta_signal = max(0, np.var(theta) - noise_captain)
        var_psi_signal = max(0, np.var(psi) - noise_agent)
        
        if var_theta_signal > 0:
            lambda_captain = var_theta_signal / (var_theta_signal + sigma2_eps/n_c)
        else:
            lambda_captain = np.zeros_like(n_c, dtype=float)
        
        if var_psi_signal > 0:
            lambda_agent = var_psi_signal / (var_psi_signal + sigma2_eps/n_a)
        else:
            lambda_agent = np.zeros_like(n_a, dtype=float)
        
        theta_eb = lambda_captain * theta + (1 - lambda_captain) * np.mean(theta)
        psi_eb = lambda_agent * psi + (1 - lambda_agent) * np.mean(psi)
    
    # Build FE dataframes
    captain_fe = pd.DataFrame({
        'captain_id': captain_ids,
        'alpha_hat': theta,
        'alpha_eb': theta_eb,
        'lambda': lambda_captain,
        'n_voyages': n_c,
    })
    
    agent_fe = pd.DataFrame({
        'agent_id': agent_ids,
        'gamma_hat': psi,
        'gamma_eb': psi_eb,
        'lambda': lambda_agent,
        'n_voyages': n_a,
    })
    
    # Merge FEs back to data
    df_result = df_est.copy()
    df_result = df_result.merge(
        captain_fe[['captain_id', 'alpha_hat', 'alpha_eb']], 
        on='captain_id', how='left'
    )
    df_result = df_result.merge(
        agent_fe[['agent_id', 'gamma_hat', 'gamma_eb']], 
        on='agent_id', how='left'
    )
    
    # Observation-level variance components
    alpha_i = df_result['alpha_hat'].values
    gamma_j = df_result['gamma_hat'].values
    alpha_i_eb = df_result['alpha_eb'].values
    gamma_j_eb = df_result['gamma_eb'].values
    
    var_alpha_plugin = np.var(alpha_i)
    var_gamma_plugin = np.var(gamma_j)
    cov_plugin = np.cov(alpha_i, gamma_j)[0, 1]
    
    var_alpha_eb = np.var(alpha_i_eb)
    var_gamma_eb = np.var(gamma_j_eb)
    cov_eb = np.cov(alpha_i_eb, gamma_j_eb)[0, 1]
    
    eb_total = var_alpha_eb + var_gamma_eb + 2 * cov_eb
    
    if var_alpha_eb > 0 and var_gamma_eb > 0:
        corr_eb = cov_eb / (np.sqrt(var_alpha_eb) * np.sqrt(var_gamma_eb))
    else:
        corr_eb = 0
    
    return {
        'n': n,
        'n_captains': n_captains,
        'n_agents': n_agents,
        'r2': float(r2),
        'rmse': float(np.sqrt(np.mean(residuals**2))),
        'sigma2_eps': float(sigma2_eps),
        'var_y': float(np.var(y)),
        'control_betas': control_betas,
        # Plugin estimates
        'var_alpha_plugin': float(var_alpha_plugin),
        'var_gamma_plugin': float(var_gamma_plugin),
        'cov_plugin': float(cov_plugin),
        # EB estimates
        'var_alpha_eb': float(var_alpha_eb),
        'var_gamma_eb': float(var_gamma_eb),
        'cov_eb': float(cov_eb),
        'corr_eb': float(corr_eb),
        'eb_total': float(eb_total),
        # Shares
        'share_alpha': float(var_alpha_eb / eb_total) if eb_total > 0 else 0,
        'share_gamma': float(var_gamma_eb / eb_total) if eb_total > 0 else 0,
        'share_cov': float(2 * cov_eb / eb_total) if eb_total > 0 else 0,
        # Reliability
        'mean_lambda_captain': float(np.mean(lambda_captain)),
        'mean_lambda_agent': float(np.mean(lambda_agent)),
        # DataFrames
        'captain_fe': captain_fe,
        'agent_fe': agent_fe,
        'df_with_fe': df_result,
        'residuals': residuals,
    }


# =============================================================================
# ANALYSIS 1: BASELINE VARIANCE DECOMPOSITION (R1)
# =============================================================================

def run_r1_baseline(df_est):
    """R1: Baseline AKM variance decomposition with multiple specifications."""
    print('\n' + '='*70)
    print('R1: BASELINE VARIANCE DECOMPOSITION')
    print('='*70)
    
    results = {}
    
    # Specification 1: No controls
    print('\n  Running Spec 1: No controls...')
    r1_noctl = run_akm_with_eb(df_est, control_cols=[])
    results['no_controls'] = r1_noctl
    
    # Specification 2: Vessel controls
    print('  Running Spec 2: Vessel controls...')
    r1_vessel = run_akm_with_eb(df_est, control_cols=['log_tonnage', 'is_ship'])
    results['vessel_controls'] = r1_vessel
    
    # Specification 3: Full controls
    print('  Running Spec 3: Full controls...')
    r1_full = run_akm_with_eb(df_est, control_cols=[
        'log_tonnage', 'log_duration', 'log_captain_exp', 'is_ship'
    ])
    results['full_controls'] = r1_full
    
    # Print summary
    print('\n  --- Variance Decomposition Summary (EB-Corrected) ---')
    for name, r in results.items():
        print(f"\n  {name}:")
        print(f"    R² = {r['r2']:.4f}")
        print(f"    Captain share: {100*r['share_alpha']:.1f}%")
        print(f"    Agent share: {100*r['share_gamma']:.1f}%")
        print(f"    Sorting share: {100*r['share_cov']:.1f}%")
        print(f"    Corr(α,γ) = {r['corr_eb']:.3f}")
    
    return results


# =============================================================================
# ANALYSIS 2: COMPLEMENTARITY (θ × ψ INTERACTIONS)
# =============================================================================

def run_complementarity(df_est, akm_results):
    """Run complementarity analysis: θ × ψ interactions by ground type."""
    print('\n' + '='*70)
    print('COMPLEMENTARITY ANALYSIS: θ × ψ INTERACTIONS')
    print('='*70)
    
    # Merge FEs
    df = akm_results['df_with_fe'].copy()
    
    # Standardize FEs for interpretation
    df['theta_std'] = (df['alpha_eb'] - df['alpha_eb'].mean()) / df['alpha_eb'].std()
    df['psi_std'] = (df['gamma_eb'] - df['gamma_eb'].mean()) / df['gamma_eb'].std()
    df['theta_x_psi'] = df['theta_std'] * df['psi_std']
    
    # Classify grounds
    if 'ground_sparse' not in df.columns:
        # Use decade catch rates to classify
        if 'ground_or_route' in df.columns and 'decade' in df.columns:
            ground_decade_mean = df.groupby(['ground_or_route', 'decade'])['log_q'].transform('mean')
            df['ground_sparse'] = (ground_decade_mean < df['log_q'].median()).astype(int)
        else:
            # Fallback: use overall median
            df['ground_sparse'] = (df['log_q'] < df['log_q'].median()).astype(int)
    
    results = {}
    
    # Pooled regression
    print('\n  --- Pooled Model ---')
    X_pooled = df[['theta_std', 'psi_std', 'theta_x_psi']].dropna()
    y_pooled = df.loc[X_pooled.index, 'log_q']
    X_pooled = sm.add_constant(X_pooled)
    model_pooled = sm.OLS(y_pooled, X_pooled).fit(cov_type='HC1')
    
    results['pooled'] = {
        'n': len(y_pooled),
        'beta_theta': float(model_pooled.params['theta_std']),
        'beta_psi': float(model_pooled.params['psi_std']),
        'beta_interaction': float(model_pooled.params['theta_x_psi']),
        'se_interaction': float(model_pooled.bse['theta_x_psi']),
        'pval_interaction': float(model_pooled.pvalues['theta_x_psi']),
        'r2': float(model_pooled.rsquared),
    }
    print(f"    N = {results['pooled']['n']:,}")
    print(f"    β(θ×ψ) = {results['pooled']['beta_interaction']:.4f} (SE = {results['pooled']['se_interaction']:.4f})")
    print(f"    Interpretation: {'Substitutes' if results['pooled']['beta_interaction'] < 0 else 'Complements'}")
    
    # Sparse grounds
    print('\n  --- Sparse Grounds ---')
    df_sparse = df[df['ground_sparse'] == 1].dropna(subset=['theta_std', 'psi_std', 'theta_x_psi', 'log_q'])
    if len(df_sparse) > 100:
        X_sparse = sm.add_constant(df_sparse[['theta_std', 'psi_std', 'theta_x_psi']])
        y_sparse = df_sparse['log_q']
        model_sparse = sm.OLS(y_sparse, X_sparse).fit(cov_type='HC1')
        
        results['sparse'] = {
            'n': len(y_sparse),
            'beta_theta': float(model_sparse.params['theta_std']),
            'beta_psi': float(model_sparse.params['psi_std']),
            'beta_interaction': float(model_sparse.params['theta_x_psi']),
            'se_interaction': float(model_sparse.bse['theta_x_psi']),
            'pval_interaction': float(model_sparse.pvalues['theta_x_psi']),
        }
        print(f"    N = {results['sparse']['n']:,}")
        print(f"    β(θ×ψ) = {results['sparse']['beta_interaction']:.4f} (SE = {results['sparse']['se_interaction']:.4f})")
    
    # Rich grounds
    print('\n  --- Rich Grounds ---')
    df_rich = df[df['ground_sparse'] == 0].dropna(subset=['theta_std', 'psi_std', 'theta_x_psi', 'log_q'])
    if len(df_rich) > 100:
        X_rich = sm.add_constant(df_rich[['theta_std', 'psi_std', 'theta_x_psi']])
        y_rich = df_rich['log_q']
        model_rich = sm.OLS(y_rich, X_rich).fit(cov_type='HC1')
        
        results['rich'] = {
            'n': len(y_rich),
            'beta_theta': float(model_rich.params['theta_std']),
            'beta_psi': float(model_rich.params['psi_std']),
            'beta_interaction': float(model_rich.params['theta_x_psi']),
            'se_interaction': float(model_rich.bse['theta_x_psi']),
            'pval_interaction': float(model_rich.pvalues['theta_x_psi']),
        }
        print(f"    N = {results['rich']['n']:,}")
        print(f"    β(θ×ψ) = {results['rich']['beta_interaction']:.4f} (SE = {results['rich']['se_interaction']:.4f})")
    
    return results, df


# =============================================================================
# ANALYSIS 3: HETEROGENEOUS EFFECTS (CATE BY QUARTILE)
# =============================================================================

def run_heterogeneous_effects(df):
    """Run heterogeneous effects analysis: CATE of agent by captain skill quartile."""
    print('\n' + '='*70)
    print('HETEROGENEOUS EFFECTS: CATE BY CAPTAIN SKILL QUARTILE')
    print('='*70)
    
    # Create quartiles
    df = df.copy()
    df['theta_quartile'] = pd.qcut(df['alpha_eb'], 4, labels=['Q1 (Novice)', 'Q2', 'Q3', 'Q4 (Expert)'])
    
    results = []
    
    for q in ['Q1 (Novice)', 'Q2', 'Q3', 'Q4 (Expert)']:
        df_q = df[df['theta_quartile'] == q].dropna(subset=['psi_std', 'log_q'])
        
        if len(df_q) > 50:
            X = sm.add_constant(df_q['psi_std'])
            y = df_q['log_q']
            model = sm.OLS(y, X).fit(cov_type='HC1')
            
            cate = float(model.params['psi_std'])
            se = float(model.bse['psi_std'])
            pval = float(model.pvalues['psi_std'])
            
            stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
            
            results.append({
                'quartile': q,
                'mean_theta': float(df_q['alpha_eb'].mean()),
                'cate': cate,
                'se': se,
                'pval': pval,
                'stars': stars,
                'n': len(df_q),
            })
            
            print(f"  {q}: CATE(ψ) = {cate:.4f}{stars} (SE = {se:.4f}), Mean θ = {df_q['alpha_eb'].mean():.2f}")
    
    return pd.DataFrame(results)


# =============================================================================
# ANALYSIS 4: EVENT STUDY (CAPTAIN SWITCHING)
# =============================================================================

def run_event_study(df_est):
    """Run event study for captain switching agents."""
    print('\n' + '='*70)
    print('EVENT STUDY: CAPTAIN SWITCHING AGENTS')
    print('='*70)
    
    df = df_est.copy()
    df = df.sort_values(['captain_id', 'year_out', 'date_out'])
    
    # Identify switches
    df['prev_agent'] = df.groupby('captain_id')['agent_id'].shift(1)
    df['switched'] = (df['agent_id'] != df['prev_agent']) & (df['prev_agent'].notna())
    
    n_switches = df['switched'].sum()
    n_captains_who_switched = df[df['switched']]['captain_id'].nunique()
    
    print(f"\n  Total switches: {n_switches:,}")
    print(f"  Captains who switched: {n_captains_who_switched:,}")
    
    # For each captain, get voyage number relative to switch
    df['voyage_num'] = df.groupby('captain_id').cumcount()
    
    # Find switch voyages
    switch_voyages = df[df['switched']][['captain_id', 'voyage_num']].rename(columns={'voyage_num': 'switch_voyage'})
    
    # Merge back
    df = df.merge(switch_voyages, on='captain_id', how='left')
    
    # Calculate event time
    df['event_time'] = df['voyage_num'] - df['switch_voyage']
    
    # Filter to event window
    df_event = df[df['event_time'].between(-2, 2) & df['switch_voyage'].notna()].copy()
    
    print(f"  Observations in event window: {len(df_event):,}")
    
    # Get pre/post means
    results = []
    for t in range(-2, 3):
        df_t = df_event[df_event['event_time'] == t]
        if len(df_t) > 10:
            mean_q = df_t['log_q'].mean()
            se_q = df_t['log_q'].std() / np.sqrt(len(df_t))
            results.append({
                'event_time': t,
                'mean_log_q': float(mean_q),
                'se': float(se_q),
                'n': len(df_t),
            })
            print(f"    t={t:+d}: mean(log_q) = {mean_q:.3f} (SE = {se_q:.3f}), N = {len(df_t)}")
    
    return pd.DataFrame(results), n_switches


# =============================================================================
# ANALYSIS 5: MATCHING COUNTERFACTUALS (PAM vs AAM)
# =============================================================================

def run_matching_counterfactual(df, comp_results):
    """Run matching counterfactual: PAM vs AAM efficiency."""
    print('\n' + '='*70)
    print('MATCHING COUNTERFACTUAL: PAM vs AAM')
    print('='*70)
    
    df = df.copy()
    
    # Get interaction coefficient
    if 'sparse' in comp_results and 'rich' in comp_results:
        beta3_sparse = comp_results['sparse']['beta_interaction']
        beta3_rich = comp_results['rich']['beta_interaction']
    else:
        beta3_sparse = beta3_rich = comp_results['pooled']['beta_interaction']
    
    print(f"\n  Interaction coefficients:")
    print(f"    Sparse: β₃ = {beta3_sparse:.4f}")
    print(f"    Rich: β₃ = {beta3_rich:.4f}")
    
    # Create cells (decade × ground_sparse)
    if 'decade' in df.columns and 'ground_sparse' in df.columns:
        df['cell'] = df['decade'].astype(str) + '_' + df['ground_sparse'].astype(str)
    else:
        df['cell'] = 'all'
    
    results = []
    
    for is_sparse in [0, 1]:
        df_g = df[df['ground_sparse'] == is_sparse].copy()
        beta3 = beta3_sparse if is_sparse else beta3_rich
        ground_label = 'Sparse' if is_sparse else 'Rich'
        
        if len(df_g) < 100:
            continue
        
        # Current output
        current_mean = df_g['log_q'].mean()
        
        # PAM counterfactual: sort theta and psi in same direction within cells
        pam_gains = []
        aam_gains = []
        
        for cell in df_g['cell'].unique():
            df_cell = df_g[df_g['cell'] == cell].copy()
            if len(df_cell) < 10:
                continue
            
            # Current matching
            current = df_cell['theta_x_psi'].mean()
            
            # PAM: high-θ with high-ψ → maximize θ×ψ
            theta_sorted = np.sort(df_cell['theta_std'].values)
            psi_sorted = np.sort(df_cell['psi_std'].values)
            pam_interaction = np.mean(theta_sorted * psi_sorted)
            
            # AAM: high-θ with low-ψ → minimize θ×ψ
            psi_reverse = np.sort(df_cell['psi_std'].values)[::-1]
            aam_interaction = np.mean(theta_sorted * psi_reverse)
            
            # Output changes
            pam_delta = beta3 * (pam_interaction - current)
            aam_delta = beta3 * (aam_interaction - current)
            
            pam_gains.append(pam_delta * len(df_cell))
            aam_gains.append(aam_delta * len(df_cell))
        
        if pam_gains:
            pam_effect = sum(pam_gains) / len(df_g)
            aam_effect = sum(aam_gains) / len(df_g)
            
            results.append({
                'ground': ground_label,
                'beta3': float(beta3),
                'pam_delta_log_q': float(pam_effect),
                'pam_delta_pct': float(100 * (np.exp(pam_effect) - 1)),
                'aam_delta_log_q': float(aam_effect),
                'aam_delta_pct': float(100 * (np.exp(aam_effect) - 1)),
                'n': len(df_g),
            })
            
            print(f"\n  {ground_label} Grounds (N = {len(df_g):,}):")
            print(f"    PAM effect: {100 * (np.exp(pam_effect) - 1):+.2f}%")
            print(f"    AAM effect: {100 * (np.exp(aam_effect) - 1):+.2f}%")
    
    return pd.DataFrame(results)


# =============================================================================
# ANALYSIS 6: INSURANCE VARIANCE (LEFT-TAIL PROTECTION)
# =============================================================================

def run_insurance_variance(df):
    """Run insurance variance analysis: does high-ψ compress novice variance?"""
    print('\n' + '='*70)
    print('INSURANCE VARIANCE: LEFT-TAIL PROTECTION')
    print('='*70)
    
    df = df.copy()
    
    # Define captain experience
    if 'captain_experience' in df.columns:
        df['is_novice'] = (df['captain_experience'] <= 3).astype(int)
    else:
        # Use count of previous voyages
        df['n_prior'] = df.groupby('captain_id').cumcount()
        df['is_novice'] = (df['n_prior'] <= 3).astype(int)
    
    # Define high-ψ (top quartile)
    psi_q75 = df['gamma_eb'].quantile(0.75)
    df['high_psi'] = (df['gamma_eb'] >= psi_q75).astype(int)
    
    results = []
    
    for novice in [1, 0]:
        for high_psi in [0, 1]:
            df_cell = df[(df['is_novice'] == novice) & (df['high_psi'] == high_psi)]
            
            if len(df_cell) < 20:
                continue
            
            label = f"{'Novice' if novice else 'Expert'} × {'High-ψ' if high_psi else 'Low-ψ'}"
            
            results.append({
                'cell': label,
                'is_novice': novice,
                'high_psi': high_psi,
                'n': len(df_cell),
                'mean_log_q': float(df_cell['log_q'].mean()),
                'std_log_q': float(df_cell['log_q'].std()),
                'p10_log_q': float(df_cell['log_q'].quantile(0.10)),
                'p90_log_q': float(df_cell['log_q'].quantile(0.90)),
            })
    
    results_df = pd.DataFrame(results)
    
    # Compute variance ratio
    baseline_var = results_df[
        (results_df['is_novice'] == 1) & (results_df['high_psi'] == 0)
    ]['std_log_q'].values[0] ** 2
    
    results_df['var_ratio'] = (results_df['std_log_q'] ** 2) / baseline_var
    
    print("\n  Treatment Cells:")
    for _, row in results_df.iterrows():
        print(f"    {row['cell']}: N = {row['n']:,}, Mean = {row['mean_log_q']:.2f}, Std = {row['std_log_q']:.2f}, Var Ratio = {row['var_ratio']:.2f}")
    
    # Key finding: variance compression
    novice_lowpsi = results_df[(results_df['is_novice'] == 1) & (results_df['high_psi'] == 0)]
    novice_highpsi = results_df[(results_df['is_novice'] == 1) & (results_df['high_psi'] == 1)]
    
    if len(novice_lowpsi) > 0 and len(novice_highpsi) > 0:
        compression = 1 - novice_highpsi['var_ratio'].values[0]
        floor_raise = novice_highpsi['p10_log_q'].values[0] - novice_lowpsi['p10_log_q'].values[0]
        print(f"\n  KEY FINDING:")
        print(f"    High-ψ agents compress novice variance by {100*compression:.1f}%")
        print(f"    Floor (P10) raised by {floor_raise:+.2f} log points")
    
    return results_df


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_full_suite():
    """Run the full analysis suite."""
    print('='*70)
    print('FULL BASELINE ANALYSIS SUITE')
    print('LOO Connected Set + EB Variance Correction')
    print(f'Timestamp: {datetime.now().isoformat()}')
    print('='*70)
    
    # Load data
    print('\n[1/7] Loading data...')
    df = pd.read_parquet(DATA_DIR / 'analysis_voyage.parquet')
    
    # Prepare variables
    df['log_tonnage'] = np.log(df['tonnage'].replace(0, np.nan))
    df['is_ship'] = (df['rig'] == 'Ship').astype(float)
    df = df.sort_values(['captain_id', 'year_out', 'date_out'])
    df['captain_experience'] = df.groupby('captain_id').cumcount()
    df['log_captain_exp'] = np.log(df['captain_experience'] + 1)
    df['log_duration'] = np.log(df['duration_days'].replace(0, np.nan))
    df['log_crew'] = np.log(df['crew_count'].replace(0, np.nan))
    df['decade'] = (df['year_out'] // 10) * 10
    df['log_q'] = np.log(df['q_oil_bbl'] + 1)
    
    print(f'  Raw data: {len(df):,} voyages')
    
    # Get LOO connected set
    print('\n[2/7] Finding LOO connected set...')
    df_est, conn_stats = get_loo_connected_set(df)
    print(f'  LOO set: {conn_stats["loo"]:,} voyages ({100*conn_stats["loo"]/conn_stats["raw"]:.1f}%)')
    print(f'  Captains: {conn_stats["n_captains"]:,}')
    print(f'  Agents: {conn_stats["n_agents"]:,}')
    
    all_results = {
        'connectivity': conn_stats,
        'timestamp': datetime.now().isoformat(),
    }
    
    # R1: Baseline variance decomposition
    print('\n[3/7] Running R1: Baseline variance decomposition...')
    r1_results = run_r1_baseline(df_est)
    all_results['r1'] = {
        k: {kk: vv for kk, vv in v.items() if not isinstance(vv, pd.DataFrame)}
        for k, v in r1_results.items()
    }
    
    # Save fixed effects
    r1_results['no_controls']['captain_fe'].to_csv(TABLES_DIR / 'captain_fixed_effects.csv', index=False)
    r1_results['no_controls']['agent_fe'].to_csv(TABLES_DIR / 'agent_fixed_effects.csv', index=False)
    
    # Complementarity analysis
    print('\n[4/7] Running complementarity analysis...')
    comp_results, df_with_fe = run_complementarity(df_est, r1_results['no_controls'])
    all_results['complementarity'] = comp_results
    
    # Heterogeneous effects
    print('\n[5/7] Running heterogeneous effects analysis...')
    cate_results = run_heterogeneous_effects(df_with_fe)
    cate_results.to_csv(TABLES_DIR / 'cate_by_quartile.csv', index=False)
    all_results['cate'] = cate_results.to_dict('records')
    
    # Event study
    print('\n[6/7] Running event study...')
    event_results, n_switches = run_event_study(df_est)
    event_results.to_csv(TABLES_DIR / 'event_study.csv', index=False)
    all_results['event_study'] = {
        'n_switches': int(n_switches),
        'coefficients': event_results.to_dict('records'),
    }
    
    # Matching counterfactual
    print('\n[7/7] Running matching counterfactual...')
    cf_results = run_matching_counterfactual(df_with_fe, comp_results)
    cf_results.to_csv(TABLES_DIR / 'matching_counterfactual.csv', index=False)
    all_results['counterfactual'] = cf_results.to_dict('records')
    
    # Insurance variance
    insurance_results = run_insurance_variance(df_with_fe)
    insurance_results.to_csv(TABLES_DIR / 'insurance_variance.csv', index=False)
    all_results['insurance'] = insurance_results.to_dict('records')
    
    # Save all results (with numpy type conversion)
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        else:
            return obj
    
    all_results = convert_numpy(all_results)
    
    with open(OUTPUT_DIR / 'analysis_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate summary markdown
    generate_summary_markdown(all_results, r1_results)
    
    print('\n' + '='*70)
    print('ANALYSIS COMPLETE')
    print('='*70)
    print(f'\nOutputs saved to: {OUTPUT_DIR}')
    print(f'  - tables/captain_fixed_effects.csv')
    print(f'  - tables/agent_fixed_effects.csv')
    print(f'  - tables/cate_by_quartile.csv')
    print(f'  - tables/event_study.csv')
    print(f'  - tables/matching_counterfactual.csv')
    print(f'  - tables/insurance_variance.csv')
    print(f'  - analysis_results.json')
    print(f'  - executive_summary.md')
    
    return all_results


def generate_summary_markdown(results, r1_results):
    """Generate executive summary markdown."""
    md = []
    md.append('# Executive Summary: AKM Analysis with LOO + EB Correction')
    md.append(f'\nGenerated: {results["timestamp"]}')
    md.append('')
    
    # Sample
    md.append('## Sample')
    md.append(f'- **LOO Connected Set**: {results["connectivity"]["loo"]:,} voyages')
    md.append(f'- **Captains**: {results["connectivity"]["n_captains"]:,}')
    md.append(f'- **Agents**: {results["connectivity"]["n_agents"]:,}')
    md.append('')
    
    # Variance decomposition
    md.append('## Variance Decomposition (EB-Corrected)')
    md.append('')
    md.append('| Specification | R² | Captain Share | Agent Share | Sorting Share | Corr(α,γ) |')
    md.append('|---------------|----:|-------------:|------------:|-------------:|----------:|')
    
    for spec, r in results['r1'].items():
        md.append(f"| {spec} | {r['r2']:.4f} | {100*r['share_alpha']:.1f}% | {100*r['share_gamma']:.1f}% | {100*r['share_cov']:.1f}% | {r['corr_eb']:.3f} |")
    md.append('')
    
    # Complementarity
    md.append('## Complementarity (θ × ψ Interaction)')
    md.append('')
    if 'complementarity' in results:
        for ground, r in results['complementarity'].items():
            if isinstance(r, dict) and 'beta_interaction' in r:
                interp = 'Substitutes' if r['beta_interaction'] < 0 else 'Complements'
                md.append(f"- **{ground}**: β₃ = {r['beta_interaction']:.4f} ({interp})")
    md.append('')
    
    # CATE
    md.append('## Conditional Average Treatment Effects (CATE)')
    md.append('')
    md.append('| Captain Quartile | Mean θ | CATE(ψ) | Interpretation |')
    md.append('|------------------|-------:|--------:|----------------|')
    
    for row in results.get('cate', []):
        mechanism = 'Insurance' if row['quartile'] == 'Q1 (Novice)' else 'Diminishing Returns' if row['quartile'] == 'Q4 (Expert)' else 'Transition'
        md.append(f"| {row['quartile']} | {row['mean_theta']:.2f} | {row['cate']:.4f}{row['stars']} | {mechanism} |")
    md.append('')
    
    # Counterfactual
    md.append('## Matching Counterfactual')
    md.append('')
    md.append('| Ground | PAM Effect | AAM Effect | Optimal |')
    md.append('|--------|----------:|----------:|---------|')
    
    for row in results.get('counterfactual', []):
        optimal = 'AAM' if row['pam_delta_pct'] < row['aam_delta_pct'] else 'PAM'
        md.append(f"| {row['ground']} | {row['pam_delta_pct']:+.2f}% | {row['aam_delta_pct']:+.2f}% | {optimal} |")
    md.append('')
    
    # Reliability
    md.append('## Reliability Diagnostics')
    md.append('')
    noctl = results['r1']['no_controls']
    md.append(f"- **Captain λ (mean)**: {noctl['mean_lambda_captain']:.3f}")
    md.append(f"- **Agent λ (mean)**: {noctl['mean_lambda_agent']:.3f}")
    md.append('')
    
    # Write
    (OUTPUT_DIR / 'executive_summary.md').write_text('\n'.join(md))


if __name__ == '__main__':
    run_full_suite()
