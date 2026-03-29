"""
Phase 3B: Event Study Debugging
Reruns the switch event study with balanced windows, strict FEs, and a joint pre-trend F-test
to diagnose why older versions showed a pre-trend but recent ones don't.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'post_connectivity'

def run_event_study_debug():
    print("="*60)
    print("PHASE 3B: SWITCH EVENT STUDY DEBUGGING")
    print("="*60)
    
    # 1. Load Data
    df_raw = pd.read_parquet(PROJECT_ROOT / 'data' / 'final' / 'analysis_voyage.parquet')
    df_types = pd.read_csv(OUTPUT_DIR / 'tables' / 'type_estimation_authoritative.csv')

    df = df_raw.merge(df_types[['voyage_id', 'psi_hat']], on='voyage_id', how='inner')
    df['log_q'] = np.log(df['q_oil_bbl'] + 1)
    
    # Sort
    df = df.sort_values(['captain_id', 'year_out', 'date_out']).copy()
    
    # Identify switches
    df['prev_agent'] = df.groupby('captain_id')['agent_id'].shift(1)
    df['switched'] = (df['agent_id'] != df['prev_agent']) & (df['prev_agent'].notna())
    
    # Define switch up vs down based on psi_hat
    df['prev_psi_hat'] = df.groupby('captain_id')['psi_hat'].shift(1)
    df['psi_diff'] = df['psi_hat'] - df['prev_psi_hat']
    
    n_switches = df['switched'].sum()
    print(f"Total switches: {n_switches}")
    
    df['voyage_num'] = df.groupby('captain_id').cumcount()
    switch_voyages = df[df['switched']][['captain_id', 'voyage_num', 'psi_diff']].rename(
        columns={'voyage_num': 'switch_voyage', 'psi_diff': 'switch_psi_diff'}
    )
    
    # Use only first switch per captain for clean stacked event study
    switch_voyages = switch_voyages.drop_duplicates('captain_id', keep='first')
    
    df = df.merge(switch_voyages, on='captain_id', how='left')
    df['event_time'] = df['voyage_num'] - df['switch_voyage']

    # 2. Balanced Event Window
    # Keep captains observing [-2, -1, 0, 1, 2]
    window = [-2, -1, 0, 1, 2]
    df_window = df[df['event_time'].isin(window)].copy()

    counts = df_window.groupby('captain_id')['event_time'].nunique()
    balanced_captains = counts[counts == 5].index
    
    df_bal = df_window[df_window['captain_id'].isin(balanced_captains)].copy()
    print(f"Balanced captains: {len(balanced_captains)} / {len(switch_voyages)}")

    # 3. Create dummy variables
    # We omit t = -1 as reference
    results = []
    
    # Build models
    df_bal['route_time'] = df_bal['ground_or_route'].astype(str) + "_" + (df_bal['year_out'] // 10).astype(str)
    
    # Convert FEs to dummies
    fe_cap = pd.get_dummies(df_bal['captain_id'], prefix='cap', drop_first=True, dtype=int)
    fe_rt = pd.get_dummies(df_bal['route_time'], prefix='rt', drop_first=True, dtype=int)
    
    # Time dummies (t=-1 omitted!)
    df_bal['event_time'] = df_bal['event_time'].astype(int)
    time_dummies = pd.get_dummies(df_bal['event_time'], prefix='t', dtype=int)
    if 't_-1' in time_dummies.columns:
        time_dummies = time_dummies.drop(columns=['t_-1'])
    
    # A. OLS without FE
    X_simple = sm.add_constant(time_dummies)
    y = df_bal['log_q']
    
    model_simple = sm.OLS(y, X_simple).fit(cov_type='cluster', cov_kwds={'groups': df_bal['captain_id']})
    
    # Pre-trend joint F-test (t = -2 = 0)
    try:
        f_test_simple = model_simple.f_test("t_-2 = 0")
        pval_simple = f_test_simple.pvalue
    except:
        pval_simple = np.nan
        
    for t in [-2, 0, 1, 2]:
        col = f't_{t}'
        if col in model_simple.params.index:
            results.append({
                'model': 'Balanced_OLS',
                'event_time': t,
                'coef': model_simple.params[col],
                'se': model_simple.bse[col],
                'pval': model_simple.pvalues[col],
                'pre_trend_pvalue': pval_simple
            })
            
    # B. OLS with Captain FE
    X_fe = pd.concat([time_dummies, fe_cap], axis=1)
    # add constant? 
    X_fe = sm.add_constant(X_fe, has_constant='add')
    
    model_fe = sm.OLS(y, X_fe).fit(cov_type='cluster', cov_kwds={'groups': df_bal['captain_id']})
    
    try:
        f_test_fe = model_fe.f_test("t_-2 = 0")
        pval_fe = f_test_fe.pvalue
    except:
        pval_fe = np.nan
        
    for t in [-2, 0, 1, 2]:
        col = f't_{t}'
        if col in model_fe.params.index:
            results.append({
                'model': 'Balanced_CaptainFE',
                'event_time': t,
                'coef': model_fe.params[col],
                'se': model_fe.bse[col],
                'pval': model_fe.pvalues[col],
                'pre_trend_pvalue': pval_fe
            })
            
    # C. OLS with Captain FE + Route×Time
    X_full = pd.concat([time_dummies, fe_cap, fe_rt], axis=1)
    X_full = sm.add_constant(X_full, has_constant='add')
    model_full = sm.OLS(y, X_full).fit(cov_type='cluster', cov_kwds={'groups': df_bal['captain_id']})
    
    try:
        f_test_full = model_full.f_test("t_-2 = 0")
        pval_full = f_test_full.pvalue
    except:
        pval_full = np.nan
        
    for t in [-2, 0, 1, 2]:
        col = f't_{t}'
        if col in model_full.params.index:
            results.append({
                'model': 'Balanced_CaptainFE_RouteTimeFE',
                'event_time': t,
                'coef': model_full.params[col],
                'se': model_full.bse[col],
                'pval': model_full.pvalues[col],
                'pre_trend_pvalue': pval_full
            })

    if len(results) == 0:
        print("Empty results. Checking dummy column keys:")
        print(X_simple.columns)
        return
        
    df_res = pd.DataFrame(results)
    df_res.to_csv(OUTPUT_DIR / 'tables' / 'event_study_reconciliation.csv', index=False)
    
    print("\n--- Event Study Pre-Trend Checklist ---")
    for mod in df_res['model'].unique():
        p_val = df_res[df_res['model'] == mod]['pre_trend_pvalue'].mean()
        print(f"Model: {mod} -> Pre-trend Joint F-test P-value: {p_val:.4f}")
        for t in [-2, 0, 1, 2]:
            row = df_res[(df_res['model'] == mod) & (df_res['event_time'] == t)]
            if not row.empty:
                print(f"  t={t}: coef={row['coef'].iloc[0]:.3f} (p={row['pval'].iloc[0]:.3f})")
    
    print("\nSUCCESS: Phase 3B Event Study Reconciled.")

if __name__ == '__main__':
    run_event_study_debug()
