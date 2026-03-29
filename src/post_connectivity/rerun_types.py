"""
Phase 1: Re-estimate Authoritative Types

Re-run AKM/KSS using the exact canonical connected set built in Phase 0.
Output: type_file_authoritative.parquet and type_estimation_authoritative.csv.
"""

import pandas as pd
import numpy as np
import warnings
import json
from pathlib import Path

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'final'
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'post_connectivity'

# We import from legacy pipelines:
from src.analyses.run_full_baseline_loo_eb import run_akm_with_eb
from src.analyses.baseline_production import estimate_r1, compute_kss_correction

def run_type_estimation():
    print("="*60)
    print("PHASE 1: RE-ESTIMATE AUTHORITATIVE TYPES")
    print("="*60)

    # 1. Load canonical connected set from Phase 0
    df = pd.read_parquet(OUTPUT_DIR / 'manifests' / 'canonical_connected_set.parquet')
    
    # Needs some fields explicitly generated for models:
    df['log_tonnage'] = np.log(df['tonnage'].replace(0, np.nan))
    df['is_ship'] = (df['rig'] == 'Ship').astype(float)
    df = df.sort_values(['captain_id', 'year_out', 'date_out'])
    df['captain_experience'] = df.groupby('captain_id').cumcount()
    df['log_captain_exp'] = np.log(df['captain_experience'] + 1)
    df['log_duration'] = np.log(df['duration_days'].replace(0, np.nan))
    
    control_cols = ['log_tonnage', 'log_duration', 'log_captain_exp', 'is_ship']
    for col in control_cols:
        df[col] = df[col].fillna(df[col].median())

    # 2. Run normal AKM with Empirical Bayes Shrinkage
    print("--- Computing Plugin Estimates and EB Shrinkage ---")
    eb_results = run_akm_with_eb(df, control_cols=control_cols, outcome_col='log_q')
    
    # 3. Create the type catalog for Captains and Agents
    cap_fe = eb_results['captain_fe']
    agent_fe = eb_results['agent_fe']
    
    df_types = df[['voyage_id', 'captain_id', 'agent_id']].copy()
    df_types['connected_set'] = True  # these are all in canonical

    # Merge captain
    df_types = df_types.merge(
        cap_fe[['captain_id', 'alpha_hat', 'alpha_eb', 'lambda']].rename(
            columns={'alpha_hat': 'theta_hat_plugin', 'alpha_eb': 'theta_hat', 'lambda': 'lambda_captain'}
        ),
        on='captain_id', how='left'
    )
    
    # Merge agent
    df_types = df_types.merge(
        agent_fe[['agent_id', 'gamma_hat', 'gamma_eb', 'lambda']].rename(
            columns={'gamma_hat': 'psi_hat_plugin', 'gamma_eb': 'psi_hat', 'lambda': 'lambda_agent'}
        ),
        on='agent_id', how='left'
    )
    
    # 4. Save type database
    tables_dir = OUTPUT_DIR / 'tables'
    manifests_dir = OUTPUT_DIR / 'manifests'
    
    df_types.to_csv(tables_dir / 'type_estimation_authoritative.csv', index=False)
    df_types.to_parquet(manifests_dir / 'type_file_authoritative.parquet', index=False)
    
    # 5. KSS variance decomposition (pytwoway / FWL approach)
    #
    # Following pytwoway's FEControlEstimator:
    # Step A: Residualize y on controls (log_tonnage, log_duration, decade FEs)
    #         via OLS — this is Frisch-Waugh-Lovell.
    # Step B: Run pure two-way FE model: ỹ = α + γ + ε
    #         (zero nuisance columns → clean KSS leverages)
    # Step C: Apply heteroskedastic KSS correction.
    #
    # Sample: movers only (captains with 2+ agents).
    #
    print("--- Computing KSS Bias-Corrected Variance Shares ---")
    print("    (FWL residualization + pure two-way FE + KSS)")
    try:
        # Reload from parquet to avoid stale state from run_akm_with_eb
        df_fresh = pd.read_parquet(OUTPUT_DIR / 'manifests' / 'canonical_connected_set.parquet')
        df_fresh['log_tonnage'] = np.log(df_fresh['tonnage'].replace(0, np.nan))
        df_fresh['log_duration'] = np.log(df_fresh['duration_days'].replace(0, np.nan))
        for col in ['log_tonnage', 'log_duration']:
            df_fresh[col] = df_fresh[col].fillna(df_fresh[col].median())
        
        # Restrict to movers
        movers = df_fresh.groupby('captain_id')['agent_id'].nunique()
        mover_ids = movers[movers >= 2].index
        df_m = df_fresh[df_fresh['captain_id'].isin(mover_ids)].copy()
        print(f"    Movers: {len(mover_ids)} captains, {len(df_m)} voyages")
        
        # Step A: FWL residualization on controls + decade FEs
        y_raw = df_m['log_q'].values
        var_y_raw = np.var(y_raw)
        
        # Build control matrix: log_tonnage, log_duration, decade dummies
        decade = (df_m['year_out'] // 10) * 10
        decade_dummies = pd.get_dummies(decade, prefix='dec', drop_first=True, dtype=float)
        X_controls = np.column_stack([
            df_m['log_tonnage'].values,
            df_m['log_duration'].values,
            decade_dummies.values,
        ])
        n_decades = decade_dummies.shape[1]
        print(f"    Controls: log_tonnage, log_duration + {n_decades} decade dummies")
        
        # OLS residualization: ỹ = y - X_c @ (X_c'X_c)^{-1} X_c'y
        XtX = X_controls.T @ X_controls
        Xty = X_controls.T @ y_raw
        b_controls = np.linalg.solve(XtX, Xty)
        y_residualized = y_raw - X_controls @ b_controls
        
        print(f"    Var(y_raw) = {var_y_raw:.4f}")
        print(f"    Var(y_residualized) = {np.var(y_residualized):.4f}")
        
        # Step B: Pure two-way FE on residualized y
        # Overwrite log_q with residualized values so estimate_r1 uses them
        df_m['log_q'] = y_residualized
        
        # Remove control columns so estimate_r1 builds a PURE captain+agent design
        df_m = df_m.drop(columns=['log_tonnage', 'log_duration'], errors='ignore')
        # Also drop any _fe_ columns
        fe_cols = [c for c in df_m.columns if c.startswith('_fe_')]
        df_m = df_m.drop(columns=fe_cols, errors='ignore')
        
        r1 = estimate_r1(df_m, dependent_var='log_q', use_loo_sample=False)
        
        # Step C: KSS on the pure two-way model (zero nuisance columns)
        kss = compute_kss_correction(r1, homoskedastic=False)
        
        var_alpha_plugin = kss['var_alpha_plugin']
        var_gamma_plugin = kss['var_gamma_plugin']
        cov_plugin       = kss['cov_plugin']
        var_alpha_kss    = kss['var_alpha_kss']
        var_gamma_kss    = kss['var_gamma_kss']
        cov_kss          = kss['cov_kss']
        
        # Var(y) of the RAW outcome (before residualization)
        var_y_raw = np.var(y_raw)
        
        # Variance explained by the controls themselves
        var_controls = np.var(X_controls @ b_controls)
        
        var_eps = np.var(r1['residuals'])
        total_bias = kss['bias_var_alpha'] + kss['bias_var_gamma']
        var_eps_kss = var_eps + total_bias
        
        components = [
            ("Var(X_c β_c) - Controls & Time FEs", var_controls,      var_controls),
            ("Var(α) - Captain Skill",             var_alpha_plugin,  var_alpha_kss),
            ("Var(γ) - Agent Capability",          var_gamma_plugin,  var_gamma_kss),
            ("2×Cov(α,γ) - Sorting",               2 * cov_plugin,    2 * cov_kss),
            ("Var(ε) - Residual",                  var_eps,           var_eps_kss),
        ]
        
        rows = []
        for label, plug, kss_val in components:
            rows.append({
                'Component': label,
                'Plugin_Var': plug,
                'KSS_Var': kss_val,
                'Plugin_Share_of_VarY_Pct': (plug / var_y_raw) * 100.0,
                'KSS_Share_of_VarY_Pct': (kss_val / var_y_raw) * 100.0,
            })
        
        plugin_sum = sum(p for _, p, _ in components)
        kss_sum    = sum(k for _, _, k in components)
        rows.append({
            'Component': 'Sum of Components',
            'Plugin_Var': plugin_sum,
            'KSS_Var': kss_sum,
            'Plugin_Share_of_VarY_Pct': (plugin_sum / var_y_raw) * 100.0,
            'KSS_Share_of_VarY_Pct': (kss_sum / var_y_raw) * 100.0,
        })
        rows.append({
            'Component': 'Var(Y) - Total Outcome Variance',
            'Plugin_Var': var_y_raw,
            'KSS_Var': var_y_raw,
            'Plugin_Share_of_VarY_Pct': 100.0,
            'KSS_Share_of_VarY_Pct': 100.0,
        })
        
        decomp = pd.DataFrame(rows)
        
        print(f"\n--- Authoritative KSS Variance Decomposition (Movers, FWL) ---")
        print(f"  Var(Y_raw) = {var_y_raw:.4f}  [total variance before residualization]")
        print(f"  {'Component':<35} {'Plugin':>10} {'KSS':>10} {'Plugin%':>10} {'KSS%':>10}")
        for label, plug, kss_val in components:
            print(f"  {label:<35} {plug:>10.4f} {kss_val:>10.4f} {plug/var_y_raw*100:>9.1f}% {kss_val/var_y_raw*100:>9.1f}%")
        print(f"  {'Sum':<35} {plugin_sum:>10.4f} {kss_sum:>10.4f} {plugin_sum/var_y_raw*100:>9.1f}% {kss_sum/var_y_raw*100:>9.1f}%")
        
        corr_p = cov_plugin / (np.sqrt(var_alpha_plugin) * np.sqrt(var_gamma_plugin))
        corr_k = cov_kss / (np.sqrt(var_alpha_kss) * np.sqrt(var_gamma_kss)) if var_alpha_kss > 0 and var_gamma_kss > 0 else 0
        print(f"\n  Corr(α, γ) plugin = {corr_p:.4f}")
        print(f"  Corr(α, γ) KSS    = {corr_k:.4f}")
        
        decomp.to_csv(tables_dir / 'kss_variance_decomposition_authoritative.csv', index=False)
        print("\nSUCCESS: Phase 1 complete.")
    except BaseException as e:
        import traceback
        traceback.print_exc()
        print(f"\nFailed: {e}")

if __name__ == '__main__':
    run_type_estimation()

