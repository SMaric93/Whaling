"""
Phase 3C: Stopping Hazard — State-Dependent Search Discipline

Tests whether organizational capability (psi) alters patch-residence
behaviour differently depending on the LOCAL signal quality.

Reports BOTH signal definitions side-by-side so the reader can judge
sensitivity. All coefficients are on an "Exit Speed" scale:
  positive = the captain leaves the patch FASTER.

Key finding: "fail fast" (empty × psi) is weak and sensitive to
definition; "succeed fast" (rich × psi) is robust across all models.
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path
import warnings

try:
    from lifelines import CoxPHFitter, WeibullAFTFitter
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'post_connectivity'

def _run_ols(df, lbl, neg_col, pos_col):
    """OLS on log_duration with ocean + decade FEs."""
    formula = (f"log_duration ~ psi_std * {neg_col} + psi_std * {pos_col}"
               f" + theta_hat + log_tonnage + C(ocean_id) + C(decade)")
    m = smf.ols(formula, data=df).fit(
        cov_type='cluster', cov_kwds={'groups': df['voyage_id']})
    
    neg_int = f"psi_std:{neg_col}"
    pos_int = f"psi_std:{pos_col}"
    # OLS DV is duration → negate for Exit Speed
    return {
        'Model_Type': 'OLS',
        'Signal_Definition': lbl,
        'Exit_Speed_Empty': -m.params.get(neg_int, np.nan),
        'P_Empty': m.pvalues.get(neg_int, np.nan),
        'Exit_Speed_Rich': -m.params.get(pos_int, np.nan),
        'P_Rich': m.pvalues.get(pos_int, np.nan),
    }

def _run_survival(df_surv, lbl, neg_col, pos_col, ocean_cols, decade_cols):
    """Cox PH and Weibull AFT with the same covariates."""
    cols = (['duration_days', 'exit_event', 'psi_std', 'theta_hat',
             'log_tonnage', neg_col, pos_col] + ocean_cols + decade_cols)
    s = df_surv[cols].copy()
    s['psi_X_empty'] = s['psi_std'] * s[neg_col]
    s['psi_X_rich']  = s['psi_std'] * s[pos_col]
    
    rows = []
    
    # Cox
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(s, duration_col='duration_days', event_col='exit_event')
    rows.append({
        'Model_Type': 'Cox',
        'Signal_Definition': lbl,
        'Exit_Speed_Empty': cph.summary.loc['psi_X_empty', 'coef'],   # hazard ↑ = faster
        'P_Empty': cph.summary.loc['psi_X_empty', 'p'],
        'Exit_Speed_Rich': cph.summary.loc['psi_X_rich', 'coef'],
        'P_Rich': cph.summary.loc['psi_X_rich', 'p'],
    })
    
    # AFT
    aft = WeibullAFTFitter(penalizer=0.01)
    aft.fit(s, duration_col='duration_days', event_col='exit_event')
    rows.append({
        'Model_Type': 'AFT',
        'Signal_Definition': lbl,
        'Exit_Speed_Empty': -aft.summary.loc[('lambda_', 'psi_X_empty'), 'coef'],  # negate
        'P_Empty': aft.summary.loc[('lambda_', 'psi_X_empty'), 'p'],
        'Exit_Speed_Rich': -aft.summary.loc[('lambda_', 'psi_X_rich'), 'coef'],
        'P_Rich': aft.summary.loc[('lambda_', 'psi_X_rich'), 'p'],
    })
    
    return rows


def run_stopping_hazard():
    print("="*60)
    print("PHASE 3C: STATE-DEPENDENT SEARCH DISCIPLINE")
    print("="*60)

    patch_file = PROJECT_ROOT / 'output' / 'stopping_rule' / 'patches.csv'
    if not patch_file.exists():
        print(f"Error: {patch_file} not found"); return

    df_patches = pd.read_csv(patch_file)
    df_types = pd.read_csv(OUTPUT_DIR / 'tables' / 'type_estimation_authoritative.csv')
    df_canon = pd.read_parquet(OUTPUT_DIR / 'manifests' / 'canonical_connected_set.parquet')

    df = df_patches.merge(df_types[['voyage_id', 'psi_hat', 'theta_hat']],
                          on='voyage_id', how='inner')
    df = df.merge(df_canon[['voyage_id', 'voyage_region', 'year_out', 'tonnage']],
                  on='voyage_id', how='left')

    df['duration_days'] = df['duration_days'].replace(0, 0.5)
    df['log_duration']  = np.log(df['duration_days'])
    df['log_tonnage']   = np.log(df['tonnage'].replace(0, np.nan)).fillna(0)
    df['decade']   = (df['year_out'] // 10 * 10).astype(str)
    df['ocean_id'] = df['voyage_region'].fillna('Unknown').astype(str)
    df = df.dropna(subset=['psi_hat', 'duration_days', 'catch_rate']).copy()
    df['psi_std'] = (df['psi_hat'] - df['psi_hat'].mean()) / df['psi_hat'].std()

    pos_rates = df.loc[df['catch_rate'] > 0, 'catch_rate']
    if len(pos_rates) == 0: pos_rates = df['catch_rate']

    # ---------- Two signal definitions ----------
    q10  = pos_rates.quantile(0.10)
    q25  = pos_rates.quantile(0.25)
    q75  = pos_rates.quantile(0.75)

    df['neg_q10'] = (df['catch_rate'] <= q10).astype(int)
    df['neg_q25'] = (df['catch_rate'] <= q25).astype(int)
    df['pos_q75'] = (df['catch_rate'] >= q75).astype(int)

    signal_defs = [
        ('Q10_Q75', 'neg_q10', 'pos_q75'),   # strict
        ('Q25_Q75', 'neg_q25', 'pos_q75'),    # looser (prior rerun)
    ]

    print(f"Sample: {len(df):,} patches")
    results = []

    # OLS
    print("\n--- OLS (log duration) ---")
    for lbl, neg, pos in signal_defs:
        r = _run_ols(df, lbl, neg, pos)
        results.append(r)
        print(f"  [{lbl}] Empty×ψ = {r['Exit_Speed_Empty']:+.4f} (p={r['P_Empty']:.3f})  "
              f"Rich×ψ = {r['Exit_Speed_Rich']:+.4f} (p={r['P_Rich']:.3f})")

    # Survival models
    if HAS_LIFELINES:
        df['exit_event'] = 1
        df_surv = pd.get_dummies(df, columns=['ocean_id', 'decade'], drop_first=True)
        ocean_cols  = [c for c in df_surv.columns if c.startswith('ocean_id_')]
        decade_cols = [c for c in df_surv.columns if c.startswith('decade_')]

        print("\n--- Survival models (Cox PH, Weibull AFT) ---")
        for lbl, neg, pos in signal_defs:
            try:
                for r in _run_survival(df_surv, lbl, neg, pos, ocean_cols, decade_cols):
                    results.append(r)
                    print(f"  [{r['Model_Type']}|{lbl}] Empty×ψ = {r['Exit_Speed_Empty']:+.4f} "
                          f"(p={r['P_Empty']:.3f})  Rich×ψ = {r['Exit_Speed_Rich']:+.4f} "
                          f"(p={r['P_Rich']:.3f})")
            except Exception as e:
                print(f"  [{lbl}] survival failed: {e}")

    df_res = pd.DataFrame(results)
    df_res.to_csv(OUTPUT_DIR / 'tables' / 'stopping_hazard_authoritative.csv', index=False)

    print("\nSUCCESS: Stopping hazard saved.")
    print("Key: positive coeff = FASTER exit.  Both signal defs shown for sensitivity.")

if __name__ == '__main__':
    run_stopping_hazard()
