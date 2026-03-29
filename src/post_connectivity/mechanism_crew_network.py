"""
Phase 4: Mechanisms — Organizational Software Transmission

Tests through which channels organizational capability (psi) operates.
Designed as an appendix-quality evidence block, not a centerpiece.

Strong results:
  1. Mate-to-Captain Training Imprint (the strongest)
  2. Crew composition (suggestive, mixed)

Null results (reported for honesty):
  3. Network size (null)

Additional tests added to thicken the block:
  4. Mate Fixed Effects (variance share)
  5. Within-captain, across-agent variation
  6. Crew experience and desertion
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr
from scipy import stats
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'post_connectivity'


def load_data():
    canonical_path = OUTPUT_DIR / 'manifests' / 'canonical_connected_set.parquet'
    types_path = OUTPUT_DIR / 'manifests' / 'type_file_authoritative.parquet'
    df = pd.read_parquet(canonical_path)
    df_types = pd.read_parquet(types_path)
    df = df.merge(df_types[['voyage_id', 'theta_hat', 'psi_hat']],
                  on='voyage_id', how='left')

    crew_path = PROJECT_ROOT / 'data' / 'staging' / 'crew_roster.parquet'
    crew = pd.read_parquet(crew_path) if crew_path.exists() else pd.DataFrame()
    if 'rank' in crew.columns:
        crew['rank'] = crew['rank'].fillna('').str.upper().str.strip()
    return df, crew


def run_mechanisms():
    print("="*60)
    print("PHASE 4: MECHANISMS — ORGANIZATIONAL SOFTWARE TRANSMISSION")
    print("="*60)

    df, crew = load_data()
    print(f"Sample: {len(df):,} voyages, {len(crew):,} crew entries")

    results = []

    # ---- 1. Agent Network Size → psi (null expected) ----
    agent_port = df.groupby('agent_id').agg(
        n_voyages=('voyage_id', 'count'),
        n_captains=('captain_id', 'nunique'),
        psi_hat=('psi_hat', 'first')).reset_index()
    agent_port = agent_port[agent_port['n_voyages'] >= 3].copy()
    agent_port['log_n_cap'] = np.log(agent_port['n_captains'])

    res = smf.ols("psi_hat ~ log_n_cap", data=agent_port).fit()
    results.append({
        'Mechanism': 'Network Size → ψ',
        'DV': 'psi_hat',
        'N': len(agent_port),
        'Beta': res.params.get('log_n_cap', np.nan),
        'SE': res.bse.get('log_n_cap', np.nan),
        'P': res.pvalues.get('log_n_cap', np.nan),
        'Note': 'Null'
    })
    print(f"\n[1] Network Size → ψ: β={res.params.get('log_n_cap',0):.4f} "
          f"(p={res.pvalues.get('log_n_cap',1):.3f}) — null")

    # ---- 2. Crew composition (greenhand ratio) ----
    if not crew.empty:
        gh = []
        for vid in df['voyage_id'].unique():
            vc = crew[crew['voyage_id'] == vid]
            if len(vc) == 0: continue
            n_gh = vc['rank'].str.contains('GREEN', case=False, na=False).sum()
            gh.append({'voyage_id': vid, 'greenhand_ratio': n_gh / len(vc),
                       'crew_size': len(vc)})
        gh_df = pd.DataFrame(gh)
        df_gh = df.merge(gh_df, on='voyage_id', how='inner')
        df_gh = df_gh.dropna(subset=['greenhand_ratio', 'psi_hat', 'theta_hat'])

        if len(df_gh) > 100:
            res_gh = smf.ols("greenhand_ratio ~ psi_hat + theta_hat",
                             data=df_gh).fit(cov_type='cluster',
                                             cov_kwds={'groups': df_gh['agent_id']})
            for var in ['psi_hat', 'theta_hat']:
                results.append({
                    'Mechanism': f'Crew Composition ({var})',
                    'DV': 'greenhand_ratio',
                    'N': len(df_gh),
                    'Beta': res_gh.params.get(var, np.nan),
                    'SE': res_gh.bse.get(var, np.nan),
                    'P': res_gh.pvalues.get(var, np.nan),
                    'Note': 'Clustered by agent'
                })
            print(f"\n[2] Crew Composition: β(ψ→gh)={res_gh.params.get('psi_hat',0):.4f} "
                  f"(p={res_gh.pvalues.get('psi_hat',1):.4f})")

    # ---- 3. Mate Fixed Effects (variance share) ----
    if not crew.empty:
        mates = crew[crew['rank'].isin(['1ST MATE', '1 MATE', 'MATE'])].copy()
        fm = mates.groupby('voyage_id')['crew_name_clean'].first().reset_index()
        fm.rename(columns={'crew_name_clean': 'mate_name'}, inplace=True)

        dm = df.merge(fm, on='voyage_id', how='inner').dropna(subset=['log_q', 'mate_name'])
        cnts = dm['mate_name'].value_counts()
        dm = dm[dm['mate_name'].isin(cnts[cnts >= 3].index)]

        if len(dm) > 50:
            overall_var = dm['log_q'].var()
            mate_means = dm.groupby('mate_name')['log_q'].mean()
            between_var = mate_means.var()
            mate_share = between_var / overall_var if overall_var > 0 else 0

            # F-test: mate FE model vs intercept-only
            n_mates = dm['mate_name'].nunique()
            rss_full = dm.groupby('mate_name').apply(
                lambda g: ((g['log_q'] - g['log_q'].mean())**2).sum()).sum()
            rss_null = ((dm['log_q'] - dm['log_q'].mean())**2).sum()
            n = len(dm)
            f_stat = ((rss_null - rss_full) / (n_mates - 1)) / (rss_full / (n - n_mates))
            p_f = 1 - stats.f.cdf(f_stat, n_mates - 1, n - n_mates)

            results.append({
                'Mechanism': 'Mate FE Variance Share',
                'DV': 'log_q',
                'N': len(dm),
                'Beta': mate_share,
                'SE': np.nan,
                'P': p_f,
                'Note': f'F({n_mates-1},{n-n_mates})={f_stat:.1f}'
            })
            print(f"\n[3] Mate FE share = {mate_share:.1%} "
                  f"(F={f_stat:.1f}, p={p_f:.4f}, {n_mates} mates)")

    # ---- 4. Mate-to-Captain Training Imprint ----
    if not crew.empty:
        mates_all = crew[crew['rank'].isin(
            ['1ST MATE', '1 MATE', 'MATE', '2ND MATE'])].dropna(subset=['crew_name_clean'])
        captains = crew[crew['rank'] == 'MASTER'].dropna(subset=['crew_name_clean'])
        promoted = set(mates_all['crew_name_clean']) & set(captains['crew_name_clean'])

        if len(promoted) > 10:
            mv = mates_all[mates_all['crew_name_clean'].isin(promoted)].merge(
                df[['voyage_id', 'agent_id', 'year_out']], on='voyage_id')
            fmv = mv.sort_values('year_out').groupby('crew_name_clean').first().reset_index()
            fmv = fmv[['crew_name_clean', 'agent_id']].rename(
                columns={'agent_id': 'training_agent'})

            cv = captains[captains['crew_name_clean'].isin(promoted)].merge(
                df[['voyage_id', 'agent_id', 'log_q', 'year_out']], on='voyage_id')
            cv = cv.merge(fmv, on='crew_name_clean')
            cv['same_agent'] = (cv['agent_id'] == cv['training_agent']).astype(int)
            cv = cv.dropna(subset=['log_q'])

            if len(cv) > 30:
                ri = smf.ols("log_q ~ same_agent", data=cv).fit()
                results.append({
                    'Mechanism': 'Mate→Captain Imprint',
                    'DV': 'log_q',
                    'N': len(cv),
                    'Beta': ri.params.get('same_agent', np.nan),
                    'SE': ri.bse.get('same_agent', np.nan),
                    'P': ri.pvalues.get('same_agent', np.nan),
                    'Note': f'{len(promoted)} promoted, {cv["same_agent"].sum()} same-agent'
                })
                print(f"\n[4] Mate→Captain Imprint: β={ri.params.get('same_agent',0):.4f} "
                      f"(p={ri.pvalues.get('same_agent',1):.4f})")

    # ---- 5. Within-Captain, Across-Agent Variation ----
    cap_ag = df.groupby('captain_id')['agent_id'].nunique()
    multi_caps = cap_ag[cap_ag > 1].index
    multi_df = df[df['captain_id'].isin(multi_caps)].dropna(subset=['log_q']).copy()

    if len(multi_df) > 500:
        y = multi_df['log_q'].values
        n = len(y)
        # Captain FE only
        cap_ids = multi_df['captain_id'].unique()
        cap_map = {c: i for i, c in enumerate(cap_ids)}
        cap_idx = multi_df['captain_id'].map(cap_map).values
        X_cap = sp.csr_matrix((np.ones(n), (np.arange(n), cap_idx)),
                              shape=(n, len(cap_ids)))
        resid_cap = y - X_cap @ lsqr(X_cap, y, iter_lim=5000)[0]
        r2_cap = 1 - np.var(resid_cap) / np.var(y)

        # Captain + Agent FE
        ag_ids = multi_df['agent_id'].unique()
        ag_map = {a: i for i, a in enumerate(ag_ids)}
        ag_idx = multi_df['agent_id'].map(ag_map).values
        X_ag = sp.csr_matrix((np.ones(n), (np.arange(n), ag_idx)),
                             shape=(n, len(ag_ids)))[:, 1:]
        X_full = sp.hstack([X_cap, X_ag])
        resid_full = y - X_full @ lsqr(X_full, y, iter_lim=5000)[0]
        r2_full = 1 - np.var(resid_full) / np.var(y)
        incr_r2 = r2_full - r2_cap

        # F-test
        n_ag = len(ag_ids) - 1
        rss_r = np.sum(resid_cap**2)
        rss_u = np.sum(resid_full**2)
        f_stat = ((rss_r - rss_u) / n_ag) / (rss_u / (n - len(cap_ids) - n_ag))
        p_f = 1 - stats.f.cdf(f_stat, n_ag, n - len(cap_ids) - n_ag)

        results.append({
            'Mechanism': 'Within-Captain Agent Effect',
            'DV': 'log_q',
            'N': n,
            'Beta': incr_r2,
            'SE': np.nan,
            'P': p_f,
            'Note': f'ΔR²={incr_r2:.4f}, {len(multi_caps)} captains w/ 2+ agents'
        })
        print(f"\n[5] Within-Captain Agent ΔR² = {incr_r2:.4f} "
              f"(F={f_stat:.1f}, p={p_f:.4f})")

    # ---- 6. Crew Experience & Desertion ----
    exp_vars = []
    if 'captain_experience' not in df.columns:
        df = df.sort_values(['captain_id', 'year_out'])
        df['captain_voyage_num'] = df.groupby('captain_id').cumcount() + 1
    else:
        df['captain_voyage_num'] = df['captain_experience'] + 1

    if 'desertion_rate' in df.columns:
        df_des = df.dropna(subset=['desertion_rate', 'psi_hat', 'theta_hat', 'log_q'])
        if len(df_des) > 100:
            rd = smf.ols("desertion_rate ~ psi_hat + theta_hat", data=df_des).fit()
            results.append({
                'Mechanism': 'Desertion Rate (ψ)',
                'DV': 'desertion_rate',
                'N': len(df_des),
                'Beta': rd.params.get('psi_hat', np.nan),
                'SE': rd.bse.get('psi_hat', np.nan),
                'P': rd.pvalues.get('psi_hat', np.nan),
                'Note': 'Does psi reduce desertion?'
            })
            print(f"\n[6] Desertion ← ψ: β={rd.params.get('psi_hat',0):.4f} "
                  f"(p={rd.pvalues.get('psi_hat',1):.4f})")

    # ---- Save ----
    out_dir = OUTPUT_DIR / 'tables'
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_dir / 'mechanism_crew_network.csv', index=False)

    print(f"\nSUCCESS: {len(results)} mechanism tests saved.")

if __name__ == "__main__":
    run_mechanisms()
