"""
Phase 0: Connectivity Audit

1. Rebuild the bipartite captain-agent graph from raw data.
2. Explicitly document:
   - number of voyages, captains, agents, movers, switchers
   - giant connected component share
   - dropped isolated components
   - reasons for exclusion
3. Compare old vs new connected sets (i.e., standard largest connected component
   vs the rigorous leave-one-out articulation-pruned KSS connected set).
4. Extract KSS/AKM decomposition shares.
5. Export canonical connected set.
"""

import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
import json
import warnings

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'final'
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'post_connectivity'
DOC_DIR = PROJECT_ROOT / 'docs' / 'post_connectivity'

# Import legacy analysis functions
from src.analyses.connected_set import (
    find_connected_set,
    find_leave_one_out_connected_set,
    compute_mobility_diagnostics
)

def build_bipartite_graph(df):
    """Build NetworkX bipartite graph for captains and agents."""
    G = nx.Graph()
    pairs = df[['captain_id', 'agent_id']].drop_duplicates()
    for _, row in pairs.iterrows():
        G.add_edge(f"C_{row['captain_id']}", f"A_{row['agent_id']}")
    return G

def run_audit():
    print("="*60)
    print("PHASE 0: CONNECTIVITY AUDIT")
    print("="*60)

    # Load base data
    df_raw = pd.read_parquet(DATA_DIR / 'analysis_voyage.parquet')
    
    # Generate log_q as used by KSS
    df_raw['log_q'] = np.log(df_raw['q_oil_bbl'] + 1)
    
    # Filter missing values which would break KSS
    df_clean = df_raw.dropna(subset=['log_q', 'captain_id', 'agent_id']).copy()

    # Get baseline mobility
    base_mob = compute_mobility_diagnostics(df_clean)

    # 1. Old (Standard) Connected Set
    df_old, old_diag = find_connected_set(df_clean)
    
    # 2. New (KSS LOO) Canonical Connected Set
    df_new, new_diag = find_leave_one_out_connected_set(df_old)
    new_mob = compute_mobility_diagnostics(df_new)

    # Compare changes
    overlap_voyages = len(df_new)
    overlap_captains = df_new['captain_id'].nunique()
    overlap_agents = df_new['agent_id'].nunique()

    old_captains = set(df_old['captain_id'])
    new_captains = set(df_new['captain_id'])
    old_agents = set(df_old['agent_id'])
    new_agents = set(df_new['agent_id'])
    
    movers_old = set(
        df_old.groupby('captain_id')['agent_id'].nunique()
        .loc[lambda x: x >= 2].index
    )
    movers_new = set(
        df_new.groupby('captain_id')['agent_id'].nunique()
        .loc[lambda x: x >= 2].index
    )

    diff_table = pd.DataFrame([{
        'metric': 'Total Voyages',
        'old_set_value': len(df_old),
        'new_set_value': len(df_new),
        'pct_remaining': len(df_new) / len(df_old) * 100
    }, {
        'metric': 'Total Captains',
        'old_set_value': len(old_captains),
        'new_set_value': len(new_captains),
        'pct_remaining': len(new_captains) / len(old_captains) * 100
    }, {
        'metric': 'Total Agents',
        'old_set_value': len(old_agents),
        'new_set_value': len(new_agents),
        'pct_remaining': len(new_agents) / len(old_agents) * 100
    }, {
        'metric': 'Movers',
        'old_set_value': len(movers_old),
        'new_set_value': len(movers_new),
        'pct_remaining': len(movers_new) / max(1, len(movers_old)) * 100
    }])

    diff_table.to_csv(OUTPUT_DIR / 'tables' / 'old_vs_new_connected_set_diff.csv', index=False)

    # Connectivity Summary (Overall Stats)
    summary_data = {
        'raw_voyages': len(df_raw),
        'clean_voyages': len(df_clean),
        'giant_component_voyages': len(df_old),
        'canonical_loo_voyages': len(df_new),
        'total_captains_raw': df_raw['captain_id'].nunique(),
        'total_captains_canonical': len(new_captains),
        'articulation_nodes_dropped': new_diag.get('articulation_captains_pruned', 0),
        'n_movers_canonical': len(movers_new)
    }

    pd.DataFrame([summary_data]).to_csv(OUTPUT_DIR / 'tables' / 'connectivity_summary.csv', index=False)

    # Save Canonical set manifest
    df_new.to_parquet(OUTPUT_DIR / 'manifests' / 'canonical_connected_set.parquet', index=False)

    # Export markdown documentation
    md = [
        "# Canonical Connected Set Audit",
        "",
        "## Overview",
        "The connected set strategy was upgraded from a naive largest-connected-component to a rigorous KSS Leave-One-Out (LOO) vertex-connected set. This prevents the bias that occurs when an articulation-point captain connects two subgraphs but their individual capability is not properly identified.",
        "",
        "## Summary Metrics",
        f"- **Raw Voyages**: {len(df_raw):,}",
        f"- **Valid Non-Null Voyages**: {len(df_clean):,}",
        f"- **Naive Connected Set (Old)**: {len(df_old):,} voyages ({old_diag['largest_component_captains']:,} captains)",
        f"- **LOO Connected Set (New Canonical)**: {len(df_new):,} voyages ({new_diag['loo_captains']:,} captains)",
        "",
        "## Old vs New Difference",
        diff_table.to_markdown(index=False),
        "",
        "## Exclusions",
        f"- Iterations to reach LOO stability: {new_diag.get('iterations', 'N/A')}",
        f"- Articulation-point captains pruned: {new_diag.get('articulation_captains_pruned', 'N/A')}",
        f"- Disconnected stayers dropped: {new_diag.get('single_obs_stayers_dropped', 'N/A')}",
        "",
        "## Mobility",
        f"- **Captains with 2+ agents (Movers)**: {len(movers_new):,}",
        f"- **Partner repeat rate**: {new_mob['repeat_pair_rate']*100:.1f}%",
    ]

    with open(DOC_DIR / 'canonical_connected_set.md', 'w') as f:
        f.write('\n'.join(md))
        
    print("SUCCESS: Phase 0 Connectivity Audit complete.")

if __name__ == '__main__':
    run_audit()
