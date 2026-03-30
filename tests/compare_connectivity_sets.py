#!/usr/bin/env python
"""
Compare standard connected set vs KSS leave-one-out connected set.

Loads the actual whaling data, computes both sets, and reports
detailed diagnostics on the differences.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analyses.data_loader import prepare_analysis_sample
from src.analyses.connected_set import (
    find_connected_set,
    find_leave_one_out_connected_set,
)


def main() -> None:
    # =========================================================================
    # 1. Load data
    # =========================================================================
    print("=" * 70)
    print("CONNECTIVITY COMPARISON: Standard CC vs KSS LOO")
    print("=" * 70)
    
    df = prepare_analysis_sample()
    df_clean = df.dropna(subset=["log_q", "captain_id", "agent_id"]).copy()
    
    print(f"\n{'─' * 70}")
    print(f"FULL SAMPLE: {len(df_clean):,} voyages, "
          f"{df_clean['captain_id'].nunique():,} captains, "
          f"{df_clean['agent_id'].nunique():,} agents")
    print(f"{'─' * 70}")
    
    # =========================================================================
    # 2. Standard connected set
    # =========================================================================
    df_cc, cc_diag = find_connected_set(df_clean)
    
    # =========================================================================
    # 3. KSS LOO connected set
    # =========================================================================
    df_loo, loo_diag = find_leave_one_out_connected_set(df_cc)
    
    # =========================================================================
    # 4. Comparison table
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON: Standard CC vs KSS LOO Connected Set")
    print("=" * 70)
    
    metrics = {
        "Voyages": (len(df_cc), len(df_loo)),
        "Captains": (df_cc["captain_id"].nunique(), df_loo["captain_id"].nunique()),
        "Agents": (df_cc["agent_id"].nunique(), df_loo["agent_id"].nunique()),
        "Captain-agent pairs": (
            len(df_cc.groupby(["captain_id", "agent_id"]).size()),
            len(df_loo.groupby(["captain_id", "agent_id"]).size()),
        ),
    }
    
    print(f"\n{'Metric':<25} {'Standard CC':>12} {'KSS LOO':>12} {'Dropped':>10} {'% Lost':>8}")
    print("─" * 70)
    for name, (std, loo) in metrics.items():
        dropped = std - loo
        pct = 100 * dropped / std if std > 0 else 0
        print(f"{name:<25} {std:>12,} {loo:>12,} {dropped:>10,} {pct:>7.1f}%")
    
    # =========================================================================
    # 5. Who was lost? Characterize dropped captains
    # =========================================================================
    print("\n" + "=" * 70)
    print("CHARACTERIZING DROPPED CAPTAINS")
    print("=" * 70)
    
    cc_captains = set(df_cc["captain_id"].unique())
    loo_captains = set(df_loo["captain_id"].unique())
    dropped_captains = cc_captains - loo_captains
    
    print(f"\nDropped captains: {len(dropped_captains):,}")
    
    if dropped_captains:
        df_dropped = df_cc[df_cc["captain_id"].isin(dropped_captains)]
        df_kept = df_cc[df_cc["captain_id"].isin(loo_captains)]
        
        # Voyage counts
        dropped_voyages = df_dropped.groupby("captain_id").size()
        kept_voyages = df_kept.groupby("captain_id").size()
        
        # Agent counts (mobility)
        dropped_agents = df_dropped.groupby("captain_id")["agent_id"].nunique()

        # Mover/stayer breakdown
        dropped_movers = (dropped_agents >= 2).sum()
        dropped_stayers = (dropped_agents == 1).sum()
        
        print("\n  Breakdown:")
        print(f"    Movers (2+ agents): {dropped_movers:,}")
        print(f"    Stayers (1 agent):  {dropped_stayers:,}")
        
        print("\n  Dropped captains — voyage count distribution:")
        print(f"    Mean:   {dropped_voyages.mean():.1f}")
        print(f"    Median: {dropped_voyages.median():.0f}")
        print(f"    Min:    {dropped_voyages.min()}")
        print(f"    Max:    {dropped_voyages.max()}")
        
        print("\n  Kept captains — voyage count distribution:")
        print(f"    Mean:   {kept_voyages.mean():.1f}")
        print(f"    Median: {kept_voyages.median():.0f}")
        print(f"    Min:    {kept_voyages.min()}")
        print(f"    Max:    {kept_voyages.max()}")
        
        print("\n  Dropped captains — agent count distribution:")
        print(f"    Mean:   {dropped_agents.mean():.2f}")
        print(f"    1 agent: {(dropped_agents == 1).sum():,}")
        print(f"    2 agents: {(dropped_agents == 2).sum():,}")
        print(f"    3+ agents: {(dropped_agents >= 3).sum():,}")
        
        # Outcome variable comparison
        if "log_q" in df_cc.columns:
            print("\n  Mean log_q comparison:")
            print(f"    Dropped captains' voyages: {df_dropped['log_q'].mean():.3f} (SD={df_dropped['log_q'].std():.3f})")
            print(f"    Kept captains' voyages:    {df_kept['log_q'].mean():.3f} (SD={df_kept['log_q'].std():.3f})")
        
        # Time distribution
        if "year_out" in df_cc.columns:
            print("\n  Year distribution:")
            print(f"    Dropped captains' voyages: {df_dropped['year_out'].min():.0f}–{df_dropped['year_out'].max():.0f}, mean={df_dropped['year_out'].mean():.0f}")
            print(f"    Kept captains' voyages:    {df_kept['year_out'].min():.0f}–{df_kept['year_out'].max():.0f}, mean={df_kept['year_out'].mean():.0f}")
    
    # =========================================================================
    # 6. Dropped agents
    # =========================================================================
    cc_agents = set(df_cc["agent_id"].unique())
    loo_agents = set(df_loo["agent_id"].unique())
    dropped_agents_set = cc_agents - loo_agents
    
    print(f"\n{'─' * 70}")
    print(f"Dropped agents: {len(dropped_agents_set):,}")
    
    if dropped_agents_set:
        df_dropped_a = df_cc[df_cc["agent_id"].isin(dropped_agents_set)]
        dropped_a_captains = df_dropped_a.groupby("agent_id")["captain_id"].nunique()
        dropped_a_voyages = df_dropped_a.groupby("agent_id").size()
        
        print(f"  Mean voyages per dropped agent: {dropped_a_voyages.mean():.1f}")
        print(f"  Mean captains per dropped agent: {dropped_a_captains.mean():.2f}")
    
    # =========================================================================
    # 7. Mobility diagnostics on both sets
    # =========================================================================
    print("\n" + "=" * 70)
    print("MOBILITY COMPARISON")
    print("=" * 70)
    
    # Standard CC mobility
    cc_cap_agents = df_cc.groupby("captain_id")["agent_id"].nunique()
    cc_movers = (cc_cap_agents >= 2).sum()
    cc_mover_rate = cc_movers / len(cc_cap_agents)
    
    # LOO mobility
    loo_cap_agents = df_loo.groupby("captain_id")["agent_id"].nunique()
    loo_movers = (loo_cap_agents >= 2).sum()
    loo_mover_rate = loo_movers / len(loo_cap_agents)
    
    print(f"\n{'Metric':<35} {'Standard CC':>12} {'KSS LOO':>12}")
    print("─" * 62)
    print(f"{'Total captains':<35} {len(cc_cap_agents):>12,} {len(loo_cap_agents):>12,}")
    print(f"{'Movers (2+ agents)':<35} {cc_movers:>12,} {loo_movers:>12,}")
    print(f"{'Mover rate':<35} {100*cc_mover_rate:>11.1f}% {100*loo_mover_rate:>11.1f}%")
    print(f"{'Mean agents per captain':<35} {cc_cap_agents.mean():>12.2f} {loo_cap_agents.mean():>12.2f}")
    
    # Agent-side
    cc_agent_caps = df_cc.groupby("agent_id")["captain_id"].nunique()
    loo_agent_caps = df_loo.groupby("agent_id")["captain_id"].nunique()
    
    print(f"{'Mean captains per agent':<35} {cc_agent_caps.mean():>12.2f} {loo_agent_caps.mean():>12.2f}")
    
    # Repeat partnerships
    cc_pair_counts = df_cc.groupby(["captain_id", "agent_id"]).size()
    loo_pair_counts = df_loo.groupby(["captain_id", "agent_id"]).size()
    cc_repeat = (cc_pair_counts > 1).sum() / len(cc_pair_counts)
    loo_repeat = (loo_pair_counts > 1).sum() / len(loo_pair_counts)
    print(f"{'Repeat partnership rate':<35} {100*cc_repeat:>11.1f}% {100*loo_repeat:>11.1f}%")
    
    # =========================================================================
    # 8. Var(y) comparison
    # =========================================================================
    if "log_q" in df_cc.columns and "log_q" in df_loo.columns:
        print(f"\n{'─' * 70}")
        print("OUTCOME VARIANCE")
        print(f"  Standard CC: Var(log_q) = {np.var(df_cc['log_q']):.4f}")
        print(f"  KSS LOO:     Var(log_q) = {np.var(df_loo['log_q']):.4f}")
    
    # =========================================================================
    # 9. Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    v_pct = 100 * (len(df_cc) - len(df_loo)) / len(df_cc)
    c_pct = 100 * len(dropped_captains) / len(cc_captains)
    a_pct = 100 * len(dropped_agents_set) / len(cc_agents)
    
    print("\n  The KSS LOO connected set drops:")
    print(f"    {len(df_cc) - len(df_loo):,} voyages ({v_pct:.1f}% of standard CC)")
    print(f"    {len(dropped_captains):,} captains ({c_pct:.1f}%)")
    print(f"    {len(dropped_agents_set):,} agents ({a_pct:.1f}%)")
    print(f"\n  LOO pruning required {loo_diag['iterations']} iterations")
    print(f"  {loo_diag['articulation_captains_pruned']} articulation-point captains identified and removed")
    print(f"  {loo_diag.get('single_obs_stayers_dropped', 0)} single-observation stayers dropped")


if __name__ == "__main__":
    main()
