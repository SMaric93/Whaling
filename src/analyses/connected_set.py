"""
Connected set analysis for AKM-style fixed effects estimation.

Implements:
- Standard connected component identification
- Leave-one-out (LOO) connected set for KSS estimation
- Mobility diagnostics
- Network density metrics
"""

from typing import Dict, Tuple, Optional, Set
import warnings

import numpy as np
import pandas as pd
import networkx as nx

warnings.filterwarnings("ignore", category=FutureWarning)


def find_connected_set(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Find largest connected component in captain-agent bipartite graph.
    
    Parameters
    ----------
    df : pd.DataFrame
        Voyage data with captain_id and agent_id.
        
    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        (filtered_df, diagnostics_dict)
    """
    print("\n" + "=" * 60)
    print("FINDING CONNECTED SET")
    print("=" * 60)
    
    # Build bipartite graph
    G = nx.Graph()
    pairs = df[["captain_id", "agent_id"]].drop_duplicates()
    for _, row in pairs.iterrows():
        G.add_edge(f"C_{row['captain_id']}", f"A_{row['agent_id']}")
    
    # Find connected components
    components = list(nx.connected_components(G))
    largest_cc = max(components, key=len)
    
    # Extract IDs from largest component
    connected_captains = {n[2:] for n in largest_cc if n.startswith("C_")}
    connected_agents = {n[2:] for n in largest_cc if n.startswith("A_")}
    
    # Filter data
    df_cc = df[
        df["captain_id"].isin(connected_captains) & 
        df["agent_id"].isin(connected_agents)
    ].copy()
    
    # Diagnostics
    diagnostics = {
        "n_components": len(components),
        "largest_component_captains": len(connected_captains),
        "largest_component_agents": len(connected_agents),
        "total_captains": df["captain_id"].nunique(),
        "total_agents": df["agent_id"].nunique(),
        "voyages_in_connected_set": len(df_cc),
        "total_voyages": len(df),
        "coverage_captains": len(connected_captains) / df["captain_id"].nunique(),
        "coverage_agents": len(connected_agents) / df["agent_id"].nunique(),
        "coverage_voyages": len(df_cc) / len(df),
    }
    
    print(f"Connected components: {diagnostics['n_components']}")
    print(f"Largest connected set:")
    print(f"  Captains: {diagnostics['largest_component_captains']:,} / {diagnostics['total_captains']:,} ({100*diagnostics['coverage_captains']:.1f}%)")
    print(f"  Agents: {diagnostics['largest_component_agents']:,} / {diagnostics['total_agents']:,} ({100*diagnostics['coverage_agents']:.1f}%)")
    print(f"  Voyages: {diagnostics['voyages_in_connected_set']:,} / {diagnostics['total_voyages']:,} ({100*diagnostics['coverage_voyages']:.1f}%)")
    
    return df_cc, diagnostics


def find_leave_one_out_connected_set(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Find leave-one-out connected set for KSS estimation.
    
    An observation is in the LOO set if removing it does not disconnect
    the captain-agent graph. This requires that each edge appears at
    least twice OR both endpoints have multiple edges.
    
    Parameters
    ----------
    df : pd.DataFrame
        Voyage data with captain_id and agent_id.
        
    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        (filtered_df, diagnostics_dict)
    """
    print("\n" + "=" * 60)
    print("FINDING LEAVE-ONE-OUT CONNECTED SET (KSS)")
    print("=" * 60)
    
    df_loo = df.copy()
    initial_n = len(df_loo)
    
    # Count articulation edges (pairs appearing only once)
    initial_pairs = df_loo.groupby(["captain_id", "agent_id"]).size()
    single_pairs = (initial_pairs == 1).sum()
    total_pairs = len(initial_pairs)
    
    print(f"Initial captain-agent pairs: {total_pairs:,}")
    print(f"Single-appearance pairs (articulation edges): {single_pairs:,} ({100*single_pairs/total_pairs:.1f}%)")
    
    # Iteratively prune until stable
    prev_n = 0
    iteration = 0
    
    while len(df_loo) != prev_n:
        prev_n = len(df_loo)
        iteration += 1
        
        # Pair counts
        pair_counts = df_loo.groupby(["captain_id", "agent_id"]).size()
        df_loo["_pair"] = list(zip(df_loo["captain_id"], df_loo["agent_id"]))
        df_loo["_pair_count"] = df_loo["_pair"].map(pair_counts)
        
        # Captain and agent connectivity
        captain_n_agents = df_loo.groupby("captain_id")["agent_id"].transform("nunique")
        agent_n_captains = df_loo.groupby("agent_id")["captain_id"].transform("nunique")
        
        # Keep if: pair appears >1 OR (captain has >1 agent AND agent has >1 captain)
        keep_mask = (df_loo["_pair_count"] > 1) | ((captain_n_agents > 1) & (agent_n_captains > 1))
        df_loo = df_loo[keep_mask].copy()
        
        # Ensure still in largest connected component
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
    
    # Clean up
    df_loo = df_loo.drop(columns=["_pair", "_pair_count"], errors="ignore")
    
    # Diagnostics
    diagnostics = {
        "iterations": iteration,
        "initial_voyages": initial_n,
        "loo_voyages": len(df_loo),
        "loo_captains": df_loo["captain_id"].nunique(),
        "loo_agents": df_loo["agent_id"].nunique(),
        "coverage": len(df_loo) / initial_n if initial_n > 0 else 0,
        "initial_pairs": total_pairs,
        "articulation_edges": single_pairs,
        "articulation_rate": single_pairs / total_pairs if total_pairs > 0 else 0,
    }
    
    print(f"\nLOO pruning complete:")
    print(f"  Iterations: {diagnostics['iterations']}")
    print(f"  Voyages: {diagnostics['loo_voyages']:,} / {diagnostics['initial_voyages']:,} ({100*diagnostics['coverage']:.1f}%)")
    print(f"  Captains: {diagnostics['loo_captains']:,}")
    print(f"  Agents: {diagnostics['loo_agents']:,}")
    
    return df_loo, diagnostics


def compute_mobility_diagnostics(df: pd.DataFrame) -> Dict:
    """
    Compute mobility diagnostics for the captain-agent labor market.
    
    Parameters
    ----------
    df : pd.DataFrame
        Voyage data.
        
    Returns
    -------
    Dict
        Mobility statistics.
    """
    print("\n" + "=" * 60)
    print("MOBILITY DIAGNOSTICS")
    print("=" * 60)
    
    # Captain mobility: % with 2+ agents
    captain_agent_counts = df.groupby("captain_id")["agent_id"].nunique()
    multi_agent_captains = (captain_agent_counts >= 2).sum()
    
    # Agent hiring: % with 2+ captains
    agent_captain_counts = df.groupby("agent_id")["captain_id"].nunique()
    multi_captain_agents = (agent_captain_counts >= 2).sum()
    
    # Average number of agents per captain and vice versa
    mean_agents_per_captain = captain_agent_counts.mean()
    mean_captains_per_agent = agent_captain_counts.mean()
    
    # Pair frequency distribution
    pair_counts = df.groupby(["captain_id", "agent_id"]).size()
    single_pairs = (pair_counts == 1).sum()
    repeat_pairs = (pair_counts > 1).sum()
    
    diagnostics = {
        "n_captains": df["captain_id"].nunique(),
        "n_agents": df["agent_id"].nunique(),
        "n_voyages": len(df),
        "n_captain_agent_pairs": len(pair_counts),
        "multi_agent_captains": multi_agent_captains,
        "multi_agent_captain_rate": multi_agent_captains / df["captain_id"].nunique(),
        "multi_captain_agents": multi_captain_agents,
        "multi_captain_agent_rate": multi_captain_agents / df["agent_id"].nunique(),
        "mean_agents_per_captain": mean_agents_per_captain,
        "mean_captains_per_agent": mean_captains_per_agent,
        "single_pairs": single_pairs,
        "repeat_pairs": repeat_pairs,
        "repeat_pair_rate": repeat_pairs / len(pair_counts) if len(pair_counts) > 0 else 0,
    }
    
    print(f"Sample: {diagnostics['n_voyages']:,} voyages, {diagnostics['n_captains']:,} captains, {diagnostics['n_agents']:,} agents")
    print(f"\nCaptain mobility:")
    print(f"  Captains with 2+ agents: {diagnostics['multi_agent_captains']:,} ({100*diagnostics['multi_agent_captain_rate']:.1f}%)")
    print(f"  Mean agents per captain: {diagnostics['mean_agents_per_captain']:.2f}")
    print(f"\nAgent hiring:")
    print(f"  Agents with 2+ captains: {diagnostics['multi_captain_agents']:,} ({100*diagnostics['multi_captain_agent_rate']:.1f}%)")
    print(f"  Mean captains per agent: {diagnostics['mean_captains_per_agent']:.2f}")
    print(f"\nPartnership structure:")
    print(f"  Total captain-agent pairs: {diagnostics['n_captain_agent_pairs']:,}")
    print(f"  Repeat pairs (2+ voyages): {diagnostics['repeat_pairs']:,} ({100*diagnostics['repeat_pair_rate']:.1f}%)")
    print(f"  Single-voyage pairs: {diagnostics['single_pairs']:,} ({100*(1-diagnostics['repeat_pair_rate']):.1f}%)")
    
    return diagnostics


def compute_network_density(df: pd.DataFrame) -> Dict:
    """
    Compute network density metrics for captain-agent graph.
    
    Parameters
    ----------
    df : pd.DataFrame
        Voyage data.
        
    Returns
    -------
    Dict
        Network statistics.
    """
    # Build bipartite graph
    G = nx.Graph()
    pairs = df[["captain_id", "agent_id"]].drop_duplicates()
    for _, row in pairs.iterrows():
        G.add_edge(f"C_{row['captain_id']}", f"A_{row['agent_id']}")
    
    n_captains = df["captain_id"].nunique()
    n_agents = df["agent_id"].nunique()
    n_edges = len(pairs)
    max_edges = n_captains * n_agents
    
    diagnostics = {
        "n_nodes": len(G.nodes()),
        "n_edges": n_edges,
        "max_possible_edges": max_edges,
        "density": n_edges / max_edges if max_edges > 0 else 0,
        "avg_degree": 2 * n_edges / len(G.nodes()) if len(G.nodes()) > 0 else 0,
    }
    
    print(f"\nNetwork density:")
    print(f"  Nodes: {diagnostics['n_nodes']:,} ({n_captains:,} captains + {n_agents:,} agents)")
    print(f"  Edges: {diagnostics['n_edges']:,}")
    print(f"  Density: {diagnostics['density']:.4f}")
    print(f"  Average degree: {diagnostics['avg_degree']:.2f}")
    
    return diagnostics


def full_connected_set_analysis(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Run full connected set analysis pipeline.
    
    Parameters
    ----------
    df : pd.DataFrame
        Voyage data.
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, Dict]
        (standard_connected_df, loo_connected_df, all_diagnostics)
    """
    print("\n" + "=" * 60)
    print("FULL CONNECTED SET ANALYSIS")
    print("=" * 60)
    
    # Standard connected set
    df_cc, cc_diag = find_connected_set(df)
    
    # LOO connected set
    df_loo, loo_diag = find_leave_one_out_connected_set(df_cc)
    
    # Mobility diagnostics (on LOO set)
    mob_diag = compute_mobility_diagnostics(df_loo)
    
    # Network density
    net_diag = compute_network_density(df_loo)
    
    # Combine diagnostics
    all_diagnostics = {
        "connected_set": cc_diag,
        "loo_set": loo_diag,
        "mobility": mob_diag,
        "network": net_diag,
    }
    
    return df_cc, df_loo, all_diagnostics


def save_diagnostics_report(diagnostics: Dict, output_path: str) -> None:
    """
    Save connected set diagnostics to file.
    
    Parameters
    ----------
    diagnostics : Dict
        Diagnostics from full_connected_set_analysis.
    output_path : str
        Path to save report.
    """
    from pathlib import Path
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write("# Connected Set Diagnostics Report\n\n")
        
        f.write("## Connected Set\n")
        cc = diagnostics["connected_set"]
        f.write(f"- Components: {cc['n_components']}\n")
        f.write(f"- Captains in largest: {cc['largest_component_captains']:,} ({100*cc['coverage_captains']:.1f}%)\n")
        f.write(f"- Agents in largest: {cc['largest_component_agents']:,} ({100*cc['coverage_agents']:.1f}%)\n")
        f.write(f"- Voyages in largest: {cc['voyages_in_connected_set']:,} ({100*cc['coverage_voyages']:.1f}%)\n\n")
        
        f.write("## Leave-One-Out Set (KSS)\n")
        loo = diagnostics["loo_set"]
        f.write(f"- Iterations: {loo['iterations']}\n")
        f.write(f"- Voyages: {loo['loo_voyages']:,} ({100*loo['coverage']:.1f}%)\n")
        f.write(f"- Captains: {loo['loo_captains']:,}\n")
        f.write(f"- Agents: {loo['loo_agents']:,}\n")
        f.write(f"- Articulation edges: {loo['articulation_edges']:,} ({100*loo['articulation_rate']:.1f}%)\n\n")
        
        f.write("## Mobility Diagnostics\n")
        mob = diagnostics["mobility"]
        f.write(f"- Multi-agent captains: {mob['multi_agent_captains']:,} ({100*mob['multi_agent_captain_rate']:.1f}%)\n")
        f.write(f"- Multi-captain agents: {mob['multi_captain_agents']:,} ({100*mob['multi_captain_agent_rate']:.1f}%)\n")
        f.write(f"- Repeat partnerships: {mob['repeat_pairs']:,} ({100*mob['repeat_pair_rate']:.1f}%)\n\n")
        
        f.write("## Network Metrics\n")
        net = diagnostics["network"]
        f.write(f"- Density: {net['density']:.4f}\n")
        f.write(f"- Average degree: {net['avg_degree']:.2f}\n")
    
    print(f"Diagnostics saved to {output_path}")


if __name__ == "__main__":
    from .data_loader import prepare_analysis_sample
    
    # Test connected set analysis
    df = prepare_analysis_sample()
    df_cc, df_loo, diagnostics = full_connected_set_analysis(df)
    
    print(f"\nFinal LOO sample: {len(df_loo):,} voyages")
