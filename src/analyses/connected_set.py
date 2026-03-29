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


def _build_bipartite_graph(pairs: pd.DataFrame) -> nx.Graph:
    """Build a captain-agent bipartite graph from unique pairs."""
    graph = nx.Graph()
    if pairs.empty:
        return graph

    captain_nodes = "C_" + pairs["captain_id"].astype(str)
    agent_nodes = "A_" + pairs["agent_id"].astype(str)
    graph.add_edges_from(zip(captain_nodes.to_numpy(), agent_nodes.to_numpy()))
    return graph


def _largest_component_ids(graph: nx.Graph) -> Tuple[Set[str], Set[str]]:
    """Return captain and agent IDs from the largest connected component."""
    if graph.number_of_nodes() == 0:
        return set(), set()

    largest_cc = max(nx.connected_components(graph), key=len)
    connected_captains = {n[2:] for n in largest_cc if n.startswith("C_")}
    connected_agents = {n[2:] for n in largest_cc if n.startswith("A_")}
    return connected_captains, connected_agents


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

    total_captains = df["captain_id"].nunique()
    total_agents = df["agent_id"].nunique()
    total_voyages = len(df)

    pairs = df[["captain_id", "agent_id"]].drop_duplicates()
    G = _build_bipartite_graph(pairs)

    # Find connected components
    components = list(nx.connected_components(G))
    connected_captains, connected_agents = _largest_component_ids(G)
    
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
        "total_captains": total_captains,
        "total_agents": total_agents,
        "voyages_in_connected_set": len(df_cc),
        "total_voyages": total_voyages,
        "coverage_captains": len(connected_captains) / total_captains if total_captains else 0,
        "coverage_agents": len(connected_agents) / total_agents if total_agents else 0,
        "coverage_voyages": len(df_cc) / total_voyages if total_voyages else 0,
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
    
    Implements the algorithm from KSS (2020) Appendix B, matching the
    reference MATLAB code ``pruning_unbal_v3.m`` from the LeaveOutTwoWay
    package (Saggio, 2020).
    
    The LOO connected set is defined as the largest subset of observations
    such that removing the **entire history of any single captain (worker)**
    does not disconnect the bipartite graph. This is a vertex-connectivity
    condition on the movers-only subgraph.
    
    Algorithm
    ---------
    1. Identify movers (captains observed with 2+ distinct agents).
    2. Build the bipartite graph of unique (mover, agent) pairs.
    3. Find articulation points (cut vertices) via biconnected component
       decomposition.
    4. Identify captains that are articulation points — their removal
       would disconnect the graph.
    5. Drop **all observations** of those captains.
    6. Re-find the largest connected component.
    7. Iterate until no articulation-point captains remain.
    8. Drop stayers (single-agent captains) with only 1 observation
       (not identified in the LOO sense).
    
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
    
    # Initial mobility summary
    captain_agent_counts = df_loo.groupby("captain_id")["agent_id"].nunique()
    n_movers_initial = (captain_agent_counts >= 2).sum()
    n_stayers_initial = (captain_agent_counts == 1).sum()
    
    print(f"Initial sample: {initial_n:,} voyages")
    print(f"  Movers (2+ agents): {n_movers_initial:,} captains")
    print(f"  Stayers (1 agent):  {n_stayers_initial:,} captains")
    
    # Iteratively prune articulation-point captains until stable
    n_bad_captains = 1  # sentinel to enter loop
    iteration = 0
    total_pruned = 0
    
    while n_bad_captains >= 1:
        iteration += 1
        
        # Step 1: Identify movers (captains with 2+ distinct agents)
        captain_agents = df_loo.groupby("captain_id")["agent_id"].nunique()
        mover_captains = set(captain_agents[captain_agents >= 2].index)
        df_movers = df_loo[df_loo["captain_id"].isin(mover_captains)]
        
        if len(df_movers) == 0:
            print(f"  Iteration {iteration}: No movers remain, stopping.")
            break
        
        # Step 2: Build bipartite graph of unique (mover, agent) pairs
        # Nodes: captain IDs (prefixed C_) and agent IDs (prefixed A_)
        unique_pairs = df_movers[["captain_id", "agent_id"]].drop_duplicates()
        G = _build_bipartite_graph(unique_pairs)
        
        # Step 3: Find articulation points (cut vertices)
        artic_points = set(nx.articulation_points(G))
        
        # Step 4: Identify captains that are articulation points
        bad_captains = {node[2:] for node in artic_points if node.startswith("C_")}
        n_bad_captains = len(bad_captains)
        
        print(f"  Iteration {iteration}: {n_bad_captains} articulation-point captains found")
        
        if n_bad_captains == 0:
            break
        
        total_pruned += n_bad_captains
        
        # Step 5: Drop ALL observations of bad captains
        df_loo = df_loo[~df_loo["captain_id"].isin(bad_captains)].copy()
        
        # Step 6: Re-find largest connected component
        if len(df_loo) > 0:
            pairs_remaining = df_loo[["captain_id", "agent_id"]].drop_duplicates()
            G_remaining = _build_bipartite_graph(pairs_remaining)

            if G_remaining.number_of_nodes() > 0:
                # Reset IDs and find largest connected component
                connected_captains, connected_agents = _largest_component_ids(
                    G_remaining
                )
                df_loo = df_loo[
                    df_loo["captain_id"].isin(connected_captains) & 
                    df_loo["agent_id"].isin(connected_agents)
                ].copy()
    
    # Step 8: Drop stayers with only 1 observation (not identified)
    obs_per_captain = df_loo.groupby("captain_id").size()
    captain_agents_final = df_loo.groupby("captain_id")["agent_id"].nunique()
    stayers = set(captain_agents_final[captain_agents_final == 1].index)
    single_obs_stayers = set(obs_per_captain[obs_per_captain <= 1].index) & stayers
    
    if single_obs_stayers:
        n_dropped_stayers = len(single_obs_stayers)
        df_loo = df_loo[~df_loo["captain_id"].isin(single_obs_stayers)].copy()
        print(f"  Dropped {n_dropped_stayers} single-observation stayers")
    
    # Diagnostics
    diagnostics = {
        "iterations": iteration,
        "initial_voyages": initial_n,
        "loo_voyages": len(df_loo),
        "loo_captains": df_loo["captain_id"].nunique(),
        "loo_agents": df_loo["agent_id"].nunique(),
        "coverage": len(df_loo) / initial_n if initial_n > 0 else 0,
        "n_movers_initial": int(n_movers_initial),
        "n_stayers_initial": int(n_stayers_initial),
        "articulation_captains_pruned": total_pruned,
        "single_obs_stayers_dropped": len(single_obs_stayers) if single_obs_stayers else 0,
    }
    
    print(f"\nLOO pruning complete (KSS articulation-point algorithm):")
    print(f"  Iterations: {diagnostics['iterations']}")
    print(f"  Articulation-point captains removed: {diagnostics['articulation_captains_pruned']}")
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
        "multi_agent_captain_rate": multi_agent_captains / df["captain_id"].nunique() if df["captain_id"].nunique() else 0,
        "multi_captain_agents": multi_captain_agents,
        "multi_captain_agent_rate": multi_captain_agents / df["agent_id"].nunique() if df["agent_id"].nunique() else 0,
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
    pairs = df[["captain_id", "agent_id"]].drop_duplicates()
    n_captains = df["captain_id"].nunique()
    n_agents = df["agent_id"].nunique()
    n_edges = len(pairs)
    n_nodes = n_captains + n_agents
    max_edges = n_captains * n_agents
    
    diagnostics = {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "max_possible_edges": max_edges,
        "density": n_edges / max_edges if max_edges > 0 else 0,
        "avg_degree": 2 * n_edges / n_nodes if n_nodes > 0 else 0,
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
        
        f.write("## Leave-One-Out Set (KSS Articulation-Point Algorithm)\n")
        loo = diagnostics["loo_set"]
        f.write(f"- Iterations: {loo['iterations']}\n")
        f.write(f"- Voyages: {loo['loo_voyages']:,} ({100*loo['coverage']:.1f}%)\n")
        f.write(f"- Captains: {loo['loo_captains']:,}\n")
        f.write(f"- Agents: {loo['loo_agents']:,}\n")
        f.write(f"- Articulation-point captains pruned: {loo.get('articulation_captains_pruned', 'N/A')}\n")
        f.write(f"- Single-obs stayers dropped: {loo.get('single_obs_stayers_dropped', 'N/A')}\n\n")
        
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
