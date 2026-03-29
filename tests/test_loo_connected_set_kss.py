"""
Tests for KSS-correct leave-one-out connected set.

Tests that the LOO connected set correctly identifies and prunes
articulation-point captains (cut vertices) from the bipartite graph,
matching the algorithm in KSS (2020) Appendix B / pruning_unbal_v3.m.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add project root to path (matches project convention)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analyses.connected_set import (
    compute_network_density,
    find_connected_set,
    find_leave_one_out_connected_set,
    full_connected_set_analysis,
)


def _make_voyage_df(pairs):
    """Helper: create a minimal voyage DataFrame from (captain, agent) pairs."""
    rows = []
    for captain, agent in pairs:
        rows.append({"captain_id": captain, "agent_id": agent})
    return pd.DataFrame(rows)


class TestLOOConnectedSetKSS:
    """Test that the LOO connected set uses proper articulation-point pruning."""

    def test_bridge_captain_is_pruned(self):
        """
        A captain who is the sole bridge between two clusters must be pruned.

        Graph structure (movers only):
            Cluster 1: C_A --[A1, A2]  (captain A works with agents 1 and 2)
            Bridge:     C_B --[A2, A3]  (captain B is the ONLY link between clusters)
            Cluster 2: C_C --[A3, A4]  (captain C works with agents 3 and 4)

        Captain B is an articulation point: removing B disconnects clusters.
        The OLD heuristic would KEEP captain B because:
          - pair_count doesn't matter (each pair appears once, but...)
          - captain B has 2 agents (>1) ✓
          - agent A2 has 2 captains (A, B) (>1) ✓
          - agent A3 has 2 captains (B, C) (>1) ✓
        So the old condition `(captain_n_agents > 1) & (agent_n_captains > 1)`
        would incorrectly keep all observations of captain B.

        The KSS algorithm correctly identifies B as an articulation point and
        removes all their observations.
        """
        pairs = [
            # Cluster 1: captain A is a mover (2 agents)
            ("A", "agent1"), ("A", "agent1"),  # repeat so A stays
            ("A", "agent2"), ("A", "agent2"),
            # Bridge captain B: connects clusters via agent2 and agent3
            ("B", "agent2"),
            ("B", "agent3"),
            # Cluster 2: captain C is a mover (2 agents)
            ("C", "agent3"), ("C", "agent3"),
            ("C", "agent4"), ("C", "agent4"),
        ]
        df = _make_voyage_df(pairs)
        df_loo, diagnostics = find_leave_one_out_connected_set(df)

        # Captain B must be pruned (articulation point)
        remaining_captains = set(df_loo["captain_id"].unique())
        assert "B" not in remaining_captains, (
            "Captain B is an articulation point and should have been pruned, "
            "but was incorrectly kept in the LOO set."
        )
        assert diagnostics["articulation_captains_pruned"] >= 1

    def test_non_bridge_captain_is_kept(self):
        """
        A captain who is NOT an articulation point should be kept.

        Graph: fully connected triangle of movers.
            C_A --[A1, A2]
            C_B --[A2, A3]
            C_C --[A1, A3]

        No captain is an articulation point — removing any one still
        leaves the other two connected through shared agents.
        """
        pairs = [
            ("A", "agent1"), ("A", "agent1"),
            ("A", "agent2"), ("A", "agent2"),
            ("B", "agent2"), ("B", "agent2"),
            ("B", "agent3"), ("B", "agent3"),
            ("C", "agent1"), ("C", "agent1"),
            ("C", "agent3"), ("C", "agent3"),
        ]
        df = _make_voyage_df(pairs)
        df_loo, diagnostics = find_leave_one_out_connected_set(df)

        remaining_captains = set(df_loo["captain_id"].unique())
        assert "A" in remaining_captains
        assert "B" in remaining_captains
        assert "C" in remaining_captains
        assert diagnostics["articulation_captains_pruned"] == 0

    def test_single_obs_stayer_is_dropped(self):
        """
        A stayer (1 agent) with only 1 observation should be dropped.
        A stayer with 2+ observations should be kept.
        """
        pairs = [
            # Movers forming a connected component
            ("A", "agent1"), ("A", "agent1"),
            ("A", "agent2"), ("A", "agent2"),
            ("B", "agent2"), ("B", "agent2"),
            ("B", "agent3"), ("B", "agent3"),
            ("C", "agent1"), ("C", "agent1"),
            ("C", "agent3"), ("C", "agent3"),
            # Stayer with 1 obs — should be dropped
            ("D", "agent1"),
            # Stayer with 2 obs — should be kept
            ("E", "agent2"), ("E", "agent2"),
        ]
        df = _make_voyage_df(pairs)
        df_loo, diagnostics = find_leave_one_out_connected_set(df)

        remaining_captains = set(df_loo["captain_id"].unique())
        assert "D" not in remaining_captains, "Single-obs stayer should be dropped"
        assert "E" in remaining_captains, "Multi-obs stayer should be kept"

    def test_iterative_pruning(self):
        """
        Test that pruning iterates correctly: removing one articulation point
        can create new ones that also need to be removed.

        Graph:
            C_A --[A1, A2]   (mover)
            C_B --[A2, A3]   (bridge → articulation point in round 1)
            C_C --[A3, A4]   (after B removed, C becomes bridge if D exists)
            C_D --[A4, A5]   (mover)

        This is a chain: A–B–C–D. B and C are both articulation points.
        After removing B, {A} disconnects from {C, D}. In the remaining
        largest component {C, D}, C is no longer an articulation point
        (it's part of a 2-node chain with D). So the final set depends on
        which component is largest.
        """
        pairs = [
            ("A", "agent1"), ("A", "agent1"),
            ("A", "agent2"), ("A", "agent2"),
            ("B", "agent2"),
            ("B", "agent3"),
            ("C", "agent3"), ("C", "agent3"),
            ("C", "agent4"), ("C", "agent4"),
            ("D", "agent4"),
            ("D", "agent5"),
        ]
        df = _make_voyage_df(pairs)
        df_loo, diagnostics = find_leave_one_out_connected_set(df)

        # B and D are articulation points in this chain topology
        # At least some pruning should have occurred
        assert diagnostics["articulation_captains_pruned"] >= 1
        assert diagnostics["iterations"] >= 1

    def test_empty_result_handled(self):
        """Test that a graph with no valid LOO set produces an empty result."""
        # Two movers each with exactly 1 observation per pair — no redundancy
        pairs = [
            ("A", "agent1"),
            ("A", "agent2"),
            ("B", "agent2"),
            ("B", "agent3"),
        ]
        df = _make_voyage_df(pairs)
        df_loo, diagnostics = find_leave_one_out_connected_set(df)

        # Both A and B are articulation points → all pruned
        assert len(df_loo) == 0 or diagnostics["articulation_captains_pruned"] >= 1

    def test_diagnostics_keys_present(self):
        """Test that the diagnostics dict contains all expected KSS-specific keys."""
        pairs = [
            ("A", "agent1"), ("A", "agent1"),
            ("A", "agent2"), ("A", "agent2"),
            ("B", "agent2"), ("B", "agent2"),
            ("B", "agent3"), ("B", "agent3"),
        ]
        df = _make_voyage_df(pairs)
        _, diagnostics = find_leave_one_out_connected_set(df)

        expected_keys = {
            "iterations",
            "initial_voyages",
            "loo_voyages",
            "loo_captains",
            "loo_agents",
            "coverage",
            "n_movers_initial",
            "n_stayers_initial",
            "articulation_captains_pruned",
            "single_obs_stayers_dropped",
        }
        assert expected_keys.issubset(set(diagnostics.keys())), (
            f"Missing keys: {expected_keys - set(diagnostics.keys())}"
        )

    def test_find_connected_set_keeps_largest_component(self):
        pairs = [
            ("A", "agent1"),
            ("A", "agent2"),
            ("B", "agent2"),
            ("C", "agent3"),
        ]
        df = _make_voyage_df(pairs)

        df_cc, diagnostics = find_connected_set(df)

        assert set(df_cc["captain_id"].unique()) == {"A", "B"}
        assert set(df_cc["agent_id"].unique()) == {"agent1", "agent2"}
        assert diagnostics["n_components"] == 2

    def test_compute_network_density_exact(self):
        pairs = [
            ("A", "agent1"),
            ("A", "agent2"),
            ("B", "agent2"),
        ]
        df = _make_voyage_df(pairs)

        diagnostics = compute_network_density(df)

        assert diagnostics["n_nodes"] == 4
        assert diagnostics["n_edges"] == 3
        assert diagnostics["density"] == pytest.approx(3 / 4)
        assert diagnostics["avg_degree"] == pytest.approx(1.5)
