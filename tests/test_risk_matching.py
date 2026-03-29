"""
Tests for Risk Matching Theory analysis module.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure src is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def make_synthetic_voyage_data(n_voyages: int = 500) -> pd.DataFrame:
    """Create synthetic voyage data with known structure for testing."""
    np.random.seed(42)
    
    n_captains = 50
    n_agents = 20
    
    voyages = []
    for i in range(n_voyages):
        captain_id = f"C{i % n_captains:03d}"
        agent_id = f"A{i % n_agents:02d}"
        
        # High-variance captains have more volatile outputs
        captain_variance = (i % n_captains) / n_captains
        
        voyages.append({
            "voyage_id": f"V{i:04d}",
            "captain_id": captain_id,
            "agent_id": agent_id,
            "year_out": 1800 + (i % 70),
            "log_q": np.random.normal(6 + captain_variance, 0.3 + 0.5 * captain_variance),
            "log_tonnage": np.random.normal(5.5, 0.3),
            "ground_or_route": f"Route_{i % 5}",
            "home_port": f"Port_{i % 4}",
            "decade": 1800 + 10 * (i % 7),
            "route_time": f"Route_{i % 5}_1800",
            "vessel_period": f"Vessel_{i % 30}_Pre",
            "port_time": f"Port_{i % 4}_1800",
        })
    
    return pd.DataFrame(voyages)


class TestCaptainVarianceDecomposition:
    """Tests for RM1: Captain Variance Decomposition."""
    
    def test_with_precomputed_residuals(self):
        """Test variance computation with pre-computed residuals."""
        df = make_synthetic_voyage_data(500)
        
        # Add mock FE estimates
        df["alpha_hat"] = np.random.normal(0, 0.5, len(df))
        df["gamma_hat"] = np.random.normal(0, 0.3, len(df))
        df["residuals"] = np.random.normal(0, 0.2, len(df))
        
        from src.analyses.risk_matching import compute_captain_variance_decomposition
        
        captain_stats, diag = compute_captain_variance_decomposition(df, min_voyages=3)
        
        # Verify output structure
        assert "sigma_sq_alpha" in captain_stats.columns
        assert "alpha_hat" in captain_stats.columns
        assert "is_variance_creator" in captain_stats.columns
        
        # Verify diagnostics
        assert diag["n_captains"] > 0
        assert "mean_sigma_sq" in diag
        assert "corr_skill_variance" in diag

    def test_known_within_captain_variance(self):
        df = pd.DataFrame({
            "voyage_id": [f"V{i:02d}" for i in range(9)],
            "captain_id": ["C1"] * 3 + ["C2"] * 3 + ["C3"] * 3,
            "agent_id": ["A1"] * 9,
            "alpha_hat": [1.0] * 3 + [2.0] * 3 + [3.0] * 3,
            "residuals": [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, -1.0, 1.0, 3.0],
            "log_q": [10.0] * 9,
        })

        from src.analyses.risk_matching import compute_captain_variance_decomposition

        captain_stats, _ = compute_captain_variance_decomposition(df, min_voyages=3)
        stats = captain_stats.set_index("captain_id")

        assert stats.loc["C1", "sigma_sq_alpha"] == pytest.approx(1.0)
        assert np.isfinite(stats["sigma_alpha_std"]).all()


class TestAgentPortfolioBreadth:
    """Tests for RM2: Agent Portfolio Breadth."""
    
    def test_portfolio_computation(self):
        """Test portfolio breadth calculation."""
        df = make_synthetic_voyage_data(500)
        
        from src.analyses.risk_matching import compute_agent_portfolio_breadth
        
        agent_stats, diag = compute_agent_portfolio_breadth(df, min_voyages=3)
        
        # Verify output structure
        assert "portfolio_breadth" in agent_stats.columns
        assert "is_variance_absorber" in agent_stats.columns
        
        # Verify diagnostics
        assert diag["n_agents"] > 0
        assert "mean_portfolio_breadth" in diag

    def test_known_portfolio_breadth(self):
        df = pd.DataFrame({
            "voyage_id": [f"V{i:02d}" for i in range(6)],
            "agent_id": ["A1", "A1", "A1", "A2", "A2", "A2"],
            "captain_id": ["C1", "C2", "C3", "C1", "C1", "C2"],
            "ground_or_route": ["R1", "R1", "R2", "R3", "R3", "R3"],
            "home_port": ["P1", "P2", "P2", "P1", "P1", "P2"],
            "log_q": [1, 2, 3, 4, 5, 6],
        })

        from src.analyses.risk_matching import compute_agent_portfolio_breadth

        agent_stats, _ = compute_agent_portfolio_breadth(df, min_voyages=1)
        stats = agent_stats.set_index("agent_id")

        assert stats.loc["A1", "n_routes"] == 2
        assert stats.loc["A1", "n_ports"] == 2
        assert stats.loc["A1", "portfolio_breadth"] == 4


class TestRiskSortingRegression:
    """Tests for RM3: Risk Sorting Regression."""
    
    def test_regression_runs(self):
        """Test that regression executes without errors."""
        df = make_synthetic_voyage_data(500)
        
        # Add mock FE estimates
        df["alpha_hat"] = np.random.normal(0, 0.5, len(df))
        df["gamma_hat"] = np.random.normal(0, 0.3, len(df))
        df["residuals"] = np.random.normal(0, 0.2, len(df))
        
        from src.analyses.risk_matching import (
            compute_captain_variance_decomposition,
            compute_agent_portfolio_breadth,
            run_risk_sorting_regression,
        )
        
        captain_stats, _ = compute_captain_variance_decomposition(df, min_voyages=3)
        agent_stats, _ = compute_agent_portfolio_breadth(df, min_voyages=3)
        
        results = run_risk_sorting_regression(df, captain_stats, agent_stats)
        
        # Verify key outputs
        assert "b_risk_sorting" in results
        assert "p_risk" in results
        assert "corr_portfolio_variance" in results


class TestSortingComparison:
    """Tests for RM4: Sorting Correlation Comparison."""
    
    def test_comparison_table(self):
        """Test comparison table generation."""
        df = make_synthetic_voyage_data(500)
        
        # Add mock FE estimates
        df["alpha_hat"] = np.random.normal(0, 0.5, len(df))
        df["gamma_hat"] = np.random.normal(0, 0.3, len(df))
        df["residuals"] = np.random.normal(0, 0.2, len(df))
        
        from src.analyses.risk_matching import (
            compute_captain_variance_decomposition,
            compute_agent_portfolio_breadth,
            run_risk_sorting_regression,
            compare_sorting_correlations,
        )
        
        captain_stats, _ = compute_captain_variance_decomposition(df, min_voyages=3)
        agent_stats, _ = compute_agent_portfolio_breadth(df, min_voyages=3)
        rm3_results = run_risk_sorting_regression(df, captain_stats, agent_stats)
        
        comparison_df = compare_sorting_correlations(
            df, captain_stats, agent_stats, rm3_results
        )
        
        # Verify comparison table structure
        assert "Type" in comparison_df.columns
        assert "Correlation" in comparison_df.columns
        assert len(comparison_df) >= 2  # At least MEAN and RISK sorting


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
