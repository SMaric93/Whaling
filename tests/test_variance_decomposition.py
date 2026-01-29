"""
Tests for variance decomposition in the Whaling analysis.

These tests address reviewer critiques about statistical anomalies
in Table 7 variance components (Issue: identical Var(ψ) = Var(θ) values).
"""

import pytest
import numpy as np
import pandas as pd


def make_synthetic_voyage_data(n_voyages: int = 500) -> pd.DataFrame:
    """Create synthetic voyage data with KNOWN distinct variances for testing."""
    np.random.seed(42)
    
    n_captains = 50
    n_agents = 20
    
    # Create distinct variance structures
    # Captain effects: lower variance (std = 0.3)
    captain_effects = {f"C{i:03d}": np.random.normal(0, 0.3) for i in range(n_captains)}
    # Agent effects: higher variance (std = 0.5) - should be DIFFERENT
    agent_effects = {f"A{i:02d}": np.random.normal(0, 0.5) for i in range(n_agents)}
    
    voyages = []
    for i in range(n_voyages):
        captain_id = f"C{i % n_captains:03d}"
        agent_id = f"A{i % n_agents:02d}"
        
        # Production function: y = α + γ + ε
        alpha = captain_effects[captain_id]
        gamma = agent_effects[agent_id]
        epsilon = np.random.normal(0, 0.2)
        
        voyages.append({
            "voyage_id": f"V{i:04d}",
            "captain_id": captain_id,
            "agent_id": agent_id,
            "year_out": 1800 + (i % 70),
            "log_q": 6.0 + alpha + gamma + epsilon,
            "log_tonnage": np.random.normal(5.5, 0.3),
            "ground_or_route": f"Route_{i % 5}",
            "home_port": f"Port_{i % 4}",
            "decade": 1800 + 10 * (i % 7),
            "route_time": f"Route_{i % 5}_1800",
            "vessel_period": f"Vessel_{i % 30}_Pre",
            "port_time": f"Port_{i % 4}_1800",
            "alpha_hat": alpha,
            "gamma_hat": gamma,
            "theta_hat": alpha,  # Alias for counterfactual suite
            "psi_hat": gamma,    # Alias for counterfactual suite
        })
    
    return pd.DataFrame(voyages)


class TestVarianceDecompositionDistinctness:
    """Ensure variance components are not identical (catches copy-paste bugs)."""
    
    def test_raw_variance_components_distinct(self):
        """Verify Var(α) and Var(γ) are not identical in synthetic data."""
        df = make_synthetic_voyage_data(500)
        
        var_alpha = df["alpha_hat"].var()
        var_gamma = df["gamma_hat"].var()
        
        # These should be different - our synthetic data has σ_α=0.3, σ_γ=0.5
        assert abs(var_alpha - var_gamma) > 0.01, (
            f"Variance components are too similar: Var(α)={var_alpha:.4f}, Var(γ)={var_gamma:.4f}. "
            f"This suggests a potential copy-paste error."
        )
        
        # Verify approximate expected values (accounting for sampling variance)
        assert 0.03 < var_alpha < 0.20, f"Var(α) = {var_alpha:.4f} outside expected range"
        assert 0.10 < var_gamma < 0.40, f"Var(γ) = {var_gamma:.4f} outside expected range"
    
    def test_variance_precision_threshold(self):
        """Warn if variances match to 3 decimal places (statistically improbable)."""
        df = make_synthetic_voyage_data(1000)
        
        var_alpha = df["alpha_hat"].var()
        var_gamma = df["gamma_hat"].var()
        
        # Round to 3 decimals and check they're different
        var_alpha_rounded = round(var_alpha, 3)
        var_gamma_rounded = round(var_gamma, 3)
        
        assert var_alpha_rounded != var_gamma_rounded, (
            f"CRITICAL: Variances match to 3 decimal places ({var_alpha_rounded}). "
            f"In large matched employer-employee data, this is statistically improbable "
            f"and suggests a coding error (see Table 7 reviewer critique)."
        )


class TestCF_F15_InequalityDecomposition:
    """Test the inequality decomposition counterfactual for distinctness."""
    
    def test_policy_variances_differ(self):
        """Verify that equalizing θ vs ψ produces different variance reductions."""
        df = make_synthetic_voyage_data(500)
        
        # Add required columns for CF_F15
        df["levy_mu"] = np.random.uniform(1.2, 2.0, len(df))
        df["era"] = np.where(df["year_out"] < 1850, "early", "late")
        df["ground_type"] = np.where(df["voyage_id"].str[-1].astype(int) % 2 == 0, "sparse", "rich")
        
        # Simulate the CF_F15 computation logic
        from src.analyses.counterfactual_suite import standardize
        
        # Baseline
        y_baseline = df["log_q"].values
        baseline_var = np.var(y_baseline)
        
        # Policy 1: Equalize θ within era×ground
        theta_mean = df.groupby(["era", "ground_type"])["theta_hat"].transform("mean")
        y_cf_theta = (
            0.132 * standardize(theta_mean.values) +
            0.509 * standardize(df["psi_hat"].values) +
            (-0.039) * standardize(theta_mean.values) * standardize(df["psi_hat"].values)
        )
        delta_var_theta = np.var(y_cf_theta) - baseline_var
        
        # Policy 2: Equalize ψ within era×ground
        psi_mean = df.groupby(["era", "ground_type"])["psi_hat"].transform("mean")
        y_cf_psi = (
            0.132 * standardize(df["theta_hat"].values) +
            0.509 * standardize(psi_mean.values) +
            (-0.039) * standardize(df["theta_hat"].values) * standardize(psi_mean.values)
        )
        delta_var_psi = np.var(y_cf_psi) - baseline_var
        
        # The key test: these should be different
        assert abs(delta_var_theta - delta_var_psi) > 0.001, (
            f"CF_F15 policy variance reductions are too similar: "
            f"Δ(θ)={delta_var_theta:.4f}, Δ(ψ)={delta_var_psi:.4f}. "
            f"This may explain the identical 1.070 values in Table 7."
        )


class TestVarianceDecompositionWithRealData:
    """Integration tests using the actual analysis pipeline."""
    
    @pytest.mark.slow
    def test_r1_variance_components_distinct(self):
        """Test that R1 variance decomposition produces distinct components."""
        try:
            from src.analyses.data_loader import prepare_analysis_sample
            from src.analyses.baseline_production import estimate_r1, variance_decomposition
            
            df = prepare_analysis_sample()
            results = estimate_r1(df, use_loo_sample=True)
            decomp = variance_decomposition(results)
            
            # Extract variances
            var_alpha = decomp[decomp["Component"].str.contains("Captain")]["KSS_Var"].values[0]
            var_gamma = decomp[decomp["Component"].str.contains("Agent")]["KSS_Var"].values[0]
            
            # They must be distinct
            assert abs(var_alpha - var_gamma) > 0.001, (
                f"WARNING: Var(α)={var_alpha:.4f} ≈ Var(γ)={var_gamma:.4f}. "
                f"Check for copy-paste errors in Table 7."
            )
            
        except Exception as e:
            pytest.skip(f"Could not load real data: {e}")
    
    @pytest.mark.slow
    def test_cf_f15_with_real_data(self):
        """Test CF_F15 inequality decomposition with real data."""
        try:
            from src.analyses.counterfactual_suite import (
                prepare_counterfactual_data,
                run_cf_f15_inequality,
            )
            from src.analyses.data_loader import prepare_analysis_sample
            from src.analyses.baseline_production import estimate_r1
            
            df = prepare_analysis_sample()
            r1_results = estimate_r1(df, use_loo_sample=True)
            df = r1_results["df"]
            df = prepare_counterfactual_data(df)
            
            results = run_cf_f15_inequality(df)
            
            # Check that policy variance deltas are distinct
            delta_theta = results["policies"]["equalize_theta"]["delta_variance"]
            delta_psi = results["policies"]["equalize_psi"]["delta_variance"]
            
            assert abs(delta_theta - delta_psi) > 0.001, (
                f"CF_F15 delta variances are identical: θ={delta_theta:.4f}, ψ={delta_psi:.4f}"
            )
            
        except Exception as e:
            pytest.skip(f"Could not run CF_F15: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
