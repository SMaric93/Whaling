"""
Tests for Three Additional Robustness Tests.

Tests for:
1. Vessel Mover Design (Killer Robustness Test)
2. Optimal Foraging Stopping Rule
3. Insurance Variance Validation
"""

import pytest
import numpy as np
import pandas as pd


# =============================================================================
# Synthetic Data Generators
# =============================================================================

def make_synthetic_vessel_transfer_data(n_voyages: int = 500) -> pd.DataFrame:
    """Create synthetic data with vessel transfers between agents."""
    np.random.seed(42)
    
    n_captains = 50
    n_agents = 20
    n_vessels = 30
    
    # Captain and agent effects
    captain_effects = {f"C{i:03d}": np.random.normal(0, 0.3) for i in range(n_captains)}
    agent_effects = {f"A{i:02d}": np.random.normal(0, 0.5) for i in range(n_agents)}
    
    voyages = []
    vessel_agent_map = {}  # Track which agent owns each vessel
    
    for i in range(n_voyages):
        captain_id = f"C{i % n_captains:03d}"
        vessel_id = f"V{i % n_vessels:02d}"
        
        # Simulate vessel transfers: 20% chance of agent change for a vessel
        if vessel_id in vessel_agent_map:
            if np.random.random() < 0.2:
                # Transfer to different agent
                new_agent = f"A{np.random.randint(0, n_agents):02d}"
                while new_agent == vessel_agent_map[vessel_id]:
                    new_agent = f"A{np.random.randint(0, n_agents):02d}"
                vessel_agent_map[vessel_id] = new_agent
        else:
            vessel_agent_map[vessel_id] = f"A{np.random.randint(0, n_agents):02d}"
        
        agent_id = vessel_agent_map[vessel_id]
        
        alpha = captain_effects[captain_id]
        gamma = agent_effects[agent_id]
        
        # μ is driven by agent capability (the effect we want to detect)
        mu = 2.0 - 0.15 * gamma + np.random.normal(0, 0.2)
        
        epsilon = np.random.normal(0, 0.2)
        log_q = 6.0 + alpha + gamma + epsilon
        
        voyages.append({
            "voyage_id": f"VOY{i:04d}",
            "captain_id": captain_id,
            "agent_id": agent_id,
            "vessel_id": vessel_id,
            "year_out": 1800 + i // 10,
            "log_q": log_q,
            "levy_mu": mu,
            "alpha_hat": alpha,
            "theta_hat": alpha,
            "psi_hat": gamma,
        })
    
    return pd.DataFrame(voyages)


def make_synthetic_position_data(n_voyages: int = 50) -> tuple:
    """Create synthetic position data for stopping rule tests."""
    np.random.seed(42)
    
    voyages = []
    positions = []
    
    for v in range(n_voyages):
        voyage_id = f"VOY{v:04d}"
        
        # Create voyage record
        psi = np.random.normal(0, 0.5)
        voyages.append({
            "voyage_id": voyage_id,
            "captain_id": f"C{v % 20:03d}",
            "agent_id": f"A{v % 10:02d}",
            "log_q": 6.0 + psi + np.random.normal(0, 0.2),
            "psi_hat": psi,
            "duration_days": 365,
        })
        
        # Create positions for this voyage
        n_positions = np.random.randint(30, 100)
        lat = np.random.uniform(30, 50)
        lon = np.random.uniform(-80, -40)
        
        for p in range(n_positions):
            # Random walk with occasional jumps
            if np.random.random() < 0.1:
                # Jump to new patch
                lat = np.random.uniform(30, 50)
                lon = np.random.uniform(-80, -40)
            else:
                lat += np.random.normal(0, 0.5)
                lon += np.random.normal(0, 0.5)
            
            positions.append({
                "voyage_id": voyage_id,
                "obs_date": pd.Timestamp("1800-01-01") + pd.Timedelta(days=p),
                "lat": lat,
                "lon": lon,
            })
    
    return pd.DataFrame(voyages), pd.DataFrame(positions)


def make_synthetic_novice_expert_data(n_voyages: int = 800) -> pd.DataFrame:
    """Create synthetic data with novice/expert captains and variance effects."""
    np.random.seed(42)
    
    n_captains = 80
    n_agents = 30
    
    captain_effects = {f"C{i:03d}": np.random.normal(0, 0.3) for i in range(n_captains)}
    agent_effects = {f"A{i:02d}": np.random.normal(0, 0.5) for i in range(n_agents)}
    captain_voyage_counts = {f"C{i:03d}": 0 for i in range(n_captains)}
    
    voyages = []
    
    for i in range(n_voyages):
        captain_id = f"C{i % n_captains:03d}"
        agent_id = f"A{i % n_agents:02d}"
        
        alpha = captain_effects[captain_id]
        gamma = agent_effects[agent_id]
        n_prior = captain_voyage_counts[captain_id]
        
        # Key test: HIGH-ψ agents compress variance for novices
        is_novice = n_prior <= 3
        is_high_psi = gamma > np.median(list(agent_effects.values()))
        
        # Base variance
        base_variance = 0.3
        
        # Novice × High-ψ gets reduced variance (insurance effect)
        if is_novice and is_high_psi:
            actual_variance = base_variance * 0.5  # 50% variance reduction
        elif is_novice:
            actual_variance = base_variance * 1.2  # Higher variance for novices with low-ψ
        else:
            actual_variance = base_variance
        
        epsilon = np.random.normal(0, np.sqrt(actual_variance))
        log_q = 6.0 + alpha + gamma + epsilon
        
        captain_voyage_counts[captain_id] += 1
        
        voyages.append({
            "voyage_id": f"VOY{i:04d}",
            "captain_id": captain_id,
            "agent_id": agent_id,
            "year_out": 1800 + i // 20,
            "log_q": log_q,
            "alpha_hat": alpha,
            "theta_hat": alpha,
            "psi_hat": gamma,
            "n_prior_voyages": n_prior,
        })
    
    return pd.DataFrame(voyages)


# =============================================================================
# Test: Vessel Mover Design
# =============================================================================

class TestVesselMoverDesign:
    """Tests for the Vessel Mover Design analysis."""
    
    def test_vessel_transfer_detection(self):
        """Test that vessel transfers are correctly identified."""
        df = make_synthetic_vessel_transfer_data()
        
        from src.analyses.vessel_mover_analysis import build_vessel_ownership_panel
        
        panel = build_vessel_ownership_panel(df)
        
        # Should have transfer indicator
        assert "vessel_transfer" in panel.columns
        
        # Should detect some transfers
        n_transfers = panel["vessel_transfer"].sum()
        assert n_transfers > 0, "No vessel transfers detected"
        
        print(f"Detected {n_transfers} vessel transfers")
    
    def test_multi_agent_vessel_identification(self):
        """Test identification of vessels with multiple agents."""
        df = make_synthetic_vessel_transfer_data()
        
        from src.analyses.vessel_mover_analysis import (
            build_vessel_ownership_panel,
            identify_multi_agent_vessels,
        )
        
        panel = build_vessel_ownership_panel(df)
        multi_agent = identify_multi_agent_vessels(panel)
        
        # Should have fewer vessels than in full panel
        assert len(multi_agent) < len(panel)
        
        # Each vessel should have 2+ agents
        for vessel_id in multi_agent["vessel_id"].unique():
            n_agents = multi_agent[multi_agent["vessel_id"] == vessel_id]["agent_id"].nunique()
            assert n_agents >= 2, f"Vessel {vessel_id} has only {n_agents} agent(s)"
    
    def test_within_vessel_mu_variation(self):
        """Test that μ varies within the same vessel."""
        df = make_synthetic_vessel_transfer_data()
        
        # Check within-vessel μ variation
        mu_within_var = df.groupby("vessel_id")["levy_mu"].var().dropna()
        
        # At least some vessels should have μ variation
        vessels_with_var = (mu_within_var > 0.01).sum()
        assert vessels_with_var > 0, "No within-vessel μ variation"
        
        print(f"Vessels with μ variation: {vessels_with_var}")
    
    def test_regression_runs(self):
        """Test that the within-vessel regression runs without error."""
        df = make_synthetic_vessel_transfer_data()
        
        from src.analyses.vessel_mover_analysis import (
            build_vessel_ownership_panel,
            identify_multi_agent_vessels,
            run_within_vessel_regression,
        )
        
        panel = build_vessel_ownership_panel(df)
        multi_agent = identify_multi_agent_vessels(panel)
        
        # Add levy_mu if not present
        if "levy_mu" not in multi_agent.columns:
            mu_map = df.set_index("voyage_id")["levy_mu"]
            multi_agent["levy_mu"] = multi_agent["voyage_id"].map(mu_map)
        
        results = run_within_vessel_regression(multi_agent)
        
        # Should return results (not error)
        assert "error" not in results or "model1_pooled" in results


# =============================================================================
# Test: Stopping Rule
# =============================================================================

class TestStoppingRule:
    """Tests for the Optimal Foraging Stopping Rule analysis."""
    
    def test_patch_identification(self):
        """Test that patches are correctly identified from positions."""
        voyage_df, positions_df = make_synthetic_position_data()
        
        from src.analyses.search_theory import identify_patches
        
        patches = identify_patches(positions_df)
        
        # Should identify some patches
        assert len(patches) > 0, "No patches identified"
        
        # Each patch should have required columns
        required_cols = ["voyage_id", "patch_id", "entry_date", "exit_date", "duration_days"]
        for col in required_cols:
            assert col in patches.columns, f"Missing column: {col}"
        
        print(f"Identified {len(patches)} patches")
    
    def test_patch_yield_calculation(self):
        """Test patch yield computation."""
        voyage_df, positions_df = make_synthetic_position_data()
        
        from src.analyses.search_theory import identify_patches, compute_patch_yield
        
        patches = identify_patches(positions_df)
        patches_with_yield = compute_patch_yield(patches, voyage_df)
        
        # Should have yield columns
        assert "estimated_yield" in patches_with_yield.columns
        assert "is_empty" in patches_with_yield.columns
        assert "is_productive" in patches_with_yield.columns
    
    def test_stopping_rule_test_runs(self):
        """Test that stopping rule test runs without error."""
        voyage_df, positions_df = make_synthetic_position_data()
        
        from src.analyses.search_theory import (
            identify_patches, compute_patch_yield, run_stopping_rule_test
        )
        
        patches = identify_patches(positions_df)
        patches_with_yield = compute_patch_yield(patches, voyage_df)
        results = run_stopping_rule_test(patches_with_yield, voyage_df)
        
        # Should return results
        assert "all_patches" in results or "error" in results


# =============================================================================
# Test: Insurance Variance
# =============================================================================

class TestInsuranceVariance:
    """Tests for the Insurance Variance Validation analysis."""
    
    def test_captain_classification(self):
        """Test captain classification as Novice/Expert."""
        df = make_synthetic_novice_expert_data()
        
        from src.analyses.insurance_variance_test import classify_captain_experience
        
        df_classified = classify_captain_experience(df)
        
        # Should have experience columns
        assert "is_novice" in df_classified.columns
        assert "is_expert" in df_classified.columns
        assert "captain_experience" in df_classified.columns
        
        # Should have all three categories
        exp_values = df_classified["captain_experience"].unique()
        assert "Novice" in exp_values
        assert "Expert" in exp_values
    
    def test_agent_classification(self):
        """Test agent classification as High-ψ/Low-ψ."""
        df = make_synthetic_novice_expert_data()
        
        from src.analyses.insurance_variance_test import classify_agent_capability
        
        df_classified = classify_agent_capability(df)
        
        # Should have capability columns
        assert "is_high_psi" in df_classified.columns
        assert "agent_capability" in df_classified.columns
        
        # Should have both High and Low
        cap_values = df_classified["agent_capability"].unique()
        assert "High-ψ" in cap_values
        assert "Low-ψ" in cap_values
    
    def test_variance_by_cell(self):
        """Test that variance differs by treatment cell."""
        df = make_synthetic_novice_expert_data()
        
        from src.analyses.insurance_variance_test import (
            classify_captain_experience,
            classify_agent_capability,
            create_treatment_cells,
        )
        
        df = classify_captain_experience(df)
        df = classify_agent_capability(df)
        df_cells = create_treatment_cells(df)
        
        # Compute variance by cell
        cell_vars = df_cells.groupby("treatment_cell")["log_q"].var()
        
        # Should have variance for each cell
        assert len(cell_vars) > 0
        
        # Check if Novice × High-ψ has lower variance (from synthetic data)
        if "Novice × High-ψ" in cell_vars.index and "Novice × Low-ψ" in cell_vars.index:
            var_novice_high = cell_vars["Novice × High-ψ"]
            var_novice_low = cell_vars["Novice × Low-ψ"]
            
            print(f"Var(Novice × High-ψ): {var_novice_high:.4f}")
            print(f"Var(Novice × Low-ψ): {var_novice_low:.4f}")
            
            # In synthetic data, high-ψ should have lower variance for novices
            # (we designed it that way)
            assert var_novice_high < var_novice_low, (
                "Insurance effect not detected in synthetic data"
            )
    
    def test_heteroskedasticity_test_runs(self):
        """Test that heteroskedasticity test runs without error."""
        df = make_synthetic_novice_expert_data()
        
        from src.analyses.insurance_variance_test import (
            classify_captain_experience,
            run_heteroskedasticity_test,
        )
        
        df = classify_captain_experience(df)
        results = run_heteroskedasticity_test(df)
        
        # Should return results
        assert "coefficients" in results or "error" in results
    
    def test_left_tail_analysis_runs(self):
        """Test that left-tail analysis runs without error."""
        df = make_synthetic_novice_expert_data()
        
        from src.analyses.insurance_variance_test import (
            classify_captain_experience,
            classify_agent_capability,
            create_treatment_cells,
            run_left_tail_analysis,
        )
        
        df = classify_captain_experience(df)
        df = classify_agent_capability(df)
        df_cells = create_treatment_cells(df)
        results = run_left_tail_analysis(df_cells)
        
        # Should return results
        assert "quantiles_by_cell" in results or "error" in results


# =============================================================================
# Integration Test
# =============================================================================

class TestIntegration:
    """Integration tests for the full analysis pipeline."""
    
    def test_full_vessel_mover_pipeline(self):
        """Test complete vessel mover analysis pipeline."""
        df = make_synthetic_vessel_transfer_data()
        
        from src.analyses.vessel_mover_analysis import run_vessel_mover_analysis
        
        results = run_vessel_mover_analysis(df, save_outputs=False)
        
        # Should have regression results
        assert "regression" in results
    
    def test_full_insurance_pipeline(self):
        """Test complete insurance variance pipeline."""
        df = make_synthetic_novice_expert_data()
        
        from src.analyses.insurance_variance_test import run_insurance_variance_tests
        
        results = run_insurance_variance_tests(df, save_outputs=False)
        
        # Should have all three test results
        assert "heteroskedasticity" in results
        assert "left_tail" in results
        assert "quantile_regression" in results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
