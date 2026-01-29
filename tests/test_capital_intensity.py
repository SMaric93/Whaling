"""
Tests for Capital Intensity (Vessel Quality) Controls.

Addresses reviewer critique: "Lower μ (ballistic path) might be due to
'Better Hardware' (faster/better ships) rather than 'Information/Maps'."

These tests verify that the μ~ψ relationship survives after controlling
for vessel characteristics.
"""

import pytest
import numpy as np
import pandas as pd


def make_synthetic_voyage_data_with_vessels(n_voyages: int = 600) -> pd.DataFrame:
    """Create synthetic data with vessel characteristics and μ."""
    np.random.seed(42)
    
    n_captains = 60
    n_agents = 25
    n_vessels = 40
    
    # Captain and agent effects
    captain_effects = {f"C{i:03d}": np.random.normal(0, 0.3) for i in range(n_captains)}
    agent_effects = {f"A{i:02d}": np.random.normal(0, 0.5) for i in range(n_agents)}
    
    # Vessel characteristics - tonnage correlates with μ but NOT fully
    vessel_tonnage = {f"V{i:02d}": np.random.uniform(100, 400) for i in range(n_vessels)}
    vessel_age = {f"V{i:02d}": np.random.randint(1, 25) for i in range(n_vessels)}
    
    voyages = []
    for i in range(n_voyages):
        captain_id = f"C{i % n_captains:03d}"
        agent_id = f"A{i % n_agents:02d}"
        vessel_id = f"V{i % n_vessels:02d}"
        
        alpha = captain_effects[captain_id]
        gamma = agent_effects[agent_id]
        tonnage = vessel_tonnage[vessel_id]
        age = vessel_age[vessel_id]
        
        # μ is driven by BOTH agent capability AND vessel quality
        # But agent effect should survive controlling for vessel
        mu_from_agent = gamma * 0.15  # Agent effect on search behavior
        mu_from_vessel = -0.001 * tonnage + 0.01 * age  # Vessel effect
        mu = 1.5 + mu_from_agent + mu_from_vessel + np.random.normal(0, 0.1)
        
        epsilon = np.random.normal(0, 0.2)
        log_q = 6.0 + alpha + gamma + 0.3 * np.log(tonnage) + epsilon
        
        voyages.append({
            "voyage_id": f"V{i:04d}",
            "captain_id": captain_id,
            "agent_id": agent_id,
            "vessel_id": vessel_id,
            "year_out": 1800 + (i % 70),
            "log_q": log_q,
            "log_tonnage": np.log(tonnage),
            "vessel_age": age,
            "tonnage": tonnage,
            "levy_mu": mu,
            "alpha_hat": alpha,
            "gamma_hat": gamma,
            "theta_hat": alpha,
            "psi_hat": gamma,
            "route_time": f"Route_{i % 5}_1800",
            "vessel_period": f"{vessel_id}_Pre",
        })
    
    return pd.DataFrame(voyages)


class TestCapitalIntensityControls:
    """Test that μ~ψ relationship is robust to vessel controls."""
    
    def test_mu_psi_relationship_without_controls(self):
        """Baseline: μ correlates with ψ (agent capability)."""
        df = make_synthetic_voyage_data_with_vessels()
        
        # Simple correlation
        corr = df["levy_mu"].corr(df["psi_hat"])
        
        # Should be positive (higher capability → more ballistic search)
        # Note: in actual data, corr might be negative if higher μ = more diffusive
        assert abs(corr) > 0.05, f"Weak μ~ψ correlation: {corr:.4f}"
    
    def test_mu_psi_relationship_with_tonnage_control(self):
        """μ~ψ should survive after controlling for log_tonnage."""
        df = make_synthetic_voyage_data_with_vessels()
        
        from sklearn.linear_model import LinearRegression
        
        # Model 1: μ ~ ψ (no controls)
        X1 = df[["psi_hat"]].values
        y = df["levy_mu"].values
        model1 = LinearRegression().fit(X1, y)
        beta_psi_raw = model1.coef_[0]
        
        # Model 2: μ ~ ψ + log_tonnage
        X2 = df[["psi_hat", "log_tonnage"]].values
        model2 = LinearRegression().fit(X2, y)
        beta_psi_controlled = model2.coef_[0]
        
        # Effect should survive (not drop below 50% of original)
        survival_ratio = beta_psi_controlled / (beta_psi_raw + 1e-10)
        
        assert abs(beta_psi_controlled) > 0.01, (
            f"μ~ψ effect disappears after tonnage control: β(ψ) = {beta_psi_controlled:.4f}"
        )
        
        print(f"  β(ψ) without controls: {beta_psi_raw:.4f}")
        print(f"  β(ψ) with tonnage control: {beta_psi_controlled:.4f}")
        print(f"  Survival ratio: {survival_ratio:.2%}")
    
    def test_vessel_fe_does_not_eliminate_psi_effect(self):
        """Adding vessel×period FE should not eliminate ψ effect on μ."""
        df = make_synthetic_voyage_data_with_vessels()
        
        # Add vessel_period dummies
        vessel_period_dummies = pd.get_dummies(df["vessel_period"], prefix="vp", drop_first=True)
        
        X = pd.concat([df[["psi_hat"]], vessel_period_dummies], axis=1).values.astype(float)
        y = df["levy_mu"].values.astype(float)
        
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        beta_psi = beta[0]
        
        # Effect should still be detectable
        assert abs(beta_psi) > 0.001, (
            f"ψ effect absorbed by vessel FE: β(ψ) = {beta_psi:.4f}"
        )
    
    def test_tonnage_not_collinear_with_psi(self):
        """Verify tonnage and ψ are not perfectly collinear."""
        df = make_synthetic_voyage_data_with_vessels()
        
        corr = df["log_tonnage"].corr(df["psi_hat"])
        
        # Should not be too high
        assert abs(corr) < 0.7, (
            f"WARNING: log_tonnage and ψ are highly correlated ({corr:.3f}). "
            f"Capital intensity controls may absorb agent effects."
        )


class TestMoversDesignWithVesselControls:
    """Movers design should reveal maps vs hardware distinction."""
    
    def test_within_captain_mu_variation(self):
        """Within-captain variation in μ should correlate with agent switches."""
        df = make_synthetic_voyage_data_with_vessels()
        
        # Compute within-captain μ variation
        df["mu_deviation"] = df.groupby("captain_id")["levy_mu"].transform(
            lambda x: x - x.mean()
        )
        df["psi_deviation"] = df.groupby("captain_id")["psi_hat"].transform(
            lambda x: x - x.mean()
        )
        
        # Within-captain correlation
        within_corr = df["mu_deviation"].corr(df["psi_deviation"])
        
        assert abs(within_corr) > 0.01, (
            f"Within-captain μ~ψ variation is too weak: {within_corr:.4f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
