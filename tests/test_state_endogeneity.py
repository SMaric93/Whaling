"""
Tests for State Endogeneity (Ex-Ante Ground Classification).

Addresses reviewer critique: "If ground classification is based on realized
catch, the result is circular. State s must be defined ex-ante."

These tests verify that ex-ante (lagged) ground classification produces
consistent results and avoids endogeneity.
"""

import pytest
import numpy as np
import pandas as pd


def make_synthetic_ground_data(n_voyages: int = 800) -> pd.DataFrame:
    """Create synthetic data with ground-level variation over time."""
    np.random.seed(42)
    
    grounds = ["Pacific", "Atlantic", "Brazil", "Indian", "Arctic", "Japan"]
    years = list(range(1820, 1870))  # 50 years
    
    # Ground-level persistent productivity (some grounds are better)
    ground_base_productivity = {
        "Pacific": 5.5,
        "Atlantic": 6.5,
        "Brazil": 6.2,
        "Indian": 5.8,
        "Arctic": 5.2,
        "Japan": 5.0,
    }
    
    voyages = []
    # Generate multiple voyages per ground-year to ensure lagged merge works
    for year in years:
        for ground in grounds:
            # 2-4 voyages per ground-year
            n_voyages_this = np.random.randint(2, 5)
            for j in range(n_voyages_this):
                base = ground_base_productivity[ground]
                time_trend = 0.01 * (year - 1820)
                shock = np.random.normal(0, 0.5)
                log_q = base + time_trend + shock
                
                i = len(voyages)
                voyages.append({
                    "voyage_id": f"V{i:04d}",
                    "captain_id": f"C{i % 50:03d}",
                    "agent_id": f"A{i % 20:02d}",
                    "year_out": year,
                    "log_q": log_q,
                    "route_or_ground": ground,
                    "log_tonnage": np.random.normal(5.5, 0.3),
                })
    
    return pd.DataFrame(voyages)


class TestExAnteClassificationMethods:
    """Test the ex-ante classification methods."""
    
    def test_lagged_year_classification(self):
        """Test lagged-year classification produces sensible results."""
        from src.analyses.counterfactual_suite import classify_ground_ex_ante
        
        df = make_synthetic_ground_data()
        df_classified = classify_ground_ex_ante(df, method="lagged_year")
        
        # Should have ground_type_ex_ante column
        assert "ground_type_ex_ante" in df_classified.columns
        
        # Should have sparse, rich, and possibly unknown
        types = df_classified["ground_type_ex_ante"].value_counts()
        assert len(types) >= 2, "Classification should produce at least sparse and rich"
        
        # First year should have "unknown" (no lag data)
        first_year = df_classified["year_out"].min()
        first_year_types = df_classified[df_classified["year_out"] == first_year]["ground_type_ex_ante"]
        assert (first_year_types == "unknown").all() or len(first_year_types[first_year_types != "unknown"]) < len(first_year_types) / 2
    
    def test_decadal_average_classification(self):
        """Test decadal-average classification produces sensible results."""
        from src.analyses.counterfactual_suite import classify_ground_ex_ante
        
        df = make_synthetic_ground_data()
        df_classified = classify_ground_ex_ante(df, method="decadal_average")
        
        assert "ground_type_ex_ante" in df_classified.columns
        
        # First decade should have more unknowns
        first_decade = (df_classified["year_out"].min() // 10) * 10
        first_decade_unknowns = (df_classified[df_classified["year_out"] // 10 * 10 == first_decade]["ground_type_ex_ante"] == "unknown").sum()
        assert first_decade_unknowns > 0, "First decade should have some unknowns"
    
    def test_name_based_vs_lagged_consistency(self):
        """Ex-ante and name-based should be correlated but not identical."""
        from src.analyses.counterfactual_suite import classify_ground_ex_ante
        
        df = make_synthetic_ground_data()
        df1 = classify_ground_ex_ante(df.copy(), method="name_based")
        df2 = classify_ground_ex_ante(df.copy(), method="lagged_year")
        
        # Should not be identical
        agreement = (df1["ground_type_ex_ante"] == df2["ground_type_ex_ante"]).mean()
        
        # Some disagreement is expected (lagged method is data-driven)
        assert 0.3 < agreement < 0.95, (
            f"Classification methods too similar ({agreement:.2%}) or too different"
        )


class TestStateEndogeneityPlacebo:
    """Placebo tests to verify ex-ante classification avoids endogeneity."""
    
    def test_lagged_catch_predicts_current_classification(self):
        """Lagged catch should predict current state but not perfectly."""
        from src.analyses.counterfactual_suite import classify_ground_ex_ante
        
        df = make_synthetic_ground_data()
        df = classify_ground_ex_ante(df, method="lagged_year")
        
        # For non-unknown observations, lagged catch should predict type
        df_valid = df[df["ground_type_ex_ante"] != "unknown"].copy()
        if len(df_valid) == 0:
            pytest.skip("No valid ex-ante classifications")
        
        # Current catch should correlate with type
        sparse_catch = df_valid[df_valid["ground_type_ex_ante"] == "sparse"]["log_q"].mean()
        rich_catch = df_valid[df_valid["ground_type_ex_ante"] == "rich"]["log_q"].mean()
        
        # Rich grounds should have higher contemporaneous catch (on average)
        assert rich_catch > sparse_catch, (
            f"Ex-ante classification not predictive: sparse={sparse_catch:.3f}, rich={rich_catch:.3f}"
        )
    
    def test_no_perfect_contemporaneous_correlation(self):
        """Current-year catch should NOT perfectly determine classification."""
        from src.analyses.counterfactual_suite import classify_ground_ex_ante
        
        df = make_synthetic_ground_data()
        df = classify_ground_ex_ante(df, method="lagged_year")
        
        df_valid = df[df["ground_type_ex_ante"] != "unknown"].copy()
        if len(df_valid) == 0:
            pytest.skip("No valid ex-ante classifications")
        
        # Within-type variance should be substantial
        within_sparse_var = df_valid[df_valid["ground_type_ex_ante"] == "sparse"]["log_q"].var()
        within_rich_var = df_valid[df_valid["ground_type_ex_ante"] == "rich"]["log_q"].var()
        
        # Should have variance within each type
        assert within_sparse_var > 0.01, "No variance within sparse - classification too deterministic"
        assert within_rich_var > 0.01, "No variance within rich - classification too deterministic"
    
    def test_results_robust_to_classification_method(self):
        """CF results should be qualitatively similar across classification methods."""
        # This is a higher-level test that would run actual counterfactuals
        # For now, just verify the methods produce reasonable outputs
        from src.analyses.counterfactual_suite import classify_ground_ex_ante
        
        df = make_synthetic_ground_data()
        
        methods = ["name_based", "lagged_year", "decadal_average"]
        sparse_counts = {}
        
        for method in methods:
            df_m = classify_ground_ex_ante(df.copy(), method=method)
            sparse_counts[method] = (df_m["ground_type_ex_ante"] == "sparse").sum()
        
        # All methods should produce some sparse classifications
        for method, count in sparse_counts.items():
            assert count > 0, f"Method {method} produced no sparse classifications"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
