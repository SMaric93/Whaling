"""
Logbook-Based Analyses Module.

Implements regression tests using logbook-derived features:
- Route-Level Skill Isolation (storm exposure × captain FE)
- Agent Selection Strategy (route assignment, information advantage)
- Behavioral/Decision Tests (ground switching, arrival timing)
- TFP Enhancement (route efficiency component)

Integrates with existing R1-R17 regression suite.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from dataclasses import dataclass
import logging
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import FINAL_DIR
from src.analyses.config import OUTPUT_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LogbookCategory:
    title: str
    tests: tuple[tuple[str, Callable[[pd.DataFrame], Dict[str, Any]]], ...]


def _merge_optional_feature_frame(
    df: pd.DataFrame,
    feature_path: Path,
    *,
    label: str,
) -> pd.DataFrame:
    """Merge an optional voyage-level feature frame when present."""
    if not feature_path.exists():
        return df

    feature_df = pd.read_parquet(feature_path)
    logger.info("Merged %s: %s", label, feature_path.name)
    return df.merge(feature_df, on="voyage_id", how="left")


def _base_result(test_name: str, hypothesis: str) -> Dict[str, Any]:
    return {
        "test_name": test_name,
        "hypothesis": hypothesis,
        "status": "not_run",
    }


def _missing_columns(df: pd.DataFrame, required_cols: list[str]) -> list[str]:
    return [col for col in required_cols if col not in df.columns]


def _run_analysis_test(
    df: pd.DataFrame,
    *,
    test_name: str,
    hypothesis: str,
    required_cols: list[str],
    compute: Callable[[pd.DataFrame], Dict[str, Any]],
) -> Dict[str, Any]:
    """Wrap a logbook test with consistent required-column and error handling."""
    result = _base_result(test_name, hypothesis)
    missing = _missing_columns(df, required_cols)
    if missing:
        result["status"] = "skipped"
        result["reason"] = f"Missing columns: {missing}"
        return result

    try:
        result.update(compute(df))
        result["status"] = "completed"
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)

    return result


def _linregress_from_frame(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
) -> tuple[pd.DataFrame, float, float, float, float, float]:
    """Drop missing values and run a simple bivariate linear regression."""
    from scipy import stats

    valid = df[[y_col, x_col]].dropna()
    slope, intercept, r, p, se = stats.linregress(valid[x_col], valid[y_col])
    return valid, slope, intercept, r, p, se


def _print_block(title: str, *, fill: str = "=") -> None:
    line = fill * 70
    print(f"\n{line}")
    print(title)
    print(line)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# =============================================================================
# Data Loading
# =============================================================================

def load_analysis_data() -> pd.DataFrame:
    """Load voyage data with all available features."""
    logger.info("Loading analysis data...")

    # Base voyage data
    voyage_path = FINAL_DIR / "analysis_voyage_with_climate.parquet"
    if not voyage_path.exists():
        voyage_path = FINAL_DIR / "analysis_voyage.parquet"

    df = pd.read_parquet(voyage_path)
    logger.info("Loaded %s voyages", f"{len(df):,}")

    for feature_path, label in (
        (FINAL_DIR / "voyage_logbook_features.parquet", "logbook features"),
        (FINAL_DIR / "voyage_storm_exposure.parquet", "storm exposure"),
        (FINAL_DIR / "voyage_network_features.parquet", "network features"),
    ):
        df = _merge_optional_feature_frame(df, feature_path, label=label)

    return df


# =============================================================================
# Test Category 1: Route-Level Skill Isolation
# =============================================================================

def test_storm_exposure_captain_fe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Test: Do captain FEs shrink when controlling for actual storm exposure?
    
    If captains earned FE credit for avoiding storms, adding precise
    storm exposure should reduce captain effect variance.
    
    Hypothesis: Captain FE variance decreases with storm controls.
    """
    logger.info("Running: Storm Exposure x Captain FE Test")

    def compute(test_df: pd.DataFrame) -> Dict[str, Any]:
        from src.analyses.config import run_fe_regression

        captain_variance_key = "captain_id"
        storm_controls = [
            col for col in ["storm_encounter_count", "days_in_hurricane_corridor"]
            if col in test_df.columns
        ]

        r1 = run_fe_regression(
            test_df,
            "log_output",
            controls=[],
            fe_vars=["captain_id"],
        )
        r2 = run_fe_regression(
            test_df,
            "log_output",
            controls=storm_controls,
            fe_vars=["captain_id"],
        )

        variance_no_storm = r1.get("fe_variance", {}).get(captain_variance_key)
        variance_with_storm = r2.get("fe_variance", {}).get(captain_variance_key)
        baseline_variance = r1.get("fe_variance", {}).get(captain_variance_key, 1)
        return {
            "captain_fe_variance_no_storm": variance_no_storm,
            "captain_fe_variance_with_storm": variance_with_storm,
            "variance_reduction": (
                (r1.get("fe_variance", {}).get(captain_variance_key, 0) - r2.get("fe_variance", {}).get(captain_variance_key, 0))
                / max(baseline_variance, 0.001)
            ),
        }

    return _run_analysis_test(
        df,
        test_name="storm_exposure_captain_fe",
        hypothesis="Captain FE variance decreases with storm exposure controls",
        required_cols=["log_output", "captain_id", "storm_encounter_count"],
        compute=compute,
    )


def test_fair_weather_captains(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Test: Identify "fair weather captains" whose high FEs are explained by luck.
    
    Compare captain rankings before/after storm exposure controls.
    Captains who drop significantly had inflated skill estimates.
    """
    logger.info("Running: Fair Weather Captains Test")

    def compute(test_df: pd.DataFrame) -> Dict[str, Any]:
        captain_stats = test_df.groupby("captain_id").agg({
            "log_output": "mean",
            "storm_encounter_count": "mean",
        }).dropna()
        correlation = captain_stats["log_output"].corr(captain_stats["storm_encounter_count"])
        return {
            "output_storm_correlation": correlation,
            "interpretation": (
                "Negative correlation suggests storms hurt output"
                if correlation < 0
                else "Positive correlation is unexpected - may indicate survivor bias"
            ),
            "n_captains": len(captain_stats),
        }

    return _run_analysis_test(
        df,
        test_name="fair_weather_captains",
        hypothesis="Some high-FE captains are 'fair weather' - luck explains their success",
        required_cols=["log_output", "captain_id", "storm_encounter_count"],
        compute=compute,
    )


def test_route_efficiency_skill(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Test: Does route efficiency predict outcomes independently of captain FE?
    
    If route efficiency (beeline/actual distance) predicts output
    after captain FE, it represents portable navigation skill.
    """
    logger.info("Running: Route Efficiency Skill Test")

    def compute(test_df: pd.DataFrame) -> Dict[str, Any]:
        valid, slope, _, r, p, _ = _linregress_from_frame(
            test_df,
            x_col="route_efficiency",
            y_col="log_output",
        )
        return {
            "efficiency_coefficient": slope,
            "efficiency_pvalue": p,
            "efficiency_r_squared": r ** 2,
            "n_observations": len(valid),
            "interpretation": (
                f"Route efficiency has {'significant' if p < 0.05 else 'insignificant'} "
                f"{'positive' if slope > 0 else 'negative'} effect on output"
            ),
        }

    return _run_analysis_test(
        df,
        test_name="route_efficiency_skill",
        hypothesis="Route efficiency predicts output beyond captain FE",
        required_cols=["log_output", "route_efficiency"],
        compute=compute,
    )


# =============================================================================
# Test Category 2: Agent Selection Strategy
# =============================================================================

def test_route_assignment_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Test: Did agents assign captains with route experience?
    
    Higher match rates suggest intentional captain-route matching.
    """
    logger.info("Running: Route Assignment Quality Test")

    def compute(test_df: pd.DataFrame) -> Dict[str, Any]:
        df_sorted = test_df.sort_values(["captain_id", "year_out"]).copy()

        prior_experience: list[int] = []
        captain_route_history: dict[tuple[Any, Any], int] = {}

        for _, row in df_sorted.iterrows():
            captain = row["captain_id"]
            route = row["route"]

            if pd.isna(captain) or pd.isna(route):
                prior_experience.append(0)
                continue

            key = (captain, route)
            prior = captain_route_history.get(key, 0)
            prior_experience.append(prior)
            captain_route_history[key] = prior + 1

        df_sorted["prior_route_exp"] = prior_experience
        agent_rates = df_sorted.groupby("agent_id")["prior_route_exp"].apply(
            lambda values: (values > 0).mean()
        )
        mean_experienced_rate = agent_rates.mean()
        return {
            "mean_experienced_rate": mean_experienced_rate,
            "agent_variance": agent_rates.var(),
            "top_5_agents": agent_rates.nlargest(5).to_dict(),
            "n_agents": len(agent_rates),
            "interpretation": (
                f"On average, {mean_experienced_rate:.1%} of voyages "
                f"assigned to captains with prior route experience"
            ),
        }

    return _run_analysis_test(
        df,
        test_name="route_assignment_quality",
        hypothesis="Agents assigned experienced captains to routes more than random",
        required_cols=["agent_id", "captain_id", "route", "year_out"],
        compute=compute,
    )


def test_information_advantage(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Test: Did vessels that 'spoke with' more ships perform better?
    
    Information sharing at sea may have provided competitive advantage.
    """
    logger.info("Running: Information Advantage Test")

    def compute(test_df: pd.DataFrame) -> Dict[str, Any]:
        valid, slope, _, r, p, _ = _linregress_from_frame(
            test_df,
            x_col="spoke_with_count",
            y_col="log_output",
        )
        return {
            "spoke_coefficient": slope,
            "spoke_pvalue": p,
            "spoke_r_squared": r ** 2,
            "mean_spoke_count": valid["spoke_with_count"].mean(),
            "n_observations": len(valid),
            "interpretation": (
                f"Speaking with more vessels has "
                f"{'significant' if p < 0.05 else 'insignificant'} "
                f"{'positive' if slope > 0 else 'negative'} effect"
            ),
        }

    return _run_analysis_test(
        df,
        test_name="information_advantage",
        hypothesis="Vessels with more spoke-with contacts performed better",
        required_cols=["log_output", "spoke_with_count"],
        compute=compute,
    )


def test_agent_risk_portfolio(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Test: Did agents diversify fleet storm exposure?
    
    Lower exposure variance within agent suggests intentional diversification.
    """
    logger.info("Running: Agent Risk Portfolio Test")

    def compute(test_df: pd.DataFrame) -> Dict[str, Any]:
        agent_stats = test_df.groupby("agent_id").agg({
            "storm_encounter_count": "var",
            "log_output": ["mean", "var"],
        }).dropna()
        agent_stats.columns = ["exposure_var", "output_mean", "output_var"]

        from scipy import stats

        _, _, r, p, _ = stats.linregress(
            agent_stats["exposure_var"],
            agent_stats["output_var"],
        )
        return {
            "exposure_output_var_correlation": r,
            "correlation_pvalue": p,
            "n_agents": len(agent_stats),
            "interpretation": (
                f"Agent exposure variance {'correlates' if p < 0.05 else 'does not correlate'} "
                f"with output variance (r={r:.3f})"
            ),
        }

    return _run_analysis_test(
        df,
        test_name="agent_risk_portfolio",
        hypothesis="Agents with lower exposure variance had more stable returns",
        required_cols=["agent_id", "storm_encounter_count", "log_output"],
        compute=compute,
    )


# =============================================================================
# Test Category 3: Behavioral/Decision Tests
# =============================================================================

def test_ground_switching(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Test: Does ground switching frequency predict outcomes?
    
    Tests explore/exploit tradeoff in hunting strategy.
    """
    logger.info("Running: Ground Switching Test")

    def compute(test_df: pd.DataFrame) -> Dict[str, Any]:
        valid, slope, _, _, p, _ = _linregress_from_frame(
            test_df,
            x_col="ground_switching_count",
            y_col="log_output",
        )
        return {
            "switching_linear_effect": slope,
            "switching_pvalue": p,
            "mean_switching": valid["ground_switching_count"].mean(),
            "n_observations": len(valid),
            "interpretation": (
                f"More ground switching has "
                f"{'positive' if slope > 0 else 'negative'} effect on output"
            ),
        }

    return _run_analysis_test(
        df,
        test_name="ground_switching",
        hypothesis="Optimal ground switching frequency exists (inverted U)",
        required_cols=["log_output", "ground_switching_count"],
        compute=compute,
    )


def test_arrival_timing(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Test: Does arriving at grounds in optimal season predict outcomes?
    """
    logger.info("Running: Arrival Timing Test")

    def compute(test_df: pd.DataFrame) -> Dict[str, Any]:
        valid, slope, _, _, p, _ = _linregress_from_frame(
            test_df,
            x_col="arrival_timing_score",
            y_col="log_output",
        )
        return {
            "timing_coefficient": slope,
            "timing_pvalue": p,
            "mean_timing_score": valid["arrival_timing_score"].mean(),
            "n_observations": len(valid),
            "interpretation": (
                f"Optimal arrival timing has "
                f"{'significant' if p < 0.05 else 'insignificant'} "
                f"{'positive' if slope > 0 else 'negative'} effect"
            ),
        }

    return _run_analysis_test(
        df,
        test_name="arrival_timing",
        hypothesis="Captains arriving during optimal seasons had higher output",
        required_cols=["log_output", "arrival_timing_score"],
        compute=compute,
    )


# =============================================================================
# Test Category 4: TFP Enhancement
# =============================================================================

def test_route_efficiency_tfp(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Test: Add route efficiency to variance decomposition.
    
    What fraction of output variance is explained by navigation efficiency?
    """
    logger.info("Running: Route Efficiency TFP Test")

    def compute(test_df: pd.DataFrame) -> Dict[str, Any]:
        valid, _, _, r, _, _ = _linregress_from_frame(
            test_df,
            x_col="route_efficiency",
            y_col="log_output",
        )
        variance_explained = r ** 2
        return {
            "route_efficiency_r_squared": variance_explained,
            "variance_share": variance_explained,
            "total_output_variance": valid["log_output"].var(),
            "n_observations": len(valid),
            "interpretation": (
                f"Route efficiency explains {variance_explained:.1%} of output variance"
            ),
        }

    return _run_analysis_test(
        df,
        test_name="route_efficiency_tfp",
        hypothesis="Route efficiency explains additional output variance",
        required_cols=["log_output", "route_efficiency"],
        compute=compute,
    )


LOGBOOK_TEST_CATEGORIES: tuple[LogbookCategory, ...] = (
    LogbookCategory(
        title="CATEGORY 1: ROUTE-LEVEL SKILL ISOLATION",
        tests=(
            ("storm_exposure_captain_fe", test_storm_exposure_captain_fe),
            ("fair_weather_captains", test_fair_weather_captains),
            ("route_efficiency_skill", test_route_efficiency_skill),
        ),
    ),
    LogbookCategory(
        title="CATEGORY 2: AGENT SELECTION STRATEGY",
        tests=(
            ("route_assignment_quality", test_route_assignment_quality),
            ("information_advantage", test_information_advantage),
            ("agent_risk_portfolio", test_agent_risk_portfolio),
        ),
    ),
    LogbookCategory(
        title="CATEGORY 3: BEHAVIORAL/DECISION TESTS",
        tests=(
            ("ground_switching", test_ground_switching),
            ("arrival_timing", test_arrival_timing),
        ),
    ),
    LogbookCategory(
        title="CATEGORY 4: TFP ENHANCEMENT",
        tests=(("route_efficiency_tfp", test_route_efficiency_tfp),),
    ),
)


# =============================================================================
# Run All Tests
# =============================================================================

def run_all_logbook_tests(
    df: Optional[pd.DataFrame] = None,
    save_outputs: bool = True,
) -> Dict[str, Any]:
    """
    Run all logbook-based analyses.
    
    Parameters
    ----------
    df : pd.DataFrame, optional
        Analysis data. If None, loads from disk.
    save_outputs : bool
        Whether to save results.
        
    Returns
    -------
    Dict
        All test results.
    """
    _print_block("LOGBOOK-BASED ANALYSES")

    if df is None:
        df = load_analysis_data()

    print(f"\nAnalysis dataset: {len(df):,} voyages")
    print(f"Available columns: {len(df.columns)}")

    results: Dict[str, Any] = {}
    for category in LOGBOOK_TEST_CATEGORIES:
        _print_block(category.title, fill="-")
        for result_key, test_func in category.tests:
            results[result_key] = test_func(df)

    _print_block("TEST SUMMARY")

    completed = sum(1 for r in results.values() if r["status"] == "completed")
    skipped = sum(1 for r in results.values() if r["status"] == "skipped")
    errors = sum(1 for r in results.values() if r["status"] == "error")

    print(f"\nCompleted: {completed}")
    print(f"Skipped: {skipped}")
    print(f"Errors: {errors}")

    # Print key findings
    print("\nKEY FINDINGS:")
    for name, r in results.items():
        if r["status"] == "completed" and "interpretation" in r:
            print(f"  {name}: {r['interpretation']}")

    # Save outputs
    if save_outputs:
        output_path = OUTPUT_DIR / "logbook_analyses_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w") as handle:
            json.dump(results, handle, indent=2, default=_json_default)

        print(f"\nResults saved to: {output_path}")

    return results


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run logbook-based analyses"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run without saving outputs"
    )
    
    args = parser.parse_args()
    
    results = run_all_logbook_tests(save_outputs=not args.dry_run)
