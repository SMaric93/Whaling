"""Test 9: Rational Exit Tests. Shows high-psi exit is conditional on continuation value, not generic aggressiveness."""

from __future__ import annotations
import logging, numpy as np, pandas as pd
from src.next_round.config import OUTPUTS_TABLES, OUTPUTS_FIGURES, PSI_COL, THETA_COL, RANDOM_SEED, FIGURE_DPI, FIGURE_FORMAT

logger = logging.getLogger(__name__)

def run_rational_exit_tests(*, save_outputs: bool = True) -> dict:
    logger.info("=" * 60); logger.info("Test 9: Rational Exit Tests"); logger.info("=" * 60)
    from src.ml.build_action_dataset import build_action_dataset
    from src.reinforcement.data_builder import build_analysis_panel

    actions = build_action_dataset(force_rebuild=False, save=False)
    voyages = build_analysis_panel(require_akm=True)
    voyage_cols = ["voyage_id", PSI_COL, THETA_COL, "captain_id"]
    voyage_cols = [c for c in voyage_cols if c in voyages.columns]
    voyage_info = voyages[voyage_cols].dropna(subset=[PSI_COL]).drop_duplicates("voyage_id")
    # Drop overlapping columns from actions
    overlap = [c for c in voyage_cols if c in actions.columns and c != "voyage_id"]
    df = actions.drop(columns=overlap, errors="ignore").merge(voyage_info, on="voyage_id", how="inner")

    if "exit_patch_next" not in df.columns:
        return {"error": "no_exit_column"}

    df["psi_q"] = pd.qcut(df[PSI_COL].rank(method="first"), 4, labels=["Q1","Q2","Q3","Q4"])
    results = {}

    # Interaction tests
    interactions = {"season_remaining": "season_remaining", "scarcity": "scarcity",
                    "consecutive_empty_days": "consecutive_empty_days"}
    for name, col in interactions.items():
        if col not in df.columns: continue
        df[f"psi_x_{name}"] = df[PSI_COL] * df[col]
        try:
            import statsmodels.formula.api as smf
            formula = f"exit_patch_next ~ {PSI_COL} + {col} + psi_x_{name}"
            m = smf.ols(formula, data=df.dropna(subset=[PSI_COL, col, "exit_patch_next"])).fit(
                cov_type="cluster", cov_kwds={"groups": df.dropna(subset=[PSI_COL, col, "exit_patch_next"])["captain_id"]})
            results[name] = {"interaction_coef": m.params.get(f"psi_x_{name}", np.nan),
                            "interaction_pval": m.pvalues.get(f"psi_x_{name}", np.nan),
                            "n_obs": int(m.nobs)}
        except Exception as e:
            results[name] = {"error": str(e)}

    # Placebo: transit segments
    if "active_search_flag" in df.columns:
        transit = df[df["active_search_flag"] == 0]
        if len(transit) > 100 and "exit_patch_next" in transit.columns:
            results["placebo_transit"] = {"exit_rate": transit["exit_patch_next"].mean(), "n": len(transit)}

    if save_outputs:
        rows = [{"test": k, **v} for k,v in results.items() if isinstance(v, dict)]
        if rows: pd.DataFrame(rows).to_csv(OUTPUTS_TABLES / "rational_exit_tests.csv", index=False)
        (OUTPUTS_TABLES / "rational_exit_tests_memo.md").write_text(
            "# Test 9: Rational Exit — Memo\n\n## Identifies\nHigh-ψ exit is conditional on signals, not generic.\n\n## Does NOT identify\n- Cannot separate from unobserved outside options\n")
    return results
