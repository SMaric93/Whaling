"""Test 15: Archival Direct Mechanism Tests. Runs only if archival variables exist."""

from __future__ import annotations
import logging, numpy as np, pandas as pd
from src.next_round.config import OUTPUTS_TABLES, PSI_COL, THETA_COL

logger = logging.getLogger(__name__)

ARCHIVAL_COLS = ["agent_instructions", "correspondence", "officer_pipeline",
                 "repeated_crew", "lay_contract", "repair_outfit", "intelligence_notes"]

def run_archival_mechanisms(*, save_outputs: bool = True) -> dict:
    logger.info("=" * 60); logger.info("Test 15: Archival Mechanisms"); logger.info("=" * 60)
    from src.reinforcement.data_builder import build_analysis_panel

    df = build_analysis_panel(require_akm=True)
    available = [c for c in ARCHIVAL_COLS if c in df.columns]

    if not available:
        logger.info("No archival variables found in dataset. Skipping.")
        result = {"status": "skipped", "reason": "no_archival_data",
                  "checked_columns": ARCHIVAL_COLS}
        if save_outputs:
            pd.DataFrame([result]).to_csv(OUTPUTS_TABLES / "archival_mechanisms.csv", index=False)
            (OUTPUTS_TABLES / "archival_mechanisms_memo.md").write_text(
                "# Test 15: Archival Mechanisms — Memo\n\n## Status\nSkipped: no archival variables found.\n")
        return result

    results = {}
    for col in available:
        try:
            import statsmodels.formula.api as smf
            outcome = "q_total_index" if "q_total_index" in df.columns else "log_q"
            m = smf.ols(f"{outcome} ~ {col} + {PSI_COL}", data=df.dropna(subset=[col, PSI_COL, outcome])).fit()
            results[col] = {"coef": m.params.get(col, np.nan), "pval": m.pvalues.get(col, np.nan),
                            "n": int(m.nobs)}
        except Exception as e:
            results[col] = {"error": str(e)}

    if save_outputs:
        rows = [{"variable": k, **v} for k,v in results.items() if isinstance(v, dict)]
        if rows: pd.DataFrame(rows).to_csv(OUTPUTS_TABLES / "archival_mechanisms.csv", index=False)
        (OUTPUTS_TABLES / "archival_mechanisms_memo.md").write_text(
            "# Test 15: Archival Mechanisms — Memo\n\n## Identifies\nDirect measures of organizational governance → behavioral outcomes.\n")
    return results
