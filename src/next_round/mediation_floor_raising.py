"""Test 14: Mediation of Floor-Raising Through Search Governance."""

from __future__ import annotations
import logging, numpy as np, pandas as pd
from src.next_round.config import OUTPUTS_TABLES, OUTPUTS_FIGURES, PSI_COL, THETA_COL, FIGURE_DPI, FIGURE_FORMAT

logger = logging.getLogger(__name__)

def run_mediation_floor_raising(*, save_outputs: bool = True) -> dict:
    logger.info("=" * 60); logger.info("Test 14: Mediation Floor-Raising"); logger.info("=" * 60)
    from src.reinforcement.data_builder import build_analysis_panel

    df = build_analysis_panel(require_akm=True)
    outcome = "q_total_index" if "q_total_index" in df.columns else "log_q"
    y = df[outcome]
    df["bottom_decile"] = (y <= y.quantile(0.10)).astype(int)

    # Candidate mediators from voyage-level data
    mediators = {}
    for col in ["max_empty_streak", "days_without_catch", "n_grounds_visited", "duration_days"]:
        if col in df.columns:
            mediators[col] = col

    results = {}

    try:
        import statsmodels.formula.api as smf

        # Step 1: Total effect psi -> bottom_decile
        m_total = smf.ols(f"bottom_decile ~ {PSI_COL}", data=df.dropna(subset=[PSI_COL, "bottom_decile"])).fit(
            cov_type="cluster", cov_kwds={"groups": df.dropna(subset=[PSI_COL, "bottom_decile"])["captain_id"]})
        results["total_effect"] = {"psi_coef": m_total.params.get(PSI_COL, np.nan), "n": int(m_total.nobs)}

        # Step 2: For each mediator, sequential decomposition
        for med_name, med_col in mediators.items():
            df_med = df.dropna(subset=[PSI_COL, "bottom_decile", med_col])
            if len(df_med) < 100: continue

            # a path: psi -> mediator
            m_a = smf.ols(f"{med_col} ~ {PSI_COL}", data=df_med).fit()
            # b path: mediator -> outcome (controlling for psi)
            m_b = smf.ols(f"bottom_decile ~ {PSI_COL} + {med_col}", data=df_med).fit()

            a_coef = m_a.params.get(PSI_COL, 0)
            b_coef = m_b.params.get(med_col, 0)
            direct = m_b.params.get(PSI_COL, 0)
            indirect = a_coef * b_coef
            total = results["total_effect"]["psi_coef"]
            share = indirect / total if abs(total) > 1e-12 else np.nan

            results[med_name] = {"a_path": a_coef, "b_path": b_coef, "indirect": indirect,
                                  "direct": direct, "share_mediated": share, "n": len(df_med)}

    except ImportError:
        results["error"] = "no_statsmodels"

    if save_outputs:
        rows = [{"mediator": k, **v} for k,v in results.items() if isinstance(v, dict)]
        if rows: pd.DataFrame(rows).to_csv(OUTPUTS_TABLES / "mediation_floor_raising.csv", index=False)

        try:
            import matplotlib.pyplot as plt
            meds = []; shares = []
            for k,v in results.items():
                if isinstance(v, dict) and "share_mediated" in v and not np.isnan(v["share_mediated"]):
                    meds.append(k); shares.append(v["share_mediated"])
            if meds:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.barh(range(len(meds)), shares, color="#E91E63")
                ax.set_yticks(range(len(meds))); ax.set_yticklabels(meds)
                ax.set_xlabel("Share Mediated"); ax.set_title("Mediation of Downside-Risk Reduction")
                ax.grid(axis="x", alpha=0.3); fig.tight_layout()
                fig.savefig(OUTPUTS_FIGURES / f"mediation_floor_raising.{FIGURE_FORMAT}", dpi=FIGURE_DPI, bbox_inches="tight")
                plt.close(fig)
        except ImportError: pass

        (OUTPUTS_TABLES / "mediation_floor_raising_memo.md").write_text(
            "# Test 14: Mediation — Memo\n\n## Identifies\nDescriptive share of downside-risk reduction associated with search governance.\n\n## Does NOT identify\n- Causal mediation requires strong assumptions (sequential ignorability)\n- This is descriptive decomposition, not causal mediation\n")
    return results
