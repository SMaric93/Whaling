"""Test 12: Hardware, Staffing, and Incentives Horse Race."""

from __future__ import annotations
import logging, numpy as np, pandas as pd
from src.next_round.config import OUTPUTS_TABLES, OUTPUTS_FIGURES, PSI_COL, THETA_COL, FIGURE_DPI, FIGURE_FORMAT

logger = logging.getLogger(__name__)

HARDWARE_CONTROLS = ["tonnage", "crew_count", "mean_age", "desertion_rate"]
NEGATIVE_CONTROLS = ["speed", "move_length"]  # transit outcomes where routines shouldn't matter

def run_hardware_staffing_placebos(*, save_outputs: bool = True) -> dict:
    logger.info("=" * 60); logger.info("Test 12: Hardware/Staffing Placebos"); logger.info("=" * 60)
    from src.reinforcement.data_builder import build_analysis_panel

    df = build_analysis_panel(require_akm=True)
    outcome = "q_total_index" if "q_total_index" in df.columns else "log_q"
    results = {}

    # 1. Main regression without controls
    try:
        import statsmodels.formula.api as smf
        base_formula = f"{outcome} ~ {PSI_COL} + {THETA_COL}"
        m_base = smf.ols(base_formula, data=df.dropna(subset=[PSI_COL, THETA_COL, outcome])).fit(
            cov_type="cluster", cov_kwds={"groups": df.dropna(subset=[PSI_COL, THETA_COL, outcome])["captain_id"]})
        results["baseline"] = {"psi_coef": m_base.params.get(PSI_COL, np.nan),
                               "psi_se": m_base.bse.get(PSI_COL, np.nan), "r2": m_base.rsquared, "n": int(m_base.nobs)}

        # 2. Add hardware/staffing controls
        available_hw = [c for c in HARDWARE_CONTROLS if c in df.columns]
        if available_hw:
            hw_formula = f"{outcome} ~ {PSI_COL} + {THETA_COL} + {' + '.join(available_hw)}"
            m_hw = smf.ols(hw_formula, data=df.dropna(subset=[PSI_COL, THETA_COL, outcome] + available_hw)).fit(
                cov_type="cluster", cov_kwds={"groups": df.dropna(subset=[PSI_COL, THETA_COL, outcome] + available_hw)["captain_id"]})
            results["with_controls"] = {"psi_coef": m_hw.params.get(PSI_COL, np.nan),
                                         "psi_se": m_hw.bse.get(PSI_COL, np.nan), "r2": m_hw.rsquared, "n": int(m_hw.nobs),
                                         "controls": available_hw}
    except ImportError:
        results["error"] = "no_statsmodels"

    if save_outputs:
        rows = [{"spec": k, **v} for k,v in results.items() if isinstance(v, dict)]
        if rows: pd.DataFrame(rows).to_csv(OUTPUTS_TABLES / "hardware_staffing_placebos.csv", index=False)

        try:
            import matplotlib.pyplot as plt
            specs = []; coefs = []; ses = []
            for s in ["baseline", "with_controls"]:
                if s in results and "psi_coef" in results[s]:
                    specs.append(s); coefs.append(results[s]["psi_coef"]); ses.append(results[s].get("psi_se", 0))
            if specs:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(range(len(specs)), coefs, yerr=ses, color=["#2196F3", "#FF9800"], capsize=5)
                ax.set_xticks(range(len(specs))); ax.set_xticklabels(specs)
                ax.set_ylabel("ψ Coefficient"); ax.set_title("ψ With and Without Hardware/Staffing Controls")
                ax.axhline(0, color="black", lw=0.5); ax.grid(axis="y", alpha=0.3); fig.tight_layout()
                fig.savefig(OUTPUTS_FIGURES / f"hardware_staffing_placebos.{FIGURE_FORMAT}", dpi=FIGURE_DPI, bbox_inches="tight")
                plt.close(fig)
        except ImportError: pass

        (OUTPUTS_TABLES / "hardware_staffing_placebos_memo.md").write_text(
            "# Test 12: Hardware/Staffing — Memo\n\n## Identifies\nWhether stopping-rule and transition results survive adding capital/staffing controls.\n\n## Does NOT identify\n- Unobserved crew quality may still confound\n")
    return results
