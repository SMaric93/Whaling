"""Test 11: Information vs Routine Timing. Separates info effects from persistent routines."""

from __future__ import annotations
import logging, numpy as np, pandas as pd
from src.next_round.config import OUTPUTS_TABLES, OUTPUTS_FIGURES, PSI_COL, THETA_COL, FIGURE_DPI, FIGURE_FORMAT

logger = logging.getLogger(__name__)

def run_info_vs_routine(*, save_outputs: bool = True) -> dict:
    logger.info("=" * 60); logger.info("Test 11: Info vs Routine"); logger.info("=" * 60)
    from src.ml.build_action_dataset import build_action_dataset
    from src.reinforcement.data_builder import build_analysis_panel

    actions = build_action_dataset(force_rebuild=False, save=False)
    voyages = build_analysis_panel(require_akm=True)
    df = actions.merge(voyages[["voyage_id", PSI_COL, THETA_COL, "captain_id"]].dropna(subset=[PSI_COL]),
                       on="voyage_id", how="inner")

    results = {}

    # Split by voyage stage (early / mid / late)
    if "duration_day" in df.columns and "exit_patch_next" in df.columns:
        max_day = df.groupby("voyage_id")["duration_day"].transform("max")
        df["voyage_pct"] = df["duration_day"] / max_day.clip(1)
        df["stage"] = pd.cut(df["voyage_pct"], bins=[0, 0.33, 0.67, 1.0], labels=["early", "mid", "late"])

        for stage in ["early", "mid", "late"]:
            sub = df[df["stage"] == stage]
            if len(sub) < 100: continue
            try:
                import statsmodels.formula.api as smf
                m = smf.ols(f"exit_patch_next ~ {PSI_COL}", data=sub.dropna(subset=[PSI_COL, "exit_patch_next"])).fit()
                results[f"psi_effect_{stage}"] = {"coef": m.params.get(PSI_COL, np.nan),
                                                   "pval": m.pvalues.get(PSI_COL, np.nan), "n": int(m.nobs)}
            except Exception as e:
                results[f"psi_effect_{stage}"] = {"error": str(e)}

    # After spoken-vessel encounters
    if "encounter" in df.columns:
        for enc_val, label in [(1, "after_encounter"), (0, "no_encounter")]:
            sub = df[df["encounter"] == enc_val]
            if len(sub) < 50: continue
            try:
                import statsmodels.formula.api as smf
                m = smf.ols(f"exit_patch_next ~ {PSI_COL}", data=sub.dropna(subset=[PSI_COL, "exit_patch_next"])).fit()
                results[f"psi_{label}"] = {"coef": m.params.get(PSI_COL, np.nan),
                                            "pval": m.pvalues.get(PSI_COL, np.nan), "n": int(m.nobs)}
            except Exception as e:
                results[f"psi_{label}"] = {"error": str(e)}

    if save_outputs:
        rows = [{"test": k, **v} for k,v in results.items() if isinstance(v, dict)]
        if rows: pd.DataFrame(rows).to_csv(OUTPUTS_TABLES / "info_vs_routine.csv", index=False)

        try:
            import matplotlib.pyplot as plt
            stages = []; coefs = []
            for s in ["early", "mid", "late"]:
                r = results.get(f"psi_effect_{s}", {})
                if "coef" in r:
                    stages.append(s); coefs.append(r["coef"])
            if stages:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(range(len(stages)), coefs, color="#00BCD4")
                ax.set_xticks(range(len(stages))); ax.set_xticklabels(stages)
                ax.set_ylabel("ψ Coefficient on Exit"); ax.set_title("ψ Effects Over Voyage Stage")
                ax.axhline(0, color="black", lw=0.5); ax.grid(axis="y", alpha=0.3); fig.tight_layout()
                fig.savefig(OUTPUTS_FIGURES / f"info_vs_routine.{FIGURE_FORMAT}", dpi=FIGURE_DPI, bbox_inches="tight")
                plt.close(fig)
        except ImportError: pass

        (OUTPUTS_TABLES / "info_vs_routine_memo.md").write_text(
            "# Test 11: Info vs Routine — Memo\n\n## Identifies\nWhether ψ effects spike near info refreshes or persist deep into voyage.\n\n## Does NOT identify\n- Cannot separate agent instructions from captain internalization\n")
    return results
