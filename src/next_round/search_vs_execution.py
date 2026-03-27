"""
Test 5: Search vs Execution Decomposition.

Separates finding whales (search) from converting encounters into output
(execution). Estimates separate models for each stage of the production chain.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

from src.next_round.config import (
    OUTPUTS_TABLES, OUTPUTS_FIGURES, PSI_COL, THETA_COL,
    RANDOM_SEED, FIGURE_DPI, FIGURE_FORMAT,
)

logger = logging.getLogger(__name__)

CONTROLS = ["scarcity", "tonnage", "duration_day", "captain_voyage_num"]


def run_search_vs_execution(*, save_outputs: bool = True) -> Dict:
    """
    Decompose the production chain:
      encounter_hazard → strike|encounter → capture|strike → yield|capture

    Key question: Does ψ primarily affect search/encounter
    rather than downstream execution?
    """
    logger.info("=" * 60)
    logger.info("Test 5: Search vs Execution Decomposition")
    logger.info("=" * 60)

    from src.reinforcement.data_builder import load_logbook_positions, build_analysis_panel

    # Load daily logbook data with encounters
    positions = load_logbook_positions()
    voyages = build_analysis_panel(require_akm=True)

    # Merge psi/theta
    desired_cols = ["voyage_id", PSI_COL, THETA_COL, "captain_id", "agent_id",
                    "captain_voyage_num", "tonnage", "scarcity"]
    available_cols = [c for c in desired_cols if c in voyages.columns]
    voyage_info = voyages[available_cols].dropna(subset=[PSI_COL])
    # Drop overlapping columns from positions to avoid suffixed duplicates
    overlap_cols = [c for c in available_cols if c in positions.columns and c != "voyage_id"]
    positions_clean = positions.drop(columns=overlap_cols, errors="ignore")
    df = positions_clean.merge(voyage_info, on="voyage_id", how="inner")

    logger.info("Search/execution dataset: %d daily observations, %d voyages",
                len(df), df["voyage_id"].nunique())

    # ── Identify event layers ─────────────────────────────────────────
    event_cols = _detect_event_columns(df)
    logger.info("Detected event columns: %s", event_cols)

    # Pre-convert string columns to binary flags
    for event_key, ecol in event_cols.items():
        if df[ecol].dtype == object:
            df[ecol] = (~df[ecol].isin(["NoEnc", "No", "None", "", "0"]) & df[ecol].notna()).astype(int)
        elif not pd.api.types.is_numeric_dtype(df[ecol]):
            df[ecol] = pd.to_numeric(df[ecol], errors="coerce").fillna(0).astype(int)

    results = {}

    # ── Stage 1: Encounter hazard ─────────────────────────────────────
    if event_cols.get("encounter"):
        results["encounter"] = _estimate_stage(
            df, event_cols["encounter"], "encounter_hazard")

    # ── Stage 2: Strike conditional on encounter ──────────────────────
    if event_cols.get("encounter") and event_cols.get("strike"):
        sub = df[df[event_cols["encounter"]] > 0]
        if len(sub) > 50:
            results["strike_given_encounter"] = _estimate_stage(
                sub, event_cols["strike"], "strike|encounter")

    # ── Stage 3: Capture conditional on strike ────────────────────────
    if event_cols.get("strike") and event_cols.get("capture"):
        sub = df[df[event_cols["strike"]] > 0]
        if len(sub) > 50:
            results["capture_given_strike"] = _estimate_stage(
                sub, event_cols["capture"], "capture|strike")

    # ── Stage 4: Overall production function ──────────────────────────
    # Voyage-level
    results["voyage_output"] = _estimate_voyage_level(voyages)

    if save_outputs:
        _save_outputs(results)

    return results


def _detect_event_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Detect which encounter/strike/capture columns exist."""
    events = {}

    for col in df.columns:
        cl = col.lower()
        if "encounter" in cl or cl == "encounter":
            events["encounter"] = col
        elif "struck" in cl or "strike" in cl or cl == "n_struck":
            events["strike"] = col
        elif "tried" in cl or cl == "n_tried":
            events["capture"] = col

    # Fallback: use binary encounter column
    if "encounter" not in events:
        for col in ["encounter", "whale_encounter", "sighting"]:
            if col in df.columns:
                events["encounter"] = col
                break

    return events


def _estimate_stage(df: pd.DataFrame, target_col: str, stage_name: str) -> Dict:
    """Estimate a single stage model."""
    try:
        import statsmodels.formula.api as smf

        # Binary outcome — handle string columns
        import re
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', stage_name)
        y_col = f"_{safe_name}_binary"
        df = df.copy()
        if df[target_col].dtype == object:
            # String column: any non-empty, non-"No*" value is positive
            df[y_col] = (~df[target_col].isin(["NoEnc", "No", "None", "", "0"]) & df[target_col].notna()).astype(int)
        else:
            df[y_col] = (df[target_col] > 0).astype(int)

        available_controls = [c for c in [PSI_COL, THETA_COL] + CONTROLS
                              if c in df.columns]

        formula = f"{y_col} ~ {' + '.join(available_controls)}"
        df_reg = df[[y_col] + available_controls + ["captain_id"]].dropna()

        if len(df_reg) < 100:
            return {"error": "insufficient_data", "n": len(df_reg)}

        model = smf.ols(formula, data=df_reg).fit(
            cov_type="cluster",
            cov_kwds={"groups": df_reg["captain_id"]}
        )

        return {
            "stage": stage_name,
            "n_obs": int(model.nobs),
            "base_rate": df_reg[y_col].mean(),
            "psi_coef": model.params.get(PSI_COL, np.nan),
            "psi_se": model.bse.get(PSI_COL, np.nan),
            "psi_pval": model.pvalues.get(PSI_COL, np.nan),
            "theta_coef": model.params.get(THETA_COL, np.nan),
            "theta_se": model.bse.get(THETA_COL, np.nan),
            "theta_pval": model.pvalues.get(THETA_COL, np.nan),
            "r_squared": model.rsquared,
        }

    except ImportError:
        # Fallback: simple correlation
        df_clean = df[[target_col, PSI_COL, THETA_COL]].dropna()
        return {
            "stage": stage_name,
            "n_obs": len(df_clean),
            "base_rate": (df_clean[target_col] > 0).mean(),
            "psi_corr": df_clean[target_col].corr(df_clean[PSI_COL]),
            "theta_corr": df_clean[target_col].corr(df_clean[THETA_COL]),
        }


def _estimate_voyage_level(voyages: pd.DataFrame) -> Dict:
    """Estimate voyage-level output model."""
    try:
        import statsmodels.formula.api as smf

        outcome = "q_total_index" if "q_total_index" in voyages.columns else "log_q"
        available = [c for c in [PSI_COL, THETA_COL] + CONTROLS
                     if c in voyages.columns]

        formula = f"{outcome} ~ {' + '.join(available)}"
        df_reg = voyages[[outcome] + available + ["captain_id"]].dropna()

        model = smf.ols(formula, data=df_reg).fit(
            cov_type="cluster",
            cov_kwds={"groups": df_reg["captain_id"]}
        )

        return {
            "stage": "voyage_output",
            "n_obs": int(model.nobs),
            "psi_coef": model.params.get(PSI_COL, np.nan),
            "psi_se": model.bse.get(PSI_COL, np.nan),
            "psi_pval": model.pvalues.get(PSI_COL, np.nan),
            "theta_coef": model.params.get(THETA_COL, np.nan),
            "theta_se": model.bse.get(THETA_COL, np.nan),
            "r_squared": model.rsquared,
        }
    except Exception as e:
        return {"error": str(e)}


def _save_outputs(results: Dict):
    """Save results."""
    rows = []
    for stage, info in results.items():
        if isinstance(info, dict):
            info["stage_key"] = stage
            rows.append(info)

    if rows:
        pd.DataFrame(rows).to_csv(
            OUTPUTS_TABLES / "search_vs_execution.csv", index=False)

    # Figure
    try:
        import matplotlib.pyplot as plt

        stages = []
        psi_coefs = []
        theta_coefs = []

        for stage in ["encounter", "strike_given_encounter",
                       "capture_given_strike", "voyage_output"]:
            info = results.get(stage, {})
            if "psi_coef" in info and not np.isnan(info["psi_coef"]):
                stages.append(stage.replace("_", "\n"))
                psi_coefs.append(info["psi_coef"])
                theta_coefs.append(info.get("theta_coef", 0))

        if stages:
            x = np.arange(len(stages))
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(x - 0.2, psi_coefs, 0.35, label="Agent (ψ)", color="#FF5722")
            ax.bar(x + 0.2, theta_coefs, 0.35, label="Captain (θ)", color="#2196F3")
            ax.set_xticks(x)
            ax.set_xticklabels(stages, fontsize=8)
            ax.set_ylabel("Coefficient")
            ax.set_title("Where in the Production Chain Does ψ Enter?")
            ax.legend()
            ax.axhline(0, color="black", lw=0.5)
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(OUTPUTS_FIGURES / f"search_vs_execution.{FIGURE_FORMAT}",
                       dpi=FIGURE_DPI, bbox_inches="tight")
            plt.close(fig)
    except ImportError:
        pass

    # Memo
    memo = [
        "# Test 5: Search vs Execution — Memo",
        "",
        "## What this identifies",
        "Whether organizational capability (ψ) affects the ability to **find** whales",
        "(encounter hazard) vs the ability to **convert** encounters into output.",
        "",
        "## What this does NOT identify",
        "- Cannot separate search skill from strategic positioning",
        "- Encounter detection may be noisy in logbook data",
        "- Execution skill may confound with crew/equipment quality",
    ]
    (OUTPUTS_TABLES / "search_vs_execution_memo.md").write_text("\n".join(memo))
