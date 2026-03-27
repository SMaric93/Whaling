"""
Test 3: State Transition Governance.

Shows that organizations shape transition logic between search states.
Uses HMM-discovered states to estimate how psi affects state-to-state
transition probabilities.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from src.next_round.config import (
    OUTPUTS_TABLES, OUTPUTS_FIGURES, PSI_COL, THETA_COL,
    N_PSI_QUARTILES, RANDOM_SEED, FIGURE_DPI, FIGURE_FORMAT,
)

logger = logging.getLogger(__name__)


def run_state_transition_governance(*, save_outputs: bool = True) -> Dict:
    """
    Test whether organizational capability shapes state transitions.

    Core transitions of interest:
    - barren_search -> exit/relocation (high psi exits faster)
    - exploitation -> stay/exploitation (high psi stays longer)
    - transit -> local_search or exploitation
    """
    logger.info("=" * 60)
    logger.info("Test 3: State Transition Governance")
    logger.info("=" * 60)

    # ── Load state dataset ────────────────────────────────────────────
    from src.ml.build_state_dataset import build_state_dataset
    from src.reinforcement.data_builder import build_analysis_panel

    state_df = build_state_dataset(force_rebuild=False, save=False)
    voyages = build_analysis_panel(require_akm=True)

    # Merge psi/theta into state data
    voyage_info = voyages[["voyage_id", PSI_COL, THETA_COL,
                           "captain_id", "agent_id"]].dropna(subset=[PSI_COL])
    df = state_df.merge(voyage_info, on="voyage_id", how="inner")

    logger.info("State transition dataset: %d windows, %d voyages",
                len(df), df["voyage_id"].nunique())

    # ── Get state labels ──────────────────────────────────────────────
    state_col = None
    for col in ["hmm_state_label", "state_label", "gmm_state_label"]:
        if col in df.columns:
            state_col = col
            break

    if state_col is None:
        # Use the raw state if labels aren't available
        for col in ["hmm_state", "gmm_state", "state_id"]:
            if col in df.columns:
                state_col = col
                break

    if state_col is None:
        logger.error("No state column found")
        return {"error": "no_state_column"}

    logger.info("Using state column: %s with %d unique states",
                state_col, df[state_col].nunique())

    # ── Compute transitions ───────────────────────────────────────────
    df = df.sort_values(["voyage_id", "window_start" if "window_start" in df.columns else df.columns[0]])
    df["next_state"] = df.groupby("voyage_id")[state_col].shift(-1)
    df = df.dropna(subset=["next_state"])

    # Psi quartiles
    df["psi_quartile"] = pd.qcut(
        df[PSI_COL].rank(method="first"), N_PSI_QUARTILES,
        labels=[f"Q{i+1}" for i in range(N_PSI_QUARTILES)]
    )

    results = {}

    # ── 1. Transition matrices by psi quartile ────────────────────────
    transition_matrices = {}
    for q in df["psi_quartile"].unique():
        sub = df[df["psi_quartile"] == q]
        tm = pd.crosstab(sub[state_col], sub["next_state"], normalize="index")
        transition_matrices[str(q)] = tm

    results["transition_matrices"] = transition_matrices

    # ── 2. Multinomial logit for next state ───────────────────────────
    mnl_results = _estimate_transition_mnl(df, state_col)
    results["mnl"] = mnl_results

    # ── 3. Discrete hazard for leaving each state ─────────────────────
    hazard_results = _estimate_state_exit_hazard(df, state_col)
    results["hazard"] = hazard_results

    # ── 4. Key interaction tests ──────────────────────────────────────
    interaction_results = _test_key_interactions(df, state_col)
    results["interactions"] = interaction_results

    if save_outputs:
        _save_outputs(results, df, state_col)

    return results


def _estimate_transition_mnl(df: pd.DataFrame, state_col: str) -> Dict:
    """Multinomial logit for next state conditional on current state and psi."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import log_loss, accuracy_score

    results = {}
    states = df[state_col].unique()

    for current_state in states:
        sub = df[df[state_col] == current_state].copy()
        if len(sub) < 100 or sub["next_state"].nunique() < 2:
            continue

        le = LabelEncoder()
        y = le.fit_transform(sub["next_state"].astype(str))

        features = [PSI_COL, THETA_COL]
        for col in ["scarcity", "captain_voyage_num", "duration_day"]:
            if col in sub.columns:
                features.append(col)

        available = [f for f in features if f in sub.columns]
        X = sub[available].fillna(0).values

        # Simple split
        n = len(X)
        train_n = int(n * 0.7)
        X_tr, y_tr = X[:train_n], y[:train_n]
        X_te, y_te = X[train_n:], y[train_n:]

        try:
            lr = LogisticRegression(max_iter=300, random_state=RANDOM_SEED)
            lr.fit(X_tr, y_tr)

            psi_idx = available.index(PSI_COL) if PSI_COL in available else None
            psi_coef = lr.coef_[:, psi_idx].tolist() if psi_idx is not None else []

            results[str(current_state)] = {
                "n_transitions": len(sub),
                "n_next_states": sub["next_state"].nunique(),
                "accuracy": accuracy_score(y_te, lr.predict(X_te)),
                "psi_coefficients": psi_coef,
            }
        except Exception as e:
            logger.warning("MNL failed for state %s: %s", current_state, e)

    return results


def _estimate_state_exit_hazard(df: pd.DataFrame, state_col: str) -> Dict:
    """Discrete-time hazard for leaving each state."""
    results = {}
    df["state_exit"] = (df[state_col] != df["next_state"]).astype(int)

    for state in df[state_col].unique():
        sub = df[df[state_col] == state].copy()
        if len(sub) < 100:
            continue

        exit_rate = sub["state_exit"].mean()
        exit_by_psi = sub.groupby("psi_quartile")["state_exit"].mean()

        results[str(state)] = {
            "n_obs": len(sub),
            "overall_exit_rate": exit_rate,
            "exit_by_psi_q1": exit_by_psi.get("Q1", np.nan),
            "exit_by_psi_q4": exit_by_psi.get("Q4", np.nan),
            "q4_minus_q1": exit_by_psi.get("Q4", 0) - exit_by_psi.get("Q1", 0),
        }

    return results


def _test_key_interactions(df: pd.DataFrame, state_col: str) -> Dict:
    """Test psi × scarcity and psi × negative_signal interactions."""
    results = {}
    df_exit = df.copy()
    df_exit["state_exit"] = (df_exit[state_col] != df_exit["next_state"]).astype(int)

    try:
        import statsmodels.formula.api as smf

        # Add interaction terms
        if "scarcity" in df_exit.columns:
            df_exit["psi_x_scarcity"] = df_exit[PSI_COL] * df_exit["scarcity"]

        for col in ["max_empty_streak_window", "time_since_success"]:
            if col in df_exit.columns:
                df_exit[f"psi_x_{col}"] = df_exit[PSI_COL] * df_exit[col]

        # Regression
        controls = [PSI_COL]
        for c in ["scarcity", "psi_x_scarcity", "captain_voyage_num"]:
            if c in df_exit.columns:
                controls.append(c)

        formula = f"state_exit ~ {' + '.join(controls)}"
        model = smf.ols(formula, data=df_exit.dropna(subset=controls + ["state_exit"])).fit(
            cov_type="cluster", cov_kwds={"groups": df_exit.dropna(subset=controls + ["state_exit"])["captain_id"]}
        )

        results["interaction_model"] = {
            "params": model.params.to_dict(),
            "pvalues": model.pvalues.to_dict(),
            "n_obs": int(model.nobs),
            "r_squared": model.rsquared,
        }

    except (ImportError, Exception) as e:
        logger.warning("Interaction test failed: %s", e)
        results["error"] = str(e)

    return results


def _save_outputs(results: Dict, df: pd.DataFrame, state_col: str):
    """Save tables, figures, and memo."""
    # Table
    hazard_rows = []
    for state, info in results.get("hazard", {}).items():
        if isinstance(info, dict):
            info["state"] = state
            hazard_rows.append(info)

    if hazard_rows:
        pd.DataFrame(hazard_rows).to_csv(
            OUTPUTS_TABLES / "state_transition_governance.csv", index=False)

    # Transition heatmap figure
    try:
        import matplotlib.pyplot as plt

        tms = results.get("transition_matrices", {})
        if len(tms) >= 2:
            fig, axes = plt.subplots(1, min(len(tms), 4), figsize=(5 * min(len(tms), 4), 4))
            if not hasattr(axes, "__len__"):
                axes = [axes]

            for ax, (q, tm) in zip(axes, sorted(tms.items())):
                im = ax.imshow(tm.values, cmap="YlOrRd", vmin=0, vmax=1)
                ax.set_title(f"ψ {q}")
                ax.set_xticks(range(len(tm.columns)))
                ax.set_xticklabels(tm.columns, rotation=45, ha="right", fontsize=7)
                ax.set_yticks(range(len(tm.index)))
                ax.set_yticklabels(tm.index, fontsize=7)

            fig.suptitle("State Transition Matrices by ψ Quartile")
            fig.tight_layout()
            fig.savefig(OUTPUTS_FIGURES / f"state_transition_heatmap.{FIGURE_FORMAT}",
                       dpi=FIGURE_DPI, bbox_inches="tight")
            plt.close(fig)
    except ImportError:
        pass

    # Memo
    memo = [
        "# Test 3: State Transition Governance — Memo",
        "",
        "## What this test identifies",
        "Whether organizational capability (ψ) affects the probability of",
        "transitioning between latent search states, especially:",
        "- barren_search → exit (faster for high ψ?)",
        "- exploitation → stay (longer for high ψ?)",
        "",
        "## What this does NOT identify",
        "- Cannot separate ψ from unobserved crew quality",
        "- State labels are model-dependent (HMM specification)",
        "",
    ]
    (OUTPUTS_TABLES / "state_transition_governance_memo.md").write_text("\n".join(memo))
