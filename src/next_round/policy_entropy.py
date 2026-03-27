"""Test 10: Policy Entropy / Standardization. Measures routinization via action entropy within matched states."""

from __future__ import annotations
import logging, numpy as np, pandas as pd
from src.next_round.config import OUTPUTS_TABLES, OUTPUTS_FIGURES, PSI_COL, THETA_COL, N_PSI_QUARTILES, FIGURE_DPI, FIGURE_FORMAT

logger = logging.getLogger(__name__)

def run_policy_entropy(*, save_outputs: bool = True) -> dict:
    logger.info("=" * 60); logger.info("Test 10: Policy Entropy"); logger.info("=" * 60)
    from src.ml.build_action_dataset import build_action_dataset
    from src.reinforcement.data_builder import build_analysis_panel

    actions = build_action_dataset(force_rebuild=False, save=False)
    voyages = build_analysis_panel(require_akm=True)
    voyage_cols = ["voyage_id", PSI_COL, THETA_COL, "captain_id", "agent_id", "novice"]
    voyage_cols = [c for c in voyage_cols if c in voyages.columns]
    voyage_info = voyages[voyage_cols].dropna(subset=[PSI_COL]).drop_duplicates("voyage_id")
    overlap = [c for c in voyage_cols if c in actions.columns and c != "voyage_id"]
    df = actions.drop(columns=overlap, errors="ignore").merge(voyage_info, on="voyage_id", how="inner")

    if "next_action_class" not in df.columns:
        return {"error": "no_action_class"}

    df["psi_q"] = pd.qcut(df[PSI_COL].rank(method="first"), N_PSI_QUARTILES, labels=[f"Q{i+1}" for i in range(N_PSI_QUARTILES)])
    results = {}

    # Action entropy by agent
    agent_entropy = df.groupby("agent_id").apply(lambda g: _entropy(g["next_action_class"])).reset_index()
    agent_entropy.columns = ["agent_id", "action_entropy"]
    agent_psi = df.groupby("agent_id")[PSI_COL].mean().reset_index()
    agent_stats = agent_entropy.merge(agent_psi, on="agent_id")
    results["psi_entropy_corr"] = agent_stats["action_entropy"].corr(agent_stats[PSI_COL])

    # Cross-captain action variance by agent
    captain_means = df.groupby(["agent_id", "captain_id"])["next_action_class"].mean().reset_index()
    cross_captain_var = captain_means.groupby("agent_id")["next_action_class"].var().reset_index()
    cross_captain_var.columns = ["agent_id", "cross_captain_var"]
    agent_stats = agent_stats.merge(cross_captain_var, on="agent_id", how="left")
    results["psi_cross_var_corr"] = agent_stats["cross_captain_var"].corr(agent_stats[PSI_COL])

    # By psi quartile
    entropy_by_q = df.groupby("psi_q").apply(lambda g: _entropy(g["next_action_class"]))
    results["entropy_by_psi_q"] = entropy_by_q.to_dict()

    # Novice vs expert conditional entropy
    if "novice" in df.columns:
        for nov in [0, 1]:
            sub = df[df["novice"] == nov]
            ent_by_q = sub.groupby("psi_q").apply(lambda g: _entropy(g["next_action_class"]))
            results[f"entropy_novice_{nov}"] = ent_by_q.to_dict()

    if save_outputs:
        rows = [{"metric": k, "value": str(v)} for k,v in results.items()]
        pd.DataFrame(rows).to_csv(OUTPUTS_TABLES / "policy_entropy.csv", index=False)

        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 4))
            qs = list(entropy_by_q.index); vals = list(entropy_by_q.values)
            ax.bar(range(len(qs)), vals, color="#9C27B0")
            ax.set_xticks(range(len(qs))); ax.set_xticklabels(qs)
            ax.set_ylabel("Action Entropy (bits)"); ax.set_title("Policy Entropy by ψ Quartile")
            ax.grid(axis="y", alpha=0.3); fig.tight_layout()
            fig.savefig(OUTPUTS_FIGURES / f"policy_entropy.{FIGURE_FORMAT}", dpi=FIGURE_DPI, bbox_inches="tight")
            plt.close(fig)
        except ImportError: pass

        (OUTPUTS_TABLES / "policy_entropy_memo.md").write_text(
            "# Test 10: Policy Entropy — Memo\n\n## Identifies\nWhether high-ψ orgs reduce conditional action dispersion.\n\n## Does NOT identify\n- Cannot separate standardization from simple conservatism\n")

    return results


def _entropy(series):
    counts = series.value_counts(normalize=True)
    return -(counts * np.log2(counts.clip(1e-12))).sum()
