"""Test 13: Portability / Invariance. Tests whether high-psi policies generalize across grounds and time."""

from __future__ import annotations
import logging, numpy as np, pandas as pd
from src.next_round.config import OUTPUTS_TABLES, OUTPUTS_FIGURES, PSI_COL, THETA_COL, RANDOM_SEED, FIGURE_DPI, FIGURE_FORMAT

logger = logging.getLogger(__name__)

def run_portability_tests(*, save_outputs: bool = True) -> dict:
    logger.info("=" * 60); logger.info("Test 13: Portability Tests"); logger.info("=" * 60)
    from src.ml.build_action_dataset import build_action_dataset
    from src.reinforcement.data_builder import build_analysis_panel
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    actions = build_action_dataset(force_rebuild=False, save=False)
    voyages = build_analysis_panel(require_akm=True)
    desired = ["voyage_id", PSI_COL, THETA_COL, "captain_id", "year_out", "ground_or_route"]
    available = [c for c in desired if c in voyages.columns]
    voyage_info = voyages[available].dropna(subset=[PSI_COL]).drop_duplicates("voyage_id")
    overlap = [c for c in available if c in actions.columns and c != "voyage_id"]
    df = actions.drop(columns=overlap, errors="ignore").merge(voyage_info, on="voyage_id", how="inner")

    if "exit_patch_next" not in df.columns:
        return {"error": "no_exit_column"}

    target = "exit_patch_next"
    features = [c for c in [PSI_COL, THETA_COL, "consecutive_empty_days", "days_since_last_success",
                            "duration_day", "scarcity", "speed", "move_length"] if c in df.columns]
    results = {}

    # 1. Out-of-time: train early, test late
    median_year = df["year_out"].median()
    train_ot = df[df["year_out"] <= median_year]
    test_ot = df[df["year_out"] > median_year]

    for psi_group, label in [("high", "high_psi"), ("low", "low_psi")]:
        psi_median = df[PSI_COL].median()
        if psi_group == "high":
            tr_sub = train_ot[train_ot[PSI_COL] >= psi_median]
            te_sub = test_ot[test_ot[PSI_COL] >= psi_median]
        else:
            tr_sub = train_ot[train_ot[PSI_COL] < psi_median]
            te_sub = test_ot[test_ot[PSI_COL] < psi_median]

        if len(tr_sub) < 100 or len(te_sub) < 50: continue
        try:
            X_tr = tr_sub[features].fillna(0).values; y_tr = tr_sub[target].values
            X_te = te_sub[features].fillna(0).values; y_te = te_sub[target].values
            hgb = HistGradientBoostingClassifier(max_iter=100, max_depth=4, random_state=RANDOM_SEED)
            hgb.fit(X_tr, y_tr)
            auc = roc_auc_score(y_te, hgb.predict_proba(X_te)[:, 1]) if len(np.unique(y_te)) > 1 else np.nan
            results[f"out_of_time_{label}"] = {"auc": auc, "n_train": len(tr_sub), "n_test": len(te_sub)}
        except Exception as e:
            results[f"out_of_time_{label}"] = {"error": str(e)}

    # 2. Out-of-ground: train on some grounds, test on held-out
    if "ground_or_route" in df.columns:
        grounds = df["ground_or_route"].value_counts()
        top_grounds = grounds[grounds >= 50].index[:10].tolist()
        other_grounds = grounds.index.difference(top_grounds[:5]).tolist()

        train_og = df[df["ground_or_route"].isin(top_grounds[:5])]
        test_og = df[df["ground_or_route"].isin(other_grounds[:5])]

        if len(train_og) > 100 and len(test_og) > 50:
            try:
                X_tr = train_og[features].fillna(0).values; y_tr = train_og[target].values
                X_te = test_og[features].fillna(0).values; y_te = test_og[target].values
                hgb = HistGradientBoostingClassifier(max_iter=100, max_depth=4, random_state=RANDOM_SEED)
                hgb.fit(X_tr, y_tr)
                auc = roc_auc_score(y_te, hgb.predict_proba(X_te)[:, 1]) if len(np.unique(y_te)) > 1 else np.nan
                results["out_of_ground"] = {"auc": auc, "n_train": len(train_og), "n_test": len(test_og)}
            except Exception as e:
                results["out_of_ground"] = {"error": str(e)}

    if save_outputs:
        rows = [{"test": k, **v} for k,v in results.items() if isinstance(v, dict)]
        if rows: pd.DataFrame(rows).to_csv(OUTPUTS_TABLES / "portability_tests.csv", index=False)
        (OUTPUTS_TABLES / "portability_tests_memo.md").write_text(
            "# Test 13: Portability — Memo\n\n## Identifies\nWhether high-ψ policies generalize better across grounds and time.\n\n## Does NOT identify\n- Cannot separate policy portability from ground similarity\n")
    return results
