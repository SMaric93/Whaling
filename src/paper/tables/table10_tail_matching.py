from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .._table_common import save_table_outputs
from ..config import BuildContext
from ..data import infer_basin, load_action_dataset, load_connected_sample
from ..utils.footnotes import standard_footnote
from ..utils.inference import clustered_ols, numeric as _numeric
from ..utils.risk import expected_shortfall_proxy, lower_tail_reference


def _prepare_sample(connected: pd.DataFrame, action: pd.DataFrame) -> pd.DataFrame:
    dry_spells = (
        action.groupby("voyage_id")
        .agg(max_consecutive_empty_days=("consecutive_empty_days", "max"))
        .reset_index()
        if not action.empty
        else pd.DataFrame(columns=["voyage_id", "max_consecutive_empty_days"])
    )
    df = connected.merge(dry_spells, on="voyage_id", how="left")
    # Ensure required AKM columns exist (filled with NaN if missing)
    for required in ["theta", "psi", "scarcity"]:
        if required not in df.columns:
            df[required] = np.nan
    for col in ["q_total_index", "theta", "psi", "scarcity", "captain_experience", "max_consecutive_empty_days", "year_out"]:
        if col in df.columns:
            df[col] = _numeric(df[col], df.index)
    p10 = lower_tail_reference(df["q_total_index"], quantile=0.10)
    df["bottom_decile"] = (df["q_total_index"] <= p10).astype(float)
    df["expected_shortfall_proxy"] = expected_shortfall_proxy(df["q_total_index"], quantile=0.10)
    dry_cutoff = float(df["max_consecutive_empty_days"].quantile(0.90)) if df["max_consecutive_empty_days"].notna().any() else np.nan
    df["long_dry_spell"] = (df["max_consecutive_empty_days"] >= dry_cutoff).astype(float) if np.isfinite(dry_cutoff) else np.nan
    df["theta_x_psi"] = df["theta"] * df["psi"]
    df["theta_x_psi_x_scarcity"] = df["theta_x_psi"] * df["scarcity"]
    return df


def _panel_a(df: pd.DataFrame) -> list[dict]:
    specs = [
        ("mean output: theta × psi", "q_total_index", "theta_x_psi", ["psi", "theta", "theta_x_psi", "scarcity"]),
        ("mean output: theta × psi × scarcity", "q_total_index", "theta_x_psi_x_scarcity", ["psi", "theta", "scarcity", "theta_x_psi", "theta_x_psi_x_scarcity"]),
        ("bottom-decile risk: theta × psi", "bottom_decile", "theta_x_psi", ["psi", "theta", "theta_x_psi", "scarcity"]),
        ("expected shortfall: theta × psi", "expected_shortfall_proxy", "theta_x_psi", ["psi", "theta", "theta_x_psi", "scarcity"]),
        ("long dry spell: theta × psi", "long_dry_spell", "theta_x_psi", ["psi", "theta", "theta_x_psi", "scarcity"]),
    ]
    rows = []
    for row_label, outcome, key, regressors in specs:
        model = clustered_ols(df, outcome=outcome, regressors=regressors, cluster_col="captain_id")
        rows.append(
            {
                "panel": "Panel A",
                "row_label": row_label,
                "coefficient": model["coef"].get(key, np.nan),
                "std_error": model["se"].get(key, np.nan),
                "p_value": model["p"].get(key, np.nan),
                "n_obs": int(model["n_obs"]),
                "note": "Captain-clustered interaction estimate built directly from the connected voyage sample.",
            }
        )
    return rows


def _corr_stats(df: pd.DataFrame) -> tuple[float, float, int]:
    sample = df.dropna(subset=["theta", "psi"])
    if len(sample) < 4:
        return np.nan, np.nan, 0
    pearson = float(sample["theta"].corr(sample["psi"]))
    spearman = float(stats.spearmanr(sample["theta"], sample["psi"], nan_policy="omit").statistic)
    return pearson, spearman, int(len(sample))


def _panel_b(df: pd.DataFrame) -> list[dict]:
    rows = []
    df = df.copy()
    df["era"] = pd.cut(
        df["year_out"],
        bins=[-np.inf, 1829, 1869, np.inf],
        labels=["Pre-1830", "1830-1869", "1870+"],
    )
    scarcity_sample = df.dropna(subset=["scarcity"]).copy()
    if not scarcity_sample.empty:
        scarcity_sample["scarcity_tercile"] = pd.qcut(
            scarcity_sample["scarcity"].rank(method="first"),
            3,
            labels=["Tercile 1", "Tercile 2", "Tercile 3"],
        )
    df["basin"] = infer_basin(df["ground_or_route"])

    def add_row(row_label: str, sample: pd.DataFrame):
        pearson, spearman, n_obs = _corr_stats(sample)
        rows.append(
            {
                "panel": "Panel B",
                "row_label": row_label,
                "pearson": pearson,
                "spearman": spearman,
                "n_obs": n_obs,
                "note": "Observed theta-psi sorting moment in the connected voyage sample.",
            }
        )

    add_row("overall corr(theta, psi)", df)
    if "scarcity_tercile" in scarcity_sample.columns:
        for tercile, sample in scarcity_sample.groupby("scarcity_tercile", observed=True):
            add_row(f"by scarcity tercile: {tercile}", sample)
    for era, sample in df.groupby("era", observed=True):
        add_row(f"by era: {era}", sample)
    for port, sample in df.groupby("home_port", observed=True):
        if len(sample) >= 100:
            add_row(f"by port: {port}", sample)
    for basin, sample in df.groupby("basin", observed=True):
        if len(sample) >= 100:
            add_row(f"by basin: {basin}", sample)
    return rows


def _fit_matching_surface(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray] | tuple[None, None]:
    clean = df.dropna(subset=["q_total_index", "theta", "psi", "captain_id", "agent_id"]).copy()
    if clean.empty:
        return None, None
    scarcity_raw = _numeric(clean.get("scarcity"), clean.index)
    scarcity_median = scarcity_raw.median()
    if pd.isna(scarcity_median):
        scarcity_median = 0.0
    clean["scarcity"] = scarcity_raw.fillna(scarcity_median)
    X = np.column_stack(
        [
            np.ones(len(clean)),
            clean["theta"].to_numpy(dtype=float),
            clean["psi"].to_numpy(dtype=float),
            clean["scarcity"].to_numpy(dtype=float),
            (clean["theta"] * clean["psi"]).to_numpy(dtype=float),
            (clean["theta"] * clean["psi"] * clean["scarcity"]).to_numpy(dtype=float),
        ]
    )
    y = clean["q_total_index"].to_numpy(dtype=float)
    # Guard against NaN/inf in feature matrix
    finite_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    if finite_mask.sum() < X.shape[1] + 1:
        return None, None
    X, y, clean = X[finite_mask], y[finite_mask], clean.iloc[finite_mask.nonzero()[0]]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return clean, beta


def _predict_surface(beta: np.ndarray, theta: np.ndarray, psi: np.ndarray, scarcity: np.ndarray) -> np.ndarray:
    return (
        beta[0]
        + beta[1] * theta
        + beta[2] * psi
        + beta[3] * scarcity
        + beta[4] * theta * psi
        + beta[5] * theta * psi * scarcity
    )


def _assignment_summary(
    *,
    row_label: str,
    predicted_output: np.ndarray,
    novice_mask: np.ndarray,
    captain_ids: np.ndarray,
    assigned_agent_ids: np.ndarray,
    support_pairs: set[tuple[str, str]],
    note: str,
) -> dict:
    p10 = float(np.quantile(predicted_output, 0.10))
    tail = predicted_output[predicted_output <= p10]
    cvar = float(tail.mean()) if len(tail) else np.nan
    reassigned = np.mean([(captain, agent) not in support_pairs for captain, agent in zip(captain_ids, assigned_agent_ids)]) * 100
    return {
        "panel": "Panel C",
        "row_label": row_label,
        "mean_output": float(predicted_output.mean()),
        "novice_mean": float(predicted_output[novice_mask].mean()) if novice_mask.any() else np.nan,
        "p10": p10,
        "cvar_10": cvar,
        "expected_shortfall": cvar,
        "pct_reassigned_outside_observed_support": float(reassigned),
        "certainty_equivalent": float(predicted_output.mean() - 0.5 * predicted_output.var()),
        "note": note,
    }


def _quantile_choice(options: pd.DataFrame, quantile: float) -> tuple[str, float]:
    ordered = options.sort_values("psi").reset_index(drop=True)
    if ordered.empty:
        return "", np.nan
    q = float(np.clip(quantile, 0.0, 1.0))
    idx = int(round(q * (len(ordered) - 1)))
    row = ordered.iloc[idx]
    return str(row["agent_id"]), float(row["psi"])


def _supported_assignment_candidates(clean: pd.DataFrame, beta: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    captain_summary = clean.groupby("captain_id").agg(theta=("theta", "mean")).sort_values("theta")
    captain_rank = captain_summary["theta"].rank(method="average", pct=True).to_dict()
    support = (
        clean.groupby(["captain_id", "agent_id"], as_index=False)
        .agg(psi=("psi", "mean"))
        .sort_values(["captain_id", "psi", "agent_id"])
    )

    pam_choice: dict[str, tuple[str, float]] = {}
    nam_choice: dict[str, tuple[str, float]] = {}
    mean_choice: dict[str, tuple[str, float]] = {}
    cvar_choice: dict[str, tuple[str, float]] = {}
    ce_choice: dict[str, tuple[str, float]] = {}

    for captain_id, captain_rows in clean.groupby("captain_id", sort=False):
        captain_key = str(captain_id)
        support_options = support[support["captain_id"] == captain_id][["agent_id", "psi"]].drop_duplicates("agent_id")
        theta_rank = float(captain_rank.get(captain_id, 0.5))
        pam_choice[captain_key] = _quantile_choice(support_options, theta_rank)
        nam_choice[captain_key] = _quantile_choice(support_options, 1.0 - theta_rank)

        theta_vec = captain_rows["theta"].to_numpy(dtype=float)
        scarcity_vec = captain_rows["scarcity"].to_numpy(dtype=float)
        scored: list[tuple[str, float, float, float, float]] = []
        for _, option in support_options.iterrows():
            psi_value = float(option["psi"])
            predicted = _predict_surface(beta, theta_vec, np.full(len(theta_vec), psi_value, dtype=float), scarcity_vec)
            cutoff = float(np.quantile(predicted, 0.10))
            tail = predicted[predicted <= cutoff]
            cvar = float(tail.mean()) if len(tail) else np.nan
            mean_value = float(predicted.mean())
            ce_value = float(mean_value - 0.5 * predicted.var())
            scored.append((str(option["agent_id"]), psi_value, mean_value, cvar, ce_value))

        mean_best = max(scored, key=lambda item: item[2])
        cvar_best = max(scored, key=lambda item: item[3])
        ce_best = max(scored, key=lambda item: item[4])
        mean_choice[captain_key] = (mean_best[0], mean_best[1])
        cvar_choice[captain_key] = (cvar_best[0], cvar_best[1])
        ce_choice[captain_key] = (ce_best[0], ce_best[1])

    def materialize(choice_map: dict[str, tuple[str, float]]) -> dict[str, np.ndarray]:
        assigned_agent = clean["captain_id"].map(lambda captain: choice_map[str(captain)][0]).to_numpy(dtype=object)
        assigned_psi = clean["captain_id"].map(lambda captain: choice_map[str(captain)][1]).to_numpy(dtype=float)
        return {"agent_ids": assigned_agent, "psi": assigned_psi}

    return {
        "pam": materialize(pam_choice),
        "nam": materialize(nam_choice),
        "mean_optimal": materialize(mean_choice),
        "cvar_optimal": materialize(cvar_choice),
        "ce_optimal": materialize(ce_choice),
    }


def _panel_c(df: pd.DataFrame) -> list[dict]:
    result = _fit_matching_surface(df)
    if result[0] is None:
        return []
    clean, beta = result
    theta = clean["theta"].to_numpy(dtype=float)
    psi = clean["psi"].to_numpy(dtype=float)
    scarcity = clean["scarcity"].to_numpy(dtype=float)
    captain_ids = clean["captain_id"].astype(str).to_numpy()
    agent_ids = clean["agent_id"].astype(str).to_numpy()
    novice_mask = clean["novice"].fillna(0).astype(float).to_numpy() > 0
    support_pairs = set(zip(captain_ids, agent_ids))
    candidates = _supported_assignment_candidates(clean, beta)

    rows = {
        "observed assignment": _assignment_summary(
            row_label="observed assignment",
            predicted_output=_predict_surface(beta, theta, psi, scarcity),
            novice_mask=novice_mask,
            captain_ids=captain_ids,
            assigned_agent_ids=agent_ids,
            support_pairs=support_pairs,
            note="Predicted welfare under the observed captain-agent assignment.",
        ),
        "PAM": _assignment_summary(
            row_label="PAM",
            predicted_output=_predict_surface(beta, theta, candidates["pam"]["psi"], scarcity),
            novice_mask=novice_mask,
            captain_ids=captain_ids,
            assigned_agent_ids=candidates["pam"]["agent_ids"],
            support_pairs=support_pairs,
            note="Captain-local positive assortative heuristic using each captain's historically observed agent menu.",
        ),
        "AAM/NAM": _assignment_summary(
            row_label="AAM/NAM",
            predicted_output=_predict_surface(beta, theta, candidates["nam"]["psi"], scarcity),
            novice_mask=novice_mask,
            captain_ids=captain_ids,
            assigned_agent_ids=candidates["nam"]["agent_ids"],
            support_pairs=support_pairs,
            note="Captain-local negative assortative heuristic using each captain's historically observed agent menu.",
        ),
    }
    rows["constrained mean-optimal"] = _assignment_summary(
        row_label="constrained mean-optimal",
        predicted_output=_predict_surface(beta, theta, candidates["mean_optimal"]["psi"], scarcity),
        novice_mask=novice_mask,
        captain_ids=captain_ids,
        assigned_agent_ids=candidates["mean_optimal"]["agent_ids"],
        support_pairs=support_pairs,
        note="Captain-local supported assignment that maximizes predicted mean output over each captain's historically observed agent menu.",
    )
    rows["constrained CVaR-optimal"] = _assignment_summary(
        row_label="constrained CVaR-optimal",
        predicted_output=_predict_surface(beta, theta, candidates["cvar_optimal"]["psi"], scarcity),
        novice_mask=novice_mask,
        captain_ids=captain_ids,
        assigned_agent_ids=candidates["cvar_optimal"]["agent_ids"],
        support_pairs=support_pairs,
        note="Captain-local supported assignment that maximizes predicted CVaR(10%) over each captain's historically observed agent menu.",
    )
    rows["constrained certainty-equivalent-optimal"] = _assignment_summary(
        row_label="constrained certainty-equivalent-optimal",
        predicted_output=_predict_surface(beta, theta, candidates["ce_optimal"]["psi"], scarcity),
        novice_mask=novice_mask,
        captain_ids=captain_ids,
        assigned_agent_ids=candidates["ce_optimal"]["agent_ids"],
        support_pairs=support_pairs,
        note="Captain-local supported assignment that maximizes the certainty equivalent over each captain's historically observed agent menu.",
    )
    return list(rows.values())


def build(context: BuildContext):
    connected = load_connected_sample(context)
    action = load_action_dataset(context)
    df = _prepare_sample(connected, action)

    frame = pd.DataFrame(_panel_a(df) + _panel_b(df) + _panel_c(df))

    memo = standard_footnote(
        sample="Connected-set voyages with action-derived dry-spell measures for the tail-risk outcomes.",
        unit="Voyage throughout.",
        types_note="theta_hat and psi_hat are connected-set AKM types. Panel C rebuilds support-constrained matching summaries over the observed voyage pool using a predicted output surface.",
        fe="No fixed effects in the main interaction rows.",
        cluster="Captain clustering for Panel A. Panel B reports descriptive sorting moments. Panel C is a counterfactual welfare summary.",
        controls="Scarcity is included in all interaction models, with an explicit theta-hat × psi-hat × scarcity row for mean output.",
        interpretation="The connected voyage sample supports a tail-risk matching narrative better than a pure mean-output matching narrative: organizational value shows up most clearly in downside protection and in the interaction between captain skill and agent capability under risk.",
        caution="Panel C is a support-constrained captain-local planner exercise over historically observed captain-agent menus using a linear predicted output surface; it is not a full equilibrium assignment model.",
    )
    memo = (
        "# table10_tail_matching\n\n"
        + memo
        + "\n\nImplementation notes:\n"
        + "- Panel A is rebuilt directly from the connected voyage sample and adds explicit interaction rows for mean output, scarcity, and tail-risk outcomes.\n"
        + "- Panel B mirrors the sorting-moment logic from Table 2 but presents the manuscript-facing breakdown used in the matching section.\n"
        + "- Panel C compares observed, support-constrained PAM/AAM-style heuristics, and support-constrained objective-specific assignments under a linear predicted output surface.\n"
        + "- The expected-shortfall row uses the first nondegenerate lower-tail cutoff so the downside-severity measure remains informative when the raw 10th percentile output is zero.\n"
    )

    return save_table_outputs(
        name="table10_tail_matching",
        frame=frame,
        out_dir=context.outputs / "tables",
        context=context,
        memo=memo,
        title="Table 10. Tail Submodularity and Risk-Based Matching",
    )
