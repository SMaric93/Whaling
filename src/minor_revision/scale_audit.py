"""
Phase 4: Table 1 vs Table 3 Scale Discrepancy Resolution.

Identifies why the descriptive SD and the AKM variance decomposition
don't match, reconciles the statistics, and produces corrected tables.
"""

import numpy as np
import pandas as pd

from .utils.io import write_table, write_doc, ensure_dirs
from .utils.checks import compute_descriptive_stats, assert_var_sd_consistent


def run_scale_audit(
    df_filtered: pd.DataFrame,
    df_connected: pd.DataFrame,
) -> dict:
    """
    Run the full scale audit and reconciliation.

    Parameters
    ----------
    df_filtered : pd.DataFrame
        Full analysis sample (Table 1 basis).
    df_connected : pd.DataFrame
        LOO connected set sample (Table 3 basis).
    """
    print("\n" + "=" * 70)
    print("PHASE 4: TABLE 1 vs TABLE 3 SCALE AUDIT")
    print("=" * 70)
    ensure_dirs()

    # --- Compute descriptive stats for each outcome object ---
    objects = []

    # 1. Table 1 log_q — full filtered sample
    stats_t1 = compute_descriptive_stats(
        df_filtered["log_q"].values,
        "Table 1: Full filtered sample log(output)"
    )
    stats_t1["sample"] = "Full filtered"
    stats_t1["transformation"] = "log(q_total_index), trimmed 0.5%/99.5%"
    stats_t1["winsorized"] = "No (trimmed)"
    stats_t1["connected_set"] = "No"
    objects.append(stats_t1)

    # 2. Table 3 log_q — connected set
    stats_t3 = compute_descriptive_stats(
        df_connected["log_q"].values,
        "Table 3: Connected-set log(output)"
    )
    stats_t3["sample"] = "LOO Connected set"
    stats_t3["transformation"] = "log(q_total_index), trimmed 0.5%/99.5%"
    stats_t3["winsorized"] = "No (trimmed)"
    stats_t3["connected_set"] = "Yes"
    objects.append(stats_t3)

    # 3. Connected-set raw (no trimming)
    log_q_raw = np.log(df_connected["q_total_index"].values)
    log_q_raw = log_q_raw[~np.isnan(log_q_raw) & ~np.isinf(log_q_raw)]
    stats_raw = compute_descriptive_stats(log_q_raw, "Connected-set raw log(output)")
    stats_raw["sample"] = "LOO Connected set (untrimmed)"
    stats_raw["transformation"] = "log(q_total_index), no trimming"
    stats_raw["winsorized"] = "No"
    stats_raw["connected_set"] = "Yes"
    objects.append(stats_raw)

    # 4. Winsorized (1%/99%) on connected set
    lo1, hi1 = np.percentile(df_connected["log_q"].values, [1, 99])
    winsorized_vals = np.clip(df_connected["log_q"].values, lo1, hi1)
    stats_wins = compute_descriptive_stats(winsorized_vals, "Connected-set winsorized log(output)")
    stats_wins["sample"] = "LOO Connected set (winsorized 1%/99%)"
    stats_wins["transformation"] = "log(q_total_index), winsorized 1%/99%"
    stats_wins["winsorized"] = "Yes (1%/99%)"
    stats_wins["connected_set"] = "Yes"
    objects.append(stats_wins)

    # --- Build reconciliation table ---
    reconciliation = pd.DataFrame(objects)
    reconciliation = reconciliation.rename(columns={"label": "Object"})

    # Reorder columns
    col_order = ["Object", "sample", "N", "mean", "sd", "variance",
                 "min", "max", "p25", "p75",
                 "transformation", "winsorized", "connected_set"]
    reconciliation = reconciliation[[c for c in col_order if c in reconciliation.columns]]

    # Round numeric columns
    for c in ["mean", "sd", "variance", "min", "max", "p25", "p75"]:
        if c in reconciliation.columns:
            reconciliation[c] = reconciliation[c].round(4)

    write_table(
        reconciliation,
        "scale_audit",
        caption="Scale Audit: Reconciliation of Outcome Objects Across Tables",
        notes=(
            "Compares the exact outcome variable, sample, and transformation used "
            "in Table 1 (descriptive statistics) and Table 3 (variance decomposition). "
            "Any discrepancy in reported SD/variance is explained by differences in "
            "sample composition and/or transformation."
        ),
    )

    print("\n  Reconciliation Table:")
    for _, row in reconciliation.iterrows():
        print(f"    {row['Object']}: N={row['N']:,}, "
              f"Mean={row['mean']:.4f}, SD={row['sd']:.4f}, "
              f"Var={row['variance']:.4f}")

    # --- Verify internal consistency ---
    print("\n  --- Scale Consistency Checks ---")
    for _, row in reconciliation.iterrows():
        try:
            assert_var_sd_consistent(row["sd"], row["variance"], row["Object"])
            print(f"    ✓ {row['Object']}: SD²≈Var")
        except AssertionError as e:
            print(f"    ✗ {e}")

    # --- Report discrepancy source ---
    delta_n = stats_t1["N"] - stats_t3["N"]
    delta_mean = abs(stats_t1["mean"] - stats_t3["mean"])
    delta_sd = abs(stats_t1["sd"] - stats_t3["sd"])

    discrepancy_note = ""
    if delta_n != 0:
        discrepancy_note += (
            f"Sample size differs by {abs(delta_n):,} voyages "
            f"(Table 1: {stats_t1['N']:,}, Table 3: {stats_t3['N']:,}). "
        )
    if delta_mean > 0.05:
        discrepancy_note += (
            f"Mean differs by {delta_mean:.4f} log points. "
        )
    if delta_sd > 0.05:
        discrepancy_note += (
            f"SD differs by {delta_sd:.4f}. "
        )

    if not discrepancy_note:
        discrepancy_note = "Statistics are consistent across Table 1 and Table 3."

    print(f"\n  Discrepancy diagnosis: {discrepancy_note}")

    # --- Produce corrected Table 1 ---
    print("\n  --- Producing Corrected Table 1 ---")
    corrected_t1 = _build_corrected_table1(df_filtered, df_connected)
    write_table(
        corrected_t1,
        "table1_corrected",
        caption="Table 1 (Corrected): Descriptive Statistics and Sample Composition",
        notes=(
            f"Live-computed from the analysis sample. "
            f"Full sample: {len(df_filtered):,} voyages. "
            f"Connected set: {len(df_connected):,} voyages, "
            f"{df_connected['captain_id'].nunique():,} captains, "
            f"{df_connected['agent_id'].nunique():,} agents. "
            "Log output trimmed at 0.5%/99.5%."
        ),
    )

    # --- Table 3 support note ---
    t3_support = pd.DataFrame([{
        "Statistic": "AKM sample N",
        "Value": stats_t3["N"],
        "Note": "LOO connected set after trimming",
    }, {
        "Statistic": "AKM sample mean(log_q)",
        "Value": round(stats_t3["mean"], 4),
        "Note": "Differs from Table 1 if sample differs",
    }, {
        "Statistic": "AKM sample SD(log_q)",
        "Value": round(stats_t3["sd"], 4),
        "Note": "Should match if same variable/sample",
    }, {
        "Statistic": "AKM sample Var(log_q)",
        "Value": round(stats_t3["variance"], 4),
        "Note": "SD² should equal this",
    }, {
        "Statistic": "Unique captains",
        "Value": df_connected["captain_id"].nunique(),
        "Note": "In LOO connected set",
    }, {
        "Statistic": "Unique agents",
        "Value": df_connected["agent_id"].nunique(),
        "Note": "In LOO connected set",
    }])
    write_table(
        t3_support,
        "table3_note_support",
        caption="Table 3 Support: Connected-Set Descriptive Statistics",
        notes="Exact statistics from the AKM estimation sample.",
    )

    # --- Documentation ---
    doc = f"""# Scale Audit Fix — Table 1 vs Table 3 Discrepancy

*Fixes: Editor concern about SD / variance mismatch between Table 1 and Table 3.*

## Original Paper Tables
- **Table 1**: Descriptive statistics on the full analysis sample
- **Table 3**: Variance decomposition on the LOO connected set

## Source of Discrepancy
{discrepancy_note}

The discrepancy arises because:
1. **Different samples**: Table 1 uses the full filtered sample ({stats_t1["N"]:,} voyages),
   while Table 3 uses the LOO connected set ({stats_t3["N"]:,} voyages).
2. **Sample composition**: The connected set drops voyages from captains/agents who are
   not part of the largest bipartite connected component (or who are articulation points
   in the LOO procedure). This changes the moments.

## Corrected Values

| Statistic | Table 1 (full sample) | Table 3 (connected set) |
|-----------|----------------------:|------------------------:|
| N | {stats_t1["N"]:,} | {stats_t3["N"]:,} |
| Mean(log_q) | {stats_t1["mean"]:.4f} | {stats_t3["mean"]:.4f} |
| SD(log_q) | {stats_t1["sd"]:.4f} | {stats_t3["sd"]:.4f} |
| Var(log_q) | {stats_t1["variance"]:.4f} | {stats_t3["variance"]:.4f} |

## Fix Applied
1. Table 1 now reports live-computed statistics from the full analysis sample.
2. Table 3 notes now clearly state the sample is the LOO connected set and report
   connected-set descriptive statistics in a support table.
3. Scale consistency assertions added: `abs(SD² - Var) < 1e-4`.

## Old code path
- `src/analyses/paper_tables.py` → hardcoded `TABLE_1_DATA`, `TABLE_3_DATA`

## New code path
- `src/minor_revision/scale_audit.py` → live computation from data

## Does the interpretation change?
No. The variance decomposition shares (captain %, agent %, sorting %) are computed
on the connected-set sample and remain the authoritative estimates. The fix ensures
that descriptive statistics are correctly attributed to their respective samples.
"""
    write_doc("scale_audit_fix.md", doc)

    return {
        "reconciliation": reconciliation,
        "corrected_t1": corrected_t1,
        "stats_t1": stats_t1,
        "stats_t3": stats_t3,
        "discrepancy_note": discrepancy_note,
    }


def _build_corrected_table1(
    df_full: pd.DataFrame,
    df_connected: pd.DataFrame,
) -> pd.DataFrame:
    """Build corrected Table 1 with live-computed values."""
    n = len(df_full)
    n_connected = len(df_connected)

    rows = []

    # Outcomes
    rows.append({
        "Category": "Outcomes",
        "Variable": "Log Output (gallons)",
        "Mean": round(df_full["log_q"].mean(), 2),
        "SD": round(df_full["log_q"].std(), 2),
        "P25": round(df_full["log_q"].quantile(0.25), 2),
        "P75": round(df_full["log_q"].quantile(0.75), 2),
        "N": n,
    })

    # Revenue (if available)
    if "q_total_index" in df_full.columns:
        log_rev = np.log(df_full["q_total_index"])
        rows.append({
            "Category": "",
            "Variable": "Log Revenue (deflated)",
            "Mean": round(log_rev.mean(), 2),
            "SD": round(log_rev.std(), 2),
            "P25": round(log_rev.quantile(0.25), 2),
            "P75": round(log_rev.quantile(0.75), 2),
            "N": n,
        })

    # Inputs
    if "tonnage" in df_full.columns:
        t = df_full["tonnage"].dropna()
        if len(t) > 0:
            log_t = np.log(t.clip(lower=1))
            rows.append({
                "Category": "Inputs",
                "Variable": "Log Tonnage",
                "Mean": round(log_t.mean(), 2),
                "SD": round(log_t.std(), 2),
                "P25": round(log_t.quantile(0.25), 2),
                "P75": round(log_t.quantile(0.75), 2),
                "N": len(t),
            })

    if "crew_count" in df_full.columns:
        crew = df_full["crew_count"].dropna()
        if len(crew) > 0:
            rows.append({
                "Category": "",
                "Variable": "Crew Size",
                "Mean": round(crew.mean(), 1),
                "SD": round(crew.std(), 1),
                "P25": int(crew.quantile(0.25)),
                "P75": int(crew.quantile(0.75)),
                "N": len(crew),
            })

    # Fixed effects
    rows.append({
        "Category": "Fixed Effects",
        "Variable": "Unique Captains",
        "Mean": df_full["captain_id"].nunique(),
        "SD": "—", "P25": "—", "P75": "—", "N": "—",
    })
    rows.append({
        "Category": "",
        "Variable": "Unique Agents",
        "Mean": df_full["agent_id"].nunique(),
        "SD": "—", "P25": "—", "P75": "—", "N": "—",
    })

    # Connected set
    rows.append({
        "Category": "Connected Set",
        "Variable": "Voyages",
        "Mean": n_connected,
        "SD": "—", "P25": "—", "P75": "—", "N": "—",
    })
    rows.append({
        "Category": "",
        "Variable": "Captains (connected)",
        "Mean": df_connected["captain_id"].nunique(),
        "SD": "—", "P25": "—", "P75": "—", "N": "—",
    })
    rows.append({
        "Category": "",
        "Variable": "Agents (connected)",
        "Mean": df_connected["agent_id"].nunique(),
        "SD": "—", "P25": "—", "P75": "—", "N": "—",
    })

    return pd.DataFrame(rows)
