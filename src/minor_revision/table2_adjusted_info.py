"""
Phase 1A: Fix Table 2 — Adjusted Information Metrics.

Replaces raw in-sample Shannon routing table with:
  - Table 2A: Adjusted Mutual Information (AMI) + bootstrap CI
  - Table 2B: Conditional Mutual Information
  - Table 2C: Frequency-restricted robustness
"""

import numpy as np
import pandas as pd

from .utils.io import write_table, write_doc
from .utils.entropy import (
    raw_mi, ami, nmi, entropy_bits,
    conditional_mi_smoothed, compute_all_mi_metrics,
)
from .utils.bootstrap import voyage_bootstrap_ci
from .utils.plotting import mi_comparison_figure


def _prepare_table2_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to voyages with non-null ground_or_route."""
    mask = df["ground_or_route"].notna()
    sample = df.loc[mask].copy()
    print(f"  Table 2 sample: {len(sample):,} voyages with ground_or_route")
    print(f"    Unique grounds: {sample['ground_or_route'].nunique()}")
    print(f"    Unique captains: {sample['captain_id'].nunique()}")
    print(f"    Unique agents: {sample['agent_id'].nunique()}")
    print(f"    Unique ports: {sample['home_port'].nunique()}")
    return sample


def run_table2a_adjusted_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 2A: Compute Raw MI, AMI, and NMI for each predictor.
    """
    print("\n  --- Table 2A: Adjusted Information Metrics ---")
    sample = _prepare_table2_sample(df)

    ground = sample["ground_or_route"].values
    predictors = {
        "Home Port": sample["home_port"].fillna("Unknown").values,
        "Managing Agent": sample["agent_id"].values,
        "Captain Identity": sample["captain_id"].values,
    }

    results = []
    for name, pred in predictors.items():
        metrics = compute_all_mi_metrics(ground, pred, name)

        # Bootstrap CI for AMI
        def ami_stat(df_boot):
            return ami(df_boot["ground_or_route"].values, df_boot[pred_col].values)

        # Map predictor name to column
        pred_col_map = {
            "Home Port": "home_port",
            "Managing Agent": "agent_id",
            "Captain Identity": "captain_id",
        }
        pred_col = pred_col_map[name]

        # Ensure no duplicate columns (e.g. when pred_col == "captain_id")
        boot_cols = list(dict.fromkeys(["ground_or_route", pred_col, "captain_id"]))
        boot = voyage_bootstrap_ci(
            sample[boot_cols].dropna(),
            lambda d, pc=pred_col: ami(
                d["ground_or_route"].values, d[pc].values
            ),
            n_boot=500,
            block_col="captain_id",
        )
        metrics["AMI 95% CI"] = f"[{boot['ci_lower']:.3f}, {boot['ci_upper']:.3f}]"
        metrics["AMI SE"] = f"{boot['se']:.3f}"
        results.append(metrics)
        print(f"    {name}: Raw MI={metrics['Raw MI (bits)']}, "
              f"AMI={metrics['AMI']}, NMI={metrics['NMI']}, "
              f"CI={metrics['AMI 95% CI']}")

    # Add baseline entropy
    h_ground = entropy_bits(ground)
    results.insert(0, {
        "Predictor": "Baseline Uncertainty H(Ground)",
        "Raw MI (bits)": f"{h_ground:.3f}",
        "AMI": "—",
        "NMI": "—",
        "AMI 95% CI": "—",
        "AMI SE": "—",
    })

    df_results = pd.DataFrame(results)
    write_table(
        df_results,
        "table2_adjusted_information",
        caption="Table 2A: Adjusted Information — Route Choice Attribution",
        notes=(
            "Raw MI in bits for reproducibility. AMI adjusts for chance agreement "
            "due to high-cardinality predictors. Bootstrap CIs computed via "
            "captain-level block bootstrap (500 replications). "
            f"Sample: {len(sample):,} voyages with non-null ground_or_route."
        ),
    )

    # Plot
    mi_comparison_figure(df_results[df_results["Predictor"] != "Baseline Uncertainty H(Ground)"])

    return df_results


def run_table2b_conditional_mi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 2B: Conditional Mutual Information.

    I(Ground; Captain | Agent) — captain-specific routing knowledge beyond agent
    I(Ground; Agent | Captain) — agent-specific routing knowledge beyond captain
    """
    print("\n  --- Table 2B: Conditional Mutual Information ---")
    sample = _prepare_table2_sample(df)

    ground = sample["ground_or_route"].values
    captain = sample["captain_id"].values
    agent = sample["agent_id"].values
    port = sample["home_port"].fillna("Unknown").values

    results = []

    # I(Ground; Captain | Agent)
    cmi_cap_given_agent = conditional_mi_smoothed(ground, captain, agent, alpha=1.0)
    results.append({
        "Conditional MI": "I(Ground; Captain | Agent)",
        "Estimate (bits)": f"{cmi_cap_given_agent:.3f}",
        "Interpretation": "Captain-specific routing knowledge beyond agent portfolio",
    })
    print(f"    I(Ground; Captain | Agent) = {cmi_cap_given_agent:.3f}")

    # I(Ground; Agent | Captain)
    cmi_agent_given_captain = conditional_mi_smoothed(ground, agent, captain, alpha=1.0)
    results.append({
        "Conditional MI": "I(Ground; Agent | Captain)",
        "Estimate (bits)": f"{cmi_agent_given_captain:.3f}",
        "Interpretation": "Agent-specific routing pattern beyond captain identity",
    })
    print(f"    I(Ground; Agent | Captain) = {cmi_agent_given_captain:.3f}")

    # I(Ground; Captain | Agent, Port)
    agent_port = pd.Series(
        [f"{a}_{p}" for a, p in zip(agent, port)], dtype=str
    ).values
    cmi_cap_given_agent_port = conditional_mi_smoothed(ground, captain, agent_port, alpha=1.0)
    results.append({
        "Conditional MI": "I(Ground; Captain | Agent, Port)",
        "Estimate (bits)": f"{cmi_cap_given_agent_port:.3f}",
        "Interpretation": "Captain routing skill net of agent and port",
    })
    print(f"    I(Ground; Captain | Agent, Port) = {cmi_cap_given_agent_port:.3f}")

    # I(Ground; Agent | Captain, Port)
    captain_port = pd.Series(
        [f"{c}_{p}" for c, p in zip(captain, port)], dtype=str
    ).values
    cmi_agent_given_captain_port = conditional_mi_smoothed(ground, agent, captain_port, alpha=1.0)
    results.append({
        "Conditional MI": "I(Ground; Agent | Captain, Port)",
        "Estimate (bits)": f"{cmi_agent_given_captain_port:.3f}",
        "Interpretation": "Agent routing pattern net of captain and port",
    })
    print(f"    I(Ground; Agent | Captain, Port) = {cmi_agent_given_captain_port:.3f}")

    df_results = pd.DataFrame(results)
    write_table(
        df_results,
        "table2_conditional_information",
        caption="Table 2B: Conditional Mutual Information — Incremental Attribution",
        notes=(
            "Conditional MI computed from smoothed empirical joint counts "
            "(Dirichlet α=1.0). Separates captain-specific routing knowledge from "
            "agent portfolio effects. "
            f"Sample: {len(sample):,} voyages."
        ),
    )
    return df_results


def run_table2c_frequency_restricted(df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 2C: Frequency-restricted robustness for AMI and conditional MI.
    """
    print("\n  --- Table 2C: Frequency-Restricted Robustness ---")
    sample = _prepare_table2_sample(df)

    # Build subsamples
    captain_counts = sample["captain_id"].value_counts()
    agent_counts = sample["agent_id"].value_counts()
    ca_pair = sample["captain_id"].astype(str) + "_" + sample["agent_id"].astype(str)
    pair_counts = ca_pair.value_counts()
    repeated_pairs = pair_counts[pair_counts >= 2].index
    sample["ca_pair"] = ca_pair.values

    subsamples = {
        "Full sample": sample,
        "Captain ≥ 2 voyages": sample[
            sample["captain_id"].isin(captain_counts[captain_counts >= 2].index)
        ],
        "Captain ≥ 3 voyages": sample[
            sample["captain_id"].isin(captain_counts[captain_counts >= 3].index)
        ],
        "Agent ≥ 5 voyages": sample[
            sample["agent_id"].isin(agent_counts[agent_counts >= 5].index)
        ],
        "Repeated captain-agent pairs": sample[
            sample["ca_pair"].isin(repeated_pairs)
        ],
    }

    results = []
    for sub_name, sub_df in subsamples.items():
        if len(sub_df) < 50:
            print(f"    {sub_name}: SKIPPED (N={len(sub_df)})")
            continue

        ground = sub_df["ground_or_route"].values
        captain = sub_df["captain_id"].values
        agent = sub_df["agent_id"].values

        ami_cap = ami(ground, captain)
        ami_agent = ami(ground, agent)
        cmi_cap = conditional_mi_smoothed(ground, captain, agent, alpha=1.0)
        cmi_agent = conditional_mi_smoothed(ground, agent, captain, alpha=1.0)

        results.append({
            "Subsample": sub_name,
            "N": len(sub_df),
            "AMI(Captain)": f"{ami_cap:.3f}",
            "AMI(Agent)": f"{ami_agent:.3f}",
            "I(G;Cap|Agt)": f"{cmi_cap:.3f}",
            "I(G;Agt|Cap)": f"{cmi_agent:.3f}",
        })
        print(f"    {sub_name}: N={len(sub_df):,}, "
              f"AMI(Cap)={ami_cap:.3f}, AMI(Agt)={ami_agent:.3f}")

    df_results = pd.DataFrame(results)
    write_table(
        df_results,
        "table2_frequency_restricted",
        caption="Table 2C: Frequency-Restricted Robustness for Information Metrics",
        notes=(
            "AMI and conditional MI recomputed on subsamples to assess sensitivity "
            "to high-cardinality units. Dirichlet smoothing (α=1.0) applied to "
            "conditional MI in all subsamples."
        ),
    )
    return df_results


def run_table2_all(df: pd.DataFrame) -> dict:
    """Run all Table 2 components."""
    print("\n" + "=" * 70)
    print("PHASE 1A: FIX TABLE 2 — ADJUSTED INFORMATION METRICS")
    print("=" * 70)

    t2a = run_table2a_adjusted_info(df)
    t2b = run_table2b_conditional_mi(df)
    t2c = run_table2c_frequency_restricted(df)

    return {"table2a": t2a, "table2b": t2b, "table2c": t2c}
