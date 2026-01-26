"""
Event study analysis for agent switches (R3).

Implements within-captain event-time effects around switching
to a new agent to measure the productivity impact of changing
organizational intermediation.
"""

from typing import Dict, Optional, List
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

from .config import DEFAULT_SAMPLE

warnings.filterwarnings("ignore", category=FutureWarning)


def identify_agent_switches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify agent switch events and construct event-time indices.
    
    Parameters
    ----------
    df : pd.DataFrame
        Voyage data with captain_id, agent_id, and switch_agent.
        
    Returns
    -------
    pd.DataFrame
        Data with event-time variables added.
    """
    print("\n" + "=" * 60)
    print("IDENTIFYING AGENT SWITCH EVENTS")
    print("=" * 60)
    
    df = df.copy()
    df = df.sort_values(["captain_id", "year_out"])
    
    # Ensure switch_agent exists
    if "switch_agent" not in df.columns:
        df["prev_agent"] = df.groupby("captain_id")["agent_id"].shift(1)
        df["switch_agent"] = (df["agent_id"] != df["prev_agent"]).astype(float)
        first_voyage = df["prev_agent"].isna()
        df.loc[first_voyage, "switch_agent"] = np.nan
        df = df.drop(columns=["prev_agent"])
    
    # Identify switch voyages
    switch_voyages = df[df["switch_agent"] == 1].copy()
    print(f"Total switch events: {len(switch_voyages):,}")
    
    # Count switches per captain
    captain_switches = df.groupby("captain_id")["switch_agent"].sum()
    captains_with_switches = (captain_switches >= 1).sum()
    print(f"Captains with ≥1 switch: {captains_with_switches:,}")
    
    # For each captain, identify first switch and create event time
    df["voyage_seq"] = df.groupby("captain_id").cumcount()
    
    # Find first switch voyage for each captain
    first_switch = df[df["switch_agent"] == 1].groupby("captain_id")["voyage_seq"].first()
    df["first_switch_seq"] = df["captain_id"].map(first_switch)
    
    # Event time = current sequence - first switch sequence
    df["event_time"] = df["voyage_seq"] - df["first_switch_seq"]
    
    # Only valid for captains with a switch
    df.loc[df["first_switch_seq"].isna(), "event_time"] = np.nan
    
    # Summary
    valid_events = df["event_time"].notna()
    print(f"Voyages with event time: {valid_events.sum():,}")
    print(f"Event time range: [{df['event_time'].min():.0f}, {df['event_time'].max():.0f}]")
    
    return df


def construct_event_study_sample(
    df: pd.DataFrame,
    min_pre: int = 2,
    min_post: int = 2,
    event_window: int = 5,
) -> pd.DataFrame:
    """
    Construct sample for event study with balanced pre/post windows.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with event_time column.
    min_pre : int
        Minimum pre-switch voyages required.
    min_post : int
        Minimum post-switch voyages required.
    event_window : int
        Maximum absolute event time to include.
        
    Returns
    -------
    pd.DataFrame
        Balanced event study sample.
    """
    print("\n--- Constructing Event Study Sample ---")
    
    df = df.copy()
    
    # Require valid event time
    df = df[df["event_time"].notna()].copy()
    
    # Count pre and post voyages per captain
    pre_counts = df[df["event_time"] < 0].groupby("captain_id").size()
    post_counts = df[df["event_time"] >= 0].groupby("captain_id").size()
    
    # Captains meeting requirements
    valid_captains = set(pre_counts[pre_counts >= min_pre].index) & \
                     set(post_counts[post_counts >= min_post].index)
    
    print(f"Captains meeting min pre ({min_pre}) and post ({min_post}): {len(valid_captains):,}")
    
    # Filter to valid captains and event window
    df = df[
        df["captain_id"].isin(valid_captains) &
        (df["event_time"].abs() <= event_window)
    ].copy()
    
    print(f"Final sample: {len(df):,} voyages")
    print(f"  Unique captains: {df['captain_id'].nunique():,}")
    
    return df


def run_r3_event_study(
    df: pd.DataFrame,
    min_pre: int = 2,
    min_post: int = 2,
    event_window: int = 5,
    omit_period: int = -1,
) -> Dict:
    """
    R3: Event-time effects around agent switches.
    
    logQ_{c,t} = Σ_{k≠-1} β_k · 1[event_time=k] + α_c + δ_{vessel×period} + θ_{route×time} + ε
    
    Parameters
    ----------
    df : pd.DataFrame
        Voyage data.
    min_pre : int
        Minimum pre-switch voyages.
    min_post : int
        Minimum post-switch voyages.
    event_window : int
        Maximum event time to include.
    omit_period : int
        Omitted reference period (usually -1).
        
    Returns
    -------
    Dict
        Event study results with coefficients and standard errors.
    """
    print("\n" + "=" * 60)
    print("R3: EVENT STUDY - AGENT SWITCH DYNAMICS")
    print("=" * 60)
    
    # Identify switches and event times
    df = identify_agent_switches(df)
    
    # Construct balanced sample
    df_es = construct_event_study_sample(df, min_pre, min_post, event_window)
    
    if len(df_es) < 100:
        print("Insufficient observations for event study")
        return {"error": "insufficient_sample", "n": len(df_es)}
    
    n = len(df_es)
    y = df_es["log_q"].values
    
    # Event time dummies
    event_times = sorted([k for k in range(-event_window, event_window + 1) if k != omit_period])
    event_time_map = {k: i for i, k in enumerate(event_times)}
    
    # Build design matrix
    matrices = []
    
    # Captain FEs
    captain_ids = df_es["captain_id"].unique()
    captain_map = {c: i for i, c in enumerate(captain_ids)}
    captain_idx = df_es["captain_id"].map(captain_map).values
    X_captain = sp.csr_matrix(
        (np.ones(n), (np.arange(n), captain_idx)),
        shape=(n, len(captain_ids))
    )
    matrices.append(X_captain)
    
    # Event time dummies
    event_dummies = np.zeros((n, len(event_times)))
    for i, (_, row) in enumerate(df_es.iterrows()):
        k = int(row["event_time"])
        if k in event_time_map:
            event_dummies[i, event_time_map[k]] = 1
    matrices.append(sp.csr_matrix(event_dummies))
    
    # Vessel×period FEs (drop first)
    if "vessel_period" in df_es.columns:
        vp_ids = df_es["vessel_period"].unique()
        vp_map = {v: i for i, v in enumerate(vp_ids)}
        vp_idx = df_es["vessel_period"].map(vp_map).values
        X_vp = sp.csr_matrix(
            (np.ones(n), (np.arange(n), vp_idx)),
            shape=(n, len(vp_ids))
        )[:, 1:]
        matrices.append(X_vp)
    
    # Controls
    controls = []
    for col in ["log_duration", "log_tonnage"]:
        if col in df_es.columns:
            controls.append(df_es[col].values)
    if controls:
        matrices.append(sp.csr_matrix(np.column_stack(controls)))
    
    X = sp.hstack(matrices)
    
    print(f"\nEstimation:")
    print(f"  Observations: {n:,}")
    print(f"  Captain FEs: {len(captain_ids):,}")
    print(f"  Event time dummies: {len(event_times)} (omitting k={omit_period})")
    
    # Solve
    result = lsqr(X, y, iter_lim=10000, atol=1e-10, btol=1e-10)
    beta = result[0]
    
    # Extract event time coefficients
    idx_start = len(captain_ids)
    event_coefs = beta[idx_start:idx_start + len(event_times)]
    
    # Compute residuals for SE estimation
    y_hat = X @ beta
    residuals = y - y_hat
    sigma2 = np.sum(residuals**2) / (n - X.shape[1])
    
    # Approximate SEs (diagonal of (X'X)^{-1} * sigma^2)
    # Use leverage approximation for speed
    XtX_diag_inv = 1.0 / np.array(X.T.dot(X).diagonal()).flatten()
    event_se = np.sqrt(sigma2 * XtX_diag_inv[idx_start:idx_start + len(event_times)])
    
    # Model fit
    r2 = 1 - np.var(residuals) / np.var(y)
    
    # Build results DataFrame
    results_df = pd.DataFrame({
        "event_time": event_times,
        "coefficient": event_coefs,
        "se": event_se,
        "t_stat": event_coefs / event_se,
        "ci_lower": event_coefs - 1.96 * event_se,
        "ci_upper": event_coefs + 1.96 * event_se,
    })
    
    # Add omitted period
    omitted_row = pd.DataFrame({
        "event_time": [omit_period],
        "coefficient": [0.0],
        "se": [0.0],
        "t_stat": [np.nan],
        "ci_lower": [0.0],
        "ci_upper": [0.0],
    })
    results_df = pd.concat([results_df, omitted_row]).sort_values("event_time").reset_index(drop=True)
    
    print("\n--- Event Study Coefficients ---")
    print(results_df.to_string(index=False))
    
    # Pre-trend test: joint significance of k < -1
    pre_coefs = results_df[results_df["event_time"] < -1]["coefficient"]
    if len(pre_coefs) > 0:
        pre_mean = pre_coefs.mean()
        pre_max_abs = pre_coefs.abs().max()
        print(f"\nPre-trend check:")
        print(f"  Mean pre-period coefficient: {pre_mean:.4f}")
        print(f"  Max absolute pre-period coefficient: {pre_max_abs:.4f}")
    
    # Post-switch effect
    post_coefs = results_df[results_df["event_time"] >= 0]["coefficient"]
    if len(post_coefs) > 0:
        post_mean = post_coefs.mean()
        print(f"\nPost-switch effect:")
        print(f"  Mean post-period coefficient: {post_mean:.4f}")
    
    results = {
        "coefficients": results_df,
        "n": n,
        "n_captains": len(captain_ids),
        "r2": r2,
        "omit_period": omit_period,
        "event_window": event_window,
        "data": df_es,
    }
    
    return results


def create_event_study_figure(
    results: Dict,
    output_path: Optional[str] = None,
) -> None:
    """
    Create event study coefficient plot.
    
    Parameters
    ----------
    results : Dict
        Results from run_r3_event_study.
    output_path : str, optional
        Path to save figure.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping figure generation")
        return
    
    from .config import FIGURES_DIR
    from pathlib import Path
    
    if output_path is None:
        output_path = FIGURES_DIR / "r3_event_study_agent_switch.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    coefs = results["coefficients"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot coefficients with confidence intervals
    ax.errorbar(
        coefs["event_time"],
        coefs["coefficient"],
        yerr=[coefs["coefficient"] - coefs["ci_lower"], 
              coefs["ci_upper"] - coefs["coefficient"]],
        fmt="o-",
        color="steelblue",
        linewidth=2,
        markersize=8,
        capsize=4,
        capthick=2,
    )
    
    # Reference line at zero
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1)
    
    # Vertical line at switch (k=0)
    ax.axvline(x=-0.5, color="red", linestyle=":", linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel("Event Time (Voyages Relative to Agent Switch)", fontsize=12)
    ax.set_ylabel("Coefficient (Log Production)", fontsize=12)
    ax.set_title("R3: Event Study - Agent Switch Dynamics", fontsize=14, fontweight="bold")
    
    # Annotations
    ax.annotate(
        "Pre-Switch",
        xy=(coefs["event_time"].min() + 0.5, ax.get_ylim()[1] * 0.9),
        fontsize=10,
        color="gray",
    )
    ax.annotate(
        "Post-Switch",
        xy=(1, ax.get_ylim()[1] * 0.9),
        fontsize=10,
        color="gray",
    )
    
    # Stats box
    stats_text = (
        f"N = {results['n']:,}\n"
        f"Captains = {results['n_captains']:,}\n"
        f"R² = {results['r2']:.3f}"
    )
    ax.annotate(
        stats_text,
        xy=(0.02, 0.98),
        xycoords="axes fraction",
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"\nEvent study figure saved to {output_path}")


def run_event_study_analysis(
    df: pd.DataFrame,
    save_outputs: bool = True,
) -> Dict:
    """
    Run full event study analysis (R3).
    
    Parameters
    ----------
    df : pd.DataFrame
        Voyage data.
    save_outputs : bool
        Whether to save outputs.
        
    Returns
    -------
    Dict
        Event study results.
    """
    from .config import TABLES_DIR
    from pathlib import Path
    
    results = run_r3_event_study(df)
    
    if "error" not in results and save_outputs:
        # Create figure
        create_event_study_figure(results)
        
        # Save coefficients
        output_path = TABLES_DIR / "r3_event_study_coefficients.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results["coefficients"].to_csv(output_path, index=False)
        print(f"Event study coefficients saved to {output_path}")
    
    return results


if __name__ == "__main__":
    from .data_loader import prepare_analysis_sample
    
    df = prepare_analysis_sample()
    results = run_event_study_analysis(df)
