"""
Publication-quality plotting helpers.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List

from .io import FIGURES_DIR, ensure_dirs


# House style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def coefficient_path_figure(
    thresholds: np.ndarray,
    coefficients: np.ndarray,
    ci_lower: np.ndarray,
    ci_upper: np.ndarray,
    main_cutoff: float,
    n_empty: np.ndarray,
    share_empty: np.ndarray,
    xlabel: str = "Empty-patch percentile cutoff",
    ylabel_coef: str = r"Coefficient on $\psi \times \mathit{empty}$",
    title: str = "",
    filename: str = "appendix_stopping_threshold_curve.png",
    panel_c_coefficients: Optional[np.ndarray] = None,
    panel_c_ci_lower: Optional[np.ndarray] = None,
    panel_c_ci_upper: Optional[np.ndarray] = None,
    panel_c_label: str = "AFT Weibull",
) -> Path:
    """
    Create the two- or three-panel threshold sensitivity figure.

    Panel A: coefficient path with CI
    Panel B: share/count of empty patches
    Panel C (optional): alternative model coefficient path
    """
    ensure_dirs()

    n_panels = 3 if panel_c_coefficients is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5))

    if n_panels == 2:
        ax_a, ax_b = axes
        ax_c = None
    else:
        ax_a, ax_b, ax_c = axes

    # --- Panel A: Coefficient path ---
    ax_a.fill_between(thresholds, ci_lower, ci_upper, alpha=0.2, color="steelblue")
    ax_a.plot(thresholds, coefficients, "o-", color="steelblue", markersize=4, label="OLS")
    ax_a.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax_a.axvline(main_cutoff, color="crimson", linestyle=":", linewidth=1.2,
                 label=f"Main-text cutoff ({main_cutoff:.0f}th pctile)")
    ax_a.set_xlabel(xlabel)
    ax_a.set_ylabel(ylabel_coef)
    ax_a.set_title("Panel A: Coefficient Path")
    ax_a.legend(fontsize=8)

    # --- Panel B: Share and count ---
    color_share = "darkorange"
    ax_b.bar(thresholds, share_empty * 100, width=3, alpha=0.5, color=color_share,
             label="Share classified empty (%)")
    ax_b_twin = ax_b.twinx()
    ax_b_twin.plot(thresholds, n_empty, "s-", color="darkgreen", markersize=4,
                   label="N classified empty")
    ax_b.set_xlabel(xlabel)
    ax_b.set_ylabel("Share empty (%)", color=color_share)
    ax_b_twin.set_ylabel("N empty", color="darkgreen")
    ax_b.set_title("Panel B: Empty Patch Classification")

    # Combine legends
    h1, l1 = ax_b.get_legend_handles_labels()
    h2, l2 = ax_b_twin.get_legend_handles_labels()
    ax_b.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")

    # --- Panel C (optional): AFT coefficient path ---
    if ax_c is not None and panel_c_coefficients is not None:
        ax_c.fill_between(thresholds, panel_c_ci_lower, panel_c_ci_upper,
                          alpha=0.2, color="darkorchid")
        ax_c.plot(thresholds, panel_c_coefficients, "o-", color="darkorchid",
                  markersize=4, label=panel_c_label)
        ax_c.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax_c.axvline(main_cutoff, color="crimson", linestyle=":", linewidth=1.2)
        ax_c.set_xlabel(xlabel)
        ax_c.set_ylabel(ylabel_coef)
        ax_c.set_title(f"Panel C: {panel_c_label}")
        ax_c.legend(fontsize=8)

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

    fig.tight_layout()
    out_path = FIGURES_DIR / filename
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  → Wrote {filename}")
    return out_path


def mi_comparison_figure(
    df_results: "pd.DataFrame",
    filename: str = "table2_information_comparison.png",
) -> Path:
    """
    Bar chart comparing raw MI, AMI, and NMI across predictors.
    """
    import pandas as pd  # noqa: F811
    ensure_dirs()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    predictors = df_results["Predictor"].tolist()
    x = np.arange(len(predictors))
    width = 0.25

    raw_vals = [float(v) for v in df_results["Raw MI (bits)"]]
    ami_vals = [float(v) for v in df_results["AMI"]]
    nmi_vals = [float(v) for v in df_results["NMI"]]

    ax.bar(x - width, raw_vals, width, label="Raw MI (bits)", color="steelblue", alpha=0.8)
    ax.bar(x, ami_vals, width, label="Adjusted MI", color="darkorange", alpha=0.8)
    ax.bar(x + width, nmi_vals, width, label="Normalized MI", color="seagreen", alpha=0.8)

    ax.set_xlabel("Predictor Variable")
    ax.set_ylabel("Information Score")
    ax.set_title("Route-Choice Information: Raw vs. Adjusted Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(predictors, fontsize=9)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = FIGURES_DIR / filename
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  → Wrote {filename}")
    return out_path
