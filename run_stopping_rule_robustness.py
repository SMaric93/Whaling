"""
Stopping Rule Robustness: LOO Historical Density vs Baseline Empty Patch Definition.

Runs the stopping rule test (Table A6) under two definitions of "empty" patches:
  1. Baseline: bottom quartile of imputed within-voyage yield
  2. LOO Historical Density: bottom quartile of prior-year cell density (leave-one-out)

Prints side-by-side comparison and saves a LaTeX robustness table.
"""

import sys
from pathlib import Path

# Setup path — same as run_pipeline.py
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
from scipy import stats

from config import STAGING_DIR
from src.analyses.data_loader import prepare_analysis_sample
from src.analyses.search_theory import (
    identify_patches,
    compute_patch_yield,
    compute_patch_yield_loo,
    run_stopping_rule_test,
)

OUTPUT_DIR = PROJECT_ROOT / "output" / "paper" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Stopping Rule Regression (shared logic)
# =============================================================================

def run_stopping_rule_regression(patches, voyage_df, empty_col="is_empty"):
    """
    Run the three stopping-rule models using a specified empty indicator.

    Returns dict with results for all_patches, empty_patches, and interaction.
    """
    patches = patches.copy()

    # Get agent info
    voyage_info = voyage_df[["voyage_id", "agent_id", "captain_id"]].copy()
    if "psi_hat" in voyage_df.columns:
        voyage_info["psi_hat"] = voyage_df["psi_hat"]
    else:
        agent_means = voyage_df.groupby("agent_id")["log_q"].mean()
        voyage_info["psi_hat"] = voyage_info["agent_id"].map(agent_means)

    patches = patches.merge(voyage_info, on="voyage_id", how="left")

    # Filter
    sample = patches.dropna(subset=["duration_days", "psi_hat", empty_col]).copy()
    sample = sample[sample["duration_days"] > 0]
    sample["log_duration"] = np.log(sample["duration_days"])
    sample["is_empty_flag"] = sample[empty_col].astype(float)

    results = {"n_total": len(sample), "n_empty": int(sample["is_empty_flag"].sum())}

    # --- Model 1: All patches ---
    y = sample["log_duration"].values
    X = np.column_stack([np.ones(len(sample)), sample["psi_hat"].values])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    n, k = X.shape
    sigma_sq = np.sum(resid**2) / (n - k)
    se = np.sqrt(np.diag(sigma_sq * np.linalg.inv(X.T @ X)))
    t = beta[1] / se[1]
    p = 2 * (1 - stats.t.cdf(np.abs(t), df=n - k))
    results["m1"] = {"beta": beta[1], "se": se[1], "t": t, "p": p, "n": n}

    # --- Model 2: Empty patches only ---
    empty = sample[sample["is_empty_flag"] == 1]
    if len(empty) > 30:
        y2 = empty["log_duration"].values
        X2 = np.column_stack([np.ones(len(empty)), empty["psi_hat"].values])
        beta2 = np.linalg.lstsq(X2, y2, rcond=None)[0]
        resid2 = y2 - X2 @ beta2
        n2, k2 = X2.shape
        sigma_sq2 = np.sum(resid2**2) / (n2 - k2)
        se2 = np.sqrt(np.diag(sigma_sq2 * np.linalg.inv(X2.T @ X2)))
        t2 = beta2[1] / se2[1]
        p2 = 2 * (1 - stats.t.cdf(np.abs(t2), df=n2 - k2))
        results["m2"] = {"beta": beta2[1], "se": se2[1], "t": t2, "p": p2, "n": n2}
    else:
        results["m2"] = None

    # --- Model 3: Interaction ---
    sample["psi_x_empty"] = sample["psi_hat"] * sample["is_empty_flag"]
    y3 = sample["log_duration"].values
    X3 = np.column_stack([
        np.ones(len(sample)),
        sample["psi_hat"].values,
        sample["is_empty_flag"].values,
        sample["psi_x_empty"].values,
    ])
    beta3 = np.linalg.lstsq(X3, y3, rcond=None)[0]
    resid3 = y3 - X3 @ beta3
    n3, k3 = X3.shape
    sigma_sq3 = np.sum(resid3**2) / (n3 - k3)
    try:
        se3 = np.sqrt(np.diag(sigma_sq3 * np.linalg.inv(X3.T @ X3)))
    except np.linalg.LinAlgError:
        se3 = np.sqrt(np.abs(np.diag(sigma_sq3 * np.linalg.pinv(X3.T @ X3))))
    t3 = beta3[3] / se3[3] if se3[3] > 0 else 0
    p3 = 2 * (1 - stats.t.cdf(np.abs(t3), df=n3 - k3))
    results["m3"] = {
        "beta_psi": beta3[1], "se_psi": se3[1],
        "beta_empty": beta3[2], "se_empty": se3[2],
        "beta_interaction": beta3[3], "se_interaction": se3[3],
        "t": t3, "p": p3, "n": n3,
    }

    return results


def stars(p):
    """Significance stars."""
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.1:
        return "*"
    return ""


def fmt(val, se, p):
    """Format coefficient as string with stars and SE."""
    return f"{val:+.4f}{stars(p)}", f"({se:.4f})"


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("STOPPING RULE ROBUSTNESS: BASELINE vs LOO HISTORICAL DENSITY")
    print("=" * 70)

    # --- Load data ---
    print("\nLoading data...")
    voyage_df = prepare_analysis_sample()
    positions_path = STAGING_DIR / "logbook_positions.parquet"
    if not positions_path.exists():
        print(f"ERROR: Logbook positions not found at {positions_path}")
        sys.exit(1)
    positions_df = pd.read_parquet(positions_path)
    print(f"Positions: {len(positions_df):,}")
    print(f"Voyages: {len(voyage_df):,}")

    # --- SR1: Identify patches ---
    patches = identify_patches(positions_df)

    # --- SR2a: Baseline yield classification ---
    print("\n" + "=" * 70)
    print("DEFINITION A: BASELINE (bottom quartile of imputed yield)")
    print("=" * 70)
    patches_baseline = compute_patch_yield(patches, voyage_df)
    res_baseline = run_stopping_rule_regression(patches_baseline, voyage_df, "is_empty")

    # --- SR2b: LOO historical density classification ---
    print("\n" + "=" * 70)
    print("DEFINITION B: LOO HISTORICAL DENSITY")
    print("=" * 70)
    patches_loo = compute_patch_yield_loo(patches, positions_df, voyage_df)
    res_loo = run_stopping_rule_regression(patches_loo, voyage_df, "is_empty_loo")

    # --- Print comparison ---
    print("\n")
    print("=" * 70)
    print("COMPARISON: STOPPING RULE RESULTS")
    print("=" * 70)

    header = f"{'':30s} {'Baseline':>20s} {'LOO Hist. Density':>20s}"
    print(header)
    print("-" * 70)

    # Model 1: All patches
    b1, s1 = fmt(res_baseline["m1"]["beta"], res_baseline["m1"]["se"], res_baseline["m1"]["p"])
    l1, ls1 = fmt(res_loo["m1"]["beta"], res_loo["m1"]["se"], res_loo["m1"]["p"])
    print(f"{'(1) All Patches: β(ψ)':30s} {b1:>20s} {l1:>20s}")
    print(f"{'    SE':30s} {s1:>20s} {ls1:>20s}")
    print(f"{'    N':30s} {res_baseline['m1']['n']:>20,d} {res_loo['m1']['n']:>20,d}")
    print()

    # Model 2: Empty only
    if res_baseline["m2"] and res_loo["m2"]:
        b2, s2 = fmt(res_baseline["m2"]["beta"], res_baseline["m2"]["se"], res_baseline["m2"]["p"])
        l2, ls2 = fmt(res_loo["m2"]["beta"], res_loo["m2"]["se"], res_loo["m2"]["p"])
        print(f"{'(2) Empty Only: β(ψ)':30s} {b2:>20s} {l2:>20s}")
        print(f"{'    SE':30s} {s2:>20s} {ls2:>20s}")
        print(f"{'    N':30s} {res_baseline['m2']['n']:>20,d} {res_loo['m2']['n']:>20,d}")
        print()

    # Model 3: Interaction
    b3, s3 = fmt(res_baseline["m3"]["beta_interaction"], res_baseline["m3"]["se_interaction"], res_baseline["m3"]["p"])
    l3, ls3 = fmt(res_loo["m3"]["beta_interaction"], res_loo["m3"]["se_interaction"], res_loo["m3"]["p"])
    print(f"{'(3) Empty × ψ:':30s} {b3:>20s} {l3:>20s}")
    print(f"{'    SE':30s} {s3:>20s} {ls3:>20s}")
    print(f"{'    N':30s} {res_baseline['m3']['n']:>20,d} {res_loo['m3']['n']:>20,d}")
    print()

    # Summary
    print(f"{'Total patches':30s} {res_baseline['n_total']:>20,d} {res_loo['n_total']:>20,d}")
    print(f"{'Empty patches':30s} {res_baseline['n_empty']:>20,d} {res_loo['n_empty']:>20,d}")
    print()

    # --- Concordance ---
    # Merge both classifications on the same patch index
    both = patches_baseline[["voyage_id", "patch_id", "is_empty"]].merge(
        patches_loo[["voyage_id", "patch_id", "is_empty_loo"]],
        on=["voyage_id", "patch_id"],
        how="inner",
    )
    agree = (both["is_empty"] == both["is_empty_loo"]).mean()
    print(f"{'Classification concordance':30s} {agree:>20.1%}")
    print()

    # --- Generate LaTeX ---
    print("Generating LaTeX table...")
    generate_latex_table(res_baseline, res_loo, both)
    print(f"Saved to {OUTPUT_DIR / 'table_stopping_rule_robustness.tex'}")

    print("\n✓ ROBUSTNESS TEST COMPLETE")


def generate_latex_table(res_baseline, res_loo, both):
    """Generate LaTeX robustness comparison table."""

    def fmt_coef(r, key="beta"):
        val = r[key]
        se = r["se"] if "se" in r else r[f"se_{key.split('_')[-1]}"] if f"se_{key.split('_')[-1]}" in r else r.get("se_interaction", 0)
        p = r["p"]
        s = stars(p)
        return f"{val:+.4f}{s}", f"({se:.4f})"

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Stopping Rule Robustness: Alternative Empty Patch Definitions}",
        r"\label{tab:stopping_rule_robustness}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"& \multicolumn{3}{c}{Baseline (Imputed Yield Q1)} & \multicolumn{3}{c}{LOO Historical Density} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
        r"& (1) All & (2) Empty & (3) Interaction & (4) All & (5) Empty & (6) Interaction \\",
        r"\midrule",
    ]

    # β(ψ) row
    row = r"$\beta(\psi)$"
    for res in [res_baseline, res_loo]:
        b1 = f"{res['m1']['beta']:+.4f}{stars(res['m1']['p'])}"
        b2 = f"{res['m2']['beta']:+.4f}{stars(res['m2']['p'])}" if res["m2"] else "---"
        b3 = f"{res['m3']['beta_psi']:+.4f}{stars(2 * (1 - stats.t.cdf(abs(res['m3']['beta_psi'] / res['m3']['se_psi']), df=res['m3']['n'] - 4)))}"
        row += f" & {b1} & {b2} & {b3}"
    lines.append(row + r" \\")

    # SE row
    row = r"SE"
    for res in [res_baseline, res_loo]:
        s1 = f"({res['m1']['se']:.4f})"
        s2 = f"({res['m2']['se']:.4f})" if res["m2"] else ""
        s3 = f"({res['m3']['se_psi']:.4f})"
        row += f" & {s1} & {s2} & {s3}"
    lines.append(row + r" \\")
    lines.append(r"\addlinespace")

    # Empty × ψ row
    row = r"Empty $\times$ $\psi$"
    for res in [res_baseline, res_loo]:
        e1 = "---"
        e2 = "---"
        e3 = f"{res['m3']['beta_interaction']:+.4f}{stars(res['m3']['p'])}"
        row += f" & {e1} & {e2} & {e3}"
    lines.append(row + r" \\")

    # SE interaction row
    row = r"SE (Interaction)"
    for res in [res_baseline, res_loo]:
        row += f" & --- & --- & ({res['m3']['se_interaction']:.4f})"
    lines.append(row + r" \\")
    lines.append(r"\addlinespace")

    # N row
    row = r"Observations"
    for res in [res_baseline, res_loo]:
        n1 = f"{res['m1']['n']:,}"
        n2 = f"{res['m2']['n']:,}" if res["m2"] else "---"
        n3 = f"{res['m3']['n']:,}"
        row += f" & {n1} & {n2} & {n3}"
    lines.append(row + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\small \item Notes: Dependent variable is $\log$(patch residence time). " +
        r"Columns (1)--(3) define `empty' patches as the bottom quartile of imputed within-voyage yield (baseline). " +
        r"Columns (4)--(6) define `empty' using leave-one-out historical cell density: " +
        r"the mean $\log(q)$ of \emph{other} voyages in the same 5$^\circ$ cell in prior years (bottom quartile or no prior history). " +
        f"Classification concordance between definitions: {(both['is_empty'] == both['is_empty_loo']).mean():.1%}. " +
        r"${}^{***}p<0.01$, ${}^{**}p<0.05$, ${}^{*}p<0.10$.",
        r"\end{tablenotes}",
        r"\end{table}",
    ])

    tex_path = OUTPUT_DIR / "table_stopping_rule_robustness.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
