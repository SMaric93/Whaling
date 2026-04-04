"""
Step 14 — Auto-generated claim triage (RESULTS_SUMMARY.md).
Step 15 — QA checks.
"""
from __future__ import annotations
import json, logging
from pathlib import Path
import numpy as np, pandas as pd
from .config import CFG, INTERMEDIATES_DIR, TABLES_DIR, OUTPUT_BASE, FIGURES_DIR

logger = logging.getLogger(__name__)

def run_qa_checks():
    logger.info("="*60+"\nSTEP 15: QA CHECKS\n"+"="*60)
    flags = []

    # 1. Leakage
    for csv in TABLES_DIR.glob("*.csv"):
        if "_meta" in csv.name: continue
        try:
            df = pd.read_csv(csv)
            if "effect_object_used" in df.columns:
                bad = df[df["effect_object_used"].isin(["same_sample","full_sample"])]
                if len(bad)>0:
                    flags.append(dict(check="leakage", level="ERROR", file=csv.name,
                                      message=f"Same-sample effects in {csv.name}"))
        except: pass

    # 2. Coverage
    for csv in TABLES_DIR.glob("*.csv"):
        if "_meta" in csv.name: continue
        try:
            df = pd.read_csv(csv)
            if "n_obs" in df.columns:
                zeros = df[df["n_obs"]==0]
                if len(zeros)>0:
                    flags.append(dict(check="coverage", level="WARNING", file=csv.name,
                                      message=f"Zero-observation rows in {csv.name}"))
        except: pass

    # 3. Zero-margin check
    zm_path = TABLES_DIR / "table_zero_margin_broad.csv"
    if zm_path.exists():
        zm = pd.read_csv(zm_path)
        degen = zm[zm["spec_name"]=="DEGENERATE"]
        non_degen = zm[zm["spec_name"]!="DEGENERATE"]
        if len(non_degen)==0 and len(degen)>0:
            flags.append(dict(check="zero_margin", level="ERROR", file="table_zero_margin_broad.csv",
                              message="ALL zero-margin specs are degenerate"))
        elif len(non_degen)>0:
            logger.info("  ✓ Zero-margin has %d non-degenerate specs", len(non_degen))

    # 4. Pre-trend check
    pt_path = TABLES_DIR / "table_event_study_pretrends.csv"
    if pt_path.exists():
        pt = pd.read_csv(pt_path)
        for _, r in pt.iterrows():
            if r.get("p_value",1) < 0.05:
                flags.append(dict(check="pretrend", level="WARNING", file="table_event_study_pretrends.csv",
                                  message=f"Significant pre-trend: {r.get('sample','?')} w{r.get('window','?')} p={r['p_value']:.4f}"))

    # 5. Inference check
    qb_path = TABLES_DIR / "table_quantile_clustered_primary.csv"
    if qb_path.exists():
        qb = pd.read_csv(qb_path)
        boot = qb[qb["spec_name"].str.contains("bootstrap")]
        if len(boot)>0:
            logger.info("  ✓ Bootstrap quantile inference present (%d rows)", len(boot))
        else:
            flags.append(dict(check="inference", level="WARNING", file="table_quantile_clustered_primary.csv",
                              message="No bootstrap quantile rows found"))

    # 6. Trimming sensitivity
    ts_path = TABLES_DIR / "table_trimming_sensitivity.csv"
    if ts_path.exists():
        ts = pd.read_csv(ts_path)
        if len(ts)>1:
            coefs = ts["estimate"].values
            if np.std(coefs) / np.abs(np.mean(coefs)) > 0.3:
                flags.append(dict(check="trimming", level="WARNING", file="table_trimming_sensitivity.csv",
                                  message=f"Coefficient CV > 30% across trimming specs"))
            else:
                logger.info("  ✓ Trimming sensitivity: CV = %.1f%%", 100*np.std(coefs)/np.abs(np.mean(coefs)))

    flags_df = pd.DataFrame(flags) if flags else pd.DataFrame(columns=["check","level","file","message"])
    flags_df.to_csv(TABLES_DIR/"qa_validation_flags.csv", index=False)
    n_err = len(flags_df[flags_df["level"]=="ERROR"]) if len(flags_df)>0 else 0
    n_warn = len(flags_df[flags_df["level"]=="WARNING"]) if len(flags_df)>0 else 0
    logger.info("  QA: %d errors, %d warnings", n_err, n_warn)
    return flags_df

def generate_results_summary():
    """Step 14: auto-generated claim triage."""
    logger.info("="*60+"\nSTEP 14: GENERATING RESULTS SUMMARY\n"+"="*60)
    lines = [
        "# Revision 2026 Phase 2 — Results Summary & Claim Triage", "",
        "## Claim Triage", "",
        "| Claim | Status | Key Evidence | Rating |",
        "|---|---|---|---|",
    ]

    # 1. Persistent two-sided heterogeneity
    lines.append("| Two-sided heterogeneity | See AKM/KSS benchmark (Phase 1) | Variance decomposition | `main_text_candidate` |")

    # 2. Within-captain organizational improvement
    mover_path = TABLES_DIR / "table_mover_output_preferred.csv"
    if mover_path.exists():
        m = pd.read_csv(mover_path)
        sig = m[(m["term"]=="psi_hat")&(m["p_value"]<0.05)]
        if len(sig)>0:
            best = sig.iloc[0]
            lines.append(f"| Within-captain org improvement | β={best['estimate']:.3f}, p={best['p_value']:.4f} | "
                          f"Mover design ({best['spec_name']}) | `main_text_candidate` |")
        else:
            lines.append("| Within-captain org improvement | No significant spec | | `appendix_only` |")
    else:
        lines.append("| Within-captain org improvement | Not yet run | | `pending` |")

    # 3. Floor-raising
    qb_path = TABLES_DIR / "table_quantile_clustered_primary.csv"
    if qb_path.exists():
        qb = pd.read_csv(qb_path)
        boot_psi = qb[(qb["term"]=="psi_hat_loo_std")&(qb["spec_name"].str.contains("bootstrap"))]
        if len(boot_psi)>=3:
            p10 = boot_psi[boot_psi["spec_name"].str.contains("tau10")]
            p90 = boot_psi[boot_psi["spec_name"].str.contains("tau90")]
            if len(p10)>0 and len(p90)>0:
                ratio = float(p10.iloc[0]["estimate"]) / float(p90.iloc[0]["estimate"]) if float(p90.iloc[0]["estimate"])!=0 else 0
                lines.append(f"| Floor-raising (lower tail) | P10/P90 ratio = {ratio:.1f}, bootstrap CIs | "
                              f"Quantile path + contrasts | `main_text_candidate` |")
            else:
                lines.append("| Floor-raising (lower tail) | Quantile results available | | `main_text_candidate` |")
        else:
            lines.append("| Floor-raising (lower tail) | Insufficient bootstrap results | | `appendix_only` |")

    # 4. Zero-catch extensive margin
    zm_path = TABLES_DIR / "table_zero_margin_broad.csv"
    if zm_path.exists():
        zm = pd.read_csv(zm_path)
        non_degen = zm[(zm["spec_name"]!="DEGENERATE")&(zm["outcome"]=="Pr(zero_catch)")]
        if len(non_degen)>0:
            sig = non_degen[non_degen["p_value"]<0.05]
            if len(sig)>0:
                best = sig.iloc[0]
                lines.append(f"| Zero-catch extensive margin | β={best['estimate']:.4f}, p={best['p_value']:.4f} | "
                              f"Broad sample LPM | `main_text_candidate` |")
            else:
                lines.append("| Zero-catch extensive margin | Not significant on broad sample | | `drop_or_reword` |")
        else:
            lines.append("| Zero-catch extensive margin | Degenerate on all samples tested | | `drop_or_reword` |")

    # 5. Portable routines
    pipe_path = TABLES_DIR / "table_pipeline_null_recheck.csv"
    if pipe_path.exists():
        pipe = pd.read_csv(pipe_path)
        agent_fe = pipe[pipe["spec_name"].str.contains("agent_FE")]
        if len(agent_fe)>0:
            best = agent_fe.iloc[0]
            if best["p_value"]<0.05 and best["estimate"]>0:
                lines.append("| Portable routines (pipeline) | Positive under agent FE | | `appendix_only` |")
            else:
                lines.append(f"| Portable routines (pipeline) | β={best['estimate']:.3f}, p={best['p_value']:.3f} under agent FE | "
                              "| `drop_or_reword` |")
        else:
            lines.append("| Portable routines (pipeline) | Not tested with agent FE | | `pending` |")

    # Event study
    pt_path = TABLES_DIR / "table_event_study_pretrends.csv"
    if pt_path.exists():
        pt = pd.read_csv(pt_path)
        clean = pt[pt["p_value"]>0.05]
        if len(clean)==len(pt):
            lines.append("| Event-study pre-trends | All clean (p>0.05) | | `main_text_candidate` |")
        else:
            lines.append("| Event-study pre-trends | Some significant | | `appendix_only (flag)` |")

    lines.extend(["", "## QA Summary", ""])
    qa_path = TABLES_DIR / "qa_validation_flags.csv"
    if qa_path.exists():
        qa = pd.read_csv(qa_path)
        n_err = len(qa[qa["level"]=="ERROR"]) if len(qa)>0 else 0
        n_warn = len(qa[qa["level"]=="WARNING"]) if len(qa)>0 else 0
        lines.append(f"- **Errors:** {n_err}")
        lines.append(f"- **Warnings:** {n_warn}")
        if n_err>0:
            lines.append(""); lines.append("> [!CAUTION]")
            lines.append("> There are QA errors. Review `qa_validation_flags.csv`.")

    lines.extend(["", "## Output Manifest", ""])
    for subdir in ["tables","figures","intermediates"]:
        d = OUTPUT_BASE / subdir
        if d.exists():
            files = sorted(d.glob("*"))
            if files:
                lines.append(f"### {subdir}/")
                for f in files:
                    lines.append(f"- `{f.name}` ({f.stat().st_size:,} bytes)")
                lines.append("")

    summary_path = OUTPUT_BASE / "RESULTS_SUMMARY.md"
    summary_path.write_text("\n".join(lines)+"\n")
    logger.info("  Summary saved to %s", summary_path)
