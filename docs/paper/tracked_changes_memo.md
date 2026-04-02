# Tracked-Changes Memo

*Revision: Additive AKM/KSS Benchmark Framing — Final 2026-04-01 (DOF-corrected)*

---

## Abstract

**What changed:** Completely rewritten. Numbers reflect DOF-corrected EB shrinkage. Organizational share 65.4%, captain share 33.0%, Corr(θ,ψ) = 0.017. Training agent premium 0.279 log points (t = 3.28). CATE gradient 6× (Q1 = 1.194 vs Q4 = 0.201). Variance compression 76%. Floor-raising leads with zero-catch extensive margin (82% → 16%).

**Why:** The sigma2_eps estimator was corrected from np.var(residuals) (divides by N) to SSR/(N-k). With N/k = 2.89 (many FE parameters), this 53% upward correction in noise variance materially increases EB shrinkage, particularly for captains (who average only 3.8 voyages). The shares shifted from 42/57% to 33/65%.

**Tables referenced:** Table 2, Table 7, Table 8.

---

## Introduction

**What changed:** All numbers updated to final DOF-corrected values. Training premium 0.279. CATE "6×" (was "4.5×"). Lay-system paragraph retained.

---

## Theory Development

**What changed:** Minor — Proposition 2 sign agnostic. Duan ĉ defined in Proposition 5. No stale numbers (theory is parametric).

---

## Empirical Strategy

**What changed:** LOO set 8,176/2,156/650 (from 14,617 connected voyages). EB reliabilities updated: captain 0.498 (was 0.669), agent 0.668 (was 0.777). DOF correction explicitly documented. Interaction β₃ = −0.243.

---

## Results

**What changed:** New section written from scratch with 6 subsections. All numbers from DOF-corrected pipeline.

---

## Code Fixes Applied

| File | Fix | Impact |
|------|-----|--------|
| **run_full_baseline_loo_eb.py** line 180 | sigma2_eps changed from `np.var(residuals)` to `SSR/(N-k)` | 53% increase in noise → more EB shrinkage → captain share dropped from 42% to 33% |
| **run_full_baseline_loo_eb.py** line 706 | Duan smearing uses full-model residuals, not naive y − θ − ψ | Smearing ĉ corrected from 194 → 3.23 (previous session fix retained) |
| **paper_tables.py** Table 1 footer | Correctly distinguishes full sample (14,617) from LOO (8,176) | Resolves sample mismatch |
| **paper_tables.py** Table 5 title | Changed from "Search Geometry" to "Output Around Agent Switches" | The event study actually shows log_q, not μ |
| **paper_tables.py** Table 6 footer | 0.074 → 0.279 | Matches pipeline output |
| **paper_tables.py** Table 7 footer | Old CATE text → CausalForestDML citation with (Q1−Q4)=0.993 | Matches pipeline output |

---

## Table-by-Table Status (Old → New)

| Old # | New # | Status | Key Change |
|------:|------:|--------|------------|
| Table 1 | **Table 1** | Main text | 14,617 full / 8,176 LOO |
| Table 3 | **Table 2** | Promoted | θ = 33.0%, ψ = 65.4%, Corr = 0.017 |
| *(new)* | **Table 3** | New main text | AMI + Conditional MI |
| Table 4 | **Table 4** | Main text (retitled) | Compass Effect |
| Table A3 | **Table 5** | Promoted | Title fixed: "Output Around Agent Switches" |
| Table A9 | **Table 6** | Promoted | Training premium = 0.279 (t = 3.28) |
| Table 6 | **Table 7** | Main text (reframed) | CATE Q1 = 1.194, Q4 = 0.201 (6×) |
| Table 7 | **Table 8** | Rewritten | Panel A: PAM +81% vs obs; Panel B: zero 82→16% |
| Table A1 | **Table A1** | Appendix | Org share 60–70% across specs |
| Table 2 | **Table A2** | Demoted | Raw Shannon — historical |
| Table 5 | **Table A3** | Demoted | β₃ = −0.243, suggestive only |
| Table A4 | **Table A4** | Appendix | Within-vessel underpowered |
| Table A5 | **Table A5** | Appendix | Var ratio = 0.24 (76% compression) |
| *(A5b)* | **Table A5b** | Appendix | P10/P50 ratio = 1.15 |
| Table A6 | **Table A6** | Appendix | Adaptive threshold discipline |
| Table A7 | **Table A7** | Appendix | β₃ by complexity |
| Table A8 | **Table A8** | Appendix | Context sorting, descriptive |
| Table A2 | **Table A9** | Appendix | Scarcity robustness |
| *(new)* | **Table A10** | New appendix | Lay-system audit |
| Old AAM table | **Dropped** | — | Not recomputed |

---

## Language Replacements (Global)

| Old | New |
|-----|-----|
| "Captains own the map" | Captains retain substantial routing info conditional on org assignment, but raw dominance attenuated by AMI correction |
| "Organizations provide the compass" | Organizations alter search governance conditional on deployment |
| "Submodular production implies AAM" | Additive AKM benchmark; organizational value in behavioral governance and lower-tail compression |
| "High-ψ agents fail fast" | Adaptive, threshold-dependent stopping discipline |
| "Definitively rejects hardware" | Evidence consistent with portable routines, within-vessel design underpowered |
| "42.4% captain, 57.1% org" | 33.0% captain, 65.4% org (DOF-corrected) |
| "Corr(θ,ψ) ≈ 0.00" | Corr(θ,ψ) = 0.017 |
| "CATE 4.5× (1.181 vs 0.264)" | CATE 6× (1.194 vs 0.201) |
| "PAM vs Observed +130%" | PAM vs Observed +81% |

---

## Consistency Audit — Resolved Items

1. **Table 1 sample note**: ✅ Distinguishes 14,617 (full connected) from 8,176 (LOO)
2. **Table 5 title**: ✅ Changed from "Search Geometry" to "Output Around Agent Switches"  
3. **Table 6 coefficient**: ✅ Both Panel B and footer report 0.279
4. **sigma2_eps DOF**: ✅ Uses SSR/(N-k), not SSR/N
5. **Duan smearing**: ✅ Uses full-model residuals
6. **Stale language**: ✅ Grep confirms zero forbidden phrases in manuscript files
