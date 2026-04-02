# Matching Note: Mean-Allocation vs. Risk-Allocation

*Technical note for the revised matching section — Updated 2026-04-01 (DOF-corrected)*

---

## 1. The Additive Mean Benchmark in Levels

### Setting

The AKM/KSS variance decomposition establishes that mean log productivity is well approximated by:

$$
\ln Q_{is} = \hat\theta_i + \hat\psi_s + X_{is}'\hat\beta + \varepsilon_{is}
$$

where $\hat\theta_i$ is the captain fixed effect, $\hat\psi_s$ is the organizational (agent) fixed effect, and $X_{is}$ includes decade fixed effects and vessel controls (log tonnage, rig type) residualized via FWL. In the baseline EB/KSS decomposition (with DOF-corrected noise), captain heterogeneity explains 33.0% of persistent variance, organizational heterogeneity explains 65.4%, and the sorting covariance accounts for 1.6%, with Corr($\hat\theta$, $\hat\psi$) = 0.017 (negligible sorting).

### The Level Problem

Under the additive specification in logs, the mean-output matching problem requires converting to levels because the planner's objective is total output, not mean log output. The correct predicted level output under match ($i$, $s$) is:

$$
\hat{Q}_{is} = \exp\!\big(\hat\theta_i + \hat\psi_s + X_{is}'\hat\beta\big) \times \hat{c}
$$

where $\hat{c}$ is the Duan (1983) smearing correction factor:

$$
\hat{c} = \frac{1}{N} \sum_{j=1}^{N} \exp(\hat\varepsilon_j)
$$

**Critical implementation note:** The residuals $\hat\varepsilon_j$ must be the **full-model** residuals $y - \hat{y}$, not the naive $y - \hat\theta - \hat\psi$ (which omits decade FEs and controls). In the current pipeline, $\hat{c}$ = 3.23, computed from the LSQR model residuals with mean ≈ 0 and std ≈ 1.57.

### Implications for Matching in Levels

Under the additive AKM in logs, the production function in levels is **multiplicatively separable**:

$$
\hat{Q}_{is} = \underbrace{e^{\hat\theta_i}}_{\text{captain type}} \times \underbrace{e^{\hat\psi_s}}_{\text{org type}} \times \underbrace{e^{X_{is}'\hat\beta}\hat{c}}_{\text{controls + correction}}
$$

This means:
- **Total output** is the product of captain productivity $e^{\hat\theta_i}$ and organizational productivity $e^{\hat\psi_s}$.
- Under multiplicative separability, the **mean-output-maximizing assignment is PAM** (positive assortative matching): pair the highest captain type with the highest organizational type.

### Counterfactual Comparisons for the Mean Margin

| Assignment Rule | Mean Q̂ (level) | Δ vs Random | Interpretation |
|------|:---:|:---:|----------------|
| **Observed** | 0.86 | −10.9% | Actual captain-agent pairings |
| **Random** | 0.97 | ref | Within-decade permutation benchmark |
| **PAM** | 1.56 | +60.8% | Mean-output-maximizing under separability |

The absolute levels are normalized (FEs centered); percentage comparisons are the informative objects. PAM gains of +81% over observed (or +61% over random) reflect the convexity of exp(): reassigning captains to their output-maximizing agent concentrates the exponential tail.

### What NOT to Claim

- Do **not** claim that the old θ×ψ interaction structurally proves mean AAM. Under the additive AKM benchmark, the interaction term is zero by construction in the first-order model.
- Do **not** interpret the old sparse-ground AAM result as universal structural evidence.

---

## 2. The Risk-Based Matching Margin

### Motivation

The floor-raising result (Table 7) shows that organizational capability has roughly **6× larger effects for Q1 (novice) captains** than for Q4 (expert) captains:

| Captain Quartile | Mean θ | CATE of ψ | Mechanism |
|---|---|---|---|
| Q1 (Novice) | −3.70 | 1.194*** | Insurance / Floor Raising |
| Q2 | −2.91 | 0.743*** | Transition |
| Q3 | −2.56 | 0.464*** | Transition |
| Q4 (Expert) | −1.99 | 0.201*** | Diminishing Returns |

Additionally, the insurance variance validation (Table A5) shows that high-ψ organizations compress the outcome variance for novice captains by **76%** (variance ratio = 0.24 relative to baseline).

### The Risk-Management Margin

When organizational capability disproportionately benefits weaker captains, the assignment problem has a **second margin**: risk management. A utilitarian planner with a concave social welfare function would want to allocate stronger organizations to weaker captains, even though this reduces mean output relative to PAM.

### Floor-Raising Evidence (Table 8, Panel B)

The most striking floor-raising evidence is at the extensive margin:

| Cell | N | Zero-Catch Rate | Cond. P10 (log) |
|---|---|---|---|
| Q1 × Low-ψ | 1,242 | **82%** | 3.66 |
| Q1 × High-ψ | 803 | **16%** | 4.62 |

High-capability organizations reduce the zero-catch rate for novice captains from 82% to 16% — a 66-percentage-point reduction.

### Supporting Evidence

1. **Quantile regression** (Table A5b): The ψ effect is 15% larger at P10 than at the median (ratio = 1.15), confirming floor-raising operates through variance compression.
2. **Insurance variance cells** (Table A5): High-ψ agents compress variance by 76% for novices (Var ratio = 0.24 vs. 1.00 baseline).
3. **CATE gradient**: The 6× ratio (Q1 vs Q4) is the key primitive that makes risk-based matching welfare-improving.

---

## 3. Which Counterfactuals Are Structural vs. Descriptive

| Counterfactual | Type | Justification |
|---|---|---|
| PAM in levels (mean-optimal) | **Structural benchmark** | Follows directly from the additive AKM in logs → multiplicatively separable production in levels |
| Random assignment | **Structural benchmark** | Model-free; no assumptions beyond additive FEs |
| Treatment-cell comparisons (Q1×ψ) | **Descriptive evidence** | Documents floor-raising at the extensive and intensive margins |
| CVaR-optimal assignment | **Descriptive planner exercise** | Depends on the assumed risk-aversion parameter λ and on the estimated CATE gradient |
| Old sparse-vs-rich AAM table | **Suggestive heuristic** | Built on the θ×ψ interaction design, which is no longer the main structural model |

### Bottom Line

The mean-allocation margin is well-identified under the additive AKM: multiplicative separability in levels implies PAM. The risk-allocation margin is supported by the CATE gradient and variance-compression evidence, but the optimal assignment depends on the planner's risk preferences (λ), which are not directly estimated. The paper's contribution is to establish that both margins exist and that the risk margin is quantitatively important for weak captains — organizations raise the floor more than the ceiling.
