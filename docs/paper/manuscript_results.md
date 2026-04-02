# Results

## 5.1 Where Persistent Performance Variation Resides

### Main Decomposition (Table 2)

We begin with the additive AKM/KSS variance decomposition. Table 2 reports the baseline results using the vessel-controls specification (decade FEs + log tonnage + rig type) on the LOO connected set of 8,176 voyages.

The plug-in estimates are substantially inflated by limited-mobility bias: $\text{Var}(\hat\theta) = 2.119$ and $\text{Var}(\hat\psi) = 1.754$ (before correction). After EB shrinkage with the degrees-of-freedom corrected residual variance, the corrected estimates are $\text{Var}(\theta^{EB}) = 0.640$ and $\text{Var}(\psi^{EB}) = 1.269$, yielding variance shares of:

- **Captain heterogeneity (θ): 33.0%** of explained persistent variance
- **Organizational heterogeneity (ψ): 65.4%** of explained persistent variance
- **Sorting 2Cov(θ,ψ): 1.6%** — negligible

The sorting correlation is essentially zero: Corr(θ,ψ) = +0.017. This suggests that the captain-agent matching market in American whaling was approximately random in terms of productive sorting — stronger captains were not systematically matched with stronger agents. The implied productivity range is substantial: a one-standard-deviation increase in captain skill corresponds to roughly ±93% variation in output levels, while organizational capability implies ±125%.

The EB reliability for captains ($\bar\lambda = 0.498$) is lower than for agents ($\bar\lambda = 0.668$), reflecting the fact that captains average 3.8 voyages versus 12.6 for agents. This asymmetry in observation counts means that captain FEs are shrunk more aggressively toward the grand mean, which partly explains the higher organizational share. However, the organizational dominance is not purely mechanical: even before shrinkage, the agent plug-in variance (1.754) is 83% of the captain plug-in variance (2.119), and after DOF-corrected noise removal, the signal-to-noise ratio is more favorable for agents.

### Robustness (Table A1)

Table A1 reports the decomposition across three progressively richer specifications. The organizational share is remarkably stable: 60.4% with decade FEs only, 65.4% with vessel controls added, and 70.0% with the full control set. The R² of the full model is 0.72, indicating that the additive captain + agent + controls structure captures nearly three-quarters of outcome variation.

---

## 5.2 Where Route Information Resides

### Corrected Route-Choice Information (Table 3)

Table 3 presents the corrected route-choice information metrics. Panel A compares raw Shannon MI with adjusted MI (AMI, which penalizes high-cardinality predictors):

| Predictor | Raw MI (bits) | AMI |
|-----------|:---:|:---:|
| Captain Identity | 3.839 | 0.114 |
| Managing Agent | 2.914 | 0.182 |
| Home Port | 1.439 | 0.246 |

Raw MI assigns 83.5% of route-choice information to captain identity, but this is inflated by the high cardinality of captain identifiers. After correction, the ordering reverses: home port contributes the most AMI, followed by managing agent, with captain identity contributing the least.

Panel B reports conditional MI — the information each predictor adds given knowledge of the other:

- I(Ground; Captain | Agent) = 1.471 bits — captains retain substantial routing knowledge conditional on organizational assignment
- I(Ground; Agent | Captain) = 0.546 bits — agents contribute nontrivial routing information conditional on captain

Captains retain substantial idiosyncratic routing knowledge conditional on organizational assignment, but raw captain dominance is attenuated once high-cardinality bias is corrected. The raw Shannon table is retained in Appendix Table A2 for historical reference.

---

## 5.3 Organizations Change Search Behavior

### Within-Captain Mover Design (Table 4)

Table 4 reports the within-captain mover design, where the dependent variable is the change in search geometry (Lévy exponent μ) — a *separate outcome* from the production function. Three specifications are estimated:

- **Column (1), Baseline:** The raw coefficient on Δψ is 0.155 (p < 0.01), indicating that switching to a higher-capability agent is associated with a change in search geometry.
- **Column (2), + Route×Time FE:** After absorbing route choice via route-by-time fixed effects, the coefficient on Δψ falls to −0.004 (p < 0.05). The sign reversal after conditioning on route is interpretable: controlling for *where* the captain hunts, higher organizational capability is associated with a modest tightening of the search geometry.
- **Column (3), + Hardware Controls:** Adding vessel tonnage and rig controls barely changes the estimate (−0.003, p < 0.05), ruling out that the within-captain shift is driven by changes in vessel quality.

The key result is that the Δψ coefficient remains statistically significant across all specifications. Because the outcome is search geometry (not output), this design is not vulnerable to the same-outcome critique that applies to θ×ψ interaction specifications.

### Event Study (Table 5)

Table 5 provides the event-study complement. Mean log output is reported by event time relative to the agent-switch year, for 3,255 observed switches:

| Event Time | Mean log_q | SE | N |
|---|---|---|---|
| t−2 (Pre-Trend) | 5.720 | (0.062) | 2,069 |
| t−1 (Pre-Trend) | 5.595 | (0.051) | 3,255 |
| t=0 (Switch Year) | 5.586 | (0.049) | 3,255 |
| t+1 (Persistence) | 5.641 | (0.061) | 1,947 |
| t+2 (Persistence) | 5.553 | (0.079) | 1,204 |

The pre-trend period (t−2, t−1) shows no systematic ramp-up or decline prior to the switch, supporting the parallel-trends assumption.

---

## 5.4 Organizations Transmit Routines

### Mate-to-Captain Training Pipeline (Table 6)

Table 6 provides the most direct evidence that organizations transmit portable routines. Panel A reports the mate fixed-effect variance decomposition: between-mate variance accounts for 87.6% of total mate-voyage variation (N = 2,400 voyages, 1,985 unique mates), indicating that mate identity is a strong predictor of voyage-level outcomes.

Panel B reports the training agent premium. Among 800 promoted mates (captains who served as first mate before being promoted), we observe 1,692 captain voyages with known training-agent identity. Of these, 195 were sailed with the original training agent and 1,497 with a different agent. The coefficient on "Same Agent" is:

$$
\hat\beta = 0.279 \quad (SE = 0.085, \quad t = 3.28, \quad p < 0.01)
$$

Captains perform significantly better — 0.279 log points, or approximately 32% higher output — when sailing with the agent who originally trained them.

---

## 5.5 Floor-Raising and Downside-Risk Compression

### Heterogeneous Returns by Captain Quartile (Table 7)

Table 7 reports CausalForestDML estimates of the conditional average treatment effect of organizational capability by captain skill quartile:

| Captain Quartile | Mean θ | CATE of ψ | Mechanism |
|---|---|---|---|
| Q1 (Novice) | −3.70 | 1.194*** | Insurance / Floor Raising |
| Q2 | −2.91 | 0.743*** | Transition |
| Q3 | −2.56 | 0.464*** | Transition |
| Q4 (Expert) | −1.99 | 0.201*** | Diminishing Returns |

The CATE gradient is steep and monotonic: novice captains (Q1) benefit roughly 6× as much from organizational capability as expert captains (Q4). The difference (Q1 − Q4) = 0.993 is economically large and statistically significant.

### Insurance Variance Validation (Table A5)

Table A5 cross-tabulates outcome distributions by captain experience × organizational capability. The headline result: for novice captains matched with low-capability organizations, the outcome standard deviation is 3.19 log points and P10 = 0.0 — reflecting a large mass of zero-catch voyages. Switching novice captains to high-capability organizations reduces the standard deviation to 1.56 and raises P10 to 5.53, a **76% compression in variance** (variance ratio = 0.24).

### Quantile Regression Confirmation (Table A5b)

Quantile regressions confirm the floor-raising pattern. The ψ coefficient is:
- **P10: 1.170*** (15% larger than median)
- **P50: 1.014*** (baseline)
- **P90: 0.751*** (26% smaller than median)

The organizational effect is strongest at the bottom of the outcome distribution.

---

## 5.6 Matching Under Additive Means and Floor-Raising Risk

### Panel A: Mean-Output Allocation (Table 8)

Under the additive AKM benchmark in logs, production in levels is multiplicatively separable: $\hat{Q}_{is} = \exp(\hat\theta_i) \times \exp(\hat\psi_s) \times \hat{c}$, where $\hat{c}$ = 3.23 (Duan 1983 smearing correction computed from the full-model residuals). Mean-output-maximizing assignment is therefore PAM.

Table 8 Panel A reports within-decade counterfactual mean predictions:

| Assignment | Mean Q̂ (level) | Δ vs Random |
|---|---|---|
| Observed | 0.86 | −10.9% |
| Random | 0.97 | ref |
| PAM | 1.56 | +60.8% |

Observed assignment produces 10.9% less mean output than random assignment, and PAM would increase mean output by 81% over observed (61% over random). The absolute levels reflect normalized FEs (centered at zero); the percentage comparisons are the informative objects.

### Panel B: Risk-Based Allocation (Table 8)

Table 8 Panel B documents the floor-raising margin of assignment by focusing on Q1 (novice) captains:

| Cell | N | Zero-Catch Rate | Cond. P10 (log) |
|---|---|---|---|
| Q1 × Low-ψ | 1,242 | **82%** | 3.66 |
| Q1 × High-ψ | 803 | **16%** | 4.62 |

High-capability organizations reduce the zero-catch rate for novice captains from 82% to 16% — a 66-percentage-point reduction. This is where the paper's most distinctive matching contribution lies.

### Stopping-Rule Evidence (Table A6)

The stopping-rule evidence supports adaptive, threshold-dependent organizational discipline rather than universal fail-fast behavior. Table A6 shows that high-ψ organizations shorten overall patch residence time (coefficient = −0.096, p < 0.01), but the ψ × empty interaction is *positive* (+0.154, p < 0.01): high-capability organizations impose more discriminating stopping — faster exit in truly barren states, but not blind impatience in merely low-yield states.

### Supplementary: θ×ψ Interaction (Table A3)

Table A3 reports the interaction production function as suggestive supplementary evidence. The estimated β₃ is −0.243 (pooled, SE = 0.023), indicating negative complementarity (substitutability) in means. This is consistent with the floor-raising pattern documented in Table 7 — organizations add more value for weaker captains — but should not be interpreted as the main structural production function due to generated-regressor concerns.
