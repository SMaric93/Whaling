# Empirical Strategy

## 4.1 AKM/KSS/EB: The Main Benchmark

### Specification

The first-order empirical model is the additive AKM specification:

$$
\ln Q_{isv} = \theta_i + \psi_s + X_{isv}'\beta + \varepsilon_{isv}
$$

where $\theta_i$ is a captain fixed effect, $\psi_s$ is an agent (organizational) fixed effect, $X_{isv}$ includes decade fixed effects (residualized via FWL/Frisch-Waugh-Lovell) and vessel controls (log tonnage, rig type), and $\varepsilon_{isv}$ is the idiosyncratic error. The model is estimated by sparse least squares (LSQR) on the leave-one-out (LOO) connected set.

### Connected Set and Identification

Identification of both $\theta$ and $\psi$ requires that captains move between agents. We construct the LOO connected set by iteratively removing captains who are articulation points in the bipartite captain-agent graph (KSS algorithm). This yields 8,176 voyages from 2,156 connected captains and 650 connected agents (from a full connected sample of 14,617 voyages). The LOO pruning ensures that each captain's fixed effect remains identified even after removing any single observation.

### Bias Correction

Plug-in variance estimates of $\text{Var}(\hat\theta)$ and $\text{Var}(\hat\psi)$ are biased upward due to limited-mobility bias (Andrews et al. 2008). We apply two corrections:

1. **KSS correction** (Kline, Saggio, and Sølvsten 2020): leverages the leave-one-out structure for unbiased variance estimation.
2. **Empirical Bayes shrinkage**: applies James-Stein shrinkage to fixed-effect estimates, where the shrinkage factor $\lambda_c = \frac{\sigma^2_\theta}{\sigma^2_\theta + \sigma^2_\varepsilon / n_c}$ downweights captains with few voyages. Mean reliability is $\bar\lambda_\text{captain} = 0.498$ and $\bar\lambda_\text{agent} = 0.668$ in the baseline specification. The residual variance $\sigma^2_\varepsilon$ uses the degrees-of-freedom corrected estimator $\text{SSR}/(N-k)$.

### Why This Benchmark

The additive AKM is the appropriate first-order benchmark because:

1. **It avoids generated-regressor circularity.** Interaction designs (e.g., $\theta \times \psi$ in a production function) use estimated fixed effects as regressors in a second-stage equation, inheriting estimation error from the first stage. The additive decomposition does not require a second stage.
2. **It is standard in the employer-employee literature.** The AKM/KSS/EB toolkit is the established benchmark for separating worker and firm effects (Card, Heining, and Kline 2013; Song et al. 2019; Bonhomme et al. 2023).
3. **It organizes the key descriptive facts.** The variance shares and sorting correlation provide a transparent summary of how much captain skill and organizational capability each matter for mean productivity.

---

## 4.2 Compass Effect: Separate-Outcome Behavioral Design

### Why a Separate Outcome

The compass effect — the claim that organizations alter search governance — is estimated on **search geometry** (Lévy exponent μ), not on output. This design choice is critical because:

- A within-captain regression of **output** on Δψ is vulnerable to the critique that the estimated ψ already captures the agent's mean output contribution (mechanical correlation).
- A within-captain regression of **search geometry** on Δψ tests whether organizations change *behavior*, not whether they change *outcomes*. The behavioral channel is conceptually distinct from the mean-production channel.

### Mover Design

For captains who switch agents between consecutive voyages:

$$
\Delta\mu_{iv} = \beta \cdot \Delta\psi_{iv} + \gamma_{rt} + \delta X_{iv} + u_{iv}
$$

where $\Delta\mu_{iv}$ is the change in search geometry, $\Delta\psi_{iv}$ is the change in agent capability, $\gamma_{rt}$ are route-by-time fixed effects, and $X_{iv}$ includes vessel tonnage and rig controls. The coefficient $\beta$ captures the within-captain effect of organizational capability on search behavior.

### Event Study

To assess the dynamics:

$$
\ln Q_{iv} = \alpha_i + \sum_{k=-2}^{+2} \beta_k \cdot \mathbf{1}[t = t^*_i + k] + \gamma_{rt} + u_{iv}
$$

where $t^*_i$ is the year captain $i$ switches agents. Pre-trend coefficients ($\beta_{-2}$, $\beta_{-1}$) should show no systematic deviation from zero; the treatment effect ($\beta_0$) and persistence ($\beta_1$, $\beta_2$) should be detectable.

---

## 4.3 Training-Pipeline Identification

### Design

Among captains who were previously first mates, we compare performance when sailing with the agent who originally trained them versus a different agent:

$$
\ln Q_{iv} = \alpha_i + \beta \cdot \mathbf{1}[\text{Same Agent as Training}] + X_{iv}'\gamma + u_{iv}
$$

The coefficient $\beta$ captures the training agent premium — the additional benefit of organizational routines for captains who were socialized within that organization.

### Identifying Variation

This design exploits within-captain variation: the same captain sometimes sails with the agent who trained him and sometimes with a different agent. Captain fixed effects $\alpha_i$ absorb all persistent captain characteristics, so $\beta$ is identified from the difference in outcomes when the organizational relationship changes.

---

## 4.4 Floor-Raising Heterogeneity

### CATE Estimation

We estimate conditional average treatment effects by captain skill quartile:

$$
\hat\tau(\theta_q) = E[\ln Q \mid \psi = \text{high}, \theta \in Q_q] - E[\ln Q \mid \psi = \text{low}, \theta \in Q_q]
$$

using CausalForestDML (Chernozhukov et al. 2018) with random forest nuisance models (200 trees, minimum leaf size 20). The CausalForest approach provides doubly-robust estimates that are less sensitive to functional-form assumptions than OLS-by-quartile.

### Variance Compression

We complement the CATE with direct variance comparisons across treatment cells (captain quartile × agent median split) and quantile regressions of $\ln Q$ on $\hat\psi$ at τ = {0.10, 0.25, 0.50, 0.75, 0.90}. If floor-raising is present, the ψ coefficient should be largest at lower quantiles.

---

## 4.5 Matching Under Mean-vs-Risk Objectives

### Level-Based Counterfactuals

Under the additive AKM in logs, predicted output in levels is:

$$
\hat{Q}_{is} = \exp(\hat\theta_i + \hat\psi_s + X_{is}'\hat\beta) \times \hat{c}
$$

where $\hat{c} = \frac{1}{N}\sum_{j} \exp(\hat\varepsilon_j)$ is the Duan (1983) smearing correction computed from the full model residuals. The smearing constant accounts for the fact that $E[\exp(\varepsilon)] \neq \exp(E[\varepsilon])$. We compare:

1. **Observed assignment** vs. **Random assignment** (within-decade permutation baseline)
2. **PAM** (mean-output-maximizing in levels) vs. **Observed**
3. Treatment-cell comparisons: **Q1 captains × Low-ψ** vs. **Q1 captains × High-ψ**

### Descriptive vs. Structural

The PAM and random-assignment counterfactuals are structural (they follow directly from the additive model). Risk-based treatment-cell comparisons are descriptive — they document that high-ψ organizations compress the zero-catch rate and raise the conditional P10 for novice captains.

---

## 4.6 Supplementary: θ×ψ Interaction (Appendix)

The interaction specification:

$$
\ln Q_{isv} = \beta_1\hat\theta_i + \beta_2\hat\psi_s + \beta_3(\hat\theta_i \times \hat\psi_s) + X_{isv}'\gamma + u_{isv}
$$

is retained in the appendix as suggestive supplementary evidence. The estimated $\beta_3$ is negative and statistically significant (pooled: $-0.243$, SE = 0.023), but should not be interpreted as the main structural production function because:

1. $\hat\theta$ and $\hat\psi$ are generated regressors with estimation error that propagates into the interaction term.
2. The interaction design uses the same outcome (output) to estimate the effects and to test the interaction, creating a mechanical relationship.
3. The additive AKM benchmark makes the interaction term zero by construction in the first-order model.

---

## 4.7 Route-Choice Information (Descriptive)

Route-choice mutual information is presented using corrected adjusted mutual information (AMI) and conditional mutual information rather than raw Shannon MI. The raw MI is retained in the appendix (Table A2) for historical continuity, but AMI and conditional MI are the inferentially relevant objects:

- **Raw MI**: Captain 3.839 bits, Agent 2.914, Port 1.439 — inflated by high-cardinality captain identity.
- **AMI**: Captain 0.114, Agent 0.182, Port 0.246 — penalizes high-cardinality predictors.
- **Conditional MI**: I(G;Captain|Agent) = 1.471 vs. I(G;Agent|Captain) = 0.546 — captains retain substantial routing knowledge conditional on agent, but agent effects are also nontrivial.

Frequency-restricted robustness (captains with ≥2 and ≥3 voyages, agents with ≥5 voyages) confirms that captains retain more conditional routing information than agents across subsamples.
