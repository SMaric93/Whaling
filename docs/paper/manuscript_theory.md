# Theory Development

## 2.1 Setting

Consider a market with captains $i \in \{1, \ldots, I\}$ and managing agents (organizations) $s \in \{1, \ldots, S\}$. Captain $i$ matched with organization $s$ on voyage $v$ produces output:

$$
\ln Q_{isv} = \theta_i + \psi_s + X_{isv}'\beta + \varepsilon_{isv}
$$

where $\theta_i$ is persistent captain heterogeneity, $\psi_s$ is persistent organizational heterogeneity, $X_{isv}$ are time-varying controls (decade fixed effects and vessel characteristics), and $\varepsilon_{isv}$ is a mean-zero idiosyncratic shock. This additive specification is the benchmark model. Richer specifications — including interaction terms or nonlinear functions of $(\theta, \psi)$ — are possible but are treated as supplementary rather than first-order.

---

## Proposition 1: Additive Persistent Heterogeneity

*Both captain heterogeneity and organizational heterogeneity contribute substantially and persistently to productivity.*

Under the additive model, the variance of log output decomposes as:

$$
\text{Var}(\ln Q) = \text{Var}(\theta) + \text{Var}(\psi) + 2\text{Cov}(\theta, \psi) + \text{Var}(X'\beta) + \text{Var}(\varepsilon)
$$

Proposition 1 states that $\text{Var}(\theta) > 0$ and $\text{Var}(\psi) > 0$ after bias correction, and that both effects are portable across matches (i.e., they survive out-of-sample validation). The proposition is agnostic about the sign or magnitude of $\text{Cov}(\theta, \psi)$.

**Testable prediction:** The AKM/KSS decomposition should attribute a nontrivial share of explained variance to both captain and organizational effects, and these effects should predict outcomes in out-of-sample validation exercises.

---

## Proposition 2: Portable Routines

*Organizations alter search governance conditional on deployment, and these effects are detectable through within-captain variation.*

Let $\mu_{isv}$ denote the search geometry (Lévy exponent) on voyage $v$. Proposition 2 states that:

$$
\frac{\partial \mu_{isv}}{\partial \psi_s} \neq 0 \quad \text{(conditional on } \theta_i, X_{isv}\text{)}
$$

The sign and interpretation are empirical: if higher organizational capability alters search patterns within captain, then the organizational effect reflects behavioral governance rather than pure selection. Critically, this is estimated within captain — holding $\theta_i$ fixed by using captain fixed effects — so it cannot be explained by selection of more capable captains into better organizations.

**Testable prediction:** Within-captain mover designs should show that switching to a different-$\psi$ agent changes $\mu$, with no pre-trends in event-study specifications.

**Interpretation:** Organizations alter search behavior conditional on deployment. This is behavioral governance: the same captain, on a potentially different vessel, changes search patterns when the organizational environment changes.

---

## Proposition 3: Training-Pipeline Transmission

*Organizations transmit portable routines through mentoring relationships, and the effect is detectable in downstream captain performance.*

Let $s_0(i)$ denote the organization under which captain $i$ originally served as first mate. Proposition 3 states that, conditional on captain ability:

$$
E[\ln Q_{isv} \mid s = s_0(i)] > E[\ln Q_{isv} \mid s \neq s_0(i)]
$$

Captains who continue sailing with their training organization perform better, even after controlling for captain fixed effects and voyage characteristics. The magnitude of this "training agent premium" reflects the degree to which organizational routines are portable and specific to the captain-organization relationship established during the mentoring period.

**Testable prediction:** Promoted mates should perform significantly better when sailing with their original training agent compared to sailing with a different agent.

**Interpretation:** This is the most direct test that organizations transmit *software* (routines, decision rules, informational resources) rather than merely providing *hardware* (better vessels, larger crews). The training pipeline embeds organizational knowledge in the captain's human capital.

---

## Proposition 4: Floor-Raising

*Organizational capability disproportionately benefits weaker captains, compressing the lower tail of the outcome distribution.*

Let $\tau(\theta_i, \psi_s)$ denote the conditional average treatment effect of organizational capability for a captain of type $\theta_i$:

$$
\tau(\theta_i, \psi_s) = E[\ln Q_{isv} \mid \psi_s = \text{high}] - E[\ln Q_{isv} \mid \psi_s = \text{low}]
$$

Proposition 4 states that $\frac{\partial \tau}{\partial \theta_i} < 0$: the marginal value of organizational capability is declining in captain skill. In the extreme, this means that organizations "raise the floor" for weak captains — reducing the probability of catastrophically bad outcomes — while adding less at the top of the skill distribution.

**Testable predictions:**
1. CATE should be largest for Q1 (novice) captains and smallest for Q4 (expert) captains.
2. The ψ coefficient in quantile regressions should be larger at lower quantiles (P10) than at higher quantiles (P90).
3. High-ψ organizations should compress outcome variance for low-θ captains.

**Interpretation:** This proposition does not require a globally submodular mean production function. It is a distributional result: organizations provide the most value where uncertainty is highest and captain skill is lowest.

---

## Proposition 5: Two-Margin Assignment

*Captain-organization assignment has two margins: a mean-output margin and a risk-management margin, and their implications for optimal matching differ.*

Under the additive AKM benchmark in logs, the production function in levels is multiplicatively separable:

$$
Q_{is} = e^{\theta_i} \cdot e^{\psi_s} \cdot e^{X_{is}'\beta} \cdot c
$$

where $c = E[\exp(\varepsilon)]$ is the Duan (1983) smearing correction. This implies:

**Mean-output margin (P5a):** The mean-output-maximizing assignment is PAM (positive assortative matching): pair the highest captain type $e^{\theta_i}$ with the highest organizational type $e^{\psi_s}$.

**Risk-management margin (P5b):** When organizational capability disproportionately benefits weaker captains (Proposition 4), a planner with a concave welfare function or with a downside-protection objective would prefer to allocate stronger organizations to weaker captains. Specifically, under the certainty-equivalent objective:

$$
CE_{is} = E[Q_{is}] - \frac{\lambda}{2}\text{Var}(Q_{is})
$$

the optimal assignment transitions from PAM (when $\lambda = 0$) toward AAM (anti-assortative matching, when $\lambda$ is large) as the planner's risk aversion increases.

**Testable prediction:** The risk-management margin should be quantitatively important for Q1 captains, where organizational capability has the largest floor-raising effects and variance-compression benefits.

**Interpretation:** Matching matters not only because organizations can raise mean output by pairing with better captains, but also because organizations stabilize weak captains. The second margin is the paper's most distinctive matching contribution.
