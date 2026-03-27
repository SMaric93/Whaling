# Next-Round Strengthening Tests — Interpretation

## The Big Picture

These 15 tests were designed to make the paper's central claim — that organizational capability operates through **state-contingent search governance** — harder to explain away. Here is what the numbers actually say, what they're strong on, and where the evidence is thin.

---

## 1. The Search Governance Channel Is Real and Operates Cumulatively

**Test T5 (Search vs Execution)** delivers the cleanest decomposition. At the *daily* level, ψ has essentially zero effect on any link in the production chain:

| Stage | ψ Coefficient | SE | p-value | N |
|-------|-------------|-----|---------|---|
| Encounter hazard | 0.019 | 0.022 | 0.38 | 212,036 |
| Strike \| encounter | −0.003 | 0.058 | 0.96 | 32,612 |
| Capture \| strike | 0.053 | 0.062 | 0.39 | 6,670 |
| **Voyage output** | **707.9** | **33.7** | **<10⁻⁹⁷** | **4,949** |

**Interpretation:** High-ψ agents don't make captains better whale-finders on any given day. They don't improve strike rates or capture rates. The massive voyage-level effect (ψ = 708, R² = 0.52) is entirely *cumulative*: it comes from positioning ships in the right waters at the right times, leaving barren grounds faster, and staying in productive grounds longer. This is the search governance story — not daily execution, but strategic allocation of search effort across time and space.

**Strength:** This is the single strongest result in the battery. The contrast between daily null effects and a massive voyage-level effect is hard to explain through any mechanism other than cumulative search management.

---

## 2. Captains Do Change Policy When They Switch Agents

**Test T2 (Switch Design)** uses within-captain variation across 1,915 agent-switchers (342,515 action-level observations):

- **Pre-switch exit rate:** 20.81%
- **Post-switch exit rate:** 20.15%
- **Captain FE estimate:** −0.018 (SE = 0.005, **p = 0.0009**)

The same captain, in similar navigational states, changes stopping behavior after joining a different agent. The magnitude is small (0.66pp reduction in exit propensity), but it's precisely estimated and survives within-captain demeaning.

**Caveat:** The R² is 0.001 — almost none of the variation in exit decisions is explained. The switch dummy captures a real but tiny behavioral shift. This is consistent with the story: agents shape *strategy*, which accumulates over months, not any single day's exit decision.

---

## 3. Floor-Raising Works Primarily Through Duration Control

**Test T14 (Mediation)** decomposes the total ψ → downside risk pathway:

| Mediator | a-path (ψ→M) | b-path (M→Risk) | Indirect | Share |
|----------|--------------|-----------------|----------|-------|
| Duration (days) | 350.8 | −0.000136 | −0.048 | **75.9%** |
| N grounds visited | 0.899 | −0.00387 | −0.003 | **5.5%** |

High-ψ agents extend voyages by ~351 extra days (relative to low-ψ agents, holding else equal). Longer voyages mechanically reduce the probability of falling into the bottom decile. This is descriptive mediation (sequential ignorability is not tested), but the dominance of the duration channel is striking.

**Interpretation for the paper:** The "insurance" provided by high-ψ organizations isn't magical — it's largely that they keep voyages going long enough to accumulate sufficient output. This is both a strength (concrete mechanism) and a weakness (it could be confounded with selection into longer voyages via financing).

---

## 4. θ and ψ Are Complements in Mean Output but Substitutes in the Tail

**Test T7 (Tail Submodularity)** produces the most nuanced result:

- **Linear θ×ψ interaction:** +333.5 (SE = 56.0, **p < 10⁻⁸**) → **complements** in mean output
- **Q10 slope by θ group:**
  - Low θ: ψ↔Q10 correlation = **0.453**
  - Mid θ: 0.428
  - High θ: **0.338**

In expected output, θ and ψ reinforce each other (supermodularity). But in the *left tail* (10th percentile of the outcome distribution), ψ matters *more* when θ is low. The marginal return to organizational capability is 34% higher for weak captains than for strong ones at Q10.

**For the paper:** This resolves a potential tension. The mean-output supermodularity doesn't contradict the floor-raising story — it's consistent with ψ and θ working through different mechanisms (governance vs. skill) that combine positively in expectation but where governance substitutes for skill specifically when things go badly.

---

## 5. Exit Is State-Conditional, Not Generic

**Test T9 (Rational Exit)** directly tests whether high-ψ exit behavior is "smart" or simply "more aggressive":

| Interaction | Coefficient | p-value |
|-------------|-------------|---------|
| ψ × consecutive_empty_days | +0.000369 | **0.015** |
| ψ × season_remaining | −0.000279 | 0.35 |
| ψ × scarcity | −0.007429 | 0.74 |

Only the empty-days interaction is significant. High-ψ captains become relatively more likely to exit as barren streaks lengthen — their exit policy is *contingent on accumulated negative signal*, not on generic aggressiveness or on macro-level scarcity.

**Placebo check:** Transit segments show a 62.5% "exit" rate, which is mechanical (transit → new patch). This confirms the exit variable is measuring *active search decisions*, not transit artifacts.

---

## 6. ψ Survives Hardware and Staffing Controls, With Some Attenuation

**Test T12 (Hardware Placebos):**

| Specification | ψ Coefficient | SE | R² | N |
|---------------|-------------|-----|-----|---|
| Baseline (ψ + θ only) | 1,011 | 32 | 0.39 | 10,511 |
| + tonnage, crew, age, desertion | 713 | 71 | 0.53 | 1,638 |

Adding capital and crew quality controls reduces ψ by **29.5%** but it remains highly significant. The sample drops dramatically (to 1,638) because crew data is only available for a subset.

**Interpretation:** About 30% of the organizational capability effect operates through *observable* input choices — agents who pick better ships and better crews. The remaining 70% is not explained by hardware/staffing. This is useful for the "mechanism" section: agents operate through both input selection AND governance.

---

## 7. Sorting Is Weak

**Test T8 (Risk Matching):** ρ(θ, ψ) = **−0.029**

There is essentially zero sorting between captain skill and organizational capability in the market. This has two implications:

1. **For estimation:** Weak sorting means the AKM identifying variation is clean — movers don't systematically go from bad-to-good or good-to-bad.
2. **For welfare:** The market is not exploiting the complementarity found in T7. If θ×ψ is positive in mean output (supermodular), PAM would be efficient, but the market chooses near-random matching. This suggests frictions, information asymmetry, or that matching is constrained by geography/timing.

---

## 8. Portability Is Very High (Perhaps Too High)

**Test T13 (Portability):**

| Test | AUC |
|------|-----|
| Out-of-time, high ψ | 0.945 |
| Out-of-time, low ψ | 0.963 |
| Out-of-ground | 0.988 |

Exit policies trained on one set of grounds predict nearly perfectly on held-out grounds (AUC = 0.99). This is consistent with two stories:

1. **Portable routines:** High-ψ organizations instill decision rules that generalize, or
2. **Predictable exits:** The exit decision is so mechanically determined by state variables (empty days, duration) that any model overfits trivially.

The out-of-time AUC for *low*-ψ captains is actually *higher* (0.963 vs 0.945), which is mildly inconsistent with the "high-ψ agents teach better routines" story. More likely: exit is predictable for everyone, and the ψ-specific governance shows up in the *marginal* differences, not in absolute predictability.

---

## 9. Standardization Is Weak

**Test T10 (Policy Entropy):**

| ψ Quartile | Action Entropy (bits) |
|------------|----------------------|
| Q1 (lowest) | 1.926 |
| Q2 | 1.887 |
| Q3 | **1.857** |
| Q4 (highest) | 1.914 |

The pattern is U-shaped, not monotonically decreasing. The *most* capable agents (Q4) actually show *more* action diversity than Q3. And the cross-captain action variance is only weakly negatively correlated with ψ (−0.067).

**For the paper:** The "behavioral standardization" claim is the weakest pillar. High-ψ organizations don't obviously reduce action dispersion. Instead, they may promote *state-contingent flexibility* — doing different things in different states rather than rigidly following one policy. This is consistent with the T9 finding (exit is contingent, not blanket).

---

## 10. The Map Hierarchy Tests Are Inconclusive

**Test T1 (Hierarchical Map Choice):** Basin-level (5 classes) accuracy is only 31% on the test set, and adding θ or ψ produces negligible improvement (log-loss barely moves). Theater-level (67 classes) and major-ground (496 classes) MNL models fail due to unseen classes in the temporal test split.

**For the paper:** The destination-choice design needs a different approach — the simple MNL over raw ontology levels doesn't have enough power to detect agent effects. Consider focusing on binary "Atlantic vs Pacific" or "familiar vs novel" ground choice instead.

---

## What the Paper Should Emphasize

1. **Lead with T5.** The daily-null / voyage-massive contrast is publication-ready and hard to counter.
2. **Use T2 as a causal design.** Within-captain, cross-agent variation with captain FE is the cleanest quasi-experiment.
3. **T7 resolves the substitutes/complements tension.** Mean-output complements + tail substitutes is a coherent two-part story.
4. **T14 grounds the mechanism in something concrete** (duration ≈ capital allocation).
5. **Drop or downplay the standardization claim** (T10 is ambiguous).
6. **Redesign T1** (map choice) around a simpler binary outcome variable.
7. **T3 needs HMM states** to produce transition governance results. Run the state discovery pipeline first.
