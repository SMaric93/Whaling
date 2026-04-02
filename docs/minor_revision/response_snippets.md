# Response Snippets for Minor Revision

## Response Snippet 1: Route-Choice Information Table

### What changed in Table 2

1. **Adjusted Mutual Information (AMI)**: Replaced raw in-sample Shannon MI with
   AMI (sklearn `adjusted_mutual_info_score`), which corrects for the inflated
   information scores produced by high-cardinality predictors like captain identity.

2. **Conditional Mutual Information**: Added I(Ground; Captain | Agent) and
   I(Ground; Agent | Captain) to separate captain-specific routing knowledge from
   agent portfolio effects. Computed using smoothed empirical joint counts
   (Dirichlet α=1.0) with bootstrap CIs.

3. **Out-of-sample benchmark**: Added Table 2D comparing multinomial logistic
   regression predictions with and without captain/agent identifiers, using a
   time split at 1870. Reports log loss, top-3 accuracy, and deviance improvement.

4. **Frequency-restricted robustness**: All metrics recomputed on subsamples
   restricted to captains with ≥2 and ≥3 voyages, agents with ≥5 voyages,
   and repeated captain-agent pairs.

### Interpretation
The raw MI is retained for reproducibility but is no longer presented as the primary
metric. AMI and conditional MI provide a mathematically defensible decomposition
that separates captain-specific routing skill from agent portfolio effects.

---

## Response Snippet 2: Stopping-Rule Sensitivity

### What changed in the stopping appendix

1. **Threshold curve added**: Appendix Figure A13 shows the coefficient on
   ψ × empty_patch across empty-patch percentile cutoffs from the 5th to the
   50th percentile of estimated patch yield.

2. **Preferred threshold indicated**: A vertical line marks the main-text cutoff
   (25th percentile, bottom quartile) used in Table A6.

3. **Models reported**:
   - Panel A: OLS on log(patch residence time) with 95% CIs
   - Panel B: Share and count of patches classified as empty at each cutoff
   - Panel C: AFT Weibull coefficient path for robustness

### Interpretation
The interaction coefficient is shown to be robust across
the range of thresholds examined. The sign, magnitude, and statistical significance
of the "fail fast" discipline effect do not depend on the specific percentile
cutoff used to define empty patches.

---

## Response Snippet 3: Lay-System / Incentive Alternative

### What changed on lay contracts

**Coverage audit**: No lay/incentive contract variables exist in the dataset.

1. **Systematic search**: Searched for `captain_lay`, `mate_lay`, `crew_lay`,
   `share_fraction`, `articles_of_agreement`, and related variable names across
   all data files in the repository. None were found.

2. **Institutional evidence**: American whaling lay shares were standardized by
   rank, port, and era (Davis, Gallman & Gleiter 1997). Within port-era cells,
   lay fractions varied minimally — agents competed on voyage selection and vessel
   quality, not on incentive generosity.

3. **Conclusion**: Observed lay standardization limits the plausibility of lay
   as the main omitted variable driving agent capability effects. The agent effect
   (ψ) operates through channels other than differential incentive contracts.

---

## Response Snippet 4: Table 1 / Table 3 Scale Fix

### Exact source of mismatch

Sample size differs by 4,988 voyages (Table 1: 10,973, Table 3: 5,985). Mean differs by 0.1700 log points. SD differs by 0.0606. 

### Corrected values

| Statistic | Table 1 (full sample) | Table 3 (connected set) |
|-----------|----------------------:|------------------------:|
| N | 10,973 | 5,985 |
| Mean | 6.6681 | 6.8381 |
| SD | 1.2057 | 1.1451 |
| Var | 1.4538 | 1.3113 |

### Fix applied
1. Table 1 now uses live-computed statistics from the analysis sample.
2. Table 3 notes explicitly state the sample is the LOO connected set.
3. Scale consistency assertions added (`SD² ≈ Var`).
