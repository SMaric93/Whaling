# table10_tail_matching

Sample: Connected-set voyages with action-derived dry-spell measures for the tail-risk outcomes.. Unit: Voyage throughout.. Types: theta_hat and psi_hat are connected-set AKM types. Panel C rebuilds planner-style matching summaries over the observed voyage pool using a predicted output surface.. FE: No fixed effects in the main interaction rows.. Clustering: Captain clustering for Panel A. Panel B reports descriptive sorting moments. Panel C is a counterfactual welfare summary.. Controls: Scarcity is included in all interaction models, with an explicit theta-hat × psi-hat × scarcity row for mean output.. Interpretation: The connected voyage sample supports a tail-risk matching narrative better than a pure mean-output matching narrative: organizational value shows up most clearly in downside protection and in the interaction between captain skill and agent capability under risk.. Caution: Panel C is a constrained planner exercise over the observed voyage pool using a linear predicted output surface; it is not a full equilibrium assignment model..

Implementation notes:
- Panel A is rebuilt directly from the connected voyage sample and adds explicit interaction rows for mean output, scarcity, and tail-risk outcomes.
- Panel B mirrors the sorting-moment logic from Table 2 but presents the manuscript-facing breakdown used in the matching section.
- Panel C compares observed, PAM, and AAM/NAM assignments under a linear predicted output surface and then selects the best candidate assignment under mean, CVaR, and certainty-equivalent objectives.
- The expected-shortfall row uses the first nondegenerate lower-tail cutoff so the downside-severity measure remains informative when the raw 10th percentile output is zero.
