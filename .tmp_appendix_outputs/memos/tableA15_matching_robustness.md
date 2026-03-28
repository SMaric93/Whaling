# tableA15_matching_robustness

Supports: Table 10.

Sample: Shipped next-round and reinforcement matching outputs plus the connected-set voyage sample.. Unit: Assignment or robustness row.. Types: theta_hat and psi_hat are the connected-set types used in the paper-layer matching surface.. FE: As inherited from the shipped reinforcement submodularity output; none in the paper-layer bootstrap rows.. Clustering: Captain clustering in the reinforcement submodularity output.. Controls: Scarcity enters the paper-layer matching surface and the shipped submodularity tables.. Interpretation: The tail-risk matching results are directionally robust across shipped counterfactuals, observed-support checks, and paper-layer bootstrap uncertainty.. Caution: The paper-layer bootstrap resamples voyages from the observed support and is not a full equilibrium reassignment uncertainty analysis..

Implementation notes:
- Panel C adds the missing uncertainty and support diagnostics around the matching exercise.
