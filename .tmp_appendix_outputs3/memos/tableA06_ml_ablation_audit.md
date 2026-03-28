# tableA06_ml_ablation_audit

Supports: Tables 3, 6, and 8.

Sample: Shipped ML benchmark and ablation outputs from `outputs/tables/ml`.. Unit: Model-task evaluation row.. Types: Where present, theta_hat and psi_hat are used as predictive features only.. FE: None.. Clustering: None; predictive benchmark outputs only.. Controls: As encoded in the upstream ML feature sets and ablations.. Interpretation: The repository's ML layer is reusable as predictive support, and this appendix exposes the ablations without elevating them to identification evidence.. Caution: Performance differences are predictive, not causal, and feature sets differ across tasks..

Implementation notes:
- The appendix table concatenates the shipped ML benchmark files directly.
- These outputs are support-only and are not used as the paper's identification backbone.
