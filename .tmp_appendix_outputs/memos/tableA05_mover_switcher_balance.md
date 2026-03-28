# tableA05_mover_switcher_balance

Supports: Table 5.

Sample: Connected-set voyage sample and the shipped switch-policy output.. Unit: Voyage or switcher captain, depending on row.. Types: theta_hat and psi_hat are connected-set voyage types.. FE: Captain fixed effects for the shipped switch-policy FE row; descriptive balance elsewhere.. Clustering: Captain clustering in the shipped FE output.. Controls: Core voyage observables only in the balance rows.. Interpretation: Switchers and vessel movers are observable subsets of the connected sample, and the shipped switch-policy effect is materially different from a placebo date shift.. Caution: The placebo switch-date row is a conservative paper-layer benchmark constructed from the shipped FE effect rather than a full re-estimated switch design..

Implementation notes:
- Panels A-B compare switchers and movers to their complements.
- Panels C-D summarize the observed switch effect, a placebo-date benchmark, and switchback prevalence.
