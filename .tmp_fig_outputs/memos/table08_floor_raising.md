# table08_floor_raising

Sample: Connected-set voyages with action-derived dry-spell measures merged from the daily action panel.. Unit: Voyage throughout.. Types: theta_hat and psi_hat are connected-set AKM types; Panel A uses a high-psi indicator, while Panels B and C use continuous psi or high-vs-low psi gaps as noted.. FE: No fixed effects in the main table; within-captain dispersion in Panel C uses captain-centered squared deviations.. Clustering: Captain clustering for regression-based rows; Panel C quantile rows use nonparametric bootstrap uncertainty.. Controls: theta_hat, scarcity, and captain experience, with prior output volatility added to define heterogeneity cells.. Interpretation: High-psi organizations do not merely raise mean output; they materially reduce the left tail and compress downside volatility, especially in riskier captain or ground environments.. Caution: Expected shortfall and long-dry-spell rows use paper-layer proxies built from shipped voyage and action panels rather than a dedicated archival downside-risk file..

Implementation notes:
- `long dry spell` is defined from the voyage-level maximum consecutive empty-day streak aggregated from the action panel.
- `expected shortfall proxy` is the shortfall below the first nondegenerate lower-tail cutoff: the voyage-output 10th percentile when positive, or the positive-output 10th percentile when the overall cutoff is zero.
- Panel C quantile rows are unconditional distributional gaps between high- and low-psi voyages with bootstrap uncertainty.
