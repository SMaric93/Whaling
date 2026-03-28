# tableA17_regime_heterogeneity

Supports: Tables 3, 8, and 10.

Sample: Connected-set voyage sample, with shipped ML scarcity heterogeneity support rows appended.. Unit: Era-specific regression or ML heterogeneity row.. Types: theta_hat and psi_hat are the connected-set voyage types.. FE: None in the current paper-layer era splits.. Clustering: Captain clustering in the regression rows.. Controls: Scarcity and captain experience in both output and tail-risk regressions.. Interpretation: The governance and tail-risk patterns vary by era but do not disappear in any one historical regime.. Caution: Era-specific samples are smaller and the ML heterogeneity rows are predictive support rather than causal estimates..

Implementation notes:
- Panels A-B rebuild era-specific paper-layer regressions.
- Panel C appends the shipped ML scarcity heterogeneity summary.
