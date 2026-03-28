# tableA10_rational_exit

Supports: Table 4.

Sample: Shipped next-round rational-exit output.. Unit: Test row.. Types: Upstream next-round output; notation mapped to psi_hat/theta_hat in the memo only.. FE: Inherited from the upstream next-round builder.. Clustering: Inherited from the upstream next-round builder.. Controls: Season remaining, scarcity, consecutive empty days, and placebo transit splits as shipped.. Interpretation: The rational-exit checks help separate organizational patience from pure optimization on outside options and season remaining.. Caution: This appendix reuses the shipped next-round CSV rather than re-estimating the interactions in the paper layer..

Implementation notes:
- Rows are drawn directly from `outputs/tables/next_round/rational_exit_tests.csv`.
