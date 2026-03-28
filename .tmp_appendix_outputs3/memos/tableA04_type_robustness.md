# tableA04_type_robustness

Supports: Table 2.

Sample: AKM variance decomposition, reinforcement type summaries, and AKM tail diagnostics shipped with the repository.. Unit: Variance component or entity bin.. Types: This table standardizes `alpha_hat`/`gamma_hat` outputs to the paper-facing `theta_hat`/`psi_hat` notation.. FE: AKM/KSS decomposition as shipped upstream.. Clustering: As inherited from the upstream decomposition and split-sample routines.. Controls: No additional paper-layer controls.. Interpretation: Captain and agent types remain nontrivial after robustness checks, with split-sample stability and reliability strongest in better-observed bins.. Caution: Not every upstream output is on the same sample; this appendix is a robustness dashboard rather than a single re-estimated model..

Implementation notes:
- Panel A reproduces shipped AKM/KSS variance components.
- Panels B-D summarize held-out and reliability diagnostics from reinforcement and AKM-tail outputs.
