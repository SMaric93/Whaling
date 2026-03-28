# table01_sample

Sample: Panel A reports sample lineage across the full voyage universe; Panel B reports connected-set voyages with AKM types.. Unit: Voyage in Panel B; row-specific sample unit in Panel A.. Types: theta_hat and psi_hat use the repository's connected-set AKM estimates from outcome_ml_dataset.parquet.. FE: None for descriptive statistics.. Clustering: None for descriptive statistics.. Controls: No regression controls; state variables are voyage-level means across logged search days when only day-level inputs exist.. Interpretation: The paper's usable identification sample is materially smaller than the raw universe because AKM, logbook, and patch-day coverage intersect only for a subset of voyages.. Caution: Log revenue is proxied with log(q_total_index), and signal variables are voyage-level averages constructed from action logs rather than native voyage scalars..

Proxy notes:
- `Log revenue` uses `q_total_index` because a direct revenue variable is not present in the final voyage panel.
- `Days since last success`, `Consecutive empty days`, and `Scarcity` are aggregated from the action dataset to the voyage level.
- `Patch residence time` is the within-voyage mean patch duration from `output/stopping_rule/patches.csv`.
