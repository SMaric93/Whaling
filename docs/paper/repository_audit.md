# Repository Audit

## Enumerated runners/modules

- `src/analyses/exposure_and_proxy_tests.py`
- `src/analyses/ground_selection_test.py`
- `src/analyses/insurance_variance_test.py`
- `src/analyses/run_all.py`
- `src/analyses/run_full_baseline_loo_eb.py`
- `src/compass/run_compass_from_aowl.py`
- `src/ml/run_all.py`
- `src/next_round/portability_tests.py`
- `src/next_round/rational_exit_tests.py`
- `src/next_round/run_all.py`
- `src/pipeline/runner.py`
- `src/reinforcement/run_all.py`
- `src/reinforcement/test1_map_compass.py`
- `src/reinforcement/test2_decomposition.py`
- `src/reinforcement/test3_stopping_rule.py`
- `src/reinforcement/test4_variance_compression.py`
- `src/reinforcement/test5_submodularity.py`

## Key paper inputs currently wired
- `data/final/analysis_voyage_augmented.parquet`
- `outputs/datasets/ml/outcome_ml_dataset.parquet`
- `outputs/datasets/ml/action_dataset.parquet`
- `outputs/datasets/ml/survival_dataset.parquet`
- `data/final/akm_variance_decomposition.csv`
- `output/figures/akm_tails/split_sample_stability.csv`
- `output/reinforcement/tables/test3_stopping_rule.csv`
- `outputs/tables/next_round/*.csv`

## Status
- Main Tables 1-10 now build data-driven paper rows from shipped voyage, action, state, AKM, ontology, and next-round inputs rather than placeholder module maps.
- Appendix Tables A1-A18 now emit real paper-layer tables built from shipped datasets or upstream CSV artifacts instead of generic module wrappers.
- Main Figures 1-10 now build from the paper tables and manifests rather than placeholder line charts.
- `table03_hierarchical_map` now rebuilds basin, theater, and major-ground predictive panels directly from the repaired destination ontology.
- `table05_state_switching` now uses repository-consistent latent-state labels rebuilt from the GMM state model when shipped labels are absent.
- `table10_tail_matching` now rebuilds planner-style observed/PAM/AAM comparison rows plus constrained mean, CVaR, and certainty-equivalent selections directly from the connected sample.
- Remaining limitations are now substantive rather than scaffolding-related: direct archival governance variables are still absent, and survival-style stopping robustness relies on the closest feasible paper-layer proxies because dedicated survival packages are not installed.
