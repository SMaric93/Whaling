# QA Report

- Checked that `master_sample_lineage.parquet` and `test_registry.csv` exist.
- Checked that main Tables 1-10 no longer build as placeholder module-mapping tables.
- Checked that Tables 3, 5, and 10 now emit real hierarchy/state/matching rows rather than carrying explicit gap markers.
- Checked that Appendix Tables A1-A18 build as real data tables rather than module wrappers.
- Checked that Figures 1-10 build from paper tables/manifests rather than placeholder plots.
- Remaining risk: archival mechanism rows are still constrained by missing direct archival governance fields, and the stopping-survival robustness rows are proxy implementations because `statsmodels` and `lifelines` are not installed.
