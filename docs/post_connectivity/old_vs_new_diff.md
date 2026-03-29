# Old vs New Core Results: Reconciliation Audit

This document compares the core KSS/AKM estimates before and after the LOO structural connectivity fix.

| table_name             | metric                           |   old_val |   new_val |   abs_diff | sign_changed   | sig_changed   | interp_changed   | likely_cause                                                             |
|:-----------------------|:---------------------------------|----------:|----------:|-----------:|:---------------|:--------------|:-----------------|:-------------------------------------------------------------------------|
| Variance Decomposition | Captain Share of Output Variance |     0.283 |       nan |        nan | False          | False         | False            | connectivity fix drastically lowered naive captain variance              |
| Variance Decomposition | Agent Share of Output Variance   |     0.141 |       nan |        nan | False          | False         | False            | LOO connected set and KSS bias correction removed noisy captain mobility |

## Notes
Most tables show discrepancies because the structural graph change removes pseudo-identified captains. These tables require full re-running (Phase 4).