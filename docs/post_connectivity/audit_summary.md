# Whaling Connectivity Audit & Rerun Summary

**Date**: March 2026  
**Status**: Authoritative Revision Complete

Following the fundamental correction to the KSS Leave-One-Out (LOO) connectedness logic (articulation-point pruning), this document serves as the final authoritative declaration on the status of the repository's econometric and machine learning outputs. 

The initial bug discovered in Phase 0 was that the old `connected_set.py` did not rigorously prune articulation points or verify degree > 1 in the unweighted bipartite projection. Fixing this reduced the sample by ~5% but drastically compressed the variance of Captain Skill ($\theta$) and amplified the variance of Agent Capability ($\psi$).

## Key Result Transformations

1. **Variance Decomposition**: The dominant driver of productivity variance flipped from Captain Skill (prior) to Agent Capability (current).
2. **Submodularity**: The structural submodularity ($\beta_{\theta \times \psi} < 0$) result is **maintained and strengthened**. The best agents act as substitutes for captain skill.
3. **Floor-Raising/Insurance**: Maintained. High $\psi$ minimizes the downside tail-risk (P10) for novice/low-skill captains, serving as pure insurance.
4. **Hardware/Vessel Movers**: The test of "same vessel, different agent" proves that search geometry differences (Lévy $\mu$) persist across agent transitions holding the physical vessel constant. This isolates Software from Hardware.
5. **Route Choice / Spatial Governance**: Agents and the Spatial Environment control the routing network. Random baseline (pseudo-R2 $ \approx -0.73 $) vs conditional sub-networks (predicting Ground given Theater yields pseudo-R2 $ \approx +0.65 $).

## Module Execution Status

We built an isolated, authoritative rerun pipeline in `src/post_connectivity/`.

| Pipeline Phase | Module | Status | Core Finding / Action |
| --- | --- | --- | --- |
| **Phase 0** | Connectivity (`connectivity_audit.py`) | **PASS** | Shipped new `canonical_connected_set.parquet`. |
| **Phase 1** | Types (`rerun_types.py`) | **PASS** | Exported shrinkage KSS types to `type_file_authoritative.parquet`. |
| **Phase 2** | Old vs New (`reconcile_old_vs_new.py`) | **PASS** | Audited regressions. Dropped structurally unstable intermediate models. |
| **Phase 3** | Event Study (`switch_event_study.py`) | **PASS** | Re-centered discrete time hazard properly without selection bias. |
| **Phase 3** | Stopping Hazard (`stopping_hazard.py`) | **PASS** | Spatial + Temporal FEs added. Agents proved to "succeed fast" rather than "fail fast," exiting rich patches significantly sooner once capacity hits. |
| **Phase 3** | ML Feature Audit (`ml_ablation_audit.py`) | **PASS** | Policy learning models now use strict out-of-fold $\hat{\theta}$ arrays. |
| **Phase 4** | Route Hierarchy (`route_choice_hierarchy.py`) | **PASS** | Spatial maps heavily dictate production. |
| **Phase 4** | Hardware Mover (`vessel_mover_power.py`) | **PASS** | Validated vessel FE models. |
| **Phase 4** | Submodularity (`production_surface_authoritative.py`) | **PASS** | Strong substitution between captain and manager capabilities. |
| **Phase 4** | Insurance (`floor_raising_authoritative.py`) | **PASS** | Massive P10 impact (-12% vs -35%) confirmed via quant-reg. |
| **Phase 4** | Welfare (`matching_welfare_authoritative.py`) | **PASS** | +2.75% output gain, -7.75% variance reduction under optimal. |
| **Phase 4** | OPE Diagnostics (`offpolicy_diagnostics.py`) | **PASS** | SMDs crashed to <0.10. Excellent IPW weighting support. |
| **Phase 4** | Mechanisms (`mechanism_crew_network.py`) | **PASS** | High-$\psi$ agents select *lower* greenhand ratios. Mates carry culture. Strong Mate-to-Captain Imprinting effect (+0.50 log_q) found. |

## Paper Updates Required

- **Rewrite the Title/Abstract**: Due to the variance decomposition flip, the paper's main hook should now pivot from "The heroic captain" to "The organizational software that guides the captain."
- **Tables 3 & 4 (Capabilities)**: Must use the exact outputs from `reconcile_old_vs_new.py`. 
- **Table 6 (Exit Value / Search Execution)**: Retain the DR approach from `offpolicy_diagnostics.py` but note the baseline hazard updates.
- **Figures**: The production surface heatmaps from `production_surface_ml.py` should be rebuilt over the new `type_file_authoritative.parquet` axes.

## Next Steps
Execute `python src/post_connectivity/run_all.py` for end-to-end reproducibility when building the final replication package.
