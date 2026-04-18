# WSL Reliability ML

This directory implements the new WSL reliability ML stack requested for the whaling project.

## What I Added

I added a four-part pipeline plus orchestration code:

- `remarks_taxonomy.py`
  Builds canonical remarks text, assigns rule-based weak labels, trains calibrated baseline text models, and exports event-level semantic labels and audit files.
- `voyage_state_model.py`
  Links event mentions into voyage episodes, creates anchor labels, infers latent voyage states, and summarizes bad-state entry, recovery, and termination.
- `departure_information.py`
  Builds departure snapshots using `issue_date` as the information clock and computes public, peer, agent, registry, uncertainty, and composite information features.
- `policy_layer.py`
  Fits local observational policy-learning layers for predeparture targeting and post-distress triage, then exports policy scores, frontiers, and an information-equalization exercise.
- `utils.py`
  Provides configuration, WSL event loading, page-type separation, uncertainty weighting, voyage linkage, safe export helpers, and manifest support.
- `src/analyses/run_wsl_reliability_ml.py`
  Runs the full stack end to end and writes outputs under `outputs/wsl_reliability_ml/`.
- `analysis/run_wsl_reliability_ml.py`
  Thin wrapper entry point for convenience.
- `tests/test_wsl_reliability_ml.py`
  Regression coverage for the main fragile edges discovered during implementation.

## What It Does

The pipeline turns extracted WSL pages into uncertainty-aware ML objects that can support mechanism work in the paper:

1. Remarks taxonomy
   Converts noisy remarks and structured-field overflow into interpretable labels such as distress, interruption/repair, homebound, terminal loss, productivity, and contamination.

2. Latent voyage states
   Uses event sequences to infer whether a voyage is in outbound transit, active search, productive search, low-yield search, distress, in-port repair, homebound termination, completed arrival, or terminal loss.

3. Departure-time information stock
   Freezes all information at departure using `issue_date <= departure_issue_date` and constructs basin-, peer-, and agent-level information features plus uncertainty and coverage measures.

4. Policy layer
   Scores local on-support observational targeting rules for scarce organizational support at departure and after a first bad-state signal.

## Important Design Choices`

- Weekly event-flow pages and registry-stock pages are treated as different data-generating processes.
- `issue_date` is the information-availability clock throughout the stack.
- Raw extracted fields, `_raw`, `_flags`, and `_confidence` are preserved and carried forward.
- Uncertainty is represented with row weights and contamination-aware logic instead of being silently collapsed away.
- Composite indexes are exported together with their raw components.
- Policy outputs are framed as local observational counterfactuals, not structural optimal-policy claims.

## Output Layout

The full runner writes:

- `outputs/wsl_reliability_ml/remarks/*`
- `outputs/wsl_reliability_ml/states/*`
- `outputs/wsl_reliability_ml/information/*`
- `outputs/wsl_reliability_ml/policy/*`
- `outputs/wsl_reliability_ml/manifest.json`
- `outputs/wsl_reliability_ml/RESULTS_SUMMARY.md`

## Robustness Work Added During Implementation

Several defensive fixes were added so the pipeline can run on real WSL data rather than only toy fixtures:

- Safe parquet serialization for nested object columns.
- Repair of duplicate `event_row_id` collisions in cached flattened WSL events.
- Idempotent voyage-linkage attachment so repeated merges do not corrupt columns or double-penalize weights.
- Tolerant handling of missing optional schema columns in predeparture and triage panels.
- Safer weak-label downsampling and rare-class handling in remarks training.
- Compatibility fixes for the installed `scikit-learn` version.

## Validation Status

- The targeted regression suite in `tests/test_wsl_reliability_ml.py` passes.
- A sample end-to-end run completed successfully and produced the full artifact tree.
- The full-corpus run is heavier, especially in voyage-state inference, but uses the same code path.

## How To Run

Run the full stack:

```bash
python -m src.analyses.run_wsl_reliability_ml
```

Useful options:

```bash
python -m src.analyses.run_wsl_reliability_ml --remarks-max-train-rows 20000
python -m src.analyses.run_wsl_reliability_ml --cleaned-events-path /tmp/sample.jsonl --output-root /tmp/wsl_reliability_ml_sample
```

## Main Limitation Right Now

The slowest part of the current implementation is voyage-state inference in `voyage_state_model.py`, which still runs episode-by-episode in Python/NumPy. That is the first place to optimize if we want a faster full-corpus run or Apple Silicon acceleration.
