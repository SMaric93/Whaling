# Route Information Fix

*Fixes: Editor request to replace raw control-share table with defensible metrics.*

## Original paper table
Table 2: "The Locus of Strategy — Conditional Entropy of Ground Selection"

## What was wrong
The raw entropy/MI table presented "control shares" that were descriptive but
lacked finite-sample correction. Captain identity appeared far more granular than
port or agent, inflating its raw MI relative to lower-cardinality predictors.

## What changed
1. AMI (Adjusted Mutual Information) replaces raw MI as the primary metric
2. Conditional MI separates captain routing knowledge from agent portfolio effects
3. Frequency-restricted subsamples confirm robustness to cardinality
4. Out-of-sample prediction benchmark with time split

## Code paths
- Old: `src/analyses/paper_tables.py` → hardcoded `TABLE_2_DATA`
- New: `src/minor_revision/table2_adjusted_info.py` + `route_prediction_oos.py`

## Does the interpretation change?
The qualitative finding — that captain identity is the strongest predictor of
ground choice — is preserved. But the magnitude is now properly penalized for
cardinality, and the incremental information of captains conditional on agents
is explicitly reported.
