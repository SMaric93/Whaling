# Lower-Tail Target Audit

## Summary
- Outcome column: `log_q`
- Global 10th percentile: **0.0000**
- Global 5th percentile: **0.0000**
- Targets are distinct: **False** (overlap: 100.0%)

## Prevalence by Split

| target        | split   |    n |   prevalence_global_threshold |   prevalence_train_threshold |   global_threshold |   train_threshold |   threshold_diff_pct |
|:--------------|:--------|-----:|------------------------------:|-----------------------------:|-------------------:|------------------:|---------------------:|
| bottom_decile | train   | 6306 |                     0.131304  |                    0.131304  |                  0 |                 0 |                    0 |
| bottom_decile | val     | 2102 |                     0.0727878 |                    0.0727878 |                  0 |                 0 |                    0 |
| bottom_decile | test    | 2103 |                     0.1631    |                    0.1631    |                  0 |                 0 |                    0 |
| bottom_5pct   | train   | 6306 |                     0.131304  |                    0.131304  |                  0 |                 0 |                    0 |
| bottom_5pct   | val     | 2102 |                     0.0727878 |                    0.0727878 |                  0 |                 0 |                    0 |
| bottom_5pct   | test    | 2103 |                     0.1631    |                    0.1631    |                  0 |                 0 |                    0 |

## Recommendation

Use **train-only thresholds** to define targets. This prevents leakage
from future observations into the target definition.

## Existing Target Columns in Dataset
- `bottom_decile`: prevalence=0.1260, n_pos=1324, n_total=10511
- `bottom_5pct`: prevalence=0.1260, n_pos=1324, n_total=10511