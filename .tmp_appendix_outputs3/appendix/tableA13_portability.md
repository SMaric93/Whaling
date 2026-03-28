# Table A13. Portability / Invariance Tests

| test | auc | n_train | n_test | level | env_log_loss | types_log_loss | log_loss_improvement | env_top3_accuracy | types_top3_accuracy | n_obs |
|---|---|---|---|---|---|---|---|---|---|---|
| out_of_time_high_psi | 0.9447 | 101,241 | 89,433 |  |  |  |  |  |  |  |
| out_of_time_low_psi | 0.9625 | 115,802 | 74,064 |  |  |  |  |  |  |  |
| out_of_ground | 0.9876 | 196,815 | 3,128 |  |  |  |  |  |  |  |
| captain-group holdout |  |  |  | basin choice | 0.9528 | 0.9161 | 0.0367 | 0.9365 | 0.9388 | 1,291 |
| captain-group holdout |  |  |  | theater choice conditional on basin | 0.2926 | 0.2788 | 0.0139 | 0.9816 | 0.9888 | 1,252 |
| captain-group holdout |  |  |  | major-ground choice conditional on theater | 0.7302 | 0.7184 | 0.0118 | 0.9407 | 0.9432 | 1,180 |
| agent-group holdout |  |  |  | basin choice | 1.090 | 1.073 | 0.0177 | 0.9173 | 0.9196 | 1,306 |
| agent-group holdout |  |  |  | theater choice conditional on basin | 0.3750 | 0.3681 | 0.0069 | 0.9752 | 0.9784 | 1,252 |
| agent-group holdout |  |  |  | major-ground choice conditional on theater | 0.7008 | 0.6838 | 0.0171 | 0.9504 | 0.9485 | 1,049 |
