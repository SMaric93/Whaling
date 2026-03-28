# Table A6. ML Ablation Audit

| source | ablation | log_loss | brier_score | auc | macro_f1 | top_3_accuracy | model | split | elapsed_sec | task | target | top_2_accuracy | rmse | mae | r_squared | n_obs | calibration_slope | val_rmse | val_mae | val_r_squared | val_n_obs | val_calibration_slope | test_rmse | test_mae | test_r_squared | test_n_obs | test_calibration_slope | val_log_loss | val_brier_score | val_auc | val_macro_f1 | val_top_2_accuracy | test_log_loss | test_brier_score | test_auc | test_macro_f1 | test_top_2_accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| policy_map | env_only | 2.615 | 0.0864 | 0.5000 | 0.0220 | 0.3774 | majority_class | val | 0.0006 | map |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| policy_map | env_only | 2.789 | 0.0908 | 0.5000 | 0.0107 | 0.3268 | majority_class | test | 0.0006 | map |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| policy_map | env_only | 3.144 | 0.0901 | 0.6390 | 0.1029 | 0.4738 | logistic | val | 2.558 | map |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| policy_map | env_only | 6.559 | 0.1233 | 0.4762 | 0.0547 | 0.2307 | logistic | test | 2.558 | map |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| policy_map | env_only | 4.810 | 0.0921 | 0.7349 | 0.3921 | 0.6328 | hist_gradient_boosting | val | 3.561 | map |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| policy_map | env_only | 10.458 | 0.1528 | 0.5074 | 0.0629 | 0.2469 | hist_gradient_boosting | test | 3.561 | map |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| policy_map | env_captain | 2.615 | 0.0864 | 0.5000 | 0.0220 | 0.3774 | majority_class | val | 0.0004 | map |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| policy_map | env_captain | 2.789 | 0.0908 | 0.5000 | 0.0107 | 0.3268 | majority_class | test | 0.0004 | map |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| policy_map | env_captain | 2.594 | 0.0844 | 0.6016 | 0.0585 | 0.4721 | logistic | val | 3.071 | map |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| policy_map | env_captain | 2.534 | 0.0812 | 0.5629 | 0.0623 | 0.5269 | logistic | test | 3.071 | map |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| policy_map | env_captain | 5.007 | 0.0908 | 0.7393 | 0.3897 | 0.6328 | hist_gradient_boosting | val | 3.881 | map |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| policy_map | env_captain | 10.905 | 0.1512 | 0.5058 | 0.0557 | 0.2531 | hist_gradient_boosting | test | 3.881 | map |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| policy_map | env_agent | 2.615 | 0.0864 | 0.5000 | 0.0220 | 0.3774 | majority_class | val | 0.0021 | map |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| policy_map | env_agent | 2.789 | 0.0908 | 0.5000 | 0.0107 | 0.3268 | majority_class | test | 0.0021 | map |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| policy_map | env_agent | 3.144 | 0.0901 | 0.6390 | 0.1029 | 0.4738 | logistic | val | 4.360 | map |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| policy_map | env_agent | 6.559 | 0.1233 | 0.4762 | 0.0547 | 0.2307 | logistic | test | 4.360 | map |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| policy_map | env_agent | 4.810 | 0.0921 | 0.7349 | 0.3921 | 0.6328 | hist_gradient_boosting | val | 4.544 | map |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| policy_map | env_agent | 10.458 | 0.1528 | 0.5074 | 0.0629 | 0.2469 | hist_gradient_boosting | test | 4.544 | map |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| policy_map | env_captain_agent | 2.615 | 0.0864 | 0.5000 | 0.0220 | 0.3774 | majority_class | val | 0.0009 | map |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| policy_map | env_captain_agent | 2.789 | 0.0908 | 0.5000 | 0.0107 | 0.3268 | majority_class | test | 0.0009 | map |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| policy_map | env_captain_agent | 2.594 | 0.0844 | 0.6016 | 0.0585 | 0.4721 | logistic | val | 3.116 | map |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| policy_map | env_captain_agent | 2.534 | 0.0812 | 0.5629 | 0.0623 | 0.5269 | logistic | test | 3.116 | map |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| policy_map | env_captain_agent | 5.007 | 0.0908 | 0.7393 | 0.3897 | 0.6328 | hist_gradient_boosting | val | 4.041 | map |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| policy_map | env_captain_agent | 10.905 | 0.1512 | 0.5058 | 0.0557 | 0.2531 | hist_gradient_boosting | test | 4.041 | map |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| policy_compass | env_state | 1.275 | 0.1377 | 0.5000 | 0.1170 | 0.9611 | majority_class | val | 0.0041 | compass_next_action_class | next_action_class |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| policy_compass | env_state | 1.489 | 0.1563 | 0.5000 | 0.0934 | 0.9563 | majority_class | test | 0.0041 | compass_next_action_class | next_action_class |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| policy_compass | env_state | 1.035 | 0.1120 | 0.7525 | 0.2640 | 0.9614 | logistic | val | 52.229 | compass_next_action_class | next_action_class |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| policy_compass | env_state | 1.226 | 0.1332 | 0.7124 | 0.2276 | 0.9535 | logistic | test | 52.229 | compass_next_action_class | next_action_class |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| policy_compass | env_state | 0.8951 | 0.0966 | 0.7978 | 0.4236 | 0.9634 | hist_gradient_boosting | val | 44.363 | compass_next_action_class | next_action_class |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| policy_compass | env_state | 0.9795 | 0.1033 | 0.7823 | 0.4067 | 0.9473 | hist_gradient_boosting | test | 44.363 | compass_next_action_class | next_action_class |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
