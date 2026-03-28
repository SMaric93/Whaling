# Table A14. Lower-Tail Robustness

| source | model | split | auc | brier | prevalence | n | target | group | group_value | mean_psi | val_log_loss | val_brier_score | val_auc | val_macro_f1 | val_top_2_accuracy | test_log_loss | test_brier_score | test_auc | test_macro_f1 | test_top_2_accuracy | rmse | mae | r_squared | n_obs | calibration_slope |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| next_round_lower_tail | logistic | val | 0.8237 | 0.0460 | 0.0598 | 535.00 | bottom_decile |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| next_round_lower_tail | logistic | test | 0.8389 | 0.0428 | 0.0599 | 651.00 | bottom_decile |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| next_round_lower_tail | hist_gbt | val | 0.8054 | 0.0460 | 0.0598 | 535.00 | bottom_decile |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| next_round_lower_tail | hist_gbt | test | 0.8013 | 0.0490 | 0.0599 | 651.00 | bottom_decile |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| next_round_lower_tail | logistic | val | 0.8237 | 0.0460 | 0.0598 | 535.00 | bottom_5pct |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| next_round_lower_tail | logistic | test | 0.8389 | 0.0428 | 0.0599 | 651.00 | bottom_5pct |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| next_round_lower_tail | hist_gbt | val | 0.8054 | 0.0460 | 0.0598 | 535.00 | bottom_5pct |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| next_round_lower_tail | hist_gbt | test | 0.8013 | 0.0490 | 0.0599 | 651.00 | bottom_5pct |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| next_round_lower_tail | logistic | val | 0.9513 | 0.0021 | 0.0019 | 535.00 | catastrophic_voyage |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| next_round_lower_tail | logistic | test | 0.5935 | 0.0076 | 0.0077 | 651.00 | catastrophic_voyage |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| next_round_lower_tail | hist_gbt | val | 1.000 | 0.0004 | 0.0019 | 535.00 | catastrophic_voyage |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| next_round_lower_tail | hist_gbt | test | 0.5237 | 0.0090 | 0.0077 | 651.00 | catastrophic_voyage |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| next_round_lower_tail |  |  |  |  | 0.0848 | 5,992 | bottom_decile | experience | 1.000 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| next_round_lower_tail |  |  |  |  | 0.0848 | 5,992 | bottom_5pct | experience | 1.000 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| next_round_lower_tail |  |  |  |  | 0.1582 | 4,519 | bottom_decile | experience | 0.0000 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| next_round_lower_tail |  |  |  |  | 0.1582 | 4,519 | bottom_5pct | experience | 0.0000 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| next_round_lower_tail |  |  |  |  | 0.0317 | 63.000 | bottom_decile | scarcity | -7.372 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| next_round_lower_tail |  |  |  |  | 0.0317 | 63.000 | bottom_5pct | scarcity | -7.372 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| next_round_lower_tail |  |  |  |  | 0.0182 | 55.000 | bottom_decile | scarcity | -7.097 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| next_round_lower_tail |  |  |  |  | 0.0182 | 55.000 | bottom_5pct | scarcity | -7.097 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| next_round_lower_tail |  |  |  |  | 0.0267 | 75.000 | bottom_decile | scarcity | -7.456 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| next_round_lower_tail |  |  |  |  | 0.0267 | 75.000 | bottom_5pct | scarcity | -7.456 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| next_round_lower_tail |  |  |  |  | 0.8896 | 163.00 | bottom_decile | scarcity | -0.0000 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| next_round_lower_tail |  |  |  |  | 0.8896 | 163.00 | bottom_5pct | scarcity | -0.0000 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| next_round_lower_tail |  |  |  |  | 0.0588 | 51.000 | bottom_decile | scarcity | -6.729 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| next_round_lower_tail |  |  |  |  | 0.0588 | 51.000 | bottom_5pct | scarcity | -6.729 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| next_round_lower_tail |  |  |  |  | 0.0286 | 70.000 | bottom_decile | scarcity | -7.307 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| next_round_lower_tail |  |  |  |  | 0.0286 | 70.000 | bottom_5pct | scarcity | -7.307 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| next_round_lower_tail |  |  |  |  | 0.0345 | 58.000 | bottom_decile | scarcity | -7.392 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| next_round_lower_tail |  |  |  |  | 0.0345 | 58.000 | bottom_5pct | scarcity | -7.392 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
