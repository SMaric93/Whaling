# Table 6. Search Versus Execution and the Value of Exit

| panel | row_label | psi_hat_coefficient | theta_hat_coefficient | interaction_coefficient | std_error | p_value | n_obs | note | simple_difference | matched_estimate | ipw_estimate | doubly_robust_estimate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Panel A | encounter hazard | 0.0167 | 0.0179 | 0.0023 | 0.0252 | 0.9286 | 266,729 | Action-day encounter indicator among active-search observations. |  |  |  |  |
| Panel A | strike | encounter | -0.0185 | 0.0944 | 0.0068 | 0.0524 | 0.8972 | 61,159 | Conditional strike indicator among encounter days. |  |  |  |  |
| Panel A | capture | strike | -0.0340 | 0.1146 | 0.0580 | 0.0533 | 0.2763 | 25,546 | Conditional capture indicator among strike attempts. |  |  |  |  |
| Panel A | yield | capture | -11.103 | 42.918 | 15.328 | 20.534 | 0.4554 | 753 | Voyage-level output per observed capture, using `q_total_index / total captures` as the shipped yield proxy. |  |  |  |  |
| Panel A | total voyage output | -112.75 | 783.23 | 266.93 | 46.604 | 0.0000 | 6,510 | Voyage-level output regression on the connected-set sample. |  |  |  |  |
| Panel B | time to next encounter (30) |  |  |  |  |  | 145,739 | Barren-state exit comparison rebuilt directly from the action panel. | -0.1403 | -0.0615 | -0.1645 | -0.1143 |
| Panel B | time to next encounter (60) |  |  |  |  |  | 145,739 | Barren-state exit comparison rebuilt directly from the action panel. | -0.4654 | -0.1223 | -0.3634 | -0.3013 |
| Panel B | time to next encounter (90) |  |  |  |  |  | 145,739 | Barren-state exit comparison rebuilt directly from the action panel. | -0.9732 | -0.3105 | -0.6548 | -0.5883 |
| Panel B | future exploitation days (30) |  |  |  |  |  | 145,739 | Barren-state exit comparison rebuilt directly from the action panel. | 0.0561 | 0.0321 | 0.0434 | 0.0358 |
| Panel B | future exploitation days (60) |  |  |  |  |  | 145,739 | Barren-state exit comparison rebuilt directly from the action panel. | 0.2468 | 0.1449 | 0.1671 | 0.1578 |
| Panel B | future exploitation days (90) |  |  |  |  |  | 145,739 | Barren-state exit comparison rebuilt directly from the action panel. | 0.5593 | 0.3521 | 0.3821 | 0.3736 |
| Panel B | future output (30) |  |  |  |  |  | 145,739 | Rebuilt directly from the action panel; shipped `exit_value_eval.csv` provides a consistent simple/matched/IPW benchmark for future-output horizons. | -0.0035 | -0.0161 | -0.0132 | -0.0147 |
| Panel B | future output (60) |  |  |  |  |  | 145,739 | Rebuilt directly from the action panel; shipped `exit_value_eval.csv` provides a consistent simple/matched/IPW benchmark for future-output horizons. | -0.0220 | -0.0521 | -0.0453 | -0.0486 |
| Panel B | future output (90) |  |  |  |  |  | 145,739 | Rebuilt directly from the action panel; shipped `exit_value_eval.csv` provides a consistent simple/matched/IPW benchmark for future-output horizons. | -0.0355 | -0.0858 | -0.0732 | -0.0781 |
| Panel B | remaining voyage output |  |  |  |  |  | 145,739 | Barren-state exit comparison rebuilt directly from the action panel. | 0.6078 | 0.2554 | 0.3562 | 0.3230 |
