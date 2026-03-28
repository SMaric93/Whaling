# Table A15. Matching Robustness

| panel | assignment | mean_output | std_output | bottom_decile_rate | bottom_5pct_rate | cvar_10 | certainty_equiv | n | current_theta_psi_corr | predicted_direction | cvar_10_observed | bottom_decile_observed | note | observed_total_q | pam_total_q | nam_total_q | pam_gain_pct | nam_gain_pct | efficient_matching | gamma_hat | n_obs | specification | outcome | variable | coefficient | std_error | p_value | stars | r_squared | n_clusters | fe_structure | cluster_var | row_label | estimate | bootstrap_se | ci_lower | ci_upper |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Panel A | observed | 1,221.11 | 1,098.98 | 0.1164 | 0.1164 | 0.0000 | -602,662.88 | 10,511 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Panel A | PAM | 1,221.11 |  |  |  |  |  | 10,511 | -0.0287 | higher mean if supermodular, lower if submodular |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Panel A | NAM/AAM | 1,221.11 |  |  |  |  |  | 10,511 | -0.0287 | reduces tail risk if substitutes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Panel A | risk_optimal | 1,221.11 |  |  |  |  |  | 10,511 |  |  | 0.0000 | 0.1164 | Full optimization requires solving an assignment problem with estimated surface |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Panel B |  |  |  |  |  |  |  |  |  |  |  |  |  | 12,125.57 | 11,798.43 | 12,531.65 | -2.698 | 3.349 | NAM | -0.0566 | 2,046 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Panel B |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 2,046 | Submodularity (linear) | log(Q) | theta_heldout | 0.3550 | 0.1636 | 0.0302 | ** | 0.4778 | 525.00 | year + ground | captain_id |  |  |  |  |  |
| Panel B |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 2,046 | Submodularity (linear) | log(Q) | psi_heldout | 0.4309 | 0.0107 | 0.0000 | *** | 0.4778 | 525.00 | year + ground | captain_id |  |  |  |  |  |
| Panel B |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 2,046 | Submodularity (linear) | log(Q) | theta_x_psi | -0.0266 | 0.0246 | 0.2796 |  | 0.4778 | 525.00 | year + ground | captain_id |  |  |  |  |  |
| Panel B |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1,074 | Submodularity (triple) | log(Q) | theta_heldout | 0.0952 | 0.0893 | 0.2867 |  | 0.5909 | 425.00 | year + ground | captain_id |  |  |  |  |  |
| Panel B |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1,074 | Submodularity (triple) | log(Q) | psi_heldout | 0.2033 | 0.0095 | 0.0000 | *** | 0.5909 | 425.00 | year + ground | captain_id |  |  |  |  |  |
| Panel B |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1,074 | Submodularity (triple) | log(Q) | scarcity_index | -0.7765 | 0.0510 | 0.0000 | *** | 0.5909 | 425.00 | year + ground | captain_id |  |  |  |  |  |
| Panel B |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1,074 | Submodularity (triple) | log(Q) | theta_x_psi | 0.0032 | 0.0133 | 0.8101 |  | 0.5909 | 425.00 | year + ground | captain_id |  |  |  |  |  |
| Panel B |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1,074 | Submodularity (triple) | log(Q) | theta_x_psi_x_S | 0.0151 | 0.0072 | 0.0369 | ** | 0.5909 | 425.00 | year + ground | captain_id |  |  |  |  |  |
| Panel B |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 404.00 | Scarcity bin=S1 | log(Q) | theta_heldout | -0.0659 | 0.0939 |  |  | 0.0628 | 266.00 | year | captain_id |  |  |  |  |  |
| Panel B |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 404.00 | Scarcity bin=S1 | log(Q) | psi_heldout | 0.1812 | 0.0111 |  |  | 0.0628 | 266.00 | year | captain_id |  |  |  |  |  |
| Panel B |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 404.00 | Scarcity bin=S1 | log(Q) | theta_x_psi | 0.0214 | 0.0161 |  |  | 0.0628 | 266.00 | year | captain_id |  |  |  |  |  |
| Panel B |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 312.00 | Scarcity bin=S2 | log(Q) | theta_heldout | 0.0686 | 0.1568 |  |  | 0.3258 | 210.00 | year | captain_id |  |  |  |  |  |
| Panel B |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 312.00 | Scarcity bin=S2 | log(Q) | psi_heldout | 0.1558 | 0.0152 |  |  | 0.3258 | 210.00 | year | captain_id |  |  |  |  |  |
| Panel B |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 312.00 | Scarcity bin=S2 | log(Q) | theta_x_psi | -0.0073 | 0.0251 |  |  | 0.3258 | 210.00 | year | captain_id |  |  |  |  |  |
| Panel B |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 358.00 | Scarcity bin=S3 | log(Q) | theta_heldout | 0.2595 | 0.1045 |  |  | 0.4846 | 156.00 | year | captain_id |  |  |  |  |  |
| Panel B |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 358.00 | Scarcity bin=S3 | log(Q) | psi_heldout | 0.2226 | 0.0234 |  |  | 0.4846 | 156.00 | year | captain_id |  |  |  |  |  |
| Panel B |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 358.00 | Scarcity bin=S3 | log(Q) | theta_x_psi | -0.0214 | 0.0217 |  |  | 0.4846 | 156.00 | year | captain_id |  |  |  |  |  |
| Panel C |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | Observed support pairs | 6,223 |  |  |  |
| Panel C |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | PAM bootstrap mean output gain | 82.443 | 6.227 | 70.218 | 94.763 |
| Panel C |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | PAM bootstrap CVaR(10) gain | -82.298 | 9.793 | -101.04 | -62.312 |
| Panel C |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | AAM/NAM bootstrap mean output gain | -120.78 | 4.688 | -128.64 | -111.05 |
| Panel C |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | AAM/NAM bootstrap CVaR(10) gain | 189.32 | 11.412 | 167.16 | 209.54 |
