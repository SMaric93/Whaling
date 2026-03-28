# Table 4. State-Contingent Stopping Rules

| panel | row_label | model | coefficient | std_error | p_value | marginal_effect | n_obs | note |
|---|---|---|---|---|---|---|---|---|
| Panel A | psi_hat | Exported logit hazard | -0.0079 | 0.0049 | 0.1101 | -0.0015 | 159,457 | Average-slope marginal effect proxy uses p(1-p) from the exported survival sample. |
| Panel A | negative signal | Exported logit hazard | -1.031 | 0.0513 | 0.0000 | -0.1951 | 159,457 | Average-slope marginal effect proxy uses p(1-p) from the exported survival sample. |
| Panel A | psi_hat × negative signal | Exported logit hazard | 0.0589 | 0.0092 | 0.0000 | 0.0111 | 159,457 | Average-slope marginal effect proxy uses p(1-p) from the exported survival sample. |
| Panel A | positive signal | Exported logit hazard | -0.3958 | 0.0578 | 0.0000 | -0.0749 | 159,457 | Average-slope marginal effect proxy uses p(1-p) from the exported survival sample. |
| Panel A | psi_hat × positive signal | Exported logit hazard | -0.0395 | 0.0109 | 0.0003 | -0.0075 | 159,457 | Average-slope marginal effect proxy uses p(1-p) from the exported survival sample. |
| Panel A | psi_hat | Clustered LPM supplement | 0.0076 | 0.0141 | 0.5896 | 0.0076 | 24,403 | Linear-probability fallback used because statsmodels logit is not installed in the repository venv. |
| Panel A | negative signal | Clustered LPM supplement | 0.0808 | 0.0163 | 0.0000 | 0.0808 | 24,403 | Linear-probability fallback used because statsmodels logit is not installed in the repository venv. |
| Panel A | psi_hat × negative signal | Clustered LPM supplement | 0.0111 | 0.0301 | 0.7119 | 0.0111 | 24,403 | Linear-probability fallback used because statsmodels logit is not installed in the repository venv. |
| Panel A | positive signal | Clustered LPM supplement | -0.0576 | 0.0109 | 0.0000 | -0.0576 | 24,403 | Linear-probability fallback used because statsmodels logit is not installed in the repository venv. |
| Panel A | psi_hat × positive signal | Clustered LPM supplement | -0.0349 | 0.0215 | 0.1046 | -0.0349 | 24,403 | Linear-probability fallback used because statsmodels logit is not installed in the repository venv. |
| Panel A | scarcity | Clustered LPM supplement | -0.0207 | 0.0063 | 0.0011 | -0.0207 | 24,403 | Linear-probability fallback used because statsmodels logit is not installed in the repository venv. |
| Panel A | theta_hat | Clustered LPM supplement | -0.0058 | 0.0104 | 0.5762 | -0.0058 | 24,403 | Linear-probability fallback used because statsmodels logit is not installed in the repository venv. |
| Panel A | experience | Clustered LPM supplement | -0.0001 | 0.0022 | 0.9720 | -0.0001 | 24,403 | Linear-probability fallback used because statsmodels logit is not installed in the repository venv. |
| Panel B | consecutive empty days | Clustered LPM robustness | 0.0000 | 0.0000 | 0.4091 | 0.0000 | 266,729 | Day-level robustness from the action dataset. |
| Panel B | days since last success | Clustered LPM robustness | -0.0000 | 0.0000 | 0.7892 | -0.0000 | 243,598 | Day-level robustness from the action dataset. |
| Panel B | barren-search-state indicator | Clustered LPM robustness | 0.0031 | 0.0053 | 0.5541 | 0.0031 | 266,729 | Day-level robustness from the action dataset. |
| Panel B | leave-one-out local quality | Clustered LPM robustness | -0.0000 | 0.0000 | 0.4363 | -0.0000 | 161,693 | Day-level robustness from the action dataset. |
| Panel B | recent failure streak | Clustered LPM robustness | 0.0027 | 0.0047 | 0.5636 | 0.0027 | 266,729 | Day-level robustness from the action dataset. |
| Panel C | active search | Clustered LPM placebo | 0.0031 | 0.0053 | 0.5541 | 0.0031 | 266,729 | Day-level placebo sample. |
| Panel C | transit | Clustered LPM placebo |  |  |  |  | 0 | Existing next-round transit exit rate = 0.625. |
| Panel C | homebound | Clustered LPM placebo | -0.0161 | 0.0217 | 0.4568 | -0.0161 | 30,108 | Day-level placebo sample. |
| Panel C | productive/exploitation state | Clustered LPM placebo | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 51,587 | Day-level placebo sample. |
