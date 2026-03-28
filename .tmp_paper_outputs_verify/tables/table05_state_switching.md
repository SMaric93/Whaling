# Table 5. State Transitions and Policy Change After Switching Agents

| panel | row_label | coefficient_on_psi_hat | coefficient_on_psi_hat_x_scarcity | std_error | p_value | n_obs | note | coefficient |
|---|---|---|---|---|---|---|---|---|
| Panel A | barren_search -> exit/relocation | 0.0017 | 0.0003 | 0.0034 | 0.6219 | 176,242 | State labels rebuilt with the repository's GMM latent-state model and rule-based labeler. |  |
| Panel A | barren_search -> stay | 0.0033 | 0.0341 | 0.0093 | 0.7232 | 176,242 | State labels rebuilt with the repository's GMM latent-state model and rule-based labeler. |  |
| Panel A | exploitation -> stay | 0.0017 | -0.0492 | 0.0113 | 0.8791 | 79,444 | State labels rebuilt with the repository's GMM latent-state model and rule-based labeler. |  |
| Panel A | exploitation -> exit | -0.0001 | 0.0054 | 0.0016 | 0.9564 | 79,444 | State labels rebuilt with the repository's GMM latent-state model and rule-based labeler. |  |
| Panel A | transit -> local_search | 0.0000 | 0.0000 | 0.0000 |  | 9,894 | State labels rebuilt with the repository's GMM latent-state model and rule-based labeler. |  |
| Panel A | transit -> exploitation | -0.0017 | 0.0044 | 0.0056 | 0.7564 | 9,894 | State labels rebuilt with the repository's GMM latent-state model and rule-based labeler. |  |
| Panel B | post-switch |  |  | 0.0000 |  | 282,246 | Captain fixed-effects estimate on action-level exit behavior among captains who ever switch agents. State labels rebuilt with the repository's GMM latent-state model and rule-based labeler. | 0.0119 |
| Panel B | switch to higher-psi |  |  | 0.0000 |  | 282,246 | Captain fixed-effects estimate on action-level exit behavior among captains who ever switch agents. State labels rebuilt with the repository's GMM latent-state model and rule-based labeler. | -0.0001 |
| Panel B | switch to lower-psi |  |  | 0.0000 |  | 282,246 | Captain fixed-effects estimate on action-level exit behavior among captains who ever switch agents. State labels rebuilt with the repository's GMM latent-state model and rule-based labeler. | 0.0120 |
| Panel B | post-switch × barren state |  |  | 0.0043 | 0.0000 | 282,246 | Captain fixed-effects estimate on action-level exit behavior among captains who ever switch agents. State labels rebuilt with the repository's GMM latent-state model and rule-based labeler. | -0.0231 |
| Panel B | post-switch × exploitation state |  |  | 0.0047 | 0.0000 | 282,246 | Captain fixed-effects estimate on action-level exit behavior among captains who ever switch agents. State labels rebuilt with the repository's GMM latent-state model and rule-based labeler. | -0.0243 |
| Panel C | t-2 |  |  | 0.0452 | 0.3617 | 39 | Event-study coefficient is the within-captain change in voyage-level barren-state exit propensity relative to t-1. State labels rebuilt with the repository's GMM latent-state model and rule-based labeler. | -0.0412 |
| Panel C | t-1 |  |  | 0.0000 | 1.000 | 242 | Event-study coefficient is the within-captain change in voyage-level barren-state exit propensity relative to t-1. State labels rebuilt with the repository's GMM latent-state model and rule-based labeler. | 0.0000 |
| Panel C | switch |  |  | 0.0349 | 0.4576 | 70 | Event-study coefficient is the within-captain change in voyage-level barren-state exit propensity relative to t-1. State labels rebuilt with the repository's GMM latent-state model and rule-based labeler. | 0.0259 |
| Panel C | t+1 |  |  | 0.0406 | 0.9798 | 47 | Event-study coefficient is the within-captain change in voyage-level barren-state exit propensity relative to t-1. State labels rebuilt with the repository's GMM latent-state model and rule-based labeler. | 0.0010 |
| Panel C | t+2 |  |  | 0.0738 | 0.6980 | 27 | Event-study coefficient is the within-captain change in voyage-level barren-state exit propensity relative to t-1. State labels rebuilt with the repository's GMM latent-state model and rule-based labeler. | 0.0286 |
