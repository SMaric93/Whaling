# Ablation Feature Audit

## Summary
- Captain feature mismatches: **0**
- Agent feature mismatches: **2**
- All ablations same feature count: **False**

## Feature Counts by Ablation
- `env_only`: 5 features
- `env_captain`: 6 features
- `env_agent`: 5 features
- `env_captain_agent`: 6 features

## Audit Table

| ablation          |   n_requested |   n_available |   n_missing | missing_features                   | has_captain_features   | expected_captain   | captain_correct   | has_agent_features   | expected_agent   | agent_correct   | has_env_features   |
|:------------------|--------------:|--------------:|------------:|:-----------------------------------|:-----------------------|:-------------------|:------------------|:---------------------|:-----------------|:----------------|:-------------------|
| env_only          |             5 |             5 |           0 | none                               | False                  | False              | True              | False                | False            | True            | True               |
| env_captain       |             7 |             6 |           1 | theta_hat_holdout                  | True                   | True               | True              | False                | False            | True            | True               |
| env_agent         |             6 |             5 |           1 | psi_hat_holdout                    | False                  | False              | True              | False                | True             | False           | True               |
| env_captain_agent |             8 |             6 |           2 | theta_hat_holdout, psi_hat_holdout | True                   | True               | True              | False                | True             | False           | True               |