# table03_hierarchical_map

Sample: Connected-set voyages merged to the repository's destination ontology.. Unit: Voyage-level destination decision.. Types: The captain and agent identity specifications use one-hot captain_id and agent_id features; the type specification uses theta_hat and psi_hat directly.. FE: No fixed effects; destination choice is estimated with multinomial classifiers on a time split.. Clustering: Not reported in the predictive benchmark table.. Controls: Environment controls are year_out, tonnage, and duration_days, with basin/theater conditioning variables included in Panels B and C.. Interpretation: Destination control is hierarchical and environment-heavy: broad location choices are increasingly predictable once the ontology is respected, while captain and agent contributions vary by decision level.. Caution: This table is a predictive decomposition rather than a causal estimate, and the current build uses time-split validation rather than the full captain-group and agent-group holdout battery..

Implementation notes:
- The paper layer rebuilds all three hierarchy levels directly because the shipped next-round CSV only preserved a stale basin benchmark.
- Panel B conditions on basin and Panel C conditions on basin plus theater using the repaired destination ontology.
- The captain and agent specifications use identity features; the final paper-facing specification uses `theta_hat` and `psi_hat` directly.
