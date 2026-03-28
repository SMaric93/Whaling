# tableA16_influence

Supports: Tables 2 and 10.

Sample: Connected-set voyage sample.. Unit: Influence scenario.. Types: theta_hat and psi_hat are the connected-set types used in sorting and output regressions.. FE: None in the influence summaries.. Clustering: Captain clustering in the psi-slope summary.. Controls: theta_hat, scarcity, and captain experience in the psi-slope regression.. Interpretation: The main sorting and psi-slope patterns are not driven entirely by one dominant captain, one agent, or one ground.. Caution: This appendix is a coarse influence screen rather than a full jackknife over all entities..

Implementation notes:
- Panel A drops the most influential observed captain, agent, and ground one at a time.
- Panel B reports era-specific summaries.
