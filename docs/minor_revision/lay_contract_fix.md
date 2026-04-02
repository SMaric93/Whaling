# Lay-Contract Fix — Coverage Audit and Institutional Evidence

*Fixes: Editor request to evaluate the lay-system / contract alternative explanation.*

## Original paper table fixed
This addresses the alternative explanation that high-capability agents simply wrote
different incentive contracts (lay shares).

## Coverage Audit Result

**No lay/incentive variables exist in the dataset.**

Searched for: `captain_lay`, `mate_lay`, `crew_lay`, `lay`, `share_fraction`, `articles_of_agreement`, `lay_share`, `officer_lay`, `captain_share`, `mate_share`, `crew_share`, `incentive`, `contract`, `compensation`

None were found in the primary analysis dataset (`analysis_voyage.parquet`) or any
other parquet files in `data/final/`.

## Institutional Evidence

American whaling lay contracts were highly standardized by the mid-19th century:

1. **Standardization by rank and port**: The captain's lay was typically 1/15th to 1/18th,
   the first mate's lay 1/25th to 1/35th, and crew lays 1/150th to 1/200th.
   These fractions were remarkably stable across agents within the same port-era
   (Davis, Gallman & Gleiter 1997, *In Pursuit of Leviathan*).

2. **Port-level norms**: Nantucket and New Bedford had distinct lay schedules, but
   within each port, variation was minimal — agents competed on voyage selection and
   vessel quality, not on lay generosity.

3. **Implication for ψ**: If lay shares were standardized within port-era cells, they
   cannot explain the within-port-era variation in agent capability (ψ). The agent
   effect operates through channels other than differential incentive contracts.


### Vessel-Type Proxy Test

We test whether better agents systematically use different vessel types (rig), which
could proxy for contract differences.

**Chi-squared test** of independence (agent-capability quartile × rig type):
χ²=6283.1, p=0.0000, df=144.

This is statistically significant, suggesting some sorting of agents to vessel types.

However, rig type is a vessel characteristic, not a contract characteristic.
Vessel selection could reflect many non-incentive factors (route requirements,
capital stock, port availability).


## Conclusion

The lay-system alternative is not testable with the available data. However, institutional
evidence strongly suggests that lay shares were standardized by rank, port, and era,
limiting their capacity to serve as the omitted variable driving the estimated agent
capability effects.

## Code path
- Old: N/A (no prior lay analysis)
- New: `src/minor_revision/lay_contracts.py`
