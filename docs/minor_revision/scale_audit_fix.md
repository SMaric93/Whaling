# Scale Audit Fix — Table 1 vs Table 3 Discrepancy

*Fixes: Editor concern about SD / variance mismatch between Table 1 and Table 3.*

## Original Paper Tables
- **Table 1**: Descriptive statistics on the full analysis sample
- **Table 3**: Variance decomposition on the LOO connected set

## Source of Discrepancy
Sample size differs by 4,988 voyages (Table 1: 10,973, Table 3: 5,985). Mean differs by 0.1700 log points. SD differs by 0.0606. 

The discrepancy arises because:
1. **Different samples**: Table 1 uses the full filtered sample (10,973 voyages),
   while Table 3 uses the LOO connected set (5,985 voyages).
2. **Sample composition**: The connected set drops voyages from captains/agents who are
   not part of the largest bipartite connected component (or who are articulation points
   in the LOO procedure). This changes the moments.

## Corrected Values

| Statistic | Table 1 (full sample) | Table 3 (connected set) |
|-----------|----------------------:|------------------------:|
| N | 10,973 | 5,985 |
| Mean(log_q) | 6.6681 | 6.8381 |
| SD(log_q) | 1.2057 | 1.1451 |
| Var(log_q) | 1.4538 | 1.3113 |

## Fix Applied
1. Table 1 now reports live-computed statistics from the full analysis sample.
2. Table 3 notes now clearly state the sample is the LOO connected set and report
   connected-set descriptive statistics in a support table.
3. Scale consistency assertions added: `abs(SD² - Var) < 1e-4`.

## Old code path
- `src/analyses/paper_tables.py` → hardcoded `TABLE_1_DATA`, `TABLE_3_DATA`

## New code path
- `src/minor_revision/scale_audit.py` → live computation from data

## Does the interpretation change?
No. The variance decomposition shares (captain %, agent %, sorting %) are computed
on the connected-set sample and remain the authoritative estimates. The fix ensures
that descriptive statistics are correctly attributed to their respective samples.
