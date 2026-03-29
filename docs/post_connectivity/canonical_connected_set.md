# Canonical Connected Set Audit

## Overview
The connected set strategy was upgraded from a naive largest-connected-component to a rigorous KSS Leave-One-Out (LOO) vertex-connected set. This prevents the bias that occurs when an articulation-point captain connects two subgraphs but their individual capability is not properly identified.

## Summary Metrics
- **Raw Voyages**: 15,687
- **Valid Non-Null Voyages**: 15,687
- **Naive Connected Set (Old)**: 15,437 voyages (5,087 captains)
- **LOO Connected Set (New Canonical)**: 8,907 voyages (2,305 captains)

## Old vs New Difference
| metric         |   old_set_value |   new_set_value |   pct_remaining |
|:---------------|----------------:|----------------:|----------------:|
| Total Voyages  |           15437 |            8907 |         57.699  |
| Total Captains |            5087 |            2305 |         45.3116 |
| Total Agents   |            1493 |             623 |         41.7281 |
| Movers         |            2463 |            1762 |         71.5388 |

## Exclusions
- Iterations to reach LOO stability: 7
- Articulation-point captains pruned: 698
- Disconnected stayers dropped: 1874

## Mobility
- **Captains with 2+ agents (Movers)**: 1,762
- **Partner repeat rate**: 35.9%