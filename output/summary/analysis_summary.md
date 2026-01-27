# Whaling Industry Analysis: Key Findings

**Date:** January 26, 2026  
**Sample:** 9,229 voyages | 2,311 captains | 777 agents (LOO connected set)

---

## 1. Who Drives Success: Agent vs Captain

| Component | Variance Share |
|-----------|---------------|
| **Agent capability (γ)** | **94.1%** |
| Captain skill (α) | 11.6% |
| Sorting covariance | -5.6% |

**Bottom line:** Agents (principals) explain 8× more voyage production variance than captains. The business side—capital allocation, ship selection, routing decisions—dominates individual mariner skill.

---

## 2. First Voyage Success (ML Prediction)

**AUC = 0.887** — Highly predictable who succeeds on first voyage

| Predictor | Importance |
|-----------|------------|
| Ship tonnage | 73% |
| Year | 14% |
| Agent quality (γ) | 13% |

**Implication:** Success is largely determined by resource allocation (bigger ships) and agent selection, not innate captain ability.

---

## 3. Industry Lifecycle in Market Structure

| Decade | Top 5 Agents Share |
|--------|-------------------|
| 1800 | 77% |
| 1840 | 7% |
| 1890 | 37% |

The industry followed a classic lifecycle: early oligopoly → competitive expansion → re-consolidation during decline.

---

## 4. The Negative Learning Curve

Production **falls** 9.3% from early career (voyages 1-3) to mid-career (4-6).

**Cause:** Survivor bias. The declining whale stocks and industry meant:

- Only successful captains got repeat voyages
- Each successive voyage faced depleted grounds
- Selection dominated skill development

---

## 5. Captain Career Mobility

| Pattern | Captains |
|---------|----------|
| Upgraded ships over career | 40% |
| Downgraded ships | 25% |
| Stayed with same agent (2+ voyages) | 27% |

Partnership loyalty was modest (27% repeat), and only 38% of repeat partnerships showed improved production.

---

## 6. Production Peak & Decline

| Decade | Mean Production |
|--------|-----------------|
| 1800 | 477 barrels |
| **1850** | **1,822 barrels** |
| 1920 | 391 barrels |

The industry peaked mid-century, then declined steadily as whale stocks depleted.

---

## Key Outputs

- **Regression tables:** `output/tables/table_main_regressions.csv`
- **Variance decomposition:** `output/tables/table_variance_decomposition.csv`
- **Captain effects:** `output/tables/r1_captain_effects.csv` (2,311 estimates)
- **Agent effects:** `output/tables/r1_agent_effects.csv` (777 estimates)
- **Visualizations:** `output/figures/ml_analysis_patterns.png`

---

## Economic Interpretation

1. **Venture capital parallel:** Agents were the "VCs" of whaling—their capital deployment, ship selection, and routing choices determined returns more than the "entrepreneurs" (captains) they backed.

2. **Selection over development:** The high AUC for first-voyage prediction and negative learning curves suggest talent was identified, not developed.

3. **Market power cycles:** Concentration patterns mirror modern industries—consolidation → disruption → re-consolidation.
