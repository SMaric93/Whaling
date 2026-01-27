# Complete Exploratory Analysis: All Findings

**Date:** January 26, 2026 | **35+ Analyses** | **11,622 Voyages**

---

## 🏆 Top 10 Most Interesting Discoveries

| Rank | Finding | Estimate | Implication |
|------|---------|----------|-------------|
| 1 | **Hot Hand Effect** | +37% | Success breeds success |
| 2 | **Tonnage Dominance** | 50% R² | Capital >> Skill alone |
| 3 | **Inequality Peak** | Gini=0.67 (1850s) | Golden age = extreme dispersion |
| 4 | **War of 1812 Shock** | -0.87 log pts | External shocks devastate |
| 5 | **Vessel Aging** | +0.35 over voyages 1-6 | Ships "break in" |
| 6 | **Mean Reversion** | r=-0.40 | Bad luck corrects |
| 7 | **Star Emergence** | r=0.60 from voyage 1 | Talent visible immediately |
| 8 | **Route Depletion** | N Pacific -3.3%/decade | Whale stocks collapsed |
| 9 | **Agent Switching Cost** | -0.15 log pts | Relationship capital real |
| 10 | **Negative Assortative** | r=-0.05 | Skill and capability substitute |

---

## Round 1: Core Patterns (A1-A14)

### Labor Market

| Finding | Value |
|---------|-------|
| Captains with 2+ agents | 50% |
| Agent pairs sharing captains | 6,697 |
| 1st-voyage survival | 61% |

### Career Dynamics

| Metric | Value |
|--------|-------|
| Median career length | 2 voyages |
| Peak production | Voyage 3 |
| Autocorrelation | -0.04 (low) |

---

## Round 2: Dynamics (B1-B15)

### Switching Costs

| Switch Type | Cost |
|-------------|------|
| Route switch | +0.08 (small gain) |
| Agent switch | -0.15 (penalty) |

### War Effects

| Period | Impact |
|--------|--------|
| War of 1812 | **-0.87** log pts |
| Civil War | -0.23 log pts |
| Post-Civil | -0.49 (permanent) |

### Career Patterns

| Voyage # | Production |
|----------|------------|
| 1 | 6.68 |
| 3 | **6.86** (peak) |
| 8 | 6.27 |
| 10 | 6.18 |

---

## Round 3: Advanced (C1-C10)

### Inequality

| Decade | Gini |
|--------|------|
| 1830s | 0.29 (lowest) |
| **1850s** | **0.67** (highest) |
| 1900s | 0.56 |

### Superstars

- Top 1% threshold: 4,551 barrels
- 102 captains reached top 1%
- Max repeat: 3 voyages

### Hot Hand

| Condition | P(Good) |
|-----------|---------|
| After 2 good | **79%** |
| Otherwise | 43% |
| Boost | **+37%** |

### Vessel Aging

| Voyage # | Production |
|----------|------------|
| 1 | 6.52 |
| 6 | **6.86** (peak) |
| 10 | 6.82 |

### Factor Decomposition

| Factor | R² |
|--------|-----|
| Tonnage alone | **50%** |
| Captain α | 2% |
| Agent γ | 5% |
| Combined | 52% |

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total voyages | 11,622 |
| Unique captains | 4,044 |
| Unique agents | 1,454 |
| Unique vessels | 2,515 |
| Mean voyage duration | 856 days |
| Mean gap between voyages | 3.4 years |
| Golden Age (1840-60) premium | +0.31 |

---

## Key Takeaways

1. **Capital trumps skill** — Tonnage alone = 50% R²; FEs add only 2%
2. **Hot hand is real** — +37% boost after consecutive successes
3. **Relationships matter** — Agent switching costs 0.15 log pts
4. **Talent ID is immediate** — First voyage predicts career (r=0.60)
5. **External shocks devastate** — War of 1812 = -0.87 effect
6. **Vessels improve with use** — Peak at voyage 6
7. **Inequality tracks industry** — Peaked in Golden Age
8. **Mean reversion corrects** — Bad voyages followed by better (r=-0.40)
