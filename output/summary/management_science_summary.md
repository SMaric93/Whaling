# Whaling Industry Analysis: Management Science Summary

**Date:** January 26, 2026  
**Sample:** 11,622 voyages | 4,044 captains | 1,454 agents

---

## Core Finding

**Agents (principals) dominate voyage outcomes through capital allocation, not captain skill.**

| Component | Variance Share | Channel |
|-----------|---------------|---------|
| Agent capability (γ) | 94% | Capital allocation |
| Captain skill (α) | 12% | Human capital |
| Sorting covariance | -6% | Substitution |

---

## Key Results

### 1. Tonnage as Mechanism

- **30%** of agent variance operates through tonnage assignment
- Tonnage coefficient: β = 1.78 (highly significant)
- Agents systematically differ in capital intensity (73-550 tons)

### 2. Regime Shift Post-1870

| Metric | Pre-1870 | Post-1870 | Change |
|--------|----------|-----------|--------|
| Tonnage marginal product | 1.99 | 0.40 | **-80%** |
| Portability slope | 0.68 | 0.26 | -62% |

### 3. Concentration Effects

- Market evolved: 81% → 7% → 41% (U-curve)
- **High concentration erodes portable skill returns** (ψ = -0.01)
- Low HHI: corr(α̂, output) = 0.14
- High HHI: corr(α̂, output) = 0.05

### 4. Learning Curve Reinterpretation

- Raw experience effect: -0.033
- **With tonnage controls: -0.003** (90% attenuation)
- Negative learning is **compositional**, not skill decay

---

## Paper Storyline

1. **Agent dominance robust** — survives all specifications
2. **Capital allocation is the channel** — 30% mediation via tonnage
3. **Production technology shifted** — marginal product collapsed 80%
4. **Concentration enables embeddedness** — erodes skill portability
5. **Selection > development** — assignment explains "learning"

---

## Output Files

| Type | File |
|------|------|
| Summary stats | `output/tables/summary_statistics.csv` |
| Concentration | `output/tables/agent_concentration_by_decade.csv` |
| Portability | `output/figures/rolling_portability.png` |
| Regime shift | `output/figures/tonnage_marginal_product.png` |
| Market structure | `output/figures/agent_concentration.png` |

---

## Implications for Strategy Research

1. **Intermediaries as allocators** — principals drive returns through resource deployment
2. **Skill is regime-dependent** — portable in competitive markets, embedded in concentrated ones
3. **Negative trends can be compositional** — control for assignment before inferring decay
