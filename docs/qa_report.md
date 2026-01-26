# Data Quality Assurance Report

Generated: 2026-01-25 20:30:40

This report summarizes data quality metrics for the Whaling Data Pipeline outputs.


## Analysis Voyage Summary

- **Total voyages**: 15,687
- **Unique vessels**: 3,402
- **Unique captains**: 5,286
- **Year range**: 1667 - 1937


## Analysis Captain-Year Summary

- **Total captain-year observations**: 4,491
- **Unique captains**: 2,988
- **Census years covered**: [1850, 1860, 1870, 1880]


## Merge Coverage Rates

| Metric | Rate |
|--------|------|
| voyage_labor_coverage | 39.9% |
| voyage_route_coverage | 9.2% |
| voyage_vqi_coverage | 0.0% |
| captain_census_link_rate | nan% |

## Validation Checks


### Voyage Validations

| Check | Status | Details |
|-------|--------|---------|
| voyage_id_uniqueness | ✅ Pass | 0 duplicate voyage_ids found |
| date_consistency | ❌ Fail | 1 voyages with date_in < date_out |
| oil_quantity_range | ❌ Fail | 0 negative, 2 extreme (>10000) |
| bone_quantity_range | ❌ Fail | 0 negative, 1 extreme (>100000) |
| desertion_rate_bounds | ✅ Pass | 0 values outside [0,1] |
| duration_range | ❌ Fail | 2463 too short (<30d), 49 too long (>2000d) |
| year_range | ❌ Fail | 10 years outside whaling era (1700-1930) |

### Captain-Year Validations

| Check | Status | Details |
|-------|--------|---------|
| captain_year_uniqueness | ✅ Pass | 0 duplicate (captain_id, census_year) pairs |
| voyage_count_positive | ✅ Pass | 0 captain-years with non-positive voyage count |

## Missingness Summary (Voyage)

| Column | Missing Rate |
|--------|--------------|
| port_in | 100.0% |
| vqi_proxy | 100.0% |
| vqi_asof_year | 100.0% |
| std_lat | 90.9% |
| std_lon | 90.9% |
| days_observed | 90.8% |
| arctic_days | 90.8% |
| bering_days | 90.8% |
| mean_lat | 90.8% |
| mean_lon | 90.8% |
| min_lat | 90.8% |
| max_lat | 90.8% |
| min_lon | 90.8% |
| max_lon | 90.8% |
| min_year | 90.8% |

## Key Variable Distributions

### Oil Output (q_oil_bbl)


- Count: 15,687
- Mean: 1,082
- Median: 530
- Max: 1,702,754


## Recommendations

1. Review any failed validation checks above
2. Consider thresholds for captain-census linkage based on your analysis needs
3. Document any data quality issues discovered during analysis
