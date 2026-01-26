# Data Dictionary

This document defines all variables in the final analysis files.

## analysis_voyage.parquet

Voyage-level analysis file combining voyage records with crew metrics, route exposure, and vessel quality proxy.

### Core Identifiers

| Variable | Type | Description | Source |
|----------|------|-------------|--------|
| `voyage_id` | string | Unique voyage identifier | AOWV or computed |
| `vessel_id` | string | Vessel entity identifier | Computed from vessel_name + port |
| `captain_id` | string | Captain entity identifier | Computed from captain_name + career |
| `agent_id` | string | Agent entity identifier | Computed from agent_name + port |

### Vessel Information

| Variable | Type | Description | Source |
|----------|------|-------------|--------|
| `vessel_name_raw` | string | Original vessel name | AOWV |
| `vessel_name_clean` | string | Normalized vessel name | Computed |
| `rig` | string | Vessel rig type (ship, bark, brig, etc.) | AOWV |
| `tonnage` | float | Vessel tonnage | AOWV |
| `home_port` | string | Vessel home port | AOWV |

### Captain/Agent Information

| Variable | Type | Description | Source |
|----------|------|-------------|--------|
| `captain_name_raw` | string | Original captain name | AOWV |
| `captain_name_clean` | string | Normalized captain name | Computed |
| `agent_name_raw` | string | Original agent name | AOWV |
| `agent_name_clean` | string | Normalized agent name | Computed |

### Voyage Dates and Duration

| Variable | Type | Description | Source |
|----------|------|-------------|--------|
| `date_out` | date | Departure date | AOWV |
| `date_in` | date | Return date | AOWV |
| `year_out` | int | Departure year | Derived |
| `year_in` | int | Return year | Derived |
| `duration_days` | int | Voyage duration in days | Computed |

### Ports and Routes

| Variable | Type | Description | Source |
|----------|------|-------------|--------|
| `port_out` | string | Departure port | AOWV |
| `port_in` | string | Return port | AOWV |
| `ground_or_route` | string | Whaling ground/destination | AOWV |
| `route_year_cell` | string | Ground × Year identifier | Computed |

### Production Quantities

| Variable | Type | Units | Description | Source |
|----------|------|-------|-------------|--------|
| `q_sperm_bbl` | float | barrels | Sperm oil production | AOWV |
| `q_whale_bbl` | float | barrels | Whale oil production | AOWV |
| `q_oil_bbl` | float | barrels | Total oil (sperm + whale) | Computed |
| `q_bone_lbs` | float | pounds | Whalebone production | AOWV |
| `q_total_index` | float | index | Weighted production index | Computed |

### Crew/Labor Metrics

| Variable | Type | Description | Source |
|----------|------|-------------|--------|
| `crew_count` | int | Total crew members | Crew lists |
| `desertion_count` | int | Crew who deserted | Crew lists |
| `desertion_rate` | float | Desertion count / crew count | Computed |
| `mean_age` | float | Mean crew age (if available) | Crew lists |
| `labor_data_quality` | string | Data quality flag (good/partial/sparse) | Computed |

### Route Exposure Metrics

| Variable | Type | Description | Source |
|----------|------|-------------|--------|
| `days_observed` | int | Logbook observation count | Logbooks |
| `frac_days_in_arctic_polygon` | float | Fraction of days north of 66.5°N | Computed |
| `frac_days_in_bering` | float | Fraction of days in Bering Sea | Computed |
| `mean_lat` | float | Mean latitude of voyage track | Computed |
| `mean_lon` | float | Mean longitude of voyage track | Computed |
| `voyage_region` | string | Classified voyage region | Computed |
| `route_data_quality` | string | Data quality flag | Computed |

### Vessel Quality Proxy

| Variable | Type | Description | Source |
|----------|------|-------------|--------|
| `vqi_proxy` | float | Vessel quality index [0-1] | Insurance register |
| `vqi_asof_year` | int | Year of matched register entry | Computed |
| `vqi_asof_exact` | bool | Whether year match is exact | Computed |
| `vqi_asof_extrapolated` | bool | Whether VQI is extrapolated | Computed |

### Coverage Flags

| Variable | Type | Description |
|----------|------|-------------|
| `has_labor_data` | bool | Has crew/desertion data |
| `has_route_data` | bool | Has logbook position data |
| `has_vqi_data` | bool | Has vessel quality proxy |

---

## analysis_captain_year.parquet

Captain-year panel combining aggregated whaling outcomes with census wealth data.

### Identifiers

| Variable | Type | Description | Source |
|----------|------|-------------|--------|
| `captain_id` | string | Captain entity identifier | Voyage data |
| `census_year` | int | Census year (1850, 1860, 1870, 1880) | Panel structure |
| `HIK` | string | IPUMS Historical Individual Key | Linkage |

### Aggregated Whaling Outcomes

| Variable | Type | Description | Window |
|----------|------|-------------|--------|
| `whaling_voyages_count` | int | Total voyages | 10 years prior |
| `whaling_total_oil_bbl` | float | Total oil production (bbl) | 10 years prior |
| `whaling_total_bone_lbs` | float | Total bone production (lbs) | 10 years prior |
| `whaling_mean_oil_per_voyage` | float | Mean oil per voyage | 10 years prior |
| `whaling_mean_duration_days` | float | Mean voyage duration | 10 years prior |
| `whaling_mean_vqi_proxy` | float | Mean vessel quality | 10 years prior |
| `whaling_mean_desertion_rate` | float | Mean crew desertion rate | 10 years prior |
| `whaling_arctic_exposure` | float | Mean Arctic exposure | 10 years prior |

### Career Information

| Variable | Type | Description |
|----------|------|-------------|
| `whaling_first_voyage_year` | int | First voyage in career |
| `whaling_last_voyage_year` | int | Last voyage as of census |
| `whaling_career_years_as_of` | int | Career length at census |
| `window_years` | int | Aggregation window used |

### Linkage Quality

| Variable | Type | Description |
|----------|------|-------------|
| `link_year` | int | Year of census link |
| `match_probability` | float | Linkage confidence [0-1] |
| `match_method` | string | deterministic or probabilistic |

### Census Demographics (from IPUMS)

| Variable | Type | Description |
|----------|------|-------------|
| `NAME_CLEAN` | string | Normalized name from census |
| `AGE` | int | Age at census |
| `STATEFIP` | int | State FIPS code |
| `COUNTY` | string | County |
| `OCC` | string | Occupation |

### Wealth Variables (from IPUMS)

| Variable | Type | Units | Description | Years Available |
|----------|------|-------|-------------|-----------------|
| `REALPROP` | float | USD | Real estate value | 1850-1870 |
| `PERSPROP` | float | USD | Personal property value | 1860-1870 |
| `total_wealth` | float | USD | REALPROP + PERSPROP | Computed |

### Coverage Flags

| Variable | Type | Description |
|----------|------|-------------|
| `has_census_link` | bool | Successfully linked to census |
| `has_wealth_data` | bool | Has REALPROP or PERSPROP |
| `has_voyage_data` | bool | Has whaling voyage data |

---

## Sensitivity Variants

Captain-year files are produced in three variants based on linkage threshold:

| File | Threshold | Use Case |
|------|-----------|----------|
| `analysis_captain_year_strict_links.parquet` | ≥0.90 | High-confidence subset |
| `analysis_captain_year_medium_links.parquet` | ≥0.70 | Balanced coverage/quality |
| `analysis_captain_year_all_links.parquet` | Best match | Maximum coverage |

---

## Data Sources

| Source | Provenance | License |
|--------|------------|---------|
| AOWV Voyages | whalinghistory.org | CC BY 4.0 |
| AOWV Crew Lists | whalinghistory.org | CC BY 4.0 |
| AOWV Logbooks | whalinghistory.org | CC BY 4.0 |
| Mutual Marine Register | Archive.org | CC BY-NC-SA 4.0 |
| IPUMS USA | usa.ipums.org | IPUMS Terms of Use |
