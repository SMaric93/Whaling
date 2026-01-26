# Venture Capital of the Sea — Whaling Data Pipeline

A comprehensive data pipeline for assembling and merging American whaling voyage data with census records for captain wealth analysis.

## Overview

This pipeline pulls, cleans, standardizes, and merges all online-available whaling, census, and vessel proxy data into two analysis files:

1. **analysis_voyage.parquet** — Voyage-level outcomes with crew metrics, route exposure, and vessel quality
2. **analysis_captain_year.parquet** — Captain-year wealth panel linking whaling outcomes to census wealth

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python run_pipeline.py all

# Or run individual stages
python run_pipeline.py download
python run_pipeline.py parse
python run_pipeline.py aggregate
python run_pipeline.py assemble-voyages
python run_pipeline.py link-captains
python run_pipeline.py assemble-captains
python run_pipeline.py qa
```

## Data Sources

| Source | URL | Format | License |
|--------|-----|--------|---------|
| AOWV Voyages | whalinghistory.org | ZIP/CSV | CC BY 4.0 |
| AOWV Crew Lists | whalinghistory.org | ZIP/CSV | CC BY 4.0 |
| AOWV Logbooks | whalinghistory.org | ZIP/CSV | CC BY 4.0 |
| Mutual Marine Register | archive.org | PDF/OCR | CC BY-NC-SA 4.0 |
| IPUMS USA Full Count | usa.ipums.org | Extract | IPUMS Terms |

> **Note**: IPUMS data requires manual extract creation. See [docs/ipums_extract_instructions.md](docs/ipums_extract_instructions.md).

## Pipeline Stages

### Stage 1: Download Raw Data

Downloads and freezes source data with SHA256 hash tracking.

### Stage 2: Parse and Standardize

- Parses voyage, crew, and logbook data
- Normalizes names (UPPER, expand abbreviations, strip punctuation)
- Standardizes dates (YYYY-MM-DD)
- Creates deterministic entity IDs (vessel_id, captain_id, agent_id)

### Stage 3-5: Aggregate Metrics

- **Labor metrics**: crew_count, desertion_rate
- **Route exposure**: Arctic days, mean latitude/longitude
- **Vessel quality proxy**: Registry ratings from insurance records

### Stage 6: Build analysis_voyage

Merges voyage data with all metrics via left joins and as-of merge for vessel quality.

### Stage 7-8: Captain-Census Linkage

- Builds captain profiles with career spans and expected ages
- Links to IPUMS census via probabilistic matching (Jaro-Winkler + age + geography)
- Outputs match scores and sensitivity variants

### Stage 9: Build analysis_captain_year

Aggregates voyage outcomes to captain-year and merges with census wealth data.

## Project Structure

```
Whaling/
├── src/
│   ├── config.py              # Global configuration
│   ├── download/              # Data acquisition
│   ├── parsing/               # Parsing and normalization
│   ├── entities/              # Entity ID resolution
│   ├── aggregation/           # Metric computation
│   ├── linkage/               # Census linkage
│   ├── assembly/              # Final table assembly
│   └── qa/                    # Quality assurance
├── data/
│   ├── raw/                   # Downloaded source files
│   ├── staging/               # Intermediate tables
│   ├── final/                 # Output analysis files
│   └── crosswalks/            # Entity crosswalks
├── docs/
│   ├── data_dictionary.md     # Variable definitions
│   ├── ipums_extract_instructions.md
│   └── qa_report.md           # Quality metrics
├── run_pipeline.py            # Main runner
├── manifest.jsonl             # Download provenance
└── requirements.txt
```

## Output Files

### analysis_voyage.parquet

| Column | Description |
|--------|-------------|
| voyage_id | Unique voyage identifier |
| vessel_id, captain_id | Entity identifiers |
| q_oil_bbl, q_bone_lbs | Production quantities |
| desertion_rate | Crew desertion fraction |
| frac_days_in_arctic_polygon | Arctic exposure |
| vqi_proxy | Vessel quality index |

### analysis_captain_year.parquet

| Column | Description |
|--------|-------------|
| captain_id | Captain identifier |
| census_year | Census year (1850-1880) |
| HIK | IPUMS historical key |
| whaling_total_oil_bbl | 10-year prior oil production |
| REALPROP, PERSPROP | Census wealth variables |
| match_probability | Linkage confidence |

## Configuration

Key parameters in `src/config.py`:

- **Date tolerance**: ±30 days for voyage matching
- **Name similarity threshold**: 0.85 Jaro-Winkler
- **Linkage thresholds**: 0.90 (strict), 0.70 (medium)
- **Arctic boundary**: 66.5°N

## Citation

When using this pipeline, please cite:

> Starbuck, Alexander. *History of the American Whale Fishery*. 1878.
> American Offshore Whaling Voyages Database. WhalingHistory.org.
> IPUMS USA. Minneapolis, MN: IPUMS, [Year].

## License

MIT License

## Acknowledgments

- New Bedford Whaling Museum
- Mystic Seaport Museum
- Nantucket Historical Association
- IPUMS, University of Minnesota
