# Venture Capital of the Sea — Whaling Data Pipeline

A comprehensive data pipeline for assembling, analyzing, and linking American whaling voyage data with census records for captain wealth analysis. This project implements an AKM-style variance decomposition to quantify the contributions of captain skill and agent quality to voyage productivity in 19th-century offshore whaling.

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Data Sources](#data-sources)
- [Pipeline Architecture](#pipeline-architecture)
- [Module Reference](#module-reference)
  - [Configuration (`src/config.py`)](#configuration-srcconfigpy)
  - [Download Module (`src/download/`)](#download-module-srcdownload)
  - [Parsing Module (`src/parsing/`)](#parsing-module-srcparsing)
  - [Entities Module (`src/entities/`)](#entities-module-srcentities)
  - [Aggregation Module (`src/aggregation/`)](#aggregation-module-srcaggregation)
  - [Linkage Module (`src/linkage/`)](#linkage-module-srclinkage)
  - [Assembly Module (`src/assembly/`)](#assembly-module-srcassembly)
  - [Quality Assurance (`src/qa/`)](#quality-assurance-srcqa)
  - [AKM Analysis (`src/akm_analysis.py`)](#akm-analysis-srcakm_analysispy)
  - [Analyses Module (`src/analyses/`)](#analyses-module-srcanalyses)
- [Output Files](#output-files)
- [Project Structure](#project-structure)
- [Configuration Reference](#configuration-reference)
- [Citation](#citation)
- [License](#license)

---

## Overview

This pipeline pulls, cleans, standardizes, and merges all online-available whaling, census, and vessel proxy data into analysis-ready datasets. The primary outputs are:

| File | Description |
|------|-------------|
| `analysis_voyage.parquet` | Voyage-level outcomes with crew metrics, route exposure, and vessel quality |
| `analysis_captain_year.parquet` | Captain-year wealth panel linking whaling outcomes to census wealth |
| `analysis_voyage_augmented.parquet` | Enhanced voyage data with additional historical sources |

The analyses module implements a 17-regression empirical suite (R1–R17) for estimating value creation, organizational intermediation, risk pass-through, and labor market dynamics.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python run_pipeline.py all

# Or run individual stages
python run_pipeline.py download          # Download raw data
python run_pipeline.py parse             # Parse and standardize
python run_pipeline.py aggregate         # Compute metrics
python run_pipeline.py assemble-voyages  # Build voyage dataset
python run_pipeline.py link-captains     # Census linkage
python run_pipeline.py assemble-captains # Build captain panel
python run_pipeline.py qa                # Quality assurance

# Run the Online Voyage Augmentation Pack
python run_pipeline.py augment-all       # Full augmentation pipeline

# Run empirical analyses
python -m src.analyses.run_all           # Run all R1-R17 regressions
python -m src.analyses.run_all --quick   # Run main text regressions only
```

---

## Data Sources

| Source | URL | Format | License |
|--------|-----|--------|---------|
| AOWV Voyages | whalinghistory.org | ZIP/CSV | CC BY 4.0 |
| AOWV Crew Lists | whalinghistory.org | ZIP/CSV | CC BY 4.0 |
| AOWV Logbooks | whalinghistory.org | ZIP/CSV | CC BY 4.0 |
| Starbuck (1878) | archive.org | PDF/OCR | Public Domain |
| Maury Logbooks | whalinghistory.org | ZIP/CSV | CC BY 4.0 |
| Mutual Marine Register | archive.org | PDF/OCR | CC BY-NC-SA 4.0 |
| IPUMS USA Full Count | usa.ipums.org | Extract | IPUMS Terms |

> **Note**: IPUMS data requires manual extract creation. See [docs/ipums_extract_instructions.md](docs/ipums_extract_instructions.md).

---

## Pipeline Architecture

The pipeline consists of 16 sequential stages organized into three major phases:

### Phase 1: Core Pipeline (Stages 1–9)

```
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 1: Download          ──▶  Raw data acquisition              │
│  Stage 2: Parse             ──▶  Standardization & normalization   │
│  Stages 3-5: Aggregate      ──▶  Labor, route, vessel metrics      │
│  Stage 6: Assemble Voyages  ──▶  analysis_voyage.parquet           │
│  Stages 7-8: Link Captains  ──▶  IPUMS census matching             │
│  Stage 9: Assemble Captains ──▶  analysis_captain_year.parquet     │
└─────────────────────────────────────────────────────────────────────┘
```

### Phase 2: Online Voyage Augmentation Pack (Stages 10–16)

```
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 10: Download Online  ──▶  Starbuck, Maury, WSL sources      │
│  Stage 11: Extract WSL      ──▶  PDF event extraction              │
│  Stage 12: Crosswalk WSL    ──▶  Map events to voyages             │
│  Stage 13: Starbuck         ──▶  Parse & reconcile with AOWV       │
│  Stage 14: Maury            ──▶  Logbook validation                │
│  Stage 16: Augment          ──▶  analysis_voyage_augmented.parquet │
└─────────────────────────────────────────────────────────────────────┘
```

### Phase 3: Empirical Analysis Suite

```
┌─────────────────────────────────────────────────────────────────────┐
│  R1:  Baseline Production   ──▶  AKM decomposition                 │
│  R2:  Agent Effect          ──▶  Agent fixed effects only          │
│  R3:  Captain Effect        ──▶  Captain fixed effects only        │
│  R4-R5: Portability         ──▶  Out-of-sample validation          │
│  R6-R8: Event Studies       ──▶  Captain switching events          │
│  R9-R12: Complementarity    ──▶  Match quality estimation          │
│  R13-R15: Shock Analysis    ──▶  Risk pass-through                 │
│  R16-R17: Strategy          ──▶  Route and ground choice           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Module Reference

### Configuration (`src/config.py`)

Central configuration module that defines all project-wide constants and settings.

**Key Classes:**

| Class | Description |
|-------|-------------|
| `Units` | Standard measurement units (barrels, lbs, nominal USD) |
| `StringNormConfig` | Name normalization rules (case, punctuation, abbreviations) |
| `CrosswalkConfig` | Fuzzy matching parameters (date tolerance, name similarity) |
| `LinkageConfig` | Census linkage thresholds (strict: 0.90, medium: 0.70) |
| `ValidationConfig` | Data validation bounds (max oil barrels, voyage duration) |

**Key Constants:**

```python
ARCTIC_BOUNDARY = 66.5          # Latitude defining Arctic waters
DATE_TOLERANCE_DAYS = 30        # Tolerance for voyage date matching
MIN_NAME_SIMILARITY = 0.85      # Jaro-Winkler threshold
```

---

### Download Module (`src/download/`)

Handles data acquisition from online sources with SHA256 hash tracking for reproducibility.

| File | Purpose |
|------|---------|
| `aowv_downloader.py` | Downloads AOWV voyages, crew lists, and logbooks from whalinghistory.org |
| `archive_downloader.py` | Downloads PDF/OCR content from archive.org |
| `online_sources_downloader.py` | Downloads Starbuck, Maury, Townsend, and COML sources |
| `wsl_pdf_downloader.py` | Downloads Whalemen's Shipping List PDFs |
| `manifest.py` | `ManifestManager` class for tracking download provenance |

**Key Functions:**

```python
from src.download import aowv_downloader

# Download all AOWV data with manifest tracking
aowv_downloader.download_all_aowv_data(force=False)
```

---

### Parsing Module (`src/parsing/`)

Parses and standardizes raw data according to project conventions.

| File | Purpose |
|------|---------|
| `string_normalizer.py` | Name/text normalization with Jaro-Winkler similarity and Soundex |
| `voyage_parser.py` | Parses AOWV voyage records |
| `crew_parser.py` | Parses crew lists with role extraction |
| `logbook_parser.py` | Parses logbook entries with coordinate extraction |
| `register_parser.py` | Parses vessel registry/insurance records |
| `starbuck_parser.py` | Parses Starbuck (1878) historical data |
| `maury_parser.py` | Parses Maury logbook coordinates |
| `wsl_pdf_parser.py` | OCR text extraction from Whalemen's Shipping List |
| `wsl_event_extractor.py` | Event extraction (departures, arrivals, wrecks) |

**String Normalization Conventions:**

- **Case**: UPPER
- **Punctuation**: Stripped
- **Whitespace**: Collapsed
- **Titles**: Removed (CAPT, MR, DR, etc.)
- **Suffixes**: Preserved (JR, SR)
- **Abbreviations**: Expanded (WM → WILLIAM, CHAS → CHARLES)

**Key Functions:**

```python
from src.parsing.string_normalizer import normalize_name, jaro_winkler_similarity

# Normalize captain names
name = normalize_name("Capt. Wm. H. Plaskett Jr.")  # → "WILLIAM H PLASKETT JR"

# Compute similarity
score = jaro_winkler_similarity("PLASKETT", "PLASKET")  # → 0.96
```

---

### Entities Module (`src/entities/`)

Creates deterministic entity identifiers for vessels, captains, and agents.

| File | Purpose |
|------|---------|
| `entity_resolver.py` | `EntityResolver` class for ID generation |
| `crosswalk_builder.py` | Builds entity crosswalks linking different sources |
| `wsl_voyage_matcher.py` | Matches WSL events to AOWV voyages |
| `maury_voyage_matcher.py` | Matches Maury logbook entries to voyages |
| `starbuck_reconciler.py` | Reconciles Starbuck records with AOWV |

**Entity ID Format:**

| Entity | ID Pattern | Components |
|--------|------------|------------|
| Vessel | `V_xxxxxx` | Normalized name + home port + rig type |
| Captain | `C_xxxxxx` | Normalized name + career start decade + modal port |
| Agent | `A_xxxxxx` | Normalized name + operating port |

**Key Usage:**

```python
from src.entities.entity_resolver import EntityResolver

resolver = EntityResolver()
captain_id = resolver.resolve_captain("WILLIAM PLASKETT", career_start_decade=1820, modal_port="NEW BEDFORD")
# → "C_a3b8d1"
```

---

### Aggregation Module (`src/aggregation/`)

Computes derived metrics for voyages.

| File | Purpose |
|------|---------|
| `labor_metrics.py` | Crew count, desertion rate, role composition |
| `route_exposure.py` | Arctic exposure, mean lat/lon, voyage duration |

**Labor Metrics:**

| Metric | Formula |
|--------|---------|
| `crew_count` | Count of unique crew members |
| `desertion_rate` | Deserters / Total crew |
| `discharge_rate` | Discharged / Total crew |
| `death_rate` | Deaths / Total crew |

**Route Exposure Metrics:**

| Metric | Formula |
|--------|---------|
| `frac_days_in_arctic_polygon` | Days above 66.5°N / Total voyage days |
| `mean_latitude` | Average of all logbook coordinates |
| `mean_longitude` | Average of all logbook coordinates |

---

### Linkage Module (`src/linkage/`)

Links captains to census records using probabilistic matching.

| File | Purpose |
|------|---------|
| `captain_profiler.py` | Builds captain career profiles with expected ages |
| `ipums_loader.py` | Loads and prepares IPUMS census extracts |
| `record_linker.py` | `RecordLinker` class for probabilistic matching |

**Matching Algorithm:**

1. **Blocking**: Geography (state) + age band (±5 years)
2. **Scoring**: Weighted combination of:
   - Name similarity (Jaro-Winkler): 40%
   - Age penalty: 25%
   - Geography match: 20%
   - Occupation boost (maritime occupations): 10%
   - Spouse validation: 5%
3. **Thresholds**:
   - Strict: ≥ 0.90
   - Medium: ≥ 0.70

**Key Usage:**

```python
from src.linkage.record_linker import RecordLinker

linker = RecordLinker()
linkage_df = linker.link_captains_to_census(captain_profiles, census_data, target_year=1850)
```

---

### Assembly Module (`src/assembly/`)

Assembles final analysis-ready datasets.

| File | Purpose |
|------|---------|
| `voyage_assembly.py` | Builds `analysis_voyage.parquet` |
| `captain_assembly.py` | Builds `analysis_captain_year.parquet` |
| `voyage_augmentor.py` | Builds `analysis_voyage_augmented.parquet` |

**Assembly Logic:**

- Left joins all metrics to voyages via deterministic entity IDs
- As-of merge for vessel quality (most recent rating before voyage)
- Aggregates voyage outcomes to captain-year level
- Merges census wealth data via linkage keys

---

### Quality Assurance (`src/qa/`)

Validates data quality and generates diagnostic reports.

| File | Purpose |
|------|---------|
| `validators.py` | Data validation checks (bounds, completeness, consistency) |
| `reporters.py` | Generates QA reports with coverage statistics |

**Validation Checks:**

| Check | Rule |
|-------|------|
| Oil production | 0 ≤ q_oil_bbl ≤ 10,000 |
| Bone production | 0 ≤ q_bone_lbs ≤ 100,000 |
| Voyage duration | 30 ≤ days ≤ 2,000 |
| Desertion rate | 0.0 ≤ rate ≤ 1.0 |

---

### AKM Analysis (`src/akm_analysis.py`)

Standalone script for AKM (Abowd-Kramarz-Margolis) variance decomposition.

**Model Specification:**

```
log(Q_v) = θ_captain + ψ_agent + β₁·log(tonnage) + β₂·log(duration) + γ_decade + ε
```

**Key Functions:**

| Function | Purpose |
|----------|---------|
| `load_and_prepare_data()` | Loads voyage data with sample restrictions |
| `find_connected_set()` | Finds largest connected component in bipartite graph |
| `find_leave_one_out_connected_set()` | Computes LOO connected set for KSS estimation |
| `estimate_akm()` | Sparse least squares estimation |
| `compute_kss_correction()` | KSS (2020) bias correction for variance components |
| `variance_decomposition()` | Full variance decomposition table |
| `analyze_matching_patterns()` | Captain-agent sorting analysis |

**Output (within captain+agent labor market variance):**

- Captain skill (α): **14.2%**
- Agent capability (γ): **95.4%**
- Sorting (2×Cov): **-9.7%**
- Corr(α, γ) = -0.131 (negative = substitutes)

---

### Analyses Module (`src/analyses/`)

Comprehensive empirical analysis suite implementing 17 regression specifications.

| File | Purpose |
|------|---------|
| `config.py` | Regression specifications, fixed effects, sample filters |
| `data_loader.py` | Data preparation and sample construction |
| `connected_set.py` | Leave-one-out connected set algorithms |
| `baseline_production.py` | R1: AKM production function |
| `portability.py` | R4-R5: Out-of-sample skill portability |
| `event_study.py` | R6-R8: Captain switching event studies |
| `complementarity.py` | R9-R12: Match quality and complementarity |
| `shock_analysis.py` | R13-R15: Climate shock pass-through |
| `strategy.py` | R16-R17: Route and ground choice |
| `labor_market.py` | Labor market dynamics |
| `extensions.py` | Robustness and sensitivity analyses |
| `output_generator.py` | LaTeX tables and figures |
| `run_all.py` | Master orchestration script |

**Regression Specifications:**

| ID | Name | Fixed Effects | Interpretation |
|----|------|---------------|----------------|
| R1 | Baseline Production | Captain + Agent + Vessel×Period + Route×Time | Full AKM decomposition |
| R2 | Agent Effect | Agent + Vessel×Period + Route×Time | Agent quality without captain |
| R3 | Captain Effect | Captain + Vessel×Period + Route×Time | Captain skill without agent |
| R4 | Pre-1870 Training | Captain + Agent (1780–1870) | In-sample estimation |
| R5 | Post-1870 Validation | Captain (predicted) → Post-1870 | Out-of-sample portability |
| R6 | Switch Event | Captain + Voyage around switch | Around-switch design |
| R9 | Complementarity | Captain × Agent interactions | Match quality effects |
| R13 | Shock Pass-through | Arctic exposure × shocks | Risk allocation |

**Usage:**

```python
from src.analyses.run_all import run_all_analyses

results = run_all_analyses(quick=False, save_outputs=True)
```

---

## Output Files

### `analysis_voyage.parquet`

| Column | Type | Description |
|--------|------|-------------|
| `voyage_id` | str | Unique voyage identifier |
| `vessel_id` | str | Vessel entity ID |
| `captain_id` | str | Captain entity ID |
| `agent_id` | str | Agent entity ID |
| `year_out` | int | Departure year |
| `year_in` | int | Return year |
| `q_oil_bbl` | float | Oil production (barrels) |
| `q_bone_lbs` | float | Bone production (lbs) |
| `log_q` | float | Log total production |
| `desertion_rate` | float | Crew desertion fraction |
| `frac_days_in_arctic_polygon` | float | Arctic exposure |
| `vqi_proxy` | float | Vessel quality index |

### `analysis_captain_year.parquet`

| Column | Type | Description |
|--------|------|-------------|
| `captain_id` | str | Captain entity ID |
| `census_year` | int | Census year (1850–1880) |
| `HIK` | str | IPUMS historical key |
| `whaling_total_oil_bbl` | float | 10-year cumulative oil production |
| `voyage_count` | int | Number of voyages |
| `REALPROP` | float | Census real property value |
| `PERSPROP` | float | Census personal property value |
| `match_probability` | float | Linkage confidence score |

---

## Project Structure

```
Whaling/
├── src/
│   ├── config.py              # Global configuration
│   ├── akm_analysis.py        # Standalone AKM decomposition
│   ├── download/              # Data acquisition (5 modules)
│   │   ├── aowv_downloader.py
│   │   ├── archive_downloader.py
│   │   ├── online_sources_downloader.py
│   │   ├── wsl_pdf_downloader.py
│   │   └── manifest.py
│   ├── parsing/               # Parsing & normalization (9 modules)
│   │   ├── string_normalizer.py
│   │   ├── voyage_parser.py
│   │   ├── crew_parser.py
│   │   ├── logbook_parser.py
│   │   ├── register_parser.py
│   │   ├── starbuck_parser.py
│   │   ├── maury_parser.py
│   │   ├── wsl_pdf_parser.py
│   │   └── wsl_event_extractor.py
│   ├── entities/              # Entity resolution (5 modules)
│   │   ├── entity_resolver.py
│   │   ├── crosswalk_builder.py
│   │   ├── wsl_voyage_matcher.py
│   │   ├── maury_voyage_matcher.py
│   │   └── starbuck_reconciler.py
│   ├── aggregation/           # Metric computation
│   │   ├── labor_metrics.py
│   │   └── route_exposure.py
│   ├── linkage/               # Census linkage
│   │   ├── captain_profiler.py
│   │   ├── ipums_loader.py
│   │   └── record_linker.py
│   ├── assembly/              # Final table assembly
│   │   ├── voyage_assembly.py
│   │   ├── captain_assembly.py
│   │   └── voyage_augmentor.py
│   ├── qa/                    # Quality assurance
│   │   ├── validators.py
│   │   └── reporters.py
│   └── analyses/              # Empirical analysis suite (13 modules)
│       ├── config.py
│       ├── data_loader.py
│       ├── connected_set.py
│       ├── baseline_production.py
│       ├── portability.py
│       ├── event_study.py
│       ├── complementarity.py
│       ├── shock_analysis.py
│       ├── strategy.py
│       ├── labor_market.py
│       ├── extensions.py
│       ├── output_generator.py
│       └── run_all.py
├── data/
│   ├── raw/                   # Downloaded source files
│   ├── staging/               # Intermediate tables
│   ├── final/                 # Output analysis files
│   └── crosswalks/            # Entity crosswalks
├── output/
│   ├── tables/                # LaTeX regression tables
│   ├── figures/               # PNG figures
│   ├── diagnostics/           # AKM diagnostics
│   └── summary/               # Executive summaries
├── docs/
│   ├── data_dictionary.md     # Variable definitions
│   ├── ipums_extract_instructions.md
│   └── qa_report.md           # Quality metrics
├── tests/                     # Test suite
├── run_pipeline.py            # Main runner (CLI)
├── manifest.jsonl             # Download provenance
└── requirements.txt           # Python dependencies
```

---

## Configuration Reference

### Sample Configuration

```python
from src.analyses.config import SampleConfig

sample = SampleConfig(
    min_year=1780,              # Earliest voyage year
    max_year=1930,              # Latest voyage year
    min_captain_voyages=1,      # Minimum voyages per captain
    min_agent_voyages=1,        # Minimum voyages per agent
    oos_cutoff_year=1870,       # Out-of-sample training cutoff
    period_bin_years=5,         # Period binning width
    event_study_min_pre=2,      # Minimum pre-event voyages
    event_study_min_post=2,     # Minimum post-event voyages
    output_trim_lower_pct=0.5,  # Production trimming (lower)
    output_trim_upper_pct=99.5, # Production trimming (upper)
)
```

### Linkage Configuration

```python
from src.config import LinkageConfig

linkage = LinkageConfig(
    target_years=[1850, 1860, 1870, 1880],
    strict_threshold=0.90,
    medium_threshold=0.70,
    name_weight=0.40,
    age_weight=0.25,
    geo_weight=0.20,
    occ_weight=0.10,
    spouse_weight=0.05,
)
```

---

## Citation

When using this pipeline, please cite:

> Starbuck, Alexander. *History of the American Whale Fishery from its Earliest Inception to the Year 1876*. 1878.

> American Offshore Whaling Voyages Database. New Bedford Whaling Museum and Mystic Seaport Museum. WhalingHistory.org.

> Steven Ruggles, Sarah Flood, Ronald Goeken, et al. IPUMS USA: Version 11.0 [dataset]. Minneapolis, MN: IPUMS, 2021.

---

## License

MIT License

---

## Acknowledgments

- New Bedford Whaling Museum
- Mystic Seaport Museum
- Nantucket Historical Association
- IPUMS, University of Minnesota
- Census of Marine Life (CoML)
