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
  - [Compass Pipeline (`src/compass/`)](#compass-pipeline-srccompass)
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

# Run the complete 5-stage pipeline
python run_pipeline.py all

# Or run individual stages
python run_pipeline.py pull      # Stage 1: Download all data
python run_pipeline.py clean     # Stage 2: Parse and standardize
python run_pipeline.py merge     # Stage 3: Assemble and link datasets
python run_pipeline.py analyze   # Stage 4: Run full analysis suite
python run_pipeline.py output    # Stage 5: Generate MD and TEX tables

# Quick analysis (main regressions only)
python run_pipeline.py analyze --quick
```

### Exploratory & ML Analyses

The project includes additional exploratory analyses beyond the core regression suite:

```bash
# Quick analysis scripts (run from project root with venv activated)
source venv/bin/activate

# Load and explore data interactively
python -c "
import pandas as pd
df = pd.read_parquet('data/final/analysis_voyage_with_climate.parquet')
print(df.info())
print(df.describe())
"

# Captain fixed effects analysis
python -c "
import pandas as pd
fe = pd.read_csv('output/tables/r1_captain_effects.csv')
print(f'Captains: {len(fe)}')
print(fe['alpha_hat'].describe())
"

# Agent fixed effects analysis
python -c "
import pandas as pd
fe = pd.read_csv('output/tables/r1_agent_effects.csv')
print(f'Agents: {len(fe)}')
print(fe['gamma_hat'].describe())
"
```

### Key Data Files for Analysis

| File | Location | Description |
|------|----------|-------------|
| Main voyage data | `data/final/analysis_voyage_with_climate.parquet` | 11,622 voyages with outcomes |
| Captain FE | `output/tables/r1_captain_effects.csv` | α̂ estimates for 2,311 captains |
| Agent FE | `output/tables/r1_agent_effects.csv` | γ̂ estimates for 777 agents |
| Variance decomp | `output/tables/table_variance_decomposition.csv` | KSS-corrected shares |
| Concentration | `output/tables/agent_concentration_by_decade.csv` | HHI by decade |

### Analysis Summaries

After running analyses, find results in:

| File | Contents |
|------|----------|
| `output/summary/executive_summary.md` | Core R1-R17 findings |
| `output/summary/management_science_summary.md` | Paper-ready summary |
| `output/summary/complete_analysis_findings.md` | All 35+ exploratory analyses |
| `output/summary/extended_analyses_summary.md` | ML and pattern discovery |

### Key Columns for Analysis

```python
# Voyage-level
'voyage_id'         # Unique identifier
'captain_id'        # Captain entity ID
'agent_id'          # Agent entity ID
'sail_year'         # Departure year (derived from date_out)
'q_total_index'     # Production (barrels equivalent)
'tonnage'           # Vessel tonnage
'ground_or_route'   # Whaling ground/route
'home_port'         # Departure port
'rig'               # Vessel type (Ship, Bark, Brig, etc.)

# Fixed effects (after merge)
'alpha_hat'         # Captain skill estimate
'gamma_hat'         # Agent capability estimate
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

## Weather and Climate Data

The pipeline integrates historical climate data to control for environmental conditions and enable instrumental variable designs.

### Climate Data Sources

| Source | File | Coverage | Variables |
|--------|------|----------|-----------|
| **NAO** | `nao_annual.csv` | 1865–1920 (21%) | North Atlantic Oscillation index |
| **PDO** | `pdo_annual.csv` | 1854–1930 (34%) | Pacific Decadal Oscillation |
| **AMO** | `amo_annual.csv` | 1856–1930 (31%) | Atlantic Multidecadal Oscillation |
| **HURDAT** | `hurricane_annual.csv` | 1851–1920 (39%) | Atlantic hurricane counts, ACE |
| **NSIDC** | `sea_ice/*.csv` | 1850–1920 (40%) | NH ice extent, regional ice |
| **SILSO** | `sunspot_annual.csv` | 1700–1930 (99%) | Sunspot numbers |
| **Volcanic** | `volcanic_forcing_annual.csv` | 1750–1930 (98%) | Eruption forcing index |
| **HadCRUT5** | `global_temp_hadcrut.csv` | 1850–1930 (40%) | Global temperature anomaly |

### Key Climate Variables

```python
# Ice extent (regional)
'nh_ice_extent_mean'   # Northern Hemisphere mean ice extent
'bering_ice_mean'      # Bering Sea ice (most relevant for Arctic routes)
'chukchi_ice_mean'     # Chukchi Sea ice
'beaufort_ice_mean'    # Beaufort Sea ice

# Hurricane activity
'n_hurricanes'         # Count of Atlantic hurricanes per year
'ace_zscore'           # Accumulated Cyclone Energy (standardized)
'corridor_hurricane_days'  # Days with hurricanes in shipping corridors

# Ocean oscillations
'nao_index'            # North Atlantic Oscillation (-3 to +3)
'pdo_index'            # Pacific Decadal Oscillation
'amo_index'            # Atlantic Multidecadal Oscillation

# Other
'sunspot_number'       # Solar cycle indicator
'volcanic_forcing'     # Radiative forcing from major eruptions
'global_temp_anomaly'  # Global temperature anomaly (°C)
```

### Main Climate Effects

| Variable | β (standardized) | Interpretation |
|----------|------------------|----------------|
| **Bering Ice** | +0.30*** | More ice → higher output (whale concentration) |
| **ACE Hurricanes** | -0.17*** | Storm energy hurts voyage success |
| **AMO** | +0.23*** | Warm Atlantic → higher output |
| NAO | -0.03 | Not significant |
| PDO | -0.00 | Not significant |
| Sunspots | +0.003*** | Weak positive |

### Climate × Skill Interactions

| Interaction | β | Interpretation |
|-------------|---|----------------|
| Ice × Captain θ | +0.16** | High-skill captains exploit ice better |
| Ice × Agent ψ | -0.32*** | High-ψ agents suffer in high-ice years |

### Instrumental Variable Designs

The weather data enables several IV specifications:

#### 1. Shift-Share (Bartik) Exposure

```
Z_{a,t} = Σ_g (s_{a,g}^{pre} × Weather_{g,t})
```

- Agent's predetermined route mix × year-specific weather
- Instruments exploration intensity (F-stat ≈ 8.4)

#### 2. Ice Threshold Instrument

- High ice years → changes Arctic route feasibility
- First stage: Ice → Pr(Arctic) (t = 4.75***)

#### 3. Lagged Hurricane Exposure

- Last year's hurricanes → predicts agent switching
- First stage: Lag Hurr → Switch (t = -6.43***)

### Climate Output Files

| File | Contents |
|------|----------|
| `output/weather/weather_regressions.json` | Main climate regression results |
| `output/climate/comprehensive_climate_analysis.json` | Full climate analysis |
| `output/iv/iv_analysis_results.json` | IV estimation results |
| `output/weather/ice_vs_output_binned_scatter.png` | Visualization of ice effects |

### Running Weather Analyses

```bash
# Download weather data
python -c "from src.download.weather_downloader import download_all_weather; download_all_weather()"

# Run climate regressions
python -c "
from pathlib import Path
import pandas as pd

df = pd.read_parquet('data/final/analysis_voyage_with_climate.parquet')
print(f'Voyages with ice data: {df[\"bering_ice_mean\"].notna().sum():,}')
"
```

---

## Economic Data

The pipeline integrates historical economic data to control for demand-side factors affecting whaling productivity.

### Economic Data Sources

| Source | File | Coverage | Variables |
|--------|------|----------|-----------|
| **EIA Petroleum** | `petroleum_prices.csv` | 1860–1920 | US crude oil prices |
| **WSL Market Prices** | `wsl_price_quotes.parquet` | 1843–1914 | Whale oil, bone prices |

### Petroleum Prices

The 1859 Drake Well discovery marks the beginning of petroleum as a whale oil competitor. The `economic_downloader.py` module provides:

```python
# Key variables
'crude_oil_price_usd'  # Nominal $/barrel
'log_oil_price'        # For regression interpretation 
'oil_price_change'     # YoY percentage change
'oil_price_rel_1860'   # Relative to 1860 peak ($9.59)
'petroleum_era'        # Boolean: year >= 1860
```

**Key Story:** Oil crashed from $9.59/bbl (1860) to $0.49 (1861) after initial overproduction. By 1880s, oil was <$1/bbl—whale oil couldn't compete on price.

### WSL Market Prices

The `wsl_market_parser.py` module extracts commodity prices from Whalemen's Shipping List PDFs:

```python
from src.parsing.wsl_market_parser import extract_prices_from_issue
from src.parsing.wsl_pdf_parser import parse_wsl_issue

issue = parse_wsl_issue(Path("data/raw/wsl_pdfs/18651017.pdf"))
prices = extract_prices_from_issue(issue)
# Returns: sperm_oil, whale_oil, whalebone prices
```

### Running Economic Downloads

```bash
# Download petroleum prices
python src/download/economic_downloader.py

# Parse WSL market prices (requires PDFs)
python src/parsing/wsl_market_parser.py
```

## Pipeline Architecture

The pipeline is organized into **5 clear stages**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     5-STAGE DATA PIPELINE                          │
├─────────────────────────────────────────────────────────────────────┤
│  Stage 1: PULL      ──▶  Download AOWV, Starbuck, WSL, Weather     │
│  Stage 2: CLEAN     ──▶  Parse voyages, crew, logbooks, PDFs       │
│  Stage 3: MERGE     ──▶  Assemble voyage/captain datasets          │
│  Stage 4: ANALYZE   ──▶  R1-R17 regressions, counterfactuals       │
│  Stage 5: OUTPUT    ──▶  Generate paper tables (MD + TEX)          │
└─────────────────────────────────────────────────────────────────────┘
```

### Stage Details

| Stage | Command | Description |
|-------|---------|-------------|
| 1. PULL | `python run_pipeline.py pull` | Download AOWV, Starbuck, Maury, WSL, Weather, Economic data |
| 2. CLEAN | `python run_pipeline.py clean` | Parse and standardize all data sources |
| 3. MERGE | `python run_pipeline.py merge` | Entity resolution (Jaro-Winkler matching), linkage, assembly |
| 4. ANALYZE | `python run_pipeline.py analyze` | Run R1-R17 regressions, counterfactuals, robustness tests |
| 5. OUTPUT | `python run_pipeline.py output` | Generate all tables (Tables 1-6, A1-A8) in MD and TEX |

### Entity Matching (Stage 3)

The merge stage uses string-based entity matching with:

- **Jaro-Winkler similarity** for fuzzy name matching (threshold: 0.85)
- **Soundex** for phonetic matching
- **Name normalization**: Abbreviation expansion (WM → WILLIAM), case normalization, suffix preservation

### Empirical Analysis Suite (Stage 4)

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

### Pipeline Module (`src/pipeline/`)

**NEW**: The unified 5-stage pipeline orchestration module.

| File | Purpose |
|------|---------|
| `__init__.py` | Exports all stage functions |
| `stage1_pull.py` | Data acquisition (AOWV, Starbuck, Weather, etc.) |
| `stage2_clean.py` | Parsing and standardization |
| `stage3_merge.py` | Entity resolution, assembly, linkage (Jaro-Winkler matching) |
| `stage4_analyze.py` | Full analysis suite (R1-R17, counterfactuals) |
| `stage5_output.py` | MD and TEX output generation |
| `runner.py` | Full pipeline orchestration |

**Key Usage:**

```python
from src.pipeline import run_full_pipeline

# Run complete pipeline
run_full_pipeline()

# Or run individual stages
from src.pipeline import run_pull, run_clean, run_merge, run_analyze, run_output
run_pull()
run_clean()
run_merge()
run_analyze()
run_output()
```

---

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

### Compass Pipeline (`src/compass/`)

Micro-routing / search policy measurement pipeline. Converts vessel trajectories into regime-labelled step datasets, computes interpretable search-policy features, constructs PCA-based compass indices, and exports panel-ready datasets.

| Module | Purpose |
| --- | --- |
| `config.py` | `CompassConfig` dataclass + JSON I/O |
| `data_io.py` | Load, validate, and clean trajectory data |
| `preprocess.py` | UTM projection per voyage + optional smoothing |
| `steps.py` | Step construction, time resampling, distance thinning |
| `regimes.py` | Gaussian HMM regime segmentation (BIC model selection) |
| `features.py` | Feature suite: Hill tail index, MRL, grid coverage, loitering |
| `compass_index.py` | PCA index construction + early-window compass |
| `embedding_optional.py` | Self-supervised 1D-CNN embedding (requires `torch`) |
| `robustness.py` | Split-half ICC, step-def robustness, regime placebo, K sensitivity |
| `export.py` | Voyage-level + captain×year panels + diagnostics |
| `cli.py` | 10-step CLI orchestrator |

**Pipeline Steps:**

```
┌──────────────────────────────────────────────────────────────────────┐
│  Step 1:  Load & validate trajectories                              │
│  Step 2:  UTM projection + smoothing                                │
│  Step 3:  Step construction (raw, time-resampled, distance-thinned) │
│  Step 4:  HMM regime segmentation (transit / search / return)       │
│  Step 5:  Compass feature suite                                     │
│  Step 6:  PCA compass index (within-strata standardisation)         │
│  Step 7:  Early-window compass (reverse causality mitigation)       │
│  Step 8:  Self-supervised embedding (optional, requires torch)      │
│  Step 9:  Robustness checks (5 batteries)                           │
│  Step 10: Econometric panel exports                                 │
└──────────────────────────────────────────────────────────────────────┘
```

**Usage:**

```bash
# Full compass pipeline
python -m compass.cli --config compass_config.json

# Run specific steps only
python -m compass.cli --config compass_config.json --steps 1,2,3,4,5

# Dry run (validate config, no outputs)
python -m compass.cli --config compass_config.json --dry-run
```

**Key Outputs:**

| File | Description |
| --- | --- |
| `panel_voyage_compass.parquet` | Voyage-level features, indices, diagnostics |
| `panel_captain_year_compass.parquet` | Captain×year aggregated (search-step weighted) |
| `robustness_report.json` | Full robustness diagnostics |
| `pca_loadings.json` | PCA loadings for reproducibility |

> See [docs/README_compass.md](docs/README_compass.md) for detailed methods documentation.

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
│   ├── pipeline/              # NEW: 5-stage pipeline orchestration
│   │   ├── __init__.py        # Unified exports
│   │   ├── stage1_pull.py     # Data acquisition
│   │   ├── stage2_clean.py    # Parsing and standardization
│   │   ├── stage3_merge.py    # Assembly and linkage
│   │   ├── stage4_analyze.py  # Full analysis suite
│   │   ├── stage5_output.py   # MD and TEX generation
│   │   └── runner.py          # Full pipeline orchestration
│   ├── download/              # Data acquisition (7 modules)
│   │   ├── aowv_downloader.py
│   │   ├── archive_downloader.py
│   │   ├── online_sources_downloader.py
│   │   ├── wsl_pdf_downloader.py
│   │   ├── weather_downloader.py
│   │   ├── economic_downloader.py
│   │   └── manifest.py
│   ├── parsing/               # Parsing & normalization (11 modules)
│   │   ├── string_normalizer.py   # Jaro-Winkler, Soundex
│   │   ├── voyage_parser.py
│   │   ├── crew_parser.py
│   │   ├── logbook_parser.py
│   │   ├── register_parser.py
│   │   ├── starbuck_parser.py
│   │   ├── maury_parser.py
│   │   ├── wsl_pdf_parser.py
│   │   ├── wsl_event_extractor.py
│   │   └── wsl_market_parser.py
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
│   ├── analyses/              # Empirical analysis suite (40+ modules)
│   │   ├── config.py
│   │   ├── data_loader.py
│   │   ├── connected_set.py
│   │   ├── baseline_production.py
│   │   ├── run_full_baseline_loo_eb.py  # Main LOO+EB suite
│   │   ├── paper_tables.py              # Table generation
│   │   ├── counterfactual_suite.py      # Counterfactuals
│   │   └── ...                          # Additional analyses
│   └── compass/               # Compass: micro-routing measurement
│       ├── config.py          # CompassConfig dataclass
│       ├── data_io.py         # Load & validate trajectories
│       ├── preprocess.py      # UTM projection + smoothing
│       ├── steps.py           # Step construction + resampling
│       ├── regimes.py         # HMM regime segmentation
│       ├── features.py        # Compass feature suite
│       ├── compass_index.py   # PCA index + early window
│       ├── robustness.py      # 5 robustness batteries
│       ├── export.py          # Econometric panel exports
│       └── cli.py             # CLI orchestrator
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

## Counterfactual Simulations

The `counterfactual_suite.py` module implements five counterfactual exercises that quantify the economic value of organizational structures in 19th-century whaling.

### Core Production Function

All counterfactuals build on the estimated production function with captain-agent interaction:

```
log(Q) = β_θ·θ + β_ψ·ψ + β₃·(θ × ψ) + controls + ε
```

where:

- **θ** = captain skill (α̂ from R1)
- **ψ** = agent capability (γ̂ from R1)
- **β₃** = interaction term (substitutes if < 0, complements if > 0)

The key estimate: **β₃ varies by ground type**:

| Ground | β₃ | Interpretation |
|--------|-----|----------------|
| Sparse | **-0.052** | Strong substitutes (agents substitute for captain skill) |
| Rich | +0.011 | Weak complements |

---

### CF_A2: Map Diffusion to Sparse Grounds

**Question**: What output gains from transferring routing technology ("maps") to agents on sparse grounds?

**Method**: Uses the movers design to estimate how agent capability (ψ) shifts search geometry (Lévy μ). High-ψ agents operate with μ closer to 2.0 (optimal Lévy). We simulate transferring this μ improvement to low-ψ agents on sparse grounds.

| Metric | Value |
|--------|-------|
| Target group | 1,628 voyages (low-ψ × sparse) |
| Mean Δlog_q | **+0.47%** |

**Interpretation**: Organizational "maps" (routing knowledge embedded in agents) have modest but positive value. The effect is concentrated on sparse grounds where information advantages matter most.

---

### CF_B5: Matching Counterfactuals (PAM vs AAM)

**Question**: Is observed negative sorting efficient given β₃ < 0?

**Method**: Within route×time cells, we recompute output under:

- **PAM** (Positive Assortative Matching): high-θ captains matched with high-ψ agents
- **AAM** (Anti-Assortative Matching): high-θ captains matched with low-ψ agents

| Ground | β₃ | PAM Effect | AAM Effect |
|--------|-----|------------|------------|
| **Sparse** | -0.052 | **-5.50%** | **+2.69%** |
| **Rich** | +0.011 | +1.04% | -0.73% |
| **Overall** | -- | -1.02% | +0.32% |

**Interpretation**:

- **Sparse grounds**: Substitutes → AAM optimal. Switching to PAM would cost 5.5% output.
- **Rich grounds**: Weak complements → PAM marginally better.
- The observed matching is close to efficient AAM, confirming industry learned to sort negatively where substitution is strongest.

---

### CF_F15: Inequality Decomposition

**Question**: Which factor explains output dispersion—captain skill (θ), agent capability (ψ), or map technology (μ)?

**Method**: Sequentially equalize each factor and compute the reduction in output variance.

| Policy | Variance Reduction |
|--------|-------------------|
| Equalize ψ (agent) | **1.073** |
| Equalize θ (captain) | 1.066 |
| Equalize map tech (μ) | 0.006 |

**Interpretation**: Agent capability and captain skill explain nearly equal shares of output inequality. Map technology (search geometry) explains very little—suggesting routing knowledge is already widely diffused or less important than bilateral human capital.

---

### CF_C8 & CF_A3 (Require Weather Data)

These counterfactuals require hurricane/ice exposure data:

- **CF_C8**: Decomposes exploration into trait (captain-specific) vs forced (weather-induced) components
- **CF_A3**: Tests whether map adoption matters more in high-risk climate states

### Running Counterfactuals

```bash
# Run full counterfactual suite
python -c "
from src.analyses.counterfactual_suite import run_full_counterfactual_suite
run_full_counterfactual_suite(save_outputs=True)
"

# Outputs saved to: output/counterfactual/tables/
```

### Output Files

| File | Contents |
|------|----------|
| `cf_a2_map_diffusion.csv` | Map diffusion results by ground type |
| `cf_b5_matching_route_time.csv` | Matching counterfactuals (route×time cells) |
| `cf_b5_matching_decade_ground.csv` | Matching counterfactuals (decade×ground cells) |
| `cf_f15_inequality.csv` | Inequality decomposition |
| `counterfactual_results.md` | Full narrative summary |

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
