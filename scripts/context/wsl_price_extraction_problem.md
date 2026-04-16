# WSL Market Price Extraction — Problem Specification

## Problem Statement

Build a VLM-based extraction pipeline to recover weekly commodity price time
series from the Whalemen's Shipping List (WSL), a newspaper published in New
Bedford, Massachusetts from 1843–1914. The target output is a structured panel of
**sperm oil, whale oil, and whalebone prices** at issue-level (approximately
weekly) granularity across the full 72-year span.

---

## 1. Why This Matters (Economic Motivation)

This project is part of an empirical economics paper studying 19th-century
whaling productivity using an AKM (Abowd-Kramarz-Margolis) variance
decomposition of captain and agent fixed effects.

The existing pipeline uses **static price constants** to convert physical
production (barrels of oil, lbs of bone) into revenue:

```python
SPERM_OIL_PRICE = 1.30  # $/gallon — fixed across all years
WHALE_OIL_PRICE = 0.60  # $/gallon — fixed across all years
WHALEBONE_PRICE = 0.90  # $/lb    — fixed across all years
```

This is a significant limitation. A **time-varying price index** would:
1. Enable proper TFP (total factor productivity) analysis with time-varying
   output prices
2. Allow revenue-based decomposition (E5 specification) with actual market
   conditions
3. Capture the petroleum competition shock (post-1859) through whale oil
   price decline
4. Measure the whalebone price boom (1870s–1900s) that sustained the late
   industry

The prices are published in every WSL issue in a section called **"Marine
Markets"**, **"Prices Current"**, **"Oil Market"**, or **"New Bedford Market"**.

---

## 2. Source Data

### 2.1 PDF Corpus

- **Location on HPCC**: `/mnt/home/maricste/PDFs/{year}/wsl_{year}_{MM}_{DD}.pdf`
- **Years**: 1843–1914 (72 directories)
- **Issues per year**: ~52 (weekly publication)
- **Pages per issue**: 4–8 pages (each page is one PDF page)
- **Total PDF files**: ~3,700 issues
- **Total pages**: ~14,190

Each PDF file represents one weekly issue. Filenames encode the publication
date: `wsl_1865_10_17.pdf` = October 17, 1865.

### 2.2 Page Content Types (from prior extraction)

The V2 extraction pipeline already classified every page:

| Page Type | Count | % | Description |
|-----------|-------|---|-------------|
| shipping_table | 5,021 | 35.4% | Dense tabular registry of vessels |
| sparse | 4,444 | 31.3% | Voyage event mentions (departures, arrivals) |
| mixed | 2,349 | 16.6% | Mix of events and tables |
| skip | 2,376 | 16.7% | Advertisements, editorials, market reports |

**The price data lives primarily on "skip" and "mixed" pages**, which the
voyage-event pipeline deliberately bypassed. These pages contain editorial
content, advertisements, and — critically — the market price reports.

### 2.3 What the Market Section Looks Like

Typical WSL market report (reconstructed from OCR samples):

```
MARINE MARKET.
New Bedford, October 17, 1865.

Sperm Oil—The market has been fairly active during the past week.
Sales include 800 bbls at $2.25 to $2.30 per gallon.

Whale Oil—This article continues in moderate demand.
Northern sells at 95c to $1.02 per gallon.

Whalebone—The market is firm with sales of Arctic at $1.75 to $1.85 per lb.
Northwest Coast bone quoted at $1.50 per lb.
```

Key characteristics:
- Prices are given as **ranges** (e.g., "$2.25 to $2.30") or single values
- Units vary: **$/gallon** for oil, **$/lb** for whalebone
- Commodity subtypes exist: "Northern" whale oil, "Arctic" bone, "NW Coast" bone
- Quality grades sometimes mentioned: "dark", "bleached", etc.
- Market narrative provides context (demand level, recent transactions)
- Format evolves across the 72-year span

---

## 3. Existing Infrastructure

### 3.1 HPCC Compute

- **Cluster**: MSU ICER HPCC
- **GPU nodes**: NVIDIA L40S (46 GB VRAM), accessed via `--constraint=l40s`
- **Model**: `Qwen/Qwen3-VL-32B-Instruct-FP8` cached at
  `/mnt/research/CEO_Complementarities/maricste/hf_cache/models/Qwen3-VL-32B-Instruct-FP8`
- **Serving**: vLLM `0.18.1` with OpenAI-compatible API
- **SLURM partition**: `general-short` (2-hour wall time)

### 3.2 Existing Extraction Pipeline

The V2 extraction script is at `scripts/hpcc_extract_v2.sb`. It:

1. Starts a vLLM server on the assigned GPU
2. Iterates over PDF pages, converting each to a base64 image
3. Sends image + prompt to the vLLM server via OpenAI chat completions API
4. Parses JSON response and writes per-page JSONL records
5. Supports checkpoint resume (skips already-processed pages)

Key configuration parameters already tuned:
- `max_model_len=8192`
- `MAX_TOKENS=4096`
- `WORKER_MULTIPLIER=3` (3 concurrent inference threads per GPU)
- Dynamic `BASE_PORT` derived from `SLURM_JOB_ID`
- Health check timeout: 420 seconds

### 3.3 Existing Price Parser (Regex-Based)

`src/parsing/wsl_market_parser.py` contains a complete regex-based price
extractor that was designed for OCR text input. It includes:

- `MARKET_SECTION_PATTERNS`: regexes to find market sections
- `COMMODITY_PATTERNS`: patterns for sperm oil, whale oil, whalebone
- `PRICE_PATTERNS`: patterns for dollar ranges, cents notation, fractions
- `UNIT_PATTERNS`: per_gallon, per_lb, per_bbl detection
- `WSLPriceQuote` dataclass with: commodity, price_low, price_high,
  price_unit, confidence
- `build_annual_price_index()`: aggregation to yearly averages

This parser could be used as a **post-processing layer** on VLM output, or
as a reference for the VLM prompt design.

### 3.4 Output Directory

All extraction outputs go to:
`/mnt/research/CEO_Complementarities/maricste/data/extracted/`

The voyage extraction produced:
- `wsl_events_{year}.jsonl` — per-year page-level records
- `wsl_events_all.jsonl` — merged 224 MB dataset (272,872 events)

---

## 4. Target Output Schema

Each PDF page should produce a JSONL record. Pages without market data produce
empty price arrays:

```json
{
  "source_pdf": "wsl_1865_10_17.pdf",
  "page_number": 3,
  "issue_date": "1865-10-17",
  "has_market_section": true,
  "market_section_type": "marine_market",
  "prices": [
    {
      "commodity": "sperm_oil",
      "price_low": 2.25,
      "price_high": 2.30,
      "unit": "per_gallon",
      "grade": null,
      "raw_text": "Sales include 800 bbls at $2.25 to $2.30 per gallon",
      "confidence": 0.95
    },
    {
      "commodity": "whale_oil",
      "price_low": 0.95,
      "price_high": 1.02,
      "unit": "per_gallon",
      "grade": "northern",
      "raw_text": "Northern sells at 95c to $1.02 per gallon",
      "confidence": 0.90
    },
    {
      "commodity": "whalebone",
      "price_low": 1.75,
      "price_high": 1.85,
      "unit": "per_lb",
      "grade": "arctic",
      "raw_text": "Arctic at $1.75 to $1.85 per lb",
      "confidence": 0.90
    }
  ],
  "market_narrative": "The market has been fairly active during the past week.",
  "n_prices": 3
}
```

### 4.1 Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `commodity` | enum | `sperm_oil`, `whale_oil`, `whalebone` |
| `price_low` | float | Lower bound of quoted range (or single price) |
| `price_high` | float | Upper bound (= price_low if single quote) |
| `unit` | enum | `per_gallon`, `per_lb`, `per_bbl` |
| `grade` | string? | Quality/origin: `arctic`, `northern`, `nw_coast`, `dark`, `bleached`, null |
| `raw_text` | string | Verbatim text snippet the price was extracted from |
| `confidence` | float | 0–1 extraction confidence |

---

## 5. Design Constraints & Considerations

### 5.1 Compute Budget

Unlike the voyage extraction (272K events across 14K pages, requiring 3–4
passes), price extraction is **much lighter**:
- Only ~7,400 pages potentially contain prices (skip + mixed types)
- Most pages will return zero prices → fast inference (2–3 seconds)
- Pages with prices have short text → small JSON output
- Estimated total GPU time: 5–10 hours (vs. 50+ hours for events)

### 5.2 Key Design Decisions

1. **Scan all pages vs. only skip/mixed pages?**
   - The voyage pipeline already classified pages. You could use those
     classifications to filter, or scan everything to avoid missing prices
     on pages that were classified differently.
   - Recommendation: Scan **all** pages. The marginal cost is low and
     avoids false negatives from misclassification.

2. **Single prompt vs. two-stage (detect → extract)?**
   - Single: Ask VLM to extract prices from every page, accept empty arrays.
   - Two-stage: First classify whether page has market data, then extract.
   - The single-stage approach is simpler and probably fast enough for
     pages that obviously lack prices.

3. **Prompt for prices vs. full market section transcription?**
   - Just prices: Structured JSON extraction (like the voyage prompt).
   - Full transcription: Get the raw market text, then parse with regex.
   - Structured extraction is more reliable; the regex parser can be used
     as validation.

4. **One job per year vs. one job for all?**
   - Per-year: Parallelizable, matches the event extraction pattern.
   - All at once: Simpler orchestration, but can't parallelize.
   - Per-year is better for the 2-hour SLURM time limit.

### 5.3 Edge Cases

- **Pre-1850s format**: The earliest WSL issues (1843–1849) may have
  different market section formats or irregular price reporting.
- **Civil War era** (1861–1865): Market disruptions, irregular publication.
- **Price units change over time**: Some eras quote per barrel instead of
  per gallon (1 barrel ≈ 31.5 gallons for whale oil).
- **Multiple grades**: Arctic bone vs. NW Coast bone can have very different
  prices. The `grade` field should capture this.
- **"Nominal" vs. no sales**: Some issues report "Sperm oil — nominal" or
  "no sales reported" — these should be flagged differently from missing data.
- **Candles and other products**: Some issues quote spermaceti candles or
  other whale products — these are secondary but potentially useful.

---

## 6. Downstream Integration

The extracted price panel will be consumed by:

### 6.1 `build_annual_price_index()` (already exists)

In `src/parsing/wsl_market_parser.py` — aggregates individual quotes to
annual averages weighted by confidence. Simple to adapt for the new JSON schema.

### 6.2 `run_e5_revenue_valuation()` (needs price data)

In `src/analyses/economic_regressions.py` — currently uses hardcoded prices.
With a real price panel, the revenue variable becomes:

```
revenue_v = q_sperm_v × sperm_price_t + q_whale_v × whale_price_t + q_bone_v × bone_price_t
```

where `t` is the year of the voyage's departure.

### 6.3 Time series analysis

A monthly or quarterly price series enables:
- Structural break detection around petroleum discovery (1859)
- Granger causality tests between petroleum and whale oil prices
- Price volatility analysis as a measure of market uncertainty

---

## 7. Deliverables

1. **VLM extraction prompt** optimized for price/market data
2. **SLURM batch script** (modeled on `hpcc_extract_v2.sb`) that:
   - Starts vLLM server
   - Iterates over all pages in all PDFs for a given year
   - Extracts price quotes via the VLM
   - Writes per-page JSONL to output directory
   - Supports checkpoint resume
3. **Post-processing script** that:
   - Merges per-year JSONL into unified price panel
   - Validates price ranges (e.g., sperm oil should be $0.50–$3.00/gal)
   - Builds annual and monthly price indices
   - Exports to `data/staging/wsl_price_quotes.parquet` and
     `data/staging/wsl_annual_prices.parquet`
4. **Integration patch** to `economic_regressions.py` to use the real price
   data instead of hardcoded constants

---

## 8. Reference Files

| File | Purpose |
|------|---------|
| `scripts/hpcc_extract_v2.sb` | Working SLURM extraction pipeline (events) — copy and adapt |
| `src/parsing/wsl_market_parser.py` | Regex price parser — reference for commodities/patterns |
| `src/analyses/economic_regressions.py` | Consumer of price data — see E5 specification |
| `scripts/wsl_postprocess.py` | Event post-processor — reference for validation patterns |
| `scripts/wsl_merge.py` | Event merger — reference for merge + postprocess pattern |
| `scripts/context/wsl_extraction_briefing.md` | Full context on the event extraction campaign |

---

## 9. Success Criteria

1. **Coverage**: Price quotes for ≥90% of WSL issues (≥3,300 out of ~3,700)
2. **Accuracy**: Manual spot-check of 50 random quotes shows ≥95% correctness
3. **Completeness**: All three commodities (sperm oil, whale oil, whalebone)
   represented in the annual index for every year 1843–1914
4. **Integration**: `run_e5_revenue_valuation()` runs successfully with the
   extracted price panel and produces plausible variance decomposition results
