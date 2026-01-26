# IPUMS USA Extract Instructions

This document provides step-by-step instructions for creating the IPUMS census extracts needed for captain-to-census linkage.

## Overview

You will need to create a custom data extract from IPUMS USA containing:

1. Full-count census data for 1850, 1860, 1870, and 1880
2. (Optional) MLP linked identifiers for cross-census tracking

## Step 1: Create IPUMS Account

1. Go to [usa.ipums.org](https://usa.ipums.org)
2. Click "Register" and create a free account
3. Verify your email address

## Step 2: Start a New Extract

1. Log in to usa.ipums.org
2. Click "Get Data" → "SELECT DATA"
3. You'll be taken to the variable selection interface

## Step 3: Select Sample(s)

Click "SELECT SAMPLES" and choose:

- [x] 1850 100% (requires acceptance of use conditions)
- [x] 1860 100%
- [x] 1870 100%
- [x] 1880 100%

> **Note**: Full-count data contain every person enumerated in the census.
> For faster extracts during development, you can use 1% samples first.

## Step 4: Select Variables

### Required Variables

| Variable | Description | Why Needed |
|----------|-------------|------------|
| `YEAR` | Census year | Panel identifier |
| `SERIAL` | Household serial number | Household identifier |
| `PERNUM` | Person number in household | Person identifier |
| `HISTID` | Historical unique identifier | Linkage |
| `NAMEFRST` | First name | Name matching |
| `NAMELAST` | Last name | Name matching |
| `AGE` | Age | Age matching |
| `SEX` | Sex | Filtering |
| `BPL` | Birthplace | Geography matching |
| `STATEFIP` | State (FIPS code) | Geography filtering |
| `OCC` | Occupation | Occupation matching |

### Wealth Variables (1850-1870 only)

| Variable | Description |
|----------|-------------|
| `REALPROP` | Value of real estate owned |
| `PERSPROP` | Value of personal property owned |

> **Note**: REALPROP is only available 1850-1870. PERSPROP is only available 1860-1870.

### Family Link Variables (for spouse validation)

| Variable | Description |
|----------|-------------|
| `SPLOC` | Spouse's location in household |
| `MOMLOC` | Mother's location in household |
| `POPLOC` | Father's location in household |
| `RELATE` | Relationship to household head |

### MLP Variables (optional, for cross-census linking)

| Variable | Description |
|----------|-------------|
| `HIK` | IPUMS Historical Individual Key |

## Step 5: Apply Case Selection (Optional)

To reduce extract size, you can filter to whaling states:

1. Click on `STATEFIP`
2. Click "Select cases"
3. Choose:
   - Massachusetts (025)
   - Connecticut (009)
   - New York (036)
   - Rhode Island (044)

## Step 6: Create Extract

1. Click "VIEW CART" to review your selections
2. Click "CREATE DATA EXTRACT"
3. Choose "Stata (.dta)" or "CSV" format
4. Add a description: "Whaling captain linkage - Full count 1850-1880"
5. Click "SUBMIT EXTRACT"

## Step 7: Download and Place Files

1. Wait for email notification (large extracts may take hours)
2. Download the data file and codebook
3. Place files in: `/Users/smaric/Whaling/data/raw/ipums/`

Expected files:

```
data/raw/ipums/
├── usa_00001.csv         # or .dat/.dta
├── usa_00001.xml         # codebook
└── usa_00001.cbk         # plain text codebook (optional)
```

## Step 8: Run IPUMS Loader

```bash
cd /Users/smaric/Whaling
python -c "from src.linkage.ipums_loader import IPUMSLoader; loader = IPUMSLoader(); loader.parse(); loader.save()"
```

## Approximate File Sizes

| Years | States | Approximate Size |
|-------|--------|------------------|
| 1850-1880 | All US | 20-50 GB |
| 1850-1880 | Whaling states only | 2-5 GB |
| 1850-1880 | 1% samples | 200-500 MB |

## Troubleshooting

### Extract Takes Too Long

- Use case selection to filter to whaling states
- Start with 1850 and 1860 only
- Use 1% samples for initial testing

### Memory Issues Loading Data

- Use chunked loading (not yet implemented)
- Filter to whaling states during load
- Consider converting to Parquet format

### Missing Variables

- Some variables are only available for certain years
- Check variable availability by sample on IPUMS website

## Citation

When using IPUMS data, please cite:

> Steven Ruggles, Sarah Flood, Matthew Sobek, Danika Brockman, Grace Cooper,
> Stephanie Richards, and Megan Schouweiler. IPUMS USA: Version 14.0 [dataset].
> Minneapolis, MN: IPUMS, 2023. <https://doi.org/10.18128/D010.V14.0>

For MLP data:

> Steven Ruggles, et al. IPUMS Multigenerational Longitudinal Panel: Version 2.0
> [dataset]. Minneapolis, MN: IPUMS, 2025. <https://doi.org/10.18128/D016.V2.0>
