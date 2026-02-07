# Compass Pipeline — Micro-Routing / Search Policy Measurement

## Overview

The **Compass Pipeline** converts raw GPS-like vessel trajectories into
regime-labelled step datasets, computes interpretable search-policy features,
constructs PCA-based compass indices, and exports panel-ready datasets for
causal / movers econometric designs.

## Quick Start

```bash
# Full pipeline
python -m compass.cli --config compass_config.json

# Run specific steps only
python -m compass.cli --config compass_config.json --steps 1,2,3,4,5

# Dry run (validate config, log plan, no outputs)
python -m compass.cli --config compass_config.json --dry-run
```

## Pipeline Steps

| Step | Module | Description |
|------|--------|-------------|
| 1 | `data_io` | Load & validate trajectory + metadata |
| 2 | `preprocess` | UTM projection per voyage, optional smoothing |
| 3 | `steps` | Step construction (time-resampled & distance-thinned) |
| 4 | `regimes` | Gaussian HMM regime segmentation (transit / search / return) |
| 5 | `features` | Compass feature suite (tail, persistence, coverage, loitering) |
| 6 | `compass_index` | PCA index + within-strata standardisation |
| 7 | `compass_index` | Early-window compass (first N search steps) |
| 8 | `embedding_optional` | Self-supervised 1D-CNN embedding (requires torch) |
| 9 | `robustness` | Split-half ICC, step-definition robustness, regime placebo, K sensitivity, missingness |
| 10 | `export` | Voyage-level + captain×year panels for econometrics |

## Methods

### Regime Model

A diagonal-covariance Gaussian HMM is fitted on three standardised features
per step: `log(step_length + 1)`, `|turning_angle|`, and `speed`. The
transition matrix is initialised with high self-transition probability
(sticky regimes). BIC selects between K = 3 and K = 4; regimes are labelled
post-hoc by their emission statistics (Transit = fast + straight, Search =
slow + turning, Return = transit-like late in voyage).

### Feature Suite

Features are computed on **search-regime steps only** (or posterior weighted):

- **Tail behaviour**: Hill estimator tail index, step length quantiles
  (p50–p95), share of total distance in top decile.
- **Directional persistence**: Mean resultant length (circular), heading
  auto-correlation at lag 1, heading run-length.
- **Coverage**: Net-to-gross displacement ratio, grid-cell visitation count,
  recurrence rate.
- **Loitering**: Fraction of steps below speed threshold, median speed.

### Index Construction

Features are z-scored within `state_time_cell_id` groups (or globally), then
PCA extracts 1–3 components. `CompassIndex1` typically separates systematic
area-restricted search from ballistic relocation.

### Early-Window Compass

To mitigate reverse causality, features and indices are re-computed using only
the first N search steps after arrival to ground.

## Robustness Checks

1. **Split-half reliability** — ICC of indices from odd/even search steps.
2. **Step-definition robustness** — Spearman ρ across time-resampled vs
   distance-thinned variants.
3. **Regime placebo** — Indices on transit steps (expect null effects).
4. **HMM K sensitivity** — Compare K=3 vs K=4 index stability.
5. **Missingness** — Exclude gap-flagged voyages; compare distributions.

## Outputs

All outputs are saved to `outputs/compass/`:

| File | Description |
|------|-------------|
| `panel_voyage_compass.parquet` | Voyage-level features, indices, and diagnostics |
| `panel_captain_year_compass.parquet` | Captain×year aggregated (search-step weighted) |
| `diagnostics_compass.parquet` | Per-voyage quality metrics |
| `voyage_compass_early_window.parquet` | Early-window features and indices |
| `steps_with_regimes.parquet` | Step-level data with regime labels and posteriors |
| `pca_loadings.json` | PCA loadings for reproducibility |
| `robustness_report.json` | Full robustness diagnostics |
| `config_snapshot.json` | Config used for the run |

## Dependencies

**Required**: `pandas`, `numpy`, `scipy`, `scikit-learn`, `pyproj`, `pyarrow`

**For regime segmentation**: `hmmlearn`

**Optional (embedding)**: `torch`

Install:

```bash
pip install pandas numpy scipy scikit-learn pyproj pyarrow hmmlearn
```
