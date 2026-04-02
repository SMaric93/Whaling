# Stopping-Rule Threshold Sensitivity Fix

*Fixes: Editor request for visible threshold sensitivity.*

## What changed
- Produced coefficient path for `ψ × empty_patch` across empty-patch percentile cutoffs (5th–50th).
- Added **Appendix Figure A13** with:
  - Panel A: OLS coefficient path with 95% CI
  - Panel B: Share and count of patches classified as empty
  - Panel C: AFT Weibull coefficient path

## Old code path
- `run_stopping_rule_robustness.py` → single bottom-quartile cutoff

## New code path
- `src/minor_revision/stopping_threshold_curve.py` → full sweep

## Interpretation
- The main-text result uses the 25th percentile (bottom quartile) to define empty patches.
- The coefficient path shows whether the interaction β(ψ × empty) is robust across alternative
  definitions of "empty."

## Key finding
The interaction coefficient is stable across the threshold range examined.
