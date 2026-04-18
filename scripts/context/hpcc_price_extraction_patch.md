# HPCC V4 — Price Extraction Patch Spec

Extend `hpcc_extract_v4.sb` so the VLM produces commodity-price records
alongside vessel events. Required for the WSL price time-series that feeds
nominal-dollar productivity valuation.

## Current gap

- `classify_page()` (v4.sb:915–967) returns page_type ∈ `{skip,
  shipping_table, registry, narrative, mixed, sparse}`. **No
  `market_prices`** label, even though `src/parsing/wsl_page_routing.py`
  defines one for the text-layer path.
- All three prompts (`PROMPT_TABLE`, `PROMPT_NARRATIVE`, `PROMPT_REGISTRY`)
  ask only for `{"events": [...]}`. None ask for commodity prices.
- "Marine Markets" / "Prices Current" sections on `narrative` and `mixed`
  pages are silently dropped.

## Patch summary

Four in-place edits plus one new prompt. Keep wire format
consistent with the local post-processor (`src.parsing.wsl_price_postprocess`):
compact keys `c`, `lo`, `hi`, `u`, `cur`.

### 1. Add `PROMPT_PRICES` (insert after `PROMPT_REGISTRY`, ~v4.sb:487)

```python
PROMPT_PRICES = """Extract the current commodity prices from this Whalemen's Shipping List page.
These pages contain "Marine Markets", "Prices Current", or "Oil Market" sections
quoting sperm oil, whale oil, and whalebone prices in USD.

For each commodity-price line, extract:
- "c": commodity — one of: sperm_oil, whale_oil, whalebone
- "lo": lower bound of quoted price range (float, USD)
- "hi": upper bound of quoted price range (float, USD). If single value, lo=hi.
- "u": unit — one of: gal (per gallon), lb (per pound), bbl (per barrel)
- "cur": currency, default "USD"; set "GBP" only if £/stg is explicit
- "raw": the raw text snippet as it appears on the page

Rules:
1. Only extract commodity prices. Ignore freight, insurance, exchange rates.
2. Prices are typically in dollars and cents (e.g., $1.15, 1.20 to 1.25, 115c).
3. Convert cents-per-gallon to dollars (115c → 1.15).
4. If the page has no price quotes, return: {"prices": []}

This page may ALSO contain vessel events. Extract those into "events" per the
normal rules; prices go into "prices".

Return: {"events": [...], "prices": [...]}
Return ONLY valid JSON. No explanation. No markdown."""
```

### 2. Classifier: detect `market_prices` pages (edit `classify_page()`, v4.sb:915)

Insert **before** the existing `if ad_kw >= 2 ...` block at v4.sb:941:

```python
    price_kw = sum(1 for w in ["marine market", "prices current",
                                 "oil market", "sperm oil",
                                 "whale oil", "whalebone", "whale bone",
                                 "per gallon", "per pound"]
                    if w in text)
    # Pure price page: prices without vessel activity
    if price_kw >= 3 and shipping_kw == 0:
        return "market_prices", "price_keywords_pure"
```

And update the `shipping_kw >= 3` branch (v4.sb:964) to route price-heavy
`mixed` pages to a price-aware prompt:

```python
    elif shipping_kw >= 3:
        if price_kw >= 2:
            return "mixed_prices", "text_events_plus_prices"
        return "mixed", "text_keywords_mixed"
```

### 3. Router: wire new page types (edit `get_prompt_for_page_type()`, v4.sb:971)

```python
def get_prompt_for_page_type(page_type):
    if page_type == "shipping_table":
        return PROMPT_TABLE
    if page_type == "registry":
        return PROMPT_REGISTRY
    if page_type in ("market_prices", "mixed_prices"):
        return PROMPT_PRICES
    return PROMPT_NARRATIVE
```

### 4. Token budget (edit `TOKEN_BUDGET` dict, search `TOKEN_BUDGET = {`)

Add:
```python
    "market_prices": 1024,   # price tables are short
    "mixed_prices":  3072,   # room for both events and prices
```

### 5. Output assembly (edit the per-page record construction, v4.sb:1688, 1763)

After the existing `events = [post_process(...) for ev in raw_events]`,
unpack and emit prices too:

```python
    raw_prices = parsed.get("prices") or []
    prices = [p for p in raw_prices if _valid_price(p)]
```

Add a small guardrail helper near the post-processing block:

```python
def _valid_price(p):
    if not isinstance(p, dict): return False
    if p.get("c") not in {"sperm_oil", "whale_oil", "whalebone"}: return False
    try:
        float(p.get("lo", p.get("hi", 0)))
    except (TypeError, ValueError):
        return False
    return True
```

Then include `prices` in both the success and suspicious branches of the
page record (v4.sb:1688 and v4.sb:1763):

```python
    rec = {
        ...
        "events": events,
        "prices": prices,            # ← new
        "n_events": len(events),
        "n_prices": len(prices),     # ← new
        ...
    }
```

## Wire-format contract

The local post-processor (`src.parsing.wsl_price_postprocess.flatten_price_record`)
already accepts both compact (`c`, `lo`, `hi`, `u`, `cur`) and verbose keys,
plus a `_confidence` passthrough. No further local-side changes required
once this patch ships.

## Rollout

1. Apply the patch to `hpcc_extract_v4.sb` on a feature branch.
2. `sbatch scripts/hpcc_extract_v4.sb --test 1850` — one-year smoke test.
3. Inspect `/mnt/research/.../data/extracted/wsl_events_1850.jsonl`; confirm
   a `"prices": [...]` key appears on `market_prices` / `mixed_prices`
   pages, empty array elsewhere.
4. Locally: `python -m src.parsing.wsl_price_postprocess --year 1850` then
   `pytest tests/test_price_validators.py` against the resulting parquet.
5. If validation passes, submit backfill (`scripts/hpcc_price_backfill.sb`)
   for the remaining 71 years.

## Cost estimate

Full re-extraction is ~4 h × 72 jobs = 288 GPU-hours on 4× L40S. The
**backfill script** (next task) touches only ~2–4 price pages per issue
(the masthead and market pages), reducing that to ~20 GPU-hours for the
whole corpus.
