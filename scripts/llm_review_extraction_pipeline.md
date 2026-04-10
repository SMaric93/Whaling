# WSL Extraction Pipeline — Review & Improvement Brief

> **Purpose:** This document is a comprehensive diagnostic brief for reviewing and improving a VLM-based (Vision-Language Model) extraction pipeline that reads historical newspaper PDFs and produces structured JSON records. The pipeline is critically underperforming — extracting <1% of expected events. Your task is to diagnose the root causes and propose concrete code-level fixes.

---

## 1. Project Context

We are extracting vessel events from the **Whalemen's Shipping List (WSL)**, a weekly newspaper published in New Bedford, Massachusetts (1843–1914). Each 4-page issue contains tabular vessel departure/arrival records, at-sea sighting reports, commodity prices, and marine notices.

**Corpus:** 3,703 PDFs across 72 years, ~4 pages each (~15,000 total pages).

**Goal:** Extract every vessel event into structured JSONL for econometric analysis of 19th-century whaling productivity.

**Expected yield:** ~30–80 vessel entries per shipping-list page (pages 2–4 of each issue are dense tabular listings). A full-year test (53 issues, 219 pages) should produce **3,000–10,000+ events**.

**Actual yield from test run:** **26 events total** from 219 pages. The pipeline is capturing <1% of the data.

---

## 2. Architecture Overview

We have **two pipeline variants**, and the SLURM script is the one that ran and failed:

### A. Local Pipeline (macOS, not yet run on full corpus)
- **OCR:** Apple Vision Framework (hardware-accelerated, runs on Mac)
- **LLM:** Qwen3-VL-4B-Instruct-4bit via MLX (local inference on Apple Silicon)
- **Flow:** PDF → render page image → Apple Vision OCR → text → chunk OCR lines → LLM JSON extraction → JSONL
- **Key feature:** Page routing classifier skips advertisement pages, separates price extraction from event extraction

### B. HPC Pipeline (SLURM, the one that failed)
- **OCR:** None — the VLM reads page images directly
- **LLM:** Qwen3-VL-32B-Instruct-AWQ (4-bit quantized) via vLLM on NVIDIA L40S GPUs
- **Flow:** PDF → render page image → preprocess (grayscale, contrast, sharpen) → base64 JPEG → vLLM VLM (direct image input) → structured JSON → JSONL
- **Parallelism:** N vLLM servers (1 per GPU), round-robin page distribution via ThreadPoolExecutor

---

## 3. Test Run Results (HPC Pipeline, Year 1850)

### Performance

| Metric | Value |
|---|---|
| PDFs | 53 |
| Pages | 219 |
| GPUs | 7 of 8 (GPU 5 unreliable, 2 timeouts) |
| Runtime | 1h 23m 34s |
| Avg time/page | 124.3s |
| **Total events** | **26** |
| Prices | 25 |
| Notices | 3 |
| Zero-event pages | **195 / 219 (89%)** |

### Per-GPU Event Distribution

```
GPU 0:  32 pages,   7 events, 98.4s avg
GPU 1:  32 pages,   1 events, 124.6s avg
GPU 2:  31 pages,   4 events, 96.9s avg
GPU 3:  31 pages,   0 events, 102.4s avg
GPU 4:  31 pages,   4 events, 103.8s avg
GPU 5:  31 pages,   4 events, 208.2s avg  ← 2× slower, 2 timeouts
GPU 6:  31 pages,   6 events, 136.4s avg
```

All GPUs show uniformly poor yield, ruling out hardware-specific issues.

### Output Characteristics

**Zero-event pages output char distribution** (195 pages):
```
0 chars     :   2  (vLLM returned nothing)
1-100       :   1
101-500     :  17  (likely {"events":[],...})
501-1000    : 105  (model returned text, but no parseable events)
1001-2000   :  50  (significant output, completely failed to parse)
2000+       :  20  (multi-KB output, still no events)
```

**Confidence distribution** (all 219 pages):
```
Mean confidence:  median=0.803  mean=0.790
P10 confidence:   median=0.055  mean=0.071  ← worst 10% of tokens are near-random
```

### Examples of Extracted Events (All 26)

The few events that were parsed show quality issues:

```python
# Garbled prompt bleed — model collapsed structured output into slashes
{'vessel_name': '/bark/Adeline Gibbs/brk/Adeline Gibbs/t:bark/c:Whiteside/e:arr/p:Nantucket/d:Sept 22/hp:Nantucket/dest:Pacific Ocean,lat:null,lon:null/sp,null,/wh,null,/bn,null/,days:,agent:/Swift & Allen/by:/Ship Java/'}

# Impossible coordinates
{'vessel_name': 'Almira', 'longitude': -750}

# Nonsense captain field
{'captain': 'M.'}
{'captain': '/'}

# Invalid vessel type
{'vessel_type': 'dx'}

# Empty required fields
{'vessel_name': '', 'event_type': ''}
```

### Price Extraction Failures

```python
# Negative prices (hallucinated)
{'commodity': 'sperm_oil', 'price_lo': -69, 'price_hi': -69}
{'commodity': 'sperm_oil', 'price_lo': -35, 'price_hi': -34}

# Inverted range
{'commodity': 'sperm_oil', 'price_lo': 60, 'price_hi': 58}

# Placeholder values
{'commodity': 'sperm_oil', 'price_lo': -1, 'price_hi': -1}
```

---

## 4. Identified Issues

### 4.1 `frequency_penalty=1.5` (Almost Certainly the Primary Cause)

The vLLM call uses `frequency_penalty=1.5`, which penalizes the model for repeating any token. Structured JSON output **requires** massive token repetition — every event needs `"v":`, `"t":`, `"c":`, `"e":`, `"p":`, etc.

After the first JSON object, the model is severely penalized for emitting these field names again. Evidence:
- **No page produced more than 2 events** (and most produced 0)
- The garbled `/bark/Adeline Gibbs/brk/...` output shows the model trying to emit JSON field names but replacing `"` and `:` with `/` to avoid repetition penalties

### 4.2 Image Preprocessing May Be Counterproductive

The HPC pipeline converts images to grayscale and applies contrast/sharpness enhancement:
```python
pil = pil.convert('L')  # Grayscale
pil = ImageEnhance.Contrast(pil).enhance(1.5)
pil = ImageEnhance.Sharpness(pil).enhance(2.0)
pil = pil.convert('RGB')  # Back to RGB
```

This may be degrading image quality for an already-printed newspaper. The VLM was trained on natural images and may perform worse on artificially enhanced grayscale.

### 4.3 No Raw Output Logging

The JSONL does not store the raw LLM response text, only the parsed events. When parsing fails on a page (which happens 89% of the time), there is no way to diagnose what the model actually returned.

### 4.4 `/no_think` May Not Work with This Model/vLLM

The prompt starts with `/no_think` to suppress thinking tokens, but Qwen3-VL via vLLM may not support this text prefix. It may require an API parameter instead. If the model enters think mode, it wastes tokens on reasoning before generating JSON.

### 4.5 Prompt Is Very Long and Complex

The extraction prompt is ~2,400 tokens with extensive examples covering 3 section types (events, prices, notices), 6+ event types, and many field rules. For pages where the image is partially degraded, this complexity may overwhelm the model.

### 4.6 No Page-Type Classification or Routing

The HPC pipeline sends every page through the same prompt. Pages 1 (front page / editorial) and advertisement pages should be skipped entirely. The local pipeline has page routing (`wsl_page_routing.py`) but this was not ported to the HPC version.

### 4.7 No Retry Logic for Zero-Event Pages

Pages returning `n_events == 0` with `output_chars > 200` almost certainly had vessel data that failed to parse. There is no retry mechanism.

---

## 5. Code — HPC SLURM Script (the failing pipeline)

This is a 693-line self-contained SLURM bash script with an embedded Python extraction worker.

### Key vLLM API Call Parameters

```python
payload = {
    "model": MODEL_ID,
    "messages": [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            {"type": "text", "text": prompt},
        ]
    }],
    "max_tokens": 8192,
    "temperature": 0.0,
    "frequency_penalty": 1.5,   # ← PROBLEM
    "logprobs": True,
}
```

### Image Preprocessing

```python
def img_to_base64(img_array):
    pil = Image.fromarray(img_array)
    if pil.mode != 'L':
        pil = pil.convert('L')          # ← Force grayscale
    pil = ImageEnhance.Contrast(pil).enhance(1.5)
    pil = ImageEnhance.Sharpness(pil).enhance(2.0)
    pil = pil.convert('RGB')
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
```

### Extraction Prompt (embedded in script)

```
/no_think
You are an expert reader of the Whalemen's Shipping List (WSL), a weekly newspaper
published in New Bedford, Massachusetts (1843-1914). Each page may contain MULTIPLE
section types. Extract ALL information into the structured JSON format below.

## SECTION 1: VESSEL EVENTS
Extract every vessel entry. The WSL has several event sections:
- DEPARTURES ("Sailed"/"Cleared"): vessels leaving port
- ARRIVALS ("Arrived"): vessels returning with cargo
- SPOKEN ("Sp."/"Spoken"): at-sea sighting reports, often with coordinates and reporting ship
- REPORTED ("Rpt"): secondhand reports of vessel status
- IN PORT ("Inp"): vessels currently at a foreign port
- WRECKED/CONDEMNED ("wrk"): vessel losses

Use this schema for each event:
{"v":"vessel name", "t":"ship|bark|brig|sch", "c":"captain surname",
 "e":"dep|arr|spk|rpt|inp|wrk",
 "p":"port or location", "d":"date as printed",
 "hp":"home port", "dest":"whaling ground or destination",
 "lat":null, "lon":null,
 "sp":null, "wh":null, "bn":null,
 "days":"time at sea", "agent":"owner or agent company", "by":"reporting vessel name"}

[...field rules and 3 examples...]

## SECTION 2: PRICES CURRENT
Extract commodity prices if present:
{"commodity":"sperm_oil|whale_oil|whalebone", "price_lo":0.00, "price_hi":0.00, "unit":"per gallon|per lb|per bbl"}

## SECTION 3: MARINE NOTICES
Extract casualty reports, condemnations, crew changes:
{"type":"casualty|condemnation|crew_change|protest|other",
 "vessel":"vessel name", "detail":"brief summary", "date":"date", "location":"where"}

## RULES
- Each vessel appears ONLY ONCE per event. Do NOT repeat entries.
- Stop when you reach the end of visible content on the page.
- "c" must be a person's surname, NOT a company name.
- Parse coordinates as signed numbers.

Return: {"events":[...], "prices":[...], "notices":[...]}
Return ONLY valid JSON. No explanation.
```

### JSON Parsing Logic

```python
def parse_result(raw):
    text = raw.strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1].lstrip("json").strip()
    events, prices, notices = [], [], []
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj.get("events", []), obj.get("prices", []), obj.get("notices", [])
        if isinstance(obj, list):
            return obj, [], []
    except json.JSONDecodeError:
        pass
    # Fallback: extract individual JSON objects via regex
    for m in re.finditer(r'\{[^{}]{10,800}\}', text):
        try:
            o = json.loads(m.group())
            if "v" in o: events.append(o)
            elif "commodity" in o or ("price_lo" in o): prices.append(o)
            elif "type" in o and "detail" in o: notices.append(o)
        except: pass
    return events, prices, notices
```

### Event Normalization & Deduplication

```python
def normalize(e):
    lat = e.get("lat")
    lon = e.get("lon")
    if isinstance(lat, str):
        try: lat = float(lat)
        except: lat = None
    if isinstance(lon, str):
        try: lon = float(lon)
        except: lon = None
    return {
        "vessel_name": (e.get("v") or "").strip(),
        "vessel_type": (e.get("t") or "").strip().lower() or None,
        "captain": (e.get("c") or "").strip() or None,
        "event_type": (e.get("e") or "").strip().lower(),
        "port": (e.get("p") or "").strip() or None,
        "date": (e.get("d") or "").strip() or None,
        "home_port": (e.get("hp") or "").strip() or None,
        "destination": (e.get("dest") or "").strip() or None,
        "latitude": lat,
        "longitude": lon,
        "oil_sperm_bbls": e.get("sp"),
        "oil_whale_bbls": e.get("wh"),
        "bone_lbs": e.get("bn"),
        "days_out": (e.get("days") or "").strip() or None,
        "agent": (e.get("agent") or "").strip() or None,
        "reported_by": (e.get("by") or "").strip() or None,
    }

# Deduplication
seen_keys = set()
deduped = []
for ev in events:
    key = (ev.get('vessel_name','').lower(),
           ev.get('captain','') or '',
           ev.get('event_type',''),
           ev.get('date',''))
    if key not in seen_keys:
        seen_keys.add(key)
        deduped.append(ev)
```

---

## 6. Code — Local Pipeline (macOS, reference implementation)

The local pipeline uses a different architecture that should be considered when improving the HPC version.

### Apple Vision OCR (high-quality text extraction)

```python
def vision_ocr(img, tmp_path="/tmp/_wsl_orch_ocr.png"):
    """Run Apple Vision OCR on a numpy image array. Returns (text, confidence)."""
    tmp_path = f"/tmp/_wsl_orch_ocr_{threading.get_ident()}.png"
    cv2.imwrite(tmp_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(
        NSURL.fileURLWithPath_(tmp_path), None)
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    req.setRecognitionLanguages_(NSArray.arrayWithArray_(["en-US"]))
    req.setUsesLanguageCorrection_(True)

    success = handler.performRequests_error_([req], None)
    if not success[0]:
        return "", 0.0

    boxes = []
    for obs in (req.results() or []):
        candidates = obs.topCandidates_(1)
        if candidates and len(candidates) > 0:
            bbox = obs.boundingBox()
            boxes.append(OCRTextBox(
                points=((x0, y0), (x1, y0), (x1, y1), (x0, y1)),
                text=candidates[0].string(),
                confidence=float(candidates[0].confidence()),
            ))

    lines = ocr_boxes_to_lines(boxes, row_sort_descending=True, min_row_threshold=0.01)
    text = "\n".join(lines)
    conf = float(np.mean(confidences)) if confidences else 0.0
    return text, conf
```

### Page Classification/Routing (skips ads, routes prices vs events)

```python
def classify_wsl_page_text(text: str) -> WSLPageRoute:
    lines = _nonempty_lines(text)
    sample = "\n".join(lines[:80])
    price_hits = _count_hits(PRICE_PATTERNS, sample)
    event_hits = _count_hits(EVENT_PATTERNS, sample)
    ad_hits = _count_hits(AD_PATTERNS, sample)
    vessel_rows = sum(1 for line in lines[:80] if VESSEL_ROW_PATTERN.search(line))

    if price_hits >= 3 and event_hits == 0 and vessel_rows == 0:
        return WSLPageRoute("market_prices", run_events=False, run_prices=True)
    if ad_hits >= 3 and price_hits == 0 and event_hits == 0:
        return WSLPageRoute("advertisements", run_events=False, run_prices=False)
    if price_hits >= 2 and (event_hits >= 1 or vessel_rows >= 1):
        return WSLPageRoute("mixed", run_events=True, run_prices=True)
    if vessel_rows >= 2 or event_hits >= 2:
        return WSLPageRoute("shipping_list", run_events=True, run_prices=False)
    return WSLPageRoute("mixed", run_events=True, run_prices=price_hits >= 1)
```

### Simpler Extraction Prompt (local pipeline)

```
/no_think
Extract vessel events from this Whalemen's Shipping List OCR text.
Return a JSON array using compact keys:
[{"v":"vessel name","t":"ship|bark|brig|sch","c":"captain","e":"dep|arr|spk|rpt|inp","p":"port","d":"date if visible","sp":null,"wh":null,"bn":null}]

Rules:
- v=vessel name, t=vessel type, c=captain surname, e=event type, p=port
- d=date string (e.g. "Sept 22"), sp=sperm oil barrels, wh=whale oil barrels, bn=bone lbs
- Correct obvious OCR errors in names (e.g. "Ncw Bedford" → "New Bedford")
- Return ONLY the JSON array. No explanation, no markdown.

TEXT:
---
{text}
---
```

### Chunked Processing (text-based, not image-based)

```python
# Chunk OCR text into ~100 lines per LLM call
chunks = []
for i in range(0, len(ocr_lines), chunk_size):
    chunks.append("\n".join(ocr_lines[i:i + chunk_size]))

all_events = []
for chunk in chunks:
    prompt_text = PROMPT_TEMPLATE.format(text=chunk)
    result = generate_fn(model, processor, prompt=prompt_text,
                         max_tokens=4096, temperature=0.1)
    raw_events = parse_events_json(result.text)
    all_events.extend([normalize_event(e) for e in raw_events])
```

---

## 7. What the WSL Pages Actually Look Like

A typical WSL issue (4 pages) contains:

- **Page 1:** Front page with editorial text, some vessel news, advertisements
- **Page 2:** Dense tabular shipping list — arrivals, departures (this is the richest page)
- **Page 3:** Continuation of shipping list — spoken/reported vessels, foreign port listings
- **Page 4:** More vessel tables, "List of Vessels" (annual tabular summary), prices, ads

The tabular vessel listings look approximately like this (OCR'd from a real page):

```
ARRIVED.
At New Bedford —
Ship Adeline Gibbs, Whiteside, Pacific Ocean, 1200 sp 450 wh 8000 bn, 28 mos,
    Swift & Allen.
Bark Alto, Lakeman, Indian Ocean, 300 sp, 14 mos, T. Nye Jr & Co.
Brig Harbinger, Cox, Atlantic, 85 sp 40 wh, 6 mos.

SAILED.
From New Bedford —
Ship Columbus, Rogers, Pacific Ocean, C. R. Tucker & Co.
Bark Corinthian, Armington, Indian Ocean.

SPOKEN.
Sp. Ship Omega, Whalon, of Nantucket, Aug 10, lat 5 S, lon 110 W,
    400 sp, 12 mos, by Ship Java.
```

Each page typically has **30–80** such entries densely packed.

---

## 8. Job History on HPCC

30+ job submissions since April 1, all FAILED or CANCELLED:
- **Exit 11:** vLLM OOM crashes (55–67 GB MaxRSS)
- **Exit 1:** Environment setup or vLLM server startup failures
- **TIMEOUT:** Wall time exceeded
- **CANCELLED:** Manual cancellation during debugging

The most recent run (April 9–10) was the **first to complete the full extraction loop**, but with the catastrophically low yield documented above.

---

## 9. Hardware Constraints

### HPC (MSU ICER)

| Resource | Available |
|---|---|
| GPUs | 4–8× L40S (48 GB each) or A100 (80 GB) |
| CPU | 32 cores |
| RAM | 256 GB |
| Scratch | Node-local SSD, temporary |
| Research | Persistent NFS mount |
| Wall time | 4 hours (general-short) or longer partitions |

### Local (for testing)

| Resource | Available |
|---|---|
| Machine | Apple Silicon Mac |
| GPU | Integrated (for MLX inference) |
| RAM | 24+ GB |
| OCR | Apple Vision Framework (hardware-accelerated) |

---

## 10. Questions for You

Please analyze the above and provide:

1. **Root cause diagnosis:** Do you agree that `frequency_penalty=1.5` is the primary cause? Are there other issues I'm missing?

2. **Optimal vLLM parameters:** What should `frequency_penalty`, `temperature`, `max_tokens`, `top_p`, and `repetition_penalty` be set to for structured JSON extraction from a VLM?

3. **Image preprocessing:** Should we keep grayscale conversion + contrast/sharpness enhancement, or send the raw rendered image? What DPI is optimal for Qwen3-VL-32B?

4. **Prompt engineering:** Is the prompt too long/complex? Should we split event extraction and price extraction into separate calls? Should we use few-shot examples or zero-shot? How should we handle the `/no_think` directive for Qwen3-VL via vLLM?

5. **Architecture decision:** The local pipeline (OCR → text → small LLM) vs the HPC pipeline (image → large VLM) represent two approaches. Which is likely to yield better results for dense 19th-century tabular newspaper data? Should the HPC pipeline also use explicitly a separate OCR step followed by text-based extraction?

6. **Retry and validation:** What retry strategy should we use for pages that return 0 events? What validation rules should flag suspicious output (e.g., impossible coordinates, negative prices)?

7. **Scaling:** Once the pipeline is working, what changes are needed to scale from 53 PDFs (test) to 3,703 PDFs (full corpus)?

8. **Anything else** you think is critical that I haven't considered.
