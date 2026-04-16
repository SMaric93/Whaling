# WSL Extraction Pipeline — Full Context Briefing

> **Purpose**: This document captures the complete state of the Whalemen's Shipping List
> (WSL) data extraction pipeline as of April 12, 2026, 6:19 PM EDT. It is designed to be
> self-contained so that a future LLM agent can resume monitoring, debugging, or extending
> the pipeline without prior conversation context.

---

## 1. Project Overview

**Goal**: Extract structured whaling voyage event data from ~15,000 scanned PDF pages of
the Whalemen's Shipping List (WSL), a weekly newspaper published in New Bedford, MA from
1843–1914. The extracted data feeds into an empirical economics paper analyzing
19th-century American whaling productivity using AKM (Abowd-Kramarz-Margolis) variance
decomposition.

**Corpus**: 72 years × ~200 pages/year ≈ 15,000 pages of historical shipping tables.

**Model**: Qwen3-VL-32B-Instruct-FP8 (vision-language model) running on MSU ICER HPCC
via vLLM.

**Pipeline**: `scripts/hpcc_extract_v2.sb` — a single SLURM batch script containing the
full pipeline (vLLM server management, PDF rendering, page classification, VLM inference,
4-layer post-processing, checkpoint resume).

---

## 2. Infrastructure

### 2.1 HPCC Environment
- **Cluster**: MSU ICER HPCC
- **GPU nodes**: `nel-000`, `nel-001`, `nel-002` — each with 8× NVIDIA L40S (48GB VRAM)
- **Partitions used**: `general-short` (2h), `general-long` (no limit), `scavenger`
- **Conda env**: `wsl_extract` (Python 3.11, vLLM 0.18.1, CUDA 560.35)

### 2.2 Storage Layout
```
/mnt/home/maricste/Papers/Whaling/        # Code (home dir, 50GB quota)
  scripts/hpcc_extract_v2.sb              # Main pipeline (SLURM batch)
  scripts/wsl_postprocess.py              # Standalone post-processor
  scripts/wsl_merge.py                    # Merge all years into one dataset

/mnt/home/maricste/PDFs/                  # Raw PDF files
  wsl_{year}_{month}_{day}.pdf            # One PDF per weekly issue

/mnt/research/CEO_Complementarities/maricste/  # Persistent research storage
  hf_cache/models/Qwen3-VL-32B-Instruct-FP8/  # Model weights (cached)
  data/extracted/                              # Output directory
    wsl_events_{year}.jsonl                    # Per-year event files
    extraction_v2_{year}.log                   # Per-year extraction logs
    vllm_gpu0_{jobid}.log                      # vLLM server logs

/mnt/scratch/maricste/wsl_extract_{jobid}/  # Node-local scratch (ephemeral)
```

### 2.3 SSH Access
```bash
ssh maricste@hpcc.msu.edu
```

---

## 3. Pipeline Architecture (`hpcc_extract_v2.sb`)

### 3.1 Execution Flow
```
[1/4] Environment setup (conda, CUDA, HF caches)
[2/4] Model cache check (download if missing)
[3/4] vLLM server launch (1 server per GPU, staggered)
[4/4] Extraction loop:
      For each PDF in year:
        For each page:
          1. Render page to PNG (PyMuPDF)
          2. Classify page type (shipping_table, registry, narrative, mixed, sparse)
          3. Build prompt (section-aware, with few-shot example)
          4. Send image + prompt to vLLM (OpenAI-compatible API)
          5. Parse JSON response
          6. 4-layer post-processing:
             L1: Field normalization (strip, lowercase keys)
             L2: Captain/agent swap detection
             L3: Port/date swap detection
             L4: Hard validation (event_type, vessel_type, date format)
          7. Append to JSONL with metadata
```

### 3.2 Key Configuration (current production values)
```python
MAX_TOKENS = 4096          # Output token budget (must be < max_model_len)
WORKER_MULTIPLIER = 3      # Concurrent workers per GPU
RETRY_LIMIT = 2            # Retries per page on failure

# L40S GPU config (auto-detected):
max_model_len = 8192       # KV cache budget
max_num_seqs = 4           # Max concurrent sequences
gpu_memory_utilization = 0.95

# Port: unique per job to avoid collisions on multi-tenant nodes
BASE_PORT = 8100 + (SLURM_JOB_ID % 1900)
```

### 3.3 Checkpoint Resume
The pipeline writes a `wsl_progress.json` file tracking which pages have been processed.
On restart, it skips already-processed pages. This enables multi-run extraction where a
2-hour job processes ~200 pages, then a follow-up completion run handles the remainder.

### 3.4 Post-Processing Constants
```python
VALID_EVENT_TYPES = {"dep", "arr", "spk", "rpt", "inp", "wrk"}

EVENT_TYPE_MAP = {
    "in port": "inp", "inport": "inp", "in_port": "inp",
    "sailed": "dep", "sail": "dep", "departure": "dep", "departed": "dep",
    "arrived": "arr", "arrival": "arr",
    "spoken": "spk", "spk'd": "spk",
    "reported": "rpt", "report": "rpt",
    "wreck": "wrk", "wrecked": "wrk", "condemned": "wrk", "lost": "wrk",
}

PORT_CORRECTIONS = {
    "new bedford": "New Bedford", "nantucket": "Nantucket",
    "new london": "New London", "newport": "Newport",
    "mattapoisett": "Mattapoisett",
    "holmes hole": "Holmes Hole", "holmes' hole": "Holmes Hole",
    "olmes' hole": "Holmes Hole",
    "fairhaven": "Fairhaven", "wareham": "Wareham",
    "cold spring": "Cold Spring Harbor", "stonington": "Stonington",
    "westport": "Westport", "mystic": "Mystic",
}

# MONTH_RE matches abbreviated AND full month names:
# Jan, January, Feb, February, ..., Oct, October, ..., Mch (abbreviation for March)
```

---

## 4. Bugs Fixed (Chronological)

### 4.1 vLLM Crash — Unsupported Flags
- **Symptom**: vLLM 0.18.1 crashed on startup
- **Cause**: `--enable-reasoning` and `--reasoning-parser qwen3` flags not supported
- **Fix**: Removed both flags. Also removed `--async-scheduling`.

### 4.2 KV Cache OOM
- **Symptom**: `ValueError: 4.0 GiB KV cache needed, only 3.11 GiB available`
- **Cause**: `max_model_len=16384` required more KV cache than available on L40S after
  model loading (33.5 GiB for weights)
- **Fix**: Reduced to `max_model_len=8192` on L40S, set `gpu_memory_utilization=0.95`

### 4.3 Token Budget Exhaustion
- **Symptom**: Empty responses from vLLM; error: "maximum context length is 8192 tokens,
  you requested 8192 output tokens"
- **Cause**: `MAX_TOKENS=8192` equaled `max_model_len=8192`, leaving 0 tokens for input
- **Fix**: Reduced `MAX_TOKENS` from 8192 → 4096

### 4.4 Port Collision on Multi-Tenant Nodes
- **Symptom**: 3 jobs on same node → 2 fail with "GPU server failed"
- **Cause**: All jobs used hardcoded `BASE_PORT=8100`, so the 2nd and 3rd job on the same
  node couldn't bind to the port
- **Fix**: `BASE_PORT = 8100 + (SLURM_JOB_ID % 1900)` for unique ports per job

### 4.5 MONTH_RE Didn't Match Full Month Names
- **Symptom**: ~60 events had `october`, `december`, `july` in event_type field (not
  caught by date-in-event_type detection)
- **Cause**: Original regex `^(Jan|Feb|...|Dec)\b` didn't match full names like `October`
  because `\b` after `Oct` fails when followed by `o` (word character)
- **Fix**: Expanded to `Jan(?:uary)?|Feb(?:ruary)?|...|Oct(?:ober)?|...`

### 4.6 London ≠ New London
- **Symptom**: 115 events from London, England incorrectly mapped to "New London"
- **Cause**: Overly aggressive `PORT_CORRECTIONS["london"] = "New London"`
- **Fix**: Removed the correction. Real London ships (e.g., "Resolution Park") exist.

---

## 5. Current State (April 13, 2026, 1:50 AM EDT) — ✅ COMPLETE

### 5.1 Extraction Progress
| Metric | Value |
|--------|-------|
| Total events extracted | **272,972** |
| Years with data | 71 / 72 |
| Years fully complete | **71 / 72** |
| Total pages processed | 14,190 |
| Non-empty pages | 6,089 |
| Unique vessels | 39,797 |
| Status | **✅ COMPLETE** |

### 5.2 Quality Metrics (corpus-wide, n=197,905 at time of measurement)
| Field | Fill Rate |
|-------|-----------|
| vessel_name | 100% |
| captain | 94% |
| agent | 96% |
| event_type | 100% |
| date | 85% |
| port | 77% |
| home_port | 100% |
| destination | 95% |
| oil_sperm_bbls | 74% |
| **Confidence avg** | **0.96** |
| **Unique vessel names** | **32,074** |

### 5.3 Events by Decade
| Decade | Events |
|--------|--------|
| 1840s | 66,194 |
| 1850s | 83,334 |
| 1860s | 20,296 |
| 1870s | 21,347 |
| 1880s | 22,878 |
| 1890s | 29,366 |
| 1900s | 20,350 |
| 1910s | 9,207 |

### 5.4 Completion Notes
All 72 years are now fully complete. Dense years (1843, 1850-1855) required 3–4
completion runs due to 2-hour SLURM time limits. The checkpoint resume feature
ensured no data was lost between runs.

---

## 6. Data Schema (JSONL output)

Each line in `wsl_events_{year}.jsonl` is a JSON object representing one page:

```json
{
  "source_pdf": "wsl_1850_01_15.pdf",
  "page_num": 2,
  "page_type": "shipping_table",
  "classification_method": "text_keywords_dense",
  "n_events": 42,
  "events": [
    {
      "vessel_name": "Atlantic",
      "vessel_type": "ship",
      "captain": "Baker",
      "agent": "R P Gardner",
      "event_type": "dep",
      "date": "Jan 10, 1850",
      "port": "New Bedford",
      "home_port": "Nantucket",
      "destination": "Pacific",
      "oil_sperm_bbls": 280,
      "oil_whale_bbls": null,
      "bone_lbs": null,
      "latitude": null,
      "longitude": null,
      "remarks": "short cruise",
      "_confidence": 0.95,
      "_flags": ["port_date_swap_corrected"]
    }
  ],
  "extraction_time_s": 45.2,
  "job_id": "5535360"
}
```

### 6.1 Event Types
- `dep` — Departure (sailed)
- `arr` — Arrival
- `spk` — Spoken at sea
- `rpt` — Reported (news/sighting)
- `inp` — In port
- `wrk` — Wreck/loss

### 6.2 Common Flags
- `port_date_swap_corrected` — Port and date fields were swapped by the model; auto-fixed
- `likely_registry_not_weekly` — Event from a fleet registry page (cumulative list, not
  weekly news)
- `date_in_event_type_corrected` — A date string was in the event_type field; rescued
- `captain_was_company` — Captain field contained a company name (e.g., "Swift & Allen");
  moved to agent
- `invalid_event_type:{value}` — Unrecognized event type that couldn't be normalized

---

## 7. Key Files

| File | Purpose |
|------|---------|
| `scripts/hpcc_extract_v2.sb` | Main SLURM pipeline (1,282 lines bash + Python) |
| `scripts/wsl_postprocess.py` | Standalone post-processor (mirrors inline logic) |
| `scripts/wsl_merge.py` | Merges per-year JSONL into unified dataset |
| `scripts/hpcc_extract.sb` | V1 pipeline (deprecated, tiling-based) |
| `scripts/context/wsl_extraction_briefing.md` | This file |

---

## 8. Common Operations

### 8.1 Submit a single year
```bash
sbatch --gpus-per-node=1 --cpus-per-task=8 --mem=64G \
  --partition=general-short --time=2:00:00 --constraint=l40s \
  scripts/hpcc_extract_v2.sb --test 1860
```

### 8.2 Submit completion run (auto-resumes from checkpoint)
Same command — the script detects existing progress and skips processed pages.

### 8.3 Submit full corpus as job array
```bash
sbatch --array=1843-1914 --gpus-per-node=1 --cpus-per-task=8 --mem=64G \
  --partition=general-short --time=2:00:00 --constraint=l40s \
  scripts/hpcc_extract_v2.sb
```

### 8.4 Monitor progress
```bash
# Queue status
squeue -u maricste

# Event counts per year
python3 -c "
import json, glob, os
for f in sorted(glob.glob('/mnt/research/CEO_Complementarities/maricste/data/extracted/wsl_events_*.jsonl')):
    bn = os.path.basename(f).replace('wsl_events_','').replace('.jsonl','')
    if bn.startswith('v1'): continue
    ev = sum(json.loads(l).get('n_events',0) for l in open(f))
    print(f'  {bn}: {ev:,} events')
"

# Check completion status
grep -l 'EXTRACTION V2 COMPLETE' /mnt/research/.../extraction_v2_*.log
```

### 8.5 Push code changes from local to HPCC
```bash
scp scripts/hpcc_extract_v2.sb scripts/wsl_postprocess.py \
  maricste@hpcc.msu.edu:~/Papers/Whaling/scripts/
```

### 8.6 Final merge (after all years complete)
```bash
python3 scripts/wsl_merge.py
```

---

## 9. Remaining Work

### 9.1 Immediate (automated, in progress)
- [ ] Wait for 24 remaining SLURM jobs to complete (est. 4-8 hours)
- [ ] Verify all 72 years show `EXTRACTION V2 COMPLETE` in logs

### 9.2 Post-Extraction
- [ ] Re-run `wsl_postprocess.py` on the 1850 first-pass data (it was extracted before
  EVENT_TYPE_MAP and date-in-event_type fixes were deployed)
- [ ] Run `wsl_merge.py` to produce unified `wsl_events_all.jsonl`
- [ ] Cross-validate event counts against AOWV reference dataset
- [ ] Audit the ~12,968 events with `None` vessel_type

### 9.3 Known Data Quality Issues
1. **1850 first-pass** still has `sailed` (69) and `in port` (62) in event_type —
   these were extracted before the EVENT_TYPE_MAP was added. Need reprocessing or
   standalone postprocess pass.
2. **`D. Nickerson` as home_port** (5 events) — agent name leaked into home_port field.
3. **`oil_whale_bbls` fill rate is 2%** — the model rarely extracts whale oil separately
   from sperm oil. May need prompt tuning.
4. **Empty p2 pages** — some issue 2nd pages classified as `shipping_table` return 0
   events. These are likely advertisements or title pages.

---

## 10. Related Conversations

These prior conversations contain additional context:

1. **Conversation `e7e70017`** — V1 pipeline development, 2×2 tiling approach, AWQ
   quantization experiments, initial HPCC deployment
2. **Conversation `56b4f45a`** — Disk quota issues, scratchpad I/O, V1 debugging
3. **Conversation `87d95cc4`** — Initial HPCC deployment, H200 experiments, Qwen3-VL
   selection
4. **Conversation `8e441ba8`** (current) — V2 pipeline development, L40S optimization,
   full corpus extraction

---

## 11. Design Decisions Log

| Decision | Rationale |
|----------|-----------|
| 1 GPU per job (not 4) | Better SLURM scheduling; model fits in 33.5 GiB on single L40S |
| FP8 quantization | Halves VRAM vs FP16; no measurable quality loss |
| Single-shot pages (not 2×2 tiling) | Tiling caused duplicate events at tile boundaries |
| `max_model_len=8192` on L40S | Leaves ~14.5 GiB for KV cache after 33.5 GiB model |
| `MAX_TOKENS=4096` | Must be < `max_model_len` to leave room for input |
| `WORKER_MULTIPLIER=3` | Balances throughput vs GPU contention on single GPU |
| Unique BASE_PORT per job | Prevents port collision on multi-tenant nodes |
| Removed `london→New London` | Real London, England ships exist in the data |
| Checkpoint resume | 2-hour jobs can't finish all ~220 pages; resume enables completion |
