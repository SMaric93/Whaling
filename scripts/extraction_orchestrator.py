#!/usr/bin/env python3
"""
WSL Extraction Orchestrator
============================
Production pipeline for extracting vessel events from 3,703 Whalemen's Shipping
List PDFs using Apple Vision OCR + Qwen3-VL-4B-Instruct-4bit on MLX.

Architecture:
  PDF → render page image → Apple Vision OCR → chunk text → Qwen3 JSON parse → JSONL

Usage:
  python scripts/extraction_orchestrator.py                    # Full corpus
  python scripts/extraction_orchestrator.py --year 1850        # Single year
  python scripts/extraction_orchestrator.py --resume            # Resume interrupted
  python scripts/extraction_orchestrator.py --limit 10          # First 10 PDFs
  python scripts/extraction_orchestrator.py --workers 2         # Parallel OCR

Output:
  data/extracted/wsl_events.jsonl     — one JSON object per page
  data/extracted/wsl_progress.json    — checkpoint for resume
  data/extracted/wsl_errors.log       — failed pages

Estimated runtime: ~48 hours for full corpus on M5 Pro (24 GB).
"""

import sys
import os
import time
import json
import re
import logging
import argparse
import glob
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import numpy as np
import pypdfium2 as pdfium
import cv2

# Apple Vision Framework
import objc
from Foundation import NSURL, NSArray
import Vision

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parsing.ocr_layout import OCRTextBox, ocr_boxes_to_lines
from src.parsing.wsl_page_routing import classify_wsl_page_text, text_layer_looks_usable

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_ID = "mlx-community/Qwen3-VL-4B-Instruct-4bit"
CHUNK_SIZE = 100       # OCR lines per LLM chunk (sweet spot from benchmarks)
MAX_TOKENS = 4096      # Per-chunk token budget
RENDER_DPI = 150       # PDF rendering resolution
TEMPERATURE = 0.1      # Low temp for deterministic extraction
OUTPUT_DIR = Path("data/extracted")
JSONL_FILE = OUTPUT_DIR / "wsl_events.jsonl"
PROGRESS_FILE = OUTPUT_DIR / "wsl_progress.json"
ERROR_LOG = OUTPUT_DIR / "wsl_errors.log"

PROMPT_TEMPLATE = """/no_think
Extract vessel events from this Whalemen's Shipping List OCR text.
Return a JSON array using compact keys:
[{{"v":"vessel name","t":"ship|bark|brig|sch","c":"captain","e":"dep|arr|spk|rpt|inp","p":"port","d":"date if visible","sp":null,"wh":null,"bn":null}}]

Rules:
- v=vessel name, t=vessel type, c=captain surname, e=event type, p=port
- d=date string (e.g. "Sept 22"), sp=sperm oil barrels, wh=whale oil barrels, bn=bone lbs
- Correct obvious OCR errors in names (e.g. "Ncw Bedford" → "New Bedford")
- Return ONLY the JSON array. No explanation, no markdown.

TEXT:
---
{text}
---"""

PRICE_PROMPT = """/no_think
Extract commodity prices from this Whalemen's Shipping List text.
Return JSON: {{"prices":[{{"commodity":"sperm_oil|whale_oil|whalebone","price_low":0.0,"price_high":0.0,"unit":"per_gallon|per_pound"}}]}}
If no prices found, return {{"prices":[]}}.
Return ONLY JSON.

TEXT:
---
{text}
---"""

# ═══════════════════════════════════════════════════════════════════════════════
# Logging setup
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(ERROR_LOG, mode="a"),
        ],
    )
    return logging.getLogger("wsl_orchestrator")


# ═══════════════════════════════════════════════════════════════════════════════
# Apple Vision OCR
# ═══════════════════════════════════════════════════════════════════════════════

def vision_ocr(img, tmp_path="/tmp/_wsl_orch_ocr.png"):
    """Run Apple Vision OCR on a numpy image array. Returns (text, confidence)."""
    if tmp_path == "/tmp/_wsl_orch_ocr.png":
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
    confidences = []
    for obs in (req.results() or []):
        candidates = obs.topCandidates_(1)
        if candidates and len(candidates) > 0:
            bbox = obs.boundingBox()
            x0 = float(bbox.origin.x)
            y0 = float(bbox.origin.y)
            x1 = x0 + float(bbox.size.width)
            y1 = y0 + float(bbox.size.height)
            boxes.append(
                OCRTextBox(
                    points=((x0, y0), (x1, y0), (x1, y1), (x0, y1)),
                    text=candidates[0].string(),
                    confidence=float(candidates[0].confidence()),
                )
            )
            confidences.append(float(candidates[0].confidence()))

    lines = ocr_boxes_to_lines(
        boxes,
        row_sort_descending=True,
        min_row_threshold=0.01,
    )
    text = "\n".join(lines)
    conf = float(np.mean(confidences)) if confidences else 0.0
    return text, conf


# ═══════════════════════════════════════════════════════════════════════════════
# JSON parsing (robust, handles truncation)
# ═══════════════════════════════════════════════════════════════════════════════

def parse_events_json(raw):
    """Parse model output into list of event dicts. Handles truncated JSON."""
    text = raw.strip()

    # Strip thinking tags
    if "<think>" in text and "</think>" in text:
        text = text.split("</think>")[-1].strip()
    elif text.startswith("<think>"):
        text = text.replace("<think>", "").strip()

    # Strip markdown code fences
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1].lstrip("json").strip()

    # Try full parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        if isinstance(result, dict) and "events" in result:
            return result["events"]
        return [result] if isinstance(result, dict) and "v" in result else []
    except json.JSONDecodeError:
        pass

    # Fallback: extract individual objects from truncated JSON
    events = []
    for m in re.finditer(r'\{[^{}]{10,500}\}', text):
        try:
            obj = json.loads(m.group())
            if "v" in obj:
                events.append(obj)
        except json.JSONDecodeError:
            continue
    return events


def parse_prices_json(raw):
    """Parse price output."""
    text = raw.strip()
    if "<think>" in text and "</think>" in text:
        text = text.split("</think>")[-1].strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1].lstrip("json").strip()
    try:
        result = json.loads(text)
        if isinstance(result, dict) and "prices" in result:
            return result["prices"]
        return []
    except json.JSONDecodeError:
        return []


def normalize_event(e):
    """Normalize compact keys to full names."""
    return {
        "vessel_name": (e.get("v") or "").strip(),
        "vessel_type": (e.get("t") or e.get("vessel_type") or ""),
        "captain": (e.get("c") or "").strip() or None,
        "event_type": (e.get("e") or "").strip(),
        "port": (e.get("p") or "").strip() or None,
        "date": (e.get("d") or "").strip() or None,
        "oil_sperm_bbls": e.get("sp"),
        "oil_whale_bbls": e.get("wh"),
        "bone_lbs": e.get("bn"),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Progress / Resume
# ═══════════════════════════════════════════════════════════════════════════════

def load_progress():
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"completed": {}, "started_at": None, "total_events": 0, "total_pages": 0}


def save_progress(progress):
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def append_result(result):
    with open(JSONL_FILE, "a") as f:
        f.write(json.dumps(result) + "\n")


def _extract_text_layer_page(pdf_path, page_number):
    """Best-effort extraction of one page from the PDF text layer."""
    if pdfplumber is None:
        return None

    try:
        with pdfplumber.open(pdf_path) as pdf:
            return pdf.pages[page_number - 1].extract_text() or ""
    except Exception:
        return None


def prepare_page_text(pdf_path, page_idx, page_meta):
    """Prepare page text using text-layer extraction first, OCR otherwise."""
    t0 = time.time()
    page_number = page_idx + 1
    text_layer = _extract_text_layer_page(pdf_path, page_number)
    if text_layer and text_layer_looks_usable(text_layer):
        return {
            **page_meta,
            "ocr_text": text_layer,
            "ocr_confidence": 1.0,
            "ocr_time_s": 0.0,
            "text_source": "text_layer",
            "text_source_time_s": time.time() - t0,
        }

    doc = pdfium.PdfDocument(pdf_path)
    try:
        bitmap = doc[page_idx].render(scale=RENDER_DPI / 72.0)
        img = bitmap.to_numpy()
    finally:
        doc.close()

    ocr_text, ocr_conf = vision_ocr(img)
    return {
        **page_meta,
        "ocr_text": ocr_text,
        "ocr_confidence": ocr_conf,
        "ocr_time_s": time.time() - t0,
        "text_source": "vision_ocr",
        "text_source_time_s": time.time() - t0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Core extraction
# ═══════════════════════════════════════════════════════════════════════════════

def extract_page_from_ocr(
    model,
    processor,
    generate_fn,
    apply_template_fn,
    page_ocr,
    logger,
    *,
    chunk_size,
):
    """Extract events from OCR text that may already be prefetched.
    
    Returns dict with events, prices, timing info.
    """
    t0 = time.time()
    ocr_text = page_ocr["ocr_text"]
    ocr_conf = page_ocr["ocr_confidence"]
    ocr_time = page_ocr["ocr_time_s"]
    ocr_lines = ocr_text.splitlines()
    route = classify_wsl_page_text(ocr_text)

    if len(ocr_lines) < 3:
        return {
            **page_ocr,
            "ocr_lines": len(ocr_lines),
            "ocr_confidence": ocr_conf,
            "ocr_time_s": ocr_time,
            "llm_time_s": 0,
            "total_time_s": ocr_time,
            "events": [],
            "prices": [],
            "n_chunks": 0,
            "skipped": "too_few_ocr_lines",
        }

    if not route.run_events and not route.run_prices:
        return {
            **page_ocr,
            "ocr_lines": len(ocr_lines),
            "llm_time_s": 0,
            "total_time_s": round(time.time() - t0, 2),
            "n_chunks": 0,
            "events": [],
            "prices": [],
            "n_events": 0,
            "n_prices": 0,
            "page_type": route.page_type,
            "skipped": route.skip_reason or "page_classified_skip",
        }

    # Chunk OCR text and extract events
    chunks = []
    if route.run_events:
        for i in range(0, len(ocr_lines), chunk_size):
            chunks.append("\n".join(ocr_lines[i:i + chunk_size]))

    all_events = []
    llm_time = 0

    for chunk in chunks:
        prompt_text = PROMPT_TEMPLATE.format(text=chunk)
        prompt = apply_template_fn(
            processor, config=model.config,
            prompt=prompt_text, num_images=0,
        )
        if prompt.rstrip().endswith("<think>"):
            prompt = prompt.rstrip() + "\n</think>\n"

        t_llm = time.time()
        result = generate_fn(model, processor, prompt=prompt,
                             max_tokens=MAX_TOKENS, temperature=TEMPERATURE)
        chunk_time = time.time() - t_llm
        llm_time += chunk_time

        raw_events = parse_events_json(result.text)
        all_events.extend([normalize_event(e) for e in raw_events])

    # Extract prices (use first 500 chars which usually has the price section)
    prices = []
    price_text = "\n".join(ocr_lines[:40])
    if route.run_prices:
        prompt_p = PRICE_PROMPT.format(text=price_text)
        prompt = apply_template_fn(
            processor, config=model.config,
            prompt=prompt_p, num_images=0,
        )
        if prompt.rstrip().endswith("<think>"):
            prompt = prompt.rstrip() + "\n</think>\n"

        t_p = time.time()
        result_p = generate_fn(model, processor, prompt=prompt,
                               max_tokens=512, temperature=TEMPERATURE)
        llm_time += time.time() - t_p
        prices = parse_prices_json(result_p.text)

    total_time = time.time() - t0

    return {
        **page_ocr,
        "ocr_lines": len(ocr_lines),
        "ocr_confidence": round(ocr_conf, 3),
        "ocr_time_s": round(ocr_time, 2),
        "llm_time_s": round(llm_time, 2),
        "total_time_s": round(total_time, 2),
        "n_chunks": len(chunks),
        "events": all_events,
        "prices": prices,
        "n_events": len(all_events),
        "n_prices": len(prices),
        "page_type": route.page_type,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="WSL Extraction Orchestrator")
    parser.add_argument("--year", type=int, help="Process single year")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--limit", type=int, help="Max PDFs to process")
    parser.add_argument("--workers", type=int, default=2,
                        help="Parallel render/OCR workers to keep ahead of MLX generation")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                        help="OCR lines per extraction chunk")
    parser.add_argument("--start-page", type=int, default=1,
                        help="First page to process in each PDF (1-indexed)")
    parser.add_argument("--dry-run", action="store_true", help="List files without processing")
    args = parser.parse_args()

    logger = setup_logging()

    # Discover PDFs
    if args.year:
        pattern = f"data/raw/wsl_pdfs/{args.year}/*.pdf"
    else:
        pattern = "data/raw/wsl_pdfs/*/*.pdf"
    
    pdf_paths = sorted(glob.glob(pattern))
    if not pdf_paths:
        logger.error(f"No PDFs found matching {pattern}")
        return

    if args.limit:
        pdf_paths = pdf_paths[:args.limit]

    logger.info(f"Found {len(pdf_paths)} PDFs to process")

    if args.dry_run:
        for p in pdf_paths:
            doc = pdfium.PdfDocument(p)
            print(f"  {p} ({len(doc)} pages)")
            doc.close()
        return

    # Load progress
    progress = load_progress()
    if args.resume:
        completed = set(progress.get("completed", {}).keys())
        logger.info(f"Resuming: {len(completed)} PDFs already completed")
    else:
        completed = set()
        progress = {"completed": {}, "started_at": datetime.now().isoformat(),
                     "total_events": 0, "total_pages": 0}

    # Load model
    logger.info(f"Loading model: {MODEL_ID}")
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template

    model, processor = load(MODEL_ID)
    logger.info("Model loaded")

    # Process
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    total_pdfs = len(pdf_paths)
    pending = [p for p in pdf_paths if p not in completed]
    logger.info(f"Processing {len(pending)}/{total_pdfs} PDFs")

    session_start = time.time()
    session_events = 0
    session_pages = 0
    pages_times = []
    workers = max(1, args.workers)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for pi, pdf_path in enumerate(pending):
            pdf_name = os.path.basename(pdf_path)
            year = os.path.basename(os.path.dirname(pdf_path))

            try:
                doc = pdfium.PdfDocument(pdf_path)
                n_pages = len(doc)
                doc.close()
            except Exception as e:
                logger.error(f"Failed to open {pdf_path}: {e}")
                continue

            pdf_events = 0
            pdf_start = time.time()
            page_indices = list(range(args.start_page - 1, n_pages))
            page_iter = iter(page_indices)
            in_flight = {}

            def submit_page(page_idx):
                page_num = page_idx + 1
                page_key = f"{pdf_name}:p{page_num}"
                page_meta = {
                    "pdf": pdf_name,
                    "year": int(year),
                    "page": page_num,
                    "page_key": page_key,
                    "extracted_at": datetime.now().isoformat(),
                }
                in_flight[page_idx] = executor.submit(
                    prepare_page_text,
                    pdf_path,
                    page_idx,
                    page_meta,
                )

            for _ in range(min(workers, len(page_indices))):
                try:
                    submit_page(next(page_iter))
                except StopIteration:
                    break

            for page_idx in page_indices:
                page_num = page_idx + 1
                page_key = f"{pdf_name}:p{page_num}"

                try:
                    page_ocr = in_flight.pop(page_idx).result()
                    try:
                        submit_page(next(page_iter))
                    except StopIteration:
                        pass

                    result = extract_page_from_ocr(
                        model,
                        processor,
                        generate,
                        apply_chat_template,
                        page_ocr,
                        logger,
                        chunk_size=args.chunk_size,
                    )

                    append_result(result)
                    pdf_events += result["n_events"]
                    session_events += result["n_events"]
                    session_pages += 1
                    pages_times.append(result["total_time_s"])

                except Exception as e:
                    logger.error(f"Error on {page_key}: {e}")
                    continue

            # Update progress
            progress["completed"][pdf_path] = {
                "events": pdf_events,
                "pages": n_pages,
                "time_s": round(time.time() - pdf_start, 1),
            }
            progress["total_events"] = sum(
                v["events"] for v in progress["completed"].values()
            )
            progress["total_pages"] = sum(
                v["pages"] for v in progress["completed"].values()
            )
            save_progress(progress)

            # Progress reporting
            elapsed = time.time() - session_start
            done = pi + 1
            rate = elapsed / done if done > 0 else 0
            eta = rate * (len(pending) - done)
            avg_page_time = np.mean(pages_times[-20:]) if pages_times else 0

            if done % 5 == 0 or done <= 3:
                logger.info(
                    f"[{done}/{len(pending)}] {pdf_name}: {pdf_events} events, "
                    f"{n_pages} pages | Session: {session_events} events, "
                    f"{session_pages} pages | "
                    f"Avg: {avg_page_time:.1f}s/page | "
                    f"OCR workers: {workers} | "
                    f"ETA: {timedelta(seconds=int(eta))}"
                )

    # Final summary
    elapsed = time.time() - session_start
    logger.info(f"\n{'='*60}")
    logger.info(f"EXTRACTION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"PDFs processed: {len(progress['completed'])}")
    logger.info(f"Total pages: {progress['total_pages']}")
    logger.info(f"Total events: {progress['total_events']}")
    logger.info(f"Session time: {timedelta(seconds=int(elapsed))}")
    if pages_times:
        logger.info(f"Avg time/page: {np.mean(pages_times):.1f}s")
    logger.info(f"Output: {JSONL_FILE}")


if __name__ == "__main__":
    main()
