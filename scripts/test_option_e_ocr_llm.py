#!/usr/bin/env python3
"""
Test Option E: Apple Vision OCR → Qwen3.5-9B for structured extraction.
Two-stage: fast OCR for text, then LLM to parse the text into JSON.
"""

import sys
import time
import json
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
import cv2

import objc
from Foundation import NSURL, NSArray
import Vision

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ── Apple Vision OCR ────────────────────────────────────────────────────────

def vision_ocr_page(img_array):
    tmp = "/tmp/_wsl_e_ocr.png"
    cv2.imwrite(tmp, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(
        NSURL.fileURLWithPath_(tmp), None)
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    req.setRecognitionLanguages_(NSArray.arrayWithArray_(["en-US"]))
    req.setUsesLanguageCorrection_(True)
    handler.performRequests_error_([req], None)
    lines = []
    for obs in (req.results() or []):
        c = obs.topCandidates_(1)
        if c and len(c) > 0:
            lines.append((c[0].string(), c[0].confidence(), obs.boundingBox().origin.y))
    lines.sort(key=lambda x: -x[2])
    text = "\n".join(l[0] for l in lines)
    conf = np.mean([l[1] for l in lines]) if lines else 0.0
    return text, float(conf)


# ── LLM ─────────────────────────────────────────────────────────────────────

EXTRACTION_PROMPT = """/no_think
You are analyzing OCR text from the Whalemen's Shipping List (1843-1914).
The text may have OCR errors. Extract structured data and return ONLY a JSON object:
{{"page_type": "shipping_list|market_prices|advertisements|mixed",
 "events": [{{"vessel_name": "string", "vessel_type": "ship|bark|brig|schooner|null",
             "captain": "string|null", "port": "string|null",
             "event_type": "DEPARTURE|ARRIVAL|SPOKEN_WITH|REPORTED_AT|WRECK|LOSS|IN_PORT",
             "date": "string|null",
             "oil_sperm_bbls": null, "oil_whale_bbls": null,
             "bone_lbs": null, "tonnage": null}}],
 "prices": [{{"commodity": "sperm_oil|whale_oil|whalebone",
             "price_low": 0.0, "price_high": 0.0,
             "unit": "per_gallon|per_barrel|per_pound"}}]}}
Correct obvious OCR errors in vessel/captain names. Extract EVERY vessel.

OCR TEXT:
---
{ocr_text}
---

Return ONLY valid JSON."""


def load_llm():
    from mlx_vlm import load
    print("Loading Qwen3.5-9B-4bit on MLX...")
    t0 = time.time()
    model, processor = load("mlx-community/Qwen3.5-9B-4bit")
    print(f"Model loaded in {time.time()-t0:.1f}s")
    return model, processor


def llm_extract_text(model, processor, ocr_text):
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    prompt_text = EXTRACTION_PROMPT.format(ocr_text=ocr_text[:6000])

    # Text-only prompt (no image tokens)
    prompt = apply_chat_template(
        processor, config=model.config,
        prompt=prompt_text,
        num_images=0,
    )

    t0 = time.time()
    result = generate(
        model, processor,
        prompt=prompt,
        max_tokens=8192,
        temperature=0.1,
    )
    elapsed = time.time() - t0

    raw = result.text.strip()
    if "<think>" in raw and "</think>" in raw:
        raw = raw.split("</think>")[-1].strip()

    parsed = None
    try:
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            parsed = json.loads(raw[start:end])
        except (ValueError, json.JSONDecodeError):
            parsed = {"error": "parse_failed", "raw": raw[:500]}

    return parsed, elapsed, len(result.text)


def main():
    samples = [
        "data/raw/wsl_pdfs/1843/wsl_1843_04_11.pdf",
        "data/raw/wsl_pdfs/1850/wsl_1850_01_01.pdf",
        "data/raw/wsl_pdfs/1860/wsl_1860_01_03.pdf",
    ]
    samples = [s for s in samples if Path(s).exists()]

    model, processor = load_llm()

    print(f"\n{'='*70}")
    print(f"OPTION E: APPLE VISION OCR → QWEN3.5-9B TEXT EXTRACTION")
    print(f"{'='*70}")

    all_stats = []
    total_events = 0
    total_prices = 0

    for pdf_str in samples:
        pdf_path = Path(pdf_str)
        pdf = pdfium.PdfDocument(str(pdf_path))
        n_pages = len(pdf)
        test_pages = min(2, n_pages)

        print(f"\n{'─'*60}")
        print(f"FILE: {pdf_path.name} ({n_pages} pages, testing {test_pages})")

        for page_num in range(1, test_pages + 1):
            bitmap = pdf[page_num - 1].render(scale=150 / 72.0)
            img = bitmap.to_numpy()

            # Stage 1: Apple Vision OCR
            t0_ocr = time.time()
            ocr_text, ocr_conf = vision_ocr_page(img)
            ocr_time = time.time() - t0_ocr

            # Stage 2: LLM extraction
            result, llm_time, output_len = llm_extract_text(model, processor, ocr_text)
            total_time = ocr_time + llm_time

            n_events = len(result.get("events", []))
            n_prices = len(result.get("prices", []))
            total_events += n_events
            total_prices += n_prices

            print(f"\n  Page {page_num}: OCR={ocr_time:.1f}s (conf={ocr_conf:.2f}), "
                  f"LLM={llm_time:.1f}s, total={total_time:.1f}s")
            print(f"    OCR: {len(ocr_text)} chars, {len(ocr_text.splitlines())} lines")
            print(f"    Events: {n_events}, Prices: {n_prices}")

            if "error" in result:
                print(f"    ERROR: {result['error']}")
                print(f"    Raw: {result.get('raw','')[:200]}")

            for e in result.get("events", [])[:5]:
                print(f"      → {e.get('event_type','?')}: {e.get('vessel_name','?')} "
                      f"(capt={e.get('captain','?')}, "
                      f"sp={e.get('oil_sperm_bbls','?')}, "
                      f"wh={e.get('oil_whale_bbls','?')})")
            for p in result.get("prices", []):
                print(f"      $ {p.get('commodity','?')}: "
                      f"${p.get('price_low','?')}-${p.get('price_high','?')} "
                      f"{p.get('unit','?')}")

            all_stats.append({
                "file": pdf_path.name, "page": page_num,
                "ocr_time": ocr_time, "llm_time": llm_time,
                "total_time": total_time,
                "events": n_events, "prices": n_prices,
                "ocr_conf": ocr_conf,
            })

        pdf.close()

    # Summary
    print(f"\n{'='*70}")
    print(f"OPTION E SUMMARY")
    print(f"{'='*70}")
    n = len(all_stats)
    total_ocr = sum(s["ocr_time"] for s in all_stats)
    total_llm = sum(s["llm_time"] for s in all_stats)
    total_all = sum(s["total_time"] for s in all_stats)
    print(f"  Pages: {n}")
    print(f"  OCR:  {total_ocr:.1f}s ({total_ocr/n:.1f}s/page)")
    print(f"  LLM:  {total_llm:.1f}s ({total_llm/n:.1f}s/page)")
    print(f"  Total: {total_all:.1f}s ({total_all/n:.1f}s/page)")
    print(f"  Events: {total_events} ({total_events/n:.1f}/page)")
    print(f"  Prices: {total_prices}")
    avg = total_all / n
    print(f"\n  Projection (3,713 issues × 4 pages):")
    print(f"    Sequential: {avg * 3713 * 4 / 3600:.1f} hours")


if __name__ == "__main__":
    main()
