#!/usr/bin/env python3
"""
Test Option C vs Option E: VLM-only vs OCR+LLM for WSL extraction.
Uses Qwen3.5-9B-4bit on MLX with thinking disabled.
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

EXTRACTION_PROMPT_VLM = """Extract all vessel events from this Whalemen's Shipping List page as JSON.
Return ONLY a JSON object:
{"page_type": "shipping_list|market_prices|mixed",
 "events": [{"vessel_name": "str", "vessel_type": "ship|bark|brig|schooner|null",
   "captain": "str|null", "port": "str|null",
   "event_type": "DEPARTURE|ARRIVAL|SPOKEN_WITH|REPORTED_AT|WRECK|LOSS|IN_PORT",
   "date": "str|null", "oil_sperm_bbls": null, "oil_whale_bbls": null, "tonnage": null}],
 "prices": [{"commodity": "sperm_oil|whale_oil|whalebone",
   "price_low": 0.0, "price_high": 0.0, "unit": "per_gallon|per_pound"}]}
Extract EVERY vessel. Return ONLY valid JSON."""

EXTRACTION_PROMPT_TEXT = """Extract all vessel events from this OCR text of a Whalemen's Shipping List page.
Return ONLY a JSON object:
{{"page_type": "shipping_list|market_prices|mixed",
 "events": [{{"vessel_name": "str", "vessel_type": "ship|bark|brig|schooner|null",
   "captain": "str|null", "port": "str|null",
   "event_type": "DEPARTURE|ARRIVAL|SPOKEN_WITH|REPORTED_AT|WRECK|LOSS|IN_PORT",
   "date": "str|null", "oil_sperm_bbls": null, "oil_whale_bbls": null, "tonnage": null}}],
 "prices": [{{"commodity": "sperm_oil|whale_oil|whalebone",
   "price_low": 0.0, "price_high": 0.0, "unit": "per_gallon|per_pound"}}]}}
Correct OCR errors in names. Extract EVERY vessel. Return ONLY valid JSON.

OCR TEXT:
---
{ocr_text}
---"""


def vision_ocr_page(img_array):
    """Apple Vision OCR."""
    tmp = "/tmp/_wsl_cmp_ocr.png"
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
    return "\n".join(l[0] for l in lines), float(np.mean([l[1] for l in lines]) if lines else 0.0)


def load_model():
    from mlx_vlm import load
    print("Loading Qwen3.5-9B-4bit...")
    t0 = time.time()
    model, processor = load("mlx-community/Qwen3.5-9B-4bit")
    print(f"Loaded in {time.time()-t0:.1f}s")
    return model, processor


def run_vlm(model, processor, image_path):
    """Option C: Image → VLM → JSON."""
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    prompt = apply_chat_template(
        processor, config=model.config,
        prompt=EXTRACTION_PROMPT_VLM,
        images=[image_path], num_images=1,
    )
    # Disable thinking by injecting </think>
    if prompt.rstrip().endswith("<think>"):
        prompt = prompt.rstrip() + "\n</think>\n"

    t0 = time.time()
    result = generate(model, processor, prompt=prompt, image=image_path,
                      max_tokens=8192, temperature=0.1)
    return result.text, time.time() - t0


def run_text_llm(model, processor, ocr_text):
    """Option E: OCR text → LLM → JSON."""
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    prompt_text = EXTRACTION_PROMPT_TEXT.format(ocr_text=ocr_text[:5000])
    prompt = apply_chat_template(
        processor, config=model.config,
        prompt=prompt_text, num_images=0,
    )
    if prompt.rstrip().endswith("<think>"):
        prompt = prompt.rstrip() + "\n</think>\n"

    t0 = time.time()
    result = generate(model, processor, prompt=prompt,
                      max_tokens=8192, temperature=0.1)
    return result.text, time.time() - t0


def parse_json_output(raw):
    """Parse JSON from model output, stripping think tags."""
    text = raw.strip()
    if "<think>" in text and "</think>" in text:
        text = text.split("</think>")[-1].strip()
    try:
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            return {"error": "parse_failed", "raw": text[:300]}


def main():
    samples = [
        ("data/raw/wsl_pdfs/1850/wsl_1850_01_01.pdf", 2),  # Shipping list
        ("data/raw/wsl_pdfs/1843/wsl_1843_04_11.pdf", 4),  # Vessel list
        ("data/raw/wsl_pdfs/1850/wsl_1850_01_01.pdf", 4),  # Reports page
    ]

    model, processor = load_model()

    print(f"\n{'='*70}")
    print(f"HEAD-TO-HEAD: OPTION C (VLM) vs OPTION E (OCR+LLM)")
    print(f"{'='*70}")

    for pdf_path_str, page_num in samples:
        pdf_path = Path(pdf_path_str)
        if not pdf_path.exists():
            continue

        pdf = pdfium.PdfDocument(str(pdf_path))
        bitmap = pdf[page_num - 1].render(scale=150 / 72.0)
        img = bitmap.to_numpy()
        pdf.close()

        tmp_img = f"/tmp/wsl_cmp_p{page_num}.png"
        cv2.imwrite(tmp_img, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        print(f"\n{'─'*60}")
        print(f"{pdf_path.name} page {page_num} ({img.shape})")

        # ── Option C: VLM only ──
        raw_c, time_c = run_vlm(model, processor, tmp_img)
        result_c = parse_json_output(raw_c)
        events_c = result_c.get("events", [])
        prices_c = result_c.get("prices", [])

        print(f"\n  OPTION C (VLM only): {time_c:.1f}s")
        print(f"    Events: {len(events_c)}, Prices: {len(prices_c)}")
        if "error" in result_c:
            print(f"    ERROR: {result_c['raw'][:200]}")
        for e in events_c[:5]:
            print(f"      → {e.get('event_type','?')}: {e.get('vessel_name','?')} "
                  f"(capt={e.get('captain')}, sp={e.get('oil_sperm_bbls')}, wh={e.get('oil_whale_bbls')})")

        # ── Option E: OCR + LLM ──
        t0_ocr = time.time()
        ocr_text, ocr_conf = vision_ocr_page(img)
        ocr_time = time.time() - t0_ocr

        raw_e, llm_time = run_text_llm(model, processor, ocr_text)
        result_e = parse_json_output(raw_e)
        events_e = result_e.get("events", [])
        prices_e = result_e.get("prices", [])

        print(f"\n  OPTION E (OCR+LLM): OCR={ocr_time:.1f}s + LLM={llm_time:.1f}s = {ocr_time+llm_time:.1f}s")
        print(f"    OCR conf={ocr_conf:.2f}, {len(ocr_text.splitlines())} lines")
        print(f"    Events: {len(events_e)}, Prices: {len(prices_e)}")
        if "error" in result_e:
            print(f"    ERROR: {result_e['raw'][:200]}")
        for e in events_e[:5]:
            print(f"      → {e.get('event_type','?')}: {e.get('vessel_name','?')} "
                  f"(capt={e.get('captain')}, sp={e.get('oil_sperm_bbls')}, wh={e.get('oil_whale_bbls')})")

        # Comparison
        print(f"\n  COMPARISON:")
        print(f"    Speed: C={time_c:.1f}s vs E={ocr_time+llm_time:.1f}s → {'C faster' if time_c < ocr_time+llm_time else 'E faster'}")
        print(f"    Events: C={len(events_c)} vs E={len(events_e)}")
        print(f"    Prices: C={len(prices_c)} vs E={len(prices_e)}")

    print(f"\n{'='*70}")
    print(f"DONE")


if __name__ == "__main__":
    main()
