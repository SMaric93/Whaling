#!/usr/bin/env python3
"""
Test Option C: Pure VLM extraction (skip OCR entirely).
Send WSL page images directly to Qwen3.5-9B via MLX.
"""

import sys
import time
import json
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

EXTRACTION_PROMPT = """/no_think
Extract all vessel events from this Whalemen's Shipping List page as JSON.
Return ONLY a JSON object with this schema:
{"page_type": "shipping_list|market_prices|advertisements|mixed",
 "events": [{"vessel_name": "string", "vessel_type": "ship|bark|brig|schooner|null",
             "captain": "string|null", "port": "string|null",
             "event_type": "DEPARTURE|ARRIVAL|SPOKEN_WITH|REPORTED_AT|WRECK|LOSS|IN_PORT",
             "date": "string|null",
             "oil_sperm_bbls": "number|null", "oil_whale_bbls": "number|null",
             "bone_lbs": "number|null", "tonnage": "number|null"}],
 "prices": [{"commodity": "sperm_oil|whale_oil|whalebone",
             "price_low": 0.0, "price_high": 0.0,
             "unit": "per_gallon|per_barrel|per_pound"}]}
Extract EVERY vessel. Be precise. Return ONLY valid JSON, no other text."""


def load_vlm():
    from mlx_vlm import load
    print("Loading Qwen3.5-9B-4bit on MLX...")
    t0 = time.time()
    model, processor = load("mlx-community/Qwen3.5-9B-4bit")
    print(f"Model loaded in {time.time()-t0:.1f}s")
    return model, processor


def vlm_extract_page(model, processor, image_path):
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    # Build prompt with image token
    prompt = apply_chat_template(
        processor, config=model.config,
        prompt=EXTRACTION_PROMPT,
        images=[image_path],
        num_images=1,
    )

    t0 = time.time()
    result = generate(
        model, processor,
        prompt=prompt,
        image=image_path,
        max_tokens=8192,
        temperature=0.1,
    )
    elapsed = time.time() - t0

    raw = result.text.strip()
    # Strip thinking tags if present
    if "<think>" in raw and "</think>" in raw:
        raw = raw.split("</think>")[-1].strip()

    # Parse JSON
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

    model, processor = load_vlm()

    print(f"\n{'='*70}")
    print(f"OPTION C: PURE VLM EXTRACTION (Qwen3.5-9B)")
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
            # Render page to PNG
            bitmap = pdf[page_num - 1].render(scale=150 / 72.0)
            img = bitmap.to_numpy()
            tmp = f"/tmp/wsl_vlm_p{page_num}.png"
            cv2.imwrite(tmp, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            result, elapsed, output_len = vlm_extract_page(model, processor, tmp)

            n_events = len(result.get("events", []))
            n_prices = len(result.get("prices", []))
            total_events += n_events
            total_prices += n_prices

            print(f"\n  Page {page_num}: {elapsed:.1f}s, {output_len} chars")
            print(f"    Type: {result.get('page_type','?')}")
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
                "time": elapsed, "events": n_events, "prices": n_prices,
                "tokens": output_len,
            })

        pdf.close()

    # Summary
    print(f"\n{'='*70}")
    print(f"OPTION C SUMMARY")
    print(f"{'='*70}")
    n = len(all_stats)
    total_time = sum(s["time"] for s in all_stats)
    print(f"  Pages: {n}")
    print(f"  Total time: {total_time:.1f}s ({total_time/n:.1f}s/page)")
    print(f"  Events: {total_events} ({total_events/n:.1f}/page)")
    print(f"  Prices: {total_prices}")
    avg = total_time / n
    print(f"\n  Projection (3,713 issues × 4 pages):")
    print(f"    Sequential: {avg * 3713 * 4 / 3600:.1f} hours")


if __name__ == "__main__":
    main()
