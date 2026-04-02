#!/usr/bin/env python3
"""
Test Qwen2.5-VL-3B-4bit: Option C (VLM only) vs Option E (OCR+LLM).
No thinking mode — pure extraction.
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

MODEL_ID = "models/qwen3-vl-4b-4bit"
MODEL_LABEL = "Qwen3-VL-4B-abliterated (4-bit quantized)"

PROMPT_VLM = """Extract all vessel events from this Whalemen's Shipping List page.
Return ONLY a JSON object:
{"events": [{"vessel_name":"str","captain":"str|null","event_type":"DEPARTURE|ARRIVAL|SPOKEN_WITH|IN_PORT|WRECK","port":"str|null","oil_sperm_bbls":null,"oil_whale_bbls":null}],
"prices": [{"commodity":"sperm_oil|whale_oil|whalebone","price":0.0,"unit":"per_gallon|per_pound"}]}
Return ONLY valid JSON."""

PROMPT_TEXT = """Extract all vessel events from this OCR'd Whalemen's Shipping List text.
Return ONLY a JSON object:
{{"events": [{{"vessel_name":"str","captain":"str|null","event_type":"DEPARTURE|ARRIVAL|SPOKEN_WITH|IN_PORT|WRECK","port":"str|null","oil_sperm_bbls":null,"oil_whale_bbls":null}}],
"prices": [{{"commodity":"sperm_oil|whale_oil|whalebone","price":0.0,"unit":"per_gallon|per_pound"}}]}}
Correct OCR errors. Return ONLY valid JSON.

OCR TEXT:
---
{ocr_text}
---"""


def vision_ocr(img):
    tmp = "/tmp/_wsl_3b_ocr.png"
    cv2.imwrite(tmp, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
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
    lines.sort(key=lambda x: -x[1])
    return "\n".join(l[0] for l in lines), float(np.mean([l[1] for l in lines]) if lines else 0.0)


def parse_json(raw):
    text = raw.strip()
    # Strip thinking tags if present
    if "<think>" in text and "</think>" in text:
        text = text.split("</think>")[-1].strip()
    elif "<think>" in text:
        text = text.replace("<think>", "").strip()
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
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template

    print(f"Loading {MODEL_ID}...")
    t0 = time.time()
    model, processor = load(MODEL_ID)
    print(f"Loaded in {time.time()-t0:.1f}s\n")

    test_pages = [
        ("data/raw/wsl_pdfs/1850/wsl_1850_01_01.pdf", 2, "1850 shipping list"),
        ("data/raw/wsl_pdfs/1843/wsl_1843_04_11.pdf", 4, "1843 vessel list"),
        ("data/raw/wsl_pdfs/1850/wsl_1850_01_01.pdf", 4, "1850 reports"),
    ]

    c_stats = []
    e_stats = []

    for pdf_path_str, page_num, label in test_pages:
        pdf_path = Path(pdf_path_str)
        if not pdf_path.exists():
            continue

        pdf = pdfium.PdfDocument(str(pdf_path))
        bitmap = pdf[page_num - 1].render(scale=150 / 72.0)
        img = bitmap.to_numpy()
        pdf.close()

        tmp_img = f"/tmp/wsl_3b_p{page_num}.png"
        cv2.imwrite(tmp_img, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        print(f"{'='*60}")
        print(f"{label}: {pdf_path.name} p{page_num} ({img.shape})")
        print(f"{'='*60}")

        # ── OPTION C: VLM only ──
        prompt_c = apply_chat_template(
            processor, config=model.config,
            prompt=PROMPT_VLM,
            images=[tmp_img], num_images=1,
        )
        # Disable thinking by injecting </think>
        if prompt_c.rstrip().endswith("<think>"):
            prompt_c = prompt_c.rstrip() + "\n</think>\n"

        t0 = time.time()
        res_c = generate(model, processor, prompt=prompt_c, image=tmp_img,
                         max_tokens=4096, temperature=0.1)
        time_c = time.time() - t0

        parsed_c = parse_json(res_c.text)
        events_c = parsed_c.get("events", [])
        prices_c = parsed_c.get("prices", [])

        print(f"\n  OPTION C (VLM): {time_c:.1f}s")
        print(f"    Events: {len(events_c)}, Prices: {len(prices_c)}")
        if "error" in parsed_c:
            print(f"    ERROR: {parsed_c['raw'][:200]}")
        for e in events_c[:3]:
            print(f"      → {e.get('event_type','?')}: {e.get('vessel_name','?')} "
                  f"(capt={e.get('captain')}, sp={e.get('oil_sperm_bbls')}, wh={e.get('oil_whale_bbls')})")

        c_stats.append({"label": label, "time": time_c, "events": len(events_c), "prices": len(prices_c)})

        # ── OPTION E: OCR + LLM ──
        t0_ocr = time.time()
        ocr_text, ocr_conf = vision_ocr(img)
        ocr_time = time.time() - t0_ocr

        prompt_text = PROMPT_TEXT.format(ocr_text=ocr_text[:4000])
        prompt_e = apply_chat_template(
            processor, config=model.config,
            prompt=prompt_text, num_images=0,
        )
        if prompt_e.rstrip().endswith("<think>"):
            prompt_e = prompt_e.rstrip() + "\n</think>\n"

        t0_llm = time.time()
        res_e = generate(model, processor, prompt=prompt_e,
                         max_tokens=4096, temperature=0.1)
        llm_time = time.time() - t0_llm

        parsed_e = parse_json(res_e.text)
        events_e = parsed_e.get("events", [])
        prices_e = parsed_e.get("prices", [])

        print(f"\n  OPTION E (OCR+LLM): OCR={ocr_time:.1f}s + LLM={llm_time:.1f}s = {ocr_time+llm_time:.1f}s")
        print(f"    OCR: conf={ocr_conf:.2f}, {len(ocr_text.splitlines())} lines")
        print(f"    Events: {len(events_e)}, Prices: {len(prices_e)}")
        if "error" in parsed_e:
            print(f"    ERROR: {parsed_e['raw'][:200]}")
        for e in events_e[:3]:
            print(f"      → {e.get('event_type','?')}: {e.get('vessel_name','?')} "
                  f"(capt={e.get('captain')}, sp={e.get('oil_sperm_bbls')}, wh={e.get('oil_whale_bbls')})")

        e_stats.append({"label": label, "time": ocr_time + llm_time, "ocr": ocr_time,
                        "llm": llm_time, "events": len(events_e), "prices": len(prices_e)})

        print(f"\n  → {'C' if time_c < ocr_time+llm_time else 'E'} faster by "
              f"{abs(time_c - (ocr_time+llm_time)):.1f}s | "
              f"Events: C={len(events_c)} vs E={len(events_e)}")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY ({MODEL_LABEL})")
    print(f"{'='*60}")
    print(f"{'Page':<25} {'C time':>8} {'C events':>9} {'E time':>8} {'E events':>9}")
    print(f"{'-'*60}")
    for c, e in zip(c_stats, e_stats):
        print(f"{c['label']:<25} {c['time']:>7.1f}s {c['events']:>9} {e['time']:>7.1f}s {e['events']:>9}")
    avg_c = np.mean([s["time"] for s in c_stats])
    avg_e = np.mean([s["time"] for s in e_stats])
    print(f"\n  Avg: C={avg_c:.1f}s/page, E={avg_e:.1f}s/page")
    print(f"\n  Full corpus (3,713 issues × 4 pages = 14,852 pages):")
    print(f"    Option C: {avg_c * 14852 / 3600:.1f} hours")
    print(f"    Option E: {avg_e * 14852 / 3600:.1f} hours")


if __name__ == "__main__":
    main()
