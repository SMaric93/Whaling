#!/usr/bin/env python3
"""
Optimized WSL extraction pipeline using Qwen3-VL-4B-4bit on MLX.
Key optimizations:
  1. Compact JSON schema (single-letter keys → smaller output)
  2. max_tokens=8192
  3. OCR text chunking for Option E (200-line blocks)
  4. Graceful handling of truncated JSON
"""

import sys, time, json, re
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
import cv2

import objc
from Foundation import NSURL, NSArray
import Vision

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

MODEL_ID = "mlx-community/Qwen3-VL-4B-Instruct-4bit"

# ── Compact prompts ──────────────────────────────────────────────────────────

PROMPT_VLM = """You are reading a page from the Whalemen's Shipping List (1843-1914).
Extract ALL vessel entries as a JSON array. Use this COMPACT format:
[{"v":"vessel name","t":"ship|bark|brig|sch","c":"captain","p":"port","e":"dep|arr|spk|rpt|wrk|inp","sp":null,"wh":null,"bn":null}]
Where: v=vessel, t=type, c=captain, p=port, e=event(dep=departure,arr=arrival,spk=spoken,rpt=reported,wrk=wreck,inp=in_port), sp=sperm_oil_bbls, wh=whale_oil_bbls, bn=bone_lbs.
Also extract prices if visible: {"prices":[{"c":"sperm|whale|bone","lo":0.0,"hi":0.0}]}
Return ONLY the JSON array. No explanation."""

PROMPT_TEXT = """/no_think
Extract vessel events from this Whalemen's Shipping List OCR text.
Return a JSON array using COMPACT keys:
[{{"v":"vessel","t":"ship|bark|brig|sch","c":"captain","p":"port","e":"dep|arr|spk|rpt|wrk|inp","sp":null,"wh":null}}]
Also extract prices: {{"prices":[{{"c":"sperm|whale|bone","lo":0.0,"hi":0.0}}]}}
Correct OCR errors in names. Return ONLY JSON.

TEXT:
---
{text}
---"""

# ── Apple Vision OCR ─────────────────────────────────────────────────────────

def vision_ocr(img):
    tmp = "/tmp/_wsl_opt_ocr.png"
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
    # Sort by vertical position (top to bottom)
    lines.sort(key=lambda x: -x[2])
    return "\n".join(l[0] for l in lines), float(np.mean([l[1] for l in lines]) if lines else 0.0)


# ── JSON parsing (handles truncation) ────────────────────────────────────────

def parse_json_robust(raw):
    """Parse JSON from model output, handling truncation and thinking tags."""
    text = raw.strip()

    # Strip thinking
    if "<think>" in text and "</think>" in text:
        text = text.split("</think>")[-1].strip()
    elif text.startswith("<think>"):
        text = text.replace("<think>", "").strip()

    # Strip markdown code fences
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

    # Try direct parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return {"events": result, "prices": []}
        return result
    except json.JSONDecodeError:
        pass

    # Try to find and parse the largest valid JSON
    events = []
    prices = []

    # Extract individual event objects from truncated JSON
    for m in re.finditer(r'\{[^{}]{10,300}\}', text):
        try:
            obj = json.loads(m.group())
            if "v" in obj or "vessel_name" in obj or "vessel" in obj:
                events.append(obj)
            elif "c" in obj and ("lo" in obj or "hi" in obj):
                prices.append(obj)
        except json.JSONDecodeError:
            continue

    if events or prices:
        return {"events": events, "prices": prices}

    return {"events": [], "prices": [], "error": "parse_failed", "raw": text[:300]}


def normalize_event(e):
    """Normalize compact keys to full names."""
    return {
        "vessel_name": e.get("v") or e.get("vessel_name") or e.get("vessel", "?"),
        "vessel_type": e.get("t") or e.get("vessel_type", None),
        "captain": e.get("c") or e.get("captain", None),
        "port": e.get("p") or e.get("port", None),
        "event_type": e.get("e") or e.get("event_type", "?"),
        "oil_sperm_bbls": e.get("sp") or e.get("oil_sperm_bbls"),
        "oil_whale_bbls": e.get("wh") or e.get("oil_whale_bbls"),
        "bone_lbs": e.get("bn") or e.get("bone_lbs"),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

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

        tmp_img = f"/tmp/wsl_opt_p{page_num}.png"
        cv2.imwrite(tmp_img, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        print(f"{'='*65}")
        print(f"{label}: {pdf_path.name} p{page_num} ({img.shape})")
        print(f"{'='*65}")

        # ── OPTION C: VLM direct ──
        prompt_c = apply_chat_template(
            processor, config=model.config,
            prompt=PROMPT_VLM,
            images=[tmp_img], num_images=1,
        )
        if prompt_c.rstrip().endswith("<think>"):
            prompt_c = prompt_c.rstrip() + "\n</think>\n"

        t0 = time.time()
        res_c = generate(model, processor, prompt=prompt_c, image=tmp_img,
                         max_tokens=8192, temperature=0.1)
        time_c = time.time() - t0

        parsed_c = parse_json_robust(res_c.text)
        events_c = [normalize_event(e) for e in parsed_c.get("events", [])]
        prices_c = parsed_c.get("prices", [])

        print(f"\n  OPTION C (VLM): {time_c:.1f}s, {len(res_c.text)} chars output")
        print(f"    Events: {len(events_c)}, Prices: {len(prices_c)}")
        if "error" in parsed_c:
            print(f"    PARSE ERROR: {parsed_c.get('raw','')[:150]}")
        for e in events_c[:5]:
            print(f"      → {e['event_type']}: {e['vessel_name']} "
                  f"(capt={e['captain']}, port={e['port']}, "
                  f"sp={e['oil_sperm_bbls']}, wh={e['oil_whale_bbls']})")

        c_stats.append({"label": label, "time": time_c,
                        "events": len(events_c), "prices": len(prices_c),
                        "chars": len(res_c.text)})

        # ── OPTION E: OCR → chunked LLM ──
        t0_ocr = time.time()
        ocr_text, ocr_conf = vision_ocr(img)
        ocr_time = time.time() - t0_ocr

        ocr_lines = ocr_text.splitlines()
        CHUNK_SIZE = 150
        chunks = []
        for i in range(0, len(ocr_lines), CHUNK_SIZE):
            chunks.append("\n".join(ocr_lines[i:i + CHUNK_SIZE]))

        all_events_e = []
        all_prices_e = []
        llm_time = 0

        for ci, chunk in enumerate(chunks):
            prompt_text = PROMPT_TEXT.format(text=chunk)
            prompt_e = apply_chat_template(
                processor, config=model.config,
                prompt=prompt_text, num_images=0,
            )
            if prompt_e.rstrip().endswith("<think>"):
                prompt_e = prompt_e.rstrip() + "\n</think>\n"

            t0_llm = time.time()
            res_e = generate(model, processor, prompt=prompt_e,
                             max_tokens=4096, temperature=0.1)
            chunk_time = time.time() - t0_llm
            llm_time += chunk_time

            parsed_e = parse_json_robust(res_e.text)
            chunk_events = [normalize_event(e) for e in parsed_e.get("events", [])]
            chunk_prices = parsed_e.get("prices", [])
            all_events_e.extend(chunk_events)
            all_prices_e.extend(chunk_prices)

            if len(chunks) > 1:
                print(f"    Chunk {ci+1}/{len(chunks)}: {chunk_time:.1f}s, "
                      f"{len(chunk_events)} events, {len(chunk_prices)} prices")

        total_e = ocr_time + llm_time

        print(f"\n  OPTION E (OCR+LLM): OCR={ocr_time:.1f}s + LLM={llm_time:.1f}s "
              f"({len(chunks)} chunks) = {total_e:.1f}s")
        print(f"    OCR: conf={ocr_conf:.2f}, {len(ocr_lines)} lines")
        print(f"    Events: {len(all_events_e)}, Prices: {len(all_prices_e)}")
        for e in all_events_e[:5]:
            print(f"      → {e['event_type']}: {e['vessel_name']} "
                  f"(capt={e['captain']}, port={e['port']}, "
                  f"sp={e['oil_sperm_bbls']}, wh={e['oil_whale_bbls']})")

        e_stats.append({"label": label, "time": total_e, "ocr": ocr_time,
                        "llm": llm_time, "chunks": len(chunks),
                        "events": len(all_events_e), "prices": len(all_prices_e)})

        winner = "C" if time_c < total_e else "E"
        print(f"\n  → {winner} faster | C: {len(events_c)} events in {time_c:.1f}s | "
              f"E: {len(all_events_e)} events in {total_e:.1f}s")

    # ── Summary ──
    print(f"\n{'='*65}")
    print(f"OPTIMIZED PIPELINE SUMMARY (Qwen3-VL-4B-4bit)")
    print(f"{'='*65}")
    print(f"{'Page':<25} {'C time':>8} {'C evts':>7} {'E time':>8} {'E evts':>7} {'E chunks':>9}")
    print(f"{'-'*65}")
    for c, e in zip(c_stats, e_stats):
        print(f"{c['label']:<25} {c['time']:>7.1f}s {c['events']:>7} "
              f"{e['time']:>7.1f}s {e['events']:>7} {e['chunks']:>9}")

    avg_c = np.mean([s["time"] for s in c_stats])
    avg_e = np.mean([s["time"] for s in e_stats])
    total_events_c = sum(s["events"] for s in c_stats)
    total_events_e = sum(s["events"] for s in e_stats)

    print(f"\n  Option C: avg {avg_c:.1f}s/page, {total_events_c} total events")
    print(f"  Option E: avg {avg_e:.1f}s/page, {total_events_e} total events")
    print(f"\n  Full corpus projection (3,713 issues × 4 pages = 14,852 pages):")
    print(f"    Option C: {avg_c * 14852 / 3600:.1f} hours")
    print(f"    Option E: {avg_e * 14852 / 3600:.1f} hours")


if __name__ == "__main__":
    main()
