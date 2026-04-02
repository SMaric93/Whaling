#!/usr/bin/env python3
"""
Test chunk size impact on Option E speed.
Compare 50, 100, 150 line chunks on the densest page (514 OCR lines).
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

PROMPT = """/no_think
Extract vessel events from this Whalemen's Shipping List OCR text.
Return a JSON array: [{{"v":"vessel","c":"captain","e":"dep|arr|spk|inp","p":"port","sp":null,"wh":null}}]
Correct OCR errors. Return ONLY JSON.

TEXT:
---
{text}
---"""


def vision_ocr(img):
    tmp = "/tmp/_wsl_chunk_ocr.png"
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
    lines.sort(key=lambda x: -x[2])
    return "\n".join(l[0] for l in lines)


def count_events(raw):
    text = raw.strip()
    if "<think>" in text and "</think>" in text:
        text = text.split("</think>")[-1].strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"): text = text[4:]
    n = 0
    for m in re.finditer(r'\{[^{}]{10,300}\}', text):
        try:
            obj = json.loads(m.group())
            if "v" in obj: n += 1
        except: pass
    return n


def test_chunk_size(model, processor, ocr_lines, chunk_size):
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    chunks = []
    for i in range(0, len(ocr_lines), chunk_size):
        chunks.append("\n".join(ocr_lines[i:i + chunk_size]))

    total_events = 0
    total_time = 0
    chunk_times = []

    for ci, chunk in enumerate(chunks):
        prompt_text = PROMPT.format(text=chunk)
        prompt = apply_chat_template(
            processor, config=model.config,
            prompt=prompt_text, num_images=0,
        )
        if prompt.rstrip().endswith("<think>"):
            prompt = prompt.rstrip() + "\n</think>\n"

        t0 = time.time()
        result = generate(model, processor, prompt=prompt,
                          max_tokens=4096, temperature=0.1)
        ct = time.time() - t0
        total_time += ct
        chunk_times.append(ct)

        n = count_events(result.text)
        total_events += n

    return {
        "chunk_size": chunk_size,
        "n_chunks": len(chunks),
        "total_time": total_time,
        "total_events": total_events,
        "avg_chunk_time": np.mean(chunk_times),
        "chunk_times": chunk_times,
    }


def main():
    from mlx_vlm import load

    print(f"Loading {MODEL_ID}...")
    model, processor = load(MODEL_ID)
    print(f"Loaded.\n")

    # Use the densest page: 1843 vessel list (514 OCR lines)
    pdf = pdfium.PdfDocument("data/raw/wsl_pdfs/1843/wsl_1843_04_11.pdf")
    bitmap = pdf[3].render(scale=150 / 72.0)
    img = bitmap.to_numpy()
    pdf.close()

    ocr_text = vision_ocr(img)
    ocr_lines = ocr_text.splitlines()
    print(f"OCR: {len(ocr_lines)} lines\n")

    chunk_sizes = [50, 75, 100, 150]

    print(f"{'Chunk':>6} {'Chunks':>7} {'Time':>8} {'Events':>7} {'Avg/chk':>8} {'Per-chunk times'}")
    print(f"{'-'*70}")

    results = []
    for cs in chunk_sizes:
        r = test_chunk_size(model, processor, ocr_lines, cs)
        results.append(r)
        times_str = " ".join(f"{t:.1f}s" for t in r["chunk_times"])
        print(f"{cs:>6} {r['n_chunks']:>7} {r['total_time']:>7.1f}s {r['total_events']:>7} "
              f"{r['avg_chunk_time']:>7.1f}s  [{times_str}]")

    # Projection
    print(f"\n{'='*60}")
    print(f"FULL CORPUS PROJECTION (14,852 pages, avg ~300 OCR lines)")
    print(f"{'='*60}")
    for r in results:
        # Estimate: avg page has ~300 lines, so scale chunks
        avg_lines = 300
        est_chunks = max(1, avg_lines // r["chunk_size"])
        est_time = est_chunks * r["avg_chunk_time"] + 1.0  # +1s OCR
        print(f"  Chunk={r['chunk_size']:>3}: ~{est_chunks} chunks/page × "
              f"{r['avg_chunk_time']:.1f}s = {est_time:.1f}s/page → "
              f"{est_time * 14852 / 3600:.0f} hours")


if __name__ == "__main__":
    main()
