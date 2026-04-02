#!/usr/bin/env python3
"""
Test: Does contrast + sharpening improve VLM extraction on WSL pages?
Compares raw vs enhanced image VLM output on 1 page using Qwen3.5-9B-4bit.

Usage:
  python scripts/test_rich_prompt.py
  python scripts/test_rich_prompt.py --pdf data/raw/wsl_pdfs/1843/wsl_1843_10_17.pdf --page 2
"""

import sys
import time
import json
import re
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
import cv2
from PIL import Image, ImageEnhance

RENDER_DPI = 200

PROMPT = """/no_think
Extract all vessel events from this Whalemen's Shipping List page as JSON.
Return ONLY a JSON object with this schema:
{"page_type": "shipping_list|market_prices|advertisements|mixed",
 "events": [{"vessel_name": "string", "vessel_type": "ship|bark|brig|schooner|null",
             "captain": "string|null", "port": "string|null",
             "home_port": "string|null", "destination": "string|null",
             "event_type": "DEPARTURE|ARRIVAL|SPOKEN_WITH|REPORTED_AT|WRECK|IN_PORT",
             "date": "string|null",
             "latitude": null, "longitude": null,
             "oil_sperm_bbls": null, "oil_whale_bbls": null,
             "bone_lbs": null, "days_out": "string|null",
             "agent": "string|null", "reported_by": "string|null"}],
 "prices": [{"commodity": "sperm_oil|whale_oil|whalebone",
             "price_low": 0.0, "price_high": 0.0}],
 "notices": [{"type": "casualty|crew_change|condemnation|other",
              "vessel": "string", "detail": "string"}]}
Each vessel ONLY ONCE per event. Do NOT repeat entries.
Extract EVERY vessel. Return ONLY valid JSON."""


def enhance_image(img_path, out_path):
    """Apply contrast + sharpness enhancement, save to out_path."""
    pil = Image.open(img_path)
    if pil.mode != 'L':
        pil = pil.convert('L')
    pil = ImageEnhance.Contrast(pil).enhance(1.5)
    pil = ImageEnhance.Sharpness(pil).enhance(2.0)
    pil.convert('RGB').save(out_path)


def parse_result(raw):
    text = raw.strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1].lstrip("json").strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list):
            return {"events": obj}
    except json.JSONDecodeError:
        pass
    # Fallback
    events = []
    for m in re.finditer(r'\{[^{}]{10,800}\}', text):
        try:
            o = json.loads(m.group())
            if "vessel_name" in o or "v" in o:
                events.append(o)
        except:
            pass
    return {"events": events}


def count_fields(events):
    """Count how many events have each optional field populated."""
    fields = ["home_port", "destination", "days_out", "agent",
              "reported_by", "latitude", "longitude"]
    counts = {}
    for f in fields:
        counts[f] = sum(1 for e in events if e.get(f) is not None)
    # Event type distribution
    et = {}
    for e in events:
        t = e.get("event_type", "?")
        et[t] = et.get(t, 0) + 1
    counts["event_types"] = et
    return counts


def run_vlm(model, processor, generate_fn, apply_template_fn, image_path, label):
    """Run VLM on one image, return parsed result + timing."""
    prompt = apply_template_fn(
        processor, config=model.config,
        prompt=PROMPT,
        images=[image_path],
        num_images=1,
    )

    t0 = time.time()
    result = generate_fn(
        model, processor,
        prompt=prompt,
        image=image_path,
        max_tokens=8192,
        temperature=0.0,
    )
    elapsed = time.time() - t0

    parsed = parse_result(result.text)
    events = parsed.get("events", [])
    prices = parsed.get("prices", [])
    notices = parsed.get("notices", [])

    print(f"\n  {label}")
    print(f"  {'─'*50}")
    print(f"  Time:    {elapsed:.1f}s")
    print(f"  Events:  {len(events)}")
    print(f"  Prices:  {len(prices)}")
    print(f"  Notices: {len(notices)}")

    fc = count_fields(events)
    print(f"  Event types: {fc['event_types']}")
    print(f"  Field coverage:")
    for f in ["home_port", "destination", "days_out", "agent", "reported_by", "latitude"]:
        print(f"    {f}: {fc[f]}/{len(events)}")

    # Sample events
    for e in events[:3]:
        name = e.get("vessel_name", "?")
        captain = e.get("captain", "?")
        et = e.get("event_type", "?")
        port = e.get("port", "?")
        hp = e.get("home_port", "")
        dest = e.get("destination", "")
        sp = e.get("oil_sperm_bbls", "")
        days = e.get("days_out", "")
        agent = e.get("agent", "")
        by = e.get("reported_by", "")
        print(f"    → {et}: {name} (capt={captain}, port={port}, hp={hp}, "
              f"dest={dest}, sp={sp}, days={days}, agent={agent}, by={by})")

    return {
        "events": events, "prices": prices, "notices": notices,
        "time": elapsed, "raw": result.text,
        "field_counts": fc,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default="data/raw/wsl_pdfs/1843/wsl_1843_10_17.pdf")
    parser.add_argument("--page", type=int, default=1, help="Page (1-indexed)")
    args = parser.parse_args()

    # Render page
    doc = pdfium.PdfDocument(args.pdf)
    bmp = doc[args.page - 1].render(scale=RENDER_DPI / 72.0)
    img = bmp.to_numpy()
    doc.close()

    # Save raw and enhanced
    raw_path = "/tmp/wsl_test_raw.png"
    enh_path = "/tmp/wsl_test_enh.png"
    cv2.imwrite(raw_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    enhance_image(raw_path, enh_path)

    print(f"{'='*60}")
    print(f"  VLM Enhancement A/B Test")
    print(f"  PDF: {args.pdf} | Page: {args.page}")
    print(f"  DPI: {RENDER_DPI}")
    print(f"{'='*60}")

    # Load model
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template

    print("\nLoading Qwen3.5-9B-4bit...")
    t0 = time.time()
    model, processor = load("mlx-community/Qwen3.5-9B-4bit")
    print(f"Model loaded in {time.time()-t0:.1f}s")

    # Test A: Raw image
    result_raw = run_vlm(model, processor, generate, apply_chat_template,
                         raw_path, "TEST A: Raw image (no enhancement)")

    # Test B: Enhanced image
    result_enh = run_vlm(model, processor, generate, apply_chat_template,
                         enh_path, "TEST B: Enhanced (contrast 1.5 + sharpness 2.0)")

    # Summary comparison
    print(f"\n{'='*60}")
    print(f"  COMPARISON")
    print(f"{'='*60}")
    er, ee = result_raw, result_enh
    print(f"  {'Metric':<25} {'Raw':>10} {'Enhanced':>10} {'Delta':>10}")
    print(f"  {'-'*55}")
    print(f"  {'Events':<25} {len(er['events']):>10} {len(ee['events']):>10} {len(ee['events'])-len(er['events']):>+10}")
    print(f"  {'Prices':<25} {len(er['prices']):>10} {len(ee['prices']):>10} {len(ee['prices'])-len(er['prices']):>+10}")
    print(f"  {'Notices':<25} {len(er['notices']):>10} {len(ee['notices']):>10} {len(ee['notices'])-len(er['notices']):>+10}")
    print(f"  {'Time (s)':<25} {er['time']:>10.1f} {ee['time']:>10.1f} {ee['time']-er['time']:>+10.1f}")

    for f in ["home_port", "destination", "days_out", "agent", "reported_by", "latitude"]:
        r_val = er['field_counts'][f]
        e_val = ee['field_counts'][f]
        print(f"  {f:<25} {r_val:>10} {e_val:>10} {e_val-r_val:>+10}")

    # Save outputs
    out_dir = Path("data/extracted/test_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "vlm_raw.json", "w") as f:
        json.dump({"events": er["events"], "prices": er["prices"], "notices": er["notices"]}, f, indent=2)
    with open(out_dir / "vlm_enhanced.json", "w") as f:
        json.dump({"events": ee["events"], "prices": ee["prices"], "notices": ee["notices"]}, f, indent=2)
    print(f"\n  Output saved to {out_dir}/")


if __name__ == "__main__":
    main()
