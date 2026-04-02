#!/usr/bin/env python3
"""
Run Apple Vision OCR on a larger WSL sample (~50 issues across decades).
Includes event extraction to test full pipeline quality.
"""

import sys
import time
import json
import random
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pypdfium2 as pdfium
import cv2

import objc
from Foundation import NSURL, NSArray
import Vision


def vision_ocr_page(img_array):
    """OCR a numpy image array using Apple Vision. Returns (text, confidence)."""
    tmp_path = "/tmp/_wsl_ocr_page.png"
    bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    cv2.imwrite(tmp_path, bgr)

    handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(
        NSURL.fileURLWithPath_(tmp_path), None
    )
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setRecognitionLanguages_(NSArray.arrayWithArray_(["en-US"]))
    request.setUsesLanguageCorrection_(True)

    success, error = handler.performRequests_error_([request], None)
    if not success:
        return "", 0.0

    lines = []
    confidences = []
    for obs in (request.results() or []):
        candidates = obs.topCandidates_(1)
        if candidates and len(candidates) > 0:
            lines.append((candidates[0].string(), candidates[0].confidence(),
                          obs.boundingBox().origin.y))
            confidences.append(candidates[0].confidence())

    # Sort top-to-bottom (Vision uses bottom-left origin)
    lines.sort(key=lambda x: -x[2])
    text = "\n".join(l[0] for l in lines)
    avg_conf = np.mean(confidences) if confidences else 0.0
    return text, float(avg_conf)


def ocr_pdf_vision(pdf_path, dpi=150):
    """OCR all pages in a PDF using Apple Vision. Returns full text + metadata."""
    pdf = pdfium.PdfDocument(str(pdf_path))
    n_pages = len(pdf)

    pages = []
    total_time = 0
    for i in range(n_pages):
        bitmap = pdf[i].render(scale=dpi / 72.0)
        img = bitmap.to_numpy()

        t0 = time.time()
        text, conf = vision_ocr_page(img)
        elapsed = time.time() - t0
        total_time += elapsed

        pages.append({"page": i + 1, "text": text, "confidence": conf,
                       "time": elapsed, "n_lines": len(text.split("\n"))})

    pdf.close()
    full_text = "\n\n".join(p["text"] for p in pages)
    avg_conf = np.mean([p["confidence"] for p in pages]) if pages else 0.0
    return {
        "full_text": full_text,
        "pages": pages,
        "n_pages": n_pages,
        "avg_confidence": float(avg_conf),
        "total_ocr_time": total_time,
    }


def extract_events_from_text_simple(text, issue_id, year):
    """Lightweight event extraction (regex-based) on Vision output."""
    import re

    events = []
    segments = re.split(r'[\n.]+', text)

    vessel_pattern = re.compile(
        r'\b(?:ship|bark|brig|schooner|bk\.?|sch\.?)\s+'
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', re.IGNORECASE)
    captain_pattern = re.compile(
        r'(?:Capt(?:ain)?\.?\s+|Master\s+)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        re.IGNORECASE)

    event_keywords = {
        'ARRIVAL': re.compile(r'\b(?:arrived|arr\.?|returned|came\s+in)\b', re.I),
        'DEPARTURE': re.compile(r'\b(?:sailed|departed|cleared|left)\b', re.I),
        'SPOKEN_WITH': re.compile(r'\b(?:spoke|spoken|spk\.?)\b', re.I),
        'REPORTED_AT': re.compile(r'\b(?:reported|rep\.?|seen|heard\s+from)\b', re.I),
        'WRECK': re.compile(r'\b(?:wrecked|wreck|aground|stranded)\b', re.I),
        'LOSS': re.compile(r'\b(?:lost|abandoned|sunk|foundered)\b', re.I),
    }

    for seg in segments:
        seg = seg.strip()
        if len(seg) < 15:
            continue

        vessel_match = vessel_pattern.search(seg)
        if not vessel_match:
            continue

        event_type = 'OTHER'
        for etype, pattern in event_keywords.items():
            if pattern.search(seg):
                event_type = etype
                break

        captain = captain_pattern.search(seg)
        events.append({
            'vessel': vessel_match.group(1),
            'captain': captain.group(1) if captain else None,
            'type': event_type,
            'snippet': seg[:120],
        })

    return events


def main():
    wsl_dir = Path("data/raw/wsl_pdfs")

    # Collect all PDFs grouped by decade
    pdfs_by_decade = defaultdict(list)
    for pdf_path in sorted(wsl_dir.rglob("*.pdf")):
        year = int(pdf_path.parent.name)
        decade = (year // 10) * 10
        pdfs_by_decade[decade].append(pdf_path)

    # Sample ~7 per decade = ~50 total
    random.seed(42)
    sample = []
    for decade in sorted(pdfs_by_decade):
        available = pdfs_by_decade[decade]
        n = min(7, len(available))
        chosen = random.sample(available, n)
        sample.extend(sorted(chosen))

    print(f"Sample: {len(sample)} issues across {len(pdfs_by_decade)} decades")
    for decade in sorted(pdfs_by_decade):
        n_avail = len(pdfs_by_decade[decade])
        n_sampled = len([s for s in sample if int(s.parent.name) // 10 * 10 == decade])
        print(f"  {decade}s: {n_sampled} sampled / {n_avail} available")

    print(f"\n{'=' * 70}")

    all_stats = []
    total_events = 0
    event_type_counts = Counter()
    decade_stats = defaultdict(lambda: {"time": 0, "pages": 0, "events": 0, "issues": 0})

    for i, pdf_path in enumerate(sample):
        year = int(pdf_path.parent.name)
        decade = (year // 10) * 10

        t0 = time.time()
        result = ocr_pdf_vision(pdf_path, dpi=150)
        wall_time = time.time() - t0

        events = extract_events_from_text_simple(
            result["full_text"], pdf_path.stem, year)

        total_events += len(events)
        for e in events:
            event_type_counts[e['type']] += 1

        stat = {
            "file": pdf_path.name,
            "year": year,
            "pages": result["n_pages"],
            "conf": result["avg_confidence"],
            "ocr_time": result["total_ocr_time"],
            "wall_time": wall_time,
            "events": len(events),
            "text_len": len(result["full_text"]),
        }
        all_stats.append(stat)

        decade_stats[decade]["time"] += result["total_ocr_time"]
        decade_stats[decade]["pages"] += result["n_pages"]
        decade_stats[decade]["events"] += len(events)
        decade_stats[decade]["issues"] += 1

        if (i + 1) % 5 == 0 or i == len(sample) - 1:
            elapsed_total = sum(s["ocr_time"] for s in all_stats)
            remaining = (elapsed_total / (i + 1)) * (len(sample) - i - 1)
            print(f"[{i+1}/{len(sample)}] {pdf_path.name}: "
                  f"{result['n_pages']}p, {result['total_ocr_time']:.1f}s, "
                  f"conf={result['avg_confidence']:.2f}, {len(events)} events "
                  f"(ETA: {remaining/60:.1f}m)")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"RESULTS - {len(sample)} issues")
    print(f"{'=' * 70}")

    total_pages = sum(s["pages"] for s in all_stats)
    total_ocr = sum(s["ocr_time"] for s in all_stats)
    total_wall = sum(s["wall_time"] for s in all_stats)

    print(f"\nPerformance:")
    print(f"  Total pages: {total_pages}")
    print(f"  Total OCR time: {total_ocr:.1f}s ({total_ocr/60:.1f}m)")
    print(f"  Avg per page: {total_ocr/total_pages:.2f}s")
    print(f"  Avg per issue: {total_ocr/len(sample):.2f}s")

    print(f"\nQuality:")
    confs = [s["conf"] for s in all_stats]
    print(f"  Confidence: min={min(confs):.2f}, mean={np.mean(confs):.2f}, max={max(confs):.2f}")

    print(f"\nEvent Extraction:")
    print(f"  Total events: {total_events}")
    print(f"  Avg per issue: {total_events/len(sample):.1f}")
    print(f"  Types: {dict(event_type_counts)}")

    print(f"\nPer-Decade Breakdown:")
    print(f"  {'Decade':<8} {'Issues':<8} {'Pages':<8} {'OCR(s)':<10} {'s/page':<8} {'Events':<8} {'Ev/issue':<10}")
    for decade in sorted(decade_stats):
        d = decade_stats[decade]
        spp = d["time"] / d["pages"] if d["pages"] else 0
        epi = d["events"] / d["issues"] if d["issues"] else 0
        print(f"  {decade}s    {d['issues']:<8} {d['pages']:<8} {d['time']:<10.1f} {spp:<8.2f} {d['events']:<8} {epi:<10.1f}")

    print(f"\nProjections for full 3,713 issues:")
    avg_per_issue = total_ocr / len(sample)
    print(f"  Sequential: {avg_per_issue * 3713 / 3600:.1f} hours")
    print(f"  4 workers:  {avg_per_issue * 3713 / 3600 / 4:.1f} hours")
    print(f"  6 workers:  {avg_per_issue * 3713 / 3600 / 6:.1f} hours")

    # Show some high-quality event examples
    print(f"\nSample Events (top confident):")
    all_events_flat = []
    for pdf_path in sample[:10]:
        year = int(pdf_path.parent.name)
        result = ocr_pdf_vision(pdf_path, dpi=150)
        events = extract_events_from_text_simple(result["full_text"], pdf_path.stem, year)
        for e in events:
            if e['type'] != 'OTHER':
                all_events_flat.append(e)
    for e in all_events_flat[:10]:
        print(f"  [{e['type']}] vessel={e['vessel']}, capt={e['captain']}")
        print(f"    \"{e['snippet'][:80]}\"")


if __name__ == "__main__":
    main()
