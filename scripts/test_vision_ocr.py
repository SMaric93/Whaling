#!/usr/bin/env python3
"""
Test Apple Vision framework OCR on WSL PDF samples.
Compares speed and quality against RapidOCR baseline.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium

# Apple Vision imports
import objc
from Foundation import NSURL, NSArray
import Vision
import Quartz


def vision_ocr_image(image_path_or_array, recognition_level="accurate"):
    """
    Run Apple Vision text recognition on an image.
    
    Args:
        image_path_or_array: Path to image file or numpy array
        recognition_level: "accurate" or "fast"
    
    Returns:
        list of (text, confidence, bounding_box) tuples
    """
    # If numpy array, save to temp file (Vision needs a CGImage)
    if isinstance(image_path_or_array, np.ndarray):
        import cv2
        tmp_path = "/tmp/wsl_vision_test.png"
        # Convert RGB to BGR for cv2
        if image_path_or_array.shape[2] == 4:
            img = cv2.cvtColor(image_path_or_array, cv2.COLOR_RGBA2BGR)
        else:
            img = cv2.cvtColor(image_path_or_array, cv2.COLOR_RGB2BGR)
        cv2.imwrite(tmp_path, img)
        image_url = NSURL.fileURLWithPath_(tmp_path)
    else:
        image_url = NSURL.fileURLWithPath_(str(image_path_or_array))
    
    # Create image request handler
    handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(
        image_url, None
    )
    
    # Create text recognition request
    request = Vision.VNRecognizeTextRequest.alloc().init()
    
    if recognition_level == "fast":
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelFast)
    else:
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    
    # Set language hints
    request.setRecognitionLanguages_(NSArray.arrayWithArray_(["en-US"]))
    request.setUsesLanguageCorrection_(True)
    
    # Perform OCR
    success, error = handler.performRequests_error_([request], None)
    
    if not success:
        print(f"  Vision error: {error}")
        return []
    
    # Extract results
    results = []
    observations = request.results()
    if observations:
        for obs in observations:
            candidates = obs.topCandidates_(1)
            if candidates and len(candidates) > 0:
                text = candidates[0].string()
                confidence = candidates[0].confidence()
                bbox = obs.boundingBox()
                results.append((text, confidence, bbox))
    
    return results


def vision_ocr_pdf_page(pdf_path, page_number, dpi=150):
    """OCR a single PDF page using Vision framework."""
    pdf = pdfium.PdfDocument(str(pdf_path))
    page = pdf[page_number - 1]
    bitmap = page.render(scale=dpi / 72.0)
    img = bitmap.to_numpy()
    pdf.close()
    
    t0 = time.time()
    results = vision_ocr_image(img)
    ocr_time = time.time() - t0
    
    # Sort by vertical position (top to bottom) then left to right
    # Vision uses bottom-left origin, so invert Y
    lines = sorted(results, key=lambda r: (-r[2].origin.y, r[2].origin.x))
    
    text = "\n".join(r[0] for r in lines)
    avg_conf = np.mean([r[1] for r in lines]) if lines else 0.0
    
    return text, avg_conf, ocr_time, len(lines)


def main():
    samples = [
        "data/raw/wsl_pdfs/1843/wsl_1843_04_11.pdf",
        "data/raw/wsl_pdfs/1846/wsl_1846_06_02.pdf",
        "data/raw/wsl_pdfs/1850/wsl_1850_01_01.pdf",
        "data/raw/wsl_pdfs/1853/wsl_1853_03_01.pdf",
    ]
    
    print("=" * 70)
    print("APPLE VISION OCR TEST - WSL Samples")
    print("=" * 70)
    
    total_time = 0
    total_pages = 0
    
    for pdf_str in samples:
        pdf_path = Path(pdf_str)
        if not pdf_path.exists():
            print(f"SKIP: {pdf_path.name}")
            continue
        
        pdf = pdfium.PdfDocument(str(pdf_path))
        n_pages = len(pdf)
        pdf.close()
        
        print(f"\n{'─' * 60}")
        print(f"FILE: {pdf_path.name} ({n_pages} pages, {pdf_path.stat().st_size/1024/1024:.1f} MB)")
        
        issue_time = 0
        issue_lines = 0
        
        for page_num in range(1, n_pages + 1):
            text, conf, ocr_time, n_lines = vision_ocr_pdf_page(pdf_path, page_num, dpi=150)
            issue_time += ocr_time
            issue_lines += n_lines
            
            # Show first few lines of text
            text_lines = text.split("\n")
            preview = text_lines[:3] if text_lines else ["(empty)"]
            
            print(f"  Page {page_num}: {ocr_time:.2f}s, {n_lines} lines, conf={conf:.3f}")
            for line in preview:
                print(f"    > {line[:100]}")
        
        total_time += issue_time
        total_pages += n_pages
        
        print(f"  TOTAL: {issue_time:.2f}s for {n_pages} pages ({issue_time/n_pages:.2f}s/page), {issue_lines} lines")
    
    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"  Issues processed: {len([s for s in samples if Path(s).exists()])}")
    print(f"  Total pages: {total_pages}")
    print(f"  Total OCR time: {total_time:.1f}s")
    print(f"  Avg per page: {total_time/total_pages:.2f}s")
    print(f"  Avg per issue: {total_time/len([s for s in samples if Path(s).exists()]):.2f}s")
    print(f"  Projection for 3,713 issues: {total_time/len([s for s in samples if Path(s).exists()]) * 3713 / 3600:.1f} hours")
    print(f"\n  Compare: RapidOCR baseline was ~8.5s/page, ~30s/issue")
    print(f"  Speedup: {8.5 / (total_time/total_pages):.1f}x per page")


if __name__ == "__main__":
    main()
