#!/usr/bin/env python3
"""
WSL Image Preprocessing Module (v2)
====================================

Pre-processes rendered WSL page images before VLM extraction.
Designed to be injected between PDF render and VLM call.

Pipeline (in priority order):
    1. Frame crop       — remove black microfilm borders and neighbor-page slivers
    2. Table segment    — isolate the shipping table region from ads/masthead
    3. Horizontal band  — split dense tables into full-width overlapping bands
    4. Light deskew     — small angle correction from text baselines
    5. Conditional norm — mild contrast/background cleanup only on degraded pages

Design principles:
    - Crop and segment first; photometric enhancement only as fallback
    - Preserve table ruling lines (they help the VLM parse columns)
    - No aggressive sharpening, hard binarization, or line removal
    - Page-type-aware: different treatment for dense tables vs. mixed pages
"""

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1: Frame Crop — Remove microfilm borders and neighbor-page slivers
# ═══════════════════════════════════════════════════════════════════════════════

def crop_frame(img, dark_thresh=80, min_content_frac=0.15, margin_frac=0.005):
    """Detect and crop the printed page rectangle from the microfilm frame.

    These pages have:
      - Black film borders (very dark, <40 pixel value)
      - Neighbor-page slivers (visible on right edge of 1880 scans)
      - Gray margins between the print area and the film edge

    Strategy:
      1. Build a "bright enough" mask (pixels above dark_thresh)
      2. Find rows/cols where ≥min_content_frac of pixels are bright (i.e. the paper)
      3. Crop to the bounding box of those rows/cols
      4. Add a small margin

    Args:
        img: RGB or RGBA numpy array (H, W, C)
        dark_thresh: brightness below this is considered border/frame
        min_content_frac: fraction of a row/col that must be "paper" to be kept
        margin_frac: fraction of dimension to add as safety margin after crop
    """
    if cv2 is None:
        return img, (0, img.shape[0], 0, img.shape[1])

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img.copy()

    # Identify "paper" pixels (brighter than the dark film border)
    paper = gray > dark_thresh

    # For each row: what fraction is paper?
    row_paper = paper.mean(axis=1)
    col_paper = paper.mean(axis=0)

    # The printed page is where most of the row/col is paper
    row_is_page = row_paper > min_content_frac
    col_is_page = col_paper > min_content_frac

    if not row_is_page.any() or not col_is_page.any():
        return img, (0, h, 0, w)

    y1 = int(np.argmax(row_is_page))
    y2 = int(h - np.argmax(row_is_page[::-1]))
    x1 = int(np.argmax(col_is_page))
    x2 = int(w - np.argmax(col_is_page[::-1]))

    # Safety margin
    my = max(int(h * margin_frac), 2)
    mx = max(int(w * margin_frac), 2)
    y1 = max(0, y1 - my)
    y2 = min(h, y2 + my)
    x1 = max(0, x1 - mx)
    x2 = min(w, x2 + mx)

    # Only crop if we're actually removing something meaningful (>3% per side)
    removed_frac = 1.0 - ((y2 - y1) * (x2 - x1)) / (h * w)
    if removed_frac < 0.02:
        return img, (0, h, 0, w)

    cropped = img[y1:y2, x1:x2].copy()
    return cropped, (y1, y2, x1, x2)


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 2: Table/Ad Region Segmentation
# ═══════════════════════════════════════════════════════════════════════════════

def _vertical_projection(gray, ink_thresh=180):
    """Ink density per column (vertical projection profile)."""
    return (gray < ink_thresh).astype(np.float32).mean(axis=0)


def _horizontal_projection(gray, ink_thresh=180):
    """Ink density per row (horizontal projection profile)."""
    return (gray < ink_thresh).astype(np.float32).mean(axis=1)


def _find_deepest_valley(profile, search_start_frac, search_end_frac,
                         min_depth_ratio=0.4, min_width=5):
    """Find the deepest relative valley in a 1-D projection profile.

    Instead of looking for absolute-zero gaps (which don't exist on noisy
    microfilm scans), this finds where density drops most relative to the
    surrounding median. The 1880 ad-column boundary is a dip to ~0.11 in a
    profile that averages ~0.27 — a 60% drop that's easy to detect.

    Args:
        profile: 1-D density array
        search_start_frac: start of search window (fraction of profile length)
        search_end_frac: end of search window
        min_depth_ratio: valley must be this fraction below the median to qualify
        min_width: minimum valley width in pixels

    Returns:
        (valley_x, valley_width, depth_ratio) or None if no valley found
    """
    n = len(profile)
    i1 = int(n * search_start_frac)
    i2 = int(n * search_end_frac)
    region = profile[i1:i2]

    if len(region) < min_width * 3:
        return None

    # Median density in this search region
    med = float(np.median(region))
    if med < 0.01:
        return None  # region is basically empty

    # Smooth to suppress line-level noise
    k = max(5, len(region) // 40)
    if k % 2 == 0:
        k += 1
    smooth = np.convolve(region, np.ones(k) / k, mode='same')

    # Find the deepest point
    min_idx = int(np.argmin(smooth))
    min_val = float(smooth[min_idx])
    depth_ratio = 1.0 - min_val / med

    if depth_ratio < min_depth_ratio:
        return None  # not deep enough

    # Measure valley width (how far below half-depth)
    half_depth = med - (med - min_val) * 0.5
    below_half = smooth < half_depth
    # Find contiguous run around min_idx
    left = min_idx
    while left > 0 and below_half[left - 1]:
        left -= 1
    right = min_idx
    while right < len(below_half) - 1 and below_half[right + 1]:
        right += 1
    width = right - left + 1

    if width < min_width:
        return None

    # Convert back to full-image coordinates
    valley_x = i1 + min_idx
    return (valley_x, width, depth_ratio)


def _find_header_bottom(gray, top_frac=0.15):
    """Find the bottom of the column header row using horizontal projection.

    The header row has column labels ("VESSEL NAME", "CAPTAIN", etc.).
    Below it there's usually a thin horizontal rule or whitespace gap.

    Returns: y-coordinate of the header bottom, or a sensible default.
    """
    h = gray.shape[0]
    search_region = int(h * top_frac)

    # Horizontal projection of the top portion
    proj = _horizontal_projection(gray[:search_region, :])

    # Smooth to find the gap after the header
    kernel = max(3, search_region // 50)
    if kernel % 2 == 0:
        kernel += 1
    smoothed = np.convolve(proj, np.ones(kernel) / kernel, mode='same')

    # Find the first significant gap after some initial content
    # (the header text, then a gap before data rows start)
    found_content = False
    for i in range(len(smoothed)):
        if smoothed[i] > 0.02:
            found_content = True
        elif found_content and smoothed[i] < 0.005:
            return i + 2  # just below the gap

    # Default: 4% of height
    return int(h * 0.04)


def _count_significant_valleys(profile, min_depth_ratio=0.40, min_width=5):
    """Count deep valleys across the full projection profile.

    Purpose: distinguish intra-table column gutters (many regularly-spaced
    valleys) from ad-column boundaries (isolated valleys). Pages like 1843 p4
    have 7 column gutters that all look like "valleys", but a true ad boundary
    is typically one of only 1-2 deep troughs.

    Returns: count of significant valleys found.
    """
    n = len(profile)
    # Smooth
    k = max(5, n // 40)
    if k % 2 == 0:
        k += 1
    smooth = np.convolve(profile, np.ones(k) / k, mode='same')
    med = float(np.median(smooth))
    if med < 0.01:
        return 0

    threshold = med * (1 - min_depth_ratio)

    # Find contiguous runs below threshold
    below = smooth < threshold
    count = 0
    in_valley = False
    valley_start = 0
    for i in range(n):
        if below[i] and not in_valley:
            valley_start = i
            in_valley = True
        elif not below[i] and in_valley:
            width = i - valley_start
            if width >= min_width:
                count += 1
            in_valley = False
    if in_valley:
        width = n - valley_start
        if width >= min_width:
            count += 1

    return count


def segment_table(img, page_type="shipping_table"):
    """Segment the shipping table from ads and non-table content.

    Uses relative valley detection in the vertical projection profile to find
    the boundary between the shipping table and side ad columns. This works
    even on noisy microfilm scans where no column is truly empty.

    Also detects the left-side boundary to crop newspaper-article columns
    that flank the table (common in 1880+ issues).

    Has a multi-valley guard: if the profile has ≥4 significant valleys
    across its full width, these are intra-table column gutters (like the
    1843 8-column table) and segmentation is skipped entirely.

    Returns:
        table_img: the table region (cropped left and/or right)
        header_img: the column header row (if found)
        meta: segmentation metadata dict
    """
    if cv2 is None or page_type not in ("shipping_table", "mixed"):
        return img, None, {"segmented": False}

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img.copy()

    # ── Header detection ──
    header_bottom = _find_header_bottom(gray, top_frac=0.20 if page_type == "mixed" else 0.10)

    # ── Vertical projection of the body ──
    body_gray = gray[header_bottom:, :]
    vproj = _vertical_projection(body_gray)

    # ── Multi-valley guard ──
    # Count deep valleys across the full profile. If there are ≥4, the page
    # has many intra-table column gutters (e.g., 1843's 7-column layout) and
    # any single-valley crop would slice the table internally.
    n_valleys = _count_significant_valleys(vproj, min_depth_ratio=0.40, min_width=5)

    if n_valleys >= 4:
        header_height = max(header_bottom, 15)
        header_img = img[:header_height, :].copy()
        meta = {
            "segmented": True,
            "x_cropped": False,
            "multi_valley_guard": n_valleys,
            "header_height": header_height,
            "table_frac": 1.0,
            "original_size": (h, w),
        }
        return img, header_img, meta

    # ── Find right-side ad column boundary ──
    # Search in the right 55–95% of the page for a density valley.
    # min_width=30: a real ad-column gap is 50-170px wide at 150 DPI.
    # The center gutter of a dual-column table (e.g. 1852 p3) is only
    # ~15px, so a threshold of 30 avoids that false positive.
    right_valley = _find_deepest_valley(vproj, 0.55, 0.95,
                                         min_depth_ratio=0.40, min_width=30)
    table_x2 = w
    if right_valley is not None:
        valley_x, valley_w, depth = right_valley
        table_x2 = valley_x  # crop at the left edge of the valley

    # ── Find left-side ad/article column boundary ──
    # Search in the left 3–35% for a density valley
    left_valley = _find_deepest_valley(vproj, 0.03, 0.35,
                                        min_depth_ratio=0.35, min_width=5)
    table_x1 = 0
    if left_valley is not None:
        valley_x, valley_w, depth = left_valley
        table_x1 = valley_x + valley_w  # crop just after the valley

    # Only apply crop if it removes a meaningful chunk
    x_cropped = (table_x2 < w * 0.90) or (table_x1 > w * 0.05)

    if not x_cropped:
        table_x1 = 0
        table_x2 = w

    # ── Build outputs ──
    header_height = max(header_bottom, 15)
    header_img = img[:header_height, table_x1:table_x2].copy()
    table_img = img[:, table_x1:table_x2].copy()

    table_frac = (table_x2 - table_x1) / w
    meta = {
        "segmented": True,
        "x_cropped": x_cropped,
        "table_x1": table_x1,
        "table_x2": table_x2,
        "right_valley": right_valley,
        "left_valley": left_valley,
        "n_valleys_total": n_valleys,
        "header_height": header_height,
        "table_frac": table_frac,
        "original_size": (h, w),
    }
    return table_img, header_img, meta


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 3: Horizontal Banding for Dense Tables
# ═══════════════════════════════════════════════════════════════════════════════

def _find_row_gaps(gray, min_gap_height=2, ink_thresh=200, gap_thresh=0.005):
    """Find horizontal whitespace gaps between text rows.

    Returns list of (gap_center_y, gap_height) tuples, sorted by y.
    """
    proj = (gray < ink_thresh).astype(np.float32).mean(axis=1)

    in_gap = False
    gap_start = 0
    gaps = []

    for i in range(len(proj)):
        if proj[i] < gap_thresh and not in_gap:
            gap_start = i
            in_gap = True
        elif proj[i] >= gap_thresh and in_gap:
            gap_h = i - gap_start
            if gap_h >= min_gap_height:
                gaps.append(((gap_start + i) // 2, gap_h))
            in_gap = False

    return gaps


def band_table(table_img, header_img=None, n_bands=3, overlap_px=20,
               min_height_for_banding=500):
    """Split a dense table into full-width horizontal bands with overlap.

    Each band:
      - Spans the FULL width (preserves left-to-right column relationships)
      - Overlaps with its neighbors by overlap_px pixels
      - Has the column header prepended at the top (if available)
      - Splits at natural row gaps when possible (to avoid cutting text)

    Args:
        table_img: RGB image of the table
        header_img: RGB image of the column header (prepended to each band)
        n_bands: target number of bands
        overlap_px: pixels of overlap between adjacent bands
        min_height_for_banding: don't band if table is shorter than this
    """
    if cv2 is None:
        return [table_img], {"banded": False, "reason": "no_cv2"}

    h, w = table_img.shape[:2]

    if h < min_height_for_banding:
        return [table_img], {"banded": False, "reason": "too_short"}

    gray = cv2.cvtColor(table_img, cv2.COLOR_RGB2GRAY) if len(table_img.shape) == 3 else table_img

    # Find natural row gaps to use as split points
    row_gaps = _find_row_gaps(gray, min_gap_height=2)

    # Calculate ideal split points (evenly spaced)
    band_h = h // n_bands
    split_ys = []

    for i in range(1, n_bands):
        ideal_y = i * band_h

        # Find the nearest row gap to this ideal split point
        best_y = ideal_y
        best_dist = float('inf')
        for gap_y, gap_h in row_gaps:
            dist = abs(gap_y - ideal_y)
            # Only snap to gaps within 20% of band height
            if dist < band_h * 0.20 and dist < best_dist:
                best_dist = dist
                best_y = gap_y
        split_ys.append(best_y)

    # Build bands with overlap
    bands = []
    for i in range(n_bands):
        y1 = max(0, split_ys[i - 1] - overlap_px) if i > 0 else 0
        y2 = min(h, split_ys[i] + overlap_px) if i < n_bands - 1 else h

        band = table_img[y1:y2, :].copy()

        # Prepend header
        if header_img is not None:
            if header_img.shape[1] != band.shape[1]:
                hdr = cv2.resize(header_img, (band.shape[1], header_img.shape[0]))
            else:
                hdr = header_img
            band = np.vstack([hdr, band])

        bands.append(band)

    meta = {
        "banded": True,
        "n_bands": len(bands),
        "split_ys": split_ys,
        "overlap_px": overlap_px,
        "band_heights": [b.shape[0] for b in bands],
        "row_gaps_found": len(row_gaps),
    }
    return bands, meta


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 4: Light Deskew
# ═══════════════════════════════════════════════════════════════════════════════

def deskew(img, max_angle=2.0, min_lines=5):
    """Small-angle deskew using Hough lines on near-horizontal features.

    Conservative: only corrects angles between 0.1° and max_angle°. Pages
    that are already straight are left untouched.
    """
    if cv2 is None:
        return img, None

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                            minLineLength=w // 4, maxLineGap=10)

    if lines is None or len(lines) < min_lines:
        return img, None

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        if abs(dx) < 20:
            continue
        angle = np.degrees(np.arctan2(y2 - y1, dx))
        if abs(angle) < max_angle * 2:
            angles.append(angle)

    if len(angles) < min_lines:
        return img, None

    median_angle = float(np.median(angles))

    # Only correct if angle is noticeable but not too large
    if abs(median_angle) < 0.1 or abs(median_angle) > max_angle:
        return img, None

    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    deskewed = cv2.warpAffine(img, M, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
    return deskewed, median_angle


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 5: Conditional Contrast/Background Cleanup
# ═══════════════════════════════════════════════════════════════════════════════

def _assess_degradation(gray):
    """Score from 0 (clean) to 1 (severely degraded).

    Checks:
      - Background darkness (microfilm framing makes background gray/dark)
      - Text-to-background contrast
      - Speckle density (using Otsu threshold, not fixed 150)
    """
    h, w = gray.shape[:2]
    reasons = []
    score = 0.0

    # Sample background: take thin strips at multiple heights
    strip_ys = [int(h * f) for f in [0.2, 0.4, 0.6, 0.8]]
    bg_samples = np.concatenate([gray[y:y + 5, :] for y in strip_ys if y + 5 < h])
    bg_brightness = float(np.percentile(bg_samples, 90))  # paper-white level

    if bg_brightness < 200:
        score += 0.25
        reasons.append(f"dark_bg={bg_brightness:.0f}")
    if bg_brightness < 160:
        score += 0.25
        reasons.append("very_dark_bg")

    # Contrast between text (dark) and background (light)
    text_darkness = float(np.percentile(bg_samples, 10))
    contrast = bg_brightness - text_darkness
    if contrast < 80:
        score += 0.3
        reasons.append(f"low_contrast={contrast:.0f}")

    # Speckle: use Otsu threshold to separate text from background,
    # then measure fraction of dark pixels that vanish with one erosion
    # (isolated single-pixel noise). The fixed-150 threshold was giving
    # false positives on clean pages by treating serif details as speckle.
    otsu_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = (gray < otsu_thresh).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(binary, kernel, iterations=1)
    total_dark = max(int(binary.sum()), 1)
    speckle = int(binary.sum() - eroded.sum())
    speckle_frac = speckle / total_dark
    if speckle_frac > 0.4:
        score += 0.2
        reasons.append(f"speckle={speckle_frac:.2f}")

    return min(score, 1.0), reasons


def conditional_cleanup(img, force=False, threshold=0.35):
    """Apply mild background flattening and despeckling on degraded pages.

    Does NOT apply hard binarization, aggressive sharpening, or line removal.
    """
    if cv2 is None:
        return img, {"cleaned": False}

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img.copy()
    degradation, reasons = _assess_degradation(gray)

    if not force and degradation < threshold:
        return img, {"cleaned": False, "degradation": round(degradation, 3), "reasons": reasons}

    # Background estimation via large morphological opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

    # Subtract background, normalize to full range
    norm = gray.astype(np.float32) - background.astype(np.float32)
    norm -= norm.min()
    mx = norm.max()
    if mx > 0:
        norm = (norm / mx * 255).astype(np.uint8)
    else:
        norm = gray

    # Very light median despeckling (3×3 kernel — removes single-pixel noise only)
    cleaned = cv2.medianBlur(norm, 3)

    if len(img.shape) == 3:
        cleaned = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)

    return cleaned, {
        "cleaned": True,
        "degradation": round(degradation, 3),
        "reasons": reasons,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Full Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_page(img, page_type="shipping_table",
                    enable_banding=True, n_bands=3,
                    enable_deskew=True,
                    enable_cleanup=True):
    """Full preprocessing pipeline for a WSL page image.

    Returns:
        images: list of RGB arrays  (1 for simple pages, n_bands for dense tables)
        meta: dict with per-stage metadata
    """
    meta = {
        "original_size": (img.shape[0], img.shape[1]),
        "page_type": page_type,
        "stages": [],
    }

    # ── Stage 1: Frame crop ──
    cropped, crop_box = crop_frame(img)
    if crop_box != (0, img.shape[0], 0, img.shape[1]):
        original_area = img.shape[0] * img.shape[1]
        cropped_area = cropped.shape[0] * cropped.shape[1]
        meta["frame_crop"] = {
            "box": crop_box,
            "removed_pct": round(100 * (1 - cropped_area / original_area), 1),
        }
        meta["stages"].append("frame_crop")
        img = cropped

    meta["after_crop_size"] = (img.shape[0], img.shape[1])

    # ── Stage 2: Table segmentation ──
    header_img = None
    if page_type in ("shipping_table", "mixed"):
        table_img, header_img, seg_meta = segment_table(img, page_type)
        if seg_meta.get("x_cropped"):
            img = table_img
            meta["segmentation"] = seg_meta
            meta["stages"].append("table_segment")
        else:
            meta["segmentation"] = {"segmented": False, "reason": "no_ad_column_found"}
            # Still keep the header for banding
            if header_img is not None:
                meta["segmentation"]["header_height"] = seg_meta.get("header_height")

    # ── Stage 3: Deskew (before banding, so bands are aligned) ──
    if enable_deskew:
        deskewed, angle = deskew(img)
        if angle is not None:
            img = deskewed
            meta["deskew"] = round(angle, 3)
            meta["stages"].append(f"deskew({angle:.2f}°)")

    # ── Stage 4: Banding ──
    if enable_banding and page_type == "shipping_table" and img.shape[0] > 500:
        images, band_meta = band_table(img, header_img, n_bands=n_bands)
        meta["banding"] = band_meta
        meta["stages"].append(f"band({len(images)})")
    else:
        images = [img]
        meta["banding"] = {"banded": False}

    # ── Stage 5: Conditional cleanup ──
    if enable_cleanup:
        cleaned_images = []
        any_cleaned = False
        for band_img in images:
            cl, cl_meta = conditional_cleanup(band_img)
            cleaned_images.append(cl)
            if cl_meta.get("cleaned"):
                any_cleaned = True
        if any_cleaned:
            images = cleaned_images
            meta["stages"].append("cleanup")
        meta["cleanup"] = cl_meta

    meta["n_outputs"] = len(images)
    meta["output_sizes"] = [(im.shape[0], im.shape[1]) for im in images]

    # Pixel accounting
    orig_px = meta["original_size"][0] * meta["original_size"][1]
    out_px = sum(s[0] * s[1] for s in meta["output_sizes"])
    meta["pixel_ratio"] = round(out_px / orig_px, 3)

    return images, meta


# ═══════════════════════════════════════════════════════════════════════════════
# Test Harness
# ═══════════════════════════════════════════════════════════════════════════════

def test_on_page(pdf_path, page_idx, dpi, page_type, desc, save_dir):
    """Render one PDF page, run the pipeline, save results, print report."""
    import pypdfium2 as pdfium
    from pathlib import Path
    from PIL import Image

    doc = pdfium.PdfDocument(str(pdf_path))
    bmp = doc[page_idx].render(scale=dpi / 72.0)
    img = bmp.to_numpy()
    doc.close()

    h, w = img.shape[:2]
    print(f"  Input: {w}×{h} ({dpi} DPI)")

    images, meta = preprocess_page(img, page_type=page_type)

    stages = meta["stages"] if meta["stages"] else ["(none)"]
    print(f"  Pipeline: {' → '.join(stages)}")

    if "frame_crop" in meta:
        print(f"    Frame crop removed {meta['frame_crop']['removed_pct']:.1f}% of pixels")

    if meta.get("segmentation", {}).get("x_cropped"):
        s = meta["segmentation"]
        print(f"    Table region: x=[{s.get('table_x1', 0)}..{s['table_x2']}] "
              f"(kept {s['table_frac']:.0%} of width)")
        if s.get("right_valley"):
            vx, vw, vd = s["right_valley"]
            print(f"      Right valley at x={vx} (width={vw}px, depth={vd:.0%})")
        if s.get("left_valley"):
            vx, vw, vd = s["left_valley"]
            print(f"      Left valley at x={vx} (width={vw}px, depth={vd:.0%})")

    if "deskew" in meta:
        print(f"    Deskew: {meta['deskew']:.2f}°")

    if meta["banding"].get("banded"):
        b = meta["banding"]
        print(f"    Bands: {b['n_bands']} (heights={b['band_heights']}, "
              f"{b['row_gaps_found']} row gaps found)")

    c = meta.get("cleanup", {})
    print(f"    Cleanup: {'APPLIED' if c.get('cleaned') else 'skipped'} "
          f"(degradation={c.get('degradation', 0):.3f}"
          f"{', ' + ', '.join(c.get('reasons', [])) if c.get('reasons') else ''})")

    for i, out in enumerate(images):
        oh, ow = out.shape[:2]
        suffix = f" band{i + 1}/{len(images)}" if len(images) > 1 else ""
        print(f"    Output{suffix}: {ow}×{oh}")

    print(f"    Pixel ratio: {meta['pixel_ratio']:.2f}× "
          f"({'reduced' if meta['pixel_ratio'] < 1 else 'expanded (due to header prepend)'})")

    # Save images
    save = Path(save_dir)
    save.mkdir(parents=True, exist_ok=True)
    stem = Path(pdf_path).stem + f"_p{page_idx + 1}"

    Image.fromarray(img).save(save / f"{stem}_0_raw.jpg", quality=90)
    for i, out in enumerate(images):
        tag = f"_band{i + 1}" if len(images) > 1 else ""
        Image.fromarray(out).save(save / f"{stem}_1_preprocessed{tag}.jpg", quality=90)


def main():
    """Test across all three visual regimes: 1845 (clean), 1852 (dense), 1880 (degraded)."""
    from pathlib import Path

    samples = [
        # (pdf_path, page_idx, page_type, description)
        ("data/raw/wsl_pdfs/1845/wsl_1845_08_05.pdf", 0, "mixed",         "1845 p1 — clean front page (masthead + arrivals)"),
        ("data/raw/wsl_pdfs/1845/wsl_1845_08_05.pdf", 2, "shipping_table", "1845 p3 — clean dense fleet table"),
        ("data/raw/wsl_pdfs/1852/wsl_1852_01_06.pdf", 0, "mixed",         "1852 p1 — front page (arrivals table + market text)"),
        ("data/raw/wsl_pdfs/1852/wsl_1852_01_06.pdf", 2, "shipping_table", "1852 p3 — dense 10-col shipping table"),
        ("data/raw/wsl_pdfs/1880/wsl_1880_01_06.pdf", 3, "shipping_table", "1880 p4 — weekly table + market + ads right"),
        ("data/raw/wsl_pdfs/1880/wsl_1880_01_06.pdf", 4, "shipping_table", "1880 p5 — dense fleet table + ads right"),
        ("data/raw/wsl_pdfs/1880/wsl_1880_01_06.pdf", 5, "mixed",         "1880 p6 — full ads page (should skip)"),
        ("data/raw/wsl_pdfs/1894/wsl_1894_01_02.pdf", 0, "shipping_table", "1894 p1 — late-era shipping table"),
    ]

    save_dir = "data/wsl/preprocess_test_v2"

    print("=" * 72)
    print("WSL IMAGE PREPROCESSING TEST v2")
    print("=" * 72)

    for pdf_path, page_idx, page_type, desc in samples:
        if not Path(pdf_path).exists():
            print(f"\n  SKIP: {pdf_path}")
            continue
        print(f"\n{'─' * 72}")
        print(f"  {desc}")
        print(f"{'─' * 72}")
        try:
            test_on_page(pdf_path, page_idx, dpi=150, page_type=page_type,
                        desc=desc, save_dir=save_dir)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 72}")
    print(f"Output saved to: {save_dir}/")
    print("=" * 72)


if __name__ == "__main__":
    main()
