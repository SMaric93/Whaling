"""
WSL PDF Parser - Extract text from Whalemen's Shipping List PDFs.

Uses pdfplumber for native text extraction with OCR fallback
via pytesseract for scanned pages.
"""

import re
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
import logging

# PDF processing - graceful fallback if not installed
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False
    pdfplumber = None

try:
    from pdf2image import convert_from_path
    import pytesseract
    HAS_OCR = True
except ImportError:
    HAS_OCR = False
    convert_from_path = None
    pytesseract = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WSLPage:
    """Represents a single page of extracted WSL content."""
    page_number: int
    text: str
    extraction_method: str  # 'text_layer' or 'ocr'
    confidence: float


@dataclass
class WSLIssue:
    """Represents a complete WSL issue."""
    issue_id: str
    year: int
    month: Optional[int]
    day: Optional[int]
    pages: List[WSLPage]
    source_path: Path
    
    @property
    def full_text(self) -> str:
        """Concatenate all page texts."""
        return "\n\n".join(p.text for p in self.pages)
    
    @property
    def avg_confidence(self) -> float:
        """Average extraction confidence across pages."""
        if not self.pages:
            return 0.0
        return sum(p.confidence for p in self.pages) / len(self.pages)


def extract_text_layer(pdf_path: Path) -> List[WSLPage]:
    """
    Extract text from PDF using native text layer.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        List of WSLPage objects with extracted text
    """
    if not HAS_PDFPLUMBER:
        raise ImportError("pdfplumber is required for PDF text extraction. "
                          "Install with: pip install pdfplumber")
    
    pages = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            
            # Assess text quality
            if len(text.strip()) > 100:
                confidence = 0.9  # Good text layer
            elif len(text.strip()) > 20:
                confidence = 0.6  # Partial text
            else:
                confidence = 0.1  # Likely needs OCR
            
            pages.append(WSLPage(
                page_number=i + 1,
                text=text,
                extraction_method='text_layer',
                confidence=confidence,
            ))
    
    return pages


def ocr_page(pdf_path: Path, page_number: int, dpi: int = 200) -> WSLPage:
    """
    OCR a single page from a PDF.
    
    Args:
        pdf_path: Path to PDF file
        page_number: 1-indexed page number
        dpi: Resolution for rendering
        
    Returns:
        WSLPage with OCR'd text
    """
    if not HAS_OCR:
        raise ImportError("pytesseract and pdf2image are required for OCR. "
                          "Install with: pip install pytesseract pdf2image")
    
    # Convert single page to image
    images = convert_from_path(
        pdf_path,
        first_page=page_number,
        last_page=page_number,
        dpi=dpi,
    )
    
    if not images:
        return WSLPage(
            page_number=page_number,
            text="",
            extraction_method='ocr_failed',
            confidence=0.0,
        )
    
    # OCR the image
    text = pytesseract.image_to_string(images[0])
    
    # Estimate confidence based on text quality
    # (OCR typically has lower confidence than text layer)
    if len(text.strip()) > 200:
        confidence = 0.7
    elif len(text.strip()) > 50:
        confidence = 0.5
    else:
        confidence = 0.3
    
    return WSLPage(
        page_number=page_number,
        text=text,
        extraction_method='ocr',
        confidence=confidence,
    )


def parse_wsl_issue(
    pdf_path: Path,
    issue_id: Optional[str] = None,
    ocr_threshold: float = 0.3,
    max_ocr_pages: int = 50,
) -> WSLIssue:
    """
    Parse a complete WSL PDF issue.
    
    Attempts text layer extraction first, falls back to OCR
    for pages with low-quality text.
    
    Args:
        pdf_path: Path to PDF file
        issue_id: Issue identifier (extracted from filename if not provided)
        ocr_threshold: Confidence threshold below which to try OCR
        max_ocr_pages: Maximum pages to OCR (to prevent runaway processing)
        
    Returns:
        WSLIssue with extracted content
    """
    # Extract issue_id from filename if not provided
    if issue_id is None:
        issue_id = pdf_path.stem
    
    # Parse date from issue_id
    match = re.search(r'wsl_(\d{4})_(\d{2})_(\d{2})', issue_id)
    if match:
        year, month, day = map(int, match.groups())
    else:
        year_match = re.search(r'(\d{4})', issue_id)
        year = int(year_match.group(1)) if year_match else None
        month, day = None, None
    
    logger.info(f"Parsing WSL issue: {pdf_path.name}")
    
    # First pass: text layer extraction
    pages = extract_text_layer(pdf_path)
    
    # Second pass: OCR for low-confidence pages
    ocr_count = 0
    for i, page in enumerate(pages):
        if page.confidence < ocr_threshold and ocr_count < max_ocr_pages:
            if HAS_OCR:
                try:
                    ocr_page_result = ocr_page(pdf_path, page.page_number)
                    if ocr_page_result.confidence > page.confidence:
                        pages[i] = ocr_page_result
                        ocr_count += 1
                except Exception as e:
                    logger.warning(f"OCR failed for page {page.page_number}: {e}")
            else:
                logger.debug(f"OCR not available for low-confidence page {page.page_number}")
    
    if ocr_count > 0:
        logger.info(f"  OCR'd {ocr_count} pages with low text quality")
    
    return WSLIssue(
        issue_id=issue_id,
        year=year,
        month=month,
        day=day,
        pages=pages,
        source_path=pdf_path,
    )


def batch_parse_wsl_issues(
    pdf_paths: List[Path],
    ocr_threshold: float = 0.3,
) -> List[WSLIssue]:
    """
    Parse multiple WSL PDF issues.
    
    Args:
        pdf_paths: List of PDF file paths
        ocr_threshold: Confidence threshold for OCR fallback
        
    Returns:
        List of parsed WSLIssue objects
    """
    issues = []
    
    for i, pdf_path in enumerate(pdf_paths):
        logger.info(f"[{i+1}/{len(pdf_paths)}] {pdf_path.name}")
        try:
            issue = parse_wsl_issue(pdf_path, ocr_threshold=ocr_threshold)
            issues.append(issue)
        except Exception as e:
            logger.error(f"Failed to parse {pdf_path}: {e}")
    
    logger.info(f"Parsed {len(issues)} issues successfully")
    return issues


def check_dependencies() -> Dict[str, bool]:
    """Check which PDF processing dependencies are available."""
    return {
        'pdfplumber': HAS_PDFPLUMBER,
        'pytesseract': HAS_OCR,
        'pdf2image': HAS_OCR,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse WSL PDFs")
    parser.add_argument("pdf_path", type=Path, help="Path to WSL PDF file")
    parser.add_argument("--ocr-threshold", type=float, default=0.3,
                        help="Confidence threshold for OCR fallback")
    
    args = parser.parse_args()
    
    # Check dependencies
    deps = check_dependencies()
    print(f"Dependencies: {deps}")
    
    if not deps['pdfplumber']:
        print("ERROR: pdfplumber is required. Install with: pip install pdfplumber")
        exit(1)
    
    # Parse the PDF
    issue = parse_wsl_issue(args.pdf_path, ocr_threshold=args.ocr_threshold)
    
    print(f"\nIssue: {issue.issue_id}")
    print(f"Date: {issue.year}-{issue.month or '??'}-{issue.day or '??'}")
    print(f"Pages: {len(issue.pages)}")
    print(f"Avg Confidence: {issue.avg_confidence:.2f}")
    print(f"\nFirst 500 chars of text:")
    print(issue.full_text[:500])
