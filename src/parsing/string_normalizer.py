"""
String normalization utilities for whaling data.

Implements project conventions:
- Convert to UPPER case
- Strip punctuation
- Collapse whitespace
- Remove titles (CAPT, CAPTAIN, MASTER, MR, MRS, DR)
- Keep suffixes (JR, SR)
- Expand common abbreviations (WM→WILLIAM, etc.)
"""

import re
from typing import Optional, List
from pathlib import Path
from datetime import datetime, date

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import STRING_NORM_CONFIG, DATE_PARSE_FORMATS


def normalize_name(
    name: Optional[str],
    expand_abbrevs: bool = True,
    remove_titles: bool = True,
) -> Optional[str]:
    """
    Normalize a person name according to project conventions.
    
    Args:
        name: Raw name string
        expand_abbrevs: If True, expand common abbreviations
        remove_titles: If True, remove titles like CAPT, MR, etc.
        
    Returns:
        Normalized name string, or None if input is None/empty
    """
    if name is None or not str(name).strip():
        return None
    
    # Convert to string and uppercase
    text = str(name).upper()
    
    # Strip punctuation (keep alphanumeric and spaces)
    if STRING_NORM_CONFIG.strip_punctuation:
        text = re.sub(r"[^\w\s]", " ", text)
    
    # Collapse whitespace
    if STRING_NORM_CONFIG.collapse_whitespace:
        text = re.sub(r"\s+", " ", text).strip()
    
    # Split into tokens for processing
    tokens = text.split()
    
    # Remove titles
    if remove_titles:
        tokens = [
            t for t in tokens 
            if t not in STRING_NORM_CONFIG.remove_titles
        ]
    
    # Expand abbreviations
    if expand_abbrevs:
        tokens = [
            STRING_NORM_CONFIG.expand_abbreviations.get(t, t)
            for t in tokens
        ]
    
    # Rejoin
    result = " ".join(tokens)
    
    return result if result else None


def normalize_vessel_name(name: Optional[str]) -> Optional[str]:
    """
    Normalize a vessel name.
    
    Similar to person name normalization but without title removal
    and abbreviation expansion (vessels have different naming conventions).
    
    Args:
        name: Raw vessel name string
        
    Returns:
        Normalized vessel name, or None if input is None/empty
    """
    if name is None or not str(name).strip():
        return None
    
    # Convert to string and uppercase
    text = str(name).upper()
    
    # Strip punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    # Remove common prefixes that don't distinguish vessels
    prefixes_to_remove = {"SHIP", "BARK", "BARQUE", "BRIG", "SCHOONER", "SCH", "BK"}
    tokens = text.split()
    if tokens and tokens[0] in prefixes_to_remove:
        tokens = tokens[1:]
    
    result = " ".join(tokens)
    
    return result if result else None


def expand_abbreviations(text: Optional[str]) -> Optional[str]:
    """
    Expand known abbreviations in text.
    
    Args:
        text: Text potentially containing abbreviations
        
    Returns:
        Text with abbreviations expanded
    """
    if text is None:
        return None
    
    tokens = text.upper().split()
    tokens = [
        STRING_NORM_CONFIG.expand_abbreviations.get(t, t)
        for t in tokens
    ]
    return " ".join(tokens)


def extract_suffix(name: str) -> tuple:
    """
    Extract suffix (JR, SR, etc.) from a name.
    
    Args:
        name: Name string
        
    Returns:
        Tuple of (name_without_suffix, suffix_or_None)
    """
    tokens = name.split()
    
    if tokens and tokens[-1] in STRING_NORM_CONFIG.keep_suffixes:
        return " ".join(tokens[:-1]), tokens[-1]
    
    return name, None


def parse_name_parts(name: Optional[str]) -> dict:
    """
    Parse a name into component parts.
    
    Args:
        name: Normalized name string
        
    Returns:
        Dict with keys: first, middle, last, suffix
    """
    if name is None or not name.strip():
        return {"first": None, "middle": None, "last": None, "suffix": None}
    
    # Extract suffix first
    name_no_suffix, suffix = extract_suffix(name)
    
    tokens = name_no_suffix.split()
    
    if len(tokens) == 0:
        return {"first": None, "middle": None, "last": None, "suffix": suffix}
    elif len(tokens) == 1:
        return {"first": None, "middle": None, "last": tokens[0], "suffix": suffix}
    elif len(tokens) == 2:
        return {"first": tokens[0], "middle": None, "last": tokens[1], "suffix": suffix}
    else:
        return {
            "first": tokens[0],
            "middle": " ".join(tokens[1:-1]),
            "last": tokens[-1],
            "suffix": suffix,
        }


def parse_date(
    date_str: Optional[str],
    return_year_only: bool = False,
) -> Optional[date]:
    """
    Parse a date string into a date object.
    
    Tries multiple formats commonly found in historical whaling records.
    
    Args:
        date_str: Date string to parse
        return_year_only: If True and only year is found, return Jan 1 of that year
        
    Returns:
        date object, or None if parsing fails
    """
    if date_str is None or not str(date_str).strip():
        return None
    
    date_str = str(date_str).strip()
    
    # Try each format
    for fmt in DATE_PARSE_FORMATS:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.date()
        except ValueError:
            continue
    
    # Try to extract just a 4-digit year
    year_match = re.search(r"\b(1[78]\d{2})\b", date_str)
    if year_match and return_year_only:
        year = int(year_match.group(1))
        return date(year, 1, 1)
    
    return None


def parse_year(date_str: Optional[str]) -> Optional[int]:
    """
    Extract just the year from a date string.
    
    Args:
        date_str: Date string
        
    Returns:
        Year as integer, or None
    """
    if date_str is None:
        return None
    
    # Try full date parse first
    parsed = parse_date(str(date_str), return_year_only=True)
    if parsed:
        return parsed.year
    
    # Try direct year extraction
    year_match = re.search(r"\b(1[78]\d{2})\b", str(date_str))
    if year_match:
        return int(year_match.group(1))
    
    return None


def jaro_winkler_similarity(s1: str, s2: str) -> float:
    """
    Compute Jaro-Winkler similarity between two strings.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Similarity score between 0 and 1
    """
    if s1 == s2:
        return 1.0
    
    len1, len2 = len(s1), len(s2)
    
    if len1 == 0 or len2 == 0:
        return 0.0
    
    # Calculate match window
    match_distance = max(len1, len2) // 2 - 1
    if match_distance < 0:
        match_distance = 0
    
    s1_matches = [False] * len1
    s2_matches = [False] * len2
    
    matches = 0
    transpositions = 0
    
    # Find matches
    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break
    
    if matches == 0:
        return 0.0
    
    # Count transpositions
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    
    # Jaro similarity
    jaro = (
        matches / len1 +
        matches / len2 +
        (matches - transpositions / 2) / matches
    ) / 3
    
    # Winkler modification - boost for common prefix
    prefix_len = 0
    for i in range(min(len1, len2, 4)):
        if s1[i] == s2[i]:
            prefix_len += 1
        else:
            break
    
    return jaro + prefix_len * 0.1 * (1 - jaro)


def soundex(name: str) -> str:
    """
    Compute Soundex code for a name.
    
    Args:
        name: Name string (should be normalized)
        
    Returns:
        4-character Soundex code
    """
    if not name:
        return "0000"
    
    name = name.upper()
    
    # Remove non-alphabetic characters
    name = re.sub(r"[^A-Z]", "", name)
    
    if not name:
        return "0000"
    
    # Soundex coding
    codes = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6',
    }
    
    # Keep first letter
    result = name[0]
    
    # Encode rest
    prev_code = codes.get(name[0], '0')
    for char in name[1:]:
        code = codes.get(char, '0')
        if code != '0' and code != prev_code:
            result += code
        prev_code = code
        if len(result) >= 4:
            break
    
    # Pad with zeros
    result = (result + "0000")[:4]
    
    return result
