"""
Detects stack sizes for all players at the table using OCR.
Stacks are displayed in BB next to player names.
"""

import re
import cv2
import numpy as np

try:
    import easyocr
    _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    EASYOCR_AVAILABLE = True
except Exception:
    EASYOCR_AVAILABLE = False


BB_PATTERN = re.compile(r"(\d+[.,]?\d*)\s*[Bb]{2}", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"(\d+[.,]?\d*)")


def detect_stacks(image: np.ndarray) -> dict:
    """
    Detects hero and villain stack sizes from the screenshot.

    Args:
        image: BGR numpy array of the full screenshot.

    Returns:
        Dict with hero_stack (float) and villains (list of floats).
        Example: {"hero_stack": 37.0, "villains": [74.7, 163.1, 99.0, 164.4, 454.2]}
    """
    h, w = image.shape[:2]

    # Hero is always bottom-center
    hero_stack = _detect_hero_stack(image, h, w)

    # Villain stacks are around the perimeter of the table
    villain_stacks = _detect_villain_stacks(image, h, w)

    return {
        "hero_stack": hero_stack,
        "villains": villain_stacks,
    }


def _detect_hero_stack(image: np.ndarray, h: int, w: int) -> float:
    """
    Reads the hero's stack from the bottom-center of the image.

    Args:
        image: Full BGR screenshot.
        h: Image height.
        w: Image width.

    Returns:
        Hero stack in BB as float.
    """
    y_start = int(h * 0.70)
    y_end = int(h * 0.90)
    x_start = int(w * 0.35)
    x_end = int(w * 0.65)

    region = image[y_start:y_end, x_start:x_end]
    text = _ocr_region(region)
    return _parse_bb_value(text, default=0.0)


def _detect_villain_stacks(image: np.ndarray, h: int, w: int) -> list[float]:
    """
    Reads villain stacks from regions around the table perimeter.

    Args:
        image: Full BGR screenshot.
        h: Image height.
        w: Image width.

    Returns:
        List of villain stacks in BB.
    """
    # Approximate player positions for a 9-max table (relative coords)
    villain_regions = [
        # (y_rel_start, y_rel_end, x_rel_start, x_rel_end)
        (0.05, 0.22, 0.28, 0.55),   # Top-center (ernesto_hector area)
        (0.05, 0.22, 0.55, 0.82),   # Top-right
        (0.10, 0.30, 0.72, 0.95),   # Right-top
        (0.30, 0.55, 0.78, 0.98),   # Right-bottom
        (0.55, 0.75, 0.72, 0.95),   # Bottom-right (Panathinaia)
        (0.55, 0.75, 0.02, 0.28),   # Bottom-left (suomy61)
        (0.30, 0.55, 0.02, 0.22),   # Left-bottom (manurojas_fcb)
        (0.05, 0.25, 0.05, 0.30),   # Top-left (Furiano1975)
    ]

    stacks = []
    for (ys, ye, xs, xe) in villain_regions:
        region = image[int(h * ys):int(h * ye), int(w * xs):int(w * xe)]
        text = _ocr_region(region)
        val = _parse_bb_value(text)
        if val is not None and val > 0:
            stacks.append(val)

    return stacks


def _ocr_region(region: np.ndarray) -> str:
    """
    Runs OCR on a region and returns concatenated text.

    Args:
        region: BGR crop to read.

    Returns:
        Raw OCR text string.
    """
    if not EASYOCR_AVAILABLE or region.size == 0:
        return ""

    try:
        # Upscale for better OCR
        scale = 2
        upscaled = cv2.resize(
            region,
            (region.shape[1] * scale, region.shape[0] * scale),
            interpolation=cv2.INTER_CUBIC,
        )
        results = _reader.readtext(upscaled, detail=0)
        return " ".join(results)
    except Exception:
        return ""


def _parse_bb_value(text: str, default: float | None = None) -> float | None:
    """
    Extracts a BB numeric value from OCR text.

    Args:
        text: Raw OCR output.
        default: Value to return if parsing fails.

    Returns:
        Float value in BB, or default.
    """
    if not text:
        return default

    # Try to find a number followed by BB
    match = BB_PATTERN.search(text)
    if match:
        raw = match.group(1).replace(",", ".")
        try:
            return float(raw)
        except ValueError:
            pass

    # Fallback: any number in the region
    match = NUMBER_PATTERN.search(text)
    if match:
        raw = match.group(1).replace(",", ".")
        try:
            val = float(raw)
            if 1 <= val <= 10000:  # Sane BB range
                return val
        except ValueError:
            pass

    return default
