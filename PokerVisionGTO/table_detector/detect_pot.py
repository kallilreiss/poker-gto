"""
Detects the pot size from the poker screenshot.
The pot label (e.g. "Pote: 5,67 BB") appears above the board cards.
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

POT_PATTERN = re.compile(
    r"(?:pote?|pot|total)[:\s]*(\d+[.,]?\d*)\s*(?:bb)?",
    re.IGNORECASE,
)
BB_PATTERN = re.compile(r"(\d+[.,]?\d*)\s*[Bb]{2}", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"(\d+[.,]\d+|\d{2,})")


def detect_pot(image: np.ndarray) -> float:
    """
    Detects the current pot size in BB.

    Args:
        image: BGR numpy array of the full screenshot.

    Returns:
        Pot size in BB as float. Returns 0.0 if not detected.
    """
    h, w = image.shape[:2]

    # Pot label appears above the board, roughly in center-upper area
    y_start = int(h * 0.22)
    y_end = int(h * 0.40)
    x_start = int(w * 0.30)
    x_end = int(w * 0.70)

    region = image[y_start:y_end, x_start:x_end]
    text = _ocr_region(region)

    pot = _parse_pot(text)
    if pot is not None:
        return pot

    # Wider search
    region2 = image[int(h * 0.18):int(h * 0.45), int(w * 0.20):int(w * 0.80)]
    text2 = _ocr_region(region2)
    pot2 = _parse_pot(text2)
    if pot2 is not None:
        return pot2

    return 0.0


def _ocr_region(region: np.ndarray) -> str:
    """
    Runs OCR on a region and returns concatenated text.

    Args:
        region: BGR image to read.

    Returns:
        Raw OCR text.
    """
    if not EASYOCR_AVAILABLE or region.size == 0:
        return ""

    try:
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


def _parse_pot(text: str) -> float | None:
    """
    Parses the pot value from OCR text.

    Handles formats like:
        "Pote: 5,67 BB"
        "Pot 10 BB"
        "5.67 BB"

    Args:
        text: Raw OCR string.

    Returns:
        Float pot value in BB, or None.
    """
    if not text:
        return None

    # Look for pot keyword followed by number
    match = POT_PATTERN.search(text)
    if match:
        raw = match.group(1).replace(",", ".")
        try:
            return float(raw)
        except ValueError:
            pass

    # Number followed by BB
    match = BB_PATTERN.search(text)
    if match:
        raw = match.group(1).replace(",", ".")
        try:
            return float(raw)
        except ValueError:
            pass

    # Any decimal number (fallback)
    match = NUMBER_PATTERN.search(text)
    if match:
        raw = match.group(1).replace(",", ".")
        try:
            val = float(raw)
            if 0.1 <= val <= 5000:
                return val
        except ValueError:
            pass

    return None
