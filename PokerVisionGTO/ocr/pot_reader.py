"""
OCR utility for reading the pot size from a cropped image region.
"""

import re
import cv2
import numpy as np

try:
    import easyocr
    _reader = easyocr.Reader(["en", "pt"], gpu=False, verbose=False)
    EASYOCR_AVAILABLE = True
except Exception:
    EASYOCR_AVAILABLE = False

POT_PATTERN = re.compile(
    r"(?:pote?|pot)[:\s]*(\d+[.,]?\d*)\s*(?:bb)?",
    re.IGNORECASE,
)
BB_PATTERN = re.compile(r"(\d+[.,]?\d*)\s*[Bb]{2}", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"(\d+[.,]\d+)")


def read_pot(region: np.ndarray) -> float | None:
    """
    Reads a pot value in BB from a cropped image region.

    Args:
        region: BGR crop containing the pot label.

    Returns:
        Pot value as float, or None if unreadable.
    """
    if not EASYOCR_AVAILABLE or region is None or region.size == 0:
        return None

    try:
        scale = 2
        upscaled = cv2.resize(
            region,
            (region.shape[1] * scale, region.shape[0] * scale),
            interpolation=cv2.INTER_CUBIC,
        )
        results = _reader.readtext(upscaled, detail=0)
        text = " ".join(results)

        match = POT_PATTERN.search(text)
        if match:
            return float(match.group(1).replace(",", "."))

        match = BB_PATTERN.search(text)
        if match:
            return float(match.group(1).replace(",", "."))

        match = NUMBER_PATTERN.search(text)
        if match:
            val = float(match.group(1).replace(",", "."))
            if 0.1 <= val <= 5000:
                return val

        return None
    except Exception:
        return None
