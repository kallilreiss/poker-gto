"""
OCR utility for reading stack sizes from cropped image regions.
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
NUMBER_PATTERN = re.compile(r"(\d+[.,]\d+|\d{2,})")


def read_stack(region: np.ndarray) -> float | None:
    """
    Reads a stack value in BB from a cropped image region.

    Args:
        region: BGR crop containing the stack text.

    Returns:
        Stack value as float, or None if unreadable.
    """
    if not EASYOCR_AVAILABLE or region is None or region.size == 0:
        return None

    try:
        scale = 3
        upscaled = cv2.resize(
            region,
            (region.shape[1] * scale, region.shape[0] * scale),
            interpolation=cv2.INTER_CUBIC,
        )
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

        results = _reader.readtext(rgb, detail=0, allowlist="0123456789.,BBbb ")
        text = " ".join(results)

        match = BB_PATTERN.search(text)
        if match:
            return float(match.group(1).replace(",", "."))

        match = NUMBER_PATTERN.search(text)
        if match:
            val = float(match.group(1).replace(",", "."))
            if 1 <= val <= 10000:
                return val

        return None
    except Exception:
        return None
