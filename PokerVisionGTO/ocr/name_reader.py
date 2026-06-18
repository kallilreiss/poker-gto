"""
OCR utility for reading player names from the table.
"""

import cv2
import numpy as np

try:
    import easyocr
    _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    EASYOCR_AVAILABLE = True
except Exception:
    EASYOCR_AVAILABLE = False


def read_name(region: np.ndarray) -> str:
    """
    Reads a player name from a cropped image region.

    Args:
        region: BGR crop of the player name plate.

    Returns:
        Player name string, or empty string if unreadable.
    """
    if not EASYOCR_AVAILABLE or region is None or region.size == 0:
        return ""

    try:
        scale = 2
        upscaled = cv2.resize(
            region,
            (region.shape[1] * scale, region.shape[0] * scale),
            interpolation=cv2.INTER_CUBIC,
        )
        results = _reader.readtext(upscaled, detail=0)
        return " ".join(results).strip()
    except Exception:
        return ""
