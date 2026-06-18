"""
Detects available actions from the action buttons at the bottom of the screen.
Typical buttons: Fold, Check, Call, Bet, Raise (or their Portuguese equivalents).
"""

import cv2
import numpy as np

try:
    import easyocr
    _reader = easyocr.Reader(["en", "pt"], gpu=False, verbose=False)
    EASYOCR_AVAILABLE = True
except Exception:
    EASYOCR_AVAILABLE = False

# Map detected text to canonical action names
ACTION_MAP = {
    "fold": "FOLD",
    "check": "CHECK",
    "call": "CALL",
    "bet": "BET",
    "raise": "RAISE",
    "allin": "ALL_IN",
    "all in": "ALL_IN",
    "all-in": "ALL_IN",
    # Portuguese
    "passar": "CHECK",
    "passo": "CHECK",
    "apostar": "BET",
    "aposto": "BET",
    "cobrir": "CALL",
    "aumentar": "RAISE",
    "desistir": "FOLD",
    "tudo": "ALL_IN",
}


def detect_actions(image: np.ndarray) -> list[str]:
    """
    Detects available action buttons from the bottom of the screen.

    Args:
        image: BGR numpy array of the full screenshot.

    Returns:
        List of available action strings, e.g. ["CHECK", "BET"].
    """
    h, w = image.shape[:2]

    # Action buttons are in the bottom portion of the screen
    region = image[int(h * 0.80):h, int(w * 0.55):w]
    text = _ocr_region(region)
    actions = _parse_actions(text)

    if not actions:
        # Try the full bottom strip
        region2 = image[int(h * 0.75):h, :]
        text2 = _ocr_region(region2)
        actions = _parse_actions(text2)

    return actions if actions else ["CHECK", "BET"]  # Safe fallback


def _ocr_region(region: np.ndarray) -> str:
    """
    Runs OCR on the given region.

    Args:
        region: BGR crop to read.

    Returns:
        Concatenated OCR text.
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


def _parse_actions(text: str) -> list[str]:
    """
    Parses action keywords from raw OCR text.

    Args:
        text: Raw OCR output from the action button area.

    Returns:
        List of canonical action strings.
    """
    if not text:
        return []

    text_lower = text.lower()
    found = set()

    for keyword, action in ACTION_MAP.items():
        if keyword in text_lower:
            found.add(action)

    # Logical deduction: if we see BET but not CHECK, there was likely a prior bet
    if "BET" in found and "CHECK" not in found:
        found.add("CALL")

    return sorted(found)
