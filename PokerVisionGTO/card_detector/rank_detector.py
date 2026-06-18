"""
Detects the rank of a playing card from a cropped card image.
Uses OCR on the top-left corner of the card.
"""

import cv2
import numpy as np

try:
    import easyocr
    _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    EASYOCR_AVAILABLE = True
except Exception:
    EASYOCR_AVAILABLE = False

VALID_RANKS = {"A", "K", "Q", "J", "T", "10", "9", "8", "7", "6", "5", "4", "3", "2"}

RANK_CORRECTIONS = {
    "1": "A",
    "0": "10",
    "l": "A",
    "i": "1",
    "|": "1",
    "o": "0",
    "O": "0",
}


def detect_rank(card_image: np.ndarray) -> str | None:
    """
    Detects the rank character from a card image.

    Args:
        card_image: BGR crop of a single card.

    Returns:
        Rank string (e.g. "A", "K", "7") or None if not detected.
    """
    if card_image is None or card_image.size == 0:
        return None

    rank_crop = _extract_rank_region(card_image)
    rank = _ocr_rank(rank_crop)

    if rank:
        return rank

    # Fallback: try template-based detection
    return _template_rank(card_image)


def _extract_rank_region(card_image: np.ndarray) -> np.ndarray:
    """
    Extracts the top-left corner of the card where the rank is printed.

    Args:
        card_image: BGR crop of a single card.

    Returns:
        Grayscale crop of the rank area, upscaled for OCR.
    """
    h, w = card_image.shape[:2]

    # Top-left corner: roughly top 35% height, left 45% width
    y_end = max(int(h * 0.40), 20)
    x_end = max(int(w * 0.50), 20)

    corner = card_image[0:y_end, 0:x_end]
    gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)

    # Upscale for better OCR
    scale = 4
    gray = cv2.resize(gray, (gray.shape[1] * scale, gray.shape[0] * scale),
                      interpolation=cv2.INTER_CUBIC)

    # Binarize
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def _ocr_rank(rank_image: np.ndarray) -> str | None:
    """
    Uses EasyOCR to read the rank character from a preprocessed image.

    Args:
        rank_image: Grayscale/binary image of the rank region.

    Returns:
        Rank string or None.
    """
    if not EASYOCR_AVAILABLE:
        return None

    try:
        rgb = cv2.cvtColor(rank_image, cv2.COLOR_GRAY2RGB)
        results = _reader.readtext(rgb, allowlist="AaKkQqJjTt0123456789", detail=0)

        for text in results:
            text = text.strip().upper()
            text = "".join(RANK_CORRECTIONS.get(c, c) for c in text)

            if text in VALID_RANKS:
                return text

            if text == "10":
                return "T"

        return None
    except Exception:
        return None


def _template_rank(card_image: np.ndarray) -> str | None:
    """
    Fallback rank detection using pixel color analysis on the rank corner.
    Attempts to classify common ranks by shape features.

    Args:
        card_image: BGR crop of a single card.

    Returns:
        Best-guess rank string or None.
    """
    h, w = card_image.shape[:2]
    corner = card_image[0:int(h * 0.35), 0:int(w * 0.45)]
    if corner.size == 0:
        return None

    gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Count dark pixels (ink) relative to area
    ink_ratio = np.sum(binary > 0) / binary.size

    # Very rough heuristic based on ink density
    if ink_ratio < 0.05:
        return None
    if ink_ratio > 0.4:
        return "8"

    return None
