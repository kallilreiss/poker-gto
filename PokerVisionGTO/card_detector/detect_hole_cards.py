"""
Detects the Hero's hole cards from a poker screenshot.
Hero is always at the bottom-center of the image.
"""

import cv2
import numpy as np
from PIL import Image
from card_detector.rank_detector import detect_rank
from card_detector.suit_detector import detect_suit


RANK_MAP = {
    "A": "A", "K": "K", "Q": "Q", "J": "J", "T": "T",
    "10": "T", "9": "9", "8": "8", "7": "7", "6": "6",
    "5": "5", "4": "4", "3": "3", "2": "2",
}

SUIT_SYMBOL_MAP = {
    "clubs": "c",
    "diamonds": "d",
    "hearts": "h",
    "spades": "s",
}


def detect_hole_cards(image: np.ndarray) -> list[str]:
    """
    Detects the Hero's two hole cards from the screenshot.

    Args:
        image: BGR numpy array of the full screenshot.

    Returns:
        List of card strings, e.g. ["7c", "4c"]. Empty list if not detected.
    """
    h, w = image.shape[:2]

    # Hero area: bottom-center strip (roughly bottom 35% of image, center 40% width)
    y_start = int(h * 0.55)
    y_end = int(h * 0.85)
    x_start = int(w * 0.30)
    x_end = int(w * 0.70)

    hero_region = image[y_start:y_end, x_start:x_end]
    cards = _extract_cards_from_region(hero_region)

    if len(cards) == 2:
        return cards

    # Fallback: wider search in case layout shifts
    y_start2 = int(h * 0.50)
    x_start2 = int(w * 0.20)
    x_end2 = int(w * 0.80)
    hero_region2 = image[y_start2:y_end, x_start2:x_end2]
    cards = _extract_cards_from_region(hero_region2)

    return cards[:2] if len(cards) >= 2 else cards


def _extract_cards_from_region(region: np.ndarray) -> list[str]:
    """
    Finds and decodes card images within a region.

    Args:
        region: BGR crop to search for cards.

    Returns:
        List of card strings found.
    """
    card_contours = _find_card_contours(region)
    cards = []

    for contour in card_contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        card_crop = region[y:y + ch, x:x + cw]

        rank = detect_rank(card_crop)
        suit = detect_suit(card_crop)

        if rank and suit:
            rank_str = RANK_MAP.get(rank, rank)
            suit_str = SUIT_SYMBOL_MAP.get(suit, suit[0].lower())
            cards.append(f"{rank_str}{suit_str}")

    return cards


def _find_card_contours(region: np.ndarray) -> list:
    """
    Finds rectangular contours that match card dimensions.

    Args:
        region: BGR crop to search.

    Returns:
        List of contours sorted left-to-right.
    """
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    # Cards are typically white/light on a darker background
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rh, rw = region.shape[:2]
    min_area = (rw * rh) * 0.02
    max_area = (rw * rh) * 0.45

    card_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = cw / ch if ch > 0 else 0

        # Playing cards have aspect ratio roughly 0.6–0.8
        if 0.45 < aspect < 0.95:
            card_contours.append(cnt)

    # Sort left-to-right
    card_contours.sort(key=lambda c: cv2.boundingRect(c)[0])
    return card_contours
