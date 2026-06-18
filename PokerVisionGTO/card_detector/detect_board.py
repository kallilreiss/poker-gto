"""
Detects community cards (board) from the center of the poker table.
"""

import cv2
import numpy as np
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


def detect_board(image: np.ndarray) -> list[str]:
    """
    Detects the community cards displayed in the center of the table.

    Args:
        image: BGR numpy array of the full screenshot.

    Returns:
        List of card strings, e.g. ["4h", "Js", "Kd"]. Empty if no board cards.
    """
    h, w = image.shape[:2]

    # Board area: center of table, slightly above mid-height
    y_start = int(h * 0.28)
    y_end = int(h * 0.58)
    x_start = int(w * 0.25)
    x_end = int(w * 0.75)

    board_region = image[y_start:y_end, x_start:x_end]
    cards = _extract_board_cards(board_region)

    if not cards:
        # Try slightly different region
        y_start2 = int(h * 0.25)
        y_end2 = int(h * 0.60)
        board_region2 = image[y_start2:y_end2, x_start:x_end]
        cards = _extract_board_cards(board_region2)

    return cards[:5]  # Max 5 community cards


def _extract_board_cards(region: np.ndarray) -> list[str]:
    """
    Finds and decodes up to 5 card images in the board region.

    Args:
        region: BGR crop of the board area.

    Returns:
        List of card strings.
    """
    card_contours = _find_board_card_contours(region)
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


def _find_board_card_contours(region: np.ndarray) -> list:
    """
    Finds card-shaped white rectangles in the board region.

    Args:
        region: BGR crop to search.

    Returns:
        Sorted list of card contours (left-to-right).
    """
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    # Use adaptive threshold to handle different lighting
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=-5
    )

    # Also try simple threshold for white cards
    _, thresh2 = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)

    # Combine both
    combined = cv2.bitwise_or(thresh, thresh2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rh, rw = region.shape[:2]
    min_area = (rw * rh) * 0.01
    max_area = (rw * rh) * 0.25

    card_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = cw / ch if ch > 0 else 0

        if 0.45 < aspect < 0.95:
            card_contours.append(cnt)

    # Sort left-to-right, take up to 5
    card_contours.sort(key=lambda c: cv2.boundingRect(c)[0])
    return card_contours[:5]
