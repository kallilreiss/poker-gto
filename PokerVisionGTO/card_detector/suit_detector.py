"""
Detects the suit of a playing card using color analysis.
Red = hearts/diamonds, Black = clubs/spades.
Shape analysis distinguishes within color groups.
"""

import cv2
import numpy as np


# HSV color ranges
RED_LOWER1 = np.array([0, 100, 100])
RED_UPPER1 = np.array([10, 255, 255])
RED_LOWER2 = np.array([160, 100, 100])
RED_UPPER2 = np.array([180, 255, 255])

BLACK_LOWER = np.array([0, 0, 0])
BLACK_UPPER = np.array([180, 80, 80])


def detect_suit(card_image: np.ndarray) -> str | None:
    """
    Detects the suit of a card from its image.

    Args:
        card_image: BGR crop of a single card.

    Returns:
        Suit string: "hearts", "diamonds", "clubs", "spades", or None.
    """
    if card_image is None or card_image.size == 0:
        return None

    suit_crop = _extract_suit_region(card_image)
    color = _detect_color(suit_crop)

    if color == "red":
        return _distinguish_red_suits(suit_crop)
    elif color == "black":
        return _distinguish_black_suits(suit_crop)

    return None


def _extract_suit_region(card_image: np.ndarray) -> np.ndarray:
    """
    Extracts the suit symbol area (below the rank in the top-left corner).

    Args:
        card_image: BGR crop of a card.

    Returns:
        BGR crop of the suit symbol area.
    """
    h, w = card_image.shape[:2]

    # Suit appears below rank in top-left, or in center of card
    # Try center of card first (more reliable for larger crops)
    y_start = int(h * 0.35)
    y_end = int(h * 0.75)
    x_start = int(w * 0.15)
    x_end = int(w * 0.85)

    center = card_image[y_start:y_end, x_start:x_end]
    if center.size > 0:
        return center

    return card_image


def _detect_color(region: np.ndarray) -> str | None:
    """
    Determines if the suit symbol is red or black.

    Args:
        region: BGR crop containing the suit.

    Returns:
        "red", "black", or None.
    """
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

    red_mask1 = cv2.inRange(hsv, RED_LOWER1, RED_UPPER1)
    red_mask2 = cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    black_mask = cv2.inRange(hsv, BLACK_LOWER, BLACK_UPPER)

    red_pixels = np.sum(red_mask > 0)
    black_pixels = np.sum(black_mask > 0)
    total = region.shape[0] * region.shape[1]

    red_ratio = red_pixels / total
    black_ratio = black_pixels / total

    if red_ratio > 0.03:
        return "red"
    if black_ratio > 0.05:
        return "black"

    return None


def _distinguish_red_suits(region: np.ndarray) -> str:
    """
    Distinguishes between hearts and diamonds using shape analysis.
    Hearts are rounded at the top; diamonds are pointed on all sides.

    Args:
        region: BGR crop of the suit area.

    Returns:
        "hearts" or "diamonds".
    """
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, RED_LOWER1, RED_UPPER1)
    red2 = cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
    mask = cv2.bitwise_or(red1, red2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "hearts"

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    if w == 0 or h == 0:
        return "hearts"

    aspect = w / h

    # Diamonds are roughly square (aspect ~1.0); hearts are slightly wider than tall
    # Also check for the concave top of hearts via convexity defects
    hull = cv2.convexHull(largest, returnPoints=False)
    if len(hull) > 3 and len(largest) > 3:
        try:
            defects = cv2.convexityDefects(largest, hull)
            if defects is not None and len(defects) >= 2:
                return "hearts"
        except cv2.error:
            pass

    if aspect > 0.9:
        return "diamonds"

    return "hearts"


def _distinguish_black_suits(region: np.ndarray) -> str:
    """
    Distinguishes between clubs and spades using shape analysis.
    Spades have a pointed top; clubs have three circular lobes.

    Args:
        region: BGR crop of the suit area.

    Returns:
        "clubs" or "spades".
    """
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "spades"

    largest = max(contours, key=cv2.contourArea)

    # Check circularity: clubs have higher circularity due to lobes
    area = cv2.contourArea(largest)
    perimeter = cv2.arcLength(largest, True)
    if perimeter == 0:
        return "spades"

    circularity = 4 * np.pi * area / (perimeter ** 2)

    # Clubs tend to have lower circularity due to the stem and three lobes
    # Spades have a more compact shape with a pointed top
    if circularity < 0.35:
        return "clubs"

    return "spades"
