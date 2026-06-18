"""
Detects the number of active players at the table.
"""

import cv2
import numpy as np

try:
    import easyocr
    _reader = easyocr.Reader(["en", "pt"], gpu=False, verbose=False)
    EASYOCR_AVAILABLE = True
except Exception:
    EASYOCR_AVAILABLE = False


def detect_num_players(image: np.ndarray) -> int:
    """
    Estimates the number of active players by counting visible player name labels.

    Args:
        image: BGR numpy array of the full screenshot.

    Returns:
        Number of players (2-9). Defaults to 9 if uncertain.
    """
    h, w = image.shape[:2]

    # Player name labels appear as dark rounded rectangles around the table
    # Count them using contour detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Dark rounded rectangles (player name plates) on medium background
    _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = (w * h) * 0.005
    max_area = (w * h) * 0.06

    player_boxes = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = cw / ch if ch > 0 else 0

        # Player labels are wider than tall
        if 1.5 < aspect < 5.0:
            player_boxes += 1

    if player_boxes < 2:
        return 9

    return min(player_boxes, 9)
