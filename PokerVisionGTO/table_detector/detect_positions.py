"""
Detects the Hero's position at the table.
Uses the dealer button location and number of players to assign positions.
"""

import cv2
import numpy as np
from table_detector.detect_dealer_button import detect_dealer_button


POSITIONS_6MAX = ["BTN", "SB", "BB", "UTG", "HJ", "CO"]
POSITIONS_9MAX = ["BTN", "SB", "BB", "UTG", "UTG+1", "MP", "MP+1", "HJ", "CO"]

# Fixed seat positions (normalized 0-1) for a 9-seat table
# Hero is always seat index 0 (bottom center)
SEAT_COORDINATES_9MAX = [
    (0.50, 0.82),   # 0 = Hero (bottom center)
    (0.80, 0.68),   # 1 = bottom right
    (0.92, 0.42),   # 2 = right
    (0.82, 0.15),   # 3 = top right
    (0.62, 0.08),   # 4 = top center-right
    (0.38, 0.08),   # 5 = top center-left
    (0.18, 0.15),   # 6 = top left
    (0.08, 0.42),   # 7 = left
    (0.20, 0.68),   # 8 = bottom left
]

SEAT_COORDINATES_6MAX = [
    (0.50, 0.82),   # 0 = Hero
    (0.82, 0.60),   # 1
    (0.82, 0.25),   # 2
    (0.50, 0.10),   # 3
    (0.18, 0.25),   # 4
    (0.18, 0.60),   # 5
]


def detect_position(image: np.ndarray, num_players: int = 9) -> str:
    """
    Detects Hero's table position (e.g. "BTN", "BB", "CO").

    Args:
        image: BGR numpy array of the full screenshot.
        num_players: Number of players at the table (6 or 9).

    Returns:
        Position string.
    """
    h, w = image.shape[:2]

    dealer_pos = detect_dealer_button(image)

    if num_players <= 6:
        seats = SEAT_COORDINATES_6MAX
        positions = POSITIONS_6MAX
    else:
        seats = SEAT_COORDINATES_9MAX
        positions = POSITIONS_9MAX

    if dealer_pos is None:
        return "BB"  # Default fallback

    dealer_seat = _find_nearest_seat(dealer_pos, seats, w, h)
    hero_seat = 0  # Hero always at seat 0

    # Number of seats between dealer and hero (going clockwise)
    num_seats = len(seats)
    offset = (hero_seat - dealer_seat) % num_seats

    # offset=0 means hero IS the dealer (BTN)
    # offset=1 means hero is SB
    # offset=2 means hero is BB
    # etc.
    if offset < len(positions):
        return positions[offset]

    return "UTG"


def _find_nearest_seat(
    dealer_xy: tuple[float, float],
    seats: list[tuple[float, float]],
    img_w: int,
    img_h: int,
) -> int:
    """
    Finds the seat index closest to the detected dealer button position.

    Args:
        dealer_xy: (x, y) pixel coordinates of the dealer button.
        seats: List of normalized (x, y) seat positions.
        img_w: Image width in pixels.
        img_h: Image height in pixels.

    Returns:
        Index of the nearest seat.
    """
    dx, dy = dealer_xy
    min_dist = float("inf")
    nearest = 0

    for i, (sx, sy) in enumerate(seats):
        px = sx * img_w
        py = sy * img_h
        dist = ((dx - px) ** 2 + (dy - py) ** 2) ** 0.5
        if dist < min_dist:
            min_dist = dist
            nearest = i

    return nearest
