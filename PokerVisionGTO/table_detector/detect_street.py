"""
Detects the current street (Preflop/Flop/Turn/River) by counting board cards.
"""

import cv2
import numpy as np
from card_detector.detect_board import detect_board


def detect_street(image: np.ndarray) -> str:
    """
    Determines the current street based on the number of community cards visible.

    Args:
        image: BGR numpy array of the full screenshot.

    Returns:
        Street string: "Preflop", "Flop", "Turn", or "River".
    """
    board = detect_board(image)
    num_cards = len(board)

    if num_cards == 0:
        return "Preflop"
    elif num_cards == 3:
        return "Flop"
    elif num_cards == 4:
        return "Turn"
    elif num_cards >= 5:
        return "River"
    else:
        # 1 or 2 cards visible: likely OCR partial detection, treat as flop
        return "Flop"
