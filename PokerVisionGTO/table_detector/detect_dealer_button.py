"""
Detects the dealer button (D) position on the poker table.
The dealer button is a small circular white/light chip with "D" or dealer marker.
"""

import cv2
import numpy as np


def detect_dealer_button(image: np.ndarray) -> tuple[float, float] | None:
    """
    Finds the dealer button position in the screenshot.

    Args:
        image: BGR numpy array of the full screenshot.

    Returns:
        (x, y) pixel coordinates of the dealer button, or None if not found.
    """
    h, w = image.shape[:2]

    # Focus on the table area (exclude outer UI elements)
    table_region = image[int(h * 0.05):int(h * 0.90), int(w * 0.05):int(w * 0.92)]
    offset_x = int(w * 0.05)
    offset_y = int(h * 0.05)

    position = _find_button_by_circle(table_region)
    if position:
        return (position[0] + offset_x, position[1] + offset_y)

    position = _find_button_by_color(table_region)
    if position:
        return (position[0] + offset_x, position[1] + offset_y)

    return None


def _find_button_by_circle(region: np.ndarray) -> tuple[float, float] | None:
    """
    Uses Hough Circle Transform to find small circular dealer buttons.

    Args:
        region: BGR crop of the table area.

    Returns:
        (x, y) coordinates relative to region, or None.
    """
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    rh, rw = region.shape[:2]
    min_radius = max(8, int(min(rw, rh) * 0.012))
    max_radius = max(25, int(min(rw, rh) * 0.035))

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=30,
        param1=50,
        param2=30,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        return None

    circles = np.round(circles[0, :]).astype(int)

    # The dealer button is white/light colored
    for (x, y, r) in circles:
        if y - r < 0 or y + r >= region.shape[0] or x - r < 0 or x + r >= region.shape[1]:
            continue

        circle_patch = region[y - r:y + r, x - r:x + r]
        mean_brightness = np.mean(cv2.cvtColor(circle_patch, cv2.COLOR_BGR2GRAY))

        if mean_brightness > 160:
            return (float(x), float(y))

    return None


def _find_button_by_color(region: np.ndarray) -> tuple[float, float] | None:
    """
    Fallback: finds the dealer button by looking for small white circular blobs.

    Args:
        region: BGR crop of the table area.

    Returns:
        (x, y) coordinates relative to region, or None.
    """
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rh, rw = region.shape[:2]
    min_area = (rw * rh) * 0.0003
    max_area = (rw * rh) * 0.005

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity > 0.60:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                candidates.append((cx, cy, circularity))

    if not candidates:
        return None

    # Return the most circular candidate
    best = max(candidates, key=lambda c: c[2])
    return (best[0], best[1])
