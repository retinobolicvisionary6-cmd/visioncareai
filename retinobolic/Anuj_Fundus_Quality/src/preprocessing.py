"""
preprocessing.py — Robust image loading, validation, resizing, and fundus
Region-of-Interest (ROI) extraction.

The original image is NEVER modified.  All functions return new NumPy arrays.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np

from . import config


# ---------------------------------------------------------------------------
# Public Types
# ---------------------------------------------------------------------------

class PreprocessError(ValueError):
    """Raised when an image cannot be loaded or is invalid for analysis."""


# ---------------------------------------------------------------------------
# Loading & Validation
# ---------------------------------------------------------------------------

def load_and_validate(image_path: str) -> np.ndarray:
    """Load an image from *image_path* and validate it is suitable for analysis.

    Returns
    -------
    np.ndarray
        BGR image array (uint8).

    Raises
    ------
    PreprocessError
        If the file is missing, has an unsupported extension, is corrupt,
        is too small, or is effectively empty.
    """
    path = Path(image_path)

    # --- existence ---
    if not path.exists():
        raise PreprocessError(f"File not found: '{image_path}'")

    # --- extension ---
    ext = path.suffix.lower()
    if ext not in config.SUPPORTED_EXTENSIONS:
        raise PreprocessError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {config.SUPPORTED_EXTENSIONS}"
        )

    # --- read ---
    img = cv2.imread(str(path))
    if img is None:
        raise PreprocessError(f"Could not read image (corrupt or unreadable): '{image_path}'")

    # --- dimension check ---
    h, w = img.shape[:2]
    if h < config.MIN_IMAGE_DIM or w < config.MIN_IMAGE_DIM:
        raise PreprocessError(
            f"Image too small ({w}×{h}). "
            f"Minimum dimension is {config.MIN_IMAGE_DIM}px."
        )

    # --- non-empty check ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    nonblack = int(np.sum(gray > config.BACKGROUND_DARK_THRESHOLD))
    if nonblack < config.MIN_NONBLACK_PIXELS:
        raise PreprocessError(
            f"Image appears empty: only {nonblack} non-black pixels found."
        )

    return img


# ---------------------------------------------------------------------------
# Resizing
# ---------------------------------------------------------------------------

def resize_for_analysis(img: np.ndarray) -> np.ndarray:
    """Resize *img* so its longest edge equals ``config.RESIZE_LONG_EDGE``.

    Aspect ratio is preserved.  If the image is already smaller, it is
    returned as-is (no upscaling).
    """
    h, w = img.shape[:2]
    long_edge = max(h, w)
    target = config.RESIZE_LONG_EDGE

    if long_edge <= target:
        return img.copy()

    scale = target / long_edge
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Fundus ROI Extraction
# ---------------------------------------------------------------------------

def extract_fundus_roi(
    img: np.ndarray,
) -> Tuple[np.ndarray, Optional[Tuple[int, int, int]]]:
    """Detect the circular fundus region using Hough circles.

    Returns
    -------
    mask : np.ndarray (uint8, same H×W as *img*)
        255 inside the fundus circle, 0 outside.
        If no circle is detected, a fallback ellipse covering the central
        60 % of the frame is used.
    circle : (cx, cy, radius) or None
        Detected circle parameters.  None if fallback was used.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Blur to suppress noise before circle detection.
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    min_r = int(min(h, w) * 0.25)
    max_r = int(min(h, w) * 0.65)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=int(min(h, w) * 0.5),
        param1=50,
        param2=30,
        minRadius=min_r,
        maxRadius=max_r,
    )

    mask = np.zeros((h, w), dtype=np.uint8)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        cx, cy, r = circles[0, 0]
        cv2.circle(mask, (cx, cy), r, 255, -1)
        return mask, (int(cx), int(cy), int(r))

    # Fallback: central ellipse covering ~60 % of the smaller axis.
    cx, cy = w // 2, h // 2
    rx, ry = int(w * 0.42), int(h * 0.42)
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
    return mask, None


def apply_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero out pixels outside *mask* (logical AND per channel)."""
    masked = img.copy()
    for c in range(img.shape[2]):
        masked[:, :, c] = cv2.bitwise_and(img[:, :, c], img[:, :, c], mask=mask)
    return masked


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def preprocess(image_path: str) -> dict:
    """Full preprocessing pipeline.

    Returns a dict with keys:
        original      — original BGR image (unmodified)
        resized       — resized BGR image
        gray          — grayscale of resized
        fundus_mask   — uint8 mask of fundus ROI
        fundus_circle — (cx, cy, r) or None
        roi_img       — resized image masked to fundus ROI
    """
    original = load_and_validate(image_path)
    resized = resize_for_analysis(original)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    fundus_mask, fundus_circle = extract_fundus_roi(resized)
    roi_img = apply_mask(resized, fundus_mask)

    return {
        "original": original,
        "resized": resized,
        "gray": gray,
        "fundus_mask": fundus_mask,
        "fundus_circle": fundus_circle,
        "roi_img": roi_img,
    }
