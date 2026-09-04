"""
illumination.py — Illumination quality assessment for fundus images.

Checks performed (all on fundus ROI pixels only):
  1. Mean brightness       — too dark / too bright
  2. Clipping ratio        — overexposed pixels
  3. Contrast (std-dev)    — flat / low-contrast images
  4. Illumination uniformity — uneven lighting across quadrants

All thresholds live in config.py.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import config


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _roi_pixels(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return 1-D array of grayscale pixel values inside the fundus ROI."""
    return gray[mask > 0].astype(np.float64)


def _quadrant_means(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compute mean brightness of four equal quadrants (ROI pixels only)."""
    h, w = gray.shape
    mh, mw = h // 2, w // 2
    quadrants = [
        (gray[:mh, :mw], mask[:mh, :mw]),
        (gray[:mh, mw:], mask[:mh, mw:]),
        (gray[mh:, :mw], mask[mh:, :mw]),
        (gray[mh:, mw:], mask[mh:, mw:]),
    ]
    means = []
    for q_gray, q_mask in quadrants:
        pixels = q_gray[q_mask > 0]
        means.append(float(np.mean(pixels)) if pixels.size > 0 else 0.0)
    return np.array(means)


def _score_from_range(value: float, good_low: float, good_high: float,
                      bad_low: float, bad_high: float) -> float:
    """Score rises from 0→1 between bad_low→good_low, stays 1 between good
    bounds, and falls from 1→0 between good_high→bad_high."""
    if value < bad_low or value > bad_high:
        return 0.0
    if value < good_low:
        span = good_low - bad_low
        return float((value - bad_low) / span) if span > 0 else 0.0
    if value > good_high:
        span = bad_high - good_high
        return float((bad_high - value) / span) if span > 0 else 0.0
    return 1.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def assess_illumination(gray: np.ndarray, mask: np.ndarray) -> dict:
    """Assess illumination quality of the fundus ROI.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale resized image (uint8).
    mask : np.ndarray
        Uint8 fundus ROI mask.

    Returns
    -------
    dict with keys:
        illumination_score  — float in [0.0, 1.0]
        mean_brightness     — float (0–255)
        std_brightness      — float
        clip_ratio          — fraction of pixels at max intensity
        uniformity_std      — std of quadrant means (higher = more uneven)
        is_hard_fail        — bool
        conditions          — list[str] of detected issues
        detail              — human-readable summary
    """
    pixels = _roi_pixels(gray, mask)

    if pixels.size == 0:
        return {
            "illumination_score": 0.0,
            "mean_brightness": 0.0,
            "std_brightness": 0.0,
            "clip_ratio": 0.0,
            "uniformity_std": 0.0,
            "is_hard_fail": True,
            "conditions": ["empty_roi"],
            "detail": "No fundus ROI pixels found.",
        }

    mean_b = float(np.mean(pixels))
    std_b = float(np.std(pixels))
    clip_ratio = float(np.mean(pixels >= 254))

    quad_means = _quadrant_means(gray, mask)
    uniformity_std = float(np.std(quad_means))

    # ---------- individual sub-scores ----------
    # Brightness: ideal range ~60–180.
    brightness_score = _score_from_range(
        mean_b,
        good_low=60.0, good_high=180.0,
        bad_low=config.ILLUM_MEAN_TOO_DARK,
        bad_high=config.ILLUM_MEAN_TOO_BRIGHT,
    )

    # Contrast (std-dev).
    contrast_score = float(np.clip(
        (std_b - config.ILLUM_STD_MIN_UNGRADABLE) /
        (60.0 - config.ILLUM_STD_MIN_UNGRADABLE),
        0.0, 1.0
    ))

    # Clipping — lower clip_ratio is better.
    clip_score = float(np.clip(
        1.0 - clip_ratio / config.ILLUM_CLIP_RATIO_UNGRADABLE,
        0.0, 1.0
    ))

    # Uniformity — lower uniformity_std is better.
    uniformity_score = float(np.clip(
        1.0 - uniformity_std / config.ILLUM_UNIFORMITY_UNGRADABLE,
        0.0, 1.0
    ))

    # Weighted composite.
    illumination_score = float(np.clip(
        0.40 * brightness_score
        + 0.25 * contrast_score
        + 0.20 * clip_score
        + 0.15 * uniformity_score,
        0.0, 1.0
    ))

    # ---------- hard fail ----------
    is_hard_fail = (
        mean_b < config.ILLUM_MEAN_TOO_DARK
        or mean_b > config.ILLUM_MEAN_TOO_BRIGHT
        or clip_ratio > config.ILLUM_CLIP_RATIO_UNGRADABLE
        or std_b < config.ILLUM_STD_MIN_UNGRADABLE
    )

    # ---------- conditions detected ----------
    conditions: list[str] = []
    if mean_b < config.ILLUM_MEAN_TOO_DARK:
        conditions.append("critically_dark")
    elif mean_b < config.ILLUM_MEAN_DARK:
        conditions.append("underexposed")
    if mean_b > config.ILLUM_MEAN_TOO_BRIGHT:
        conditions.append("critically_bright")
    elif mean_b > config.ILLUM_MEAN_BRIGHT:
        conditions.append("overexposed")
    if clip_ratio > config.ILLUM_CLIP_RATIO_UNGRADABLE:
        conditions.append("severe_clipping")
    elif clip_ratio > config.ILLUM_CLIP_RATIO_BORDERLINE:
        conditions.append("moderate_clipping")
    if std_b < config.ILLUM_STD_MIN_UNGRADABLE:
        conditions.append("no_contrast")
    elif std_b < config.ILLUM_STD_MIN_BORDERLINE:
        conditions.append("low_contrast")
    if uniformity_std > config.ILLUM_UNIFORMITY_UNGRADABLE:
        conditions.append("severely_uneven_illumination")
    elif uniformity_std > config.ILLUM_UNIFORMITY_BORDERLINE:
        conditions.append("uneven_illumination")

    # ---------- human detail ----------
    if conditions:
        cond_str = ", ".join(c.replace("_", " ") for c in conditions)
        detail = f"Illumination issues detected: {cond_str}. "
        detail += f"(mean={mean_b:.0f}, std={std_b:.1f}, clip={clip_ratio:.2%})"
    else:
        detail = (
            f"Illumination is good "
            f"(mean={mean_b:.0f}, std={std_b:.1f}, clip={clip_ratio:.2%})."
        )

    return {
        "illumination_score": round(illumination_score, 4),
        "mean_brightness": round(mean_b, 2),
        "std_brightness": round(std_b, 2),
        "clip_ratio": round(clip_ratio, 4),
        "uniformity_std": round(uniformity_std, 2),
        "is_hard_fail": is_hard_fail,
        "conditions": conditions,
        "detail": detail,
    }
