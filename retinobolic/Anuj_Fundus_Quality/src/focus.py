"""
focus.py — Focus / sharpness assessment for fundus images.

Uses two interpretable classical metrics:
  1. Laplacian variance  — sensitive to blur
  2. Tenengrad (Sobel)   — edge / gradient strength

Both are computed on the fundus ROI (masked grayscale image) to avoid the
dark background skewing the result.

All thresholds live in config.py.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import config


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _laplacian_variance(gray_roi: np.ndarray, mask: np.ndarray) -> float:
    """Laplacian variance computed only over the fundus ROI pixels."""
    lap = cv2.Laplacian(gray_roi, cv2.CV_64F)
    # Restrict to pixels inside the fundus mask.
    roi_pixels = lap[mask > 0]
    if roi_pixels.size == 0:
        return 0.0
    return float(np.var(roi_pixels))


def _tenengrad(gray_roi: np.ndarray, mask: np.ndarray) -> float:
    """Mean Sobel gradient magnitude inside the fundus ROI."""
    gx = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx**2 + gy**2)
    roi_pixels = magnitude[mask > 0]
    if roi_pixels.size == 0:
        return 0.0
    return float(np.mean(roi_pixels))


def _metric_to_score(value: float, low: float, high: float) -> float:
    """Linearly map *value* from [low, high] → [0.0, 1.0], clamped."""
    if high <= low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def assess_focus(gray: np.ndarray, mask: np.ndarray) -> dict:
    """Assess the focus / sharpness of a fundus image.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale version of the *resized* image.
    mask : np.ndarray
        Uint8 fundus ROI mask (255 inside, 0 outside).

    Returns
    -------
    dict with keys:
        focus_score       — float in [0.0, 1.0]
        laplacian_var     — raw Laplacian variance value
        tenengrad_mean    — raw Tenengrad mean value
        is_hard_fail      — bool: True if focus is critically insufficient
        detail            — human-readable description of what was found
    """
    lap_var = _laplacian_variance(gray, mask)
    ten_mean = _tenengrad(gray, mask)

    # Normalise each metric to [0, 1].
    lap_score = _metric_to_score(
        lap_var,
        config.FOCUS_LAP_VARIANCE_UNGRADABLE,
        config.FOCUS_LAP_VARIANCE_GOOD,
    )
    grad_score = _metric_to_score(
        ten_mean,
        config.FOCUS_GRAD_UNGRADABLE,
        config.FOCUS_GRAD_GOOD,
    )

    # Weighted combination.
    focus_score = (
        config.FOCUS_LAP_WEIGHT * lap_score
        + config.FOCUS_GRAD_WEIGHT * grad_score
    )
    focus_score = float(np.clip(focus_score, 0.0, 1.0))

    # Hard fail: EITHER metric below the ungradable threshold.
    # Primary discriminator: Tenengrad (3.5 for very blurred vs 6.1 for borderline).
    # Secondary: Laplacian variance (0.92 very blurred vs 1.49 borderline).
    is_hard_fail = (
        lap_var < config.FOCUS_LAP_VARIANCE_UNGRADABLE
        or ten_mean < config.FOCUS_GRAD_UNGRADABLE
    )

    # Human-readable detail.
    if is_hard_fail:
        detail = (
            f"Image is critically blurred "
            f"(Laplacian={lap_var:.1f}, Tenengrad={ten_mean:.2f}). "
            "Retinal structures are not distinguishable."
        )
    elif focus_score < 0.5:
        detail = (
            f"Image sharpness is low "
            f"(Laplacian={lap_var:.1f}, Tenengrad={ten_mean:.2f}). "
            "Fine retinal details may be unclear."
        )
    elif focus_score < 0.75:
        detail = (
            f"Image sharpness is acceptable "
            f"(Laplacian={lap_var:.1f}, Tenengrad={ten_mean:.2f})."
        )
    else:
        detail = (
            f"Image is well focused "
            f"(Laplacian={lap_var:.1f}, Tenengrad={ten_mean:.2f})."
        )

    return {
        "focus_score": round(focus_score, 4),
        "laplacian_var": round(lap_var, 2),
        "tenengrad_mean": round(ten_mean, 4),
        "is_hard_fail": is_hard_fail,
        "detail": detail,
    }
