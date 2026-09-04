"""
field_of_view.py — Field-of-View (FOV) assessment for fundus images.

Estimates whether enough retinal area is visible by analysing:
  1. Detected fundus circle vs. image area ratio.
  2. Fill ratio of the fundus disk (non-black pixels inside the circle).
  3. Fallback: raw non-black pixel fraction when no circle was detected.

Prototype note: this does NOT perform full anatomical segmentation.
All thresholds live in config.py.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import config


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def assess_field_of_view(
    gray: np.ndarray,
    mask: np.ndarray,
    fundus_circle,          # (cx, cy, radius) or None
) -> dict:
    """Assess field of view coverage of the fundus image.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale resized image (uint8).
    mask : np.ndarray
        Uint8 fundus ROI mask.
    fundus_circle : tuple or None
        (cx, cy, radius) from Hough circle detection, or None if fallback
        ellipse was used.

    Returns
    -------
    dict with keys:
        field_of_view_score   — float in [0.0, 1.0]
        fundus_area_ratio     — detected fundus area / total image area
        disk_fill_ratio       — non-black / total pixels inside detected circle
        circle_detected       — bool
        is_hard_fail          — bool
        detail                — human-readable description
    """
    h, w = gray.shape
    total_pixels = h * w

    # ---- circle-based assessment ----
    if fundus_circle is not None:
        cx, cy, r = fundus_circle
        circle_area = np.pi * r * r

        # Fraction of total image covered by circle.
        fundus_area_ratio = float(circle_area / total_pixels)

        # Fill ratio: non-dark pixels inside the circle.
        inside_pixels = gray[mask > 0]
        if inside_pixels.size == 0:
            disk_fill_ratio = 0.0
        else:
            disk_fill_ratio = float(
                np.mean(inside_pixels > config.BACKGROUND_DARK_THRESHOLD)
            )

        circle_detected = True

        # Score based on disk fill (primary) and area ratio (secondary).
        fill_score = float(np.clip(
            (disk_fill_ratio - config.FOV_DISK_FILL_UNGRADABLE) /
            (1.0 - config.FOV_DISK_FILL_UNGRADABLE),
            0.0, 1.0
        ))
        # Reward larger detected circle (up to 50 % of image is excellent).
        area_score = float(np.clip(fundus_area_ratio / 0.50, 0.0, 1.0))

        fov_score = float(np.clip(0.70 * fill_score + 0.30 * area_score, 0.0, 1.0))

        is_hard_fail = (
            disk_fill_ratio < config.FOV_DISK_FILL_UNGRADABLE
            or fundus_area_ratio < config.FOV_MIN_FUNDUS_RATIO
        )

    else:
        # Fallback: no circle found → use whole-image non-black fraction.
        circle_area = float(np.sum(mask > 0))
        fundus_area_ratio = circle_area / total_pixels

        nonblack_in_mask = float(
            np.sum(gray[mask > 0] > config.BACKGROUND_DARK_THRESHOLD)
        )
        disk_fill_ratio = (nonblack_in_mask / circle_area) if circle_area > 0 else 0.0

        circle_detected = False

        fov_score = float(np.clip(
            (disk_fill_ratio - config.FOV_NONBLACK_RATIO_UNGRADABLE) /
            (1.0 - config.FOV_NONBLACK_RATIO_UNGRADABLE),
            0.0, 1.0
        ))

        is_hard_fail = disk_fill_ratio < config.FOV_NONBLACK_RATIO_UNGRADABLE

    # ---- human-readable detail ----
    if is_hard_fail:
        detail = (
            f"Insufficient fundus area visible "
            f"(fill={disk_fill_ratio:.1%}, area_ratio={fundus_area_ratio:.1%}). "
            "Image is likely heavily cropped or mostly background."
        )
    elif fov_score < 0.65:
        detail = (
            f"Limited fundus field of view "
            f"(fill={disk_fill_ratio:.1%}, area_ratio={fundus_area_ratio:.1%}). "
            "Some peripheral retinal area may be cut off."
        )
    else:
        detail = (
            f"Adequate field of view "
            f"(fill={disk_fill_ratio:.1%}, area_ratio={fundus_area_ratio:.1%})."
        )

    return {
        "field_of_view_score": round(fov_score, 4),
        "fundus_area_ratio": round(fundus_area_ratio, 4),
        "disk_fill_ratio": round(disk_fill_ratio, 4),
        "circle_detected": circle_detected,
        "is_hard_fail": is_hard_fail,
        "detail": detail,
    }
