"""
retinal_visibility.py — Prototype-level retinal content visibility assessment.

Uses interpretable proxy metrics (no deep learning):
  - Edge density inside the fundus ROI as a vessel / structure visibility proxy.
    Denser edges → more retinal structures visible.

This is a first-version heuristic, NOT a clinical anatomical analysis.
All thresholds live in config.py.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import config


def assess_retinal_visibility(gray: np.ndarray, mask: np.ndarray) -> dict:
    """Estimate retinal content visibility using edge density.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale resized image (uint8).
    mask : np.ndarray
        Uint8 fundus ROI mask.

    Returns
    -------
    dict with keys:
        retinal_visibility_score — float in [0.0, 1.0]
        edge_density             — fraction of edge pixels inside ROI
        detail                   — human-readable description
    """
    # CLAHE-equalised version improves vessel contrast for edge detection.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Apply mask: zero out background before Canny.
    enhanced_roi = cv2.bitwise_and(enhanced, enhanced, mask=mask)

    # Adaptive Canny thresholds: sigma=0.33 heuristic.
    v = float(np.median(enhanced_roi[mask > 0])) if np.any(mask > 0) else 128.0
    lo = max(0.0, (1.0 - 0.33) * v)
    hi = min(255.0, (1.0 + 0.33) * v)

    edges = cv2.Canny(enhanced_roi, lo, hi)

    # Edge density = fraction of ROI pixels that are edges.
    roi_area = float(np.sum(mask > 0))
    if roi_area == 0:
        edge_density = 0.0
    else:
        edge_density = float(np.sum(edges[mask > 0] > 0)) / roi_area

    # Normalise to [0, 1].
    # Real good fundus images have edge density ~0.18-0.25.
    # Ceiling raised from 0.15 to 0.25 to reflect real image data.
    RETVIS_GOOD_CEILING = 0.25
    score = float(np.clip(
        (edge_density - config.RETVIS_EDGE_DENSITY_UNGRADABLE) /
        (RETVIS_GOOD_CEILING - config.RETVIS_EDGE_DENSITY_UNGRADABLE),
        0.0, 1.0
    ))

    # Human detail.
    if edge_density < config.RETVIS_EDGE_DENSITY_UNGRADABLE:
        detail = (
            f"Very few retinal structures visible "
            f"(edge density={edge_density:.3f}). "
            "Image may be too blurred or obscured."
        )
    elif edge_density < config.RETVIS_EDGE_DENSITY_BORDERLINE:
        detail = (
            f"Limited retinal structure visibility "
            f"(edge density={edge_density:.3f})."
        )
    else:
        detail = (
            f"Retinal structures appear adequately visible "
            f"(edge density={edge_density:.3f})."
        )

    return {
        "retinal_visibility_score": round(score, 4),
        "edge_density": round(edge_density, 6),
        "detail": detail,
    }
