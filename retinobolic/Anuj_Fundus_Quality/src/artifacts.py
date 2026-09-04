"""
artifacts.py — Basic detection of major quality-degrading artifacts in
fundus images.

Checks:
  1. Glare / bright reflections — fraction of fully-saturated pixels
  2. Noise level — estimated via high-frequency energy of the fundus ROI

Prototype version: focuses on the most common artifacts in low-resource
fundus cameras.  Over-engineering is avoided per project guidelines.

All thresholds live in config.py.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import config


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _glare_ratio(bgr: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of ROI pixels where ALL channels are saturated (≥250)."""
    b, g, r = cv2.split(bgr)
    saturated = (b >= 250) & (g >= 250) & (r >= 250)
    roi = saturated[mask > 0]
    return float(np.mean(roi)) if roi.size > 0 else 0.0


def _noise_level(gray: np.ndarray, mask: np.ndarray) -> float:
    """High-frequency energy ratio as a noise proxy.

    Apply a 3×3 Laplacian and measure the RMS of the result relative to the
    image mean brightness.  High ratio → noisy.
    """
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    roi_pixels = np.abs(lap[mask > 0])
    img_mean = float(np.mean(gray[mask > 0])) if np.any(mask > 0) else 128.0
    if roi_pixels.size == 0 or img_mean < 1:
        return 0.0
    rms = float(np.sqrt(np.mean(roi_pixels**2)))
    # Normalise by mean brightness so dark/bright images are comparable.
    return float(np.clip(rms / (img_mean + 1e-6), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def assess_artifacts(bgr: np.ndarray, gray: np.ndarray, mask: np.ndarray) -> dict:
    """Detect major artifacts in the fundus ROI.

    Parameters
    ----------
    bgr : np.ndarray
        Colour resized image (uint8, BGR).
    gray : np.ndarray
        Grayscale resized image (uint8).
    mask : np.ndarray
        Uint8 fundus ROI mask.

    Returns
    -------
    dict with keys:
        artifact_score   — float in [0.0, 1.0] (higher = fewer artifacts)
        glare_ratio      — fraction of saturated pixels
        noise_level      — normalised noise estimate (0–1)
        artifacts_found  — list[str]
        detail           — human-readable description
    """
    glare = _glare_ratio(bgr, mask)
    noise = _noise_level(gray, mask)

    # Convert to per-artifact sub-scores (1.0 = no artifact).
    glare_score = float(np.clip(
        1.0 - glare / config.ARTIFACT_GLARE_UNGRADABLE,
        0.0, 1.0
    ))
    noise_score = float(np.clip(
        1.0 - noise / config.ARTIFACT_NOISE_UNGRADABLE,
        0.0, 1.0
    ))

    # Composite artifact score.
    artifact_score = float(np.clip(0.60 * glare_score + 0.40 * noise_score, 0.0, 1.0))

    # List detected artifacts.
    artifacts_found: list[str] = []
    if glare > config.ARTIFACT_GLARE_UNGRADABLE:
        artifacts_found.append("severe_glare")
    elif glare > config.ARTIFACT_GLARE_BORDERLINE:
        artifacts_found.append("moderate_glare")
    if noise > config.ARTIFACT_NOISE_UNGRADABLE:
        artifacts_found.append("severe_noise")
    elif noise > config.ARTIFACT_NOISE_BORDERLINE:
        artifacts_found.append("moderate_noise")

    if not artifacts_found:
        detail = f"No significant artifacts detected (glare={glare:.3f}, noise={noise:.3f})."
    else:
        af_str = ", ".join(a.replace("_", " ") for a in artifacts_found)
        detail = (
            f"Artifacts detected: {af_str}. "
            f"(glare={glare:.3f}, noise={noise:.3f})"
        )

    return {
        "artifact_score": round(artifact_score, 4),
        "glare_ratio": round(glare, 4),
        "noise_level": round(noise, 4),
        "artifacts_found": artifacts_found,
        "detail": detail,
    }
