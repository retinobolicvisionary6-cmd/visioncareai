"""
enhancement.py — Modular image enhancement pipeline for BORDERLINE fundus images.

Enhancement is ONLY applied to borderline images and is NOT guaranteed to
improve quality.  The re-assessment step determines the final outcome.

Pipeline:
    CLAHE on L-channel (LAB colour space)  →  illumination normalisation
    →  mild non-local-means denoising  →  output BGR image

All parameters live in config.py.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import config


# ---------------------------------------------------------------------------
# Internal steps
# ---------------------------------------------------------------------------

def _apply_clahe(bgr: np.ndarray) -> np.ndarray:
    """Apply CLAHE to the Luminance channel in LAB colour space."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT,
        tileGridSize=config.CLAHE_TILE_GRID,
    )
    l_eq = clahe.apply(l_ch)

    lab_eq = cv2.merge([l_eq, a_ch, b_ch])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def _illumination_normalise(bgr: np.ndarray) -> np.ndarray:
    """Mild gamma correction to bring mean brightness closer to a target.

    Uses gamma correction: out = in^(1/gamma).
    If image is too dark, gamma > 1.0 brightens it.
    If too bright, gamma < 1.0 darkens it.
    Gamma is capped to a mild range to avoid over-correction.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mean_b = float(np.mean(gray))
    target = 128.0  # neutral mid-tone target

    if mean_b < 5:
        return bgr  # avoid division by zero on black images

    # gamma such that target = mean^(1/gamma) → gamma = log(mean)/log(target)
    gamma = np.log(mean_b) / np.log(target) if target > 0 and mean_b > 0 else 1.0
    # Clamp to mild range.  For very dark images push gamma higher.
    gamma = float(np.clip(gamma, 0.4, 3.0))

    lut = np.array(
        [int(np.clip(((i / 255.0) ** (1.0 / gamma)) * 255.0, 0, 255)) for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(bgr, lut)


def _denoise(bgr: np.ndarray) -> np.ndarray:
    """Apply mild non-local-means denoising (colour).

    Skipped when mean brightness is very low (<30) to prevent NLM from
    introducing artefacts on near-black images.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if float(np.mean(gray)) < 30.0:
        return bgr
    return cv2.fastNlMeansDenoisingColored(
        bgr,
        None,
        h=config.DENOISE_H,
        hColor=config.DENOISE_H,
        templateWindowSize=7,
        searchWindowSize=21,
    )


def _unsharp_mask(bgr: np.ndarray, strength: float = 1.5) -> np.ndarray:
    """Apply unsharp masking to sharpen blurry fundus images.

    Unsharp mask = original + strength * (original - blurred)
    This boosts edge contrast, improving sharpness scores and retinal
    vessel visibility after denoising smoothed the image.

    strength: 1.0 = subtle, 1.5 = moderate (default), 2.0 = strong
    """
    blurred = cv2.GaussianBlur(bgr, (0, 0), sigmaX=2.0)
    sharpened = cv2.addWeighted(bgr, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enhance(bgr: np.ndarray) -> np.ndarray:
    """Run the full enhancement pipeline on *bgr* (BGR uint8 image).

    Steps applied in order:
        1. CLAHE on luminance channel       — improves local contrast
        2. Illumination normalisation        — fixes brightness
        3. Mild denoising                    — reduces noise
        4. Unsharp mask sharpening           — restores edge sharpness lost during denoise

    Returns a new BGR uint8 array.  Original is not modified.
    """
    out = bgr.copy()
    out = _apply_clahe(out)
    out = _illumination_normalise(out)
    out = _denoise(out)
    out = _unsharp_mask(out)   # sharpen after denoise to recover edge detail
    return out
