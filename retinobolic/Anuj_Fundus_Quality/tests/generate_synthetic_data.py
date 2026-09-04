"""
generate_synthetic_data.py — Creates controlled synthetic test images from
a valid fundus sample for prototype testing.

ALL images produced here are synthetic degradations.
They are NOT clinically validated test cases.

Usage:
    python tests/generate_synthetic_data.py
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

import cv2
import numpy as np

# Ensure project root is on the path.
sys.path.insert(0, str(Path(__file__).parent.parent))


OUTPUT_DIR = Path("sample_data")
OUTPUT_DIR.mkdir(exist_ok=True)


def _make_base_fundus(size: int = 400) -> np.ndarray:
    """Create a synthetic circular fundus-like BGR image."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cx, cy, r = size // 2, size // 2, int(size * 0.45)

    # Dark background.
    img[:] = (0, 0, 0)

    # Orange-red fundus circle.
    cv2.circle(img, (cx, cy), r, (30, 80, 160), -1)

    # Simulate veins (dark lines).
    rng = np.random.default_rng(42)
    for _ in range(8):
        angle = rng.uniform(0, 2 * np.pi)
        dx, dy = int(np.cos(angle) * r * 0.8), int(np.sin(angle) * r * 0.8)
        cv2.line(img, (cx, cy), (cx + dx, cy + dy), (10, 40, 80), 2)

    # Simulate optic disc (bright spot).
    odx, ody = cx + int(r * 0.2), cy
    cv2.circle(img, (odx, ody), int(r * 0.12), (150, 190, 220), -1)

    # Add mild texture.
    noise = rng.integers(-15, 15, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


def save(img: np.ndarray, name: str) -> Path:
    p = OUTPUT_DIR / name
    cv2.imwrite(str(p), img)
    print(f"  Saved: {p}")
    return p


def generate_all() -> None:
    print("Generating synthetic prototype test images …")
    base = _make_base_fundus(400)

    # 1. Good fundus
    save(base, "good_fundus.jpg")

    # 2. Blurred (Gaussian blur — simulates out-of-focus capture)
    blurred = cv2.GaussianBlur(base, (31, 31), 15)
    save(blurred, "blurred_fundus.jpg")

    # 3. Very blurred (critically ungradable)
    very_blurred = cv2.GaussianBlur(base, (61, 61), 30)
    save(very_blurred, "very_blurred_fundus.jpg")

    # 4. Dark (underexposed)
    dark = (base * 0.25).astype(np.uint8)
    save(dark, "dark_fundus.jpg")

    # 5. Very dark (critically dark)
    # 0.12 keeps a few non-black pixels so load validation passes,
    # but mean brightness (~15) still triggers the critical-dark gate.
    very_dark = (base * 0.12).astype(np.uint8)
    save(very_dark, "very_dark_fundus.jpg")

    # 6. Overexposed (bright)
    bright = np.clip(base.astype(np.int32) + 120, 0, 255).astype(np.uint8)
    save(bright, "overexposed_fundus.jpg")

    # 7. Very overexposed (critically bright)
    very_bright = np.clip(base.astype(np.int32) + 200, 0, 255).astype(np.uint8)
    save(very_bright, "very_overexposed_fundus.jpg")

    # 8. Reduced FOV (heavily cropped — mostly black background)
    small = _make_base_fundus(100)
    fov_limited = np.zeros((400, 400, 3), dtype=np.uint8)
    fov_limited[150:250, 150:250] = small
    save(fov_limited, "poor_fov_fundus.jpg")

    # 9. Noisy
    rng = np.random.default_rng(7)
    noise_arr = rng.integers(-60, 60, base.shape, dtype=np.int16)
    noisy = np.clip(base.astype(np.int16) + noise_arr, 0, 255).astype(np.uint8)
    save(noisy, "noisy_fundus.jpg")

    # 10. Glare simulation
    glare = base.copy()
    cv2.circle(glare, (220, 180), 40, (255, 255, 255), -1)
    save(glare, "glare_fundus.jpg")

    # 11. Borderline (mild blur + mild dark)
    borderline = cv2.GaussianBlur(base, (11, 11), 4)
    borderline = (borderline * 0.65).astype(np.uint8)
    save(borderline, "borderline_fundus.jpg")

    print(f"\nGenerated {11} synthetic test images in '{OUTPUT_DIR}/'")
    print("NOTE: These are synthetic prototype test cases — NOT clinically validated.")


if __name__ == "__main__":
    generate_all()
