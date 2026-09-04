"""
Synthetic Sample Generator for OOD Testing
===========================================
Generates controlled synthetic images for:
  - in_distribution/  : simulated fundus-like images (bright disc, blood vessel
                        arcade patterns, macula-like dark spot, radial gradient)
  - out_of_distribution/: blank, noise, checkerboard, color-inverted, extreme-crop
                          variants clearly labelled as SYNTHETIC OOD

IMPORTANT
---------
These images are SYNTHETIC and CONTROLLED.
They are for software testing and threshold calibration ONLY.
They do NOT represent real fundus pathology or clinical OOD validation.

Usage
-----
    python sample_data/generate_samples.py
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# Allow running from project root or sample_data/
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

OUT_ID = ROOT / "sample_data" / "in_distribution"
OUT_OOD = ROOT / "sample_data" / "out_of_distribution"
SIZE = (224, 224)
RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path))
    print(f"  [OK] {path.relative_to(ROOT)}")


def make_base_fundus(rng: np.random.Generator, variation: float = 0.0) -> np.ndarray:
    """
    Create a synthetic fundus-like RGB image array (H, W, 3) uint8.
    Features:
      - Dark reddish circular background (fundus cup shape)
      - Bright optic disc (white-yellow patch, nasal side)
      - Radial blood-vessel-like darkened arcs
      - Macula-like dark circular spot (temporal centre)
    """
    H, W = SIZE
    img = np.zeros((H, W, 3), dtype=np.float32)

    # Background — dark retinal red
    base_r = float(np.clip(0.35 + variation * rng.uniform(-0.08, 0.08), 0.1, 0.7))
    base_g = float(np.clip(0.10 + variation * rng.uniform(-0.04, 0.04), 0.02, 0.3))
    base_b = float(np.clip(0.05 + variation * rng.uniform(-0.02, 0.02), 0.01, 0.2))
    img[:, :, 0] = base_r
    img[:, :, 1] = base_g
    img[:, :, 2] = base_b

    # Circular fundus vignette mask (darken edges)
    cy, cx = H / 2.0, W / 2.0
    ys, xs = np.mgrid[0:H, 0:W]
    r = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2) / (min(cy, cx))
    vignette = np.clip(1.0 - r ** 2 * 0.6, 0.0, 1.0)
    for c in range(3):
        img[:, :, c] *= vignette

    # Optic disc — bright circular patch (nasal, ~left-centre)
    disc_cx = int(W * (0.35 + variation * rng.uniform(-0.03, 0.03)))
    disc_cy = int(H * (0.50 + variation * rng.uniform(-0.03, 0.03)))
    disc_r = int(min(H, W) * (0.09 + variation * rng.uniform(-0.01, 0.02)))
    disc_r = max(disc_r, 8)
    disc_mask = ((ys - disc_cy) ** 2 + (xs - disc_cx) ** 2) <= disc_r ** 2
    disc_intensity = float(np.clip(0.85 + variation * rng.uniform(-0.05, 0.05), 0.6, 1.0))
    img[disc_mask, 0] = disc_intensity
    img[disc_mask, 1] = disc_intensity * 0.9
    img[disc_mask, 2] = disc_intensity * 0.7

    # Blood vessels — radial dark arcs emanating from disc
    angles = np.linspace(0, 2 * np.pi, 80)
    for angle in angles[::4]:
        for radius in np.linspace(disc_r, min(H, W) * 0.45, 30):
            vx = int(disc_cx + radius * np.cos(angle))
            vy = int(disc_cy + radius * np.sin(angle))
            if 0 <= vy < H and 0 <= vx < W:
                thickness = max(1, int(3 - radius / (min(H, W) * 0.45) * 2))
                for dy in range(-thickness, thickness + 1):
                    for dx in range(-thickness, thickness + 1):
                        ny, nx = vy + dy, vx + dx
                        if 0 <= ny < H and 0 <= nx < W:
                            img[ny, nx, 0] *= 0.75
                            img[ny, nx, 1] *= 0.65
                            img[ny, nx, 2] *= 0.65

    # Macula — dark circular spot (temporal side, right of centre)
    mac_cx = int(W * (0.62 + variation * rng.uniform(-0.03, 0.03)))
    mac_cy = int(H * (0.50 + variation * rng.uniform(-0.03, 0.03)))
    mac_r = int(min(H, W) * (0.06 + variation * rng.uniform(-0.01, 0.01)))
    mac_r = max(mac_r, 5)
    mac_dist = np.sqrt((ys - mac_cy) ** 2 + (xs - mac_cx) ** 2) / (mac_r + 1e-8)
    mac_mask = mac_dist <= 1.0
    attenuation = np.clip(0.3 + mac_dist * 0.3, 0.0, 0.8)
    for c in range(3):
        img[:, :, c] = np.where(mac_mask, img[:, :, c] * attenuation, img[:, :, c])

    # Soft blur to simulate optics
    pil = Image.fromarray((np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(radius=1.5)
    )
    result = np.asarray(pil, dtype=np.uint8)
    return result


# ---------------------------------------------------------------------------
# In-distribution images
# ---------------------------------------------------------------------------

def generate_in_distribution() -> None:
    print("\n[IN-DISTRIBUTION] Generating synthetic fundus-like images...")

    for i in range(12):
        arr = make_base_fundus(RNG, variation=1.0)
        save(Image.fromarray(arr), OUT_ID / f"fundus_{i+1:02d}.png")


# ---------------------------------------------------------------------------
# Out-of-distribution images
# ---------------------------------------------------------------------------

def generate_out_of_distribution() -> None:
    print("\n[OUT-OF-DISTRIBUTION — SYNTHETIC] Generating OOD test images...")

    # 1. Blank black
    save(Image.fromarray(np.zeros((*SIZE, 3), dtype=np.uint8)),
         OUT_OOD / "blank_black.png")

    # 2. Blank white
    save(Image.fromarray(np.full((*SIZE, 3), 255, dtype=np.uint8)),
         OUT_OOD / "blank_white.png")

    # 3. Pure random noise
    noise = RNG.integers(0, 256, (*SIZE, 3), dtype=np.uint8)
    save(Image.fromarray(noise), OUT_OOD / "random_noise.png")

    # 4. Checkerboard pattern (natural image domain shift)
    H, W = SIZE
    checker = np.zeros((H, W, 3), dtype=np.uint8)
    block = 16
    for i in range(0, H, block):
        for j in range(0, W, block):
            val = 220 if ((i // block + j // block) % 2 == 0) else 40
            checker[i:i+block, j:j+block] = val
    checker[:, :, 1] = checker[:, :, 0] // 2
    save(Image.fromarray(checker), OUT_OOD / "checkerboard.png")

    # 5. Colour-inverted fundus (extreme colour shift)
    arr = make_base_fundus(RNG, variation=0.0)
    inverted = 255 - arr
    save(Image.fromarray(inverted), OUT_OOD / "color_inverted.png")

    # 6. Extreme crop — tiny centre patch scaled up (severe context loss)
    arr = make_base_fundus(RNG, variation=0.0)
    pil = Image.fromarray(arr)
    crop_region = (60, 60, 160, 160)  # tiny centre patch
    cropped = pil.crop(crop_region).resize(SIZE, Image.NEAREST)
    save(cropped, OUT_OOD / "extreme_crop.png")

    # 7. Heavy salt-and-pepper noise overlaid on fundus
    arr = make_base_fundus(RNG, variation=0.0).astype(np.float32)
    noise_mask = RNG.random(arr.shape[:2]) < 0.25  # 25% pixel corruption
    arr[noise_mask] = RNG.integers(0, 256, (noise_mask.sum(), 3))
    save(Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)),
         OUT_OOD / "heavy_noise.png")

    # 8. Solid red — completely wrong modality
    red = np.zeros((*SIZE, 3), dtype=np.uint8)
    red[:, :, 0] = 200
    save(Image.fromarray(red), OUT_OOD / "solid_red.png")

    # 9. Severe brightness shift — near-black fundus (underexposed)
    arr = make_base_fundus(RNG, variation=0.0).astype(np.float32) * 0.05
    save(Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)),
         OUT_OOD / "severely_underexposed.png")

    # 10. Gradient sweep — no fundus structure at all
    H, W = SIZE
    gradient = np.zeros((H, W, 3), dtype=np.uint8)
    gradient[:, :, 0] = np.linspace(0, 255, W, dtype=np.uint8)
    gradient[:, :, 1] = np.linspace(255, 0, W, dtype=np.uint8)
    gradient[:, :, 2] = np.linspace(0, 128, W, dtype=np.uint8)
    save(Image.fromarray(gradient), OUT_OOD / "gradient.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("SYNTHETIC OOD TEST DATA GENERATOR")
    print("=" * 60)
    print("NOTE: These images are SYNTHETIC for software testing only.")
    print("They do NOT represent real fundus images or clinical OOD cases.")
    print()

    generate_in_distribution()
    generate_out_of_distribution()

    id_count = len(list(OUT_ID.glob("*.png")))
    ood_count = len(list(OUT_OOD.glob("*.png")))

    print(f"\n[DONE] Generated {id_count} in-distribution and {ood_count} OOD test images.")
    print(f"   ID  -> {OUT_ID.relative_to(ROOT)}/")
    print(f"   OOD -> {OUT_OOD.relative_to(ROOT)}/")
