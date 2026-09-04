"""
config.py — Configurable thresholds for the Fundus Quality Assessment engine.

All numeric thresholds live here so they can be tuned without touching
algorithm logic.  None of these values are clinically validated; they are
engineering starting points for a prototype.
"""

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

# Resize the longest edge to this value before analysis (preserves aspect ratio).
# Large images slow down histogram analysis; 512 is a good prototype tradeoff.
RESIZE_LONG_EDGE: int = 512

# Minimum meaningful image dimension (px).  Images smaller than this in any
# axis are rejected as too small to assess.
MIN_IMAGE_DIM: int = 64

# Minimum number of non-black pixels to consider an image non-empty.
MIN_NONBLACK_PIXELS: int = 1000

# Supported input file extensions (lowercase, with dot).
SUPPORTED_EXTENSIONS: tuple = (".jpg", ".jpeg", ".png")

# ---------------------------------------------------------------------------
# Fundus ROI Detection
# ---------------------------------------------------------------------------

# Minimum fraction of total image pixels that must belong to the detected
# fundus circle.  Below this the image is considered too heavily cropped.
FOV_MIN_FUNDUS_RATIO: float = 0.30

# Grayscale threshold below which a pixel is considered "black background".
BACKGROUND_DARK_THRESHOLD: int = 20

# ---------------------------------------------------------------------------
# Focus / Sharpness
# ---------------------------------------------------------------------------

# Laplacian variance on the fundus ROI.
# Values are image-size independent because we work on the resized ROI.
FOCUS_LAP_VARIANCE_UNGRADABLE: float = 1.2   # hard gate: below → ungradable (calibrated to synthetic data)
FOCUS_LAP_VARIANCE_BORDERLINE: float = 5.0   # below → borderline
FOCUS_LAP_VARIANCE_GOOD: float = 30.0        # above → 1.0 focus score

# Weight of Laplacian vs. Tenengrad (gradient magnitude) when computing
# the combined focus score.
FOCUS_LAP_WEIGHT: float = 0.6
FOCUS_GRAD_WEIGHT: float = 0.4

# Tenengrad (Sobel gradient magnitude mean) thresholds.
FOCUS_GRAD_UNGRADABLE: float = 4.0    # hard gate (separates ungradable ~3.5 from borderline ~6.2)
FOCUS_GRAD_BORDERLINE: float = 7.0
FOCUS_GRAD_GOOD: float = 20.0

# ---------------------------------------------------------------------------
# Illumination
# ---------------------------------------------------------------------------

# Mean pixel intensity (0–255 on grayscale) thresholds.
ILLUM_MEAN_TOO_DARK: float = 30.0    # hard gate: below → ungradable
ILLUM_MEAN_DARK: float = 50.0        # below → borderline
ILLUM_MEAN_BRIGHT: float = 200.0     # above → borderline
ILLUM_MEAN_TOO_BRIGHT: float = 230.0 # hard gate: above → ungradable

# Fraction of pixels that may be clipped (≥254) before flagging overexposure.
ILLUM_CLIP_RATIO_BORDERLINE: float = 0.05   # 5 % clipped → borderline
ILLUM_CLIP_RATIO_UNGRADABLE: float = 0.20   # 20 % clipped → ungradable

# Minimum standard deviation of pixel intensities (contrast proxy).
ILLUM_STD_MIN_BORDERLINE: float = 20.0   # below → borderline (low contrast)
ILLUM_STD_MIN_UNGRADABLE: float = 8.0    # below → ungradable

# Illumination uniformity: std of per-quadrant mean brightness (0–255).
# High value → uneven illumination.
ILLUM_UNIFORMITY_BORDERLINE: float = 30.0
ILLUM_UNIFORMITY_UNGRADABLE: float = 60.0

# ---------------------------------------------------------------------------
# Field of View
# ---------------------------------------------------------------------------

# Fraction of the detected fundus disk area that must be non-black.
# If the detected disk is mostly black, it is heavily cropped or incorrect.
FOV_DISK_FILL_BORDERLINE: float = 0.65
FOV_DISK_FILL_UNGRADABLE: float = 0.40

# Fallback: if no circle is detected, use fraction of non-black pixels
# over total image area.
FOV_NONBLACK_RATIO_BORDERLINE: float = 0.35
FOV_NONBLACK_RATIO_UNGRADABLE: float = 0.20

# ---------------------------------------------------------------------------
# Retinal Visibility (prototype heuristic)
# ---------------------------------------------------------------------------

# Vessel density proxy: fraction of edge pixels inside the fundus ROI
# (after Canny edge detection).  Low value → poor vessel visibility.
RETVIS_EDGE_DENSITY_BORDERLINE: float = 0.04
RETVIS_EDGE_DENSITY_UNGRADABLE: float = 0.01

# ---------------------------------------------------------------------------
# Artifact Detection
# ---------------------------------------------------------------------------

# Glare: fraction of pixels at max intensity (255) in any channel.
ARTIFACT_GLARE_BORDERLINE: float = 0.03
ARTIFACT_GLARE_UNGRADABLE: float = 0.10

# Noise: estimated via high-frequency energy ratio.
ARTIFACT_NOISE_BORDERLINE: float = 0.18
ARTIFACT_NOISE_UNGRADABLE: float = 0.30

# ---------------------------------------------------------------------------
# Quality Score Fusion Weights
# ---------------------------------------------------------------------------

# Weights must sum to 1.0.
WEIGHT_FOCUS: float = 0.30
WEIGHT_ILLUMINATION: float = 0.25
WEIGHT_FIELD_OF_VIEW: float = 0.20
WEIGHT_RETINAL_VISIBILITY: float = 0.15
WEIGHT_ARTIFACT: float = 0.10

# Final composite quality score thresholds.
QUALITY_SCORE_GOOD: float = 0.75        # ≥ this → good
QUALITY_SCORE_BORDERLINE_LOW: float = 0.45  # < this → ungradable

# ---------------------------------------------------------------------------
# Enhancement
# ---------------------------------------------------------------------------

# CLAHE parameters.
CLAHE_CLIP_LIMIT: float = 2.0
CLAHE_TILE_GRID: tuple = (8, 8)

# Denoise strength (Non-local means).
DENOISE_H: int = 7
