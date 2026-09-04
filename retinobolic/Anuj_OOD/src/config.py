"""
OOD Detection Module — Configuration
=====================================
Centralised configuration for the Out-of-Distribution detection pipeline.
All tunable parameters live here; nothing is hard-coded inside detection functions.

Threshold guidance
------------------
OOD_THRESHOLD is a *prototype* value.
It MUST be calibrated on representative in-distribution validation images
before use in a clinical pipeline.
Recommended calibration: set threshold at the 95th–99th percentile of
in-distribution validation distances.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

REFERENCE_DIR = PROJECT_ROOT / "reference"
REFERENCE_STATS_FILE = REFERENCE_DIR / "reference_statistics.json"
REFERENCE_EMBEDDINGS_FILE = REFERENCE_DIR / "reference_embeddings.npy"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------------------
# Embedding Extractor
# ---------------------------------------------------------------------------
# Active extractor type.  Options:
#   "classical"    — multi-feature spatial/color/texture extractor (zero deep-learning deps)
#   "pretrained"   — lightweight PyTorch encoder (MobileNet-V2 by default)
#   "dr_model"     — adapter for Vinayak's trained DR model feature layer
EXTRACTOR_TYPE: str = "classical"

# If using the pretrained torch extractor
PRETRAINED_MODEL_NAME: str = "mobilenet_v2"  # or "resnet18"
PRETRAINED_FEATURE_LAYER: str = "features"   # layer from which to extract

# If using the DR model extractor — set this path when Vinayak's model is available
DR_MODEL_CHECKPOINT: str = ""  # e.g., "models/vinayak_dr_model.pth"
DR_MODEL_FEATURE_LAYER: str = "avgpool"

# ---------------------------------------------------------------------------
# Distance Method
# ---------------------------------------------------------------------------
# Active distance metric.  Options:
#   "mahalanobis"  — Mahalanobis distance (recommended; requires reference fit)
#   "cosine"       — Cosine distance from reference mean
#   "euclidean"    — L2 distance from reference mean (normalised)
#   "nearest"      — Nearest-centroid distance (multi-cluster fallback)
DISTANCE_METRIC: str = "mahalanobis"

# Regularisation strength added to covariance diagonal to prevent singularity
# (Ledoit–Wolf shrinkage is also used automatically).
COVARIANCE_REGULARISATION: float = 1e-5

# ---------------------------------------------------------------------------
# OOD Threshold
# ---------------------------------------------------------------------------
# PROTOTYPE VALUE — requires calibration on real in-distribution + OOD data.
# For Mahalanobis distance: typical in-distribution scores cluster near 0–1;
# values >> 1 suggest OOD.  Normalised cosine/euclidean sit in [0, 1].
OOD_THRESHOLD: float = 12.74

# Percentile of reference distribution distances to use as auto-threshold
# when build_reference.py is run (override with --threshold flag).
AUTO_THRESHOLD_PERCENTILE: float = 99.0

# ---------------------------------------------------------------------------
# Image Preprocessing
# ---------------------------------------------------------------------------
# Target size for embedding extraction (H, W)
IMAGE_SIZE: tuple[int, int] = (224, 224)

# Supported input extensions
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")

# ---------------------------------------------------------------------------
# Logging / Debug
# ---------------------------------------------------------------------------
VERBOSE: bool = False
