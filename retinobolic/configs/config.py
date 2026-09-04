"""
Central Configuration for VISIONARY6 — VINAYAK Module.
DR Classification & Explainable AI (Grad-CAM).

All tuneable parameters are defined here. Do NOT hard-code values in other modules.
Adheres strictly to the SIH Visionary6 Vinayak workmap specification.
"""
from pathlib import Path
import torch
import logging

# ---------------------------------------------------------------------------
# Base Directory Layout
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR          = BASE_DIR / "data"
RAW_DATA_DIR      = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR        = DATA_DIR / "splits"
METADATA_DIR      = DATA_DIR / "metadata"

MODELS_DIR        = BASE_DIR / "models"
CHECKPOINTS_DIR   = MODELS_DIR / "checkpoints"
FINAL_MODELS_DIR  = MODELS_DIR / "final"

OUTPUTS_DIR           = BASE_DIR / "outputs"
METRICS_DIR           = OUTPUTS_DIR / "metrics"
CONFUSION_MATRIX_DIR  = OUTPUTS_DIR / "confusion_matrix"
GRADCAM_OUTPUT_DIR    = OUTPUTS_DIR / "gradcam"
PREDICTIONS_DIR       = OUTPUTS_DIR / "predictions"

EXPERIMENTS_DIR = BASE_DIR / "experiments"

# ---------------------------------------------------------------------------
# Ensure all directories exist at import time
# ---------------------------------------------------------------------------
for _d in [
    RAW_DATA_DIR, PROCESSED_DATA_DIR, SPLITS_DIR, METADATA_DIR,
    CHECKPOINTS_DIR, FINAL_MODELS_DIR,
    METRICS_DIR, CONFUSION_MATRIX_DIR, GRADCAM_OUTPUT_DIR, PREDICTIONS_DIR,
    EXPERIMENTS_DIR,
]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 5-Class DR Label Contract
# ---------------------------------------------------------------------------
# Raw APTOS labels: 0, 1, 2, 3, 4  (5 classes)
# Mapped to 5 classes:
#   0 → 0  (No DR)
#   1 → 1  (Mild DR)
#   2 → 2  (Moderate DR)
#   3 → 3  (Severe DR)
#   4 → 4  (Proliferative DR)

CLASS_NAMES: dict[int, str] = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR"
}
NUM_CLASSES: int = len(CLASS_NAMES)  # 5

# ---------------------------------------------------------------------------
# Hardware Detection
# ---------------------------------------------------------------------------
def detect_device() -> str:
    """
    Auto-detects the best available compute device.
    Prints device, GPU name, and memory info for transparency.
    """
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[Config] Device : cuda ({gpu_name})")
        print(f"[Config] VRAM   : {mem_total:.1f} GB")
    else:
        device = "cpu"
        print("[Config] Device : cpu  (no CUDA GPU detected)")
    return device


DEVICE: str = detect_device()

# ---------------------------------------------------------------------------
# Model Settings
# ---------------------------------------------------------------------------
# Supported architectures: "efficientnet_b0", "resnet50", "mobilenet_v3_small"
MODEL_NAME: str = "efficientnet_b0"
DROPOUT_RATE: float = 0.3

# Checkpoint paths
CHECKPOINT_PATH   = CHECKPOINTS_DIR / "best_model.pth"
LAST_CHECKPOINT   = CHECKPOINTS_DIR / "last_model.pth"
FINAL_MODEL_PATH  = FINAL_MODELS_DIR / "final_model.pth"
CLASS_MAPPING_PATH = MODELS_DIR / "class_mapping.json"
TRAIN_CONFIG_PATH  = MODELS_DIR / "training_config.json"

# ---------------------------------------------------------------------------
# Preprocessing & Image Settings
# ---------------------------------------------------------------------------
IMAGE_SIZE: tuple[int, int] = (384, 384)   # (width, height)
IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float]  = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Augmentation Settings (training only)
# ---------------------------------------------------------------------------
AUGMENTATION_CONFIG: dict = {
    "random_horizontal_flip": True,
    "random_vertical_flip": True,
    "rotation_degrees": 15,
    "brightness_jitter": 0.1,
    "contrast_jitter": 0.1,
    "saturation_jitter": 0.1,
    "hue_jitter": 0.0,          # keep 0.0 — hue changes alter clinical appearance
}

# ---------------------------------------------------------------------------
# Training Hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE: int   = 16
NUM_EPOCHS: int   = 20
LEARNING_RATE: float = 1e-4
WEIGHT_DECAY: float  = 1e-4
RANDOM_SEED: int     = 42

# Mixed precision — only enabled on CUDA
USE_AMP: bool = (DEVICE == "cuda")

# ---------------------------------------------------------------------------
# Class Imbalance Handling
# ---------------------------------------------------------------------------
USE_WEIGHTED_SAMPLER: bool  = True   # WeightedRandomSampler for train loader
USE_CLASS_WEIGHTS_LOSS: bool = True  # Weighted CrossEntropyLoss

# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------
EARLY_STOPPING_PATIENCE: int = 5    # stop if no val loss improvement for N epochs

# ---------------------------------------------------------------------------
# Dataset Splits
# ---------------------------------------------------------------------------
TRAIN_RATIO: float = 0.70
VAL_RATIO:   float = 0.15
TEST_RATIO:  float = 0.15

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: int = logging.INFO

def get_logger(name: str) -> logging.Logger:
    """Returns a module-level logger with consistent formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(LOG_LEVEL)
    return logger

# ---------------------------------------------------------------------------
# Print startup banner
# ---------------------------------------------------------------------------
print(f"[Config] VISIONARY6 — VINAYAK Module loaded.")
print(f"[Config] BASE_DIR   : {BASE_DIR}")
print(f"[Config] IMAGE_SIZE : {IMAGE_SIZE}")
print(f"[Config] BATCH_SIZE : {BATCH_SIZE}")
print(f"[Config] DEVICE     : {DEVICE}")
print(f"[Config] USE_AMP    : {USE_AMP}")
