"""
config.py — Central configuration for the Model Confidence Module.

All class mappings, confidence thresholds, and validation tolerances are
defined here as the single source of truth.

IMPORTANT:
  Confidence thresholds (HIGH_CONFIDENCE_THRESHOLD, MEDIUM_CONFIDENCE_THRESHOLD)
  are PROTOTYPE ENGINEERING THRESHOLDS ONLY.
  They have NOT been clinically validated and must NOT be used to make
  medical decisions without independent clinical evaluation.
"""

# ---------------------------------------------------------------------------
# Class mapping — Diabetic Retinopathy grading scale (ICDR)
# ---------------------------------------------------------------------------

NUM_CLASSES: int = 5

CLASS_MAPPING: dict[int, str] = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR (PDR)",
}

# ---------------------------------------------------------------------------
# Probability validation tolerances
# ---------------------------------------------------------------------------

# Acceptable deviation from sum=1.0 due to floating-point precision.
# Distributions whose sum differs from 1.0 by more than this value are
# treated as materially invalid and will raise InvalidProbabilityError.
PROBABILITY_SUM_TOLERANCE: float = 1e-3

# ---------------------------------------------------------------------------
# Confidence level thresholds
# ---------------------------------------------------------------------------
# Prototype engineering thresholds — require clinical validation before use.

HIGH_CONFIDENCE_THRESHOLD: float = 0.80
MEDIUM_CONFIDENCE_THRESHOLD: float = 0.50

# Canonical string labels for the three confidence levels.
CONFIDENCE_LEVEL_HIGH: str = "high"
CONFIDENCE_LEVEL_MEDIUM: str = "medium"
CONFIDENCE_LEVEL_LOW: str = "low"
