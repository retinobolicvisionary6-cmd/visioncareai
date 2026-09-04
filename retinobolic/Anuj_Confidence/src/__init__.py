"""
src — Model Confidence Module package.

Public exports:
    calculate_confidence    — Primary API: compute confidence from DR model output.
    validate_probabilities  — Validate a raw probability dict/list directly.
    get_confidence_level    — Map a scalar confidence to "high" | "medium" | "low".

    ConfidenceModuleError   — Base exception.
    InvalidInputFormatError — Raised on structural format errors.
    InvalidProbabilityError — Raised on invalid probability values.

    CLASS_MAPPING           — DR grade → human-readable label mapping.
    NUM_CLASSES             — Number of DR classes (4).
"""

from src.confidence import (
    ConfidenceModuleError,
    InvalidInputFormatError,
    InvalidProbabilityError,
    calculate_confidence,
    get_confidence_level,
    validate_probabilities,
)
from src.config import CLASS_MAPPING, NUM_CLASSES

__all__ = [
    "calculate_confidence",
    "validate_probabilities",
    "get_confidence_level",
    "ConfidenceModuleError",
    "InvalidInputFormatError",
    "InvalidProbabilityError",
    "CLASS_MAPPING",
    "NUM_CLASSES",
]
