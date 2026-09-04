"""
DR Prediction Uncertainty Engine — Module 2 (SIH26038)

Public API exports for the uncertainty module.

Usage
-----
    from src import calculate_uncertainty, UncertaintyConfig
    from src import compute_shannon_entropy, compute_normalized_uncertainty
    from src import compute_probability_margin
    from src import InvalidProbabilityError, ValidationError

Primary function:
    calculate_uncertainty(dr_result, config=None, confidence=None) -> dict

    Consumes Vinayak's DR model output (probabilities dict) and returns:
        {
            "predicted_grade":    int,
            "uncertainty":        float,   # normalized entropy in [0, 1]
            "uncertainty_level":  str,     # "low" | "medium" | "high"
            "review_recommended": bool,
            "probability_margin": float,   # top - second-highest probability
        }
"""

from .config import UncertaintyConfig, DEFAULT_CONFIG
from .validation import InvalidProbabilityError, ValidationError, validate_probabilities
from .uncertainty import (
    calculate_uncertainty,
    compute_shannon_entropy,
    compute_normalized_uncertainty,
    compute_probability_margin,
    determine_uncertainty_level,
)

__all__ = [
    # Primary API
    "calculate_uncertainty",
    # Building blocks
    "compute_shannon_entropy",
    "compute_normalized_uncertainty",
    "compute_probability_margin",
    "determine_uncertainty_level",
    # Validation
    "validate_probabilities",
    # Exceptions
    "InvalidProbabilityError",
    "ValidationError",
    # Config
    "UncertaintyConfig",
    "DEFAULT_CONFIG",
]
