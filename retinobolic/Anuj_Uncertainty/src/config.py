"""
config.py - Configuration for the DR Prediction Uncertainty Engine.

All tunable parameters live here so that thresholds and tolerances can be
adjusted (or loaded from a file / environment) without touching the core math.

---------------------------------------------------------------------------
IMPORTANT - THRESHOLD DISCLAIMER
---------------------------------------------------------------------------
LOW_UNCERTAINTY_MAX and HIGH_UNCERTAINTY_MIN are PROTOTYPE ENGINEERING
THRESHOLDS chosen for development convenience. They have NOT been calibrated
on a clinically representative dataset. They MUST be validated by domain
experts and/or empirical calibration studies before deployment in any
clinical or screening setting.
---------------------------------------------------------------------------
"""

from dataclasses import dataclass


@dataclass
class UncertaintyConfig:
    """
    Centralised configuration for the Uncertainty Engine.

    Attributes
    ----------
    NUM_CLASSES : int
        Number of output classes produced by Vinayak's DR model.
        Must be 4 (No DR / Mild / Moderate / Severe+Proliferative).

    PROB_SUM_TOLERANCE : float
        Maximum allowed deviation of sum(probabilities) from 1.0 before the
        input is rejected as invalid.  Tiny floating-point rounding errors
        (e.g. 0.9999999) are tolerated; materially incorrect distributions
        (e.g. 1.3) are rejected.

    EPSILON : float
        Small constant used for zero-masking inside the log computation so
        that the standard convention 0 * log(0) == 0 is faithfully honoured.

    LOW_UNCERTAINTY_MAX : float
        Uncertainty scores at or below this threshold map to level "low".
        PROTOTYPE threshold - requires validation on representative data.

    HIGH_UNCERTAINTY_MIN : float
        Uncertainty scores at or above this threshold map to level "high"
        and trigger review_recommended = True.
        PROTOTYPE threshold - requires validation on representative data.
    """
    NUM_CLASSES: int = 5
    PROB_SUM_TOLERANCE: float = 1e-3
    EPSILON: float = 1e-12

    # --- Uncertainty-level thresholds ---
    # PROTOTYPE THRESHOLDS - NOT CLINICALLY VALIDATED.
    LOW_UNCERTAINTY_MAX: float = 0.35
    HIGH_UNCERTAINTY_MIN: float = 0.70

    # Human-readable level labels
    LEVEL_LOW: str = "low"
    LEVEL_MEDIUM: str = "medium"
    LEVEL_HIGH: str = "high"

    def validate_self(self) -> None:
        """Sanity-check the configuration values. Raises ValueError if inconsistent."""
        if self.NUM_CLASSES < 2:
            raise ValueError(f"NUM_CLASSES must be >= 2, got {self.NUM_CLASSES}")
        if not (0.0 < self.PROB_SUM_TOLERANCE < 0.1):
            raise ValueError(
                f"PROB_SUM_TOLERANCE should be a small positive float (0 < tol < 0.1), "
                f"got {self.PROB_SUM_TOLERANCE}"
            )
        if not (0.0 < self.EPSILON < 1e-6):
            raise ValueError(f"EPSILON should be a very small positive float, got {self.EPSILON}")
        if not (0.0 <= self.LOW_UNCERTAINTY_MAX < self.HIGH_UNCERTAINTY_MIN <= 1.0):
            raise ValueError(
                f"Threshold ordering must satisfy "
                f"0 <= LOW_UNCERTAINTY_MAX ({self.LOW_UNCERTAINTY_MAX}) "
                f"< HIGH_UNCERTAINTY_MIN ({self.HIGH_UNCERTAINTY_MIN}) <= 1.0"
            )


# Module-level default instance - import and use directly if no customisation needed.
DEFAULT_CONFIG = UncertaintyConfig()
