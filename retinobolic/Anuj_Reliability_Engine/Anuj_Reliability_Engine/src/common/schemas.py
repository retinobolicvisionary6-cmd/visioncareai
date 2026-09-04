"""
src/common/schemas.py — Data schemas and status enumerations.

Defines the typed result contracts exchanged between modules and
consumed by the Reliability Engine.  Using plain dataclasses keeps
this zero-dependency (no Pydantic) while remaining explicit.

NOTE: These are internal contracts for the Reliability Engine only.
Each upstream module retains its own public output format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Status enum
# ---------------------------------------------------------------------------

class ReliabilityStatus(str, Enum):
    """
    The three possible outcomes of the Reliability Engine.

    Priority order (highest to lowest):
        REVIEW_REQUIRED  — at least one critical signal triggered
        CAUTION          — intermediate / borderline signals
        ACCEPTABLE       — all signals in acceptable ranges

    IMPORTANT: This is an engineering reliability classification only.
    It does NOT constitute a medical diagnosis, clinical recommendation,
    or treatment guidance.
    """
    REVIEW_REQUIRED = "review_required"
    CAUTION = "caution"
    ACCEPTABLE = "acceptable"


# ---------------------------------------------------------------------------
# Module result dataclasses (typed views into upstream dicts)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConfidenceResult:
    """
    Typed view of the output from calculate_confidence().

    Fields extracted from the upstream dict:
        confidence       : float in [0, 1] — max class probability
        confidence_level : str — "high" | "medium" | "low"
    """
    confidence: float
    confidence_level: str       # "high" | "medium" | "low"

    # Optional: carried through for downstream display
    predicted_grade: Optional[int] = None
    predicted_class_name: Optional[str] = None
    confidence_percent: Optional[float] = None
    margin: Optional[float] = None


@dataclass(frozen=True)
class UncertaintyResult:
    """
    Typed view of the output from calculate_uncertainty().

    Fields extracted from the upstream dict:
        uncertainty       : float in [0, 1] — normalized Shannon entropy
        uncertainty_level : str — "low" | "medium" | "high"
        review_recommended: bool
    """
    uncertainty: float
    uncertainty_level: str      # "low" | "medium" | "high"
    review_recommended: bool

    # Optional: auxiliary signal
    probability_margin: Optional[float] = None


@dataclass(frozen=True)
class OODResult:
    """
    Typed view of the output from detect_ood().

    Fields extracted from the upstream dict:
        ood        : bool — True if out-of-distribution
        ood_status : str  — "in_distribution" | "review_required"
        ood_score  : float — computed distance score (lower = more in-dist)
    """
    ood: bool
    ood_status: str
    ood_score: float

    # Optional: carry through for display
    threshold: Optional[float] = None
    distance_metric: Optional[str] = None
    reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Final reliability output dataclass
# ---------------------------------------------------------------------------

@dataclass
class ReliabilityResult:
    """
    The single, unified output produced by the Reliability Engine.

    This is a JSON-serialisable, stable output contract.  All fields are
    either primitive types (float, bool, str, int) or None.

    IMPORTANT SAFETY NOTES
    ----------------------
    - reliability_status is an ENGINEERING classification only.
    - It does NOT represent diagnostic certainty or medical accuracy.
    - confidence != clinical accuracy; ood_status != a disease finding.
    - review_required = True means the pipeline flags this case for human
      review — it is NOT a diagnosis and NOT a treatment recommendation.
    """
    # --- Core reliability decision ---
    reliability_status: str          # "acceptable" | "caution" | "review_required"
    review_required: bool            # True when status == "review_required"
    reason: str                      # Human-readable explanation

    # --- Preserved upstream signals ---
    confidence: float                # from Confidence Module
    confidence_level: str            # "high" | "medium" | "low"
    uncertainty: float               # from Uncertainty Module
    uncertainty_level: str           # "low" | "medium" | "high"
    ood: bool                        # from OOD Module
    ood_status: str                  # "in_distribution" | "review_required"
    ood_score: float                 # distance score from OOD Module

    # --- Optional engineering score (mathematically defined) ---
    reliability_score: Optional[float] = None  # [0, 1]; None if not computed

    # --- Optional metadata pass-through ---
    predicted_grade: Optional[int] = None
    predicted_class_name: Optional[str] = None

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict of all non-None fields."""
        out = {
            "reliability_status": self.reliability_status,
            "review_required": self.review_required,
            "reason": self.reason,
            "confidence": self.confidence,
            "confidence_level": self.confidence_level,
            "uncertainty": self.uncertainty,
            "uncertainty_level": self.uncertainty_level,
            "ood": self.ood,
            "ood_status": self.ood_status,
            "ood_score": self.ood_score,
        }
        if self.reliability_score is not None:
            out["reliability_score"] = self.reliability_score
        if self.predicted_grade is not None:
            out["predicted_grade"] = self.predicted_grade
        if self.predicted_class_name is not None:
            out["predicted_class_name"] = self.predicted_class_name
        return out
