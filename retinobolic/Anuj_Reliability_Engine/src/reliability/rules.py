"""
src/reliability/rules.py — Deterministic reliability decision rules.

Implements the 5-priority rule hierarchy specified in the Reliability Engine
design.  All logic is transparent, explicit, and testable in isolation.

Priority order (highest = evaluated first):
    1. OOD        — OOD = True                    → review_required
    2. Uncertainty — uncertainty_level = "high"   → review_required
    3. Confidence  — confidence_level = "low"     → review_required
    4. Caution     — intermediate signals          → caution
    5. Acceptable  — all signals in good range     → acceptable

KEY SAFETY RULE:
    OOD takes highest priority.
    Even if confidence = 0.99 and uncertainty = 0.0, OOD = True MUST
    return review_required.

    Rationale: a high-confidence prediction on an out-of-distribution
    input provides false assurance and is MORE dangerous, not less.

CONFLICTING SIGNALS handled deterministically:
    Case A: high confidence + low uncertainty + OOD = True  → review_required
    Case B: high confidence + high uncertainty + OOD = False → review_required
    Case C: low confidence  + low uncertainty + OOD = False  → review_required
    Case D: medium confidence + medium uncertainty + OOD = False → caution
    Case E: high confidence + low uncertainty + OOD = False → acceptable

IMPORTANT: This file contains engineering logic only.
           It does NOT make medical diagnoses or clinical recommendations.
"""

from __future__ import annotations

from .config import ReliabilityConfig, DEFAULT_CONFIG
from ..common.schemas import (
    ConfidenceResult,
    OODResult,
    ReliabilityStatus,
    UncertaintyResult,
)


# ---------------------------------------------------------------------------
# Reason strings
# ---------------------------------------------------------------------------

_REASON_ACCEPTABLE = (
    "High model confidence, low uncertainty and in-distribution input."
)
_REASON_CAUTION_BASE = (
    "Model confidence or uncertainty is intermediate; "
    "additional review may be appropriate."
)
_REASON_CAUTION_MEDIUM_CONF = (
    "Model confidence is intermediate; additional review may be appropriate."
)
_REASON_CAUTION_MEDIUM_UNCERT = (
    "Model uncertainty is intermediate; additional review may be appropriate."
)
_REASON_HIGH_UNCERTAINTY = "Prediction has high model uncertainty."
_REASON_LOW_CONFIDENCE = "Model confidence is low."
_REASON_OOD = (
    "Input appears outside the configured reference distribution."
)
_REASON_BORDERLINE_OOD = (
    "Input is near the OOD threshold boundary; treat with additional caution."
)


def _build_reason(*parts: str) -> str:
    """
    Combine multiple reason parts into a single human-readable string.

    Parts are joined with a space and any empty strings are ignored.
    """
    clean = [p.strip() for p in parts if p and p.strip()]
    return " ".join(clean)


# ---------------------------------------------------------------------------
# Rule evaluation
# ---------------------------------------------------------------------------

def evaluate_rules(
    confidence: ConfidenceResult,
    uncertainty: UncertaintyResult,
    ood: OODResult,
    config: ReliabilityConfig = DEFAULT_CONFIG,
) -> tuple[ReliabilityStatus, bool, str]:
    """
    Apply deterministic reliability rules and return a decision.

    Parameters
    ----------
    confidence  : ConfidenceResult   — validated output from Confidence Module
    uncertainty : UncertaintyResult  — validated output from Uncertainty Module
    ood         : OODResult          — validated output from OOD Module
    config      : ReliabilityConfig  — threshold configuration

    Returns
    -------
    (status, review_required, reason) : tuple
        status          : ReliabilityStatus enum value
        review_required : bool — True when status == REVIEW_REQUIRED
        reason          : str  — human-readable explanation

    Decision logic (evaluated in priority order):
    --------------------------------------------
    PRIORITY 1 — OOD
        IF ood.ood == True AND config.OOD_STRICT_MODE
            → REVIEW_REQUIRED
            Reason: "Input appears outside the configured reference distribution."
            [Additional signals are reported alongside for transparency]

    PRIORITY 2 — High Uncertainty
        IF uncertainty.uncertainty_level == "high"
            → REVIEW_REQUIRED
            Reason: "Prediction has high model uncertainty."

    PRIORITY 3 — Low Confidence
        IF confidence.confidence_level == "low"
            → REVIEW_REQUIRED
            Reason: "Model confidence is low."

    PRIORITY 4 — Caution (intermediate)
        IF confidence.confidence_level == "medium"
           OR uncertainty.uncertainty_level == "medium"
            → CAUTION
            Reason: descriptive intermediate signal message

    PRIORITY 5 — Acceptable
        (All signals pass — high confidence, low uncertainty, in-distribution)
            → ACCEPTABLE
            Reason: "High model confidence, low uncertainty and in-distribution input."
    """

    # Collect all active issue signals for combined reason generation
    issues: list[str] = []

    # ---------------------------------------------------------------------------
    # PRIORITY 1: OOD check (highest priority)
    # ---------------------------------------------------------------------------
    if ood.ood and config.OOD_STRICT_MODE:
        # Collect any additional issues for transparency even when OOD triggers
        if uncertainty.uncertainty_level == "high":
            issues.append(_REASON_HIGH_UNCERTAINTY)
        if confidence.confidence_level == "low":
            issues.append(_REASON_LOW_CONFIDENCE)

        if issues:
            reason = _build_reason(_REASON_OOD, *issues)
        else:
            reason = _REASON_OOD

        return ReliabilityStatus.REVIEW_REQUIRED, True, reason

    # ---------------------------------------------------------------------------
    # PRIORITY 2: High uncertainty
    # ---------------------------------------------------------------------------
    if uncertainty.uncertainty_level == "high":
        # Report low confidence as additional issue if also present
        if confidence.confidence_level == "low":
            reason = _build_reason(_REASON_HIGH_UNCERTAINTY, _REASON_LOW_CONFIDENCE)
        else:
            reason = _REASON_HIGH_UNCERTAINTY

        return ReliabilityStatus.REVIEW_REQUIRED, True, reason

    # ---------------------------------------------------------------------------
    # PRIORITY 3: Low confidence
    # ---------------------------------------------------------------------------
    if confidence.confidence_level == "low":
        return ReliabilityStatus.REVIEW_REQUIRED, True, _REASON_LOW_CONFIDENCE

    # ---------------------------------------------------------------------------
    # PRIORITY 4: Caution (intermediate signals)
    # ---------------------------------------------------------------------------
    caution_triggers: list[str] = []

    if confidence.confidence_level == "medium":
        caution_triggers.append(_REASON_CAUTION_MEDIUM_CONF)

    if uncertainty.uncertainty_level == "medium":
        caution_triggers.append(_REASON_CAUTION_MEDIUM_UNCERT)

    if caution_triggers:
        reason = _build_reason(*caution_triggers)
        return ReliabilityStatus.CAUTION, False, reason

    # ---------------------------------------------------------------------------
    # PRIORITY 5: Acceptable
    # ---------------------------------------------------------------------------
    return ReliabilityStatus.ACCEPTABLE, False, _REASON_ACCEPTABLE
