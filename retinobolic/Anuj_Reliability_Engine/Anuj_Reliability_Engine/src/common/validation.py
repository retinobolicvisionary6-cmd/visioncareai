"""
src/common/validation.py — Input/output validation for the Reliability Engine.

Validates the structured outputs of each upstream module before they
are passed to the rules engine.  Does NOT recompute any module logic.

Validation philosophy:
    - Fail loudly on any invalid signal.
    - Never silently ignore or coerce NaN / Inf values.
    - Preserve original module outputs; do not modify them.
    - All validators return a clean typed dataclass on success.

IMPORTANT: This module validates engineering signals only.
           It does NOT assess clinical validity of DR model outputs.
"""

from __future__ import annotations

import math
from typing import Any

from .schemas import ConfidenceResult, OODResult, UncertaintyResult


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class ValidationError(Exception):
    """Raised when a module output fails structural or value validation."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_float_in_range(
    value: Any,
    name: str,
    lo: float = 0.0,
    hi: float = 1.0,
) -> float:
    """
    Ensure *value* is a finite float within [lo, hi].

    Raises ValidationError with a descriptive message on any violation.
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(
            f"'{name}' must be a numeric float, got {type(value).__name__!r}."
        )
    v = float(value)
    if math.isnan(v):
        raise ValidationError(f"'{name}' is NaN — the module produced an invalid result.")
    if math.isinf(v):
        raise ValidationError(f"'{name}' is Infinite — the module produced an invalid result.")
    if not (lo <= v <= hi):
        raise ValidationError(
            f"'{name}' = {v:.6f} is outside the valid range [{lo}, {hi}]."
        )
    return v


def _check_required_keys(result: dict, keys: list[str], source: str) -> None:
    """Raise ValidationError if any required key is absent from *result*."""
    missing = [k for k in keys if k not in result]
    if missing:
        raise ValidationError(
            f"{source} result is missing required keys: {missing}. "
            f"Got keys: {list(result.keys())}"
        )


def _check_string_in_set(value: Any, name: str, valid: set[str]) -> str:
    """Raise ValidationError if *value* is not one of the *valid* strings."""
    if not isinstance(value, str):
        raise ValidationError(
            f"'{name}' must be a str, got {type(value).__name__!r}."
        )
    if value not in valid:
        raise ValidationError(
            f"'{name}' = {value!r} is not a recognised value. "
            f"Expected one of: {sorted(valid)}"
        )
    return value


# ---------------------------------------------------------------------------
# DR result pre-validation
# ---------------------------------------------------------------------------

def validate_dr_input(dr_result: Any) -> dict:
    """
    Light structural validation of the upstream DR model output before
    passing it to downstream modules.

    Only checks for dict type and presence of 'probabilities'.
    Full probability validation is left to each downstream module.

    Parameters
    ----------
    dr_result : any
        Raw DR model result expected to be a dict.

    Returns
    -------
    dict — validated dr_result (unchanged).

    Raises
    ------
    ValidationError
    """
    if not isinstance(dr_result, dict):
        raise ValidationError(
            f"dr_result must be a dict; got {type(dr_result).__name__!r}."
        )
    if "probabilities" not in dr_result:
        raise ValidationError(
            "'probabilities' key is missing from dr_result. "
            "The DR model must supply a probability distribution."
        )
    return dr_result


# ---------------------------------------------------------------------------
# Module output validators
# ---------------------------------------------------------------------------

def validate_confidence_result(result: Any) -> ConfidenceResult:
    """
    Validate the output dict from calculate_confidence().

    Parameters
    ----------
    result : any
        Raw dict returned by the Confidence Module.

    Returns
    -------
    ConfidenceResult — typed, validated dataclass.

    Raises
    ------
    ValidationError — on structural, type, range, or NaN/Inf errors.
    """
    if not isinstance(result, dict):
        raise ValidationError(
            f"Confidence result must be a dict; got {type(result).__name__!r}."
        )

    _check_required_keys(result, ["confidence", "confidence_level"], "Confidence")

    confidence = _check_float_in_range(result["confidence"], "confidence", 0.0, 1.0)

    confidence_level = _check_string_in_set(
        result["confidence_level"],
        "confidence_level",
        {"high", "medium", "low"},
    )

    return ConfidenceResult(
        confidence=confidence,
        confidence_level=confidence_level,
        predicted_grade=result.get("predicted_grade"),
        predicted_class_name=result.get("predicted_class_name"),
        confidence_percent=result.get("confidence_percent"),
        margin=result.get("margin"),
    )


def validate_uncertainty_result(result: Any) -> UncertaintyResult:
    """
    Validate the output dict from calculate_uncertainty().

    Parameters
    ----------
    result : any
        Raw dict returned by the Uncertainty Module.

    Returns
    -------
    UncertaintyResult — typed, validated dataclass.

    Raises
    ------
    ValidationError — on structural, type, range, or NaN/Inf errors.
    """
    if not isinstance(result, dict):
        raise ValidationError(
            f"Uncertainty result must be a dict; got {type(result).__name__!r}."
        )

    _check_required_keys(
        result, ["uncertainty", "uncertainty_level", "review_recommended"], "Uncertainty"
    )

    uncertainty = _check_float_in_range(result["uncertainty"], "uncertainty", 0.0, 1.0)

    uncertainty_level = _check_string_in_set(
        result["uncertainty_level"],
        "uncertainty_level",
        {"low", "medium", "high"},
    )

    review_recommended = result["review_recommended"]
    if not isinstance(review_recommended, bool):
        raise ValidationError(
            f"'review_recommended' must be a bool; got {type(review_recommended).__name__!r}."
        )

    return UncertaintyResult(
        uncertainty=uncertainty,
        uncertainty_level=uncertainty_level,
        review_recommended=review_recommended,
        probability_margin=result.get("probability_margin"),
    )


def validate_ood_result(result: Any) -> OODResult:
    """
    Validate the output dict from detect_ood().

    Parameters
    ----------
    result : any
        Raw dict returned by the OOD Module.

    Returns
    -------
    OODResult — typed, validated dataclass.

    Raises
    ------
    ValidationError — on structural, type, or NaN/Inf errors.
    """
    if not isinstance(result, dict):
        raise ValidationError(
            f"OOD result must be a dict; got {type(result).__name__!r}."
        )

    _check_required_keys(result, ["ood", "ood_status", "ood_score"], "OOD")

    ood_flag = result["ood"]
    if not isinstance(ood_flag, bool):
        raise ValidationError(
            f"'ood' must be a bool; got {type(ood_flag).__name__!r}."
        )

    ood_status = _check_string_in_set(
        result["ood_status"],
        "ood_status",
        {"in_distribution", "review_required"},
    )

    # ood_score: finite float (NOT constrained to [0,1] — Mahalanobis can exceed 1)
    ood_score_raw = result["ood_score"]
    if not isinstance(ood_score_raw, (int, float)):
        raise ValidationError(
            f"'ood_score' must be numeric; got {type(ood_score_raw).__name__!r}."
        )
    ood_score = float(ood_score_raw)
    if math.isnan(ood_score):
        raise ValidationError("'ood_score' is NaN — the OOD module produced an invalid result.")
    if math.isinf(ood_score):
        raise ValidationError("'ood_score' is Infinite — the OOD module produced an invalid result.")
    if ood_score < 0.0:
        raise ValidationError(
            f"'ood_score' = {ood_score:.6f} is negative — distance scores must be >= 0."
        )

    return OODResult(
        ood=ood_flag,
        ood_status=ood_status,
        ood_score=ood_score,
        threshold=result.get("threshold"),
        distance_metric=result.get("distance_metric"),
        reason=result.get("reason"),
    )
