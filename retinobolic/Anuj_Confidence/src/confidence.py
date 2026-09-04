"""
confidence.py — Model Confidence Module (Step 3.1)

Calculates the probability-based confidence signal from the Diabetic
Retinopathy (DR) model's output probability distribution.

PUBLIC API
----------
    calculate_confidence(dr_result, include_top2=True) -> dict

PURPOSE
-------
This module answers ONLY:
    "How strong is the DR model's current probability-based confidence
     in its selected class?"

It does NOT:
    - Diagnose Diabetic Retinopathy.
    - Estimate clinical certainty.
    - Detect out-of-distribution (OOD) samples.
    - Quantify epistemic / aleatoric uncertainty.
    - Determine referral priority.
    - Replace or advise a physician.

IMPORTANT DISTINCTION
---------------------
    Model Confidence  ≠  Model Accuracy  ≠  Clinical Certainty

A model can output Confidence = 0.95 and still be wrong.
Confidence is a model-output signal only. It will later be combined with
Uncertainty, OOD, and Camera Reliability scores by the Reliability Engine.

INTEGRATION
-----------
    from src.confidence import calculate_confidence

    dr_result = {
        "grade": 2,
        "probabilities": {"0": 0.03, "1": 0.08, "2": 0.81, "3": 0.08}
    }
    result = calculate_confidence(dr_result)
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np

from src.config import (
    CLASS_MAPPING,
    CONFIDENCE_LEVEL_HIGH,
    CONFIDENCE_LEVEL_LOW,
    CONFIDENCE_LEVEL_MEDIUM,
    HIGH_CONFIDENCE_THRESHOLD,
    MEDIUM_CONFIDENCE_THRESHOLD,
    NUM_CLASSES,
    PROBABILITY_SUM_TOLERANCE,
)

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class ConfidenceModuleError(Exception):
    """Base exception for all errors raised by the Confidence Module."""


class InvalidInputFormatError(ConfidenceModuleError):
    """
    Raised when the input structure is malformed:
    wrong type, missing keys, unexpected keys, wrong number of classes, etc.
    """


class InvalidProbabilityError(ConfidenceModuleError):
    """
    Raised when individual probability values or their distribution are invalid:
    NaN, Infinity, negative, > 1.0, or sum materially ≠ 1.0.
    """


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_VALID_KEYS = {str(i) for i in range(NUM_CLASSES)} | {i for i in range(NUM_CLASSES)}


def _parse_probabilities(raw: object) -> np.ndarray:
    """
    Convert a raw probability input into an ordered numpy array of shape (4,).

    Accepts:
        - dict with string keys "0".."3"  → {"0": 0.03, "1": 0.08, ...}
        - dict with int keys 0..3         → {0: 0.03, 1: 0.08, ...}
        - list / tuple of 4 values        → [0.03, 0.08, 0.81, 0.08]
        - numpy array of shape (4,)

    All other types raise InvalidInputFormatError.
    """
    if isinstance(raw, dict):
        # Normalise keys to integers.
        parsed: dict[int, float] = {}
        for k, v in raw.items():
            if isinstance(k, str):
                if not k.isdigit():
                    raise InvalidInputFormatError(
                        f"Dictionary key '{k}' is not a valid class index string. "
                        f"Expected string digits '0'–'{NUM_CLASSES - 1}'."
                    )
                parsed[int(k)] = v
            elif isinstance(k, int):
                parsed[k] = v
            else:
                raise InvalidInputFormatError(
                    f"Dictionary key '{k}' has unsupported type {type(k).__name__}. "
                    "Expected str or int."
                )

        # Verify all expected classes are present.
        expected = set(range(NUM_CLASSES))
        actual = set(parsed.keys())

        missing = expected - actual
        extra = actual - expected

        if missing:
            raise InvalidInputFormatError(
                f"Missing probability classes: {sorted(missing)}. "
                f"All {NUM_CLASSES} classes (0–{NUM_CLASSES - 1}) must be provided."
            )
        if extra:
            raise InvalidInputFormatError(
                f"Unexpected probability classes: {sorted(extra)}. "
                f"Only classes 0–{NUM_CLASSES - 1} are valid."
            )

        return np.array([parsed[i] for i in range(NUM_CLASSES)], dtype=float)

    elif isinstance(raw, (list, tuple, np.ndarray)):
        arr = np.asarray(raw, dtype=float)
        if arr.ndim != 1 or len(arr) != NUM_CLASSES:
            raise InvalidInputFormatError(
                f"Expected a 1-D sequence with exactly {NUM_CLASSES} probabilities, "
                f"got shape {arr.shape}."
            )
        return arr

    else:
        raise InvalidInputFormatError(
            f"'probabilities' must be a dict, list, or numpy array; "
            f"got {type(raw).__name__}."
        )


def _validate_probability_values(probs: np.ndarray) -> None:
    """
    Validate that every element of *probs* is a finite real number in [0, 1]
    and that the distribution sums to approximately 1.

    Raises:
        InvalidProbabilityError — on any constraint violation.
    """
    # 1. Check for NaN.
    nan_mask = np.isnan(probs)
    if nan_mask.any():
        bad = np.where(nan_mask)[0].tolist()
        raise InvalidProbabilityError(
            f"NaN detected in probability values at class indices {bad}. "
            "The DR model output is invalid."
        )

    # 2. Check for Infinity.
    inf_mask = np.isinf(probs)
    if inf_mask.any():
        bad = np.where(inf_mask)[0].tolist()
        raise InvalidProbabilityError(
            f"Infinite value detected in probability values at class indices {bad}. "
            "The DR model output is invalid."
        )

    # 3. Check for negative probabilities.
    neg_mask = probs < 0.0
    if neg_mask.any():
        bad_vals = {int(i): float(probs[i]) for i in np.where(neg_mask)[0]}
        raise InvalidProbabilityError(
            f"Negative probability values detected: {bad_vals}. "
            "Probabilities must be in [0, 1]."
        )

    # 4. Check for values > 1.
    over_mask = probs > 1.0
    if over_mask.any():
        bad_vals = {int(i): float(probs[i]) for i in np.where(over_mask)[0]}
        raise InvalidProbabilityError(
            f"Probability values exceeding 1.0 detected: {bad_vals}. "
            "Probabilities must be in [0, 1]."
        )

    # 5. Check sum ≈ 1.0.
    total = float(probs.sum())
    if not math.isclose(total, 1.0, abs_tol=PROBABILITY_SUM_TOLERANCE):
        raise InvalidProbabilityError(
            f"Probability distribution sums to {total:.8f}, which deviates from "
            f"1.0 by more than the allowed tolerance ({PROBABILITY_SUM_TOLERANCE}). "
            "The DR model output is materially invalid and will not be silently normalised."
        )


def validate_probabilities(
    raw: object,
) -> np.ndarray:
    """
    Parse and validate a raw probability input.

    Returns:
        np.ndarray of shape (4,) — ordered probabilities for classes 0..3.

    Raises:
        InvalidInputFormatError — structural / format errors.
        InvalidProbabilityError — value / distribution errors.
    """
    probs = _parse_probabilities(raw)
    _validate_probability_values(probs)
    return probs


# ---------------------------------------------------------------------------
# Confidence level
# ---------------------------------------------------------------------------


def get_confidence_level(confidence: float) -> str:
    """
    Map a scalar confidence value to a human-readable level string.

    Thresholds (from config.py) are prototype engineering values only —
    they require clinical validation before operational use.

    Returns:
        "high"   if confidence >= HIGH_CONFIDENCE_THRESHOLD
        "medium" if confidence >= MEDIUM_CONFIDENCE_THRESHOLD
        "low"    otherwise
    """
    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
        return CONFIDENCE_LEVEL_HIGH
    elif confidence >= MEDIUM_CONFIDENCE_THRESHOLD:
        return CONFIDENCE_LEVEL_MEDIUM
    else:
        return CONFIDENCE_LEVEL_LOW


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def calculate_confidence(
    dr_result: dict,
    include_top2: bool = True,
) -> dict:
    """
    Calculate the Model Confidence from a DR model output dictionary.

    Parameters
    ----------
    dr_result : dict
        Output from the DR model. Must contain a "probabilities" key.
        The "grade" and "gradcam_path" keys are accepted but ignored.

        Example::

            {
                "grade": 2,
                "probabilities": {
                    "0": 0.03, "1": 0.08, "2": 0.81, "3": 0.08
                }
            }

    include_top2 : bool, optional (default=True)
        When True, includes top-2 probability analysis fields in the output
        ("top_class", "top_probability", "second_class", "second_probability",
        "margin"). These are informational and intended for downstream
        uncertainty analysis. Setting to False removes them from the output.

    Returns
    -------
    dict
        Core fields (always present):
            predicted_grade (int):
                Class index with the highest probability (argmax).
            predicted_class_name (str):
                Human-readable label for the predicted grade from CLASS_MAPPING.
            confidence (float):
                Maximum class probability, in [0, 1].
                This is the canonical Model Confidence value.
            confidence_percent (float):
                confidence × 100.0, for display convenience only.
                Canonical numerical value remains `confidence`.
            confidence_level (str):
                Human-readable level: "high" | "medium" | "low".

        Optional top-2 fields (when include_top2=True):
            top_class (int):
                Class index with the highest probability (same as predicted_grade).
            top_probability (float):
                Probability of the top class.
            second_class (int):
                Class index with the second-highest probability.
            second_probability (float):
                Probability of the second class.
            margin (float):
                top_probability − second_probability.
                Larger margin → less ambiguity between top two classes.

    Raises
    ------
    InvalidInputFormatError
        If dr_result is not a dict, or 'probabilities' is missing or
        structurally invalid.
    InvalidProbabilityError
        If any probability value is NaN, Infinite, negative, > 1.0,
        or the distribution sum deviates materially from 1.0.

    Notes
    -----
    * This function does NOT mutate the input dr_result object.
    * Confidence is a model-output signal only:
        Model Confidence ≠ Model Accuracy ≠ Clinical Certainty.
    * Thresholds in config.py are prototype engineering values only.
    """
    # --- Input format check ------------------------------------------------
    if not isinstance(dr_result, dict):
        raise InvalidInputFormatError(
            f"dr_result must be a dict; got {type(dr_result).__name__}."
        )

    if "probabilities" not in dr_result:
        raise InvalidInputFormatError(
            "'probabilities' key is missing from dr_result. "
            "The DR model must supply a probability distribution."
        )

    # --- Validate probabilities ---------------------------------------------
    probs = validate_probabilities(dr_result["probabilities"])

    # --- Predicted grade (argmax) -------------------------------------------
    predicted_grade = int(np.argmax(probs))
    predicted_class_name = CLASS_MAPPING[predicted_grade]

    # --- Confidence (max probability) ---------------------------------------
    confidence = float(probs[predicted_grade])
    confidence_percent = round(confidence * 100.0, 2)
    confidence_level = get_confidence_level(confidence)

    # --- Build output dict --------------------------------------------------
    result: dict = {
        "predicted_grade": predicted_grade,
        "predicted_class_name": predicted_class_name,
        "confidence": confidence,
        "confidence_percent": confidence_percent,
        "confidence_level": confidence_level,
    }

    # --- Optional top-2 margin analysis -------------------------------------
    if include_top2:
        sorted_indices = np.argsort(probs)[::-1]  # descending order
        top_idx = int(sorted_indices[0])
        second_idx = int(sorted_indices[1])

        top_probability = float(probs[top_idx])
        second_probability = float(probs[second_idx])
        margin = round(top_probability - second_probability, 10)

        result["top_class"] = top_idx
        result["top_probability"] = top_probability
        result["second_class"] = second_idx
        result["second_probability"] = second_probability
        result["margin"] = margin

    return result
