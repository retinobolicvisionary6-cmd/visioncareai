"""
validation.py - Strict probability distribution validation for the Uncertainty Engine.

Validates raw model output before any calculations are performed.
All validation failures produce clear, human-readable error messages.
"""

from __future__ import annotations

import math
from typing import Any, Union

import numpy as np

from .config import UncertaintyConfig, DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class InvalidProbabilityError(ValueError):
    """
    Raised when the probability distribution from the DR model is invalid.

    This exception is intentionally specific so that downstream code can
    catch only probability-related errors without swallowing broader
    ValueErrors.
    """


class ValidationError(ValueError):
    """
    Raised when the top-level input structure (e.g. missing keys, wrong types)
    is malformed before even inspecting the probabilities.
    """


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_EXPECTED_KEYS = {"0", "1", "2", "3", "4"}

def _parse_prob_dict(prob_dict: dict) -> list:
    """
    Convert a probability dictionary with string or integer keys to an ordered
    list [p0, p1, p2, p3].

    Accepts both {"0": 0.5, "1": 0.3, ...} and {0: 0.5, 1: 0.3, ...}.
    Raises InvalidProbabilityError for missing or extra class keys.
    """
    str_keys = {str(k): v for k, v in prob_dict.items()}
    provided_keys = set(str_keys.keys())

    missing = _EXPECTED_KEYS - provided_keys
    extra = provided_keys - _EXPECTED_KEYS

    messages = []
    if missing:
        messages.append(f"Missing class key(s): {sorted(missing)}")
    if extra:
        messages.append(f"Unexpected class key(s): {sorted(extra)} (expected only 0-4)")

    if messages:
        raise InvalidProbabilityError(
            "Invalid probability distribution:\n"
            + "\n".join(f"  - {m}" for m in messages)
            + "\nExpected exactly 5 class probabilities with keys '0', '1', '2', '3', '4'."
        )

    return [str_keys["0"], str_keys["1"], str_keys["2"], str_keys["3"], str_keys["4"]]


def _check_individual_values(values: list) -> None:
    """
    Check each probability value for type, finiteness, and bounds [0, 1].
    Collects all errors before raising so the user sees all problems at once.
    """
    errors = []
    for i, v in enumerate(values):
        if v is None:
            errors.append(f"  - Class {i}: value is None (expected a float in [0, 1])")
            continue
        if not isinstance(v, (int, float)):
            errors.append(
                f"  - Class {i}: value is {type(v).__name__!r} ({v!r}), "
                f"expected a numeric float in [0, 1]"
            )
            continue
        if math.isnan(v):
            errors.append(f"  - Class {i}: value is NaN (not a valid probability)")
            continue
        if math.isinf(v):
            errors.append(f"  - Class {i}: value is {'positive' if v > 0 else 'negative'} "
                          f"infinity (not a valid probability)")
            continue
        if v < 0.0:
            errors.append(
                f"  - Class {i}: value {v!r} is negative (probabilities must be >= 0)"
            )
        if v > 1.0:
            errors.append(
                f"  - Class {i}: value {v!r} exceeds 1.0 (probabilities must be <= 1)"
            )

    if errors:
        raise InvalidProbabilityError(
            "Invalid probability distribution:\n"
            + "\n".join(errors)
            + "\nAll 4 class probabilities must be finite numeric values in [0, 1]."
        )


def _check_sum(values: list, tolerance: float) -> None:
    """
    Verify that the probabilities sum to approximately 1.0.
    """
    total = sum(values)
    if abs(total - 1.0) > tolerance:
        raise InvalidProbabilityError(
            f"Invalid probability distribution:\n"
            f"  - Probabilities sum to {total:.6f}, expected approximately 1.0 "
            f"(tolerance: +/-{tolerance}).\n"
            f"  - Values: {values}\n"
            f"This may indicate a bug in the DR model's softmax output."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_probabilities(
    probabilities: Any,
    config: UncertaintyConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """
    Validate and normalise a raw probability input into a clean NumPy array.

    Accepted input formats
    ----------------------
    - dict with string keys:  {"0": 0.03, "1": 0.08, "2": 0.81, "3": 0.08}
    - dict with integer keys: {0: 0.03, 1: 0.08, 2: 0.81, 3: 0.08}
    - list / tuple of length 4: [0.03, 0.08, 0.81, 0.08]
    - numpy ndarray of shape (4,)

    Parameters
    ----------
    probabilities : Any
        Raw probability input from the DR model output.
    config : UncertaintyConfig
        Active configuration (controls tolerance, expected class count, etc.).

    Returns
    -------
    np.ndarray
        Validated probability vector of dtype float64 and shape (NUM_CLASSES,).

    Raises
    ------
    InvalidProbabilityError
        If the distribution is malformed in any way.
    ValidationError
        If the input type is entirely unsupported.
    """
    if probabilities is None:
        raise InvalidProbabilityError(
            "Invalid probability distribution:\n"
            "  - Received None. Expected a dict, list, tuple, or numpy array "
            "of 4 class probabilities."
        )

    # ---- Step 1: parse into a flat list ----
    if isinstance(probabilities, dict):
        if len(probabilities) == 0:
            raise InvalidProbabilityError(
                "Invalid probability distribution:\n"
                "  - Received an empty dictionary. Expected 5 class probabilities."
            )
        values_list = _parse_prob_dict(probabilities)

    elif isinstance(probabilities, (list, tuple)):
        if len(probabilities) == 0:
            raise InvalidProbabilityError(
                "Invalid probability distribution:\n"
                "  - Received an empty sequence. Expected exactly 5 class probabilities."
            )
        if len(probabilities) != config.NUM_CLASSES:
            raise InvalidProbabilityError(
                f"Invalid probability distribution:\n"
                f"  - Expected {config.NUM_CLASSES} probabilities, "
                f"got {len(probabilities)}.\n"
                f"  - Values: {list(probabilities)}"
            )
        values_list = list(probabilities)

    elif isinstance(probabilities, np.ndarray):
        if probabilities.size == 0:
            raise InvalidProbabilityError(
                "Invalid probability distribution:\n"
                "  - Received an empty NumPy array."
            )
        flat = probabilities.flatten()
        if flat.shape[0] != config.NUM_CLASSES:
            raise InvalidProbabilityError(
                f"Invalid probability distribution:\n"
                f"  - Expected {config.NUM_CLASSES} probabilities, "
                f"got array with {flat.shape[0]} elements."
            )
        values_list = flat.tolist()

    else:
        raise ValidationError(
            f"Unsupported input type for probabilities: {type(probabilities).__name__!r}.\n"
            f"Expected dict, list, tuple, or numpy.ndarray."
        )

    # ---- Step 2: validate individual values ----
    _check_individual_values(values_list)

    # ---- Step 3: validate sum ----
    _check_sum(values_list, config.PROB_SUM_TOLERANCE)

    return np.array(values_list, dtype=np.float64)


def validate_dr_input(
    dr_result: Any,
    config: UncertaintyConfig = DEFAULT_CONFIG,
) -> tuple[np.ndarray, int | None]:
    """
    Validate the top-level DR model output dictionary and extract probabilities.

    Parameters
    ----------
    dr_result : Any
        The full output dictionary from Vinayak's model.
        Must contain at least the "probabilities" key.
    config : UncertaintyConfig
        Active configuration.

    Returns
    -------
    probs : np.ndarray
        Validated probability array of shape (4,).
    predicted_grade : int or None
        The "grade" field if present and valid, otherwise None.

    Raises
    ------
    ValidationError
        If the top-level structure is wrong (not a dict, missing keys, etc.).
    InvalidProbabilityError
        If the probability values are invalid.
    """
    if not isinstance(dr_result, dict):
        raise ValidationError(
            f"DR model output must be a dictionary, got {type(dr_result).__name__!r}.\n"
            f"Expected format: {{\"grade\": int, \"probabilities\": {{\"0\": p0, ...}}}}"
        )

    if "probabilities" not in dr_result:
        raise ValidationError(
            "DR model output is missing the required 'probabilities' key.\n"
            "Expected format: {\"grade\": int, \"probabilities\": {\"0\": p0, ...}}"
        )

    # Extract optional grade
    predicted_grade: int | None = None
    if "grade" in dr_result:
        grade_val = dr_result["grade"]
        if isinstance(grade_val, (int, float)) and not math.isnan(grade_val):
            predicted_grade = int(grade_val)

    probs = validate_probabilities(dr_result["probabilities"], config)
    return probs, predicted_grade
