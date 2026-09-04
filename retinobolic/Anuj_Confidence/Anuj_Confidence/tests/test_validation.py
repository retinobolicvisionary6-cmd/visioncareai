"""
test_validation.py — Unit tests for probability input validation.

Tests the validate_probabilities() function and the structural / value
constraints enforced before any confidence calculation is attempted.
"""

import math

import numpy as np
import pytest

from src.confidence import (
    InvalidInputFormatError,
    InvalidProbabilityError,
    validate_probabilities,
)


# ---------------------------------------------------------------------------
# Valid input formats — should NOT raise
# ---------------------------------------------------------------------------


class TestValidInputFormats:
    """validate_probabilities should accept all valid equivalent formats."""

    def test_dict_string_keys(self):
        """Standard DR model output with string keys."""
        probs = validate_probabilities({"0": 0.03, "1": 0.08, "2": 0.81, "3": 0.08})
        assert probs.shape == (4,)
        assert math.isclose(probs[2], 0.81)

    def test_dict_int_keys(self):
        """Dict with integer keys — also valid."""
        probs = validate_probabilities({0: 0.03, 1: 0.08, 2: 0.81, 3: 0.08})
        assert probs.shape == (4,)
        assert math.isclose(probs[2], 0.81)

    def test_list_input(self):
        """Python list of 4 floats — valid."""
        probs = validate_probabilities([0.03, 0.08, 0.81, 0.08])
        assert probs.shape == (4,)

    def test_numpy_array_input(self):
        """Numpy array of shape (4,) — valid."""
        arr = np.array([0.03, 0.08, 0.81, 0.08])
        probs = validate_probabilities(arr)
        assert probs.shape == (4,)

    def test_returns_ordered_array(self):
        """Probabilities are ordered by class index 0..3 regardless of dict ordering."""
        # Deliberately provide out-of-order keys.
        probs = validate_probabilities({"3": 0.08, "0": 0.03, "2": 0.81, "1": 0.08})
        assert math.isclose(probs[0], 0.03)
        assert math.isclose(probs[1], 0.08)
        assert math.isclose(probs[2], 0.81)
        assert math.isclose(probs[3], 0.08)

    def test_sum_within_tolerance(self):
        """Sum slightly off from 1.0 due to floating-point — should be accepted."""
        # 0.999999 is within the default tolerance of 1e-3.
        probs = validate_probabilities({"0": 0.249999, "1": 0.25, "2": 0.25, "3": 0.25})
        assert probs is not None

    def test_high_confidence_valid(self):
        """High confidence distribution is valid."""
        probs = validate_probabilities({"0": 0.02, "1": 0.03, "2": 0.92, "3": 0.03})
        assert math.isclose(probs.sum(), 1.0, abs_tol=1e-3)

    def test_uniform_valid(self):
        """Uniform distribution sums to exactly 1.0 — valid."""
        probs = validate_probabilities({"0": 0.25, "1": 0.25, "2": 0.25, "3": 0.25})
        assert math.isclose(probs.sum(), 1.0)


# ---------------------------------------------------------------------------
# Structural / format errors → InvalidInputFormatError
# ---------------------------------------------------------------------------


class TestStructuralErrors:
    """Structural input errors should raise InvalidInputFormatError."""

    def test_wrong_type_string(self):
        """A plain string is not a valid input."""
        with pytest.raises(InvalidInputFormatError):
            validate_probabilities("0.8, 0.1, 0.05, 0.05")

    def test_wrong_type_none(self):
        """None is not a valid input."""
        with pytest.raises(InvalidInputFormatError):
            validate_probabilities(None)

    def test_wrong_type_int(self):
        """A scalar integer is not a valid input."""
        with pytest.raises(InvalidInputFormatError):
            validate_probabilities(1)

    def test_missing_class_3_keys(self):
        """Only 3 classes provided — class 3 is missing."""
        with pytest.raises(InvalidInputFormatError, match="Missing probability classes"):
            validate_probabilities({"0": 0.20, "1": 0.30, "2": 0.50})

    def test_extra_class_5_keys(self):
        """5 classes provided — extra class 4."""
        with pytest.raises(InvalidInputFormatError, match="Unexpected probability classes"):
            validate_probabilities({"0": 0.2, "1": 0.2, "2": 0.2, "3": 0.2, "4": 0.2})

    def test_list_too_short(self):
        """List with fewer than 4 elements."""
        with pytest.raises(InvalidInputFormatError):
            validate_probabilities([0.3, 0.4, 0.3])

    def test_list_too_long(self):
        """List with more than 4 elements."""
        with pytest.raises(InvalidInputFormatError):
            validate_probabilities([0.2, 0.2, 0.2, 0.2, 0.2])

    def test_empty_dict(self):
        """Empty dict."""
        with pytest.raises(InvalidInputFormatError):
            validate_probabilities({})

    def test_empty_list(self):
        """Empty list."""
        with pytest.raises(InvalidInputFormatError):
            validate_probabilities([])

    def test_dict_non_digit_string_key(self):
        """Non-numeric string key like 'grade'."""
        with pytest.raises(InvalidInputFormatError):
            validate_probabilities({"grade": 0.5, "1": 0.2, "2": 0.2, "3": 0.1})

    def test_dict_invalid_key_type_float(self):
        """Float key — not supported."""
        with pytest.raises(InvalidInputFormatError):
            validate_probabilities({0.0: 0.25, 1.0: 0.25, 2.0: 0.25, 3.0: 0.25})

    def test_nested_list(self):
        """2-D nested list is not a valid 1-D sequence."""
        with pytest.raises(InvalidInputFormatError):
            validate_probabilities([[0.25, 0.25], [0.25, 0.25]])


# ---------------------------------------------------------------------------
# Probability value errors → InvalidProbabilityError
# ---------------------------------------------------------------------------


class TestProbabilityValueErrors:
    """Invalid probability values should raise InvalidProbabilityError."""

    def test_negative_probability(self):
        """Negative probability value."""
        with pytest.raises(InvalidProbabilityError, match="Negative"):
            validate_probabilities({"0": -0.10, "1": 0.30, "2": 0.40, "3": 0.40})

    def test_probability_greater_than_one(self):
        """Probability > 1.0 on one class."""
        with pytest.raises(InvalidProbabilityError, match="exceeding 1.0"):
            validate_probabilities({"0": 1.5, "1": 0.0, "2": 0.0, "3": 0.0})

    def test_nan_probability(self):
        """NaN in probability values."""
        with pytest.raises(InvalidProbabilityError, match="NaN"):
            validate_probabilities({"0": float("nan"), "1": 0.3, "2": 0.4, "3": 0.3})

    def test_positive_infinity(self):
        """Positive infinity in probability values."""
        with pytest.raises(InvalidProbabilityError, match="Infinite"):
            validate_probabilities({"0": float("inf"), "1": 0.3, "2": 0.3, "3": 0.4})

    def test_negative_infinity(self):
        """Negative infinity in probability values."""
        with pytest.raises(InvalidProbabilityError, match="Infinite"):
            validate_probabilities({"0": float("-inf"), "1": 0.3, "2": 0.3, "3": 0.4})

    def test_nan_in_list(self):
        """NaN in a list input."""
        with pytest.raises(InvalidProbabilityError, match="NaN"):
            validate_probabilities([float("nan"), 0.3, 0.4, 0.3])

    def test_sum_too_high(self):
        """Probability sum materially exceeds 1.0."""
        with pytest.raises(InvalidProbabilityError, match="sums to"):
            validate_probabilities({"0": 0.4, "1": 0.4, "2": 0.4, "3": 0.1})

    def test_sum_too_low(self):
        """Probability sum materially below 1.0."""
        with pytest.raises(InvalidProbabilityError, match="sums to"):
            validate_probabilities({"0": 0.1, "1": 0.1, "2": 0.1, "3": 0.1})

    def test_all_zeros(self):
        """All zero probabilities — sum is 0.0."""
        with pytest.raises(InvalidProbabilityError, match="sums to"):
            validate_probabilities({"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0})

    def test_string_probability_value(self):
        """Probability given as a string — numpy will raise on conversion."""
        # numpy will raise a ValueError when converting "high" to float.
        with pytest.raises(Exception):
            validate_probabilities({"0": "high", "1": 0.3, "2": 0.4, "3": 0.3})
