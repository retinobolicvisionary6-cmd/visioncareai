"""
test_confidence.py — Unit tests for confidence calculation logic.

Tests calculate_confidence() correctness: predicted grade, confidence value,
confidence level, top-2 margin, and output contract behaviour.
"""

import math

import pytest

from src.confidence import (
    InvalidInputFormatError,
    InvalidProbabilityError,
    calculate_confidence,
)
from src.config import CLASS_MAPPING, CONFIDENCE_LEVEL_HIGH, CONFIDENCE_LEVEL_LOW, CONFIDENCE_LEVEL_MEDIUM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dr_result(probs: dict) -> dict:
    """Wrap probabilities in a minimal dr_result dict."""
    return {"grade": 0, "probabilities": probs}


# ---------------------------------------------------------------------------
# Test 1 — Normal prediction
# ---------------------------------------------------------------------------


class TestNormalPrediction:
    """Test case 1: [0.03, 0.08, 0.81, 0.08]"""

    def setup_method(self):
        self.result = calculate_confidence(
            {"grade": 2, "probabilities": {"0": 0.03, "1": 0.08, "2": 0.81, "3": 0.08}}
        )

    def test_predicted_grade(self):
        assert self.result["predicted_grade"] == 2

    def test_confidence_value(self):
        assert math.isclose(self.result["confidence"], 0.81, rel_tol=1e-9)

    def test_confidence_percent(self):
        assert math.isclose(self.result["confidence_percent"], 81.0, rel_tol=1e-6)

    def test_confidence_level(self):
        assert self.result["confidence_level"] == CONFIDENCE_LEVEL_HIGH

    def test_predicted_class_name(self):
        assert self.result["predicted_class_name"] == CLASS_MAPPING[2]

    def test_confidence_in_unit_range(self):
        """Canonical confidence must be stored in [0, 1], not as a percentage."""
        assert 0.0 <= self.result["confidence"] <= 1.0


# ---------------------------------------------------------------------------
# Test 2 — Strong confidence
# ---------------------------------------------------------------------------


class TestStrongConfidence:
    """Test case 2: [0.02, 0.03, 0.92, 0.03]"""

    def setup_method(self):
        self.result = calculate_confidence(
            _make_dr_result({"0": 0.02, "1": 0.03, "2": 0.92, "3": 0.03})
        )

    def test_confidence_value(self):
        assert math.isclose(self.result["confidence"], 0.92, rel_tol=1e-9)

    def test_predicted_grade(self):
        assert self.result["predicted_grade"] == 2

    def test_confidence_level(self):
        assert self.result["confidence_level"] == CONFIDENCE_LEVEL_HIGH

    def test_margin_is_large(self):
        """Strong prediction should have a large margin."""
        assert self.result["margin"] > 0.8


# ---------------------------------------------------------------------------
# Test 3 — Uniform probabilities
# ---------------------------------------------------------------------------


class TestUniformProbabilities:
    """Test case 3: [0.25, 0.25, 0.25, 0.25]"""

    def setup_method(self):
        self.result = calculate_confidence(
            _make_dr_result({"0": 0.25, "1": 0.25, "2": 0.25, "3": 0.25})
        )

    def test_confidence_value(self):
        assert math.isclose(self.result["confidence"], 0.25, rel_tol=1e-9)

    def test_confidence_level_is_low(self):
        assert self.result["confidence_level"] == CONFIDENCE_LEVEL_LOW

    def test_margin_is_zero(self):
        """Uniform distribution → margin between top-1 and top-2 is 0."""
        assert math.isclose(self.result["margin"], 0.0, abs_tol=1e-9)

    def test_confidence_not_interpreted_as_reliable(self):
        """Confidence = 0.25 — we document this should NOT be treated as reliable."""
        # This test enforces that 0.25 is indeed low (level = "low").
        assert self.result["confidence_level"] == CONFIDENCE_LEVEL_LOW


# ---------------------------------------------------------------------------
# Test 4 — Invalid sum
# ---------------------------------------------------------------------------


class TestInvalidProbabilitySum:
    """Test case 4: [0.4, 0.4, 0.4, 0.1] — sum = 1.3 → validation error."""

    def test_raises_invalid_probability_error(self):
        with pytest.raises(InvalidProbabilityError, match="sums to"):
            calculate_confidence(
                _make_dr_result({"0": 0.4, "1": 0.4, "2": 0.4, "3": 0.1})
            )


# ---------------------------------------------------------------------------
# Test 5 — Negative probability
# ---------------------------------------------------------------------------


class TestNegativeProbability:
    """Test case 5: [-0.1, 0.3, 0.4, 0.4] → validation error."""

    def test_raises_invalid_probability_error(self):
        with pytest.raises(InvalidProbabilityError, match="Negative"):
            calculate_confidence(
                _make_dr_result({"0": -0.1, "1": 0.3, "2": 0.4, "3": 0.4})
            )


# ---------------------------------------------------------------------------
# Test 6 — Missing class
# ---------------------------------------------------------------------------


class TestMissingClass:
    """Test case 6: only 3 classes provided → validation error."""

    def test_raises_invalid_input_format_error(self):
        with pytest.raises(InvalidInputFormatError, match="Missing probability classes"):
            calculate_confidence(
                _make_dr_result({"0": 0.2, "1": 0.3, "2": 0.5})
            )


# ---------------------------------------------------------------------------
# Test 7 — NaN / Infinity
# ---------------------------------------------------------------------------


class TestNaNAndInfinity:
    """Test case 7: NaN and Infinity values must fail safely."""

    def test_nan_raises(self):
        with pytest.raises(InvalidProbabilityError, match="NaN"):
            calculate_confidence(
                _make_dr_result({"0": float("nan"), "1": 0.3, "2": 0.4, "3": 0.3})
            )

    def test_positive_inf_raises(self):
        with pytest.raises(InvalidProbabilityError, match="Infinite"):
            calculate_confidence(
                _make_dr_result({"0": float("inf"), "1": 0.0, "2": 0.0, "3": 0.0})
            )

    def test_negative_inf_raises(self):
        with pytest.raises(InvalidProbabilityError, match="Infinite"):
            calculate_confidence(
                _make_dr_result({"0": float("-inf"), "1": 0.5, "2": 0.3, "3": 0.2})
            )


# ---------------------------------------------------------------------------
# Medium confidence level
# ---------------------------------------------------------------------------


class TestMediumConfidence:
    """Medium confidence: [0.10, 0.15, 0.60, 0.15]"""

    def setup_method(self):
        self.result = calculate_confidence(
            _make_dr_result({"0": 0.10, "1": 0.15, "2": 0.60, "3": 0.15})
        )

    def test_confidence_value(self):
        assert math.isclose(self.result["confidence"], 0.60, rel_tol=1e-9)

    def test_confidence_level(self):
        assert self.result["confidence_level"] == CONFIDENCE_LEVEL_MEDIUM

    def test_predicted_grade(self):
        assert self.result["predicted_grade"] == 2


# ---------------------------------------------------------------------------
# Low confidence margin test
# ---------------------------------------------------------------------------


class TestLowConfidenceMargin:
    """Low confidence: [0.24, 0.27, 0.25, 0.24]"""

    def setup_method(self):
        self.result = calculate_confidence(
            _make_dr_result({"0": 0.24, "1": 0.27, "2": 0.25, "3": 0.24})
        )

    def test_predicted_grade(self):
        assert self.result["predicted_grade"] == 1

    def test_confidence_value(self):
        assert math.isclose(self.result["confidence"], 0.27, rel_tol=1e-9)

    def test_confidence_level(self):
        assert self.result["confidence_level"] == CONFIDENCE_LEVEL_LOW

    def test_small_margin(self):
        """Very small margin reflects ambiguity between class 1 and class 2."""
        assert self.result["margin"] == pytest.approx(0.02, abs=1e-9)


# ---------------------------------------------------------------------------
# Class name mapping
# ---------------------------------------------------------------------------


class TestClassNameMapping:
    """Predicted class name must resolve from CLASS_MAPPING."""

    @pytest.mark.parametrize("grade, expected_name", [
        (0, "No DR"),
        (1, "Mild DR"),
        (2, "Moderate DR"),
        (3, "Severe/PDR"),
    ])
    def test_class_names(self, grade, expected_name):
        probs = [0.0, 0.0, 0.0, 0.0]
        probs[grade] = 1.0
        result = calculate_confidence(_make_dr_result(
            {str(i): p for i, p in enumerate(probs)}
        ))
        assert result["predicted_class_name"] == expected_name


# ---------------------------------------------------------------------------
# Non-mutation of input
# ---------------------------------------------------------------------------


class TestInputNonMutation:
    """calculate_confidence must NOT mutate the caller's dr_result object."""

    def test_original_dict_unchanged(self):
        original = {
            "grade": 2,
            "probabilities": {"0": 0.03, "1": 0.08, "2": 0.81, "3": 0.08},
            "gradcam_path": "outputs/gradcam/image_001.jpg",
        }
        original_copy = {
            "grade": original["grade"],
            "probabilities": dict(original["probabilities"]),
            "gradcam_path": original["gradcam_path"],
        }
        calculate_confidence(original)
        assert original == original_copy


# ---------------------------------------------------------------------------
# Optional top-2 inclusion/exclusion
# ---------------------------------------------------------------------------


class TestTop2Fields:
    """Top-2 fields present when include_top2=True, absent when False."""

    TOP2_KEYS = {"top_class", "top_probability", "second_class", "second_probability", "margin"}

    def test_top2_present_by_default(self):
        result = calculate_confidence(
            _make_dr_result({"0": 0.03, "1": 0.08, "2": 0.81, "3": 0.08})
        )
        for key in self.TOP2_KEYS:
            assert key in result, f"Expected key '{key}' in result"

    def test_top2_absent_when_disabled(self):
        result = calculate_confidence(
            _make_dr_result({"0": 0.03, "1": 0.08, "2": 0.81, "3": 0.08}),
            include_top2=False,
        )
        for key in self.TOP2_KEYS:
            assert key not in result, f"Key '{key}' should not be in result when include_top2=False"

    def test_top2_margin_correctness(self):
        result = calculate_confidence(
            _make_dr_result({"0": 0.03, "1": 0.08, "2": 0.81, "3": 0.08})
        )
        expected_margin = 0.81 - 0.08
        assert math.isclose(result["margin"], expected_margin, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# dr_result structural errors
# ---------------------------------------------------------------------------


class TestDrResultFormat:
    """calculate_confidence enforces dr_result structure."""

    def test_not_a_dict_raises(self):
        with pytest.raises(InvalidInputFormatError):
            calculate_confidence([0.03, 0.08, 0.81, 0.08])

    def test_missing_probabilities_key_raises(self):
        with pytest.raises(InvalidInputFormatError, match="'probabilities' key is missing"):
            calculate_confidence({"grade": 2})

    def test_gradcam_path_is_ignored(self):
        """gradcam_path in dr_result should not cause errors."""
        result = calculate_confidence({
            "grade": 2,
            "probabilities": {"0": 0.03, "1": 0.08, "2": 0.81, "3": 0.08},
            "gradcam_path": "outputs/gradcam/image_001.jpg",
        })
        assert result["predicted_grade"] == 2
