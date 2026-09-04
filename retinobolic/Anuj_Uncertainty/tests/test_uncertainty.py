"""
test_uncertainty.py - Deterministic unit and numerical tests for the core uncertainty engine.

Tests cover:
  1. All 7 required test cases from the specification.
  2. Exact numerical correctness via pytest.approx() for known distributions.
  3. Configurable threshold behavior.
  4. Margin calculation.
  5. Tie-breaking behavior (uniform distribution).
  6. API contract (output keys, types, bounds).
"""

import math
import pytest
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.uncertainty import (
    compute_shannon_entropy,
    compute_normalized_uncertainty,
    compute_probability_margin,
    determine_uncertainty_level,
    calculate_uncertainty,
)
from src.validation import InvalidProbabilityError
from src.config import UncertaintyConfig, DEFAULT_CONFIG


# ===========================================================================
# Constants for numerical verification
# ===========================================================================

LN4 = math.log(4)     # = ln(4) ~= 1.386294361
LN2 = math.log(2)     # = ln(2) ~= 0.693147181
APPROX_TOL = 1e-4     # tolerance for pytest.approx on normalized scores


# ===========================================================================
# 1. Shannon entropy — numerical correctness
# ===========================================================================

class TestShannonEntropy:

    def test_deterministic_distribution(self):
        """H([1, 0, 0, 0]) = 0 (no uncertainty)."""
        probs = np.array([1.0, 0.0, 0.0, 0.0])
        h = compute_shannon_entropy(probs)
        assert h == pytest.approx(0.0, abs=1e-10)

    def test_uniform_distribution_entropy(self):
        """H([0.25, 0.25, 0.25, 0.25]) = ln(4) (maximum entropy for 4 classes)."""
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        h = compute_shannon_entropy(probs)
        assert h == pytest.approx(LN4, abs=1e-10)

    def test_binary_uniform_entropy(self):
        """H([0.5, 0.5, 0, 0]) = ln(2) = 0.693..."""
        probs = np.array([0.5, 0.5, 0.0, 0.0])
        h = compute_shannon_entropy(probs)
        assert h == pytest.approx(LN2, abs=1e-10)

    def test_entropy_is_nonnegative(self):
        """Entropy must always be >= 0."""
        for probs in [
            [0.5, 0.5, 0.0, 0.0],
            [0.03, 0.08, 0.81, 0.08],
            [0.25, 0.25, 0.25, 0.25],
        ]:
            assert compute_shannon_entropy(np.array(probs)) >= 0.0

    def test_entropy_never_exceeds_log4(self):
        """Entropy must never exceed ln(4) for a 4-class problem."""
        for probs in [
            [0.5, 0.5, 0.0, 0.0],
            [0.03, 0.08, 0.81, 0.08],
            [0.25, 0.25, 0.25, 0.25],
        ]:
            assert compute_shannon_entropy(np.array(probs)) <= LN4 + 1e-9

    def test_zero_probabilities_handled_safely(self):
        """[1, 0, 0, 0] must not produce NaN or Infinity."""
        probs = np.array([1.0, 0.0, 0.0, 0.0])
        h = compute_shannon_entropy(probs)
        assert math.isfinite(h)

    def test_near_zero_probabilities_stable(self):
        """[0.997, 0.001, 0.001, 0.001] must be numerically stable."""
        probs = np.array([0.001, 0.001, 0.997, 0.001])
        h = compute_shannon_entropy(probs)
        assert math.isfinite(h)
        assert h >= 0.0


# ===========================================================================
# 2. Normalized uncertainty — numerical correctness
# ===========================================================================

class TestNormalizedUncertainty:

    def test_uniform_gives_1(self):
        """Uniform distribution must produce uncertainty = 1.0."""
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        u = compute_normalized_uncertainty(probs)
        assert u == pytest.approx(1.0, abs=1e-6)

    def test_deterministic_gives_0(self):
        """Deterministic distribution must produce uncertainty = 0.0."""
        probs = np.array([0.0, 0.0, 1.0, 0.0])
        u = compute_normalized_uncertainty(probs)
        assert u == pytest.approx(0.0, abs=1e-6)

    def test_binary_uniform_gives_half(self):
        """[0.5, 0.5, 0, 0]: U = ln(2) / ln(4) = 0.5 exactly."""
        probs = np.array([0.5, 0.5, 0.0, 0.0])
        u = compute_normalized_uncertainty(probs)
        assert u == pytest.approx(0.5, abs=1e-6)

    def test_result_in_unit_interval(self):
        """Normalized uncertainty must always be in [0, 1]."""
        test_cases = [
            [0.03, 0.08, 0.81, 0.08],
            [0.1, 0.15, 0.60, 0.15],
            [0.25, 0.25, 0.25, 0.25],
            [0.001, 0.001, 0.997, 0.001],
        ]
        for probs in test_cases:
            u = compute_normalized_uncertainty(np.array(probs))
            assert 0.0 <= u <= 1.0, f"Out of range for {probs}: {u}"


# ===========================================================================
# 3. Required test cases (spec sections 19-20)
# ===========================================================================

class TestRequiredCases:

    def test_1_low_uncertainty(self):
        """TEST 1: [0.02, 0.03, 0.92, 0.03] -> low uncertainty."""
        dr_result = {
            "grade": 2,
            "probabilities": {"0": 0.02, "1": 0.03, "2": 0.92, "3": 0.03},
        }
        result = calculate_uncertainty(dr_result)
        assert result["uncertainty_level"] == "low"
        assert result["review_recommended"] is False
        # Verify score is numerically in low zone
        assert result["uncertainty"] <= DEFAULT_CONFIG.LOW_UNCERTAINTY_MAX

    def test_2_medium_uncertainty(self):
        """
        TEST 2: Medium-range behavior.

        Probabilities [0.05, 0.10, 0.75, 0.10] produce U ~= 0.596, which
        lies within the medium band (0.35, 0.70) with default thresholds.

        NOTE: The spec's original example [0.10, 0.15, 0.60, 0.15] produces
        U ~= 0.798 (high), not medium — that was an informal approximation in
        the spec. The test uses mathematically verified probabilities instead.
        The spec requirement is that medium-range behavior is correctly mapped;
        the specific probabilities used here satisfy that requirement exactly.
        """
        dr_result = {
            "probabilities": {"0": 0.05, "1": 0.10, "2": 0.75, "3": 0.10},
        }
        result = calculate_uncertainty(dr_result)
        # Analytically: U ~= 0.596, which is in (0.35, 0.70)
        u = result["uncertainty"]
        assert DEFAULT_CONFIG.LOW_UNCERTAINTY_MAX < u < DEFAULT_CONFIG.HIGH_UNCERTAINTY_MIN, (
            f"Expected medium range (0.35 < U < 0.70), got uncertainty={u}"
        )
        assert result["uncertainty_level"] == "medium"
        assert result["review_recommended"] is False

    def test_3_maximum_uncertainty(self):
        """TEST 3: [0.25, 0.25, 0.25, 0.25] -> uncertainty ~= 1.0, high, review_recommended."""
        dr_result = {
            "probabilities": {"0": 0.25, "1": 0.25, "2": 0.25, "3": 0.25},
        }
        result = calculate_uncertainty(dr_result)
        assert result["uncertainty"] == pytest.approx(1.0, abs=APPROX_TOL)
        assert result["uncertainty_level"] == "high"
        assert result["review_recommended"] is True

    def test_3_maximum_uncertainty_tie_breaking(self):
        """Tie-breaking: uniform dist -> grade should be 0 (first argmax)."""
        dr_result = {
            "probabilities": {"0": 0.25, "1": 0.25, "2": 0.25, "3": 0.25},
        }
        result = calculate_uncertainty(dr_result)
        assert result["predicted_grade"] == 0  # first index wins on tie

    def test_4_strongly_concentrated(self):
        """TEST 4: [0.001, 0.001, 0.997, 0.001] -> uncertainty ~= 0."""
        dr_result = {
            "grade": 2,
            "probabilities": {"0": 0.001, "1": 0.001, "2": 0.997, "3": 0.001},
        }
        result = calculate_uncertainty(dr_result)
        assert result["uncertainty"] == pytest.approx(0.0, abs=0.05)
        assert result["uncertainty_level"] == "low"

    def test_5_invalid_sum(self):
        """TEST 5: [0.4, 0.4, 0.4, 0.1] sums to 1.3 -> validation error."""
        dr_result = {"probabilities": {"0": 0.4, "1": 0.4, "2": 0.4, "3": 0.1}}
        with pytest.raises(InvalidProbabilityError):
            calculate_uncertainty(dr_result)

    def test_6_negative_probability(self):
        """TEST 6: [-0.1, 0.3, 0.4, 0.4] -> validation error."""
        dr_result = {"probabilities": {"0": -0.1, "1": 0.3, "2": 0.4, "3": 0.4}}
        with pytest.raises(InvalidProbabilityError, match="negative"):
            calculate_uncertainty(dr_result)

    def test_7_nan_probability(self):
        """TEST 7a: NaN probability -> validation error."""
        dr_result = {"probabilities": {"0": float("nan"), "1": 0.33, "2": 0.33, "3": 0.34}}
        with pytest.raises(InvalidProbabilityError, match="NaN"):
            calculate_uncertainty(dr_result)

    def test_7_infinity_probability(self):
        """TEST 7b: Infinite probability -> validation error."""
        dr_result = {"probabilities": {"0": float("inf"), "1": 0.0, "2": 0.0, "3": 0.0}}
        with pytest.raises(InvalidProbabilityError, match="infinity"):
            calculate_uncertainty(dr_result)


# ===========================================================================
# 4. Probability margin (auxiliary signal)
# ===========================================================================

class TestProbabilityMargin:

    def test_margin_clear_prediction(self):
        """0.81 - 0.08 = 0.73 for [0.03, 0.08, 0.81, 0.08]."""
        probs = np.array([0.03, 0.08, 0.81, 0.08])
        margin = compute_probability_margin(probs)
        assert margin == pytest.approx(0.73, abs=1e-4)

    def test_margin_uniform(self):
        """Uniform distribution -> margin = 0 (tied top classes)."""
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        margin = compute_probability_margin(probs)
        assert margin == pytest.approx(0.0, abs=1e-6)

    def test_margin_binary_split(self):
        """[0.5, 0.5, 0, 0] -> margin = 0."""
        probs = np.array([0.5, 0.5, 0.0, 0.0])
        margin = compute_probability_margin(probs)
        assert margin == pytest.approx(0.0, abs=1e-6)

    def test_margin_deterministic(self):
        """[1, 0, 0, 0] -> margin = 1.0."""
        probs = np.array([1.0, 0.0, 0.0, 0.0])
        margin = compute_probability_margin(probs)
        assert margin == pytest.approx(1.0, abs=1e-6)

    def test_margin_in_result_dict(self):
        """calculate_uncertainty must include probability_margin in output."""
        dr_result = {
            "grade": 2,
            "probabilities": {"0": 0.03, "1": 0.08, "2": 0.81, "3": 0.08},
        }
        result = calculate_uncertainty(dr_result)
        assert "probability_margin" in result
        assert result["probability_margin"] == pytest.approx(0.73, abs=1e-4)


# ===========================================================================
# 5. Configurable thresholds
# ===========================================================================

class TestConfigurableThresholds:

    def test_custom_threshold_changes_level(self):
        """Using a very high LOW_UNCERTAINTY_MAX forces 'low' for medium scores."""
        permissive_config = UncertaintyConfig(
            LOW_UNCERTAINTY_MAX=0.99,
            HIGH_UNCERTAINTY_MIN=1.0,
        )
        # [0.10, 0.15, 0.60, 0.15] would normally be 'medium' with defaults
        dr_result = {"probabilities": {"0": 0.10, "1": 0.15, "2": 0.60, "3": 0.15}}
        result = calculate_uncertainty(dr_result, config=permissive_config)
        assert result["uncertainty_level"] == "low"
        assert result["review_recommended"] is False

    def test_strict_threshold_forces_high(self):
        """Using a very low HIGH_UNCERTAINTY_MIN forces 'high' for moderate scores."""
        strict_config = UncertaintyConfig(
            LOW_UNCERTAINTY_MAX=0.10,
            HIGH_UNCERTAINTY_MIN=0.20,
        )
        # LOW uncertainty case normally, but strict config makes it "high"
        dr_result = {"probabilities": {"0": 0.03, "1": 0.08, "2": 0.81, "3": 0.08}}
        result = calculate_uncertainty(dr_result, config=strict_config)
        # uncertainty ~= 0.31, which exceeds HIGH_UNCERTAINTY_MIN=0.20
        assert result["uncertainty_level"] == "high"
        assert result["review_recommended"] is True


# ===========================================================================
# 6. API contract — output structure
# ===========================================================================

class TestOutputContract:

    def _run(self, probs_dict):
        dr_result = {"grade": 2, "probabilities": probs_dict}
        return calculate_uncertainty(dr_result)

    def test_required_keys_present(self):
        result = self._run({"0": 0.03, "1": 0.08, "2": 0.81, "3": 0.08})
        for key in ["predicted_grade", "uncertainty", "uncertainty_level", "review_recommended", "probability_margin"]:
            assert key in result, f"Missing key: {key}"

    def test_predicted_grade_is_int(self):
        result = self._run({"0": 0.03, "1": 0.08, "2": 0.81, "3": 0.08})
        assert isinstance(result["predicted_grade"], int)

    def test_uncertainty_is_float(self):
        result = self._run({"0": 0.03, "1": 0.08, "2": 0.81, "3": 0.08})
        assert isinstance(result["uncertainty"], float)

    def test_uncertainty_in_unit_interval(self):
        result = self._run({"0": 0.03, "1": 0.08, "2": 0.81, "3": 0.08})
        assert 0.0 <= result["uncertainty"] <= 1.0

    def test_uncertainty_level_is_valid_string(self):
        result = self._run({"0": 0.03, "1": 0.08, "2": 0.81, "3": 0.08})
        assert result["uncertainty_level"] in ("low", "medium", "high")

    def test_review_recommended_is_bool(self):
        result = self._run({"0": 0.03, "1": 0.08, "2": 0.81, "3": 0.08})
        assert isinstance(result["review_recommended"], bool)

    def test_confidence_metadata_forwarded(self):
        """If confidence is provided, it should appear in the output dict."""
        dr_result = {"grade": 2, "probabilities": {"0": 0.03, "1": 0.08, "2": 0.81, "3": 0.08}}
        result = calculate_uncertainty(dr_result, confidence=0.81)
        assert "confidence" in result
        assert result["confidence"] == pytest.approx(0.81)

    def test_confidence_metadata_absent_when_not_provided(self):
        """If confidence is not provided, key must be absent (not None)."""
        dr_result = {"grade": 2, "probabilities": {"0": 0.03, "1": 0.08, "2": 0.81, "3": 0.08}}
        result = calculate_uncertainty(dr_result)
        assert "confidence" not in result


# ===========================================================================
# 7. Exact numerical spot-checks from the specification
# ===========================================================================

class TestNumericalSpotChecks:

    def test_example_output_from_spec(self):
        """
        Specification example:
            {"grade": 2, "probabilities": {"0": 0.03, "1": 0.08, "2": 0.81, "3": 0.08}}

        The spec's informal notation says 'uncertainty 31%' but the exact
        Shannon entropy normalized by ln(4) gives U ~= 0.4905 (medium).
        The spec's value was a rounded approximation for illustrative purposes.

        This test verifies the EXACT analytical value using pytest.approx.
        """
        dr_result = {
            "grade": 2,
            "probabilities": {"0": 0.03, "1": 0.08, "2": 0.81, "3": 0.08},
        }
        result = calculate_uncertainty(dr_result)
        # Analytically: H = -(0.03*ln0.03 + 0.08*ln0.08 + 0.81*ln0.81 + 0.08*ln0.08)
        expected_entropy = -(
            0.03 * math.log(0.03)
            + 0.08 * math.log(0.08)
            + 0.81 * math.log(0.81)
            + 0.08 * math.log(0.08)
        )
        expected_normalized = expected_entropy / LN4  # ~= 0.4905
        assert result["uncertainty"] == pytest.approx(expected_normalized, abs=APPROX_TOL)
        assert result["predicted_grade"] == 2
        # U ~= 0.49 is above LOW_UNCERTAINTY_MAX=0.35 -> "medium"
        assert result["uncertainty_level"] == "medium"
        assert result["probability_margin"] == pytest.approx(0.73, abs=1e-4)

    def test_binary_uniform_exact(self):
        """[0.5, 0.5, 0, 0]: normalized uncertainty = ln(2)/ln(4) = 0.5."""
        probs = np.array([0.5, 0.5, 0.0, 0.0])
        u = compute_normalized_uncertainty(probs)
        assert u == pytest.approx(LN2 / LN4, abs=1e-6)

    def test_no_nan_in_any_output(self):
        """No field in the result dict should ever be NaN or Infinity."""
        dr_result = {
            "probabilities": {"0": 0.03, "1": 0.08, "2": 0.81, "3": 0.08},
        }
        result = calculate_uncertainty(dr_result)
        for key, val in result.items():
            if isinstance(val, float):
                assert math.isfinite(val), f"Field {key!r} is not finite: {val}"
