"""
tests/test_reliability.py — Unit tests for src/reliability/fusion.py

Tests calculate_reliability() directly, using mock module outputs.
No real module calls — exercises only the Reliability Engine fusion
and the optional reliability_score computation.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reliability.fusion import calculate_reliability


# ---------------------------------------------------------------------------
# Test 1: Fully reliable (acceptable)
# ---------------------------------------------------------------------------

class TestAcceptableCase:
    def test_full_reliability_acceptable(
        self, mock_confidence_high, mock_uncertainty_low, mock_ood_in_distribution
    ):
        result = calculate_reliability(
            mock_confidence_high, mock_uncertainty_low, mock_ood_in_distribution
        )
        assert result["reliability_status"] == "acceptable"
        assert result["review_required"] is False
        assert "High model confidence" in result["reason"]

    def test_output_contains_all_required_keys(
        self, mock_confidence_high, mock_uncertainty_low, mock_ood_in_distribution
    ):
        result = calculate_reliability(
            mock_confidence_high, mock_uncertainty_low, mock_ood_in_distribution
        )
        required = [
            "reliability_status", "review_required", "reason",
            "confidence", "confidence_level",
            "uncertainty", "uncertainty_level",
            "ood", "ood_status", "ood_score",
        ]
        for key in required:
            assert key in result, f"Missing required key: '{key}'"

    def test_original_signals_preserved(
        self, mock_confidence_high, mock_uncertainty_low, mock_ood_in_distribution
    ):
        result = calculate_reliability(
            mock_confidence_high, mock_uncertainty_low, mock_ood_in_distribution
        )
        assert result["confidence"] == 0.88
        assert result["uncertainty"] == 0.18
        assert result["ood"] is False


# ---------------------------------------------------------------------------
# Test 2: High uncertainty → review_required
# ---------------------------------------------------------------------------

class TestHighUncertaintyCase:
    def test_high_uncertainty_triggers_review(
        self, mock_confidence_high, mock_uncertainty_high, mock_ood_in_distribution
    ):
        result = calculate_reliability(
            mock_confidence_high, mock_uncertainty_high, mock_ood_in_distribution
        )
        assert result["reliability_status"] == "review_required"
        assert result["review_required"] is True
        assert "uncertainty" in result["reason"].lower()


# ---------------------------------------------------------------------------
# Test 3: OOD despite high confidence → review_required
# ---------------------------------------------------------------------------

class TestOODOverridesHighConfidence:
    def test_ood_overrides_high_confidence(
        self, mock_confidence_high, mock_uncertainty_low, mock_ood_out_of_distribution
    ):
        result = calculate_reliability(
            mock_confidence_high, mock_uncertainty_low, mock_ood_out_of_distribution
        )
        assert result["reliability_status"] == "review_required"
        assert result["review_required"] is True
        assert "outside the configured reference distribution" in result["reason"]

    def test_ood_score_preserved_in_output(
        self, mock_confidence_high, mock_uncertainty_low, mock_ood_out_of_distribution
    ):
        result = calculate_reliability(
            mock_confidence_high, mock_uncertainty_low, mock_ood_out_of_distribution
        )
        assert result["ood_score"] == 5.80
        assert result["ood"] is True


# ---------------------------------------------------------------------------
# Test 4: Low confidence
# ---------------------------------------------------------------------------

class TestLowConfidenceCase:
    def test_low_confidence_triggers_review(
        self, mock_confidence_low, mock_uncertainty_medium, mock_ood_in_distribution
    ):
        result = calculate_reliability(
            mock_confidence_low, mock_uncertainty_medium, mock_ood_in_distribution
        )
        assert result["reliability_status"] == "review_required"
        assert result["review_required"] is True

    def test_low_confidence_low_uncertainty_review(
        self, mock_confidence_low, mock_uncertainty_low, mock_ood_in_distribution
    ):
        """Low confidence alone (with low uncertainty) → review_required."""
        result = calculate_reliability(
            mock_confidence_low, mock_uncertainty_low, mock_ood_in_distribution
        )
        assert result["reliability_status"] == "review_required"
        assert result["review_required"] is True


# ---------------------------------------------------------------------------
# Test 5: Multiple failures
# ---------------------------------------------------------------------------

class TestMultipleFailures:
    def test_all_failures_review_required(
        self, mock_confidence_low, mock_uncertainty_high, mock_ood_out_of_distribution
    ):
        result = calculate_reliability(
            mock_confidence_low, mock_uncertainty_high, mock_ood_out_of_distribution
        )
        assert result["reliability_status"] == "review_required"
        assert result["review_required"] is True
        assert len(result["reason"]) > 0


# ---------------------------------------------------------------------------
# Caution case
# ---------------------------------------------------------------------------

class TestCautionCase:
    def test_medium_signals_produce_caution(
        self, mock_confidence_medium, mock_uncertainty_medium, mock_ood_in_distribution
    ):
        result = calculate_reliability(
            mock_confidence_medium, mock_uncertainty_medium, mock_ood_in_distribution
        )
        assert result["reliability_status"] == "caution"
        assert result["review_required"] is False


# ---------------------------------------------------------------------------
# Optional reliability_score
# ---------------------------------------------------------------------------

class TestReliabilityScore:
    def test_score_present_by_default(
        self, mock_confidence_high, mock_uncertainty_low, mock_ood_in_distribution
    ):
        result = calculate_reliability(
            mock_confidence_high, mock_uncertainty_low, mock_ood_in_distribution
        )
        assert "reliability_score" in result
        score = result["reliability_score"]
        assert 0.0 <= score <= 1.0

    def test_score_zero_when_ood(
        self, mock_confidence_high, mock_uncertainty_low, mock_ood_out_of_distribution
    ):
        """OOD = True → reliability_score = 0.0."""
        result = calculate_reliability(
            mock_confidence_high, mock_uncertainty_low, mock_ood_out_of_distribution
        )
        assert result["reliability_score"] == 0.0

    def test_score_absent_when_disabled(
        self, mock_confidence_high, mock_uncertainty_low, mock_ood_in_distribution
    ):
        result = calculate_reliability(
            mock_confidence_high, mock_uncertainty_low, mock_ood_in_distribution,
            include_score=False,
        )
        assert "reliability_score" not in result

    def test_score_not_overriding_status(
        self, mock_confidence_high, mock_uncertainty_low, mock_ood_out_of_distribution
    ):
        """High reliability_score cannot make OOD case acceptable."""
        result = calculate_reliability(
            mock_confidence_high, mock_uncertainty_low, mock_ood_out_of_distribution
        )
        # Score is 0.0 for OOD, and status is still review_required
        assert result["reliability_status"] == "review_required"
        assert result["reliability_score"] == 0.0

    def test_acceptable_score_is_high(
        self, mock_confidence_high, mock_uncertainty_low, mock_ood_in_distribution
    ):
        """High confidence + low uncertainty + in-dist → score should be > 0.7."""
        result = calculate_reliability(
            mock_confidence_high, mock_uncertainty_low, mock_ood_in_distribution
        )
        assert result["reliability_score"] > 0.7


# ---------------------------------------------------------------------------
# Validation propagation
# ---------------------------------------------------------------------------

class TestValidationPropagation:
    def test_invalid_confidence_result_raises(
        self, mock_uncertainty_low, mock_ood_in_distribution
    ):
        from src.common.validation import ValidationError
        with pytest.raises(ValidationError):
            calculate_reliability(
                {"confidence": float("nan"), "confidence_level": "high"},
                mock_uncertainty_low,
                mock_ood_in_distribution,
            )

    def test_invalid_ood_result_raises(
        self, mock_confidence_high, mock_uncertainty_low
    ):
        from src.common.validation import ValidationError
        with pytest.raises(ValidationError):
            calculate_reliability(
                mock_confidence_high,
                mock_uncertainty_low,
                {"ood": "yes", "ood_status": "in_distribution", "ood_score": 1.0},
            )
