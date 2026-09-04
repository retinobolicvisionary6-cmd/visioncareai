"""
tests/test_validation.py — Unit tests for src/common/validation.py

Tests the validation layer that guards all three upstream module outputs
before they enter the rules engine.

Coverage:
    - validate_dr_input: structure checks
    - validate_confidence_result: confidence, level, NaN, Inf, range, type
    - validate_uncertainty_result: uncertainty, level, review flag
    - validate_ood_result: ood bool, ood_status, ood_score
"""

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.validation import (
    ValidationError,
    validate_confidence_result,
    validate_dr_input,
    validate_ood_result,
    validate_uncertainty_result,
)


# ===========================================================================
# validate_dr_input
# ===========================================================================

class TestValidateDRInput:
    def test_valid_dr_result(self):
        dr = {"grade": 2, "probabilities": {"0": 0.03, "1": 0.08, "2": 0.81, "3": 0.08}}
        result = validate_dr_input(dr)
        assert result is dr  # returned unchanged

    def test_missing_probabilities(self):
        with pytest.raises(ValidationError, match="probabilities"):
            validate_dr_input({"grade": 2})

    def test_not_a_dict(self):
        with pytest.raises(ValidationError, match="dict"):
            validate_dr_input([0.03, 0.08, 0.81, 0.08])

    def test_none_input(self):
        with pytest.raises(ValidationError):
            validate_dr_input(None)


# ===========================================================================
# validate_confidence_result
# ===========================================================================

class TestValidateConfidenceResult:
    def test_valid_high_confidence(self, mock_confidence_high):
        result = validate_confidence_result(mock_confidence_high)
        assert result.confidence == 0.88
        assert result.confidence_level == "high"

    def test_valid_low_confidence(self, mock_confidence_low):
        result = validate_confidence_result(mock_confidence_low)
        assert result.confidence_level == "low"

    def test_missing_confidence_key(self):
        with pytest.raises(ValidationError, match="confidence"):
            validate_confidence_result({"confidence_level": "high"})

    def test_missing_level_key(self):
        with pytest.raises(ValidationError, match="confidence_level"):
            validate_confidence_result({"confidence": 0.80})

    def test_nan_confidence(self):
        with pytest.raises(ValidationError, match="NaN"):
            validate_confidence_result({"confidence": float("nan"), "confidence_level": "high"})

    def test_inf_confidence(self):
        with pytest.raises(ValidationError, match="Infinite"):
            validate_confidence_result({"confidence": float("inf"), "confidence_level": "high"})

    def test_negative_confidence(self):
        with pytest.raises(ValidationError, match="outside the valid range"):
            validate_confidence_result({"confidence": -0.1, "confidence_level": "low"})

    def test_confidence_above_one(self):
        with pytest.raises(ValidationError, match="outside the valid range"):
            validate_confidence_result({"confidence": 1.1, "confidence_level": "high"})

    def test_invalid_level_string(self):
        with pytest.raises(ValidationError, match="not a recognised value"):
            validate_confidence_result({"confidence": 0.8, "confidence_level": "very_high"})

    def test_non_dict_input(self):
        with pytest.raises(ValidationError, match="dict"):
            validate_confidence_result("high")

    def test_boundary_confidence_zero(self):
        result = validate_confidence_result({"confidence": 0.0, "confidence_level": "low"})
        assert result.confidence == 0.0

    def test_boundary_confidence_one(self):
        result = validate_confidence_result({"confidence": 1.0, "confidence_level": "high"})
        assert result.confidence == 1.0


# ===========================================================================
# validate_uncertainty_result
# ===========================================================================

class TestValidateUncertaintyResult:
    def test_valid_low_uncertainty(self, mock_uncertainty_low):
        result = validate_uncertainty_result(mock_uncertainty_low)
        assert result.uncertainty == 0.18
        assert result.uncertainty_level == "low"
        assert result.review_recommended is False

    def test_valid_high_uncertainty(self, mock_uncertainty_high):
        result = validate_uncertainty_result(mock_uncertainty_high)
        assert result.uncertainty_level == "high"
        assert result.review_recommended is True

    def test_missing_uncertainty_key(self):
        with pytest.raises(ValidationError, match="uncertainty"):
            validate_uncertainty_result({"uncertainty_level": "low", "review_recommended": False})

    def test_nan_uncertainty(self):
        with pytest.raises(ValidationError, match="NaN"):
            validate_uncertainty_result({
                "uncertainty": float("nan"),
                "uncertainty_level": "low",
                "review_recommended": False,
            })

    def test_uncertainty_out_of_range(self):
        with pytest.raises(ValidationError, match="outside the valid range"):
            validate_uncertainty_result({
                "uncertainty": 1.5,
                "uncertainty_level": "high",
                "review_recommended": True,
            })

    def test_invalid_level_string(self):
        with pytest.raises(ValidationError, match="not a recognised value"):
            validate_uncertainty_result({
                "uncertainty": 0.5,
                "uncertainty_level": "extreme",
                "review_recommended": False,
            })

    def test_review_recommended_not_bool(self):
        with pytest.raises(ValidationError, match="bool"):
            validate_uncertainty_result({
                "uncertainty": 0.5,
                "uncertainty_level": "medium",
                "review_recommended": "yes",
            })


# ===========================================================================
# validate_ood_result
# ===========================================================================

class TestValidateOODResult:
    def test_valid_in_distribution(self, mock_ood_in_distribution):
        result = validate_ood_result(mock_ood_in_distribution)
        assert result.ood is False
        assert result.ood_status == "in_distribution"

    def test_valid_ood(self, mock_ood_out_of_distribution):
        result = validate_ood_result(mock_ood_out_of_distribution)
        assert result.ood is True
        assert result.ood_status == "review_required"

    def test_missing_ood_key(self):
        with pytest.raises(ValidationError, match="ood"):
            validate_ood_result({"ood_status": "in_distribution", "ood_score": 1.0})

    def test_ood_not_bool(self):
        with pytest.raises(ValidationError, match="bool"):
            validate_ood_result({"ood": 1, "ood_status": "in_distribution", "ood_score": 1.0})

    def test_ood_score_nan(self):
        with pytest.raises(ValidationError, match="NaN"):
            validate_ood_result({"ood": False, "ood_status": "in_distribution", "ood_score": float("nan")})

    def test_ood_score_negative(self):
        with pytest.raises(ValidationError, match="negative"):
            validate_ood_result({"ood": False, "ood_status": "in_distribution", "ood_score": -0.5})

    def test_invalid_ood_status(self):
        with pytest.raises(ValidationError, match="not a recognised value"):
            validate_ood_result({"ood": False, "ood_status": "unknown", "ood_score": 1.0})

    def test_non_dict_input(self):
        with pytest.raises(ValidationError, match="dict"):
            validate_ood_result(False)
