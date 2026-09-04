"""
tests/test_validation.py — Unit tests for src/validation.py.

Verifies that the input validators:
    - Accept valid upstream outputs
    - Reject invalid/missing required fields with clear errors
    - Handle missing optional fields gracefully
    - Never alter the DR grade or probabilities
    - Handle None/empty clinical context without creating risk signals
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make src importable from the tests directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.validation import (
    validate_quality_result,
    validate_dr_result,
    validate_reliability_result,
    validate_clinical_context,
    validate_gradcam_metadata,
    validate_all_inputs,
    QualityResult,
    DRResult,
    ReliabilityResult,
    ClinicalContext,
    DecisionValidationError,
)


# ============================================================================
# Quality result validation
# ============================================================================

class TestValidateQualityResult:

    def test_valid_good(self):
        data = {
            "status": "good", "quality_score": 0.88, "action": "continue",
            "reason": "Suitable.", "enhanced": False, "enhanced_image_path": None, "error": None,
        }
        result = validate_quality_result(data)
        assert isinstance(result, QualityResult)
        assert result.status == "good"
        assert result.quality_score == 0.88
        assert result.is_gradable is True
        assert result.is_ungradable is False

    def test_valid_ungradable(self):
        data = {
            "status": "ungradable", "quality_score": 0.20, "action": "recapture",
            "reason": "Too blurry.", "error": None,
        }
        result = validate_quality_result(data)
        assert result.status == "ungradable"
        assert result.is_ungradable is True
        assert result.is_gradable is False

    def test_valid_borderline(self):
        data = {
            "status": "borderline", "quality_score": 0.55, "action": "enhance_and_recheck",
            "reason": "Marginal quality.", "error": None,
        }
        result = validate_quality_result(data)
        assert result.status == "borderline"
        assert result.is_gradable is True  # borderline is still gradable

    def test_none_raises(self):
        with pytest.raises(DecisionValidationError, match="quality_result is None"):
            validate_quality_result(None)

    def test_non_dict_raises(self):
        with pytest.raises(DecisionValidationError, match="must be a dict"):
            validate_quality_result("bad_input")

    def test_missing_status_raises(self):
        with pytest.raises(DecisionValidationError, match="'status' is missing"):
            validate_quality_result({"quality_score": 0.8})

    def test_invalid_status_raises(self):
        with pytest.raises(DecisionValidationError, match="Must be one of"):
            validate_quality_result({"status": "invalid", "quality_score": 0.8})

    def test_quality_score_out_of_range(self):
        with pytest.raises(DecisionValidationError, match="outside \\[0, 1\\]"):
            validate_quality_result({"status": "good", "quality_score": 1.5})

    def test_quality_score_negative(self):
        with pytest.raises(DecisionValidationError, match="outside \\[0, 1\\]"):
            validate_quality_result({"status": "good", "quality_score": -0.1})

    def test_action_derived_from_status_when_missing(self):
        data = {"status": "ungradable", "quality_score": 0.2, "reason": "Blurry."}
        result = validate_quality_result(data)
        assert result.action == "recapture"

    def test_enhanced_fields_preserved(self):
        data = {
            "status": "good", "quality_score": 0.85, "action": "continue",
            "reason": "OK.", "enhanced": True,
            "enhanced_image_path": "outputs/enhanced/img.jpg",
        }
        result = validate_quality_result(data)
        assert result.enhanced is True
        assert result.enhanced_image_path == "outputs/enhanced/img.jpg"


# ============================================================================
# DR result validation
# ============================================================================

class TestValidateDRResult:

    def test_valid_grade_0_dict_probs(self):
        data = {
            "grade": 0,
            "probabilities": {"0": 0.92, "1": 0.05, "2": 0.02, "3": 0.01},
        }
        result = validate_dr_result(data)
        assert isinstance(result, DRResult)
        assert result.grade == 0
        assert result.class_name == "No DR"
        assert result.gradcam_path is None

    def test_valid_grade_3_list_probs(self):
        data = {
            "grade": 3,
            "probabilities": [0.01, 0.02, 0.07, 0.90],
        }
        result = validate_dr_result(data)
        assert result.grade == 3
        assert result.class_name == "Severe / PDR"

    def test_gradcam_path_preserved(self):
        data = {
            "grade": 2,
            "probabilities": {"0": 0.02, "1": 0.04, "2": 0.88, "3": 0.06},
            "gradcam_path": "outputs/gradcam/image.jpg",
        }
        result = validate_dr_result(data)
        assert result.gradcam_path == "outputs/gradcam/image.jpg"

    def test_none_raises(self):
        with pytest.raises(DecisionValidationError, match="dr_result is None"):
            validate_dr_result(None)

    def test_missing_grade_raises(self):
        with pytest.raises(DecisionValidationError, match="'grade' is missing"):
            validate_dr_result({"probabilities": {"0": 0.9, "1": 0.05, "2": 0.03, "3": 0.02}})

    def test_invalid_grade_minus_1(self):
        with pytest.raises(DecisionValidationError, match="not a valid DR grade"):
            validate_dr_result({
                "grade": -1,
                "probabilities": {"0": 0.9, "1": 0.05, "2": 0.03, "3": 0.02},
            })

    def test_invalid_grade_5(self):
        with pytest.raises(DecisionValidationError, match="not a valid DR grade"):
            validate_dr_result({
                "grade": 5,
                "probabilities": {"0": 0.9, "1": 0.05, "2": 0.03, "3": 0.02},
            })

    def test_missing_probabilities_raises(self):
        with pytest.raises(DecisionValidationError, match="'probabilities' is missing"):
            validate_dr_result({"grade": 2})

    def test_wrong_prob_count_list(self):
        with pytest.raises(DecisionValidationError, match="exactly 4 elements"):
            validate_dr_result({"grade": 1, "probabilities": [0.5, 0.5]})

    def test_prob_sum_too_far_from_1(self):
        with pytest.raises(DecisionValidationError, match="sum"):
            validate_dr_result({
                "grade": 1,
                "probabilities": {"0": 0.5, "1": 0.5, "2": 0.5, "3": 0.5},
            })

    def test_prob_negative_value(self):
        with pytest.raises(DecisionValidationError, match="outside \\[0, 1\\]"):
            validate_dr_result({
                "grade": 0,
                "probabilities": {"0": -0.1, "1": 0.5, "2": 0.3, "3": 0.3},
            })

    def test_grade_not_altered(self):
        """The validator must NOT modify the DR grade."""
        data = {
            "grade": 2,
            "probabilities": {"0": 0.02, "1": 0.04, "2": 0.88, "3": 0.06},
        }
        result = validate_dr_result(data)
        assert result.grade == 2  # unchanged


# ============================================================================
# Reliability result validation
# ============================================================================

class TestValidateReliabilityResult:

    def _valid_reliable(self):
        return {
            "reliability_status": "acceptable",
            "review_required": False,
            "reason": "High model confidence, low uncertainty and in-distribution input.",
            "confidence": 0.92,
            "confidence_level": "high",
            "uncertainty": 0.18,
            "uncertainty_level": "low",
            "ood": False,
            "ood_status": "in_distribution",
            "ood_score": 1.42,
            "reliability_score": 0.87,
            "predicted_grade": 0,
            "predicted_class_name": "No DR",
        }

    def test_valid_acceptable(self):
        result = validate_reliability_result(self._valid_reliable())
        assert isinstance(result, ReliabilityResult)
        assert result.reliability_status == "acceptable"
        assert result.is_acceptable is True
        assert result.is_review_required is False
        assert result.ood is False

    def test_valid_review_required(self):
        data = self._valid_reliable()
        data["reliability_status"] = "review_required"
        data["review_required"] = True
        data["ood"] = True
        result = validate_reliability_result(data)
        assert result.is_review_required is True
        assert result.ood is True

    def test_valid_caution(self):
        data = self._valid_reliable()
        data["reliability_status"] = "caution"
        data["confidence_level"] = "medium"
        data["uncertainty_level"] = "medium"
        result = validate_reliability_result(data)
        assert result.is_caution is True

    def test_none_raises(self):
        with pytest.raises(DecisionValidationError, match="reliability_result is None"):
            validate_reliability_result(None)

    def test_missing_reliability_status(self):
        data = self._valid_reliable()
        del data["reliability_status"]
        with pytest.raises(DecisionValidationError, match="'reliability_status' is missing"):
            validate_reliability_result(data)

    def test_invalid_reliability_status(self):
        data = self._valid_reliable()
        data["reliability_status"] = "unknown_status"
        with pytest.raises(DecisionValidationError, match="Must be one of"):
            validate_reliability_result(data)

    def test_confidence_out_of_range(self):
        data = self._valid_reliable()
        data["confidence"] = 1.5
        with pytest.raises(DecisionValidationError, match="outside \\[0, 1\\]"):
            validate_reliability_result(data)

    def test_missing_ood_raises(self):
        data = self._valid_reliable()
        del data["ood"]
        with pytest.raises(DecisionValidationError, match="'ood' is missing"):
            validate_reliability_result(data)

    def test_optional_reliability_score_missing_ok(self):
        data = self._valid_reliable()
        del data["reliability_score"]
        result = validate_reliability_result(data)
        assert result.reliability_score is None


# ============================================================================
# Clinical context validation
# ============================================================================

class TestValidateClinicalContext:

    def test_none_returns_gracefully(self):
        result = validate_clinical_context(None)
        assert isinstance(result, ClinicalContext)
        assert result.clinical_context_complete is False
        assert result.validation_passed is True  # Not failed, just not provided

    def test_empty_dict_returns_gracefully(self):
        result = validate_clinical_context({})
        assert result.clinical_context_complete is False
        assert result.validation_passed is True

    def test_invalid_type_returns_gracefully(self):
        result = validate_clinical_context("not_a_dict")
        assert result.validation_passed is False
        assert result.clinical_context_complete is False

    def test_valid_complete_context(self):
        data = {
            "validation_passed": True,
            "validation_errors": [],
            "clinical_context": {
                "age": {"value": 55, "unit": "years", "status": "provided"},
                "hba1c": {"value": 7.8, "unit": "%", "status": "provided"},
            },
            "data_quality": {
                "complete": True,
                "clinical_context_complete": True,
                "missing_fields": [],
            }
        }
        result = validate_clinical_context(data)
        assert result.clinical_context_complete is True
        assert result.validation_passed is True

    def test_missing_clinical_context_does_not_create_high_risk(self):
        """
        Missing clinical data must NEVER automatically create a high-risk flag.
        This is a critical safety requirement.
        """
        result = validate_clinical_context(None)
        # The clinical context object carries no risk escalation information
        assert result.clinical_context_complete is False
        assert result.hba1c is None  # Missing data, not fabricated
        assert result.age is None
        assert result.clinical_data is None

    def test_age_accessor(self):
        data = {
            "validation_passed": True,
            "validation_errors": [],
            "clinical_context": {"age": {"value": 60, "unit": "years", "status": "provided"}},
            "data_quality": {"complete": False, "clinical_context_complete": False, "missing_fields": ["hba1c"]},
        }
        result = validate_clinical_context(data)
        assert result.age == 60

    def test_hba1c_accessor(self):
        data = {
            "validation_passed": True,
            "validation_errors": [],
            "clinical_context": {"hba1c": {"value": 8.5, "unit": "%", "status": "provided"}},
            "data_quality": {"complete": False, "clinical_context_complete": False, "missing_fields": []},
        }
        result = validate_clinical_context(data)
        assert result.hba1c == 8.5

    def test_validation_failed_context(self):
        data = {
            "validation_passed": False,
            "validation_errors": ["age: invalid value"],
            "clinical_context": None,
            "data_quality": {"complete": False, "clinical_context_complete": False, "missing_fields": []},
        }
        result = validate_clinical_context(data)
        assert result.validation_passed is False
        assert len(result.validation_errors) == 1
        assert result.clinical_context_complete is False


# ============================================================================
# Gradcam metadata validation
# ============================================================================

class TestValidateGradcamMetadata:

    def test_no_path_anywhere(self):
        result = validate_gradcam_metadata(
            dr_result_raw={}, reliability_result_raw={},
            explicit_gradcam_path=None, warn_if_missing=False,
        )
        assert result.gradcam_path is None
        assert result.gradcam_exists is False

    def test_explicit_path_takes_priority(self):
        result = validate_gradcam_metadata(
            dr_result_raw={"gradcam_path": "from_dr.jpg"},
            reliability_result_raw={},
            explicit_gradcam_path="explicit.jpg",
        )
        assert result.gradcam_path == "explicit.jpg"

    def test_dr_result_path_used_when_no_explicit(self):
        result = validate_gradcam_metadata(
            dr_result_raw={"gradcam_path": "from_dr.jpg"},
            reliability_result_raw={},
            explicit_gradcam_path=None,
        )
        assert result.gradcam_path == "from_dr.jpg"

    def test_nonexistent_path_does_not_raise(self):
        result = validate_gradcam_metadata(
            dr_result_raw={"gradcam_path": "nonexistent/path.jpg"},
            reliability_result_raw={},
            explicit_gradcam_path=None,
        )
        assert result.gradcam_path == "nonexistent/path.jpg"
        assert result.gradcam_exists is False  # File doesn't exist on disk


# ============================================================================
# validate_all_inputs integration
# ============================================================================

class TestValidateAllInputs:

    def _make_inputs(self):
        quality = {
            "status": "good", "quality_score": 0.88,
            "action": "continue", "reason": "OK.", "error": None,
        }
        dr = {
            "grade": 0, "probabilities": {"0": 0.92, "1": 0.05, "2": 0.02, "3": 0.01},
        }
        rel = {
            "reliability_status": "acceptable", "review_required": False,
            "reason": "Acceptable.", "confidence": 0.92, "confidence_level": "high",
            "uncertainty": 0.18, "uncertainty_level": "low",
            "ood": False, "ood_status": "in_distribution", "ood_score": 1.42,
        }
        return quality, dr, rel

    def test_valid_complete_inputs(self):
        q, d, r = self._make_inputs()
        inputs = validate_all_inputs(q, d, r)
        assert inputs.quality.status == "good"
        assert inputs.dr.grade == 0
        assert inputs.reliability.reliability_status == "acceptable"
        assert inputs.clinical.clinical_context_complete is False

    def test_missing_quality_raises(self):
        _, d, r = self._make_inputs()
        with pytest.raises(DecisionValidationError):
            validate_all_inputs(None, d, r)

    def test_missing_dr_raises(self):
        q, _, r = self._make_inputs()
        with pytest.raises(DecisionValidationError):
            validate_all_inputs(q, None, r)

    def test_missing_reliability_raises(self):
        q, d, _ = self._make_inputs()
        with pytest.raises(DecisionValidationError):
            validate_all_inputs(q, d, None)

    def test_none_clinical_context_does_not_raise(self):
        q, d, r = self._make_inputs()
        inputs = validate_all_inputs(q, d, r, clinical_context=None)
        assert inputs.clinical.clinical_context_complete is False
