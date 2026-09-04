"""
tests/test_engine.py — Integration and public API tests for the Decision Layer engine.

Tests:
    - make_final_decision() public API
    - run_screening_decision() API equivalence
    - Integration with sample data files
    - Custom policy overrides
    - Priority module: Grade 3 → urgent
    - Reason generator: non-diagnostic content checks
    - Error handling: DecisionValidationError propagation
    - End-to-end pipeline simulation (mimics full project flow)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.engine import make_final_decision, run_screening_decision
from src.validation import DecisionValidationError
from src.config import DecisionPolicy, load_policy

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"


def load_sample(name: str) -> Dict[str, Any]:
    with open(SAMPLE_DIR / name, encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# run_screening_decision equivalence
# ============================================================================

class TestRunScreeningDecision:

    def test_run_is_alias_of_make(self):
        """run_screening_decision must produce identical output to make_final_decision."""
        q = {"status": "good", "quality_score": 0.88, "action": "continue", "reason": "OK."}
        d = {"grade": 0, "probabilities": {"0": 0.92, "1": 0.05, "2": 0.02, "3": 0.01}}
        r = {
            "reliability_status": "acceptable", "review_required": False,
            "reason": "Acceptable.", "confidence": 0.92, "confidence_level": "high",
            "uncertainty": 0.18, "uncertainty_level": "low",
            "ood": False, "ood_status": "in_distribution", "ood_score": 1.42,
        }
        result_make = make_final_decision(q, d, r)
        result_run = run_screening_decision(q, d, r)
        assert result_make == result_run


# ============================================================================
# Sample data integration tests
# ============================================================================

class TestSampleDataIntegration:

    def _run_sample(self, filename: str) -> Dict[str, Any]:
        data = load_sample(filename)
        return make_final_decision(
            quality_result=data["quality_result"],
            dr_result=data["dr_result"],
            reliability_result=data["reliability_result"],
            clinical_context=data.get("clinical_context"),
        )

    def test_routine_sample(self):
        data = load_sample("routine.json")
        result = self._run_sample("routine.json")
        assert result["action"] == data["expected_output"]["action"]
        assert result["priority"] == data["expected_output"]["priority"]

    def test_recapture_sample(self):
        data = load_sample("recapture.json")
        result = self._run_sample("recapture.json")
        assert result["action"] == data["expected_output"]["action"]
        assert result["priority"] == data["expected_output"]["priority"]

    def test_review_sample(self):
        data = load_sample("review.json")
        result = self._run_sample("review.json")
        assert result["action"] == data["expected_output"]["action"]
        assert result["priority"] == data["expected_output"]["priority"]

    def test_refer_sample(self):
        data = load_sample("refer.json")
        result = self._run_sample("refer.json")
        assert result["action"] == data["expected_output"]["action"]
        assert result["priority"] == data["expected_output"]["priority"]

    def test_conflict_cases(self):
        conflict_data = load_sample("conflict_cases.json")
        for case in conflict_data:
            result = make_final_decision(
                quality_result=case["quality_result"],
                dr_result=case["dr_result"],
                reliability_result=case["reliability_result"],
                clinical_context=case.get("clinical_context"),
            )
            expected = case["expected_output"]
            assert result["action"] == expected["action"], (
                f"Case '{case.get('_name', '?')}': "
                f"expected action='{expected['action']}', got '{result['action']}'"
            )
            assert result["priority"] == expected["priority"], (
                f"Case '{case.get('_name', '?')}': "
                f"expected priority='{expected['priority']}', got '{result['priority']}'"
            )


# ============================================================================
# Custom policy tests
# ============================================================================

class TestCustomPolicy:

    def test_custom_threshold_grade_1_refers(self):
        """Override threshold to 1 → Grade 1 should refer."""
        policy = DecisionPolicy()
        policy.referral.referable_grade_threshold = 1

        q = {"status": "good", "quality_score": 0.88, "action": "continue", "reason": "OK."}
        d = {"grade": 1, "probabilities": {"0": 0.02, "1": 0.93, "2": 0.03, "3": 0.02}}
        r = {
            "reliability_status": "acceptable", "review_required": False,
            "reason": "Acceptable.", "confidence": 0.93, "confidence_level": "high",
            "uncertainty": 0.12, "uncertainty_level": "low",
            "ood": False, "ood_status": "in_distribution", "ood_score": 1.50,
        }
        result = make_final_decision(q, d, r, policy=policy)
        assert result["action"] == "refer"

    def test_custom_threshold_grade_3_only_refers(self):
        """Override threshold to 3 → Grade 2 should NOT refer (routine)."""
        policy = DecisionPolicy()
        policy.referral.referable_grade_threshold = 3
        policy.referral.urgent_referral_grades = []

        q = {"status": "good", "quality_score": 0.88, "action": "continue", "reason": "OK."}
        d = {"grade": 2, "probabilities": {"0": 0.02, "1": 0.04, "2": 0.88, "3": 0.06}}
        r = {
            "reliability_status": "acceptable", "review_required": False,
            "reason": "Acceptable.", "confidence": 0.88, "confidence_level": "high",
            "uncertainty": 0.20, "uncertainty_level": "low",
            "ood": False, "ood_status": "in_distribution", "ood_score": 1.55,
        }
        result = make_final_decision(q, d, r, policy=policy)
        assert result["action"] == "routine"

    def test_caution_allows_referral_policy(self):
        """When caution_allows_referral=True, caution status permits referral."""
        policy = DecisionPolicy()
        policy.reliability.caution_allows_referral = True

        q = {"status": "good", "quality_score": 0.85, "action": "continue", "reason": "OK."}
        d = {"grade": 2, "probabilities": {"0": 0.02, "1": 0.04, "2": 0.88, "3": 0.06}}
        r = {
            "reliability_status": "caution", "review_required": False,
            "reason": "Medium confidence.", "confidence": 0.65, "confidence_level": "medium",
            "uncertainty": 0.52, "uncertainty_level": "medium",
            "ood": False, "ood_status": "in_distribution", "ood_score": 1.80,
        }
        result = make_final_decision(q, d, r, policy=policy)
        assert result["action"] == "refer"


# ============================================================================
# Priority tests
# ============================================================================

class TestPriority:

    def test_grade3_urgent_by_default(self):
        q = {"status": "good", "quality_score": 0.88, "action": "continue", "reason": "OK."}
        d = {"grade": 3, "probabilities": {"0": 0.01, "1": 0.01, "2": 0.01, "3": 0.97}}
        r = {
            "reliability_status": "acceptable", "review_required": False,
            "reason": "Acceptable.", "confidence": 0.97, "confidence_level": "high",
            "uncertainty": 0.08, "uncertainty_level": "low",
            "ood": False, "ood_status": "in_distribution", "ood_score": 1.30,
        }
        result = make_final_decision(q, d, r)
        assert result["action"] == "refer"
        assert result["priority"] == "urgent"

    def test_grade2_high_priority(self):
        q = {"status": "good", "quality_score": 0.88, "action": "continue", "reason": "OK."}
        d = {"grade": 2, "probabilities": {"0": 0.02, "1": 0.04, "2": 0.88, "3": 0.06}}
        r = {
            "reliability_status": "acceptable", "review_required": False,
            "reason": "Acceptable.", "confidence": 0.88, "confidence_level": "high",
            "uncertainty": 0.22, "uncertainty_level": "low",
            "ood": False, "ood_status": "in_distribution", "ood_score": 1.55,
        }
        result = make_final_decision(q, d, r)
        assert result["priority"] == "high"

    def test_routine_low_priority(self):
        q = {"status": "good", "quality_score": 0.88, "action": "continue", "reason": "OK."}
        d = {"grade": 0, "probabilities": {"0": 0.92, "1": 0.05, "2": 0.02, "3": 0.01}}
        r = {
            "reliability_status": "acceptable", "review_required": False,
            "reason": "Acceptable.", "confidence": 0.92, "confidence_level": "high",
            "uncertainty": 0.18, "uncertainty_level": "low",
            "ood": False, "ood_status": "in_distribution", "ood_score": 1.42,
        }
        result = make_final_decision(q, d, r)
        assert result["priority"] == "low"

    def test_recapture_medium_priority(self):
        q = {"status": "ungradable", "quality_score": 0.20, "action": "recapture", "reason": "Blurry."}
        d = {"grade": 0, "probabilities": {"0": 0.92, "1": 0.05, "2": 0.02, "3": 0.01}}
        r = {
            "reliability_status": "acceptable", "review_required": False,
            "reason": "Acceptable.", "confidence": 0.92, "confidence_level": "high",
            "uncertainty": 0.18, "uncertainty_level": "low",
            "ood": False, "ood_status": "in_distribution", "ood_score": 1.42,
        }
        result = make_final_decision(q, d, r)
        assert result["priority"] == "medium"

    def test_doctor_review_high_priority(self):
        q = {"status": "good", "quality_score": 0.85, "action": "continue", "reason": "OK."}
        d = {"grade": 1, "probabilities": {"0": 0.05, "1": 0.65, "2": 0.20, "3": 0.10}}
        r = {
            "reliability_status": "review_required", "review_required": True,
            "reason": "OOD.", "confidence": 0.65, "confidence_level": "medium",
            "uncertainty": 0.52, "uncertainty_level": "medium",
            "ood": True, "ood_status": "review_required", "ood_score": 5.80,
        }
        result = make_final_decision(q, d, r)
        assert result["priority"] == "high"


# ============================================================================
# Error handling
# ============================================================================

class TestErrorHandling:

    def test_missing_quality_raises_validation_error(self):
        d = {"grade": 0, "probabilities": {"0": 0.92, "1": 0.05, "2": 0.02, "3": 0.01}}
        r = {
            "reliability_status": "acceptable", "review_required": False,
            "reason": "OK.", "confidence": 0.92, "confidence_level": "high",
            "uncertainty": 0.18, "uncertainty_level": "low",
            "ood": False, "ood_status": "in_distribution", "ood_score": 1.42,
        }
        with pytest.raises(DecisionValidationError):
            make_final_decision(None, d, r)

    def test_missing_dr_raises_validation_error(self):
        q = {"status": "good", "quality_score": 0.88, "action": "continue", "reason": "OK."}
        r = {
            "reliability_status": "acceptable", "review_required": False,
            "reason": "OK.", "confidence": 0.92, "confidence_level": "high",
            "uncertainty": 0.18, "uncertainty_level": "low",
            "ood": False, "ood_status": "in_distribution", "ood_score": 1.42,
        }
        with pytest.raises(DecisionValidationError):
            make_final_decision(q, None, r)

    def test_missing_reliability_raises_validation_error(self):
        q = {"status": "good", "quality_score": 0.88, "action": "continue", "reason": "OK."}
        d = {"grade": 0, "probabilities": {"0": 0.92, "1": 0.05, "2": 0.02, "3": 0.01}}
        with pytest.raises(DecisionValidationError):
            make_final_decision(q, d, None)

    def test_invalid_dr_grade_raises(self):
        q = {"status": "good", "quality_score": 0.88, "action": "continue", "reason": "OK."}
        d = {"grade": 99, "probabilities": {"0": 0.92, "1": 0.05, "2": 0.02, "3": 0.01}}
        r = {
            "reliability_status": "acceptable", "review_required": False,
            "reason": "OK.", "confidence": 0.92, "confidence_level": "high",
            "uncertainty": 0.18, "uncertainty_level": "low",
            "ood": False, "ood_status": "in_distribution", "ood_score": 1.42,
        }
        with pytest.raises(DecisionValidationError, match="not a valid DR grade"):
            make_final_decision(q, d, r)


# ============================================================================
# End-to-end pipeline simulation
# ============================================================================

class TestEndToEndPipelineSimulation:
    """
    Simulates the complete project pipeline flow without invoking real models.

    Pipeline:
        Fundus Image (mocked)
        → Quality Result (mocked from anuj-fundus-quality output contract)
        → DR Result (mocked from Vinayak's DR model output contract)
        → Reliability Result (mocked from anuj-reliability output contract)
        → Clinical Context (mocked from anuj-clinical-context output contract)
        → Decision Layer (REAL engine call)
        → Final Decision Output (verified)
    """

    def test_full_pipeline_routine(self):
        """Typical healthy patient — No DR, reliable, good image."""
        # Mock quality module output (anuj-fundus-quality)
        quality_output = {
            "status": "good",
            "quality_score": 0.91,
            "focus_score": 0.93,
            "illumination_score": 0.89,
            "field_of_view_score": 0.94,
            "retinal_visibility_score": 0.88,
            "artifact_score": 0.92,
            "reason": "Image is suitable for screening.",
            "action": "continue",
            "enhanced": False,
            "enhanced_image_path": None,
            "error": None,
        }

        # Mock DR model output (Vinayak — not yet integrated)
        dr_output = {
            "grade": 0,
            "probabilities": {"0": 0.92, "1": 0.05, "2": 0.02, "3": 0.01},
            "gradcam_path": None,  # Grad-CAM not yet available
        }

        # Mock reliability pipeline output (anuj-reliability)
        reliability_output = {
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

        # Mock clinical context output (anuj-clinical-context)
        clinical_output = {
            "validation_passed": True,
            "validation_errors": [],
            "clinical_context": {
                "age": {"value": 42, "unit": "years", "status": "provided"},
                "hba1c": {"value": 6.2, "unit": "%", "status": "provided"},
            },
            "data_quality": {
                "complete": True,
                "clinical_context_complete": True,
                "missing_fields": [],
            },
        }

        # Run the REAL Decision Layer
        result = run_screening_decision(
            quality_result=quality_output,
            dr_result=dr_output,
            reliability_result=reliability_output,
            clinical_context=clinical_output,
        )

        # Verify final output
        assert result["action"] == "routine"
        assert result["priority"] == "low"
        assert result["dr_grade"] == 0
        assert result["reliability_status"] == "acceptable"
        assert result["review_required"] is False
        assert result["evidence"]["clinical_context_complete"] is True
        assert result["evidence"]["confidence"] == 0.92
        assert result["evidence"]["ood"] is False

    def test_full_pipeline_referral(self):
        """Patient with Moderate DR (Grade 2), reliable prediction."""
        quality_output = {
            "status": "good", "quality_score": 0.88,
            "action": "continue", "reason": "Image is suitable for screening.",
            "enhanced": False, "enhanced_image_path": None, "error": None,
        }
        dr_output = {
            "grade": 2,
            "probabilities": {"0": 0.02, "1": 0.04, "2": 0.88, "3": 0.06},
            "gradcam_path": "outputs/gradcam/patient_003.jpg",
        }
        reliability_output = {
            "reliability_status": "acceptable", "review_required": False,
            "reason": "High model confidence, low uncertainty and in-distribution input.",
            "confidence": 0.88, "confidence_level": "high",
            "uncertainty": 0.22, "uncertainty_level": "low",
            "ood": False, "ood_status": "in_distribution", "ood_score": 1.55,
            "reliability_score": 0.84,
        }
        result = run_screening_decision(
            quality_result=quality_output,
            dr_result=dr_output,
            reliability_result=reliability_output,
        )
        assert result["action"] == "refer"
        assert result["priority"] == "high"
        assert result["evidence"]["gradcam_path"] == "outputs/gradcam/patient_003.jpg"

    def test_full_pipeline_does_not_duplicate_calculations(self):
        """
        The Decision Layer must NOT recalculate confidence, uncertainty, or OOD.
        It only reads the pre-computed values from reliability_result.
        """
        # Deliberately set conflicting values in reliability vs DR result
        # (a real scenario would never have this, but this tests that the
        # engine reads from reliability_result, not from dr_result directly)
        quality_output = {
            "status": "good", "quality_score": 0.88,
            "action": "continue", "reason": "OK.", "error": None,
        }
        dr_output = {
            "grade": 3,
            "probabilities": {"0": 0.01, "1": 0.01, "2": 0.01, "3": 0.97},
        }
        # Reliability result says OOD=True (which should override)
        reliability_output = {
            "reliability_status": "review_required", "review_required": True,
            "reason": "OOD detected.",
            "confidence": 0.97, "confidence_level": "high",
            "uncertainty": 0.08, "uncertainty_level": "low",
            "ood": True,
            "ood_status": "review_required", "ood_score": 5.5,
        }
        result = run_screening_decision(quality_output, dr_output, reliability_output)
        # Must use reliability_result's ood=True, not recompute from dr_result
        assert result["action"] == "doctor_review"
        assert result["evidence"]["ood"] is True
