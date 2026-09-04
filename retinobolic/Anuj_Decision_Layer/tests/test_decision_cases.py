"""
tests/test_decision_cases.py — All required test scenarios for the Decision Layer.

Tests the 9 specified scenarios + conflict cases + clinical context scenarios,
using the full make_final_decision() public API.

TEST 1: Good + Reliable + Non-referable → routine
TEST 2: Ungradable → recapture
TEST 3: High uncertainty → doctor_review
TEST 4: OOD → doctor_review
TEST 5: Reliable referable DR → refer
TEST 6: High confidence but OOD → doctor_review
TEST 7: High DR grade but ungradable → recapture
TEST 8: Missing clinical context → basic workflow still works
TEST 9: Conflicting signals → highest-priority safety rule wins

CONFLICT A: Ungradable + Grade 3 + 0.97 confidence → recapture
CONFLICT B: Grade 3 + High confidence + High uncertainty → doctor_review
CONFLICT C: Grade 2 + High confidence + OOD → doctor_review
CONFLICT D: Multiple failures → doctor_review

CLINICAL EXTRA 1: Missing clinical context + refer case → still refers (not blocked)
CLINICAL EXTRA 2: Invalid clinical context → still processes (non-fatal)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.engine import make_final_decision


# ============================================================================
# Fixtures / Builders
# ============================================================================

def quality(status: str, score: float = 0.85) -> dict:
    action_map = {"good": "continue", "borderline": "enhance_and_recheck", "ungradable": "recapture"}
    return {
        "status": status,
        "quality_score": score,
        "action": action_map[status],
        "reason": f"Test quality status: {status}.",
        "enhanced": False,
        "enhanced_image_path": None,
        "error": None,
    }


def dr(grade: int, gradcam_path: Optional[str] = None) -> dict:
    base = [0.01, 0.01, 0.01, 0.01]
    base[grade] = 0.97
    return {
        "grade": grade,
        "probabilities": {str(i): p for i, p in enumerate(base)},
        "gradcam_path": gradcam_path,
    }


def reliability(
    status: str = "acceptable",
    confidence_level: str = "high",
    uncertainty_level: str = "low",
    ood: bool = False,
    ood_score: float = 1.5,
) -> dict:
    conf_val = {"high": 0.92, "medium": 0.65, "low": 0.35}.get(confidence_level, 0.92)
    unc_val = {"low": 0.15, "medium": 0.52, "high": 0.78}.get(uncertainty_level, 0.15)
    return {
        "reliability_status": status,
        "review_required": status == "review_required",
        "reason": "Test reliability reason.",
        "confidence": conf_val,
        "confidence_level": confidence_level,
        "uncertainty": unc_val,
        "uncertainty_level": uncertainty_level,
        "ood": ood,
        "ood_status": "review_required" if ood else "in_distribution",
        "ood_score": ood_score,
        "reliability_score": 0.0 if ood else 0.85,
    }


def clinical(complete: bool = True, patient_id: str = "TEST001") -> dict:
    if complete:
        return {
            "validation_passed": True,
            "validation_errors": [],
            "clinical_context": {
                "age": {"value": 50, "unit": "years", "status": "provided"},
                "hba1c": {"value": 7.2, "unit": "%", "status": "provided"},
            },
            "data_quality": {
                "complete": True,
                "clinical_context_complete": True,
                "missing_fields": [],
            },
        }
    return {}  # Empty = not provided


# ============================================================================
# TEST 1: Good + Reliable + Non-referable → routine
# ============================================================================

def test_1_routine():
    result = make_final_decision(
        quality_result=quality("good"),
        dr_result=dr(0),
        reliability_result=reliability("acceptable"),
    )
    assert result["action"] == "routine", f"Expected 'routine', got '{result['action']}'"
    assert result["priority"] == "low"
    assert result["review_required"] is False
    assert result["dr_grade"] == 0
    assert result["reliability_status"] == "acceptable"


def test_1_routine_grade_1():
    """Grade 1 with default threshold=2 → routine."""
    result = make_final_decision(
        quality_result=quality("good"),
        dr_result=dr(1),
        reliability_result=reliability("acceptable"),
    )
    assert result["action"] == "routine"


# ============================================================================
# TEST 2: Ungradable → recapture
# ============================================================================

def test_2_recapture_ungradable():
    result = make_final_decision(
        quality_result=quality("ungradable", score=0.20),
        dr_result=dr(0),
        reliability_result=reliability("acceptable"),
    )
    assert result["action"] == "recapture"
    assert result["priority"] == "medium"
    assert result["review_required"] is False
    assert result["dr_grade"] is None            # No grade when ungradable
    assert result["reliability_status"] is None  # No reliability when ungradable


# ============================================================================
# TEST 3: High uncertainty → doctor_review
# ============================================================================

def test_3_high_uncertainty():
    result = make_final_decision(
        quality_result=quality("good"),
        dr_result=dr(2),
        reliability_result=reliability(
            "review_required",
            confidence_level="low",
            uncertainty_level="high",
            ood=False,
        ),
    )
    assert result["action"] == "doctor_review"
    assert result["priority"] == "high"
    assert result["review_required"] is True


# ============================================================================
# TEST 4: OOD → doctor_review
# ============================================================================

def test_4_ood_triggers_review():
    result = make_final_decision(
        quality_result=quality("good"),
        dr_result=dr(1),
        reliability_result=reliability(
            "review_required",
            confidence_level="high",
            uncertainty_level="low",
            ood=True,
            ood_score=5.80,
        ),
    )
    assert result["action"] == "doctor_review"
    assert result["review_required"] is True
    assert result["evidence"]["ood"] is True


# ============================================================================
# TEST 5: Reliable referable DR → refer
# ============================================================================

def test_5_reliable_referable_refer_grade2():
    result = make_final_decision(
        quality_result=quality("good"),
        dr_result=dr(2),
        reliability_result=reliability("acceptable"),
    )
    assert result["action"] == "refer"
    assert result["priority"] == "high"
    assert result["review_required"] is False
    assert result["dr_grade"] == 2


def test_5_reliable_referable_refer_grade3_urgent():
    """Grade 3 should be "urgent" priority by default."""
    result = make_final_decision(
        quality_result=quality("good"),
        dr_result=dr(3),
        reliability_result=reliability("acceptable"),
    )
    assert result["action"] == "refer"
    assert result["priority"] == "urgent"
    assert result["dr_grade"] == 3


# ============================================================================
# TEST 6: High confidence but OOD → doctor_review (safety override)
# ============================================================================

def test_6_high_confidence_ood_overrides():
    """
    Case from spec: OOD=True must block referral even when confidence=0.99.
    A high-confidence prediction on an OOD input is MORE dangerous, not less.
    """
    result = make_final_decision(
        quality_result=quality("good"),
        dr_result=dr(2),
        reliability_result=reliability(
            "review_required",
            confidence_level="high",   # High confidence
            uncertainty_level="low",   # Low uncertainty
            ood=True,                  # BUT OOD!
        ),
    )
    assert result["action"] == "doctor_review", (
        "OOD must override high confidence and prevent automatic referral"
    )
    assert result["evidence"]["ood"] is True
    assert result["evidence"]["confidence_level"] == "high"


# ============================================================================
# TEST 7: High DR grade but ungradable → recapture
# ============================================================================

def test_7_high_grade_ungradable_recaptures():
    """
    Conflict Case A from spec: Grade 3, confidence 0.99, but ungradable image.
    Expected: recapture (Rule 1 overrides everything).
    """
    # Override probabilities for high Grade 3 confidence
    dr_data = {
        "grade": 3,
        "probabilities": {"0": 0.01, "1": 0.01, "2": 0.01, "3": 0.97},
        "gradcam_path": None,
    }
    result = make_final_decision(
        quality_result=quality("ungradable", score=0.18),
        dr_result=dr_data,
        reliability_result=reliability("acceptable", confidence_level="high"),
    )
    assert result["action"] == "recapture", (
        "Ungradable image must ALWAYS recapture regardless of DR grade or confidence"
    )
    assert result["dr_grade"] is None   # Grade not used when recapture


# ============================================================================
# TEST 8: Missing clinical context → basic workflow still works
# ============================================================================

def test_8_missing_clinical_context_routine():
    result = make_final_decision(
        quality_result=quality("good"),
        dr_result=dr(0),
        reliability_result=reliability("acceptable"),
        clinical_context=None,  # Not provided
    )
    assert result["action"] == "routine"
    assert result["evidence"]["clinical_context_complete"] is False


def test_8_missing_clinical_context_refer():
    result = make_final_decision(
        quality_result=quality("good"),
        dr_result=dr(2),
        reliability_result=reliability("acceptable"),
        clinical_context=None,
    )
    assert result["action"] == "refer"  # Not blocked by missing clinical context


def test_8_missing_clinical_context_does_not_create_urgent():
    """Missing clinical data must NEVER automatically create urgent priority."""
    result = make_final_decision(
        quality_result=quality("good"),
        dr_result=dr(1),   # Non-referable
        reliability_result=reliability("acceptable"),
        clinical_context=None,
    )
    assert result["action"] == "routine"
    assert result["priority"] == "low"  # Missing context doesn't escalate to urgent


# ============================================================================
# TEST 9: Conflicting signals → highest-priority safety rule wins
# ============================================================================

def test_9_conflict_ungradable_overrides_ood():
    """Rule 1 (ungradable) beats Rule 2 (OOD)."""
    result = make_final_decision(
        quality_result=quality("ungradable"),
        dr_result=dr(1),
        reliability_result=reliability("review_required", ood=True),
    )
    assert result["action"] == "recapture"
    assert result["metadata"]["rule_applied"] == "RULE_1_UNGRADABLE"


def test_9_conflict_ood_overrides_referral():
    """Rule 2 (OOD) beats Rule 3 (referral)."""
    result = make_final_decision(
        quality_result=quality("good"),
        dr_result=dr(3),
        reliability_result=reliability("review_required", ood=True),
    )
    assert result["action"] == "doctor_review"
    assert result["metadata"]["rule_applied"] == "RULE_2_RELIABILITY_FAILURE"


def test_9_conflict_high_uncertainty_overrides_referral():
    """Rule 2 (high uncertainty) beats Rule 3 (referral)."""
    result = make_final_decision(
        quality_result=quality("good"),
        dr_result=dr(3),
        reliability_result=reliability(
            "review_required",
            confidence_level="low",
            uncertainty_level="high",
        ),
    )
    assert result["action"] == "doctor_review"


# ============================================================================
# CONFLICT CASE A: Ungradable + Grade 3 + 0.97 confidence → recapture
# ============================================================================

def test_conflict_a_ungradable_beats_high_grade():
    """From spec Case A: quality=ungradable, DR Grade=3, Confidence=0.99 → recapture."""
    dr_data = {"grade": 3, "probabilities": {"0": 0.01, "1": 0.01, "2": 0.01, "3": 0.97}}
    result = make_final_decision(
        quality_result=quality("ungradable"),
        dr_result=dr_data,
        reliability_result=reliability("acceptable"),
    )
    assert result["action"] == "recapture"


# ============================================================================
# CONFLICT CASE B: Grade 3 + High confidence + High uncertainty → doctor_review
# ============================================================================

def test_conflict_b_high_confidence_high_uncertainty():
    """From spec Case B: Grade 3, Confidence=0.95, Uncertainty=high → doctor_review."""
    dr_data = {"grade": 3, "probabilities": {"0": 0.02, "1": 0.02, "2": 0.02, "3": 0.94}}
    result = make_final_decision(
        quality_result=quality("good"),
        dr_result=dr_data,
        reliability_result=reliability(
            "review_required",
            confidence_level="high",
            uncertainty_level="high",
        ),
    )
    assert result["action"] == "doctor_review"


# ============================================================================
# CONFLICT CASE C: Grade 2 + High confidence + OOD → doctor_review
# ============================================================================

def test_conflict_c_grade2_reliable_ood():
    """Grade 2, high confidence, but OOD=True → doctor_review."""
    dr_data = {"grade": 2, "probabilities": {"0": 0.02, "1": 0.04, "2": 0.88, "3": 0.06}}
    result = make_final_decision(
        quality_result=quality("good"),
        dr_result=dr_data,
        reliability_result=reliability(
            "review_required",
            confidence_level="high",
            uncertainty_level="low",
            ood=True,
        ),
    )
    assert result["action"] == "doctor_review"


# ============================================================================
# Output contract verification
# ============================================================================

class TestOutputContract:
    """Verify the stable JSON output contract for all actions."""

    REQUIRED_KEYS = {"action", "priority", "reason", "dr_grade", "reliability_status",
                     "review_required", "evidence", "metadata"}
    EVIDENCE_KEYS = {"quality_status", "quality_score", "confidence", "confidence_level",
                     "uncertainty", "uncertainty_level", "ood", "ood_score",
                     "gradcam_path", "clinical_context_complete", "reliability_signals"}
    METADATA_KEYS = {"rule_applied", "engine_version"}

    def _verify_output(self, result: dict):
        assert set(result.keys()) >= self.REQUIRED_KEYS
        assert set(result.get("evidence", {}).keys()) >= self.EVIDENCE_KEYS
        assert set(result.get("metadata", {}).keys()) >= self.METADATA_KEYS
        assert result["action"] in {"recapture", "doctor_review", "refer", "routine"}
        assert result["priority"] in {"low", "medium", "high", "urgent"}
        assert isinstance(result["reason"], str) and len(result["reason"]) > 0
        assert isinstance(result["review_required"], bool)

    def test_contract_routine(self):
        result = make_final_decision(quality("good"), dr(0), reliability("acceptable"))
        self._verify_output(result)

    def test_contract_recapture(self):
        result = make_final_decision(quality("ungradable"), dr(1), reliability("acceptable"))
        self._verify_output(result)
        assert result["dr_grade"] is None
        assert result["reliability_status"] is None

    def test_contract_doctor_review(self):
        result = make_final_decision(
            quality("good"), dr(2),
            reliability("review_required", ood=True),
        )
        self._verify_output(result)
        assert result["review_required"] is True

    def test_contract_refer(self):
        result = make_final_decision(quality("good"), dr(2), reliability("acceptable"))
        self._verify_output(result)
        assert result["dr_grade"] == 2
        assert result["reliability_status"] == "acceptable"

    def test_reason_is_non_diagnostic(self):
        """Reasons must not contain diagnostic statements."""
        for grade_data, rel_data in [
            (dr(0), reliability("acceptable")),
            (dr(2), reliability("acceptable")),
        ]:
            result = make_final_decision(quality("good"), grade_data, rel_data)
            reason = result["reason"].lower()
            assert "patient definitely has dr" not in reason
            assert "patient is disease-free" not in reason
            assert "treatment required" not in reason

    def test_gradcam_path_preserved_in_evidence(self):
        """Grad-CAM path must be preserved in evidence output."""
        dr_data = {
            "grade": 2,
            "probabilities": {"0": 0.02, "1": 0.04, "2": 0.88, "3": 0.06},
            "gradcam_path": "outputs/gradcam/test.jpg",
        }
        result = make_final_decision(quality("good"), dr_data, reliability("acceptable"))
        assert result["evidence"]["gradcam_path"] == "outputs/gradcam/test.jpg"

    def test_no_evidence_when_disabled(self):
        result = make_final_decision(
            quality("good"), dr(0), reliability("acceptable"),
            include_evidence=False,
        )
        assert "evidence" not in result
        assert "metadata" in result

    def test_metadata_contains_rule_applied(self):
        result = make_final_decision(quality("good"), dr(0), reliability("acceptable"))
        assert "rule_applied" in result["metadata"]
        assert result["metadata"]["rule_applied"] == "RULE_4_ROUTINE"

    def test_metadata_engine_version(self):
        result = make_final_decision(quality("good"), dr(0), reliability("acceptable"))
        assert "engine_version" in result["metadata"]


# ============================================================================
# Clinical context integration
# ============================================================================

class TestClinicalContextIntegration:

    def test_complete_clinical_context_routine(self):
        result = make_final_decision(
            quality("good"), dr(0), reliability("acceptable"),
            clinical_context=clinical(complete=True),
        )
        assert result["action"] == "routine"
        assert result["evidence"]["clinical_context_complete"] is True

    def test_invalid_clinical_context_does_not_crash(self):
        """Non-fatal: invalid clinical context must not crash the engine."""
        result = make_final_decision(
            quality("good"), dr(0), reliability("acceptable"),
            clinical_context="bad_clinical_data",
        )
        assert result["action"] == "routine"
        assert result["evidence"]["clinical_context_complete"] is False
