"""
Test suite for non-retinal image rejection and fundus verification.
Verifies that wrong images (selfies, human faces, buildings, arbitrary photos)
are properly rejected without producing false diabetic retinopathy grades.
"""

import sys
from pathlib import Path
import pytest

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from Anuj_Fundus_Quality.src.fundus_validator import verify_fundus_image
from Anuj_Fundus_Quality.src.quality import assess_quality
from Integration.pipeline.screening_pipeline import run_pipeline
import cv2

RETINA_SAMPLE = PROJ_ROOT / "tests" / "sample_test_retina.jpg"
FACE_SAMPLE = PROJ_ROOT / "static" / "images" / "avatar_designer.jpg"
PERSON_SAMPLE = PROJ_ROOT / "static" / "images" / "doctor_hero.jpg"
BUILDING_SAMPLE = PROJ_ROOT / "static" / "images" / "hospital_building.jpg"


def test_fundus_validator_accepts_retina():
    assert RETINA_SAMPLE.exists(), f"Sample retina missing: {RETINA_SAMPLE}"
    img = cv2.imread(str(RETINA_SAMPLE))
    result = verify_fundus_image(img)
    assert result["is_fundus"] is True, f"Valid retina was rejected: {result}"
    assert result["confidence"] > 0.8


def test_fundus_validator_rejects_face():
    assert FACE_SAMPLE.exists(), f"Face sample missing: {FACE_SAMPLE}"
    img = cv2.imread(str(FACE_SAMPLE))
    result = verify_fundus_image(img)
    assert result["is_fundus"] is False, f"Face photo was accepted as fundus: {result}"
    assert len(result["reasons"]) > 0


def test_fundus_validator_rejects_building():
    assert BUILDING_SAMPLE.exists(), f"Building sample missing: {BUILDING_SAMPLE}"
    img = cv2.imread(str(BUILDING_SAMPLE))
    result = verify_fundus_image(img)
    assert result["is_fundus"] is False, f"Building photo was accepted as fundus: {result}"


def test_quality_gate_rejects_non_fundus():
    q_face = assess_quality(str(FACE_SAMPLE))
    assert q_face["status"] == "invalid_image"
    assert q_face["is_fundus"] is False
    assert q_face["action"] == "reject"
    assert "rejected" in q_face["reason"].lower()


def test_pipeline_rejection_end_to_end():
    res = run_pipeline(str(FACE_SAMPLE))
    
    # 1. Domain flag
    assert res.get("is_valid_fundus") is False, "is_valid_fundus must be False for face photo"
    
    # 2. Final Decision
    decision = res["final_decision"]
    assert decision["action"] == "REJECTED", f"Action should be REJECTED, got {decision['action']}"
    assert decision["dr_grade"] is None, f"DR Grade must be None, got {decision['dr_grade']}"
    
    # 3. DR Model Result
    dr_result = res["dr_result"]
    assert dr_result["grade"] is None, f"dr_result grade must be None, got {dr_result['grade']}"
    assert dr_result["probabilities"] is None, "Probabilities must be None for non-retinal image"
    assert dr_result["gradcam_path"] == "", "Grad-CAM must be empty string for non-retinal image"


def test_pipeline_valid_retina_end_to_end():
    res = run_pipeline(str(RETINA_SAMPLE))
    
    # 1. Domain flag
    assert res.get("is_valid_fundus") is True, "is_valid_fundus must be True for valid retina"
    
    # 2. Final Decision
    decision = res["final_decision"]
    assert decision["action"] != "REJECTED", f"Valid retina was rejected: {decision}"
    assert decision["dr_grade"] is not None, "DR Grade must not be None for valid retina"
    
    # 3. DR Model Result
    dr_result = res["dr_result"]
    assert dr_result["grade"] is not None, "dr_result grade must not be None"
    assert dr_result["probabilities"] is not None, "Probabilities must exist for valid retina"


if __name__ == "__main__":
    test_fundus_validator_accepts_retina()
    print("PASS: test_fundus_validator_accepts_retina")
    test_fundus_validator_rejects_face()
    print("PASS: test_fundus_validator_rejects_face")
    test_fundus_validator_rejects_building()
    print("PASS: test_fundus_validator_rejects_building")
    test_quality_gate_rejects_non_fundus()
    print("PASS: test_quality_gate_rejects_non_fundus")
    test_pipeline_rejection_end_to_end()
    print("PASS: test_pipeline_rejection_end_to_end")
    test_pipeline_valid_retina_end_to_end()
    print("PASS: test_pipeline_valid_retina_end_to_end")
    print("\nALL UNIT AND INTEGRATION TESTS PASSED SUCCESSFULLY!")
