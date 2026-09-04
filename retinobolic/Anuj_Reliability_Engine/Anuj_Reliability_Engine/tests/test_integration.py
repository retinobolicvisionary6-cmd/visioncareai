"""
tests/test_integration.py — End-to-end integration tests for the Reliability Engine.

These tests call the REAL upstream modules:
    - calculate_confidence  (anuj-confidence)
    - calculate_uncertainty (anuj-uncertainty)
    - detect_ood            (anuj-ood)

and pass their outputs through the full Reliability Engine.

A lightweight mock DR result is used (Vinayak's model not required here).
Real fundus images from anuj-ood/sample_data are used for OOD testing.

Test scenarios (from specification):
    Test 1 — Fully reliable        : high conf, low uncert, in-dist  → acceptable
    Test 2 — High uncertainty      : uniform probs + in-dist          → review_required
    Test 3 — OOD despite high conf : high conf, low uncert, OOD=True → review_required
    Test 4 — Low confidence        : low max prob + in-dist           → review_required
    Test 5 — Multiple failures     : uniform + OOD                    → review_required
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Real module imports via adapters
from src.confidence import calculate_confidence
from src.uncertainty import calculate_uncertainty
from src.ood import detect_ood
from src.reliability.fusion import calculate_reliability
from src.reliability.engine import run_reliability_pipeline

# Sample images — IN_DIST_IMAGE updated to verified APTOS image after APTOS calibration
# (fundus_01.png scores OOD=28.8 against the new APTOS reference; use a real APTOS image instead)
OOD_ROOT = Path("d:/anuj-ood")
APTOS_IMAGES = Path(r"E:\retinobolic\data\raw\train_images")
IN_DIST_IMAGE = APTOS_IMAGES / "00a8624548a9.png"   # APTOS Grade 0, OOD score=4.1
OUT_DIST_IMAGE = OOD_ROOT / "sample_data" / "out_of_distribution" / "random_noise.png"


# ---------------------------------------------------------------------------
# Integration Test 1 — Fully reliable
# ---------------------------------------------------------------------------

class TestIntegrationFullyReliable:
    """High confidence, low uncertainty, in-distribution → acceptable."""

    DR_RESULT = {
        "grade": 2,
        "probabilities": {"0": 0.01, "1": 0.01, "2": 0.97, "3": 0.01},
    }

    @pytest.mark.skipif(not IN_DIST_IMAGE.exists(), reason="Sample in-distribution image not found")
    def test_acceptable_pipeline(self):
        result = run_reliability_pipeline(self.DR_RESULT, str(IN_DIST_IMAGE))

        assert result["review_required"] is False
        assert result["reliability_status"] == "acceptable"
        assert result["confidence"] >= 0.80   # high confidence
        assert result["ood"] is False
        assert "High model confidence" in result["reason"]

    @pytest.mark.skipif(not IN_DIST_IMAGE.exists(), reason="Sample in-distribution image not found")
    def test_full_pipeline_returns_all_keys(self):
        result = run_reliability_pipeline(self.DR_RESULT, str(IN_DIST_IMAGE))
        for key in [
            "reliability_status", "review_required", "reason",
            "confidence", "confidence_level",
            "uncertainty", "uncertainty_level",
            "ood", "ood_status", "ood_score",
        ]:
            assert key in result, f"Missing key: {key}"

    @pytest.mark.skipif(not IN_DIST_IMAGE.exists(), reason="Sample in-distribution image not found")
    def test_module_outputs_are_preserved(self):
        """Upstream confidence/uncertainty/ood values must appear in final result."""
        conf_raw = calculate_confidence(self.DR_RESULT)
        uncert_raw = calculate_uncertainty(self.DR_RESULT)
        ood_raw = detect_ood(str(IN_DIST_IMAGE))

        result = run_reliability_pipeline(self.DR_RESULT, str(IN_DIST_IMAGE))

        assert result["confidence"] == conf_raw["confidence"]
        assert result["uncertainty"] == uncert_raw["uncertainty"]
        assert result["ood"] == ood_raw["ood"]
        assert result["ood_score"] == ood_raw["ood_score"]


# ---------------------------------------------------------------------------
# Integration Test 2 — High uncertainty
# ---------------------------------------------------------------------------

class TestIntegrationHighUncertainty:
    """Uniform probabilities → high Shannon entropy → review_required."""

    DR_RESULT = {
        "grade": 1,
        "probabilities": {"0": 0.26, "1": 0.28, "2": 0.24, "3": 0.22},
    }

    @pytest.mark.skipif(not IN_DIST_IMAGE.exists(), reason="Sample in-distribution image not found")
    def test_high_uncertainty_triggers_review(self):
        result = run_reliability_pipeline(self.DR_RESULT, str(IN_DIST_IMAGE))
        assert result["reliability_status"] == "review_required"
        assert result["review_required"] is True
        assert result["uncertainty_level"] == "high"


# ---------------------------------------------------------------------------
# Integration Test 3 — OOD despite high confidence
# ---------------------------------------------------------------------------

class TestIntegrationOODHighConfidence:
    """High confidence + low uncertainty + OOD image → review_required."""

    DR_RESULT = {
        "grade": 2,
        "probabilities": {"0": 0.01, "1": 0.01, "2": 0.97, "3": 0.01},
    }

    @pytest.mark.skipif(not OUT_DIST_IMAGE.exists(), reason="Sample OOD image not found")
    def test_ood_overrides_high_confidence(self):
        result = run_reliability_pipeline(self.DR_RESULT, str(OUT_DIST_IMAGE))
        assert result["reliability_status"] == "review_required"
        assert result["review_required"] is True
        assert result["ood"] is True
        assert "outside the configured reference distribution" in result["reason"]

    @pytest.mark.skipif(not OUT_DIST_IMAGE.exists(), reason="Sample OOD image not found")
    def test_confidence_remains_high_even_when_ood(self):
        """Confidence signal is preserved — OOD does not alter it."""
        result = run_reliability_pipeline(self.DR_RESULT, str(OUT_DIST_IMAGE))
        assert result["confidence"] >= 0.80
        assert result["confidence_level"] == "high"


# ---------------------------------------------------------------------------
# Integration Test 4 — Low confidence
# ---------------------------------------------------------------------------

class TestIntegrationLowConfidence:
    """Low max probability → low confidence → review_required."""

    DR_RESULT = {
        "grade": 0,
        "probabilities": {"0": 0.35, "1": 0.25, "2": 0.20, "3": 0.20},
    }

    @pytest.mark.skipif(not IN_DIST_IMAGE.exists(), reason="Sample in-distribution image not found")
    def test_low_confidence_triggers_review(self):
        result = run_reliability_pipeline(self.DR_RESULT, str(IN_DIST_IMAGE))
        assert result["review_required"] is True
        assert result["reliability_status"] in ("review_required", "caution")


# ---------------------------------------------------------------------------
# Integration Test 5 — Multiple failures
# ---------------------------------------------------------------------------

class TestIntegrationMultipleFailures:
    """Uniform distribution + OOD image → everything fails → review_required."""

    DR_RESULT = {
        "grade": 0,
        "probabilities": {"0": 0.27, "1": 0.24, "2": 0.25, "3": 0.24},
    }

    @pytest.mark.skipif(not OUT_DIST_IMAGE.exists(), reason="Sample OOD image not found")
    def test_all_failures_review_required(self):
        result = run_reliability_pipeline(self.DR_RESULT, str(OUT_DIST_IMAGE))
        assert result["reliability_status"] == "review_required"
        assert result["review_required"] is True
        assert result["ood"] is True
        assert len(result["reason"]) > 0


# ---------------------------------------------------------------------------
# No-duplicate-calculation check (structural, not cryptographic)
# ---------------------------------------------------------------------------

class TestNoDuplication:
    """Verify that the pipeline result values match individual module outputs."""

    DR_RESULT = {
        "grade": 2,
        "probabilities": {"0": 0.01, "1": 0.01, "2": 0.97, "3": 0.01},
    }

    @pytest.mark.skipif(not IN_DIST_IMAGE.exists(), reason="Sample in-distribution image not found")
    def test_reliability_does_not_recalculate_confidence(self):
        """Confidence in output must equal what calculate_confidence() returns."""
        expected_conf = calculate_confidence(self.DR_RESULT)["confidence"]
        result = run_reliability_pipeline(self.DR_RESULT, str(IN_DIST_IMAGE))
        assert result["confidence"] == expected_conf

    @pytest.mark.skipif(not IN_DIST_IMAGE.exists(), reason="Sample in-distribution image not found")
    def test_reliability_does_not_recalculate_uncertainty(self):
        expected_uncert = calculate_uncertainty(self.DR_RESULT)["uncertainty"]
        result = run_reliability_pipeline(self.DR_RESULT, str(IN_DIST_IMAGE))
        assert result["uncertainty"] == expected_uncert
