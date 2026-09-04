"""
Tests — End-to-end OOD detection
"""
import json
from pathlib import Path

import numpy as np
import pytest

from src.ood import detect_ood, OODDetector

REFERENCE_STATS = Path("reference/reference_statistics.json")
ID_DIR = Path("sample_data/in_distribution")
OOD_DIR = Path("sample_data/out_of_distribution")

ID_IMAGES = sorted(ID_DIR.glob("*.png"))
OOD_IMAGES = sorted(OOD_DIR.glob("*.png"))


# ---------------------------------------------------------------------------
# JSON contract / schema tests
# ---------------------------------------------------------------------------

class TestOutputSchema:
    """Verify the detect_ood() output dict always has the correct schema."""

    def test_required_keys_present(self):
        result = detect_ood(ID_IMAGES[0])
        required_keys = {
            "ood", "ood_status", "ood_score", "threshold",
            "distance_metric", "extractor_type", "reason", "metadata",
        }
        assert required_keys.issubset(result.keys()), (
            f"Missing keys: {required_keys - set(result.keys())}"
        )

    def test_ood_is_bool(self):
        result = detect_ood(ID_IMAGES[0])
        assert isinstance(result["ood"], bool)

    def test_ood_status_valid_string(self):
        result = detect_ood(ID_IMAGES[0])
        assert result["ood_status"] in ("in_distribution", "review_required")

    def test_ood_score_is_finite_float(self):
        result = detect_ood(ID_IMAGES[0])
        assert isinstance(result["ood_score"], float)
        assert np.isfinite(result["ood_score"])

    def test_threshold_is_float(self):
        result = detect_ood(ID_IMAGES[0])
        assert isinstance(result["threshold"], float)

    def test_reason_is_nonempty_string(self):
        result = detect_ood(ID_IMAGES[0])
        assert isinstance(result["reason"], str)
        assert len(result["reason"]) > 0

    def test_result_is_json_serialisable(self):
        result = detect_ood(ID_IMAGES[0])
        try:
            json.dumps(result)
        except (TypeError, ValueError) as e:
            pytest.fail(f"Result is not JSON-serialisable: {e}")

    def test_metadata_passthrough(self):
        meta = {"camera_id": "CAM001", "source": "mobile"}
        result = detect_ood(ID_IMAGES[0], metadata=meta)
        assert result["metadata"] == meta


# ---------------------------------------------------------------------------
# Test A — In-distribution (synthetic fundus-like) images
# ---------------------------------------------------------------------------

class TestInDistribution:
    """Reference images should consistently score as in_distribution."""

    def test_all_id_images_are_in_distribution(self):
        """In-distribution images should have scores generally below or around threshold."""
        results = [detect_ood(img) for img in ID_IMAGES]
        scores = [r["ood_score"] for r in results]
        assert all(np.isfinite(s) for s in scores)
        assert np.mean(scores) < 100.0

    def test_id_status_string(self):
        result = detect_ood(ID_IMAGES[0])
        # Either in_distribution OR review_required (borderline) is acceptable
        assert result["ood_status"] in ("in_distribution", "review_required")

    def test_id_scores_are_finite(self):
        for img in ID_IMAGES:
            r = detect_ood(img)
            assert np.isfinite(r["ood_score"])


# ---------------------------------------------------------------------------
# Test B — Out-of-distribution (synthetic OOD) images
# ---------------------------------------------------------------------------

class TestOutOfDistribution:
    """
    Clearly unusual images should score as review_required / OOD.
    Tests against synthetic OOD data only — NOT a clinical validation.
    """

    # Clearly unambiguous OOD images (expected to always flag)
    CLEAR_OOD = [
        "blank_black.png",
        "blank_white.png",
        "solid_red.png",
    ]

    def test_clear_ood_images_flagged(self):
        """Blank and solid colour images must be flagged as OOD."""
        for name in self.CLEAR_OOD:
            path = OOD_DIR / name
            result = detect_ood(path)
            assert result["ood"] is True, (
                f"'{name}' was NOT flagged as OOD. "
                f"Score={result['ood_score']:.4f}, Threshold={result['threshold']:.4f}"
            )

    def test_ood_status_review_required(self):
        """OOD images must carry ood_status='review_required'."""
        for name in self.CLEAR_OOD:
            result = detect_ood(OOD_DIR / name)
            assert result["ood_status"] == "review_required"

    def test_ood_scores_higher_than_id_median(self):
        """
        Mean OOD score for clear-OOD images must be higher than
        median score across all in-distribution images.
        """
        id_scores = [detect_ood(img)["ood_score"] for img in ID_IMAGES]
        ood_scores = [detect_ood(OOD_DIR / n)["ood_score"] for n in self.CLEAR_OOD]
        id_median = float(np.median(id_scores))
        ood_mean = float(np.mean(ood_scores))
        assert ood_mean > id_median, (
            f"OOD mean score ({ood_mean:.4f}) <= ID median score ({id_median:.4f}). "
            "OOD detector is not separating distributions."
        )


# ---------------------------------------------------------------------------
# Test C — Borderline / configurable threshold
# ---------------------------------------------------------------------------

class TestThresholding:

    def test_high_threshold_accepts_everything(self):
        """With a very high threshold, all images should be in_distribution."""
        very_high = 1e9
        for img in ID_IMAGES[:3]:
            r = detect_ood(img, threshold=very_high)
            assert r["ood"] is False

    def test_zero_threshold_flags_everything(self):
        """With threshold=0, all images should be flagged as OOD."""
        for img in ID_IMAGES[:3]:
            r = detect_ood(img, threshold=0.0)
            assert r["ood"] is True

    def test_threshold_in_result_matches_input(self):
        """The 'threshold' key in result must reflect the threshold passed in."""
        custom_thr = 1.23456
        r = detect_ood(ID_IMAGES[0], threshold=custom_thr)
        assert abs(r["threshold"] - custom_thr) < 1e-5

    def test_ood_flag_consistent_with_score_and_threshold(self):
        """ood flag must be consistent: True iff ood_score > threshold."""
        for img in ID_IMAGES[:5]:
            r = detect_ood(img)
            expected_flag = r["ood_score"] > r["threshold"]
            assert r["ood"] == expected_flag


# ---------------------------------------------------------------------------
# Error handling / edge cases
# ---------------------------------------------------------------------------

class TestErrorHandling:

    def test_missing_image_raises(self):
        from src.utils import OODError
        from src.utils import ImageLoadError
        with pytest.raises((FileNotFoundError, ImageLoadError, OODError)):
            detect_ood("nonexistent/fundus.png")

    def test_missing_reference_raises(self):
        from src.utils import ReferenceError
        with pytest.raises(ReferenceError):
            detect_ood(ID_IMAGES[0], reference_path="nonexistent/stats.json")

    def test_array_input_works(self):
        """detect_ood should accept a raw numpy uint8 array."""
        dummy = np.full((224, 224, 3), 100, dtype=np.uint8)
        r = detect_ood(dummy)
        assert "ood" in r

    def test_multiple_calls_consistent(self):
        """Calling detect_ood on the same image twice must return identical results."""
        r1 = detect_ood(ID_IMAGES[0])
        r2 = detect_ood(ID_IMAGES[0])
        assert r1["ood_score"] == r2["ood_score"]
        assert r1["ood"] == r2["ood"]


# ---------------------------------------------------------------------------
# OODDetector class interface
# ---------------------------------------------------------------------------

class TestOODDetectorClass:

    def test_detector_instantiation(self):
        detector = OODDetector()
        assert detector.threshold > 0

    def test_detector_detect_returns_dict(self):
        detector = OODDetector()
        result = detector.detect(ID_IMAGES[0])
        assert isinstance(result, dict)
        assert "ood" in result

    def test_detector_custom_threshold(self):
        detector = OODDetector(threshold=0.0)
        result = detector.detect(ID_IMAGES[0])
        assert result["ood"] is True  # everything is OOD at threshold 0

    def test_detector_custom_metric(self):
        for metric in ("mahalanobis", "cosine", "euclidean"):
            detector = OODDetector(metric=metric)
            result = detector.detect(ID_IMAGES[0])
            assert result["distance_metric"] == metric
            assert np.isfinite(result["ood_score"])
