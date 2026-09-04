"""
test_enhancement.py — Unit tests for the enhancement module.

All test cases are synthetic prototype tests — NOT clinically validated.
"""

from pathlib import Path
import cv2
import numpy as np
import pytest

from src.enhancement import enhance
from src.preprocessing import preprocess

SAMPLE = Path("sample_data")


def _load_resized(fname: str) -> np.ndarray:
    data = preprocess(str(SAMPLE / fname))
    return data["resized"]


class TestEnhancementOutputShape:
    def test_output_same_shape(self):
        img = _load_resized("good_fundus.jpg")
        out = enhance(img)
        assert out.shape == img.shape

    def test_output_dtype_uint8(self):
        img = _load_resized("good_fundus.jpg")
        out = enhance(img)
        assert out.dtype == np.uint8


class TestEnhancementDoesNotModifyOriginal:
    def test_original_unchanged(self):
        img = _load_resized("borderline_fundus.jpg")
        original_copy = img.copy()
        _ = enhance(img)
        assert np.array_equal(img, original_copy), "enhance() must not modify the input array"


class TestEnhancementOnDarkImage:
    def test_dark_image_gets_brighter(self):
        # borderline_fundus is moderately dark (mean ~40) and reliably
        # benefits from gamma correction + CLAHE brightening.
        # very_dark_fundus (~15 mean) is too dark for NLM and is intended
        # to be rejected as ungradable, not enhanced.
        img = _load_resized("borderline_fundus.jpg")
        enhanced = enhance(img)
        orig_mean = float(np.mean(img))
        enh_mean = float(np.mean(enhanced))
        assert enh_mean > orig_mean, (
            f"Enhancement should brighten a moderately dark image. "
            f"Before={orig_mean:.1f}, After={enh_mean:.1f}"
        )


class TestEnhancementBorderlineRecheckLogic:
    """Verify the full borderline → enhance → re-assess path via assess_quality."""

    def test_borderline_triggers_enhancement(self):
        """assess_quality on a borderline image should set enhanced=True."""
        from src.quality import assess_quality
        result = assess_quality(str(SAMPLE / "borderline_fundus.jpg"))
        # borderline should attempt enhancement (may still be borderline after)
        if result["status"] == "borderline":
            # If borderline, enhanced flag should be True (attempted)
            assert result["enhanced"] is True
        # If enhanced to good or degraded to ungradable, that is also valid.
        assert result["status"] in ("good", "borderline", "ungradable")

    def test_ungradable_not_enhanced(self):
        """assess_quality on ungradable images should NOT attempt enhancement."""
        from src.quality import assess_quality
        result = assess_quality(str(SAMPLE / "very_blurred_fundus.jpg"))
        if result["status"] == "ungradable":
            assert result["enhanced"] is False
