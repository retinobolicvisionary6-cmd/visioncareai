"""
test_illumination.py — Unit tests for the illumination assessment module.

All test cases are synthetic prototype tests — NOT clinically validated.
"""

from pathlib import Path
import pytest

from src.illumination import assess_illumination
from src.preprocessing import preprocess

SAMPLE = Path("sample_data")


def _get_illum_result(fname: str) -> dict:
    data = preprocess(str(SAMPLE / fname))
    return assess_illumination(data["gray"], data["fundus_mask"])


class TestIlluminationGood:
    def test_score_high(self):
        r = _get_illum_result("good_fundus.jpg")
        assert r["illumination_score"] >= 0.5

    def test_no_hard_fail(self):
        r = _get_illum_result("good_fundus.jpg")
        assert not r["is_hard_fail"]

    def test_score_in_range(self):
        r = _get_illum_result("good_fundus.jpg")
        assert 0.0 <= r["illumination_score"] <= 1.0


class TestIlluminationDark:
    def test_dark_score_lower_than_good(self):
        good = _get_illum_result("good_fundus.jpg")["illumination_score"]
        dark = _get_illum_result("dark_fundus.jpg")["illumination_score"]
        assert dark < good

    def test_very_dark_detects_condition(self):
        r = _get_illum_result("very_dark_fundus.jpg")
        conditions = r["conditions"]
        assert any("dark" in c or "contrast" in c for c in conditions), (
            f"Very dark image should flag darkness, got conditions: {conditions}"
        )

    def test_very_dark_hard_fail(self):
        r = _get_illum_result("very_dark_fundus.jpg")
        assert r["is_hard_fail"], "Very dark image should be a hard illumination fail"


class TestIlluminationBright:
    def test_bright_score_lower_than_good(self):
        good = _get_illum_result("good_fundus.jpg")["illumination_score"]
        bright = _get_illum_result("overexposed_fundus.jpg")["illumination_score"]
        assert bright < good

    def test_very_bright_hard_fail(self):
        r = _get_illum_result("very_overexposed_fundus.jpg")
        assert r["is_hard_fail"], "Very overexposed image should be a hard illumination fail"


class TestIlluminationOutputContract:
    def test_required_keys(self):
        r = _get_illum_result("good_fundus.jpg")
        for key in ["illumination_score", "mean_brightness", "std_brightness",
                    "clip_ratio", "uniformity_std", "is_hard_fail",
                    "conditions", "detail"]:
            assert key in r, f"Missing key: {key}"
