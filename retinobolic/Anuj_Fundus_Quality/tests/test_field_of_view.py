"""
test_field_of_view.py — Unit tests for the field of view assessment module.

All test cases are synthetic prototype tests — NOT clinically validated.
"""

from pathlib import Path
import pytest

from src.field_of_view import assess_field_of_view
from src.preprocessing import preprocess

SAMPLE = Path("sample_data")


def _get_fov_result(fname: str) -> dict:
    data = preprocess(str(SAMPLE / fname))
    return assess_field_of_view(data["gray"], data["fundus_mask"], data["fundus_circle"])


class TestFOVGood:
    def test_score_high(self):
        r = _get_fov_result("good_fundus.jpg")
        assert r["field_of_view_score"] >= 0.5

    def test_no_hard_fail(self):
        r = _get_fov_result("good_fundus.jpg")
        assert not r["is_hard_fail"]

    def test_score_in_range(self):
        r = _get_fov_result("good_fundus.jpg")
        assert 0.0 <= r["field_of_view_score"] <= 1.0


class TestFOVPoor:
    def test_poor_fov_low_score(self):
        r = _get_fov_result("poor_fov_fundus.jpg")
        good = _get_fov_result("good_fundus.jpg")["field_of_view_score"]
        assert r["field_of_view_score"] < good, (
            f"Poor FOV image should score lower than good. Got {r['field_of_view_score']} vs {good}"
        )

    def test_poor_fov_hard_fail_or_low(self):
        r = _get_fov_result("poor_fov_fundus.jpg")
        assert r["is_hard_fail"] or r["field_of_view_score"] < 0.5


class TestFOVOutputContract:
    def test_required_keys(self):
        r = _get_fov_result("good_fundus.jpg")
        for key in ["field_of_view_score", "fundus_area_ratio", "disk_fill_ratio",
                    "circle_detected", "is_hard_fail", "detail"]:
            assert key in r, f"Missing key: {key}"

    def test_disk_fill_ratio_in_range(self):
        r = _get_fov_result("good_fundus.jpg")
        assert 0.0 <= r["disk_fill_ratio"] <= 1.0
