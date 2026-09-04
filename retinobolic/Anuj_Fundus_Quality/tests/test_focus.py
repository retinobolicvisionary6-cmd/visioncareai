"""
test_focus.py — Unit tests for the focus assessment module.

Uses synthetic images from sample_data/.
All test cases are synthetic prototype tests — NOT clinically validated.
"""

from pathlib import Path
import cv2
import numpy as np
import pytest

from src.focus import assess_focus
from src.preprocessing import preprocess

SAMPLE = Path("sample_data")


def _get_focus_result(fname: str) -> dict:
    data = preprocess(str(SAMPLE / fname))
    return assess_focus(data["gray"], data["fundus_mask"])


class TestFocusGoodImage:
    def test_score_high(self):
        r = _get_focus_result("good_fundus.jpg")
        assert r["focus_score"] >= 0.5, (
            f"Good image should have decent focus score, got {r['focus_score']}"
        )

    def test_no_hard_fail(self):
        r = _get_focus_result("good_fundus.jpg")
        assert not r["is_hard_fail"]


class TestFocusBlurredImage:
    def test_score_lower_than_good(self):
        good = _get_focus_result("good_fundus.jpg")["focus_score"]
        blurred = _get_focus_result("blurred_fundus.jpg")["focus_score"]
        assert blurred < good, "Blurred image should have a lower focus score than good"

    def test_very_blurred_is_hard_fail(self):
        r = _get_focus_result("very_blurred_fundus.jpg")
        # Very blurred should be a hard fail or have very low score.
        assert r["is_hard_fail"] or r["focus_score"] < 0.4, (
            f"Very blurred image should fail hard or score very low: {r}"
        )


class TestFocusMetrics:
    def test_laplacian_and_tenengrad_present(self):
        r = _get_focus_result("good_fundus.jpg")
        assert "laplacian_var" in r
        assert "tenengrad_mean" in r

    def test_score_in_range(self):
        for fname in ["good_fundus.jpg", "blurred_fundus.jpg", "dark_fundus.jpg"]:
            r = _get_focus_result(fname)
            assert 0.0 <= r["focus_score"] <= 1.0, (
                f"focus_score out of [0,1] for {fname}: {r['focus_score']}"
            )
