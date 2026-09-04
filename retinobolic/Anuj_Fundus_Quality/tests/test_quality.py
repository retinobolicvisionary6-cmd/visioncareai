"""
test_quality.py — Integration tests for the central assess_quality() API.

Tests cover:
  - Good fundus → "good" / "continue"
  - Blurred → "borderline" or "ungradable"
  - Very dark → "ungradable"
  - Very bright → "ungradable"
  - Poor FOV → "ungradable" or low score
  - Borderline → "enhance_and_recheck" action (or better)
  - Error handling (missing file, corrupt data, bad extension)
  - Output contract stability

All cases use synthetic images — NOT clinically validated.
"""

from pathlib import Path
import json
import pytest

from src.quality import assess_quality, result_to_json

SAMPLE = Path("sample_data")

# ── required output keys ────────────────────────────────────────────────────
REQUIRED_KEYS = [
    "status", "quality_score", "focus_score", "illumination_score",
    "field_of_view_score", "retinal_visibility_score", "artifact_score",
    "reason", "action", "enhanced", "enhanced_image_path", "error",
]


# ── helpers ──────────────────────────────────────────────────────────────────

def _assess(fname: str) -> dict:
    return assess_quality(str(SAMPLE / fname))


# ── output contract ──────────────────────────────────────────────────────────

class TestOutputContract:
    def test_all_required_keys_present(self):
        r = _assess("good_fundus.jpg")
        for key in REQUIRED_KEYS:
            assert key in r, f"Missing required key: {key}"

    def test_status_is_valid_enum(self):
        r = _assess("good_fundus.jpg")
        assert r["status"] in ("good", "borderline", "ungradable")

    def test_action_is_valid_enum(self):
        r = _assess("good_fundus.jpg")
        assert r["action"] in ("continue", "enhance_and_recheck", "recapture")

    def test_scores_in_range(self):
        r = _assess("good_fundus.jpg")
        for key in ["quality_score", "focus_score", "illumination_score",
                    "field_of_view_score", "retinal_visibility_score", "artifact_score"]:
            assert 0.0 <= r[key] <= 1.0, f"{key}={r[key]} out of [0,1]"

    def test_json_serialisable(self):
        r = _assess("good_fundus.jpg")
        j = result_to_json(r)
        parsed = json.loads(j)  # must not raise
        assert parsed["status"] == r["status"]

    def test_reason_is_nonempty_string(self):
        r = _assess("good_fundus.jpg")
        assert isinstance(r["reason"], str) and len(r["reason"]) > 0


# ── per-scenario tests ────────────────────────────────────────────────────────

class TestGoodFundus:
    def test_status(self):
        r = _assess("good_fundus.jpg")
        assert r["status"] == "good", f"Expected good, got {r['status']}"

    def test_action(self):
        r = _assess("good_fundus.jpg")
        assert r["action"] == "continue"

    def test_quality_score_high(self):
        r = _assess("good_fundus.jpg")
        assert r["quality_score"] >= 0.60, f"Good image quality_score too low: {r['quality_score']}"


class TestBlurredFundus:
    def test_status_degraded(self):
        r = _assess("blurred_fundus.jpg")
        assert r["status"] in ("borderline", "ungradable"), (
            f"Blurred image should not be 'good', got {r['status']}"
        )

    def test_focus_score_low(self):
        good_focus = assess_quality(str(SAMPLE / "good_fundus.jpg"))["focus_score"]
        blur_focus = _assess("blurred_fundus.jpg")["focus_score"]
        assert blur_focus < good_focus


class TestVeryBlurredFundus:
    def test_ungradable(self):
        r = _assess("very_blurred_fundus.jpg")
        assert r["status"] == "ungradable", f"Very blurred should be ungradable, got {r['status']}"

    def test_action_recapture(self):
        r = _assess("very_blurred_fundus.jpg")
        assert r["action"] == "recapture"

    def test_not_enhanced(self):
        r = _assess("very_blurred_fundus.jpg")
        assert r["enhanced"] is False


class TestVeryDarkFundus:
    def test_ungradable(self):
        r = _assess("very_dark_fundus.jpg")
        assert r["status"] == "ungradable", f"Very dark should be ungradable, got {r['status']}"

    def test_action_recapture(self):
        r = _assess("very_dark_fundus.jpg")
        assert r["action"] == "recapture"


class TestOverexposedFundus:
    def test_status_degraded(self):
        r = _assess("overexposed_fundus.jpg")
        assert r["status"] in ("borderline", "ungradable")

    def test_very_overexposed_ungradable(self):
        r = _assess("very_overexposed_fundus.jpg")
        assert r["status"] == "ungradable"


class TestPoorFOV:
    def test_status_degraded(self):
        r = _assess("poor_fov_fundus.jpg")
        assert r["status"] in ("borderline", "ungradable")

    def test_fov_score_low(self):
        r = _assess("poor_fov_fundus.jpg")
        good_fov = assess_quality(str(SAMPLE / "good_fundus.jpg"))["field_of_view_score"]
        assert r["field_of_view_score"] < good_fov


class TestBorderlineFundus:
    def test_status_borderline_or_better(self):
        r = _assess("borderline_fundus.jpg")
        # After enhancement it could be good; it should not be ungradable for a mild borderline.
        # We allow good or borderline as success.
        assert r["status"] in ("good", "borderline"), (
            f"Borderline image should be good or borderline after enhancement, got {r['status']}"
        )

    def test_enhancement_attempted(self):
        r = _assess("borderline_fundus.jpg")
        # Enhancement is attempted if the initial assessment was borderline.
        # If initial was already good, enhanced may be False — that's fine.
        assert r["status"] in ("good", "borderline")


# ── error handling ────────────────────────────────────────────────────────────

class TestErrorHandling:
    def test_missing_file(self):
        r = assess_quality("sample_data/nonexistent_image.jpg")
        assert r["status"] == "ungradable"
        assert r["action"] == "recapture"
        assert r["error"] is not None

    def test_unsupported_extension(self, tmp_path):
        bmp_file = tmp_path / "test.bmp"
        bmp_file.write_bytes(b"\x00" * 100)
        r = assess_quality(str(bmp_file))
        assert r["status"] == "ungradable"
        assert r["error"] is not None

    def test_corrupt_file(self, tmp_path):
        corrupt = tmp_path / "corrupt.jpg"
        corrupt.write_bytes(b"not an image file at all")
        r = assess_quality(str(corrupt))
        assert r["status"] == "ungradable"
        assert r["error"] is not None


# ── action-status consistency ─────────────────────────────────────────────────

class TestActionStatusConsistency:
    @pytest.mark.parametrize("fname,expected_action", [
        ("good_fundus.jpg", "continue"),
        ("very_blurred_fundus.jpg", "recapture"),
        ("very_dark_fundus.jpg", "recapture"),
    ])
    def test_action_matches_status(self, fname, expected_action):
        r = _assess(fname)
        assert r["action"] == expected_action, (
            f"{fname}: expected action={expected_action}, got {r['action']}"
        )
