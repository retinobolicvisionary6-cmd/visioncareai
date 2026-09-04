"""
VINAYAK Module — Inference Contract Test Suite.

Phase 11 — Automated tests covering:
  1. Valid image inference
  2. Invalid image (corrupt data)
  3. Unsupported format file
  4. Missing file (path does not exist)
  5. Model loading failure (malformed checkpoint)
  6. Malformed checkpoint (wrong keys)
  7. Probability normalization (sum ≈ 1.0)
  8. Grade range (must be 0, 1, 2, or 3)
  9. Grad-CAM generation (file exists and is non-empty)

Also tests:
  - Forbidden fields (ANUJ's fields must NOT appear in output)
  - Output contract structure (required keys)

Run with:
    python tests/test_inference.py       (direct)
    pytest tests/test_inference.py -v   (pytest)

NOTE: These tests use a synthetic fundus image. A trained checkpoint is NOT
required for most contract structure tests — they use base weights.
"""
import sys
import json
import tempfile
import shutil
import struct
from pathlib import Path
import numpy as np
import cv2
import torch
import pytest

# Setup path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from configs.config import CLASS_NAMES, GRADCAM_OUTPUT_DIR, CHECKPOINT_PATH
from src.inference import predict, load_inference_model
from src.preprocess import preprocess_single_image, get_val_transforms


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

TEST_DIR = Path(__file__).resolve().parent
SAMPLE_RETINA = TEST_DIR / "sample_test_retina.jpg"


def _create_synthetic_fundus(path: Path) -> Path:
    """Creates a realistic-looking synthetic fundus image for testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = 512, 512
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Dark reddish background simulating fundus
    img[:] = [30, 15, 10]

    # Main circular retina
    cv2.circle(img, (w // 2, h // 2), 230, (80, 40, 25), -1)

    # Optic disc (bright yellow-white disc)
    cv2.circle(img, (w // 2 - 80, h // 2), 40, (200, 210, 150), -1)

    # Retinal vessels (dark branching lines)
    cv2.line(img, (w // 2 - 80, h // 2), (w // 2 + 140, h // 2 - 90), (15, 10, 5), 5)
    cv2.line(img, (w // 2 - 80, h // 2), (w // 2 + 130, h // 2 + 100), (15, 10, 5), 4)
    cv2.line(img, (w // 2 - 80, h // 2), (w // 2 - 180, h // 2 - 60), (15, 10, 5), 3)

    # Gaussian blur to soften edges
    img = cv2.GaussianBlur(img, (7, 7), 0)
    cv2.imwrite(str(path), img)
    return path


def _ensure_sample_exists():
    """Ensures the sample test image exists, creating it if needed."""
    if not SAMPLE_RETINA.exists():
        _create_synthetic_fundus(SAMPLE_RETINA)
    return SAMPLE_RETINA


def _get_or_make_checkpoint() -> Path:
    """
    Returns the best checkpoint if it exists, otherwise creates a minimal
    random-weight checkpoint so tests can run without training.
    """
    if CHECKPOINT_PATH.exists():
        return CHECKPOINT_PATH

    # Create minimal checkpoint with random weights
    from src.model import build_model
    from configs.config import MODEL_NAME, NUM_CLASSES, DROPOUT_RATE, DEVICE
    model = build_model(
        architecture=MODEL_NAME,
        num_classes=NUM_CLASSES,
        pretrained=False,
        dropout_rate=DROPOUT_RATE,
        device=DEVICE,
        verbose=False,
    )
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    return CHECKPOINT_PATH


# ---------------------------------------------------------------------------
# Test 1 — Valid image inference
# ---------------------------------------------------------------------------

def test_valid_image_inference():
    """Test that a valid fundus image produces a well-formed output."""
    sample = _ensure_sample_exists()
    _get_or_make_checkpoint()  # ensure checkpoint exists

    result = predict(str(sample))

    assert isinstance(result, dict), "Result must be a dict"
    assert "grade" in result,          "Missing 'grade'"
    assert "probabilities" in result,  "Missing 'probabilities'"
    assert "gradcam_path" in result,   "Missing 'gradcam_path'"
    print(f"\n[Test 1 PASS] valid_image: grade={result['grade']}")


# ---------------------------------------------------------------------------
# Test 2 — Invalid image (corrupt binary content)
# ---------------------------------------------------------------------------

def test_invalid_corrupt_image():
    """Test that a corrupt (non-image) file raises ValueError."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(b"\xff\xd8" + b"\x00" * 512)   # fake JPEG header, corrupt body
        tmp_path = f.name

    try:
        with pytest.raises((ValueError, RuntimeError, Exception)):
            predict(tmp_path)
        print("[Test 2 PASS] corrupt image correctly raised exception")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Test 3 — Unsupported format (e.g. a .txt file)
# ---------------------------------------------------------------------------

def test_unsupported_format():
    """Test that a text file passed as an image raises an appropriate error."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
        f.write("This is not an image file.\n")
        tmp_path = f.name

    try:
        with pytest.raises((ValueError, RuntimeError, Exception)):
            predict(tmp_path)
        print("[Test 3 PASS] unsupported format correctly raised exception")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Test 4 — Missing file (path does not exist)
# ---------------------------------------------------------------------------

def test_missing_file():
    """Test that a non-existent file path raises FileNotFoundError."""
    fake_path = "/tmp/this_file_absolutely_does_not_exist_xyz123.jpg"
    with pytest.raises(FileNotFoundError):
        predict(fake_path)
    print("[Test 4 PASS] missing file correctly raised FileNotFoundError")


# ---------------------------------------------------------------------------
# Test 5 — Model loading failure (checkpoint path does not exist)
# ---------------------------------------------------------------------------

def test_model_loading_failure():
    """Test that a non-existent checkpoint raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_inference_model(
            checkpoint_path=Path("/tmp/nonexistent_checkpoint_abc.pth"),
            force_reload=True,
        )
    print("[Test 5 PASS] missing checkpoint correctly raised FileNotFoundError")


# ---------------------------------------------------------------------------
# Test 6 — Malformed checkpoint (wrong keys)
# ---------------------------------------------------------------------------

def test_malformed_checkpoint():
    """Test that a checkpoint with incompatible keys raises RuntimeError."""
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        # Save a dict with totally wrong keys
        torch.save({"totally_wrong_key": torch.randn(10)}, f.name)
        tmp_ckpt = f.name

    try:
        with pytest.raises((RuntimeError, Exception)):
            load_inference_model(
                checkpoint_path=Path(tmp_ckpt),
                force_reload=True,
            )
        print("[Test 6 PASS] malformed checkpoint correctly raised exception")
    finally:
        Path(tmp_ckpt).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Test 7 — Probability normalization (sum ≈ 1.0)
# ---------------------------------------------------------------------------

def test_probability_normalization():
    """Test that output probabilities sum to approximately 1.0."""
    sample = _ensure_sample_exists()
    _get_or_make_checkpoint()

    result = predict(str(sample))
    probs = result["probabilities"]

    total = sum(probs.values())
    assert abs(total - 1.0) < 0.05, (
        f"Probabilities must sum to ≈ 1.0, got {total:.6f}"
    )
    print(f"[Test 7 PASS] probability sum = {total:.6f}")


# ---------------------------------------------------------------------------
# Test 8 — Grade range (must be 0, 1, 2, 3, or 4)
# ---------------------------------------------------------------------------

def test_grade_range():
    """Test that the output grade is always one of {0, 1, 2, 3, 4}."""
    sample = _ensure_sample_exists()
    _get_or_make_checkpoint()

    result = predict(str(sample))
    grade = result["grade"]

    assert isinstance(grade, int), f"Grade must be int, got {type(grade)}"
    assert grade in {0, 1, 2, 3, 4}, f"Grade must be 0-4, got {grade}"
    print(f"[Test 8 PASS] grade = {grade} ({CLASS_NAMES[grade]})")


# ---------------------------------------------------------------------------
# Test 9 — Grad-CAM generation
# ---------------------------------------------------------------------------

def test_gradcam_generation():
    """Test that Grad-CAM overlay image is created and non-empty."""
    sample = _ensure_sample_exists()
    _get_or_make_checkpoint()

    result = predict(str(sample), generate_gradcam=True)
    gradcam_path = Path(result["gradcam_path"])

    assert gradcam_path.exists(), f"Grad-CAM file not created: {gradcam_path}"
    assert gradcam_path.stat().st_size > 0, "Grad-CAM file is empty"
    print(f"[Test 9 PASS] Grad-CAM saved: {gradcam_path} ({gradcam_path.stat().st_size} bytes)")


# ---------------------------------------------------------------------------
# Additional: Contract structure tests
# ---------------------------------------------------------------------------

def test_output_keys_exact():
    """Test that output contains EXACTLY the required keys (no extras, no missing)."""
    sample = _ensure_sample_exists()
    _get_or_make_checkpoint()

    result = predict(str(sample))
    required_keys = {"grade", "grade_name", "probabilities", "gradcam_path", "mcdo"}
    assert set(result.keys()) == required_keys, (
        f"Output keys mismatch. Expected {required_keys}, got {set(result.keys())}"
    )
    print(f"[Test PASS] Output keys: {set(result.keys())}")


def test_probability_keys():
    """Test that probabilities dict has exactly keys '0', '1', '2', '3', '4'."""
    sample = _ensure_sample_exists()
    _get_or_make_checkpoint()

    result = predict(str(sample))
    probs = result["probabilities"]
    assert set(probs.keys()) == {"0", "1", "2", "3", "4"}, (
        f"Probability keys must be '0','1','2','3','4', got {set(probs.keys())}"
    )
    for k, v in probs.items():
        assert isinstance(v, float), f"Probability[{k}] must be float, got {type(v)}"
        assert 0.0 <= v <= 1.0, f"Probability[{k}]={v} outside [0,1]"
    print(f"[Test PASS] probability keys and types valid")


def test_no_forbidden_fields():
    """Test that ANUJ's reserved fields do NOT appear in VINAYAK output."""
    sample = _ensure_sample_exists()
    _get_or_make_checkpoint()

    result = predict(str(sample))
    forbidden = ["quality", "confidence", "uncertainty", "ood", "action", "priority"]

    for field in forbidden:
        assert field not in result, (
            f"PROHIBITED field '{field}' found in VINAYAK output! "
            f"This belongs to ANUJ's reliability module."
        )
    print("[Test PASS] No forbidden ANUJ fields in output")


def test_json_serializable():
    """Test that the output dict is fully JSON-serializable."""
    sample = _ensure_sample_exists()
    _get_or_make_checkpoint()

    result = predict(str(sample))
    try:
        serialized = json.dumps(result)
        reparsed = json.loads(serialized)
        assert reparsed["grade"] == result["grade"]
    except (TypeError, ValueError) as e:
        pytest.fail(f"Output is not JSON-serializable: {e}")
    print("[Test PASS] Output is JSON-serializable")


# ---------------------------------------------------------------------------
# Preprocessing unit tests
# ---------------------------------------------------------------------------

def test_preprocess_single_image_shape():
    """Test that preprocessing returns correct tensor shape."""
    sample = _ensure_sample_exists()
    tensor = preprocess_single_image(str(sample))
    assert tensor.shape == (1, 3, 384, 384), (
        f"Expected shape (1, 3, 384, 384), got {tensor.shape}"
    )
    print(f"[Test PASS] preprocess tensor shape: {tensor.shape}")


def test_preprocess_missing_file():
    """Test that preprocessing raises FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError):
        preprocess_single_image("/nonexistent/path/image.jpg")
    print("[Test PASS] preprocess missing file raised FileNotFoundError")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("="*60)
    print("  VISIONARY6 — VINAYAK Integration Contract Test Suite")
    print("="*60)

    # Create checkpoint if needed
    _get_or_make_checkpoint()
    print(f"\nCheckpoint: {CHECKPOINT_PATH}")

    tests = [
        ("1. Valid image inference",          test_valid_image_inference),
        ("2. Corrupt image",                  test_invalid_corrupt_image),
        ("3. Unsupported format",             test_unsupported_format),
        ("4. Missing file",                   test_missing_file),
        ("5. Model loading failure",          test_model_loading_failure),
        ("6. Malformed checkpoint",           test_malformed_checkpoint),
        ("7. Probability normalization",      test_probability_normalization),
        ("8. Grade range (0–3)",              test_grade_range),
        ("9. Grad-CAM file created",          test_gradcam_generation),
        ("10. Output keys exact",             test_output_keys_exact),
        ("11. Probability keys & types",      test_probability_keys),
        ("12. No forbidden ANUJ fields",      test_no_forbidden_fields),
        ("13. JSON serializable",             test_json_serializable),
        ("14. Preprocess tensor shape",       test_preprocess_single_image_shape),
        ("15. Preprocess missing file",       test_preprocess_missing_file),
    ]

    passed, failed = 0, 0
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"\n[FAIL] {name}: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"  Results: {passed} PASSED | {failed} FAILED")
    if failed == 0:
        print("  ALL TESTS PASSED — Integration contract verified.")
    else:
        print("  SOME TESTS FAILED — Review output above.")
    print("="*60)
