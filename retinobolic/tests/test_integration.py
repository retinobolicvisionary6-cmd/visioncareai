"""
Integration Contract Test for Anuj / Downstream Integration.
Simulates a test retina image, runs predict(), and validates the strict output contract.
"""
from pathlib import Path
import json
import numpy as np
import cv2


import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.config import CLASS_NAMES
from src.inference import predict


def create_dummy_fundus_image(output_path: Path):
    """
    Creates a synthetic retinal fundus test image (circular fundus with blood vessels).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    
    # Draw circular retina (orange-reddish fundus background)
    cv2.circle(img, (256, 256), 220, (30, 80, 200), -1)
    
    # Draw optic disc (bright yellowish circle)
    cv2.circle(img, (160, 256), 35, (100, 220, 240), -1)
    
    # Draw sample retinal vessels (curved dark lines)
    cv2.line(img, (160, 256), (360, 180), (10, 30, 120), 4)
    cv2.line(img, (160, 256), (340, 340), (10, 30, 120), 3)
    cv2.line(img, (250, 220), (320, 120), (10, 30, 120), 2)
    
    # Add slight gaussian blur
    img = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.imwrite(str(output_path), img)
    return output_path


def test_inference_integration_contract():
    """
    Validates that predict() conforms strictly to the integration JSON contract.
    """
    dummy_img_path = Path(__file__).resolve().parent / "sample_test_retina.jpg"
    create_dummy_fundus_image(dummy_img_path)

    # Call inference
    result = predict(str(dummy_img_path))
    print("\n--- Output Contract Result ---")
    print(json.dumps(result, indent=2))

    # 1. Verify required keys
    assert "grade" in result, "Missing 'grade' field"
    assert "grade_name" in result, "Missing 'grade_name' field"
    assert "probabilities" in result, "Missing 'probabilities' field"
    assert "gradcam_path" in result, "Missing 'gradcam_path' field"

    # 2. Verify types & values
    assert isinstance(result["grade"], int), "'grade' must be an integer"
    assert result["grade"] in [0, 1, 2, 3, 4], f"Invalid grade: {result['grade']}"
    assert result["grade_name"] in CLASS_NAMES.values(), f"Invalid grade name: {result['grade_name']}"

    # 3. Verify probabilities dictionary
    probs = result["probabilities"]
    for i in ["0", "1", "2", "3", "4"]:
        assert i in probs, f"Missing class '{i}' in probabilities"
        assert isinstance(probs[i], float), f"Probability for class {i} is not float"
    
    total_prob = sum(probs.values())
    assert abs(total_prob - 1.0) < 0.05, f"Probabilities do not sum to 1.0 (Sum: {total_prob})"

    # 4. Verify Grad-CAM file output
    gradcam_path = Path(result["gradcam_path"])
    assert gradcam_path.exists(), f"Grad-CAM file was not created at: {gradcam_path}"
    assert gradcam_path.stat().st_size > 0, "Grad-CAM file is empty"

    # 5. Verify NO forbidden fields (belonging to Anuj) are present
    forbidden_fields = ["quality", "uncertainty", "ood", "action", "priority"]
    for field in forbidden_fields:
        assert field not in result, f"Prohibited field '{field}' found in inference result! Belongs to Anuj."

    print("\n[SUCCESS] Integration Contract Test Passed Perfectly!")


if __name__ == "__main__":
    test_inference_integration_contract()
