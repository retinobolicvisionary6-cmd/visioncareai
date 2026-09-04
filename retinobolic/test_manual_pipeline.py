"""
Manual Pipeline Tester — Image Quality Gate + DR Model + Grad-CAM Explainability.
"""

import sys
import json
import argparse
from pathlib import Path

VINAYAK_ROOT = Path(r"C:\Users\asus\Desktop\html\c3lite\retinobolic")
ANUJ_ROOT = Path(r"C:\anuj-fundus-quality")

# 1. Import Vinayak
sys.path.insert(0, str(VINAYAK_ROOT))
from configs.config import CLASS_NAMES
import src.inference
vinayak_predict = src.inference.predict

# 2. Swap sys.path to load Anuj's src package cleanly
sys.path.remove(str(VINAYAK_ROOT))
sys.path.insert(0, str(ANUJ_ROOT))
if 'src' in sys.modules:
    del sys.modules['src']
if 'src.quality' in sys.modules:
    del sys.modules['src.quality']

import src.quality
assess_quality = src.quality.assess_quality

# Re-insert Vinayak root for any file references
sys.path.insert(0, str(VINAYAK_ROOT))


def run_manual_test(image_path: str):
    image_path = Path(image_path).resolve()
    if not image_path.exists():
        print(f"\n[ERROR] Image not found: {image_path}")
        return

    print("=" * 70)
    print("  VISIONARY6 — MANUAL INTEGRATED TEST")
    print("  (Image Quality Gate + DR Model + Grad-CAM)")
    print(f"  Input Image: {image_path.name}")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. ANUJ'S IMAGE QUALITY MODULE
    # -------------------------------------------------------------
    print("\n[Step 1] Running Anuj's Image Quality Assessment Gate...")
    q_result = assess_quality(str(image_path))
    
    q_status = str(q_result.get("status", "unknown")).upper()
    q_score = q_result.get("quality_score", 0.0)
    
    print(f"  - Quality Status     : {q_status}")
    print(f"  - Total Quality Score: {q_score:.4f} / 1.0")
    print(f"  - Focus Score        : {q_result.get('focus_score', 0.0):.4f}")
    print(f"  - Illumination Score : {q_result.get('illumination_score', 0.0):.4f}")
    print(f"  - Field of View (FOV): {q_result.get('field_of_view_score', 0.0):.4f}")
    print(f"  - Retinal Visibility : {q_result.get('retinal_visibility_score', 0.0):.4f}")
    print(f"  - Artifact Score     : {q_result.get('artifact_score', 0.0):.4f}")
    print(f"  - Assessment Reason  : {q_result.get('reason', 'N/A')}")

    # Check if ungradable
    if q_result.get("status") == "ungradable":
        print("\n" + "=" * 70)
        print("  [HARD QUALITY GATE TRIGGERED] IMAGE IS UNGRADABLE")
        print("  Action: Recapture image (image quality is insufficient for DR screening)")
        print("=" * 70)
        return

    # Check if enhanced
    image_to_screen = str(image_path)
    if q_result.get("enhanced") and q_result.get("enhanced_image_path"):
        image_to_screen = q_result["enhanced_image_path"]
        print(f"\n  [ENHANCEMENT] Borderline image recovered using enhancement!")
        print(f"  - Enhanced Image Path: {image_to_screen}")

    # -------------------------------------------------------------
    # 2. VINAYAK'S DR CLASSIFICATION + GRAD-CAM
    # -------------------------------------------------------------
    print("\n[Step 2] Running Vinayak's DR Model & Grad-CAM on GPU...")
    dr_result = vinayak_predict(
        image_path=image_to_screen,
        generate_gradcam=True
    )

    pred_grade = dr_result["grade"]
    grade_label = CLASS_NAMES.get(pred_grade, f"Class {pred_grade}")
    probs = dr_result["probabilities"]
    gradcam_path = dr_result["gradcam_path"]

    print(f"\n  [PREDICTION] Grade {pred_grade} ({grade_label})")
    print("  - Class Probability Distribution:")
    for cls_idx, p_val in probs.items():
        name = CLASS_NAMES.get(int(cls_idx), f"Class {cls_idx}")
        bar = "#" * int(p_val * 30)
        print(f"     [{cls_idx}] {name:<14}: {p_val * 100:5.1f}%  | {bar}")

    print(f"\n[Step 3] Grad-CAM Explainability Map:")
    print(f"  - Overlay file saved : {gradcam_path}")

    # -------------------------------------------------------------
    # Summary Table
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PIPELINE OUTPUT SUMMARY (Manual Test)")
    print("=" * 70)
    summary_output = {
        "input_image": str(image_path),
        "quality_gate": {
            "status": q_result.get("status"),
            "quality_score": round(q_score, 4),
            "reason": q_result.get("reason"),
            "enhanced": q_result.get("enhanced", False)
        },
        "dr_model": {
            "grade": pred_grade,
            "grade_name": grade_label,
            "probabilities": probs
        },
        "explainability": {
            "gradcam_path": gradcam_path
        }
    }
    print(json.dumps(summary_output, indent=2))
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual Pipeline Test")
    parser.add_argument("image_path", nargs="?", default=None, help="Path to fundus image")
    args = parser.parse_args()

    test_img = args.image_path
    if not test_img:
        sample_from_dataset = Path(r"C:\Users\asus\Desktop\html\c3lite\retinobolic\data\raw\train_images\92d8a7c8e718.png")
        if sample_from_dataset.exists():
            test_img = str(sample_from_dataset)
        else:
            test_img = str(VINAYAK_ROOT / "tests" / "sample_test_retina.jpg")

    run_manual_test(test_img)
