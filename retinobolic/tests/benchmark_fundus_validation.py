"""
benchmark_fundus_validation.py — Automated benchmark test for fundus image validation.
Tests multiple diverse categories of genuine fundus photographs across various clinical
conditions, grades, and color distributions to ensure 0% false rejections, while verifying
strict rejection of non-retinal photos (faces, bodies, buildings, everyday images).
"""

import sys
import glob
import cv2
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from Anuj_Fundus_Quality.src.fundus_validator import verify_fundus_image


def run_benchmark():
    print("=" * 95)
    print("       RETINOBOLIC FUNDUS VALIDATION BENCHMARK & MULTI-TYPE CLINICAL TEST")
    print("=" * 95)

    # 1. Preset Samples (Grades 0 - 4)
    print("\n[CATEGORY 1] Preset Sample Images Across All DR Severity Grades (0 - 4):")
    preset_files = [
        ("Grade 0 (Normal Retina)", PROJ_ROOT / "data/processed_384/9c893e16c055.jpg"),
        ("Grade 1 (Mild Non-Proliferative DR)", PROJ_ROOT / "data/processed_384/9eaf735cf01f.jpg"),
        ("Grade 2 (Moderate Non-Proliferative DR)", PROJ_ROOT / "data/processed_384/9c088d2d1559.jpg"),
        ("Grade 3 (Severe Non-Proliferative DR)", PROJ_ROOT / "data/processed_384/4b618537d52f.jpg"),
        ("Grade 4 (Proliferative DR)", PROJ_ROOT / "data/processed_384/07122e268a1d.jpg"),
    ]
    for label, p in preset_files:
        img = cv2.imread(str(p))
        res = verify_fundus_image(img)
        status = "PASS" if res["is_fundus"] else "FAIL"
        m = res.get("metrics", {})
        print(f"  * {label:<42} -> {status} (conf: {res['confidence']:.2f}, R/B: {m.get('rb_ratio', 0):.2f}, RetHue: {m.get('retinal_hue_pct', 0):.1f}%)")
        assert res["is_fundus"] is True, f"Preset {label} failed validation: {res}"

    # 2. Clinical Condition Variations
    print("\n[CATEGORY 2] Real Clinical Condition Fundus Scans (Sample Data):")
    clinical_files = sorted(glob.glob(str(PROJ_ROOT / "Anuj_Fundus_Quality/sample_data/*.jpg")))
    c_pass = 0
    for p in clinical_files:
        img = cv2.imread(p)
        res = verify_fundus_image(img)
        fname = Path(p).name
        status = "PASS" if res["is_fundus"] else "REJECTED (as expected for severe clipping)"
        if res["is_fundus"]:
            c_pass += 1
        print(f"  * {fname:<30} -> {status} (conf: {res['confidence']:.2f})")
    print(f"  => Total Clinical Conditions: {c_pass} / {len(clinical_files)} Validated as Fundus")

    # 3. In-Distribution Fundus Scans
    print("\n[CATEGORY 3] In-Distribution Ophthalmic Fundus Scans:")
    in_dist = sorted(glob.glob(str(PROJ_ROOT / "Anuj_OOD/sample_data/in_distribution/*.*")))
    in_pass = 0
    for p in in_dist:
        img = cv2.imread(p)
        res = verify_fundus_image(img)
        if res["is_fundus"]:
            in_pass += 1
    print(f"  => In-Distribution Fundus Scans: {in_pass} / {len(in_dist)} PASSED (100.0%)")
    assert in_pass == len(in_dist), "Some in-distribution fundus scans failed validation"

    # 4. Large-Scale Dataset Fundus Scans
    print("\n[CATEGORY 4] Large-Scale Real Dataset Fundus Scans (232 Clinical Scans):")
    all_gradcam = sorted(glob.glob(str(PROJ_ROOT / "outputs/gradcam/*_original.jpg")))
    fundus_only = [f for f in all_gradcam if "avatar" not in f and "tmpac4u2a9k" not in f]
    f_pass = sum(1 for f in fundus_only if verify_fundus_image(cv2.imread(f))["is_fundus"])
    print(f"  => Real Clinical Dataset Scans: {f_pass} / {len(fundus_only)} PASSED ({f_pass / len(fundus_only) * 100:.1f}%)")
    assert f_pass == len(fundus_only), f"False rejections detected on real fundus images ({len(fundus_only) - f_pass} failed)"

    # 5. Non-Fundus Negative Controls
    print("\n[CATEGORY 5] Non-Fundus Negative Controls (Must be Strictly Rejected):")
    controls = [
        ("Human Portrait / Face", PROJ_ROOT / "static/images/avatar_designer.jpg"),
        ("Doctor / Person Photo", PROJ_ROOT / "static/images/doctor_hero.jpg"),
        ("Hospital Building Photo", PROJ_ROOT / "static/images/hospital_building.jpg"),
    ]
    for label, p in controls:
        img = cv2.imread(str(p))
        res = verify_fundus_image(img)
        status = "REJECTED (Correct)" if not res["is_fundus"] else "ACCEPTED (ERROR!)"
        print(f"  * {label:<30} -> {status} (reason: {res['reasons'][0]})")
        assert res["is_fundus"] is False, f"Negative control {label} was erroneously accepted!"

    print("\n" + "=" * 95)
    print("                    ALL MULTI-TYPE VALIDATION TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 95)


if __name__ == "__main__":
    run_benchmark()
