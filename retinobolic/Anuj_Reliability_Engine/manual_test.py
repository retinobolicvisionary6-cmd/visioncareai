"""
manual_test.py — Interactive Manual Testing Tool for Module 4 (Reliability Engine)
==================================================================================
SIH Problem ID: SIH26038
Explainable AI for Diabetic Retinopathy Screening in Rural India

Run:
    python manual_test.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reliability.engine import run_reliability_pipeline
from src.reliability.fusion import calculate_reliability

OOD_ROOT = Path("d:/anuj-ood")
APTOS_IMAGES = Path(r"E:\retinobolic\data\raw\train_images")
SAMPLE_IN_DIST = APTOS_IMAGES / "00a8624548a9.png" if (APTOS_IMAGES / "00a8624548a9.png").exists() else (OOD_ROOT / "sample_data" / "in_distribution" / "fundus_01.png")
SAMPLE_OOD = OOD_ROOT / "sample_data" / "out_of_distribution" / "random_noise.png"


def print_banner():
    print("\n" + "=" * 65)
    print("   MANUAL RELIABILITY ENGINE TESTER — MODULE 4 (SIH26038)")
    print("=" * 65)
    print("Classes: 0: No DR | 1: Mild DR | 2: Moderate DR | 3: Severe DR")
    print("-----------------------------------------------------------------")


def print_result_card(res: dict):
    status_icons = {
        "acceptable": "[OK] ACCEPTABLE",
        "caution": "[!] CAUTION",
        "review_required": "[X] REVIEW REQUIRED",
    }
    status = res.get("reliability_status", "unknown")
    icon_status = status_icons.get(status, status.upper())

    print("\n" + "-" * 65)
    print(f"  RELIABILITY DECISION : {icon_status}")
    print(f"  Review Required      : {'YES' if res.get('review_required') else 'NO'}")
    print(f"  Engineering Score    : {res.get('reliability_score', 0.0):.4f} / 1.0000")
    print("-" * 65)
    print("  FUSED SIGNALS:")
    print(f"    • Confidence   : {res.get('confidence', 0.0) * 100:.1f}% ({res.get('confidence_level', '').upper()})")
    print(f"    • Uncertainty  : {res.get('uncertainty', 0.0) * 100:.1f}% ({res.get('uncertainty_level', '').upper()})")
    print(f"    • OOD Status   : {'OOD DETECTED' if res.get('ood') else 'IN DISTRIBUTION'} (score={res.get('ood_score', 0.0):.3f})")
    print("-" * 65)
    print(f"  Reason:")
    print(f"    {res.get('reason', '')}")
    print("-" * 65 + "\n")


def run_preset(preset_num: int):
    presets = {
        1: {
            "name": "Fully Reliable (High Conf + Low Uncert + In-Dist)",
            "probs": [0.01, 0.01, 0.97, 0.01],
            "img": str(SAMPLE_IN_DIST),
        },
        2: {
            "name": "OOD Override (High Conf + Low Uncert + OOD Image)",
            "probs": [0.01, 0.01, 0.97, 0.01],
            "img": str(SAMPLE_OOD),
        },
        3: {
            "name": "High Uncertainty (Borderline/Confused probabilities)",
            "probs": [0.26, 0.28, 0.24, 0.22],
            "img": str(SAMPLE_IN_DIST),
        },
        4: {
            "name": "Caution (Moderate probabilities)",
            "probs": [0.10, 0.25, 0.55, 0.10],
            "img": str(SAMPLE_IN_DIST),
        },
    }

    item = presets.get(preset_num)
    if not item:
        print("Invalid preset number.")
        return

    print(f"\n>>> Running Preset {preset_num}: {item['name']}")
    dr_payload = {
        "grade": int(max(range(4), key=lambda i: item["probs"][i])),
        "probabilities": {str(i): p for i, p in enumerate(item["probs"])},
    }
    try:
        res = run_reliability_pipeline(dr_payload, item["img"])
        print_result_card(res)
    except Exception as e:
        print(f"Error executing pipeline: {e}")


def run_custom():
    print("\n--- Custom Probability & Image Input ---")
    print("Enter 4 probabilities summing to 1.0 (e.g. 0.02 0.03 0.90 0.05)")
    val = input("Probabilities (p0 p1 p2 p3) > ").strip()
    if not val:
        return

    parts = val.replace(",", " ").split()
    if len(parts) != 4:
        print(f"Error: Expected 4 probabilities, got {len(parts)}.")
        return

    try:
        probs = [float(p) for p in parts]
    except ValueError:
        print("Error: All values must be valid numbers.")
        return

    print("\nSelect Image Source:")
    print("  1. Use sample In-Distribution fundus image")
    print("  2. Use sample Out-of-Distribution noise image")
    print("  3. Enter custom image path")
    choice = input("Choice (1/2/3) [default=1] > ").strip() or "1"

    if choice == "1":
        img_path = str(SAMPLE_IN_DIST)
    elif choice == "2":
        img_path = str(SAMPLE_OOD)
    elif choice == "3":
        img_path = input("Enter full image path > ").strip().strip('"')
    else:
        print("Invalid choice, defaulting to sample in-distribution image.")
        img_path = str(SAMPLE_IN_DIST)

    dr_payload = {
        "grade": int(max(range(4), key=lambda i: probs[i])),
        "probabilities": {str(i): p for i, p in enumerate(probs)},
    }

    try:
        res = run_reliability_pipeline(dr_payload, img_path)
        print_result_card(res)
    except Exception as e:
        print(f"\n[!] Pipeline Error: {e}\n")


def main():
    print_banner()
    while True:
        print("OPTIONS:")
        print("  1. Preset 1 (Fully Reliable Case)")
        print("  2. Preset 2 (OOD Case with High Confidence)")
        print("  3. Preset 3 (High Uncertainty Case)")
        print("  4. Preset 4 (Caution / Intermediate Case)")
        print("  5. Enter Custom Probabilities & Image")
        print("  q. Quit")
        choice = input("\nSelect an option > ").strip().lower()

        if choice in ("q", "quit", "exit"):
            print("Exiting Manual Reliability Tester.")
            break
        elif choice in ("1", "2", "3", "4"):
            run_preset(int(choice))
        elif choice == "5":
            run_custom()
        else:
            print("Invalid option. Please enter 1-5 or 'q'.\n")


if __name__ == "__main__":
    main()
