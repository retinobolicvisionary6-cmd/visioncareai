#!/usr/bin/env python
"""
demo.py — Reliability Engine Demo
===================================
SIH Problem ID: SIH26038
Explainable AI for Diabetic Retinopathy Screening in Rural India

Demonstrates the Reliability Engine across five engineering scenarios.
Uses real Confidence and Uncertainty modules, and real OOD detection
on sample fundus images (in-distribution) and out-of-distribution images.

Usage:
    python demo.py

IMPORTANT:
    This demo is for engineering evaluation only.
    It does NOT constitute a medical diagnosis or clinical recommendation.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reliability.engine import run_reliability_pipeline
from src.reliability.fusion import calculate_reliability

# ---------------------------------------------------------------------------
# Sample image paths (from anuj-ood)
# ---------------------------------------------------------------------------
OOD_ROOT = Path("d:/anuj-ood")
APTOS_IMAGES = Path(r"E:\retinobolic\data\raw\train_images")
IN_DIST_IMAGE = APTOS_IMAGES / "00a8624548a9.png" if (APTOS_IMAGES / "00a8624548a9.png").exists() else (OOD_ROOT / "sample_data" / "in_distribution" / "fundus_01.png")
OUT_DIST_IMAGE = OOD_ROOT / "sample_data" / "out_of_distribution" / "random_noise.png"


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

_SEPARATOR = "-" * 52

_STATUS_LABELS = {
    "acceptable":      "✅  ACCEPTABLE",
    "caution":         "⚠️   CAUTION",
    "review_required": "🔴  REVIEW REQUIRED",
}


def _yn(flag: bool) -> str:
    return "YES" if flag else "NO"


def print_result(title: str, result: dict) -> None:
    print(f"\n{'=' * 52}")
    print(f"  {title}")
    print(_SEPARATOR)
    print(f"  Model Confidence   : {result['confidence'] * 100:.1f}% ({result['confidence_level'].upper()})")
    print(f"  Model Uncertainty  : {result['uncertainty'] * 100:.1f}% ({result['uncertainty_level'].upper()})")
    print(f"  OOD Detected       : {_yn(result['ood'])} (score={result['ood_score']:.3f})")
    if "reliability_score" in result:
        print(f"  Engineering Score  : {result['reliability_score']:.3f} / 1.000")
    print(_SEPARATOR)
    label = _STATUS_LABELS.get(result["reliability_status"], result["reliability_status"].upper())
    print(f"  Reliability Status : {label}")
    print(f"  Review Required    : {_yn(result['review_required'])}")
    print(_SEPARATOR)
    print(f"  Reason:")
    # Word-wrap reason at 48 chars
    words = result["reason"].split()
    line = "    "
    for word in words:
        if len(line) + len(word) + 1 > 52:
            print(line)
            line = "    " + word + " "
        else:
            line += word + " "
    if line.strip():
        print(line)
    print()


def check_images() -> tuple[bool, bool]:
    in_ok = IN_DIST_IMAGE.exists()
    out_ok = OUT_DIST_IMAGE.exists()
    if not in_ok:
        print(f"⚠  In-distribution image not found: {IN_DIST_IMAGE}")
        print("   Scenarios using OOD detection will be skipped.")
    if not out_ok:
        print(f"⚠  OOD image not found: {OUT_DIST_IMAGE}")
        print("   Scenarios requiring OOD=True will be skipped.")
    return in_ok, out_ok


# ---------------------------------------------------------------------------
# Demo scenarios
# ---------------------------------------------------------------------------

def demo_fusion_only(
    title: str,
    confidence_dict: dict,
    uncertainty_dict: dict,
    ood_dict: dict,
) -> None:
    """Run just the fusion layer (no real image needed)."""
    result = calculate_reliability(confidence_dict, uncertainty_dict, ood_dict)
    print_result(title, result)


def run_demo() -> None:
    print("\n" + "=" * 52)
    print("  RELIABILITY ENGINE — Module 4 Demo")
    print("  SIH26038: Explainable AI for DR Screening")
    print("=" * 52)
    print()
    print("  ⚠  ENGINEERING EVALUATION ONLY.")
    print("  ⚠  NOT a medical diagnosis tool.")
    print()

    in_ok, out_ok = check_images()

    # -----------------------------------------------------------------
    # Scenario 1: Fully Reliable — Acceptable
    # -----------------------------------------------------------------
    print("\n>>> SCENARIO 1 — Fully Reliable")
    print("    (High confidence, low uncertainty, in-distribution)")
    dr1 = {"grade": 2, "probabilities": {"0": 0.01, "1": 0.01, "2": 0.97, "3": 0.01}}
    if in_ok:
        result1 = run_reliability_pipeline(dr1, str(IN_DIST_IMAGE))
    else:
        # Fusion-only fallback (mock OOD)
        from src.confidence import calculate_confidence
        from src.uncertainty import calculate_uncertainty
        c = calculate_confidence(dr1)
        u = calculate_uncertainty(dr1)
        mock_ood = {"ood": False, "ood_status": "in_distribution", "ood_score": 1.50,
                    "threshold": 3.17, "distance_metric": "mahalanobis",
                    "extractor_type": "classical", "reason": "Mock in-distribution", "metadata": {}}
        result1 = calculate_reliability(c, u, mock_ood)
    print_result("Scenario 1: Fully Reliable", result1)

    # -----------------------------------------------------------------
    # Scenario 2: OOD Despite High Confidence → Review Required
    # -----------------------------------------------------------------
    print("\n>>> SCENARIO 2 — OOD Despite High Confidence")
    print("    (High confidence, low uncertainty, OOD=True)")
    dr2 = {"grade": 2, "probabilities": {"0": 0.01, "1": 0.01, "2": 0.97, "3": 0.01}}
    if out_ok:
        result2 = run_reliability_pipeline(dr2, str(OUT_DIST_IMAGE))
    else:
        from src.confidence import calculate_confidence
        from src.uncertainty import calculate_uncertainty
        c = calculate_confidence(dr2)
        u = calculate_uncertainty(dr2)
        mock_ood = {"ood": True, "ood_status": "review_required", "ood_score": 5.80,
                    "threshold": 3.17, "distance_metric": "mahalanobis",
                    "extractor_type": "classical",
                    "reason": "Input embedding is substantially distant from the reference fundus distribution.",
                    "metadata": {}}
        result2 = calculate_reliability(c, u, mock_ood)
    print_result("Scenario 2: OOD Overrides High Confidence", result2)

    # -----------------------------------------------------------------
    # Scenario 3: High Uncertainty → Review Required
    # -----------------------------------------------------------------
    print("\n>>> SCENARIO 3 — High Model Uncertainty")
    print("    (Near-uniform distribution, in-distribution image)")
    dr3 = {"grade": 1, "probabilities": {"0": 0.26, "1": 0.28, "2": 0.24, "3": 0.22}}
    if in_ok:
        result3 = run_reliability_pipeline(dr3, str(IN_DIST_IMAGE))
    else:
        from src.confidence import calculate_confidence
        from src.uncertainty import calculate_uncertainty
        c = calculate_confidence(dr3)
        u = calculate_uncertainty(dr3)
        mock_ood = {"ood": False, "ood_status": "in_distribution", "ood_score": 1.50,
                    "threshold": 3.17, "distance_metric": "mahalanobis",
                    "extractor_type": "classical", "reason": "Mock in-distribution", "metadata": {}}
        result3 = calculate_reliability(c, u, mock_ood)
    print_result("Scenario 3: High Uncertainty", result3)

    # -----------------------------------------------------------------
    # Scenario 4: Caution (intermediate signals)
    # -----------------------------------------------------------------
    print("\n>>> SCENARIO 4 — Intermediate Signals (Caution)")
    print("    (Medium confidence, medium uncertainty, in-distribution)")
    # Use fusion-only with mocked results for precise level control
    demo_fusion_only(
        "Scenario 4: Caution — Intermediate Signals",
        confidence_dict={
            "predicted_grade": 2, "predicted_class_name": "Moderate DR",
            "confidence": 0.62, "confidence_percent": 62.0,
            "confidence_level": "medium", "margin": 0.42,
        },
        uncertainty_dict={
            "predicted_grade": 2, "uncertainty": 0.52,
            "uncertainty_level": "medium", "review_recommended": False,
            "probability_margin": 0.42,
        },
        ood_dict={
            "ood": False, "ood_status": "in_distribution", "ood_score": 1.80,
            "threshold": 3.17, "distance_metric": "mahalanobis",
            "extractor_type": "classical", "reason": "In-distribution.", "metadata": {},
        },
    )

    # -----------------------------------------------------------------
    # Scenario 5: Multiple failures
    # -----------------------------------------------------------------
    print("\n>>> SCENARIO 5 — Multiple Failures")
    print("    (Low confidence, high uncertainty, OOD=True)")
    demo_fusion_only(
        "Scenario 5: Multiple Failures",
        confidence_dict={
            "predicted_grade": 0, "predicted_class_name": "No DR",
            "confidence": 0.28, "confidence_percent": 28.0,
            "confidence_level": "low", "margin": 0.03,
        },
        uncertainty_dict={
            "predicted_grade": 0, "uncertainty": 0.88,
            "uncertainty_level": "high", "review_recommended": True,
            "probability_margin": 0.03,
        },
        ood_dict={
            "ood": True, "ood_status": "review_required", "ood_score": 6.20,
            "threshold": 3.17, "distance_metric": "mahalanobis",
            "extractor_type": "classical",
            "reason": "Input embedding is substantially distant from the reference fundus distribution.",
            "metadata": {},
        },
    )

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    print("=" * 52)
    print("  DEMO COMPLETE")
    print("=" * 52)
    print()
    print("  All scenarios executed successfully.")
    print("  The Reliability Engine integrates:")
    print("    • Confidence Module  (anuj-confidence)")
    print("    • Uncertainty Module (anuj-uncertainty)")
    print("    • OOD Module         (anuj-ood)")
    print()
    print("  Next steps in pipeline:")
    print("    → Clinical Context Module")
    print("    → Final Decision Layer")
    print()
    print("  ⚠  This output is for engineering evaluation only.")
    print("  ⚠  It is NOT a medical diagnosis or clinical recommendation.")
    print()


if __name__ == "__main__":
    run_demo()
