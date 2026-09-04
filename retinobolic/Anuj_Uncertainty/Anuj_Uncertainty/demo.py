"""
demo.py - Interactive demonstration of the DR Prediction Uncertainty Engine.

Run with:
    python demo.py

This script does NOT require any trained DR model.
It consumes example probability distributions to demonstrate all
features of the Uncertainty Engine in a readable format.
"""

import math
import sys
import os

# Allow running from the project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import calculate_uncertainty, UncertaintyConfig
from src.calibration import TemperatureScaler
from src.validation import InvalidProbabilityError, ValidationError

DIVIDER = "-" * 52
HEADER  = "=" * 52


def fmt_pct(val: float) -> str:
    return f"{val * 100:.1f}%"


def fmt_grade(grade: int) -> str:
    labels = {0: "No DR", 1: "Mild DR", 2: "Moderate DR", 3: "Severe/Proliferative DR"}
    return f"{grade} ({labels.get(grade, 'Unknown')})"


def print_result(scenario: str, result: dict, probs: dict) -> None:
    print(f"\n{HEADER}")
    print(f"  SCENARIO: {scenario}")
    print(HEADER)
    print(f"  Probabilities : {probs}")
    print(DIVIDER)
    print(f"  Predicted Grade    : {fmt_grade(result['predicted_grade'])}")
    print(f"  Model Uncertainty  : {fmt_pct(result['uncertainty'])}  ({result['uncertainty']})")
    print(f"  Uncertainty Level  : {result['uncertainty_level'].upper()}")
    print(f"  Review Recommended : {'YES  *** Prediction is uncertain; additional review is recommended. ***' if result['review_recommended'] else 'NO'}")
    print(f"  Probability Margin : {result['probability_margin']:.4f}  (top - 2nd highest)")
    if "confidence" in result:
        print(f"  Confidence (Mod 1) : {fmt_pct(result['confidence'])}  (from Module 1)")
    print(DIVIDER)


def demo_scenario(title, dr_result, config=None, confidence=None):
    result = calculate_uncertainty(dr_result, config=config, confidence=confidence)
    print_result(title, result, dr_result["probabilities"])
    return result


def main():
    print(f"\n{HEADER}")
    print("   DR MODEL UNCERTAINTY ENGINE — DEMO")
    print("   Module 2 | SIH26038 | Anuj")
    print(HEADER)

    print("""
SEMANTIC DISTINCTION
--------------------
  Confidence (Module 1):
      How dominant is the TOP predicted class?
      = max(probabilities)

  Uncertainty (THIS module):
      How spread out / ambiguous is the FULL distribution?
      = normalized Shannon entropy = H(P) / ln(4)
      Range: 0 (clear) to 1 (maximally ambiguous)

NOTE: These thresholds are PROTOTYPE ENGINEERING THRESHOLDS.
They are NOT clinically validated. Validation on representative
data is required before clinical or screening deployment.
""")

    # ----------------------------------------------------------------
    # Scenario 1: Clear Moderate DR (spec example)
    # ----------------------------------------------------------------
    demo_scenario(
        "1. Clear Moderate DR (Spec Example)",
        {
            "grade": 2,
            "probabilities": {"0": 0.03, "1": 0.08, "2": 0.81, "3": 0.08},
            "gradcam_path": "outputs/gradcam/image_001.jpg",
        },
        confidence=0.81,
    )

    # ----------------------------------------------------------------
    # Scenario 2: Low uncertainty (strongly concentrated)
    # ----------------------------------------------------------------
    demo_scenario(
        "2. Strongly Concentrated — Uncertainty near 0",
        {
            "grade": 2,
            "probabilities": {"0": 0.001, "1": 0.001, "2": 0.997, "3": 0.001},
        },
    )

    # ----------------------------------------------------------------
    # Scenario 3: Medium uncertainty (ambiguous)
    # ----------------------------------------------------------------
    demo_scenario(
        "3. Ambiguous Prediction — Medium Uncertainty",
        {
            "probabilities": {"0": 0.10, "1": 0.15, "2": 0.60, "3": 0.15},
        },
    )

    # ----------------------------------------------------------------
    # Scenario 4: Maximum uncertainty (uniform)
    # ----------------------------------------------------------------
    demo_scenario(
        "4. Uniform Distribution — MAXIMUM Uncertainty",
        {
            "probabilities": {"0": 0.25, "1": 0.25, "2": 0.25, "3": 0.25},
        },
    )
    print("  TIE-BREAKING NOTE: When multiple classes share the maximum")
    print("  probability (e.g. uniform distribution), the predicted grade")
    print("  is the FIRST maximum index (grade=0). This is deterministic")
    print("  behavior from numpy.argmax and is documented.")

    # ----------------------------------------------------------------
    # Scenario 5: Custom thresholds
    # ----------------------------------------------------------------
    print(f"\n{HEADER}")
    print("  SCENARIO: 5. Custom Strict Thresholds")
    print(HEADER)
    strict_config = UncertaintyConfig(
        LOW_UNCERTAINTY_MAX=0.20,
        HIGH_UNCERTAINTY_MIN=0.40,
    )
    dr_result_5 = {
        "grade": 2,
        "probabilities": {"0": 0.03, "1": 0.08, "2": 0.81, "3": 0.08},
    }
    result_5 = calculate_uncertainty(dr_result_5, config=strict_config)
    print(f"  Config: LOW < {strict_config.LOW_UNCERTAINTY_MAX}, HIGH >= {strict_config.HIGH_UNCERTAINTY_MIN}")
    print(f"  Input:  {dr_result_5['probabilities']}")
    print(DIVIDER)
    print(f"  Uncertainty       : {fmt_pct(result_5['uncertainty'])}")
    print(f"  Level             : {result_5['uncertainty_level'].upper()}")
    print(f"  Review Recommended: {'YES' if result_5['review_recommended'] else 'NO'}")
    print(f"  (Same probabilities as Scenario 1, but stricter thresholds)")
    print(DIVIDER)

    # ----------------------------------------------------------------
    # Scenario 6: Optional temperature scaling
    # ----------------------------------------------------------------
    print(f"\n{HEADER}")
    print("  SCENARIO: 6. Optional Temperature Scaling (T=2.0)")
    print(HEADER)
    raw_logits = [4.5, 0.8, 0.3, -0.2]
    scaler = TemperatureScaler(temperature=2.0)
    cal_probs = scaler.scale_logits(raw_logits)
    cal_probs_dict = {str(i): round(float(p), 5) for i, p in enumerate(cal_probs)}
    dr_result_6 = {"probabilities": cal_probs_dict}
    result_6 = calculate_uncertainty(dr_result_6)
    print(f"  Raw logits         : {raw_logits}")
    print(f"  Calibrated probs   : {cal_probs_dict}")
    print(DIVIDER)
    print(f"  Model Uncertainty  : {fmt_pct(result_6['uncertainty'])}")
    print(f"  Uncertainty Level  : {result_6['uncertainty_level'].upper()}")
    print(DIVIDER)
    print("  NOTE: Temperature scaling is OPTIONAL and requires offline")
    print("  fitting on a calibration dataset. Never fit during inference.")

    # ----------------------------------------------------------------
    # Scenario 7: Validation error handling
    # ----------------------------------------------------------------
    print(f"\n{HEADER}")
    print("  SCENARIO: 7. Graceful Validation Error Handling")
    print(HEADER)
    bad_cases = [
        ("Sum mismatch",        {"probabilities": {"0": 0.4, "1": 0.4, "2": 0.4, "3": 0.1}}),
        ("Negative probability",{"probabilities": {"0": -0.1, "1": 0.3, "2": 0.4, "3": 0.4}}),
        ("NaN probability",     {"probabilities": {"0": float("nan"), "1": 0.33, "2": 0.33, "3": 0.34}}),
    ]
    for label, bad_result in bad_cases:
        try:
            calculate_uncertainty(bad_result)
            print(f"  [{label}]: FAILED - should have raised error!")
        except (InvalidProbabilityError, ValidationError) as e:
            first_line = str(e).split("\n")[0]
            print(f"  [{label}]:")
            print(f"    -> {first_line}")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print(f"\n{HEADER}")
    print("   DEMO COMPLETE")
    print(HEADER)
    print("""
  Module outputs can be combined later with:
    confidence (Module 1)    — how dominant is the top class?
    uncertainty (Module 2)   — how ambiguous is the distribution?   [THIS MODULE]
    OOD score (Module 3)     — is this image from a known domain?
         +
    Camera Reliability
    Clinical Context
         |
    Decision / Reliability Layer
""")


if __name__ == "__main__":
    main()
