"""
demo.py — Model Confidence Module demonstration.

Run:
    python demo.py

This script demonstrates calculate_confidence() on several example
probability distributions and illustrates correct error handling for
invalid inputs.

IMPORTANT:
    Confidence is a model-output signal.
    Model Confidence ≠ Model Accuracy ≠ Clinical Certainty.
    A model can output Confidence = 0.95 and still be wrong.
"""

import json
import sys
from pathlib import Path

# Ensure the project root is on sys.path so 'from src.confidence import ...' works
# whether this script is run from the project root or from another directory.
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.confidence import (
    ConfidenceModuleError,
    calculate_confidence,
)
from src.config import CLASS_MAPPING


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

_SEPARATOR = "-" * 55
_LEVEL_LABELS = {
    "high": "HIGH   [OK]",
    "medium": "MEDIUM [WARN]",
    "low": "LOW    [!]",
}


def print_header(title: str) -> None:
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print(f"{'=' * 55}")


def print_result(label: str, dr_result: dict) -> None:
    """Run calculate_confidence and print formatted output."""
    print(f"\n>> {label}")
    print(_SEPARATOR)

    try:
        result = calculate_confidence(dr_result, include_top2=True)

        grade = result["predicted_grade"]
        class_name = result["predicted_class_name"]
        confidence = result["confidence"]
        confidence_pct = result["confidence_percent"]
        level_raw = result["confidence_level"]
        level_display = _LEVEL_LABELS.get(level_raw, level_raw.upper())

        print(f"  Predicted Grade     : {grade}  ({class_name})")
        print(f"  Model Confidence    : {confidence_pct:.1f}%")
        print(f"  Confidence Level    : {level_display}")

        if "margin" in result:
            print(f"  Margin (top-2 gap)  : {result['margin']:.4f}")
            second_name = CLASS_MAPPING[result["second_class"]]
            print(f"  Second class        : {result['second_class']}  ({second_name})")

        print()
        print("  NOTE: Model Confidence != Clinical Certainty.")
        print("        A high confidence score does NOT guarantee a correct prediction.")

    except ConfidenceModuleError as exc:
        print(f"  [!]  Confidence module error: {exc}")
    except Exception as exc:
        print(f"  [X]  Unexpected error: {exc}")


# ---------------------------------------------------------------------------
# Demo cases
# ---------------------------------------------------------------------------

DEMO_CASES = [
    (
        "Case 1 — Normal prediction (Moderate DR, 81%)",
        {"grade": 2, "probabilities": {"0": 0.03, "1": 0.08, "2": 0.81, "3": 0.08}},
    ),
    (
        "Case 2 — High confidence (Moderate DR, 92%)",
        {"grade": 2, "probabilities": {"0": 0.02, "1": 0.03, "2": 0.92, "3": 0.03}},
    ),
    (
        "Case 3 — Medium confidence (Moderate DR, 60%)",
        {"grade": 2, "probabilities": {"0": 0.10, "1": 0.15, "2": 0.60, "3": 0.15}},
    ),
    (
        "Case 4 — Low confidence (ambiguous near-uniform)",
        {"grade": 1, "probabilities": {"0": 0.24, "1": 0.27, "2": 0.25, "3": 0.24}},
    ),
    (
        "Case 5 — Uniform distribution (minimum confidence)",
        {"grade": 0, "probabilities": {"0": 0.25, "1": 0.25, "2": 0.25, "3": 0.25}},
    ),
    (
        "Case 6 — No DR, high confidence",
        {"grade": 0, "probabilities": {"0": 0.91, "1": 0.04, "2": 0.03, "3": 0.02}},
    ),
    (
        "Case 7 — Vinayak integration format (with gradcam_path, ignored)",
        {
            "grade": 2,
            "probabilities": {"0": 0.03, "1": 0.08, "2": 0.81, "3": 0.08},
            "gradcam_path": "outputs/gradcam/image_001.jpg",
        },
    ),
]

INVALID_CASES = [
    (
        "Invalid A — Probability sum > 1 (sum = 1.3)",
        {"grade": 2, "probabilities": {"0": 0.40, "1": 0.40, "2": 0.40, "3": 0.10}},
    ),
    (
        "Invalid B — Negative probability",
        {"grade": 2, "probabilities": {"0": -0.10, "1": 0.30, "2": 0.40, "3": 0.40}},
    ),
    (
        "Invalid C — Missing class (only 3 classes)",
        {"grade": 2, "probabilities": {"0": 0.20, "1": 0.30, "2": 0.50}},
    ),
    (
        "Invalid D — NaN probability",
        {"grade": 2, "probabilities": {"0": float("nan"), "1": 0.30, "2": 0.40, "3": 0.30}},
    ),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print_header("DR MODEL CONFIDENCE - Demonstration")
    print(
        "\n  This script demonstrates the Model Confidence Module.\n"
        "  Module: src/confidence.py\n"
        "  Answers: 'How strong is the DR model's probability-based confidence?'\n"
        "\n  Model Confidence  !=  Model Accuracy  !=  Clinical Certainty.\n"
    )

    print_header("VALID PREDICTIONS")
    for label, dr_result in DEMO_CASES:
        print_result(label, dr_result)

    print_header("INVALID INPUTS - Error Handling")
    for label, dr_result in INVALID_CASES:
        print_result(label, dr_result)

    print(f"\n{'=' * 55}")
    print("  Demonstration complete.")
    print(
        "  Next modules: Uncertainty -> OOD/Reliability -> Clinical Context -> Decision Layer."
    )
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    main()
