"""
manual_test.py - Interactive manual testing tool for Module 2.
Run:
    python manual_test.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import calculate_uncertainty, UncertaintyConfig
from src.validation import InvalidProbabilityError, ValidationError

def main():
    print("=" * 60)
    print("   MANUAL UNCERTAINTY TESTER — MODULE 2 (SIH26038)")
    print("=" * 60)
    print("Classes: 0: No DR | 1: Mild DR | 2: Moderate DR | 3: Severe DR")
    print("Type 4 probabilities separated by spaces (e.g., 0.03 0.08 0.81 0.08)")
    print("Type 'q' or 'exit' to quit.\n")

    while True:
        try:
            user_input = input("Enter 4 probabilities (p0 p1 p2 p3) > ").strip()
            if user_input.lower() in ("q", "exit", "quit"):
                print("Exiting manual test.")
                break
            if not user_input:
                continue

            parts = user_input.replace(",", " ").split()
            if len(parts) != 4:
                print(f"  [!] Please enter exactly 4 numbers. Got {len(parts)}.\n")
                continue

            probs = [float(p) for p in parts]
            payload = {
                "probabilities": {str(i): p for i, p in enumerate(probs)}
            }

            result = calculate_uncertainty(payload)

            print("\n" + "-" * 50)
            print(f"  Input Probabilities : {probs}")
            print(f"  Predicted Grade     : {result['predicted_grade']}")
            print(f"  Model Uncertainty   : {result['uncertainty'] * 100:.2f}% ({result['uncertainty']})")
            print(f"  Uncertainty Level   : {result['uncertainty_level'].upper()}")
            print(f"  Review Recommended  : {'YES' if result['review_recommended'] else 'NO'}")
            print(f"  Probability Margin  : {result['probability_margin']:.4f}")
            print("-" * 50 + "\n")

        except ValueError as e:
            print(f"  [!] Number parsing error: {e}\n")
        except (InvalidProbabilityError, ValidationError) as e:
            print(f"  [!] Validation Error:\n{e}\n")
        except KeyboardInterrupt:
            print("\nExiting.")
            break

if __name__ == "__main__":
    main()
