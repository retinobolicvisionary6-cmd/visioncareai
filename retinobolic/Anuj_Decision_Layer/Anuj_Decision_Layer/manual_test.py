"""
manual_test.py — Interactive / CLI Manual Tester for Decision Layer.

Usage:
    # Run interactive menu:
    python manual_test.py

    # Or run with a sample file:
    python manual_test.py --file sample_data/refer.json
    python manual_test.py --file sample_data/recapture.json
    python manual_test.py --file sample_data/conflict_cases.json
"""

import argparse
import json
import sys
from pathlib import Path
from src import make_final_decision

def run_test(quality_result, dr_result, reliability_result, clinical_context=None):
    decision = make_final_decision(
        quality_result=quality_result,
        dr_result=dr_result,
        reliability_result=reliability_result,
        clinical_context=clinical_context,
    )
    print("\n" + "="*50)
    print("         FINAL DECISION OUTPUT")
    print("="*50)
    print(json.dumps(decision, indent=2))
    print("="*50 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Manual Decision Layer Tester")
    parser.add_argument("--file", help="Path to sample json file with module outputs", default=None)
    args = parser.parse_args()

    if args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"File not found: {path}")
            sys.exit(1)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if isinstance(data, list):
            for i, case in enumerate(data):
                print(f"\n--- Testing Case {i+1}: {case.get('_name', '')} ---")
                run_test(
                    case["quality_result"],
                    case["dr_result"],
                    case["reliability_result"],
                    case.get("clinical_context"),
                )
        else:
            run_test(
                data["quality_result"],
                data["dr_result"],
                data["reliability_result"],
                data.get("clinical_context"),
            )
    else:
        # Default quick test
        print("Running quick sample test (Moderate DR, Grade 2, Reliable):")
        q = {"status": "good", "quality_score": 0.90, "action": "continue", "reason": "Suitable"}
        dr = {"grade": 2, "probabilities": {"0": 0.02, "1": 0.04, "2": 0.88, "3": 0.06}, "gradcam_path": "outputs/gradcam/sample.jpg"}
        rel = {
            "reliability_status": "acceptable", "review_required": False, "reason": "Acceptable",
            "confidence": 0.88, "confidence_level": "high",
            "uncertainty": 0.22, "uncertainty_level": "low",
            "ood": False, "ood_status": "in_distribution", "ood_score": 1.5, "reliability_score": 0.85
        }
        run_test(q, dr, rel)

if __name__ == "__main__":
    main()
