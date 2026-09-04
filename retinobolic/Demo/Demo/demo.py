import argparse
import json
import sys
from pathlib import Path

# Add project root to sys path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from Integration.pipeline.screening_pipeline import run_pipeline

def main():
    parser = argparse.ArgumentParser(description="SIH26038 DR Screening Demo")
    parser.add_argument("--image", required=True, help="Path to fundus image")
    parser.add_argument("--age", type=int, help="Patient age")
    parser.add_argument("--bp-systolic", type=int, help="Systolic BP")
    parser.add_argument("--bp-diastolic", type=int, help="Diastolic BP")
    parser.add_argument("--hba1c", type=float, help="HbA1c level")
    parser.add_argument("--diabetes-duration", type=int, help="Diabetes duration in years")
    parser.add_argument("--checkpoint", type=str, help="Path to custom DR model checkpoint")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of formatted text")
    args = parser.parse_args()

    try:
        result = run_pipeline(
            image_path=args.image,
            age=args.age,
            bp_systolic=args.bp_systolic,
            bp_diastolic=args.bp_diastolic,
            hba1c=args.hba1c,
            diabetes_duration_years=args.diabetes_duration,
            dr_model_checkpoint=args.checkpoint
        )
        
        if args.json:
            print(json.dumps(result, indent=2))
            return
        
        # Formatted output
        quality = result.get("quality", {})
        final_decision = result.get("final_decision", {})
        
        print("\nSIH26038 DR SCREENING")
        print("=====================\n")
        
        # QUALITY
        status = quality.get("status", "unknown").upper()
        print("IMAGE QUALITY")
        print(f"Status: {status}")
        if "quality_score" in quality:
            print(f"Quality: {int(quality['quality_score']*100)}%")
        print()
        
        # If ungradable, we stop here
        if status == "UNGRADABLE":
            print("IMAGE NOT SUITABLE")
            print("🔴 UNGRADABLE")
            print("Image quality is insufficient for screening.")
            print("\n[ Recapture Image ]")
            return
            
        # DR RESULT
        dr_res = result.get("dr_result", {})
        print("DR RESULT")
        print(f"Grade: {dr_res.get('grade', 'N/A')}")
        
        rel_res = result.get("reliability", {})
        
        if "confidence" in rel_res:
            print(f"Model Confidence: {int(rel_res['confidence']*100)}%")
        if "uncertainty_level" in rel_res:
            print(f"Model Uncertainty: {rel_res['uncertainty_level'].upper()}")
        if "ood" in rel_res:
            print(f"OOD: {'YES' if rel_res['ood'] else 'NO'}")
        print()
        
        # RELIABILITY
        print("RELIABILITY")
        rel_status = rel_res.get("reliability_status", "unknown").upper()
        print(f"Status: {rel_status}\n")
        
        # XAI
        print("XAI")
        gradcam = dr_res.get("gradcam_path")
        if gradcam:
            print("Grad-CAM: GENERATED")
        else:
            print("Grad-CAM: NOT AVAILABLE")
        print()
            
        # CLINICAL CONTEXT
        clin = result.get("clinical_context", {})
        provided_fields = len(clin.get("data_quality", {}).get("provided_fields", []))
        print("CLINICAL CONTEXT")
        if provided_fields > 0:
            print("Status: PROVIDED")
        else:
            print("Status: NOT PROVIDED")
        print()
        
        # FINAL ACTION
        print("FINAL ACTION")
        print(final_decision.get("action", "unknown").upper())
        print(f"\nPriority: {final_decision.get('priority', 'normal').upper()}")
        print(f"\nReason:\n{final_decision.get('reason', 'N/A')}")
        print("Final clinical decision rests with the examining physician.\n")

    except Exception as e:
        print(f"Error during screening: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
