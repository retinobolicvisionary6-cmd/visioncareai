"""
demo.py — CLI demo for the Fundus Image Quality Assessment engine.

Usage:
    python demo.py --image sample_data/good_fundus.jpg
    python demo.py --image <path_to_fundus_image.jpg>

Outputs:
    - Formatted quality report to terminal
    - JSON result saved to outputs/quality/<image_stem>_quality.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _bar(score: float, width: int = 20) -> str:
    """ASCII progress bar for a score in [0, 1]."""
    filled = int(round(score * width))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def _pct(score: float) -> str:
    return f"{score * 100:.0f}%"


def _status_icon(status: str) -> str:
    return {"good": "[GOOD]", "borderline": "[BORDERLINE]", "ungradable": "[UNGRADABLE]"}.get(
        status, status.upper()
    )


def _artifact_label(artifact_score: float) -> str:
    if artifact_score >= 0.80:
        return "Acceptable"
    elif artifact_score >= 0.55:
        return "Moderate"
    else:
        return "Significant"


def run_demo(image_path: str) -> None:
    # Import here so the CLI fails gracefully if deps are missing.
    try:
        from src.quality import assess_quality, result_to_json
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you have installed requirements: pip install -r requirements.txt")
        sys.exit(1)

    print()
    print("=" * 52)
    print("   Fundus Image Quality Assessment")
    print("=" * 52)
    print(f"   Image: {image_path}")
    print("=" * 52)
    print("   Running analysis …")
    print()

    result = assess_quality(image_path)

    if result.get("error") and result["status"] == "ungradable":
        # Load error.
        print(f"❌  ERROR: {result['error']}")
        print(f"   Action : RECAPTURE")
        print()
        return

    status = result["status"]
    print(f"   Status        : {_status_icon(status)}")
    print(f"   Image Quality : {_pct(result['quality_score'])}  {_bar(result['quality_score'])}")
    print()
    print("   Component Scores")
    print("   " + "-" * 45)
    print(f"   Focus          : {_pct(result['focus_score']):>4}  {_bar(result['focus_score'])}")
    print(f"   Illumination   : {_pct(result['illumination_score']):>4}  {_bar(result['illumination_score'])}")
    print(f"   Field of View  : {_pct(result['field_of_view_score']):>4}  {_bar(result['field_of_view_score'])}")
    print(f"   Retinal Vis.   : {_pct(result['retinal_visibility_score']):>4}  {_bar(result['retinal_visibility_score'])}")
    print(f"   Artifacts      : {_artifact_label(result['artifact_score'])}")
    print()

    if result.get("enhanced"):
        print(f"   >> Enhancement applied.")
        if result.get("enhanced_image_path"):
            print(f"      Enhanced image saved -> {result['enhanced_image_path']}")
        print()

    print("   " + "-" * 45)
    print(f"   Action  : {result['action'].upper().replace('_', ' ')}")
    print(f"   Reason  : {result['reason']}")
    print()

    # Locate the saved JSON.
    stem = Path(image_path).stem
    json_path = Path("outputs") / "quality" / f"{stem}_quality.json"
    if json_path.exists():
        print(f"   JSON result saved -> {json_path}")
    print("=" * 52)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fundus Image Quality Assessment Demo"
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to a fundus image (JPG, JPEG, or PNG)",
    )
    args = parser.parse_args()
    run_demo(args.image)


if __name__ == "__main__":
    main()
