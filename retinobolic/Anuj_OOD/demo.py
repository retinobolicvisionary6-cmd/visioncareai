"""
OOD Detection Module — Interactive Demo
=========================================
Usage:
    python demo.py --image sample_data/in_distribution/fundus_01.png
    python demo.py --image sample_data/out_of_distribution/blank_black.png
    python demo.py --image sample_data/out_of_distribution/checkerboard.png --json
    python demo.py --image sample_data/in_distribution/fundus_01.png --metric cosine
    python demo.py --batch sample_data/in_distribution sample_data/out_of_distribution
"""

import argparse
import json
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent))

from src import config
from src.ood import detect_ood
from src.utils import OODError


def format_result(result: dict, image_path: str = "") -> str:
    """Format a detect_ood() result for clean terminal output."""
    ood = result["ood"]
    score = result["ood_score"]
    threshold = result["threshold"]
    status = result["ood_status"]
    reason = result["reason"]
    metric = result["distance_metric"]
    extractor = result["extractor_type"]

    sep = "-" * 52

    if image_path:
        header = f"\nOOD ANALYSIS  |  {Path(image_path).name}"
    else:
        header = "\nOOD ANALYSIS"

    status_label = "REVIEW REQUIRED" if ood else "IN-DISTRIBUTION"
    review_flag = "YES" if ood else "NO"

    output = f"""
{sep}
{header}
{sep}

  OOD Score   : {score:.6f}
  Threshold   : {threshold:.6f}
  Metric      : {metric}
  Extractor   : {extractor}

  Status      : {status_label}
  Review Req. : {review_flag}

  Reason:
    {reason}

{sep}
"""
    return output


def run_single(args: argparse.Namespace) -> None:
    """Run OOD detection on a single image."""
    try:
        result = detect_ood(
            args.image,
            threshold=args.threshold,
            metric=args.metric,
            extractor_type=args.extractor,
        )
    except OODError as exc:
        print(f"\n[ERROR] OOD detection failed: {exc}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(format_result(result, image_path=args.image))


def run_batch(args: argparse.Namespace) -> None:
    """Run OOD detection on all images in one or more directories."""
    all_dirs = args.batch
    all_images = []
    for d in all_dirs:
        p = Path(d)
        if not p.is_dir():
            print(f"[WARN] Not a directory: {d}", file=sys.stderr)
            continue
        for ext in config.SUPPORTED_EXTENSIONS:
            all_images.extend(sorted(p.glob(f"*{ext}")))

    if not all_images:
        print("[ERROR] No images found in specified directories.", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"BATCH OOD ANALYSIS  |  {len(all_images)} images")
    print(f"{'='*60}")
    print(f"  {'Image':<35} {'Score':>10}  {'Status'}")
    print(f"  {'-'*35} {'-'*10}  {'-'*16}")

    n_ood = 0
    results = []
    for img_path in all_images:
        try:
            r = detect_ood(
                img_path,
                threshold=args.threshold,
                metric=args.metric,
                extractor_type=args.extractor,
            )
            status_short = "REVIEW REQUIRED" if r["ood"] else "in-distribution"
            if r["ood"]:
                n_ood += 1
            print(f"  {img_path.name:<35} {r['ood_score']:>10.4f}  {status_short}")
            results.append({"image": str(img_path), **r})
        except Exception as exc:
            print(f"  {img_path.name:<35} {'ERROR':>10}  {exc}")

    print(f"\n  Summary: {n_ood}/{len(all_images)} images flagged as REVIEW REQUIRED")
    print(f"  Threshold: {args.threshold or config.OOD_THRESHOLD:.4f}")
    print(f"{'='*60}")

    if args.json:
        print("\n--- JSON output ---")
        print(json.dumps(results, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OOD Detection Demo — Module 3 (SIH26038)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py --image sample_data/in_distribution/fundus_01.png
  python demo.py --image sample_data/out_of_distribution/blank_black.png
  python demo.py --image sample_data/in_distribution/fundus_01.png --json
  python demo.py --batch sample_data/in_distribution sample_data/out_of_distribution
  python demo.py --image sample_data/in_distribution/fundus_01.png --metric cosine --threshold 0.1
        """,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image", metavar="PATH", help="Path to a single image to analyse.")
    mode.add_argument(
        "--batch",
        metavar="DIR",
        nargs="+",
        help="One or more directories of images to run in batch.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=f"OOD threshold (default: config.OOD_THRESHOLD = {config.OOD_THRESHOLD}).",
    )
    parser.add_argument(
        "--metric",
        default=None,
        choices=["mahalanobis", "cosine", "euclidean", "nearest"],
        help=f"Distance metric (default: {config.DISTANCE_METRIC}).",
    )
    parser.add_argument(
        "--extractor",
        default=None,
        choices=["classical", "pretrained", "dr_model"],
        help=f"Embedding extractor (default: {config.EXTRACTOR_TYPE}).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON.",
    )

    args = parser.parse_args()

    if args.image:
        run_single(args)
    else:
        run_batch(args)


if __name__ == "__main__":
    main()
