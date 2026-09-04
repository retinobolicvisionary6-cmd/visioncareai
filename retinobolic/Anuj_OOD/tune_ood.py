"""
OOD Detection Module — Simple Dataset Tuner
============================================
A lightweight, simple tuning script designed to calibrate thresholds
and select the best distance metric even on small datasets (5-20 images).

Usage:
    python tune_ood.py
    python tune_ood.py --id_dir sample_data/in_distribution --ood_dir sample_data/out_of_distribution
    python tune_ood.py --save   (automatically updates src/config.py with optimal settings)
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src import config
from src.embedding import extract_embedding
from src.reference import ReferenceDistribution
from src.distance import compute_distance
from src.utils import OODError

logging.basicConfig(level=logging.WARNING)


def collect_images(folder: Path) -> list[Path]:
    """Collect image paths from folder."""
    images = []
    for ext in config.SUPPORTED_EXTENSIONS:
        images.extend(folder.glob(f"*{ext}"))
        images.extend(folder.glob(f"*{ext.upper()}"))
    return sorted(set(images))


def tune_on_dataset(
    id_dir: Path,
    ood_dir: Path,
    extractor_type: str = "classical",
    save_to_config: bool = False,
) -> None:
    id_images = collect_images(id_dir)
    ood_images = collect_images(ood_dir)

    if len(id_images) < 2:
        print(f"[ERROR] Need at least 2 In-Distribution images in '{id_dir}'. Found: {len(id_images)}")
        sys.exit(1)
    if len(ood_images) < 1:
        print(f"[ERROR] Need at least 1 OOD image in '{ood_dir}'. Found: {len(ood_images)}")
        sys.exit(1)

    print("\n" + "=" * 65)
    print("      OOD MODULE TUNER - SMALL DATASET CALIBRATION")
    print("=" * 65)
    print(f"  In-Distribution images : {len(id_images)} ({id_dir})")
    print(f"  Out-of-Distribution    : {len(ood_images)} ({ood_dir})")
    print(f"  Active Extractor       : {extractor_type}")
    print("-" * 65)

    # 1. Extract ID embeddings & fit reference
    print("\n1. Fitting reference distribution on In-Distribution images...")
    id_embeddings = []
    for img in id_images:
        try:
            emb = extract_embedding(img, extractor_type=extractor_type)
            id_embeddings.append(emb)
        except Exception as e:
            print(f"   [WARN] Skipping ID image {img.name}: {e}")

    id_embeddings = np.array(id_embeddings)
    ref = ReferenceDistribution()
    ref.fit(id_embeddings, extractor_type=extractor_type)

    # Also save the updated reference files
    ref.save(config.REFERENCE_STATS_FILE, config.REFERENCE_EMBEDDINGS_FILE)
    print(f"   [OK] Reference fitted (dimensions: {ref.embedding_dim})")

    # 2. Extract OOD embeddings
    print("2. Extracting features for Out-of-Distribution images...")
    ood_embeddings = []
    for img in ood_images:
        try:
            emb = extract_embedding(img, extractor_type=extractor_type)
            ood_embeddings.append(emb)
        except Exception as e:
            print(f"   [WARN] Skipping OOD image {img.name}: {e}")
    ood_embeddings = np.array(ood_embeddings)

    # 3. Test all distance metrics
    metrics = ["mahalanobis", "cosine", "euclidean"]
    results = {}

    print("\n3. Evaluating distance metrics & separation margins:")
    print("-" * 65)
    print(f"  {'Metric':<14} {'ID Max':>10} {'OOD Min':>10} {'Gap / Margin':>14} {'Separation'}")
    print("-" * 65)

    best_metric = None
    best_ratio = -1.0
    best_recommended_threshold = 0.0

    for metric in metrics:
        id_scores = [compute_distance(emb, ref, metric=metric) for emb in id_embeddings]
        ood_scores = [compute_distance(emb, ref, metric=metric) for emb in ood_embeddings]

        id_max = float(np.max(id_scores))
        id_p99 = float(np.percentile(id_scores, 99))
        ood_min = float(np.min(ood_scores))

        gap = ood_min - id_max
        ratio = (ood_min / (id_max + 1e-12)) if id_max > 0 else 0.0

        # Suggested threshold: midpoint of the safe gap with safety buffer
        if gap > 0:
            suggested_thresh = id_max + (gap * 0.20)  # 20% into the safe gap above ID
            sep_status = "EXCELLENT" if ratio > 5 else "GOOD"
        else:
            suggested_thresh = id_p99
            sep_status = "OVERLAP"

        results[metric] = {
            "id_max": id_max,
            "ood_min": ood_min,
            "gap": gap,
            "ratio": ratio,
            "suggested_threshold": suggested_thresh,
            "status": sep_status,
        }

        print(f"  {metric:<14} {id_max:>10.4f} {ood_min:>10.4f} {gap:>14.4f}  {sep_status} ({ratio:.1f}x)")

        if ratio > best_ratio:
            best_ratio = ratio
            best_metric = metric
            best_recommended_threshold = suggested_thresh

    print("-" * 65)
    print("\n4. TUNING RECOMMENDATION:")
    print("=" * 65)
    print(f"  * Recommended Metric    : {best_metric.upper()}")
    print(f"  * Recommended Threshold : {best_recommended_threshold:.4f}")
    print(f"  * Separation Quality    : {results[best_metric]['status']} (OOD is {results[best_metric]['ratio']:.1f}x higher than ID)")
    print(f"  * Safe In-Distribution  : Any score < {best_recommended_threshold:.4f}")
    print(f"  * Triggers OOD Review   : Any score >= {best_recommended_threshold:.4f}")
    print("=" * 65)

    if save_to_config:
        _update_config_file(best_metric, best_recommended_threshold)
        print("\n[OK] Automatically updated 'src/config.py' with the recommended settings!")
    else:
        print("\nTip: To automatically apply this configuration, run:")
        print(f"  python tune_ood.py --save")
        print("\nOr update 'src/config.py' manually:")
        print(f"  DISTANCE_METRIC = \"{best_metric}\"")
        print(f"  OOD_THRESHOLD = {best_recommended_threshold:.4f}")


def _update_config_file(metric: str, threshold: float) -> None:
    """Updates src/config.py with calibrated values."""
    config_path = Path(__file__).parent / "src" / "config.py"
    content = config_path.read_text(encoding="utf-8")

    # Update metric
    import re
    content = re.sub(
        r'DISTANCE_METRIC:\s*str\s*=\s*["\'].*?["\']',
        f'DISTANCE_METRIC: str = "{metric}"',
        content
    )
    # Update threshold
    content = re.sub(
        r'OOD_THRESHOLD:\s*float\s*=\s*[0-9.]+',
        f'OOD_THRESHOLD: float = {threshold:.4f}',
        content
    )
    config_path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune OOD threshold & metric on a small dataset.")
    parser.add_argument(
        "--id_dir",
        type=Path,
        default=Path("sample_data/in_distribution"),
        help="Folder containing known in-distribution fundus images.",
    )
    parser.add_argument(
        "--ood_dir",
        type=Path,
        default=Path("sample_data/out_of_distribution"),
        help="Folder containing abnormal/non-fundus/OOD images.",
    )
    parser.add_argument(
        "--extractor",
        default=config.EXTRACTOR_TYPE,
        choices=["classical", "pretrained", "dr_model"],
        help="Feature extractor to use.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Automatically save recommended metric and threshold into src/config.py.",
    )

    args = parser.parse_args()
    tune_on_dataset(
        id_dir=args.id_dir,
        ood_dir=args.ood_dir,
        extractor_type=args.extractor,
        save_to_config=args.save,
    )


if __name__ == "__main__":
    main()
