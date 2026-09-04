"""
Calibrate OOD Detection on Real APTOS 2019 Dataset
===================================================
Uses a fast, representative sample (50-100 images) of the real APTOS 2019
dataset located at E:\\retinobolic\\data\\raw\\train_images to calibrate
the reference distribution and decision threshold with high precision.

Usage:
    python calibrate_aptos.py
    python calibrate_aptos.py --num_samples 100 --save
"""

import argparse
import json
import logging
import sys
from pathlib import Path
import numpy as np

# Project root
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src import config
from src.embedding import extract_embedding
from src.reference import ReferenceDistribution
from src.distance import compute_distance

logging.basicConfig(level=logging.WARNING)

APTOS_DIR = Path(r"E:\retinobolic\data\raw\train_images")
NATURAL_OOD_DIR = Path(r"E:\PHOTOSSS")
SYNTHETIC_OOD_DIR = ROOT / "sample_data" / "out_of_distribution"


def calibrate_aptos(
    num_samples: int = 80,
    save: bool = True,
) -> None:
    if not APTOS_DIR.exists():
        print(f"[ERROR] APTOS directory not found at: {APTOS_DIR}")
        sys.exit(1)

    print("=" * 68)
    print("       CALIBRATING OOD MODULE WITH REAL APTOS 2019 DATASET")
    print("=" * 68)

    # 1. Collect APTOS images
    all_aptos = sorted(list(APTOS_DIR.glob("*.png")) + list(APTOS_DIR.glob("*.jpg")))
    print(f"  Total APTOS images found : {len(all_aptos)}")

    # Deterministic sampling for reproducibility
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(len(all_aptos), size=min(num_samples, len(all_aptos)), replace=False)
    selected_aptos = [all_aptos[i] for i in sample_indices]
    print(f"  Calibration sample size  : {len(selected_aptos)} images (fast & efficient)")

    # 2. Collect OOD images (Real natural photos + synthetic extreme cases)
    ood_images = []
    if NATURAL_OOD_DIR.exists():
        natural_photos = sorted(list(NATURAL_OOD_DIR.glob("*.jpg")) + list(NATURAL_OOD_DIR.glob("*.png")))
        ood_images.extend(natural_photos[:15])  # Real camera photos
    if SYNTHETIC_OOD_DIR.exists():
        synth_photos = sorted(list(SYNTHETIC_OOD_DIR.glob("*.png")))
        ood_images.extend(synth_photos)  # Blank, noise, inverted, extreme artifacts

    print(f"  OOD test sample size     : {len(ood_images)} images (real-world natural photos + synthetic anomalies)")
    print("-" * 68)

    # 3. Extract In-Distribution Embeddings
    print("\n1. Extracting features from real APTOS fundus images...")
    id_embeddings = []
    for i, img_path in enumerate(selected_aptos, 1):
        try:
            emb = extract_embedding(img_path, extractor_type="classical")
            id_embeddings.append(emb)
            if i % 20 == 0 or i == len(selected_aptos):
                print(f"   Processed {i}/{len(selected_aptos)} APTOS images...")
        except Exception as e:
            print(f"   [WARN] Skipping {img_path.name}: {e}")

    id_embeddings = np.array(id_embeddings)
    print(f"   [OK] In-distribution feature matrix: {id_embeddings.shape}")

    # 4. Fit Reference Distribution
    print("\n2. Fitting Reference Distribution (Mean + Precision Matrix)...")
    ref = ReferenceDistribution()
    ref.fit(id_embeddings, extractor_type="classical")
    print(f"   [OK] Reference distribution fitted.")

    # 5. Extract OOD Embeddings
    print("\n3. Extracting features from Out-of-Distribution test set...")
    ood_embeddings = []
    for img_path in ood_images:
        try:
            emb = extract_embedding(img_path, extractor_type="classical")
            ood_embeddings.append(emb)
        except Exception as e:
            print(f"   [WARN] Skipping OOD {img_path.name}: {e}")

    ood_embeddings = np.array(ood_embeddings)
    print(f"   [OK] OOD feature matrix: {ood_embeddings.shape}")

    # 6. Evaluate Separation
    print("\n4. Evaluating Distance Metrics & Separation Margins:")
    print("-" * 68)
    print(f"  {'Metric':<14} {'APTOS Max':>10} {'OOD Min':>10} {'Gap / Margin':>14} {'Separation'}")
    print("-" * 68)

    metrics = ["mahalanobis", "cosine", "euclidean"]
    best_metric = None
    best_ratio = -1.0
    best_threshold = 0.0

    for metric in metrics:
        id_scores = [compute_distance(emb, ref, metric=metric) for emb in id_embeddings]
        ood_scores = [compute_distance(emb, ref, metric=metric) for emb in ood_embeddings]

        id_max = float(np.max(id_scores))
        id_p99 = float(np.percentile(id_scores, 99))
        ood_min = float(np.min(ood_scores))

        gap = ood_min - id_max
        ratio = (ood_min / (id_max + 1e-12)) if id_max > 0 else 0.0

        if gap > 0:
            suggested_thresh = id_max + (gap * 0.15)  # 15% safety buffer above max APTOS
            status = "EXCELLENT" if ratio > 3 else "GOOD"
        else:
            suggested_thresh = id_p99
            status = "OVERLAP"

        print(f"  {metric:<14} {id_max:>10.4f} {ood_min:>10.4f} {gap:>14.4f}  {status} ({ratio:.1f}x)")

        if ratio > best_ratio:
            best_ratio = ratio
            best_metric = metric
            best_threshold = suggested_thresh

    print("-" * 68)
    print("\n5. CALIBRATION SUMMARY & RECOMMENDATION:")
    print("=" * 68)
    print(f"  * Calibrated on         : Real APTOS 2019 Retinal Fundus Dataset")
    print(f"  * Optimal Metric        : {best_metric.upper()}")
    print(f"  * Optimal Threshold     : {best_threshold:.4f}")
    print(f"  * Real Fundus Scores    : Typically 0.0 to {np.max(id_scores):.2f}")
    print(f"  * OOD/Non-Fundus Scores : Minimum {ood_min:.2f} (up to thousands)")
    print(f"  * Separation Ratio      : {best_ratio:.1f}x clear margin")
    print("=" * 68)

    if save:
        # Save reference files
        ref.save(config.REFERENCE_STATS_FILE, config.REFERENCE_EMBEDDINGS_FILE)
        print("\n[OK] Saved calibrated reference to 'reference/reference_statistics.json'!")

        # Update config.py
        config_path = ROOT / "src" / "config.py"
        content = config_path.read_text(encoding="utf-8")
        import re
        content = re.sub(
            r'DISTANCE_METRIC:\s*str\s*=\s*["\'].*?["\']',
            f'DISTANCE_METRIC: str = "{best_metric}"',
            content
        )
        content = re.sub(
            r'OOD_THRESHOLD:\s*float\s*=\s*[0-9.]+',
            f'OOD_THRESHOLD: float = {best_threshold:.4f}',
            content
        )
        config_path.write_text(content, encoding="utf-8")
        print("[OK] Updated 'src/config.py' with calibrated APTOS threshold!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate OOD on real APTOS dataset.")
    parser.add_argument("--num_samples", type=int, default=80, help="Number of APTOS images to sample.")
    parser.add_argument("--no_save", action="store_true", help="Do not save calibration to config.")
    args = parser.parse_args()

    calibrate_aptos(num_samples=args.num_samples, save=not args.no_save)


if __name__ == "__main__":
    main()
