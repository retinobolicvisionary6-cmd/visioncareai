"""
OOD Detection Module — Reference Distribution Builder
======================================================
CLI utility to pre-compute and save the reference distribution from a folder
of in-distribution fundus images.

Usage
-----
    python build_reference.py \\
        --data_dir  sample_data/in_distribution \\
        --output_dir reference \\
        --extractor  classical \\
        --threshold_percentile 99

Output files
------------
    reference/reference_statistics.json   — serialised distribution statistics
    reference/reference_embeddings.npy    — distance array (for future kNN methods)

PROTOTYPE NOTE
--------------
The threshold suggested by this script is computed from the p99 of the
*in-distribution* reference distances.  This is a starting point only.
Calibrate against real out-of-distribution samples before clinical deployment.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent))

from src import config
from src.embedding import extract_embedding, get_extractor
from src.reference import ReferenceDistribution
from src.utils import ImageLoadError, EmbeddingError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_reference")


def collect_images(data_dir: Path) -> list[Path]:
    """Collect all supported image files from *data_dir*."""
    images = []
    for ext in config.SUPPORTED_EXTENSIONS:
        images.extend(data_dir.glob(f"*{ext}"))
        images.extend(data_dir.glob(f"*{ext.upper()}"))
    return sorted(set(images))


def build_reference(
    data_dir: Path,
    output_dir: Path,
    extractor_type: str,
    threshold_percentile: float,
) -> None:
    images = collect_images(data_dir)

    if not images:
        log.error("No images found in '%s'. Supported: %s", data_dir, config.SUPPORTED_EXTENSIONS)
        sys.exit(1)

    log.info("Found %d reference images in '%s'.", len(images), data_dir)
    log.info("Extractor: %s", extractor_type)

    embeddings = []
    failed = []

    for img_path in images:
        try:
            emb = extract_embedding(img_path, extractor_type=extractor_type)
            embeddings.append(emb)
            log.debug("  ✓ %s  (dim=%d)", img_path.name, emb.shape[0])
        except (ImageLoadError, EmbeddingError) as exc:
            log.warning("  ✗ Skipped '%s': %s", img_path.name, exc)
            failed.append(img_path)

    if len(embeddings) < 2:
        log.error(
            "Need at least 2 valid embeddings to fit reference distribution; "
            "only %d succeeded.",
            len(embeddings),
        )
        sys.exit(1)

    embeddings_array = np.stack(embeddings)  # (n, d)
    log.info("Fitting reference distribution: n=%d, d=%d.", *embeddings_array.shape)

    ref = ReferenceDistribution()
    ref.fit(embeddings_array, extractor_type=extractor_type)

    output_dir = Path(output_dir)
    ref.save(
        stats_path=output_dir / "reference_statistics.json",
        embeddings_path=output_dir / "reference_embeddings.npy",
    )

    # Suggest threshold
    suggested_threshold = ref.percentiles.get(
        f"p{int(threshold_percentile)}", ref.percentiles["p99"]
    )

    print("\n" + "=" * 60)
    print("REFERENCE DISTRIBUTION BUILT")
    print("=" * 60)
    print(f"  Reference images      : {len(embeddings)} / {len(images)}")
    print(f"  Skipped (errors)      : {len(failed)}")
    print(f"  Embedding dimensions  : {embeddings_array.shape[1]}")
    print(f"  Extractor             : {extractor_type}")
    print()
    print("  In-distribution distance percentiles:")
    for k, v in ref.percentiles.items():
        print(f"    {k:>5} : {v:.6f}")
    print()
    print(f"  [!] PROTOTYPE THRESHOLD ({threshold_percentile}th pct): {suggested_threshold:.6f}")
    print()
    print("  To use this threshold, set in src/config.py:")
    print(f"    OOD_THRESHOLD = {suggested_threshold:.6f}")
    print()
    print("  [NOTE] Validate this threshold on representative OOD samples")
    print("    before using in a clinical pipeline.")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build OOD reference distribution from in-distribution fundus images."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("sample_data/in_distribution"),
        help="Directory containing reference in-distribution images.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("reference"),
        help="Directory to save reference_statistics.json and reference_embeddings.npy.",
    )
    parser.add_argument(
        "--extractor",
        default=config.EXTRACTOR_TYPE,
        choices=["classical", "pretrained", "dr_model"],
        help="Embedding extractor to use (default: %(default)s).",
    )
    parser.add_argument(
        "--threshold_percentile",
        type=float,
        default=config.AUTO_THRESHOLD_PERCENTILE,
        help="Percentile of in-distribution distances to suggest as threshold (default: %(default)s).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.data_dir.exists():
        log.error("Data directory '%s' not found.", args.data_dir)
        sys.exit(1)

    build_reference(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        extractor_type=args.extractor,
        threshold_percentile=args.threshold_percentile,
    )


if __name__ == "__main__":
    main()
