"""
Dataset Loading, Cleaning, Splitting & PyTorch Dataset — VINAYAK Module.

Phases covered:
  Phase 1 – Dataset inspection & metadata documentation
  Phase 2 – Data cleaning: path validation, corrupt detection, missing/invalid
             labels, duplicate detection (MD5), class distribution, patient
             grouping, reproducible splits, leakage checks
  Phase 3 – PyTorch Dataset and DataLoader construction

Supported datasets (primary): APTOS 2019
Future:                        EyePACS, Messidor-2, IDRiD
"""
import sys
import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.config import (
    CLASS_NAMES, NUM_CLASSES,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED,
    SPLITS_DIR, METADATA_DIR,
)
from src.preprocess import crop_retina_circle, apply_clahe, get_train_transforms, get_val_transforms
from src.utils import get_logger, md5_file

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Label Mapping
# ---------------------------------------------------------------------------

VALID_RAW_LABELS: set[int] = {0, 1, 2, 3, 4}  # APTOS raw labels


def map_raw_label_to_5class(raw_label: int) -> int:
    """
    Validates 5-class APTOS labels.
    """
    raw_label = int(raw_label)
    if raw_label not in VALID_RAW_LABELS:
        raise ValueError(f"Invalid raw DR label: {raw_label}. Expected one of {VALID_RAW_LABELS}.")
    return raw_label


# ---------------------------------------------------------------------------
# Helper: resolve image path with extension fallback
# ---------------------------------------------------------------------------

def _resolve_image_path(img_dir: Path, img_name: str) -> Optional[Path]:
    """
    Finds an image file by name, trying multiple extensions if needed.

    Returns the resolved Path if found, else None.
    """
    direct = img_dir / img_name
    if direct.exists():
        return direct
    for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
        candidate = img_dir / f"{img_name}{ext}"
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Phase 2 – Dataset Cleaning & Report
# ---------------------------------------------------------------------------

def inspect_and_clean_dataset(
    csv_path: Path,
    img_dir: Path,
    output_summary_path: Optional[Path] = None,
    run_duplicate_check: bool = True,
) -> pd.DataFrame:
    """
    Inspects all images, verifies readability, validates labels, detects
    duplicates via MD5, maps labels to 4-class, and writes a dataset report.

    Args:
        csv_path:            Path to the labels CSV (APTOS: train.csv).
        img_dir:             Directory containing fundus images.
        output_summary_path: Where to write the JSON dataset report.
                             Defaults to data/metadata/dataset_report.json.
        run_duplicate_check: If True, compute MD5 hashes to find duplicates.
                             Disable for large datasets where speed matters.

    Returns:
        Clean DataFrame with columns:
            image_id, raw_grade, mapped_grade, patient_id, file_path

    Notes:
        - APTOS 2019 CSV uses columns: id_code, diagnosis
        - patient_id is derived from image prefix (APTOS has no explicit patient ID)
        - Leakage check is performed when patient IDs are available
    """
    csv_path = Path(csv_path)
    img_dir = Path(img_dir)
    if output_summary_path is None:
        output_summary_path = METADATA_DIR / "dataset_report.json"

    log.info("Starting dataset inspection: %s", csv_path)
    df = pd.read_csv(csv_path)

    # --- Auto-detect column names ---
    id_col = _detect_col(df, ["id_code", "image_id", "image"])
    label_col = _detect_col(df, ["diagnosis", "level", "dr_grade", "label"])
    patient_col = _detect_col(df, ["patient_id"], required=False)

    log.info("CSV shape: %s | id_col=%s | label_col=%s", df.shape, id_col, label_col)

    valid_rows: list[dict] = []
    rejected_files: list[dict] = []
    seen_hashes: dict[str, str] = {}   # md5 → first image_id
    duplicates: list[dict] = []

    for idx, row in df.iterrows():
        img_name = str(row[id_col]).strip()
        raw_label = row[label_col]

        # 1. Missing label check
        if pd.isna(raw_label):
            rejected_files.append({"image_id": img_name, "reason": "Missing label (NaN)"})
            continue

        # 2. Invalid label check
        try:
            mapped = map_raw_label_to_5class(raw_label)
        except ValueError as e:
            rejected_files.append({"image_id": img_name, "reason": str(e)})
            continue

        # 3. File existence check
        img_path = _resolve_image_path(img_dir, img_name)
        if img_path is None:
            rejected_files.append({
                "image_id": img_name,
                "reason": f"File not found in {img_dir}",
            })
            continue

        # 4. Corrupt / unreadable image check
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                rejected_files.append({
                    "image_id": img_name,
                    "reason": "Corrupt or unreadable image (cv2.imread returned None)",
                })
                continue
            if img.shape[0] < 10 or img.shape[1] < 10:
                rejected_files.append({
                    "image_id": img_name,
                    "reason": f"Image too small: {img.shape}",
                })
                continue
        except Exception as e:
            rejected_files.append({"image_id": img_name, "reason": f"Read error: {e}"})
            continue

        # 5. Duplicate detection (MD5 hash)
        if run_duplicate_check:
            file_hash = md5_file(img_path)
            if file_hash in seen_hashes:
                duplicates.append({
                    "image_id": img_name,
                    "duplicate_of": seen_hashes[file_hash],
                })
                # Still include in dataset — flag for user review
            else:
                seen_hashes[file_hash] = img_name

        # 6. Patient ID: use explicit column or derive from image name prefix
        if patient_col and patient_col in row.index:
            patient_id = str(row[patient_col])
        else:
            # APTOS naming: e.g. "0a4e1a29ffff" — no patient ID embedded
            # Use image prefix as a best-effort grouping; flag this assumption
            patient_id = img_name.split("_")[0]

        valid_rows.append({
            "image_id":    img_path.name,
            "raw_grade":   int(raw_label),
            "mapped_grade": mapped,
            "patient_id":  patient_id,
            "file_path":   str(img_path),
        })

    clean_df = pd.DataFrame(valid_rows)

    # --- Class distribution ---
    class_counts: dict[str, int] = {}
    if len(clean_df):
        dist = clean_df["mapped_grade"].value_counts().sort_index()
        class_counts = {CLASS_NAMES.get(k, str(k)): int(v) for k, v in dist.items()}

    # --- Patient count ---
    unique_patients = int(clean_df["patient_id"].nunique()) if len(clean_df) else 0

    # --- Leakage warning ---
    leakage_warnings: list[str] = []
    if unique_patients == len(clean_df):
        leakage_warnings.append(
            "ASSUMPTION: Patient IDs appear 1-to-1 with images (no explicit patient column). "
            "Splits will be stratified by class. Verify if a patient ID mapping is available."
        )

    # --- Write dataset report ---
    report = {
        "dataset": "APTOS 2019 (prototype)",
        "source": "https://www.kaggle.com/c/aptos2019-blindness-detection",
        "label_definition": (
            "5-class APTOS labels (0–4) mapped to 4 classes: "
            "0=No DR, 1=Mild, 2=Moderate, 3=Severe/PDR (merged 3&4)"
        ),
        "annotation_type": "image-level grade by ophthalmologist panel",
        "license": "Kaggle competition data — research use, see terms",
        "patient_identifiers": "Not available in APTOS 2019 public dataset",
        "total_raw_entries": int(len(df)),
        "valid_images":      int(len(clean_df)),
        "rejected_count":    int(len(rejected_files)),
        "duplicate_count":   int(len(duplicates)),
        "unique_patients":   unique_patients,
        "class_distribution": class_counts,
        "rejected_files":    rejected_files[:50],  # cap for readability
        "duplicates":        duplicates[:50],
        "leakage_warnings":  leakage_warnings,
        "assumption_note": (
            "APTOS 2019 does not provide explicit patient IDs. "
            "Each image is treated as an independent sample for splits. "
            "Patient-level leakage prevention not possible without ID mapping."
        ),
    }
    output_summary_path = Path(output_summary_path)
    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_summary_path, "w") as f:
        json.dump(report, f, indent=2)

    log.info(
        "Inspection done. Valid=%d | Rejected=%d | Duplicates=%d | Classes=%s",
        len(clean_df), len(rejected_files), len(duplicates), class_counts,
    )
    print(f"\n{'='*55}")
    print(f"  Dataset Inspection Report")
    print(f"  Valid images   : {len(clean_df)}")
    print(f"  Rejected       : {len(rejected_files)}")
    print(f"  Duplicates     : {len(duplicates)}")
    print(f"  Class dist.    : {class_counts}")
    if leakage_warnings:
        for w in leakage_warnings:
            print(f"  [LEAKAGE WARN] : {w}")
    print(f"  Report saved   : {output_summary_path}")
    print(f"{'='*55}\n")

    return clean_df


def _detect_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> Optional[str]:
    """Returns the first matching column name from candidates, or None/raises if missing."""
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(
            f"Could not find any of {candidates} in CSV columns: {list(df.columns)}"
        )
    return None


# ---------------------------------------------------------------------------
# Phase 2 – Reproducible Train/Val/Test Splits
# ---------------------------------------------------------------------------

def create_splits(
    df: pd.DataFrame,
    splits_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Creates reproducible stratified or patient-level train/val/test splits.

    Strategy:
        1. If patient IDs appear truly non-unique (real patient groupings exist),
           split by patient to prevent data leakage.
        2. Otherwise, use stratified split by mapped_grade.

    The splits are saved as train.csv / val.csv / test.csv in splits_dir.
    Splits are verified for class representation and logged.

    Args:
        df:         Clean DataFrame from inspect_and_clean_dataset().
        splits_dir: Directory to save split CSVs. Defaults to data/splits/.

    Returns:
        (train_df, val_df, test_df)
    """
    if splits_dir is None:
        splits_dir = SPLITS_DIR
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    unique_patients = df["patient_id"].nunique()
    n_total = len(df)
    log.info("Total samples: %d | Unique patients: %d", n_total, unique_patients)

    # Decide split strategy
    if unique_patients < n_total and unique_patients >= 10:
        # Patient-level split (prevents leakage when real patient IDs exist)
        log.info("Using PATIENT-LEVEL split to prevent data leakage.")
        patient_ids = df["patient_id"].unique()
        train_pts, temp_pts = train_test_split(
            patient_ids,
            test_size=(VAL_RATIO + TEST_RATIO),
            random_state=RANDOM_SEED,
        )
        rel_test = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
        val_pts, test_pts = train_test_split(
            temp_pts, test_size=rel_test, random_state=RANDOM_SEED
        )
        train_df = df[df["patient_id"].isin(train_pts)].copy()
        val_df   = df[df["patient_id"].isin(val_pts)].copy()
        test_df  = df[df["patient_id"].isin(test_pts)].copy()
        strategy = "patient_level"
    else:
        # Stratified split by class label
        log.info("Using STRATIFIED split (no patient grouping available).")
        train_df, temp_df = train_test_split(
            df,
            test_size=(VAL_RATIO + TEST_RATIO),
            stratify=df["mapped_grade"],
            random_state=RANDOM_SEED,
        )
        rel_test = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=rel_test,
            stratify=temp_df["mapped_grade"],
            random_state=RANDOM_SEED,
        )
        strategy = "stratified"

    # --- Leakage verification ---
    _verify_no_leakage(train_df, val_df, test_df)

    # --- Save splits ---
    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "val.csv",   index=False)
    test_df.to_csv(splits_dir / "test.csv",  index=False)

    # --- Split report ---
    split_report = {
        "strategy": strategy,
        "random_seed": RANDOM_SEED,
        "train_ratio": TRAIN_RATIO,
        "val_ratio":   VAL_RATIO,
        "test_ratio":  TEST_RATIO,
        "splits": {
            "train": {"count": len(train_df), "class_dist": _class_dist(train_df)},
            "val":   {"count": len(val_df),   "class_dist": _class_dist(val_df)},
            "test":  {"count": len(test_df),  "class_dist": _class_dist(test_df)},
        },
    }
    with open(splits_dir / "split_report.json", "w") as f:
        json.dump(split_report, f, indent=2)

    log.info("Splits: Train=%d | Val=%d | Test=%d", len(train_df), len(val_df), len(test_df))
    print(f"\n{'='*55}")
    print(f"  Split Strategy : {strategy} (seed={RANDOM_SEED})")
    print(f"  Train          : {len(train_df)} images | {_class_dist(train_df)}")
    print(f"  Validation     : {len(val_df)} images | {_class_dist(val_df)}")
    print(f"  Test           : {len(test_df)} images | {_class_dist(test_df)}")
    print(f"  Saved to       : {splits_dir}")
    print(f"{'='*55}\n")

    return train_df, val_df, test_df


def _class_dist(df: pd.DataFrame) -> dict[str, int]:
    dist = df["mapped_grade"].value_counts().sort_index()
    return {CLASS_NAMES.get(int(k), str(k)): int(v) for k, v in dist.items()}


def _verify_no_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """
    Checks for image-level leakage across splits (same image_id in multiple splits).
    Logs a WARNING if any overlap is detected.
    """
    train_ids = set(train_df["image_id"])
    val_ids   = set(val_df["image_id"])
    test_ids  = set(test_df["image_id"])

    tv = train_ids & val_ids
    tt = train_ids & test_ids
    vt = val_ids   & test_ids

    if tv or tt or vt:
        log.warning("DATA LEAKAGE DETECTED!")
        if tv:
            log.warning("  Train ∩ Val:  %d images overlap: %s", len(tv), list(tv)[:5])
        if tt:
            log.warning("  Train ∩ Test: %d images overlap: %s", len(tt), list(tt)[:5])
        if vt:
            log.warning("  Val ∩ Test:   %d images overlap: %s", len(vt), list(vt)[:5])
    else:
        log.info("Leakage check PASSED — no overlapping image_ids across splits.")


# ---------------------------------------------------------------------------
# Phase 3 – PyTorch Dataset
# ---------------------------------------------------------------------------

class FundusDataset(Dataset):
    """
    PyTorch Dataset for Retinal Fundus Images.

    Loads images on demand, applies the appropriate transform pipeline,
    and returns (image_tensor, label_tensor) pairs.

    Args:
        df:          DataFrame with columns: image_id, mapped_grade (+ optional file_path)
        img_dir:     Directory containing images (used when file_path not in df)
        transform:   torchvision transform to apply. Defaults to val_transforms.
        is_training: Informational flag (transform should already be set appropriately)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: Path,
        transform=None,
        is_training: bool = False,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform if transform is not None else get_val_transforms()
        self.is_training = is_training

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img_name = str(row["image_id"])

        # Prefer pre-resolved file_path if available
        if "file_path" in row.index and pd.notna(row["file_path"]):
            img_path = Path(str(row["file_path"]))
        else:
            img_path = _resolve_image_path(self.img_dir, img_name)

        if img_path is None or not img_path.exists():
            raise FileNotFoundError(
                f"Image not found: '{img_name}' in directory '{self.img_dir}'"
            )

        # Fast path for pre-processed JPEGs (retina cropped + CLAHE + 384x384 already done)
        if img_path.suffix.lower() in (".jpg", ".jpeg"):
            pil_img = Image.open(img_path).convert("RGB")
            image_tensor = self.transform(pil_img)
            label = int(row["mapped_grade"])
            return image_tensor, torch.tensor(label, dtype=torch.long)

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise RuntimeError(f"cv2 could not decode image: {img_path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        cropped = crop_retina_circle(img_rgb)
        clahe_img = apply_clahe(cropped)
        pil_img = Image.fromarray(clahe_img)

        image_tensor = self.transform(pil_img)
        label = int(row["mapped_grade"])
        return image_tensor, torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloaders(
    splits_dir: Optional[Path] = None,
    img_dir: Optional[Path] = None,
    batch_size: int = 16,
    image_size: tuple[int, int] = (224, 224),
    use_weighted_sampler: bool = True,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Builds and returns (train_loader, val_loader, test_loader).

    Args:
        splits_dir:           Directory containing train/val/test.csv.
        img_dir:              Directory containing fundus images.
        batch_size:           Batch size for all loaders.
        image_size:           Image resize target (width, height).
        use_weighted_sampler: If True, uses WeightedRandomSampler to handle
                              class imbalance in the training loader.
        num_workers:          DataLoader worker processes. Use 0 on Windows.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    from configs.config import SPLITS_DIR as DEFAULT_SPLITS
    from configs.config import RAW_DATA_DIR

    if splits_dir is None:
        splits_dir = DEFAULT_SPLITS
    splits_dir = Path(splits_dir)

    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df   = pd.read_csv(splits_dir / "val.csv")
    test_df  = pd.read_csv(splits_dir / "test.csv")

    # Resolve img_dir from file_path column if no explicit dir given
    if img_dir is None:
        if "file_path" in train_df.columns and pd.notna(train_df["file_path"].iloc[0]):
            img_dir = Path(train_df["file_path"].iloc[0]).parent
        else:
            img_dir = RAW_DATA_DIR / "train_images"

    train_dataset = FundusDataset(
        train_df, img_dir,
        transform=get_train_transforms(image_size),
        is_training=True,
    )
    val_dataset = FundusDataset(
        val_df, img_dir,
        transform=get_val_transforms(image_size),
    )
    test_dataset = FundusDataset(
        test_df, img_dir,
        transform=get_val_transforms(image_size),
    )

    # Weighted sampler for class imbalance
    sampler = None
    shuffle = True
    if use_weighted_sampler:
        class_counts = train_df["mapped_grade"].value_counts().sort_index().values
        class_weights = 1.0 / (class_counts.astype(float) + 1e-9)
        sample_weights = np.array([class_weights[int(label)] for label in train_df["mapped_grade"]])
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).float(),
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False  # mutually exclusive with sampler

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=shuffle, sampler=sampler,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True,
    )

    log.info(
        "DataLoaders ready | Train=%d | Val=%d | Test=%d | batch=%d | workers=%d",
        len(train_dataset), len(val_dataset), len(test_dataset), batch_size, num_workers,
    )
    return train_loader, val_loader, test_loader
