#!/usr/bin/env python3
"""
Combined APTOS + IDRiD Training Pipeline -- VISIONARY6 VINAYAK
================================================================

Trains EfficientNet-B0 on the merged APTOS-2019 + IDRiD dataset for
maximum generalization across Indian fundus image sources.

Strategy:
    1. Merge APTOS splits + IDRiD data into unified train/val/test CSVs.
    2. Oversample IDRiD images (smaller dataset) to balance dataset sources.
    3. Train from ImageNet pretrained weights (fresh backbone).
    4. Use class-weighted loss + weighted sampler (handles class imbalance).
    5. Save the combined checkpoint separately as best_model_combined.pth.
    6. Post-training evaluation on BOTH APTOS-test and IDRiD-test sets.

Usage:
    python src/train_combined.py
    python src/train_combined.py --epochs 25 --lr 1e-4 --batch_size 16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import (
    BASE_DIR,
    CLASS_NAMES,
    NUM_CLASSES,
    MODEL_NAME,
    DROPOUT_RATE,
    CHECKPOINTS_DIR,
    SPLITS_DIR,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    RANDOM_SEED,
    DEVICE,
    USE_AMP,
    EXPERIMENTS_DIR,
    IMAGE_SIZE,
    get_logger,
)
from src.model import build_model
from src.preprocess import get_train_transforms, get_val_transforms
from src.dataset import FundusDataset
from src.utils import (
    set_seed,
    print_device_info,
    generate_experiment_id,
    create_experiment_dir,
    save_json,
)
from src.train import run_train_epoch, run_val_epoch, save_training_curves

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Use pre-processed splits (224x224 images, retina already cropped) for 10x speed
APTOS_SPLITS_DIR    = BASE_DIR / "data" / "processed_splits"           # pre-processed!
IDRID_CSV           = BASE_DIR / "data" / "idrid" / "idrid_processed.csv"  # pre-processed!
COMBINED_SPLITS_DIR = BASE_DIR / "data" / "combined" / "splits"
COMBINED_CHECKPOINT = CHECKPOINTS_DIR / "best_model_combined.pth"
COMBINED_LAST_CKPT  = CHECKPOINTS_DIR / "last_model_combined.pth"

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
COMBINED_EPOCHS    = 25
COMBINED_BATCH     = 16
COMBINED_LR        = 1e-4
COMBINED_WD        = 1e-4
COMBINED_PATIENCE  = 7

# ---------------------------------------------------------------------------
# Fast Dataset (pre-processed images — PIL direct load, no cv2 crop)
# ---------------------------------------------------------------------------

class FastFundusDataset(Dataset):
    """
    Ultra-fast dataset for pre-processed 224x224 JPEG images.

    Images are already: retina-cropped + resized to 224x224.
    So we just load them with PIL and apply torchvision transforms.
    No heavy OpenCV processing — ~10x faster than FundusDataset.
    """

    def __init__(self, df: pd.DataFrame, transform=None) -> None:
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = Path(str(row["file_path"]))

        if not img_path.exists():
            raise FileNotFoundError(f"Pre-processed image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = int(row["mapped_grade"])
        return img, torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------------------------
# Dataset merging
# ---------------------------------------------------------------------------


def _fix_idrid_paths(df: pd.DataFrame) -> pd.DataFrame:
    """Convert relative backslash IDRiD paths to absolute paths."""
    def _to_abs(rel_path: str) -> str:
        p = Path(str(rel_path).replace("\\", "/"))
        if not p.is_absolute():
            p = BASE_DIR / p
        return str(p)
    df = df.copy()
    df["file_path"] = df["file_path"].apply(_to_abs)
    return df


def _create_combined_splits(
    aptos_splits_dir: Path,
    idrid_csv: Path,
    output_dir: Path,
    idrid_oversample: int = 3,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Merge APTOS train/val/test splits with IDRiD data.

    IDRiD is ~8x smaller than APTOS, so we oversample IDRiD rows in the
    training set to balance dataset representation (val/test are NOT
    oversampled to keep evaluation fair).

    Args:
        aptos_splits_dir:  Directory containing APTOS train.csv, val.csv, test.csv.
        idrid_csv:         Path to idrid_standardized.csv.
        output_dir:        Directory to write merged split CSVs.
        idrid_oversample:  Repeat IDRiD training rows N times (default 3x).
        seed:              Random seed for IDRiD stratified split.

    Returns:
        (train_df, val_df, test_df) -- merged DataFrames.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load APTOS splits ---
    aptos_train = pd.read_csv(aptos_splits_dir / "train.csv")
    aptos_val   = pd.read_csv(aptos_splits_dir / "val.csv")
    aptos_test  = pd.read_csv(aptos_splits_dir / "test.csv")

    print(f"  APTOS  : Train={len(aptos_train)} | Val={len(aptos_val)} | Test={len(aptos_test)}")

    # --- Load & split IDRiD ---
    idrid_df = pd.read_csv(idrid_csv)
    idrid_df = _fix_idrid_paths(idrid_df)
    if "patient_id" not in idrid_df.columns:
        idrid_df["patient_id"] = idrid_df["image_id"]

    idrid_train, idrid_temp = train_test_split(
        idrid_df, test_size=0.30,
        stratify=idrid_df["mapped_grade"], random_state=seed,
    )
    idrid_val, idrid_test = train_test_split(
        idrid_temp, test_size=0.50,
        stratify=idrid_temp["mapped_grade"], random_state=seed,
    )

    print(f"  IDRiD  : Train={len(idrid_train)} | Val={len(idrid_val)} | Test={len(idrid_test)}")

    # --- Add source column for tracking ---
    aptos_train["source"] = "APTOS"
    aptos_val["source"]   = "APTOS"
    aptos_test["source"]  = "APTOS"
    idrid_train["source"] = "IDRiD"
    idrid_val["source"]   = "IDRiD"
    idrid_test["source"]  = "IDRiD"

    # --- Oversample IDRiD training data ---
    if idrid_oversample > 1:
        idrid_train_repeated = pd.concat(
            [idrid_train] * idrid_oversample, ignore_index=True
        )
        print(f"  IDRiD train oversampled: {len(idrid_train)} -> {len(idrid_train_repeated)} ({idrid_oversample}x)")
    else:
        idrid_train_repeated = idrid_train

    # --- Merge ---
    # Ensure both DataFrames have the same columns
    common_cols = ["image_id", "raw_grade", "mapped_grade", "patient_id", "file_path", "source"]

    for df in [aptos_train, aptos_val, aptos_test]:
        for col in common_cols:
            if col not in df.columns:
                df[col] = ""

    for df in [idrid_train_repeated, idrid_val, idrid_test]:
        for col in common_cols:
            if col not in df.columns:
                df[col] = ""

    train_df = pd.concat(
        [aptos_train[common_cols], idrid_train_repeated[common_cols]],
        ignore_index=True,
    ).sample(frac=1, random_state=seed).reset_index(drop=True)   # shuffle

    val_df = pd.concat(
        [aptos_val[common_cols], idrid_val[common_cols]],
        ignore_index=True,
    )

    test_df = pd.concat(
        [aptos_test[common_cols], idrid_test[common_cols]],
        ignore_index=True,
    )

    print(f"\n  Combined Train : {len(train_df)} images  (APTOS={len(aptos_train)}, IDRiD={len(idrid_train_repeated)})")
    print(f"  Combined Val   : {len(val_df)} images")
    print(f"  Combined Test  : {len(test_df)} images")

    # --- Save ---
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    # --- Report ---
    split_report = {
        "dataset": "APTOS_2019 + IDRiD (combined)",
        "strategy": "APTOS pre-split + IDRiD stratified + IDRiD oversample",
        "idrid_oversample_factor": idrid_oversample,
        "random_seed": seed,
        "splits": {
            "train": {
                "total": len(train_df),
                "aptos": len(aptos_train),
                "idrid_original": len(idrid_train),
                "idrid_oversampled": len(idrid_train_repeated),
                "class_dist": train_df["mapped_grade"].value_counts().sort_index().to_dict(),
            },
            "val": {
                "total": len(val_df),
                "class_dist": val_df["mapped_grade"].value_counts().sort_index().to_dict(),
            },
            "test": {
                "total": len(test_df),
                "class_dist": test_df["mapped_grade"].value_counts().sort_index().to_dict(),
            },
        },
    }
    save_json(split_report, output_dir / "split_report.json")

    return train_df, val_df, test_df


def _compute_class_weights(train_df: pd.DataFrame, device: str) -> torch.Tensor:
    """Inverse-frequency class weights from training split."""
    counts = train_df["mapped_grade"].value_counts().sort_index()
    total  = len(train_df)
    weights = []
    for c in range(NUM_CLASSES):
        cnt = counts.get(c, 0)
        w = total / (NUM_CLASSES * cnt) if cnt > 0 else 1.0
        weights.append(w)
    wt = torch.tensor(weights, dtype=torch.float32, device=device)
    log.info("Combined class weights: %s", [round(w, 4) for w in wt.tolist()])
    return wt


def _build_weighted_sampler(train_df: pd.DataFrame) -> WeightedRandomSampler:
    """WeightedRandomSampler for the combined imbalanced dataset."""
    labels = train_df["mapped_grade"].values
    class_counts = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
    class_weights = 1.0 / (class_counts + 1e-9)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(sample_weights),
        replacement=True,
    )


def _evaluate_subset(model, test_df, source_name, device):
    """Evaluate model on a subset of the test set filtered by source."""
    subset = test_df[test_df["source"] == source_name].copy()
    if len(subset) == 0:
        print(f"  No {source_name} images in test set -- skipping.")
        return

    dummy_dir = BASE_DIR / "data"
    ds = FastFundusDataset(subset, transform=get_val_transforms(IMAGE_SIZE))
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds = model(images).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    class_names_list = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]

    print(f"\n  --- {source_name} Test Subset ({len(subset)} images) ---")
    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"  Macro F1 : {f1*100:.2f}%")
    print(classification_report(all_labels, all_preds, target_names=class_names_list, zero_division=0))
    return {"accuracy": round(acc, 6), "macro_f1": round(f1, 6), "count": len(subset)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train_combined(
    num_epochs: int = COMBINED_EPOCHS,
    batch_size: int = COMBINED_BATCH,
    learning_rate: float = COMBINED_LR,
    weight_decay: float = COMBINED_WD,
    patience: int = COMBINED_PATIENCE,
    idrid_oversample: int = 3,
    random_seed: int = RANDOM_SEED,
) -> dict:
    """
    Train on merged APTOS + IDRiD dataset.
    """
    set_seed(random_seed)
    print_device_info(DEVICE)
    use_amp = USE_AMP

    # -----------------------------------------------------------------------
    # 1. Create combined splits
    # -----------------------------------------------------------------------
    print("\n[Step 1/5] Creating combined APTOS + IDRiD splits...")
    train_df, val_df, test_df = _create_combined_splits(
        APTOS_SPLITS_DIR, IDRID_CSV, COMBINED_SPLITS_DIR,
        idrid_oversample=idrid_oversample, seed=random_seed,
    )

    # -----------------------------------------------------------------------
    # 2. DataLoaders
    # -----------------------------------------------------------------------
    print("\n[Step 2/5] Building DataLoaders (using pre-processed 224x224 images)...")
    train_transforms = get_train_transforms(IMAGE_SIZE)
    val_transforms   = get_val_transforms(IMAGE_SIZE)

    train_dataset = FastFundusDataset(train_df, transform=train_transforms)
    val_dataset   = FastFundusDataset(val_df,   transform=val_transforms)

    sampler = _build_weighted_sampler(train_df)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=sampler, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=True,
    )

    print(f"  Train samples : {len(train_dataset)}")
    print(f"  Val samples   : {len(val_dataset)}")
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val batches   : {len(val_loader)}")

    # -----------------------------------------------------------------------
    # 3. Model (fresh from ImageNet)
    # -----------------------------------------------------------------------
    print("\n[Step 3/5] Building model (ImageNet pretrained)...")
    model = build_model(
        architecture=MODEL_NAME,
        num_classes=NUM_CLASSES,
        pretrained=True,
        dropout_rate=DROPOUT_RATE,
        device=DEVICE,
        verbose=True,
    )

    # -----------------------------------------------------------------------
    # Loss, optimizer, scheduler
    # -----------------------------------------------------------------------
    class_weights = _compute_class_weights(train_df, DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6,
    )
    scaler = GradScaler(enabled=use_amp)

    # Experiment dir
    experiment_id = f"combined_{generate_experiment_id(MODEL_NAME, random_seed)}"
    exp_dir = create_experiment_dir(EXPERIMENTS_DIR, experiment_id)
    COMBINED_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 4. Training loop
    # -----------------------------------------------------------------------
    print(f"\n[Step 4/5] Training for up to {num_epochs} epochs...")
    print(f"{'='*65}")

    best_val_loss = float("inf")
    best_val_acc  = 0.0
    no_improve    = 0

    history: dict[str, list[float]] = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
        "lr": [],
    }

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_loss, train_acc = run_train_epoch(
            model, train_loader, criterion, optimizer,
            DEVICE, scaler, use_amp, epoch, num_epochs,
        )
        val_loss, val_acc = run_val_epoch(
            model, val_loader, criterion, DEVICE, epoch, num_epochs,
        )
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        history["train_loss"].append(round(train_loss, 6))
        history["train_acc"].append(round(train_acc, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["val_acc"].append(round(val_acc, 6))
        history["lr"].append(lr_now)

        print(
            f"Epoch {epoch:02d}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc*100:.2f}% | "
            f"LR: {lr_now:.2e} | {elapsed:.1f}s"
        )

        # Save last
        torch.save(model.state_dict(), COMBINED_LAST_CKPT)

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc  = val_acc
            no_improve    = 0
            torch.save(model.state_dict(), COMBINED_CHECKPOINT)
            log.info(
                "[*] Combined best ckpt -- epoch %d (Val Loss: %.4f | Acc: %.2f%%)",
                epoch, best_val_loss, best_val_acc * 100,
            )
        else:
            no_improve += 1

        if no_improve >= patience:
            log.info("Early stopping at epoch %d.", epoch)
            print(f"\n[Early Stopping] Triggered at epoch {epoch}.")
            break

    print(f"{'='*65}")
    print(f"  Training Complete")
    print(f"  Best Val Loss  : {best_val_loss:.4f}")
    print(f"  Best Val Acc   : {best_val_acc*100:.2f}%")
    print(f"  Checkpoint     : {COMBINED_CHECKPOINT}")

    # Save curves
    save_training_curves(history, exp_dir)
    combined_metrics_dir = BASE_DIR / "outputs" / "metrics" / "combined"
    combined_metrics_dir.mkdir(parents=True, exist_ok=True)
    save_training_curves(history, combined_metrics_dir)

    # -----------------------------------------------------------------------
    # 5. Post-training evaluation
    # -----------------------------------------------------------------------
    print(f"\n[Step 5/5] Evaluating on test sets...")

    model.load_state_dict(
        torch.load(COMBINED_CHECKPOINT, map_location=DEVICE, weights_only=True)
    )
    model.eval()

    # Evaluate on each dataset separately
    aptos_metrics = _evaluate_subset(model, test_df, "APTOS", DEVICE)
    idrid_metrics = _evaluate_subset(model, test_df, "IDRiD", DEVICE)

    # Overall test evaluation
    test_dataset = FastFundusDataset(test_df, transform=get_val_transforms(IMAGE_SIZE))
    test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            preds = model(images).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    overall_acc = accuracy_score(all_labels, all_preds)
    overall_f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    class_names_list = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]
    print(f"\n  --- Overall Combined Test ({len(test_df)} images) ---")
    print(f"  Accuracy : {overall_acc*100:.2f}%")
    print(f"  Macro F1 : {overall_f1*100:.2f}%")
    print(classification_report(all_labels, all_preds, target_names=class_names_list, zero_division=0))

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

    # -----------------------------------------------------------------------
    # Save summary
    # -----------------------------------------------------------------------
    train_summary = {
        "experiment_id":         experiment_id,
        "architecture":          MODEL_NAME,
        "dataset":               "APTOS 2019 + IDRiD (combined)",
        "idrid_oversample":      idrid_oversample,
        "num_classes":           NUM_CLASSES,
        "epochs_run":            len(history["train_loss"]),
        "max_epochs":            num_epochs,
        "batch_size":            batch_size,
        "learning_rate":         learning_rate,
        "weight_decay":          weight_decay,
        "optimizer":             "AdamW",
        "scheduler":             "CosineAnnealingLR",
        "loss":                  "CrossEntropyLoss (weighted)",
        "early_stopping_patience": patience,
        "use_amp":               use_amp,
        "device":                str(DEVICE),
        "random_seed":           random_seed,
        "best_val_loss":         round(best_val_loss, 6),
        "best_val_acc":          round(best_val_acc, 6),
        "test_overall": {
            "accuracy": round(overall_acc, 6),
            "macro_f1": round(overall_f1, 6),
        },
        "test_aptos":  aptos_metrics,
        "test_idrid":  idrid_metrics,
        "confusion_matrix":      cm.tolist(),
        "checkpoint_path":       str(COMBINED_CHECKPOINT),
        "training_history":      history,
        "data_split": {
            "train": len(train_df),
            "val":   len(val_df),
            "test":  len(test_df),
        },
    }
    save_json(train_summary, exp_dir / "training_summary.json")
    save_json(train_summary, combined_metrics_dir / "combined_training_summary.json")

    print(f"\n  Experiment dir : {exp_dir}")
    print(f"  Checkpoint     : {COMBINED_CHECKPOINT}")
    print(f"{'='*65}\n")

    return train_summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Train VINAYAK DR model on combined APTOS + IDRiD dataset."
    )
    parser.add_argument("--epochs",      type=int,   default=COMBINED_EPOCHS)
    parser.add_argument("--batch_size",  type=int,   default=COMBINED_BATCH)
    parser.add_argument("--lr",          type=float, default=COMBINED_LR)
    parser.add_argument("--patience",    type=int,   default=COMBINED_PATIENCE)
    parser.add_argument("--oversample",  type=int,   default=3,
                        help="IDRiD oversample factor (default: 3)")
    parser.add_argument("--seed",        type=int,   default=RANDOM_SEED)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_combined(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        idrid_oversample=args.oversample,
        random_seed=args.seed,
    )
