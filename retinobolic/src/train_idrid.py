#!/usr/bin/env python3
"""
IDRiD Fine-Tuning Pipeline — VISIONARY6 VINAYAK
=================================================

Fine-tunes the APTOS-2019-pretrained EfficientNet-B0 on the IDRiD dataset
(Indian Diabetic Retinopathy Image Dataset) for improved cross-dataset
generalization.

Strategy:
    1. Load the APTOS-pretrained checkpoint as starting weights.
    2. Create stratified train/val/test splits from the IDRiD standardized CSV.
    3. Apply aggressive augmentation (small dataset → more augmentation).
    4. Use differential learning rates: low for backbone, higher for classifier.
    5. Save new checkpoints separately (does NOT overwrite APTOS checkpoint).
    6. Run post-training evaluation on the IDRiD test split.

Usage:
    python src/train_idrid.py
    python src/train_idrid.py --epochs 30 --lr 3e-4 --batch_size 8
    python src/train_idrid.py --from_scratch   # skip APTOS pretrained weights
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
from torch.utils.data import DataLoader, WeightedRandomSampler
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
    CHECKPOINT_PATH,        # APTOS checkpoint (source for transfer learning)
    CHECKPOINTS_DIR,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    RANDOM_SEED,
    DEVICE,
    USE_AMP,
    EXPERIMENTS_DIR,
    METRICS_DIR,
    IMAGE_SIZE,
    get_logger,
)
from src.model import build_model
from src.preprocess import get_train_transforms, get_val_transforms, crop_retina_circle
from src.dataset import FundusDataset
from src.utils import (
    set_seed,
    print_device_info,
    generate_experiment_id,
    create_experiment_dir,
    save_json,
    print_model_summary,
)
from src.train import run_train_epoch, run_val_epoch, save_training_curves

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# IDRiD-specific paths
# ---------------------------------------------------------------------------
IDRID_CSV        = BASE_DIR / "data" / "idrid" / "idrid_standardized.csv"
IDRID_SPLITS_DIR = BASE_DIR / "data" / "idrid" / "splits"
IDRID_CHECKPOINT = CHECKPOINTS_DIR / "best_model_idrid.pth"
IDRID_LAST_CKPT  = CHECKPOINTS_DIR / "last_model_idrid.pth"

# ---------------------------------------------------------------------------
# IDRiD training hyper-parameters (tuned for small dataset fine-tuning)
# ---------------------------------------------------------------------------
IDRID_EPOCHS          = 30          # more epochs since dataset is small
IDRID_BATCH_SIZE      = 8           # small batches — only 456 images
IDRID_LR_BACKBONE     = 3e-5        # low LR to preserve APTOS features
IDRID_LR_CLASSIFIER   = 3e-4        # higher LR for the new classifier head
IDRID_WEIGHT_DECAY    = 1e-4
IDRID_PATIENCE        = 8           # more patience — small val set is noisy
IDRID_TRAIN_RATIO     = 0.70
IDRID_VAL_RATIO       = 0.15
IDRID_TEST_RATIO      = 0.15


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fix_file_paths(df: pd.DataFrame) -> pd.DataFrame:
    """Convert relative backslash paths in the CSV to absolute paths."""
    def _to_abs(rel_path: str) -> str:
        p = Path(rel_path.replace("\\", "/"))
        if not p.is_absolute():
            p = BASE_DIR / p
        return str(p)

    df = df.copy()
    df["file_path"] = df["file_path"].apply(_to_abs)
    return df


def _create_idrid_splits(
    csv_path: Path,
    splits_dir: Path,
    train_ratio: float = IDRID_TRAIN_RATIO,
    val_ratio: float = IDRID_VAL_RATIO,
    test_ratio: float = IDRID_TEST_RATIO,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/val/test splits from IDRiD standardized CSV.
    Saves train.csv, val.csv, test.csv into splits_dir.
    """
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = _fix_file_paths(df)

    # Add a dummy patient_id (each image = unique patient for IDRiD)
    if "patient_id" not in df.columns:
        df["patient_id"] = df["image_id"]

    log.info("IDRiD dataset: %d images", len(df))
    log.info("Class distribution:\n%s", df["mapped_grade"].value_counts().sort_index())

    # Stratified split by mapped_grade
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        stratify=df["mapped_grade"],
        random_state=seed,
    )
    rel_test = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=rel_test,
        stratify=temp_df["mapped_grade"],
        random_state=seed,
    )

    # Verify no overlap
    train_ids = set(train_df["image_id"])
    val_ids   = set(val_df["image_id"])
    test_ids  = set(test_df["image_id"])
    assert train_ids.isdisjoint(val_ids),  "Train/Val overlap detected!"
    assert train_ids.isdisjoint(test_ids), "Train/Test overlap detected!"
    assert val_ids.isdisjoint(test_ids),   "Val/Test overlap detected!"

    # Save
    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "val.csv",     index=False)
    test_df.to_csv(splits_dir / "test.csv",   index=False)

    # Report
    split_report = {
        "dataset": "IDRiD",
        "strategy": "stratified",
        "random_seed": seed,
        "train_ratio": train_ratio,
        "val_ratio":   val_ratio,
        "test_ratio":  test_ratio,
        "splits": {
            "train": {"count": len(train_df), "class_dist": train_df["mapped_grade"].value_counts().sort_index().to_dict()},
            "val":   {"count": len(val_df),   "class_dist": val_df["mapped_grade"].value_counts().sort_index().to_dict()},
            "test":  {"count": len(test_df),  "class_dist": test_df["mapped_grade"].value_counts().sort_index().to_dict()},
        },
    }
    save_json(split_report, splits_dir / "split_report.json")

    print(f"\n{'='*60}")
    print(f"  IDRiD Split Strategy : stratified (seed={seed})")
    print(f"  Train                : {len(train_df):>4d} images")
    print(f"  Validation           : {len(val_df):>4d} images")
    print(f"  Test                 : {len(test_df):>4d} images")
    print(f"  Saved to             : {splits_dir}")
    print(f"{'='*60}\n")

    return train_df, val_df, test_df


def _compute_class_weights(train_df: pd.DataFrame, device: str) -> torch.Tensor:
    """Compute inverse-frequency class weights from the training split."""
    counts = train_df["mapped_grade"].value_counts().sort_index()
    total  = len(train_df)
    weights = []
    for c in range(NUM_CLASSES):
        cnt = counts.get(c, 0)
        w = total / (NUM_CLASSES * cnt) if cnt > 0 else 1.0
        weights.append(w)
    wt = torch.tensor(weights, dtype=torch.float32, device=device)
    log.info("IDRiD class weights: %s", wt.tolist())
    return wt


def _build_weighted_sampler(train_df: pd.DataFrame) -> WeightedRandomSampler:
    """WeightedRandomSampler for imbalanced IDRiD classes (Mild DR only 22 imgs)."""
    labels = train_df["mapped_grade"].values
    class_counts = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
    class_weights = 1.0 / (class_counts + 1e-9)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(sample_weights),
        replacement=True,
    )


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_idrid(
    num_epochs: int = IDRID_EPOCHS,
    batch_size: int = IDRID_BATCH_SIZE,
    lr_backbone: float = IDRID_LR_BACKBONE,
    lr_classifier: float = IDRID_LR_CLASSIFIER,
    weight_decay: float = IDRID_WEIGHT_DECAY,
    patience: int = IDRID_PATIENCE,
    from_scratch: bool = False,
    random_seed: int = RANDOM_SEED,
) -> dict:
    """
    Fine-tune EfficientNet-B0 on IDRiD with transfer learning from APTOS.

    Args:
        num_epochs:     Maximum epochs.
        batch_size:     Batch size (8 recommended for 456 images).
        lr_backbone:    Learning rate for backbone (feature extractor) layers.
        lr_classifier:  Learning rate for classifier head.
        weight_decay:   L2 regularization.
        patience:       Early stopping patience.
        from_scratch:   If True, skip loading APTOS checkpoint (ImageNet only).
        random_seed:    Random seed.

    Returns:
        Training summary dict.
    """
    set_seed(random_seed)
    print_device_info(DEVICE)
    use_amp = USE_AMP

    # -----------------------------------------------------------------------
    # 1. Create IDRiD splits
    # -----------------------------------------------------------------------
    print("\n[Step 1/5] Creating IDRiD data splits...")
    train_df, val_df, test_df = _create_idrid_splits(IDRID_CSV, IDRID_SPLITS_DIR, seed=random_seed)

    # -----------------------------------------------------------------------
    # 2. Build DataLoaders
    # -----------------------------------------------------------------------
    print("[Step 2/5] Building DataLoaders...")

    train_transforms = get_train_transforms(IMAGE_SIZE)
    val_transforms   = get_val_transforms(IMAGE_SIZE)

    # img_dir is a dummy — file_path column has absolute paths
    dummy_img_dir = BASE_DIR / "data" / "raw_idrid"

    train_dataset = FundusDataset(train_df, dummy_img_dir, transform=train_transforms, is_training=True)
    val_dataset   = FundusDataset(val_df,   dummy_img_dir, transform=val_transforms,   is_training=False)
    test_dataset  = FundusDataset(test_df,  dummy_img_dir, transform=val_transforms,   is_training=False)

    sampler = _build_weighted_sampler(train_df)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=sampler, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=True,
    )

    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val batches   : {len(val_loader)}")
    print(f"  Test batches  : {len(test_loader)}")

    # -----------------------------------------------------------------------
    # 3. Build model + load APTOS weights (transfer learning)
    # -----------------------------------------------------------------------
    print("[Step 3/5] Building model...")
    model = build_model(
        architecture=MODEL_NAME,
        num_classes=NUM_CLASSES,
        pretrained=True,          # ImageNet weights as base
        dropout_rate=DROPOUT_RATE,
        device=DEVICE,
        verbose=True,
    )

    if not from_scratch and CHECKPOINT_PATH.exists():
        print(f"  Loading APTOS checkpoint: {CHECKPOINT_PATH}")
        state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        log.info("Successfully loaded APTOS-pretrained weights for fine-tuning.")
        print("  [OK] APTOS weights loaded - fine-tuning mode")
    elif from_scratch:
        print("  [!] Training from scratch (ImageNet weights only, no APTOS)")
    else:
        print("  [!] APTOS checkpoint not found - training from ImageNet weights only")

    # -----------------------------------------------------------------------
    # Differential learning rates
    # -----------------------------------------------------------------------
    # Separate backbone parameters from classifier head
    backbone_params = []
    classifier_params = []

    for name, param in model.named_parameters():
        if "classifier" in name or "fc" in name:
            classifier_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = [
        {"params": backbone_params,    "lr": lr_backbone},
        {"params": classifier_params,  "lr": lr_classifier},
    ]

    log.info(
        "Differential LR — Backbone: %.2e | Classifier: %.2e",
        lr_backbone, lr_classifier,
    )
    print(f"  Backbone LR   : {lr_backbone:.2e}")
    print(f"  Classifier LR : {lr_classifier:.2e}")

    # -----------------------------------------------------------------------
    # Loss, optimizer, scheduler
    # -----------------------------------------------------------------------
    class_weights = _compute_class_weights(train_df, DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-7,
    )
    scaler = GradScaler(enabled=use_amp)

    # -----------------------------------------------------------------------
    # Experiment directory
    # -----------------------------------------------------------------------
    experiment_id = f"idrid_{generate_experiment_id(MODEL_NAME, random_seed)}"
    exp_dir = create_experiment_dir(EXPERIMENTS_DIR, experiment_id)
    IDRID_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)

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

        # Save last checkpoint
        torch.save(model.state_dict(), IDRID_LAST_CKPT)

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc  = val_acc
            no_improve    = 0
            torch.save(model.state_dict(), IDRID_CHECKPOINT)
            log.info(
                "[*] IDRiD best ckpt saved — epoch %d (Val Loss: %.4f | Acc: %.2f%%)",
                epoch, best_val_loss, best_val_acc * 100,
            )
        else:
            no_improve += 1
            log.info("No improvement for %d epoch(s) (patience=%d).", no_improve, patience)

        if no_improve >= patience:
            log.info("Early stopping at epoch %d.", epoch)
            print(f"\n[Early Stopping] Triggered at epoch {epoch}.")
            break

    print(f"{'='*65}")
    print(f"  Training Complete")
    print(f"  Best Val Loss  : {best_val_loss:.4f}")
    print(f"  Best Val Acc   : {best_val_acc*100:.2f}%")
    print(f"  Checkpoint     : {IDRID_CHECKPOINT}")

    # -----------------------------------------------------------------------
    # Save training curves & experiment metadata
    # -----------------------------------------------------------------------
    save_training_curves(history, exp_dir)

    idrid_metrics_dir = BASE_DIR / "outputs" / "metrics" / "idrid"
    idrid_metrics_dir.mkdir(parents=True, exist_ok=True)
    save_training_curves(history, idrid_metrics_dir)

    # -----------------------------------------------------------------------
    # 5. Post-training evaluation on IDRiD test split
    # -----------------------------------------------------------------------
    print(f"\n[Step 5/5] Evaluating on IDRiD test split ({len(test_df)} images)...")

    # Load best checkpoint for eval
    model.load_state_dict(
        torch.load(IDRID_CHECKPOINT, map_location=DEVICE, weights_only=True)
    )
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    test_acc = accuracy_score(all_labels, all_preds)
    test_f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    class_names_list = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]

    print(f"\n{'='*65}")
    print(f"  IDRiD Test Set Evaluation (after fine-tuning)")
    print(f"{'='*65}")
    print(f"  Accuracy   : {test_acc*100:.2f}%")
    print(f"  Macro F1   : {test_f1*100:.2f}%")
    print(f"\n{classification_report(all_labels, all_preds, target_names=class_names_list, zero_division=0)}")

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

    # -----------------------------------------------------------------------
    # Save summary
    # -----------------------------------------------------------------------
    train_summary = {
        "experiment_id":         experiment_id,
        "architecture":          MODEL_NAME,
        "dataset":               "IDRiD (fine-tuned from APTOS 2019)",
        "transfer_learning":     not from_scratch,
        "source_checkpoint":     str(CHECKPOINT_PATH) if not from_scratch else "None",
        "num_classes":           NUM_CLASSES,
        "epochs_run":            len(history["train_loss"]),
        "max_epochs":            num_epochs,
        "batch_size":            batch_size,
        "lr_backbone":           lr_backbone,
        "lr_classifier":         lr_classifier,
        "weight_decay":          weight_decay,
        "optimizer":             "AdamW (differential LR)",
        "scheduler":             "CosineAnnealingLR",
        "loss":                  "CrossEntropyLoss (weighted)",
        "early_stopping_patience": patience,
        "use_amp":               use_amp,
        "device":                str(DEVICE),
        "random_seed":           random_seed,
        "best_val_loss":         round(best_val_loss, 6),
        "best_val_acc":          round(best_val_acc, 6),
        "test_accuracy":         round(test_acc, 6),
        "test_macro_f1":         round(test_f1, 6),
        "confusion_matrix":      cm.tolist(),
        "checkpoint_path":       str(IDRID_CHECKPOINT),
        "training_history":      history,
        "idrid_split": {
            "train": len(train_df),
            "val":   len(val_df),
            "test":  len(test_df),
        },
    }
    save_json(train_summary, exp_dir / "training_summary.json")
    save_json(train_summary, idrid_metrics_dir / "idrid_training_summary.json")

    print(f"\n  Experiment dir : {exp_dir}")
    print(f"  IDRiD checkpoint saved at: {IDRID_CHECKPOINT}")
    print(f"{'='*65}\n")

    return train_summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune VINAYAK DR model on IDRiD dataset."
    )
    parser.add_argument("--epochs",       type=int,   default=IDRID_EPOCHS,
                        help=f"Max epochs (default: {IDRID_EPOCHS})")
    parser.add_argument("--batch_size",   type=int,   default=IDRID_BATCH_SIZE,
                        help=f"Batch size (default: {IDRID_BATCH_SIZE})")
    parser.add_argument("--lr",           type=float, default=IDRID_LR_CLASSIFIER,
                        help=f"Classifier LR (default: {IDRID_LR_CLASSIFIER})")
    parser.add_argument("--lr_backbone",  type=float, default=IDRID_LR_BACKBONE,
                        help=f"Backbone LR (default: {IDRID_LR_BACKBONE})")
    parser.add_argument("--patience",     type=int,   default=IDRID_PATIENCE,
                        help=f"Early stopping patience (default: {IDRID_PATIENCE})")
    parser.add_argument("--seed",         type=int,   default=RANDOM_SEED,
                        help=f"Random seed (default: {RANDOM_SEED})")
    parser.add_argument("--from_scratch", action="store_true",
                        help="Skip APTOS checkpoint, train from ImageNet only")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_idrid(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr_backbone=args.lr_backbone,
        lr_classifier=args.lr,
        patience=args.patience,
        from_scratch=args.from_scratch,
        random_seed=args.seed,
    )
