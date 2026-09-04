"""
Training Pipeline — VISIONARY6 VINAYAK Module.

Phases covered:
  Phase 6  – Full training loop with checkpointing
  Phase 5  – Class imbalance handling (weighted loss + WeightedRandomSampler)
  Phase 14 – All hyperparameters read from config
  Phase 15 – Hardware auto-detection, mixed precision
  Phase 17 – Experiment tracking (timestamped directories)

Usage:
    python src/train.py
    python src/train.py --model resnet50 --epochs 20 --batch_size 8
"""
import sys
import json
import time
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server/script use
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.config import (
    BASE_DIR,
    CLASS_NAMES, NUM_CLASSES,
    MODEL_NAME, DROPOUT_RATE,
    CHECKPOINT_PATH, LAST_CHECKPOINT, FINAL_MODEL_PATH,
    CLASS_MAPPING_PATH, TRAIN_CONFIG_PATH,
    SPLITS_DIR, RAW_DATA_DIR,
    METRICS_DIR,
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, RANDOM_SEED,
    DEVICE, USE_AMP,
    USE_WEIGHTED_SAMPLER, USE_CLASS_WEIGHTS_LOSS,
    EARLY_STOPPING_PATIENCE,
    EXPERIMENTS_DIR,
    get_logger,
)
from src.dataset import get_dataloaders
from src.model import build_model
from src.utils import (
    set_seed, print_device_info,
    generate_experiment_id, create_experiment_dir, save_experiment_config,
    save_json, format_metrics_table,
)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Loss with class weights
# ---------------------------------------------------------------------------

def compute_class_weights(train_csv: Path, device: str) -> torch.Tensor:
    """
    Computes inverse-frequency class weights from the training split.

    Returns a tensor of shape [NUM_CLASSES] on the specified device.
    """
    df = pd.read_csv(train_csv)
    counts = df["mapped_grade"].value_counts().sort_index()
    # Ensure all classes are represented
    counts = counts.reindex(range(NUM_CLASSES), fill_value=1)
    total = counts.sum()
    weights = [total / (NUM_CLASSES * c) for c in counts.values]
    log.info("Class weights: %s", {CLASS_NAMES[i]: round(w, 3) for i, w in enumerate(weights)})
    return torch.tensor(weights, dtype=torch.float32).to(device)


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------

def run_train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    scaler: GradScaler,
    use_amp: bool,
    epoch: int,
    num_epochs: int,
) -> tuple[float, float]:
    """Single training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    with tqdm(loader, desc=f"Epoch {epoch:02d}/{num_epochs} [Train]", leave=False) as pbar:
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            if use_amp:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            bs = labels.size(0)
            total_loss += loss.item() * bs
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += bs
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Validation epoch
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_val_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str,
    epoch: int,
    num_epochs: int,
    split_name: str = "Val",
) -> tuple[float, float]:
    """Validation or test epoch. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with tqdm(loader, desc=f"Epoch {epoch:02d}/{num_epochs} [{split_name}]", leave=False) as pbar:
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            bs = labels.size(0)
            total_loss += loss.item() * bs
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += bs

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_training_curves(history: dict, output_dir: Path) -> None:
    """Saves training loss and validation accuracy curves."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", marker="o", markersize=3)
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   marker="o", markersize=3)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["val_acc"], label="Val Accuracy", color="green", marker="o", markersize=3)
    axes[1].plot(epochs, history["train_acc"], label="Train Accuracy", color="blue", marker="o", markersize=3)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curves")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close()
    log.info("Training curves saved to: %s", output_dir / "training_curves.png")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_model(
    model_name: str = MODEL_NAME,
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    splits_dir: Path = SPLITS_DIR,
    img_dir: Path = None,
    device: str = DEVICE,
    use_amp: bool = USE_AMP,
    use_weighted_sampler: bool = USE_WEIGHTED_SAMPLER,
    use_class_weights_loss: bool = USE_CLASS_WEIGHTS_LOSS,
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
    random_seed: int = RANDOM_SEED,
    experiment_id: str = None,
    resume: bool = False,
) -> dict:
    """
    Full training pipeline: dataset → model → training → validation →
    checkpointing → experiment logging.

    Args:
        model_name:              Architecture key (e.g. 'efficientnet_b0').
        num_epochs:              Maximum training epochs.
        batch_size:              Batch size.
        learning_rate:           Initial learning rate (AdamW).
        weight_decay:            L2 regularisation coefficient.
        splits_dir:              Directory containing train/val/test CSV files.
        img_dir:                 Directory with fundus image files (auto-detected if None).
        device:                  'cuda' or 'cpu'.
        use_amp:                 Enable automatic mixed precision (CUDA only).
        use_weighted_sampler:    Use WeightedRandomSampler in training DataLoader.
        use_class_weights_loss:  Apply inverse-frequency class weights to the loss.
        early_stopping_patience: Stop training if val loss doesn't improve for N epochs.
        random_seed:             Global random seed.
        experiment_id:           Optional experiment ID. Auto-generated if None.

    Returns:
        Dictionary with training summary and best metrics.
    """
    # --- Reproducibility ---
    set_seed(random_seed)
    print_device_info(device)

    # --- Experiment directory ---
    if experiment_id is None:
        experiment_id = generate_experiment_id(model_name, random_seed)
    exp_dir = create_experiment_dir(EXPERIMENTS_DIR, experiment_id)

    log.info("="*60)
    log.info("VISIONARY6 — VINAYAK Training Run")
    log.info("Experiment ID : %s", experiment_id)
    log.info("Model         : %s", model_name)
    log.info("Epochs        : %d", num_epochs)
    log.info("Batch size    : %d", batch_size)
    log.info("Learning rate : %s", learning_rate)
    log.info("Device        : %s", device)
    log.info("AMP           : %s", use_amp)
    log.info("="*60)

    splits_dir = Path(splits_dir)
    splits_384 = BASE_DIR / "data" / "splits_384"
    if splits_384.exists() and (splits_384 / "train.csv").exists():
        splits_dir = splits_384
        log.info("Using 384x384 pre-processed splits for ~10x training acceleration: %s", splits_dir)
    train_csv = splits_dir / "train.csv"

    # --- DataLoaders ---
    train_loader, val_loader, _ = get_dataloaders(
        splits_dir=splits_dir,
        img_dir=img_dir,
        batch_size=batch_size,
        use_weighted_sampler=use_weighted_sampler,
    )

    # --- Model ---
    model = build_model(
        architecture=model_name,
        num_classes=NUM_CLASSES,
        pretrained=True,
        dropout_rate=DROPOUT_RATE,
        device=device,
        verbose=True,
    )
    
    if resume:
        if CHECKPOINT_PATH.exists():
            model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
            log.info(f"Successfully resumed weights from {CHECKPOINT_PATH}")
        else:
            log.warning(f"Resume requested but {CHECKPOINT_PATH} not found!")

    # --- Loss ---
    if use_class_weights_loss:
        class_weights = compute_class_weights(train_csv, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        log.info("Using weighted CrossEntropyLoss.")
    else:
        criterion = nn.CrossEntropyLoss()

    # --- Optimiser & Scheduler ---
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )
    scaler = GradScaler(enabled=use_amp)

    # --- Training state ---
    best_val_loss = float("inf")
    best_val_acc  = 0.0
    no_improve_count = 0
    history: dict[str, list[float]] = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
        "lr":         [],
    }

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # --- Training loop ---
    for epoch in range(1, num_epochs + 1):
        t_start = time.time()

        train_loss, train_acc = run_train_epoch(
            model, train_loader, criterion, optimizer,
            device, scaler, use_amp, epoch, num_epochs
        )
        val_loss, val_acc = run_val_epoch(
            model, val_loader, criterion, device, epoch, num_epochs
        )
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t_start

        history["train_loss"].append(round(train_loss, 6))
        history["train_acc"].append(round(train_acc, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["val_acc"].append(round(val_acc, 6))
        history["lr"].append(current_lr)

        print(
            f"Epoch {epoch:02d}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% | "
            f"LR: {current_lr:.2e} | Time: {elapsed:.1f}s"
        )

        # --- Save last checkpoint ---
        torch.save(model.state_dict(), LAST_CHECKPOINT)

        # --- Save best checkpoint ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc  = val_acc
            no_improve_count = 0
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            log.info(
                "[*] Best checkpoint saved at epoch %d (Val Loss: %.4f | Acc: %.2f%%)",
                epoch, best_val_loss, best_val_acc * 100,
            )
        else:
            no_improve_count += 1
            log.info(
                "No improvement for %d epoch(s) (patience=%d).",
                no_improve_count, early_stopping_patience,
            )

        # --- Early stopping ---
        if no_improve_count >= early_stopping_patience:
            log.info(
                "Early stopping triggered at epoch %d (no improvement for %d epochs).",
                epoch, early_stopping_patience,
            )
            print(f"\n[Early Stopping] Triggered at epoch {epoch}.")
            break

    # --- Save training curves ---
    save_training_curves(history, METRICS_DIR)
    save_training_curves(history, exp_dir)   # also in experiment dir

    # --- Save class mapping ---
    save_json(
        {str(k): v for k, v in CLASS_NAMES.items()},
        CLASS_MAPPING_PATH,
    )

    # --- Save training config ---
    train_summary = {
        "experiment_id":     experiment_id,
        "architecture":      model_name,
        "dataset":           "APTOS 2019",
        "num_classes":       NUM_CLASSES,
        "epochs_run":        len(history["train_loss"]),
        "max_epochs":        num_epochs,
        "batch_size":        batch_size,
        "learning_rate":     learning_rate,
        "weight_decay":      weight_decay,
        "optimizer":         "AdamW",
        "scheduler":         "CosineAnnealingLR",
        "loss":              "CrossEntropyLoss",
        "use_class_weights": use_class_weights_loss,
        "use_weighted_sampler": use_weighted_sampler,
        "early_stopping_patience": early_stopping_patience,
        "use_amp":           use_amp,
        "device":            str(device),
        "random_seed":       random_seed,
        "augmentation":      "HFlip, VFlip, Rotate±15, ColorJitter(b=0.1)",
        "best_val_loss":     round(best_val_loss, 6),
        "best_val_acc":      round(best_val_acc, 6),
        "checkpoint_path":   str(CHECKPOINT_PATH),
        "training_history":  history,
    }
    save_json(train_summary, TRAIN_CONFIG_PATH)
    save_json(train_summary, exp_dir / "training_summary.json")

    log.info("Training complete. Best checkpoint: %s", CHECKPOINT_PATH)
    print(f"\n{'='*55}")
    print(f"  Training Complete")
    print(f"  Best Val Loss  : {best_val_loss:.4f}")
    print(f"  Best Val Acc   : {best_val_acc*100:.2f}%")
    print(f"  Checkpoint     : {CHECKPOINT_PATH}")
    print(f"  Experiment dir : {exp_dir}")
    print(f"{'='*55}\n")

    return train_summary


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Train VINAYAK 4-class DR classification model."
    )
    parser.add_argument("--model",      default=MODEL_NAME,    help="Architecture name")
    parser.add_argument("--epochs",     type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=LEARNING_RATE)
    parser.add_argument("--seed",       type=int, default=RANDOM_SEED)
    parser.add_argument("--no_amp",     action="store_true", help="Disable AMP")
    parser.add_argument("--resume",     action="store_true", help="Resume from best_model.pth")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_model(
        model_name=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        random_seed=args.seed,
        use_amp=USE_AMP and not args.no_amp,
        resume=args.resume
    )
