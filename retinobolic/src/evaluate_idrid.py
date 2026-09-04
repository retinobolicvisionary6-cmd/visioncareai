"""
src/evaluate_idrid.py — Cross-Dataset External Validation on IDRiD.

Evaluates our trained EfficientNet-B0 model (trained on APTOS from Tamil Nadu)
on the external IDRiD dataset (from Maharashtra) to demonstrate Domain Generalization.

Usage:
    python src/evaluate_idrid.py
"""

import sys
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, recall_score, f1_score

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from configs.config import CLASS_NAMES, CHECKPOINT_PATH, DEVICE, NUM_CLASSES
from src.model import build_model
from src.dataset import FundusDataset
from src.preprocess import get_val_transforms
from src.inference import _apply_calibrated_thresholds

def run_idrid_validation():
    csv_path = BASE_DIR / "data" / "idrid" / "idrid_standardized.csv"
    if not csv_path.exists():
        print(f"[ERROR] Standardized IDRiD CSV not found at: {csv_path}")
        print("Please run: python data/prepare_idrid.py --idrid_dir path/to/idrid first.")
        return

    df = pd.read_csv(csv_path)
    print("=" * 65)
    print("  VISIONARY6 — ZERO-SHOT EXTERNAL VALIDATION (IDRiD)")
    print(f"  Total External Images: {len(df)}")
    print("=" * 65)

    img_dir = Path(df["file_path"].iloc[0]).parent
    dataset = FundusDataset(df, img_dir=img_dir, transform=get_val_transforms())
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Load model
    model = build_model(architecture="efficientnet_b0", num_classes=NUM_CLASSES, pretrained=False, device=DEVICE)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt)
    model.eval()

    all_probs, all_targets = [], []
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.numpy())

    all_probs = np.vstack(all_probs)
    all_targets = np.concatenate(all_targets)

    # Apply our calibrated thresholding engine
    preds = np.array([_apply_calibrated_thresholds(p) for p in all_probs])

    acc = accuracy_score(all_targets, preds)
    f1 = f1_score(all_targets, preds, average="macro")
    recall = recall_score(all_targets, preds, average="macro")

    print("\n[RESULTS] External Validation Metrics on Unseen Indian Clinic (IDRiD):")
    print(f"  • External Accuracy     : {acc * 100:.2f}%")
    print(f"  • Macro F1-Score        : {f1 * 100:.2f}%")
    print(f"  • Macro Sensitivity     : {recall * 100:.2f}%")

    print("\nPer-Class Sensitivity/Recall:")
    for i, name in CLASS_NAMES.items():
        rec_i = recall_score(all_targets, preds, labels=[i], average="macro", zero_division=0)
        print(f"  [{i}] {name:<15}: {rec_i * 100:5.1f}%")

    print("=" * 65)

if __name__ == "__main__":
    run_idrid_validation()
