"""
tune_engine.py  —  Complete Reliability Engine Calibration on APTOS
====================================================================
SIH26038: Explainable AI for Diabetic Retinopathy Screening

Runs with:  py -3.12 D:\\anuj-reliability\\tune_engine.py

What this script does (using only 80 validation images):
  1. Loads Vinayak's trained EfficientNet-B0 from checkpoint
  2. Runs inference on 80 APTOS val images (fast, CPU only)
  3. Computes per-image confidence and Shannon entropy
  4. Finds optimal thresholds for Confidence and Uncertainty modules
  5. Verifies the OOD calibration we already ran
  6. Writes final tuned thresholds to:
       D:\\anuj-reliability\\config\\thresholds.json
       D:\\anuj-reliability\\config\\tuning_report.json
"""
from __future__ import annotations

import sys
import json
import math
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
RETINOBOLIC_ROOT = Path(r"E:\retinobolic")
ANUJ_OOD_ROOT    = Path(r"D:\anuj-ood")
RELIABILITY_ROOT = Path(r"D:\anuj-reliability")

sys.path.insert(0, str(RETINOBOLIC_ROOT))        # for configs.*
sys.path.insert(0, str(RETINOBOLIC_ROOT / "src")) # for src.model, src.preprocess
sys.path.insert(0, str(ANUJ_OOD_ROOT))

print("\n" + "=" * 65)
print("  RELIABILITY ENGINE — Complete APTOS Calibration")
print("  SIH26038 | Modules: Confidence, Uncertainty, OOD")
print("=" * 65)

# ---------------------------------------------------------------------------
# 1. Load PyTorch model
# ---------------------------------------------------------------------------
import torch
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n[1/5] Loading Vinayak's DR Model on {device.upper()}...")

# Remap Vinayak's hardcoded Windows path to this machine's E: drive
CKPT = RETINOBOLIC_ROOT / "models" / "checkpoints" / "best_model.pth"
if not CKPT.exists():
    print(f"ERROR: Checkpoint not found at {CKPT}"); sys.exit(1)

import torchvision.models as models
import torch.nn as nn

def build_efficientnet_b0(num_classes=4, dropout_rate=0.3):
    backbone = models.efficientnet_b0(weights=None)
    in_features = backbone.classifier[1].in_features
    backbone.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return backbone

model = build_efficientnet_b0()
state = torch.load(str(CKPT), map_location=device, weights_only=True)
# The checkpoint keys are prefixed with "backbone." — strip that prefix
new_state = {}
for k, v in state.items():
    if k.startswith("backbone."):
        new_state[k[len("backbone."):]] = v
    elif not k.startswith("_target_layer"):
        new_state[k] = v
model.load_state_dict(new_state, strict=False)
model.eval()
model.to(device)
print(f"  Model loaded from {CKPT.name}")

# ---------------------------------------------------------------------------
# 2. Load val.csv and remap image paths
# ---------------------------------------------------------------------------
print("\n[2/5] Loading 80 stratified APTOS val images...")

import pandas as pd

VAL_CSV = RETINOBOLIC_ROOT / "data" / "splits" / "val.csv"
df = pd.read_csv(VAL_CSV)
df["local_path"] = df["image_id"].apply(
    lambda x: str(RETINOBOLIC_ROOT / "data" / "raw" / "train_images" / x)
)
df = df[df["local_path"].apply(lambda p: Path(p).exists())]

# Take 20 images per grade (balanced), max 80 total
samples = []
for grade in [0, 1, 2, 3]:
    grade_df = df[df["mapped_grade"] == grade].head(20)
    samples.append(grade_df)

import pandas as pd
df_sample = pd.concat(samples).reset_index(drop=True)
print(f"  Using {len(df_sample)} images | Grade distribution:")
for g in [0, 1, 2, 3]:
    n = (df_sample["mapped_grade"] == g).sum()
    print(f"    Grade {g}: {n} images")

# ---------------------------------------------------------------------------
# 3. Run inference — collect probabilities
# ---------------------------------------------------------------------------
print("\n[3/5] Running inference (CPU)...")

import cv2
from PIL import Image
import torchvision.transforms as T

_VAL_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess_single_image(image_path):
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    return _VAL_TRANSFORM(pil_img).unsqueeze(0)

results = []
for i, row in df_sample.iterrows():
    try:
        tensor = preprocess_single_image(row["local_path"])
        tensor = tensor.to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()

        true_grade = int(row["mapped_grade"])
        pred_grade = int(probs.argmax())
        max_prob   = float(probs.max())
        correct    = (pred_grade == true_grade)

        # Normalized Shannon entropy [0, 1] where 1 = maximum uncertainty
        entropy = -sum(p * math.log(p + 1e-9) for p in probs) / math.log(4)

        results.append({
            "image_id":   row["image_id"],
            "true_grade": true_grade,
            "pred_grade": pred_grade,
            "correct":    correct,
            "probs":      probs.tolist(),
            "max_prob":   max_prob,
            "entropy":    entropy,
        })
        if (i + 1) % 20 == 0:
            done = i + 1
            print(f"  Processed {done}/{len(df_sample)}...")
    except Exception as e:
        print(f"  SKIP {row['image_id']}: {e}")

print(f"  Done — {len(results)} inference results collected")

# ---------------------------------------------------------------------------
# 4. Compute threshold statistics
# ---------------------------------------------------------------------------
print("\n[4/5] Computing optimal thresholds...")

import numpy as np

correct_probs   = [r["max_prob"]  for r in results if r["correct"]]
incorrect_probs = [r["max_prob"]  for r in results if not r["correct"]]
correct_ent     = [r["entropy"]   for r in results if r["correct"]]
incorrect_ent   = [r["entropy"]   for r in results if not r["correct"]]
overall_acc     = sum(r["correct"] for r in results) / len(results)

print(f"\n  Overall Accuracy on sample: {overall_acc * 100:.1f}%")

# -------- Confidence thresholds --------
#  HIGH_CONFIDENCE: top % of correct predictions (p90 of correct probs)
#  LOW_CONFIDENCE : below which model mostly fails (p50 of incorrect probs)
HIGH_CONF = round(float(np.percentile(correct_probs, 50)), 2)   # median correct
LOW_CONF  = round(float(np.percentile(incorrect_probs, 75)), 2) # 75th pct of wrong

print(f"\n  CONFIDENCE:")
print(f"    Median prob when CORRECT  : {np.median(correct_probs):.4f}")
print(f"    Median prob when WRONG    : {np.median(incorrect_probs):.4f}")
print(f"  -> HIGH_CONFIDENCE_THRESHOLD = {HIGH_CONF}")
print(f"  -> LOW_CONFIDENCE_THRESHOLD  = {LOW_CONF}")

# -------- Uncertainty thresholds --------
#  LOW_UNCERTAINTY_MAX : 90th percentile of correct-case entropy
#  HIGH_UNCERTAINTY_MIN: 25th percentile of incorrect-case entropy
LOW_UNC  = round(float(np.percentile(correct_ent, 90)), 2)
HIGH_UNC = round(float(np.percentile(incorrect_ent, 25)), 2)

print(f"\n  UNCERTAINTY (Shannon Entropy, normalised 0-1):")
print(f"    Median entropy when CORRECT: {np.median(correct_ent):.4f}")
print(f"    Median entropy when WRONG  : {np.median(incorrect_ent):.4f}")
print(f"  -> LOW_UNCERTAINTY_MAX  = {LOW_UNC}")
print(f"  -> HIGH_UNCERTAINTY_MIN = {HIGH_UNC}")

# -------- OOD Verification --------
print("\n[5/5] Verifying OOD calibration on APTOS val images...")

sys.path.insert(0, str(ANUJ_OOD_ROOT))
from src.ood import detect_ood

ood_pass = 0
ood_flag = 0
ood_errors = 0
for r in results[:40]:  # check 40 known-good fundus images
    try:
        img_path = str(RETINOBOLIC_ROOT / "data" / "raw" / "train_images" / r["image_id"])
        ood_result = detect_ood(img_path)
        if ood_result["ood"]:
            ood_flag += 1
        else:
            ood_pass += 1
    except Exception as e:
        ood_errors += 1

total_checked = ood_pass + ood_flag + ood_errors
ood_pass_rate = (ood_pass / max(total_checked - ood_errors, 1)) * 100

print(f"  APTOS images -> In-distribution: {ood_pass} | Flagged OOD: {ood_flag} | Errors: {ood_errors}")
print(f"  OOD Pass Rate for real retina images: {ood_pass_rate:.1f}%")

# -------- Safety guards --------
# Ensure thresholds don't become degenerate
HIGH_CONF = max(HIGH_CONF, 0.55)
LOW_CONF  = min(LOW_CONF, 0.65)
LOW_UNC   = max(LOW_UNC,  0.25)
HIGH_UNC  = min(HIGH_UNC, 0.80)

# -------- Write thresholds.json --------
thresholds = {
    "_comment": "Reliability Engine — APTOS-calibrated thresholds (SIH26038).",
    "_warning": "Calibrated on 80 stratified APTOS 2019 val images. Requires clinical validation before operational use.",
    "_accuracy_on_sample": round(overall_acc * 100, 1),
    "_ood_pass_rate_pct": round(ood_pass_rate, 1),
    "HIGH_CONFIDENCE_THRESHOLD": HIGH_CONF,
    "LOW_CONFIDENCE_THRESHOLD":  LOW_CONF,
    "LOW_UNCERTAINTY_MAX":       LOW_UNC,
    "HIGH_UNCERTAINTY_MIN":      HIGH_UNC,
    "OOD_STRICT_MODE":           True
}

THRESHOLDS_PATH = RELIABILITY_ROOT / "config" / "thresholds.json"
with open(THRESHOLDS_PATH, "w") as f:
    json.dump(thresholds, f, indent=2)
print(f"\n  Thresholds written -> {THRESHOLDS_PATH}")

# -------- Write detailed tuning report --------
report = {
    "calibration_date": str(pd.Timestamp.now()),
    "dataset": "APTOS 2019 validation split",
    "n_images_used": len(results),
    "overall_accuracy_pct": round(overall_acc * 100, 1),
    "ood_pass_rate_pct": round(ood_pass_rate, 1),
    "confidence": {
        "correct_median": round(np.median(correct_probs), 4),
        "incorrect_median": round(np.median(incorrect_probs), 4),
        "HIGH_CONFIDENCE_THRESHOLD": HIGH_CONF,
        "LOW_CONFIDENCE_THRESHOLD": LOW_CONF,
    },
    "uncertainty": {
        "correct_median_entropy": round(np.median(correct_ent), 4),
        "incorrect_median_entropy": round(np.median(incorrect_ent), 4),
        "LOW_UNCERTAINTY_MAX": LOW_UNC,
        "HIGH_UNCERTAINTY_MIN": HIGH_UNC,
    },
    "ood": {
        "threshold": 12.74,
        "n_checked": total_checked,
        "flagged": ood_flag,
        "passed": ood_pass,
    },
    "per_image_results": results[:10]  # first 10 for reference
}

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'item'):
            return obj.item()
        return super().default(obj)

REPORT_PATH = RELIABILITY_ROOT / "config" / "tuning_report.json"
with open(REPORT_PATH, "w") as f:
    json.dump(report, f, indent=2, cls=NumpyEncoder)
print(f"  Tuning report written -> {REPORT_PATH}")

# ---------------------------------------------------------------------------
# Final Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("  CALIBRATION COMPLETE")
print("=" * 65)
print(f"  Model Accuracy on sample     : {overall_acc * 100:.1f}%")
print(f"  OOD Pass Rate (APTOS images) : {ood_pass_rate:.1f}%")
print()
print("  TUNED THRESHOLDS:")
print(f"    HIGH_CONFIDENCE_THRESHOLD  : {HIGH_CONF}")
print(f"    LOW_CONFIDENCE_THRESHOLD   : {LOW_CONF}")
print(f"    LOW_UNCERTAINTY_MAX        : {LOW_UNC}")
print(f"    HIGH_UNCERTAINTY_MIN       : {HIGH_UNC}")
print(f"    OOD Threshold (anuj-ood)   : 12.74")
print("=" * 65)
print("  Written: config/thresholds.json")
print("  Written: config/tuning_report.json")
print()
print("  Run: python -m pytest tests/ -v  to verify all tests still pass")
print("=" * 65)
