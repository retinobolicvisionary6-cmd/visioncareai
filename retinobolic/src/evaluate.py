"""
Evaluation Module — VISIONARY6 VINAYAK Module.

Phase 7 — Full evaluation suite:
  - Accuracy, Precision, Recall/Sensitivity, Specificity, F1
  - Per-class metrics (all of the above)
  - Confusion Matrix (plot + JSON)
  - ROC-AUC (OvR macro-average)
  - PR-AUC (OvR macro-average)
  - Prediction CSV export
  - Concise evaluation report

IMPORTANT:
  Metrics are computed on the held-out test split only.
  This is a standard dataset split evaluation — NOT clinical validation.
  Do not interpret these results as proof of clinical deployment readiness.

Usage:
    python src/evaluate.py
    python src/evaluate.py --checkpoint models/checkpoints/best_model.pth
"""
import sys
import json
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.config import (
    CLASS_NAMES, NUM_CLASSES,
    CHECKPOINT_PATH, MODEL_NAME, DROPOUT_RATE,
    SPLITS_DIR, RAW_DATA_DIR,
    METRICS_DIR, CONFUSION_MATRIX_DIR, PREDICTIONS_DIR,
    DEVICE, get_logger,
)
from src.dataset import FundusDataset
from src.preprocess import get_val_transforms
from src.model import build_model
from src.utils import save_json, format_metrics_table

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Specificity helper
# ---------------------------------------------------------------------------

def compute_per_class_specificity(cm: np.ndarray) -> np.ndarray:
    """
    Computes per-class specificity from a confusion matrix.
    Specificity (True Negative Rate) = TN / (TN + FP)

    Args:
        cm: Square confusion matrix of shape [C, C].

    Returns:
        Array of shape [C] with per-class specificity values.
    """
    n = cm.shape[0]
    total = cm.sum()
    specificities = []
    for i in range(n):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total - tp - fp - fn
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(spec)
    return np.array(specificities)


# ---------------------------------------------------------------------------
# Inference pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs model inference on the full dataloader.

    Returns:
        all_targets:  Ground truth labels [N]
        all_preds:    Predicted class indices [N]
        all_probs:    Softmax probability distributions [N, C]
    """
    model.eval()
    targets_list: list[np.ndarray] = []
    preds_list: list[np.ndarray] = []
    probs_list: list[np.ndarray] = []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)

        targets_list.append(labels.numpy())
        preds_list.append(preds)
        probs_list.append(probs)

    return (
        np.concatenate(targets_list),
        np.concatenate(preds_list),
        np.vstack(probs_list),
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    output_dir: Path,
) -> Path:
    """Saves a heatmap of the confusion matrix. Returns output path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "confusion_matrix.png"

    plt.figure(figsize=(8, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5,
    )
    plt.xlabel("Predicted Grade", fontsize=12)
    plt.ylabel("Actual Grade",    fontsize=12)
    plt.title("Confusion Matrix — 4-Class DR Grading\n(AI-Assisted Screening | Not Clinically Validated)", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    log.info("Confusion matrix saved: %s", output_path)
    return output_path


def plot_roc_curves(
    all_targets: np.ndarray,
    all_probs: np.ndarray,
    class_names: list[str],
    output_dir: Path,
) -> Path:
    """
    Plots One-vs-Rest ROC curves for each class.
    Returns output path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "roc_curves.png"

    n_classes = len(class_names)
    targets_bin = label_binarize(all_targets, classes=list(range(n_classes)))

    plt.figure(figsize=(9, 7))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, (name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(targets_bin[:, i], all_probs[:, i])
        auc_val = roc_auc_score(targets_bin[:, i], all_probs[:, i])
        plt.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc_val:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("ROC Curves — One-vs-Rest (per class)\n(Model Evidence | Not Clinical Validation)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    log.info("ROC curves saved: %s", output_path)
    return output_path


def plot_pr_curves(
    all_targets: np.ndarray,
    all_probs: np.ndarray,
    class_names: list[str],
    output_dir: Path,
) -> Path:
    """
    Plots One-vs-Rest Precision-Recall curves for each class.
    Returns output path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "pr_curves.png"

    n_classes = len(class_names)
    targets_bin = label_binarize(all_targets, classes=list(range(n_classes)))

    plt.figure(figsize=(9, 7))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, (name, color) in enumerate(zip(class_names, colors)):
        prec, rec, _ = precision_recall_curve(targets_bin[:, i], all_probs[:, i])
        ap = average_precision_score(targets_bin[:, i], all_probs[:, i])
        plt.plot(rec, prec, color=color, lw=2, label=f"{name} (AP={ap:.3f})")

    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves — One-vs-Rest\n(Model Evidence | Not Clinical Validation)")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    log.info("PR curves saved: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_model(
    checkpoint_path: Optional[Path] = None,
    model_name: str = MODEL_NAME,
    splits_dir: Optional[Path] = None,
    img_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    batch_size: int = 16,
    device: str = DEVICE,
) -> dict:
    """
    Evaluates the best checkpoint on the held-out test split.

    Saves:
        - confusion_matrix.png
        - roc_curves.png
        - pr_curves.png
        - evaluation_report.json
        - predictions.csv

    Returns:
        Dictionary with all computed metrics.

    Notes:
        This evaluation uses the standard held-out test split only.
        It does NOT constitute clinical validation.
    """
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_PATH
    if splits_dir is None:
        splits_dir = SPLITS_DIR
    if output_dir is None:
        output_dir = METRICS_DIR

    checkpoint_path = Path(checkpoint_path)
    splits_dir = Path(splits_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("="*60)
    log.info("VISIONARY6 — VINAYAK Evaluation")
    log.info("Checkpoint : %s", checkpoint_path)
    log.info("Device     : %s", device)
    log.info("="*60)

    # --- Load test data ---
    test_csv = splits_dir / "test.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"Test split not found: {test_csv}. Run splitting first.")

    test_df = pd.read_csv(test_csv)

    if img_dir is None:
        if "file_path" in test_df.columns and pd.notna(test_df["file_path"].iloc[0]):
            img_dir = Path(test_df["file_path"].iloc[0]).parent
        else:
            img_dir = RAW_DATA_DIR / "train_images"

    test_dataset = FundusDataset(test_df, img_dir, transform=get_val_transforms())
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # --- Load model ---
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Please train the model first: python src/train.py"
        )

    model = build_model(
        architecture=model_name,
        num_classes=NUM_CLASSES,
        pretrained=False,
        dropout_rate=DROPOUT_RATE,
        device=device,
        verbose=False,
    )
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    log.info("Checkpoint loaded from: %s", checkpoint_path)

    # --- Collect predictions ---
    all_targets, all_preds, all_probs = collect_predictions(model, test_loader, device)

    # --- Overall metrics ---
    class_names = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]

    acc = float(accuracy_score(all_targets, all_preds))
    prec, rec, f1, support = precision_recall_fscore_support(
        all_targets, all_preds, average=None, zero_division=0, labels=list(range(NUM_CLASSES))
    )
    macro_prec = float(np.mean(prec))
    macro_rec  = float(np.mean(rec))
    macro_f1   = float(np.mean(f1))

    cm = confusion_matrix(all_targets, all_preds, labels=list(range(NUM_CLASSES)))
    per_class_spec = compute_per_class_specificity(cm)
    macro_spec = float(np.mean(per_class_spec))

    # ROC-AUC and PR-AUC (OvR macro)
    targets_bin = label_binarize(all_targets, classes=list(range(NUM_CLASSES)))
    roc_auc_macro = float(roc_auc_score(targets_bin, all_probs, multi_class="ovr", average="macro"))
    pr_auc_macro  = float(np.mean([
        average_precision_score(targets_bin[:, i], all_probs[:, i])
        for i in range(NUM_CLASSES)
    ]))

    # Per-class detail
    per_class = {}
    for i in range(NUM_CLASSES):
        per_class[CLASS_NAMES[i]] = {
            "precision":           round(float(prec[i]), 4),
            "recall_sensitivity":  round(float(rec[i]),  4),
            "specificity":         round(float(per_class_spec[i]), 4),
            "f1_score":            round(float(f1[i]),   4),
            "roc_auc":             round(float(roc_auc_score(targets_bin[:, i], all_probs[:, i])), 4),
            "pr_auc":              round(float(average_precision_score(targets_bin[:, i], all_probs[:, i])), 4),
            "support":             int(support[i]),
        }

    # --- Classification report (text) ---
    cls_report_text = classification_report(
        all_targets, all_preds, target_names=class_names, zero_division=0
    )

    # --- Assemble report ---
    report = {
        "evaluation_note": (
            "Results computed on the held-out test split. "
            "This is NOT clinical validation. "
            "Do not interpret as proof of deployment readiness."
        ),
        "model":      model_name,
        "checkpoint": str(checkpoint_path),
        "test_split": str(splits_dir / "test.csv"),
        "overall_metrics": {
            "accuracy":                round(acc, 4),
            "macro_precision":         round(macro_prec, 4),
            "macro_recall_sensitivity": round(macro_rec, 4),
            "macro_specificity":       round(macro_spec, 4),
            "macro_f1":                round(macro_f1,   4),
            "roc_auc_ovr_macro":       round(roc_auc_macro, 4),
            "pr_auc_macro":            round(pr_auc_macro,  4),
        },
        "per_class_metrics": per_class,
        "confusion_matrix":  cm.tolist(),
        "classification_report_text": cls_report_text,
    }

    save_json(report, output_dir / "evaluation_report.json")

    # --- Save predictions CSV ---
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    pred_df = pd.DataFrame({
        "image_id":      test_df["image_id"].values,
        "true_grade":    all_targets,
        "pred_grade":    all_preds,
        "prob_no_dr":    all_probs[:, 0].round(4),
        "prob_mild":     all_probs[:, 1].round(4),
        "prob_moderate": all_probs[:, 2].round(4),
        "prob_severe":   all_probs[:, 3].round(4),
        "correct":       (all_targets == all_preds),
    })
    pred_df.to_csv(PREDICTIONS_DIR / "test_predictions.csv", index=False)

    # --- Plots ---
    plot_confusion_matrix(cm, class_names, CONFUSION_MATRIX_DIR)
    plot_roc_curves(all_targets, all_probs, class_names, output_dir)
    plot_pr_curves( all_targets, all_probs, class_names, output_dir)

    # --- Print summary ---
    summary = {
        "Accuracy":          acc,
        "Macro Precision":   macro_prec,
        "Macro Recall/Sensitivity": macro_rec,
        "Macro Specificity": macro_spec,
        "Macro F1":          macro_f1,
        "ROC-AUC (OvR macro)": roc_auc_macro,
        "PR-AUC (macro)":    pr_auc_macro,
    }
    print(f"\n{'='*55}")
    print("  VINAYAK — Test Set Evaluation (AI-assisted DR Screening)")
    print(f"{'='*55}")
    print(format_metrics_table(summary))
    print(f"\n  Per-class details in: {output_dir / 'evaluation_report.json'}")
    print(f"  Confusion matrix   : {CONFUSION_MATRIX_DIR / 'confusion_matrix.png'}")
    print(f"  ROC curves         : {output_dir / 'roc_curves.png'}")
    print(f"  PR curves          : {output_dir / 'pr_curves.png'}")
    print(f"  Predictions CSV    : {PREDICTIONS_DIR / 'test_predictions.csv'}")
    print(f"\n  NOTE: This is a dataset-split evaluation only.")
    print(f"        It does NOT constitute clinical validation.")
    print(f"{'='*55}\n")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VINAYAK DR model on test set.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model",      type=str, default=MODEL_NAME)
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate_model(
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        model_name=args.model,
        batch_size=args.batch_size,
    )
