"""
Model Comparison Framework — VISIONARY6 VINAYAK Module.

Phase 8 — Compare ResNet50 vs EfficientNet-B0 vs MobileNet-V3-Small.

Evaluates each model checkpoint against the held-out test set and
produces a comparison table covering:
  - Accuracy, Precision, Recall/Sensitivity, Specificity, F1
  - ROC-AUC (OvR macro), PR-AUC
  - Parameter count, model size (MB), inference time (ms/image)

Also guides the final model selection process.

Usage:
    python src/compare_models.py
"""
import sys
import time
import json
from pathlib import Path

import torch
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.config import (
    NUM_CLASSES, DROPOUT_RATE, DEVICE, SPLITS_DIR, RAW_DATA_DIR,
    MODELS_DIR, METRICS_DIR, IMAGE_SIZE, get_logger,
)
from src.model import build_model
from src.evaluate import evaluate_model, collect_predictions
from src.dataset import FundusDataset
from src.preprocess import preprocess_single_image, get_val_transforms
from src.utils import count_parameters, get_model_size_mb, save_json
from torch.utils.data import DataLoader
import pandas as pd

log = get_logger(__name__)

ARCHITECTURES = {
    "EfficientNet-B0": "efficientnet_b0",
    "ResNet-50":       "resnet50",
    "MobileNet-V3-S":  "mobilenet_v3_small",
}

# Expected checkpoint per architecture — named by architecture
def _get_checkpoint(arch_key: str) -> Path:
    """Returns checkpoint path for a named architecture."""
    return MODELS_DIR / "checkpoints" / f"best_{arch_key}.pth"


def benchmark_inference_speed(model: torch.nn.Module, device: str, n_runs: int = 50) -> float:
    """
    Measures average inference time per image in milliseconds.

    Args:
        model:   Loaded model in eval mode.
        device:  'cuda' or 'cpu'.
        n_runs:  Number of warmup + timed forward passes.

    Returns:
        Average time per image in milliseconds.
    """
    dummy = torch.randn(1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]).to(device)
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            model(dummy)

    # Timed
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(dummy)
            times.append((time.perf_counter() - t0) * 1000)

    return round(float(np.mean(times)), 2)


def compare_all_models(
    splits_dir: Path = SPLITS_DIR,
    img_dir: Path = None,
    output_dir: Path = METRICS_DIR,
) -> pd.DataFrame:
    """
    Evaluates all available model checkpoints and builds a comparison table.

    Only models with existing checkpoints are evaluated.
    Missing checkpoints are documented in the table as 'not trained'.

    Returns:
        DataFrame with one row per model.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve img_dir
    if img_dir is None:
        test_csv = splits_dir / "test.csv"
        if test_csv.exists():
            import pandas as _pd
            tdf = _pd.read_csv(test_csv)
            if "file_path" in tdf.columns and pd.notna(tdf["file_path"].iloc[0]):
                img_dir = Path(tdf["file_path"].iloc[0]).parent
            else:
                img_dir = RAW_DATA_DIR / "train_images"
        else:
            img_dir = RAW_DATA_DIR / "train_images"

    rows = []
    print(f"\n{'='*70}")
    print("  VISIONARY6 — VINAYAK Model Comparison")
    print(f"{'='*70}")

    for display_name, arch_key in ARCHITECTURES.items():
        ckpt = _get_checkpoint(arch_key)
        print(f"\n  Evaluating: {display_name} ({arch_key})")

        if not ckpt.exists():
            # Check fallback: generic best_model.pth if only one arch was trained
            fallback = MODELS_DIR / "checkpoints" / "best_model.pth"
            if fallback.exists():
                log.info("Using fallback checkpoint for %s: %s", display_name, fallback)
                ckpt = fallback
            else:
                print(f"    [SKIP] Checkpoint not found: {ckpt}")
                rows.append({"Model": display_name, "Status": "not trained"})
                continue

        # Load model
        model = build_model(
            architecture=arch_key,
            num_classes=NUM_CLASSES,
            pretrained=False,
            dropout_rate=DROPOUT_RATE,
            device=DEVICE,
            verbose=False,
        )
        try:
            state = torch.load(ckpt, map_location=DEVICE)
            model.load_state_dict(state)
            model.eval()
        except Exception as e:
            print(f"    [SKIP] Failed to load checkpoint: {e}")
            rows.append({"Model": display_name, "Status": f"load error: {e}"})
            continue

        # Model size
        total_params, _ = count_parameters(model)
        size_mb = get_model_size_mb(model)

        # Inference speed
        infer_ms = benchmark_inference_speed(model, DEVICE)

        # Evaluation metrics
        try:
            report = evaluate_model(
                checkpoint_path=ckpt,
                model_name=arch_key,
                splits_dir=splits_dir,
                img_dir=img_dir,
                output_dir=output_dir / arch_key,
                batch_size=16,
                device=DEVICE,
            )
            om = report["overall_metrics"]
            row = {
                "Model":           display_name,
                "Architecture":    arch_key,
                "Checkpoint":      str(ckpt),
                "Status":          "evaluated",
                "Accuracy":        om["accuracy"],
                "Precision":       om["macro_precision"],
                "Recall/Sensitivity": om["macro_recall_sensitivity"],
                "Specificity":     om["macro_specificity"],
                "F1":              om["macro_f1"],
                "ROC-AUC":        om["roc_auc_ovr_macro"],
                "PR-AUC":         om["pr_auc_macro"],
                "Params (M)":     round(total_params / 1e6, 2),
                "Size (MB)":      size_mb,
                "Inference (ms)": infer_ms,
            }
        except Exception as e:
            print(f"    [ERROR] Evaluation failed: {e}")
            row = {
                "Model": display_name,
                "Architecture": arch_key,
                "Status": f"eval error: {e}",
                "Params (M)": round(total_params / 1e6, 2),
                "Size (MB)": size_mb,
                "Inference (ms)": infer_ms,
            }

        rows.append(row)
        print(f"    Params: {total_params/1e6:.1f}M | Size: {size_mb}MB | Speed: {infer_ms}ms/img")

    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)
    save_json(comparison_df.to_dict(orient="records"), output_dir / "model_comparison.json")

    print(f"\n{'='*70}")
    print("  Comparison Table (evaluated models only):")
    print(f"{'='*70}")
    evaluated = comparison_df[comparison_df.get("Status", "evaluated") == "evaluated"]
    if len(evaluated):
        cols = ["Model", "Accuracy", "F1", "ROC-AUC", "Specificity", "Params (M)", "Size (MB)", "Inference (ms)"]
        disp_cols = [c for c in cols if c in evaluated.columns]
        print(evaluated[disp_cols].to_string(index=False))
    else:
        print("  No models evaluated yet. Train at least one checkpoint first.")

    print(f"\n  Full comparison saved: {output_dir / 'model_comparison.csv'}")
    print(f"\n  MODEL SELECTION GUIDANCE:")
    print("  - Do NOT select based on accuracy alone.")
    print("  - Prioritize: Macro Recall/Sensitivity (critical for screening)")
    print("  - Also consider: F1, ROC-AUC, model size, inference speed")
    print("  - EfficientNet-B0 is recommended as primary model (accuracy/size balance).")
    print("  - ResNet-50 is recommended as the research baseline.")
    print("  - MobileNet-V3 for lightweight deployment experiments only.")
    print(f"{'='*70}\n")

    return comparison_df


if __name__ == "__main__":
    compare_all_models()
