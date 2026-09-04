# VISIONARY6 — VINAYAK AI/ML Module
# Diabetic Retinopathy Screening (4-Class) + Grad-CAM Explainability

> **Scope**: This repository contains ONLY the AI/ML module for the Visionary6 SIH project.
> It is the responsibility of **Vinayak** and is consumed by **Anuj's** reliability module via a fixed JSON contract.

---

## ⚠️ Important Disclaimer

This module provides **AI-assisted DR screening** and is a **research prototype**.

It does NOT:
- Provide autonomous medical diagnosis
- Replace ophthalmologists or qualified medical professionals
- Constitute a clinically validated tool
- Guarantee medical accuracy

All outputs must be interpreted as **model predictions** and **model evidence visualizations**, reviewed by qualified medical professionals.

---

## 🎯 Module Scope

### Included (Vinayak):
- Dataset management, cleaning, and reproducible splits
- Deterministic preprocessing pipeline
- 4-class DR classification (EfficientNet / ResNet / MobileNet)
- Transfer learning and fine-tuning
- Evaluation: Accuracy, Precision, Recall/Sensitivity, Specificity, F1, ROC-AUC, PR-AUC
- Model comparison framework
- Inference pipeline with fixed JSON contract
- Grad-CAM attention/evidence visualization

### Not Included (handled by other team members):
- Frontend (Healthcare Worker / Doctor UI)
- FastAPI backend / Database / Auth
- Image Quality Gate
- Image Enhancement
- Uncertainty estimation
- Out-of-Distribution (OOD) detection
- Decision Agent / Referral workflow (Anuj)

---

## 📋 4-Class DR Classification Schema

| Grade | Class Name     | APTOS Raw Labels | Clinical Description                                      |
|:-----:|:---------------|:----------------:|:----------------------------------------------------------|
|   0   | No DR          | 0                | Normal retina, no observable microaneurysms               |
|   1   | Mild DR        | 1                | Microaneurysms only                                       |
|   2   | Moderate DR    | 2                | Microaneurysms, haemorrhages, hard exudates               |
|   3   | Severe / PDR   | 3, 4             | Severe NPDR or Proliferative DR (merged from raw 3 and 4) |

---

## 📁 Repository Structure

```
retinobolic/
├── configs/
│   └── config.py               ← Central configuration (ALL parameters here)
│
├── data/
│   ├── raw/                    ← Place APTOS dataset here
│   │   ├── train.csv
│   │   └── train_images/
│   ├── processed/
│   ├── splits/                 ← Auto-generated: train/val/test.csv
│   └── metadata/
│       ├── aptos2019_metadata.json
│       └── dataset_report.json
│
├── models/
│   ├── checkpoints/            ← best_model.pth, last_model.pth
│   ├── final/
│   ├── class_mapping.json
│   └── training_config.json
│
├── outputs/
│   ├── metrics/                ← evaluation_report.json, training_curves.png
│   ├── confusion_matrix/       ← confusion_matrix.png
│   ├── gradcam/                ← *_original.jpg, *_heatmap.jpg, *_overlay.jpg
│   └── predictions/            ← test_predictions.csv
│
├── src/
│   ├── config.py               ← (imports configs/config.py)
│   ├── utils.py                ← Shared utilities
│   ├── preprocess.py           ← Crop, resize, normalize, augment
│   ├── dataset.py              ← Cleaning, splitting, PyTorch Dataset
│   ├── model.py                ← Model factory (EfficientNet/ResNet/MobileNet)
│   ├── train.py                ← Training pipeline
│   ├── evaluate.py             ← Full evaluation suite
│   ├── inference.py            ← predict() — fixed JSON contract
│   ├── gradcam.py              ← Grad-CAM heatmap generator
│   └── compare_models.py       ← Model comparison framework
│
├── tests/
│   └── test_inference.py       ← 15 automated contract tests
│
├── experiments/                ← Timestamped experiment logs
├── notebooks/
├── data/prepare_dataset.py     ← Dataset prep entry point
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚡ Installation

```bash
pip install -r requirements.txt
```

**Hardware**: GPU (CUDA) recommended. CPU fallback is automatic.  
The module detects hardware at startup and prints device info.

---

## 🗂️ Dataset Setup (APTOS 2019)

1. Create a Kaggle account: https://www.kaggle.com/
2. Go to: https://www.kaggle.com/c/aptos2019-blindness-detection/data
3. Accept competition rules and download `train.zip` + `train.csv`
4. Extract to:
   ```
   data/raw/train.csv
   data/raw/train_images/
   ```
5. Run preparation:
   ```bash
   python data/prepare_dataset.py
   ```

This validates all images, removes corrupt files, maps 5-class → 4-class labels,
and creates reproducible 70/15/15 train/val/test splits.

**Note**: APTOS 2019 does not provide explicit patient IDs. Splits are stratified by class label.

---

## 🚀 Commands

### Step 1: Prepare dataset
```bash
python data/prepare_dataset.py
```

### Step 2: Train (EfficientNet-B0 default)
```bash
python src/train.py
```

### Train with custom settings
```bash
python src/train.py --model resnet50 --epochs 25 --batch_size 8 --lr 5e-5
```

Supported `--model` values: `efficientnet_b0`, `resnet50`, `mobilenet_v3_small`

### Step 3: Evaluate on test set
```bash
python src/evaluate.py
```

### Evaluate a specific checkpoint
```bash
python src/evaluate.py --checkpoint models/checkpoints/best_model.pth --model efficientnet_b0
```

### Step 4: Compare all trained models
```bash
python src/compare_models.py
```

### Step 5: Inference (single image)
```bash
python src/inference.py path/to/retina_image.jpg
```

### Step 6: Grad-CAM only
```bash
python src/gradcam.py path/to/retina_image.jpg
python src/gradcam.py path/to/retina_image.jpg --target_class 2
```

### Step 7: Run tests
```bash
python tests/test_inference.py
# or with pytest:
pytest tests/test_inference.py -v
```

---

## 🔒 Integration Output Contract (VINAYAK → ANUJ)

Calling `predict(image_path)` in `src/inference.py` returns:

```json
{
  "grade": 2,
  "probabilities": {
    "0": 0.03,
    "1": 0.08,
    "2": 0.81,
    "3": 0.08
  },
  "gradcam_path": "outputs/gradcam/image_001_overlay.jpg"
}
```

| Field          | Type    | Description                                           |
|:---------------|:--------|:------------------------------------------------------|
| `grade`        | int     | Predicted class index: 0=No DR, 1=Mild, 2=Moderate, 3=Severe/PDR |
| `probabilities`| dict    | Softmax distribution across all 4 classes (sum ≈ 1.0) |
| `gradcam_path` | string  | Path to Grad-CAM overlay image                        |

**ANUJ adds**: `quality`, `confidence`, `uncertainty`, `ood`, `action`, `priority`  
**VINAYAK does NOT include those fields.**

---

## 📊 Evaluation Metrics

The evaluation suite (`src/evaluate.py`) computes:

| Metric              | Level      | Notes                                     |
|:--------------------|:-----------|:------------------------------------------|
| Accuracy            | Overall    | Not used as sole selection criterion      |
| Macro Precision     | Overall    | Macro-averaged across 4 classes           |
| Macro Recall/Sensitivity | Overall | Critical for DR screening               |
| Macro Specificity   | Overall    | TN rate per class, macro-averaged         |
| Macro F1            | Overall    | Harmonic mean of precision and recall     |
| ROC-AUC (OvR)       | Overall    | One-vs-Rest macro-averaged                |
| PR-AUC              | Overall    | Average precision macro-averaged          |
| Per-class metrics   | Per class  | All above metrics per DR grade            |

---

## 🧠 Grad-CAM Interpretation

Grad-CAM outputs three images per prediction:

| Output file          | Description                            |
|:---------------------|:---------------------------------------|
| `*_original.jpg`     | Cropped fundus reference image         |
| `*_heatmap.jpg`      | Standalone colour heatmap              |
| `*_overlay.jpg`      | Heatmap superimposed on fundus image   |

> **Important**: Grad-CAM shows **model attention/evidence regions**.
> It does NOT detect exact lesions and does NOT prove causality.
> Highlighted regions indicate where the model focused — not a clinical diagnosis.

---

## ⚙️ Configuration

All parameters are in [`configs/config.py`](configs/config.py):

| Parameter              | Default         | Description                         |
|:-----------------------|:----------------|:------------------------------------|
| `MODEL_NAME`           | efficientnet_b0 | Architecture                        |
| `IMAGE_SIZE`           | (224, 224)      | Input image dimensions              |
| `BATCH_SIZE`           | 16              | Training batch size                 |
| `NUM_EPOCHS`           | 20              | Max training epochs                 |
| `LEARNING_RATE`        | 1e-4            | Initial learning rate               |
| `RANDOM_SEED`          | 42              | Global reproducibility seed         |
| `EARLY_STOPPING_PATIENCE` | 5           | Stop if no val improvement for N epochs |
| `USE_AMP`              | True (CUDA only)| Automatic mixed precision           |
| `AUGMENTATION_CONFIG`  | dict            | Per-augmentation toggle settings    |

---

## 🔁 Reproducibility

Every training run:
- Sets a global random seed (`RANDOM_SEED = 42` by default)
- Saves the full experiment config to `experiments/TIMESTAMP_model_seed/`
- Saves `training_config.json` with all hyperparameters
- Saves `class_mapping.json` for label contract documentation
- Saves both best and last checkpoints

To reproduce a run exactly:
```bash
python src/train.py --model efficientnet_b0 --seed 42 --epochs 20
```

---

## ⚠️ Model Limitations

1. **Research prototype only** — not clinically validated
2. **APTOS 2019 dataset** — acquired from specific Indian centres; may not generalise to all fundus cameras or populations
3. **Image quality** — model performance degrades on low-quality images; use an Image Quality Gate upstream
4. **Class imbalance** — No DR is the majority class; minority class metrics (Mild DR) may be less reliable
5. **No patient context** — model sees single images, no patient history or demographics
6. **Grad-CAM** — attention heatmaps are not validated diagnostic markers
7. **Threshold** — the argmax prediction threshold is not calibrated for any specific clinical use case

---

## 📚 Dataset Citation

> APTOS 2019 Blindness Detection, Kaggle Competition, 2019.
> https://www.kaggle.com/c/aptos2019-blindness-detection

No peer-reviewed paper exists for the APTOS 2019 public competition dataset.  
See `data/metadata/aptos2019_metadata.json` for full documentation.
