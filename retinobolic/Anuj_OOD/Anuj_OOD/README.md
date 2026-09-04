# Module 3 — Out-of-Distribution (OOD) Detection
### Explainable AI for Diabetic Retinopathy Screening | SIH26038

---

## What OOD Detection Does

The OOD Detection Module answers a single reliability question:

> **"Does this input image look sufficiently similar to the fundus images the DR model was trained on — or is it unusually different?"**

It produces an **OOD Score** and an **OOD Status** that the downstream Reliability Engine can use alongside Confidence and Uncertainty to make a final safety decision.

---

## Why It Matters

A deep learning model can sometimes produce **high-confidence predictions on unfamiliar inputs**. Without an OOD check, this creates a silent failure risk:

```
Unusual or corrupted fundus image
    ↓
DR model assigns e.g. Grade 2, confidence 90%
    ↓
No warning issued
    ↓
Clinician trusts incorrect prediction
```

OOD detection provides an additional signal that can catch these cases:

```
Unusual input
    ↓
OOD = review_required
    ↓
Reliability Engine flags for human review
```

---

## What OOD Does NOT Mean

> [!IMPORTANT]
> OOD = **"This input appears outside the reference distribution."**
>
> OOD does **NOT** mean:
> - The image is medically abnormal
> - The patient does not have Diabetic Retinopathy
> - The DR model's prediction is definitely wrong
> - The image is fraudulent or fake
>
> It only means: **The input's feature representation is substantially distant from the reference distribution used by the screening model.**

---

## Pipeline Position

```
Fundus Image
     ↓
Anuj — Image Quality Gate
     ↓
Enhancement / Re-check
     ↓
Vinayak — 4-Class DR Model
     ↓
Vinayak — Grad-CAM / XAI
     ↓
Anuj — Confidence
     ↓
Anuj — Uncertainty
     ↓
Anuj — OOD  ← this module
     ↓
Clinical Context
     ↓
Final Reliability / Decision Layer
```

---

## OOD vs Uncertainty

| | Uncertainty | OOD |
|---|---|---|
| **Question** | How ambiguous is the prediction? | Does this input look unlike training data? |
| **Example** | Good fundus image, model unsure between grades | Corrupted/wrong image, model confidently wrong |
| **Signal** | Model's output distribution | Distance from reference embedding distribution |

These are **separate signals** and must not be combined inside this module.

---

## Architecture

```
Input Image
     ↓
Input Validation (format, existence, corruption check)
     ↓
Embedding Extractor (3-tier modular)
  ├── Tier 1: Vinayak DR Model hook (when available)
  ├── Tier 2: Pretrained PyTorch encoder (MobileNet-V2)
  └── Tier 3: Classical feature extractor (active; zero deep-learning deps)
     ↓
Reference Distribution (pre-computed, loaded once)
  mean μ, precision matrix Σ⁻¹, percentile calibration
     ↓
Distance Calculation
  Mahalanobis (default) | Cosine | Euclidean | Nearest-Centroid
     ↓
Threshold Comparison
     ↓
OOD Result (JSON)
```

---

## Project Structure

```
anuj-ood/
├── src/
│   ├── __init__.py       — public API surface
│   ├── config.py         — all tunable parameters
│   ├── utils.py          — image loading, validation, exceptions
│   ├── embedding.py      — modular 3-tier embedding extractor
│   ├── reference.py      — reference distribution (fit / save / load)
│   ├── distance.py       — distance metrics
│   └── ood.py            — OODDetector class + detect_ood() API
├── tests/
│   ├── test_embedding.py  — embedding extraction tests
│   ├── test_distance.py   — distance metric tests
│   ├── test_validation.py — input validation + edge case tests
│   └── test_ood.py        — end-to-end OOD pipeline tests
├── sample_data/
│   ├── in_distribution/   — synthetic fundus-like reference images
│   ├── out_of_distribution/ — synthetic OOD test images
│   └── generate_samples.py  — synthetic data generator
├── reference/
│   ├── reference_statistics.json  — pre-computed distribution stats
│   └── reference_embeddings.npy   — reference distance array
├── outputs/               — (reserved for batch output CSVs)
├── build_reference.py     — reference distribution builder CLI
├── demo.py                — interactive demo CLI
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate synthetic test images (first time only)

```bash
python sample_data/generate_samples.py
```

### 3. Build the reference distribution

```bash
python build_reference.py \
    --data_dir  sample_data/in_distribution \
    --output_dir reference \
    --extractor  classical
```

### 4. Run the demo

**In-distribution image:**
```bash
python demo.py --image sample_data/in_distribution/fundus_01.png
```
```
OOD Score   : 3.123038
Threshold   : 3.171500
Status      : IN-DISTRIBUTION
Review Req. : NO
```

**OOD image:**
```bash
python demo.py --image sample_data/out_of_distribution/blank_black.png
```
```
OOD Score   : 341.710439
Threshold   : 3.171500
Status      : REVIEW REQUIRED
Review Req. : YES
```

**Batch mode (all images):**
```bash
python demo.py --batch sample_data/in_distribution sample_data/out_of_distribution
```

**JSON output:**
```bash
python demo.py --image sample_data/in_distribution/fundus_01.png --json
```

### 5. Run tests

```bash
pytest -v tests/
```

---

## Public API

### `detect_ood(image_path, ...)`

```python
from src import detect_ood

result = detect_ood("path/to/fundus.jpg")
```

**Returns:**
```json
{
    "ood": false,
    "ood_status": "in_distribution",
    "ood_score": 3.123038,
    "threshold": 3.1715,
    "distance_metric": "mahalanobis",
    "extractor_type": "classical",
    "reason": "Input embedding is within the configured reference distribution.",
    "metadata": {}
}
```

OOD example:
```json
{
    "ood": true,
    "ood_status": "review_required",
    "ood_score": 341.71,
    "threshold": 3.1715,
    "distance_metric": "mahalanobis",
    "extractor_type": "classical",
    "reason": "Input embedding is substantially distant from the reference fundus distribution.",
    "metadata": {}
}
```

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `image_path` | required | Image path (str/Path) or uint8 numpy array |
| `reference_path` | config default | Path to `reference_statistics.json` |
| `threshold` | `config.OOD_THRESHOLD` | Decision threshold |
| `metric` | `config.DISTANCE_METRIC` | `"mahalanobis"` / `"cosine"` / `"euclidean"` |
| `extractor_type` | `config.EXTRACTOR_TYPE` | `"classical"` / `"pretrained"` / `"dr_model"` |
| `metadata` | `{}` | Pass-through camera / domain metadata |

### `OODDetector` class

For batch processing (avoids reloading reference on each call):

```python
from src.ood import OODDetector

detector = OODDetector(threshold=3.1715, metric="mahalanobis")

for image_path in image_list:
    result = detector.detect(image_path)
    print(result["ood_status"])
```

---

## OOD Score Semantics

> [!IMPORTANT]
> **The OOD score is a distance metric, NOT a probability.**
> Do NOT label it "OOD Probability" without proper calibration.

| Score range | Interpretation |
|---|---|
| `score << threshold` | Clearly in-distribution |
| `score ≈ threshold` | Borderline — treat with additional caution |
| `score >> threshold` | Clearly OOD — REVIEW REQUIRED |

For the classical Mahalanobis-based extractor:
- In-distribution scores (synthetic fundus): `2.6 – 3.17`
- OOD scores (blank, noise, checkerboard, etc.): `245 – 1134`

The gap is very clear for strongly OOD inputs.

---

## Embedding Methods

### Tier 3 — Classical Feature Extractor (currently active)

Zero deep-learning dependencies. Produces a **71-dimensional** feature vector:

| Feature Group | Dimensions |
|---|---|
| Global RGB colour moments (mean, std, skew) | 9 |
| HSV colour moments (mean, std) | 6 |
| Brightness / contrast statistics | 4 |
| Radial profile (8 rings × 3 channels) | 24 |
| Gradient statistics (Sobel magnitude) | 3 |
| High-frequency energy (Laplacian variance) | 1 |
| Colour histograms (8 bins × 3 channels) | 24 |
| **Total** | **71** |

### Tier 2 — Pretrained PyTorch Encoder

Requires `torch` and `torchvision`. Uses MobileNet-V2 (1280-dim) or ResNet-18 (512-dim) global average pooled features. Activate by setting `EXTRACTOR_TYPE = "pretrained"` in `src/config.py`.

### Tier 1 — Vinayak DR Model Hook

Integration point for Vinayak's trained DR model. Set `DR_MODEL_CHECKPOINT` in `src/config.py` once the model checkpoint is available.

---

## Threshold Calibration

> [!WARNING]
> `OOD_THRESHOLD = 3.1715` is a **PROTOTYPE VALUE** calibrated on the p99 of synthetic reference images.
> It MUST be re-calibrated on real fundus images before clinical deployment.

**Calibration procedure:**
1. Collect a representative set of real in-distribution fundus images.
2. Collect a set of known OOD examples (corrupted scans, wrong modalities, etc.).
3. Run `build_reference.py` on the in-distribution set.
4. Inspect the suggested percentile thresholds.
5. Choose a threshold that correctly classifies your validation OOD set with acceptable false-positive rate.

---

## Configuration (`src/config.py`)

| Key | Default | Description |
|---|---|---|
| `EXTRACTOR_TYPE` | `"classical"` | Active embedding extractor |
| `DISTANCE_METRIC` | `"mahalanobis"` | Active distance metric |
| `OOD_THRESHOLD` | `3.1715` | Decision threshold (prototype) |
| `AUTO_THRESHOLD_PERCENTILE` | `99.0` | Percentile for auto-threshold suggestion |
| `IMAGE_SIZE` | `(224, 224)` | Resize target |
| `COVARIANCE_REGULARISATION` | `1e-5` | Diagonal regularisation strength |
| `REFERENCE_STATS_FILE` | `reference/reference_statistics.json` | Reference statistics path |

---

## Integration with Vinayak's DR Model

Vinayak's output contract:
```json
{
    "grade": 2,
    "probabilities": {"0": 0.03, "1": 0.08, "2": 0.81, "3": 0.08},
    "gradcam_path": "..."
}
```

This module's output:
```json
{
    "ood": false,
    "ood_status": "in_distribution",
    "ood_score": 3.12,
    "threshold": 3.1715,
    ...
}
```

These two outputs remain **completely independent** and are combined only in the final Reliability Engine.

**Future DR model embedding integration:**
1. Set `config.DR_MODEL_CHECKPOINT` to the checkpoint path.
2. Set `config.EXTRACTOR_TYPE = "dr_model"`.
3. Update `DRModelExtractorHook` in `src/embedding.py` with Vinayak's model class.
4. Re-build the reference: `python build_reference.py --extractor dr_model`.

---

## Synthetic Test Data

> [!NOTE]
> The images in `sample_data/` are **SYNTHETIC** and intended only for software testing and threshold exploration.
> They are NOT real fundus images and do NOT represent clinical OOD validation.

| Category | Images | Type |
|---|---|---|
| In-distribution | `fundus_01.png` – `fundus_12.png` | Synthetic fundus patterns (disc, vessels, macula) |
| OOD — blank | `blank_black.png`, `blank_white.png` | Featureless images |
| OOD — noise | `random_noise.png`, `heavy_noise.png` | Random pixel noise |
| OOD — structural | `checkerboard.png`, `gradient.png` | Non-fundus patterns |
| OOD — domain shift | `color_inverted.png`, `solid_red.png` | Colour domain extremes |
| OOD — degraded | `severely_underexposed.png`, `extreme_crop.png` | Severe acquisition artefacts |

---

## Test Suite

```bash
pytest -v tests/
# 79 passed
```

| Test file | Coverage |
|---|---|
| `test_embedding.py` | Shape, dtype, finiteness, determinism, from-path/array API, caching |
| `test_distance.py` | All 4 metrics, dispatcher, singular covariance stability |
| `test_validation.py` | Image loading, NaN/Inf/empty/multi-dim, roundtrip serialisation |
| `test_ood.py` | JSON schema, Test A (ID), Test B (OOD), Test C (thresholding), errors |

---

## Error Handling

| Condition | Error raised |
|---|---|
| Missing image file | `FileNotFoundError` |
| Corrupt or undecodable image | `ImageLoadError` |
| Unsupported file extension | `ImageLoadError` |
| NaN / Inf in embedding | `EmbeddingError` |
| Missing reference file | `ReferenceError` |
| Embedding dimension mismatch | `DimensionMismatchError` |
| Unknown extractor / metric | `EmbeddingError` |

---

## Dependencies

```
numpy>=1.24
scipy>=1.10
Pillow>=9.0
scikit-learn>=1.2
pytest>=7.0
```

Optional (for Tier 2 pretrained extractor):
```
torch
torchvision
```

---

## Team Context

| Owner | Module |
|---|---|
| Anuj | Image Quality Gate, Enhancement, **OOD (this)**, Confidence, Uncertainty, Clinical Context, Reliability Engine |
| Vinayak | DR Dataset, DR Model Training, Evaluation, Inference, Grad-CAM |
