# Module 2: DR Prediction Uncertainty Engine

**SIH26038 — Explainable AI for Diabetic Retinopathy Screening in Rural India**

**Owner:** Anuj  
**Pipeline Position:** After Confidence Engine → before OOD / Reliability

---

## What This Module Does

This module answers a single question:

> **"How ambiguous is the DR model's probability distribution?"**

It does **NOT** predict DR. It does **NOT** assess image quality.
It takes the output probability distribution from Vinayak's 4-class DR model and
quantifies how **spread out** or **concentrated** that distribution is.

---

## Core Concept: Shannon Entropy

The uncertainty score is based on **Shannon entropy**:

```
H(P) = -∑ pᵢ · log(pᵢ)
```

### Why Entropy?

**Example A — Clear prediction (low uncertainty)**

```
[0.02, 0.03, 0.92, 0.03]
```

One class dominates. The distribution is **concentrated**.
Entropy is low → the model is not ambiguous.

**Example B — Ambiguous prediction (high uncertainty)**

```
[0.25, 0.25, 0.25, 0.25]
```

All classes are equally likely. The distribution is **spread out**.
Entropy is at its maximum → the model is maximally ambiguous.

### Normalization

Entropy is normalized by the maximum possible entropy for 4 classes (ln 4):

```
Model Uncertainty = H(P) / ln(4)
```

This gives a score in **[0, 1]**:
- `0.0` → completely certain (one class has all probability)
- `1.0` → completely uncertain (all classes equally likely)

---

## Confidence vs Uncertainty

These two signals are **related but not identical**.

| Signal | Formula | Meaning |
|--------|---------|---------|
| **Confidence** (Module 1) | `max(probabilities)` | How dominant is the top class? |
| **Uncertainty** (Module 2) | `H(P) / ln(4)` | How spread is the full distribution? |

Examples:

| Probabilities | Confidence | Uncertainty | Interpretation |
|---------------|------------|-------------|----------------|
| [0.02, 0.03, 0.92, 0.03] | 0.92 (high) | ~0.31 (low) | Strong clear prediction |
| [0.25, 0.25, 0.25, 0.25] | 0.25 (low) | 1.00 (high) | Ambiguous prediction |
| [0.40, 0.38, 0.12, 0.10] | 0.40 (medium) | ~0.80 (high) | Top class slight edge but very spread |

**Neither Confidence nor Uncertainty equates to clinical truth.**

---

## Input Contract

The module consumes output from Vinayak's DR model:

```json
{
  "grade": 2,
  "probabilities": {
    "0": 0.03,
    "1": 0.08,
    "2": 0.81,
    "3": 0.08
  },
  "gradcam_path": "outputs/gradcam/image_001.jpg"
}
```

Only `probabilities` is required. `grade` and `gradcam_path` are optional.

### Class Labels

| Class | DR Grade |
|-------|----------|
| 0 | No DR |
| 1 | Mild DR |
| 2 | Moderate DR |
| 3 | Severe / Proliferative DR |

---

## Output Contract

```json
{
  "predicted_grade": 2,
  "uncertainty": 0.3124,
  "uncertainty_level": "low",
  "review_recommended": false,
  "probability_margin": 0.73
}
```

| Field | Type | Description |
|-------|------|-------------|
| `predicted_grade` | int | DR grade inferred from argmax of probabilities |
| `uncertainty` | float [0,1] | Normalized Shannon entropy (canonical score) |
| `uncertainty_level` | str | `"low"` / `"medium"` / `"high"` (configurable thresholds) |
| `review_recommended` | bool | True when uncertainty ≥ HIGH_UNCERTAINTY_MIN |
| `probability_margin` | float | Top probability − 2nd-highest probability |

Optional field (if confidence is passed from Module 1):

| Field | Type | Description |
|-------|------|-------------|
| `confidence` | float | Forwarded from Module 1 for later combination |

---

## Uncertainty Levels & Thresholds

```
uncertainty ≤ 0.35           → "low"    review_recommended = False
0.35 < uncertainty < 0.70    → "medium" review_recommended = False
uncertainty ≥ 0.70           → "high"   review_recommended = True
```

> ⚠️ **IMPORTANT — PROTOTYPE THRESHOLDS**
>
> These thresholds are **prototype engineering values** chosen for development
> convenience. They have **NOT** been calibrated on a clinically representative
> dataset. They **MUST** be validated by domain experts and empirical calibration
> studies before use in any clinical or screening setting.
>
> Thresholds are fully configurable via `UncertaintyConfig`.

When `review_recommended = True`, the message is:

> *"Prediction is uncertain; additional review is recommended."*

This is a **downstream safety signal only**. It does NOT mean the model is wrong.

---

## Tie-Breaking Behavior

When multiple classes share the maximum probability (e.g. the uniform distribution
`[0.25, 0.25, 0.25, 0.25]`), the predicted grade is the **first index** returned
by `numpy.argmax`. This is deterministic and documented. Grade 0 wins ties.

---

## Project Structure

```
anuj-uncertainty/
├── src/
│   ├── __init__.py          # Public API exports
│   ├── config.py            # UncertaintyConfig — all tunable parameters
│   ├── validation.py        # Strict probability validation & custom exceptions
│   ├── uncertainty.py       # Core entropy math + public calculate_uncertainty()
│   └── calibration.py       # Optional: TemperatureScaler, ECE, reliability diagram
│
├── tests/
│   ├── test_validation.py   # Input validation tests
│   ├── test_uncertainty.py  # Unit + numerical tests for entropy & API
│   └── test_calibration.py  # Optional calibration and ECE tests
│
├── sample_data/
│   └── sample_predictions.json   # Example model outputs for testing
│
├── outputs/                 # Reserved for future output artifacts
├── requirements.txt
├── demo.py                  # CLI demonstration
└── README.md
```

---

## Quick Start

```python
from src import calculate_uncertainty

dr_result = {
    "grade": 2,
    "probabilities": {"0": 0.03, "1": 0.08, "2": 0.81, "3": 0.08}
}

result = calculate_uncertainty(dr_result)
print(result)
# {
#     "predicted_grade": 2,
#     "uncertainty": 0.3124,
#     "uncertainty_level": "low",
#     "review_recommended": False,
#     "probability_margin": 0.73
# }
```

### With Custom Thresholds

```python
from src import calculate_uncertainty, UncertaintyConfig

config = UncertaintyConfig(
    LOW_UNCERTAINTY_MAX=0.30,
    HIGH_UNCERTAINTY_MIN=0.60,
)
result = calculate_uncertainty(dr_result, config=config)
```

### With Confidence from Module 1

```python
result = calculate_uncertainty(dr_result, confidence=0.81)
# result will include "confidence": 0.81 for downstream combination
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

Expected: **all tests pass**.

Test coverage includes:
- All 7 required test cases from spec
- Exact numerical entropy calculations (pytest.approx)
- Configurable threshold behavior
- All invalid input types (NaN, Inf, negative, wrong count, bad sum)
- Output contract validation
- Optional temperature scaling and ECE

---

## Running the Demo

```bash
python demo.py
```

---

## Optional: Temperature Scaling

For overconfident neural network outputs, optional post-hoc calibration is available:

```python
from src.calibration import TemperatureScaler

# Fit offline on calibration dataset (never during inference!)
scaler = TemperatureScaler()
scaler.fit(val_logits, val_labels)

# Apply to raw logits before uncertainty computation
calibrated_probs = scaler.scale_logits(raw_logits)
probs_dict = {str(i): float(p) for i, p in enumerate(calibrated_probs)}
result = calculate_uncertainty({"probabilities": probs_dict})
```

## Optional: Expected Calibration Error (ECE)

For offline model evaluation only — NOT for real-time use:

```python
from src.calibration import compute_ece

ece = compute_ece(confidences, predictions, labels, n_bins=10)
print(f"ECE: {ece:.4f}")   # lower is better; 0 = perfect calibration
```

---

## Future Integration

This module is designed to be combined with:

```
Confidence (Module 1)    — max(probabilities)
Uncertainty (Module 2)   — normalized entropy      [THIS MODULE]
OOD Score (Module 3)     — is the image in-distribution?
Camera Reliability
Clinical Context
        ↓
Reliability / Decision Layer
```

Output fields can be extended without breaking existing consumers.

---

## Dependencies

- `numpy` — all core math
- `pytest` — test runner

No PyTorch, TensorFlow, OpenCV, LangChain, or LLM APIs are used or required.

---

## Scope Boundaries

This module does **NOT**:
- Predict DR grade
- Train or modify Vinayak's model
- Assess image quality
- Perform OOD detection
- Implement referral logic or clinical decisions
- Connect to any frontend, backend, or database
- Use any LLM or generative AI
