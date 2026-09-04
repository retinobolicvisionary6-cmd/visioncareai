# Model Confidence Module

**SIH Problem ID:** SIH26038 — Explainable AI for Diabetic Retinopathy Screening in Rural India
**Module Owner:** Anuj
**Step:** 3.1 — Model Confidence

---

## What This Module Does

The Model Confidence Module calculates a **probability-based confidence signal** from the Diabetic Retinopathy (DR) model's output probability distribution.

It answers exactly one question:

> **"How strong is the DR model's current probability-based confidence in its selected class?"**

### Confidence Definition

```
confidence = max(class probabilities) = argmax(p_i)
```

Example:

| Class | Grade      | Probability |
|-------|------------|-------------|
| 0     | No DR      | 0.03        |
| 1     | Mild DR    | 0.08        |
| **2** | **Moderate DR** | **0.81** |
| 3     | Severe/PDR | 0.08        |

```
confidence       = 0.81
confidence_level = "high"
predicted_grade  = 2
```

---

## ⚠ What This Module Does NOT Do

| ❌ Out of Scope                       | ✅ Module scope only              |
|--------------------------------------|-----------------------------------|
| Diagnose Diabetic Retinopathy        | Compute confidence from probs     |
| Estimate clinical certainty          | Validate probability distribution |
| Detect out-of-distribution samples   | Return confidence level           |
| Quantify epistemic uncertainty       | Provide top-2 margin              |
| Determine referral priority          |                                   |
| Replace or advise a physician        |                                   |
| Process image data                   |                                   |
| Interpret Grad-CAM results           |                                   |

---

## ⚠ Critical Distinction

```
Model Confidence  ≠  Model Accuracy  ≠  Clinical Certainty
```

A model can output **Confidence = 0.95** and still produce a **wrong prediction**.

The confidence score is a **model-output signal** only. It will later be combined with:

```
Confidence
+
Uncertainty (entropy / MC-Dropout)
+
OOD Detection
+
Camera Reliability
→ Reliability Engine → Clinical Context → Decision Layer
```

At this stage, the module outputs confidence only — no referral, no clinical decision.

---

## Pipeline Position

```
Fundus Image
      ↓
Anuj — Image Quality Gate
      ↓
Vinayak — DR Model
      ↓
DR Grade + Class Probabilities
      ↓
Vinayak — Grad-CAM / XAI
      ↓
► Anuj — Confidence Module  ◄  (this module)
      ↓
Anuj — Uncertainty
      ↓
Anuj — OOD / Camera Reliability
      ↓
Anuj — Clinical Context
      ↓
Final Decision Layer
```

---

## Project Structure

```
anuj-confidence/
├── src/
│   ├── __init__.py              # Public package exports
│   ├── confidence.py            # Core validation & confidence engine
│   └── config.py                # Class mapping, thresholds, tolerances
├── tests/
│   ├── test_confidence.py       # Confidence calculation tests
│   └── test_validation.py       # Input validation tests
├── sample_data/
│   └── sample_predictions.json  # Synthetic example distributions
├── outputs/                     # Reserved for future output artefacts
├── requirements.txt
├── demo.py
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `pytest` only. No deep learning frameworks required.

---

## Quick Start

```python
from src.confidence import calculate_confidence

dr_result = {
    "grade": 2,
    "probabilities": {
        "0": 0.03,
        "1": 0.08,
        "2": 0.81,
        "3": 0.08
    }
}

result = calculate_confidence(dr_result)
print(result)
```

Output:

```json
{
  "predicted_grade": 2,
  "predicted_class_name": "Moderate DR",
  "confidence": 0.81,
  "confidence_percent": 81.0,
  "confidence_level": "high",
  "top_class": 2,
  "top_probability": 0.81,
  "second_class": 1,
  "second_probability": 0.08,
  "margin": 0.73
}
```

### Disable top-2 fields

```python
result = calculate_confidence(dr_result, include_top2=False)
```

---

## API Reference

### `calculate_confidence(dr_result, include_top2=True) → dict`

**Input:**

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `probabilities` | dict or list | ✅ Yes | 4-class probability distribution |
| `grade` | int | Optional | Accepted but not used by this module |
| `gradcam_path` | str | Optional | Accepted but not used by this module |

**Probabilities format:** dict with string keys `"0"`–`"3"`, int keys `0`–`3`, or a list/array of 4 values. Sum must equal 1.0 (within floating-point tolerance).

**Output fields:**

| Field | Type | Description |
|-------|------|-------------|
| `predicted_grade` | int | Class with highest probability (argmax) |
| `predicted_class_name` | str | Human-readable class name |
| `confidence` | float | Maximum class probability ∈ [0, 1] — **canonical value** |
| `confidence_percent` | float | `confidence × 100` — display only |
| `confidence_level` | str | `"high"` / `"medium"` / `"low"` |
| `top_class` | int | Top-1 class index (= `predicted_grade`) |
| `top_probability` | float | Top-1 probability |
| `second_class` | int | Top-2 class index |
| `second_probability` | float | Top-2 probability |
| `margin` | float | `top_probability − second_probability` |

**Exceptions:**

| Exception | Raised when |
|-----------|-------------|
| `InvalidInputFormatError` | Missing key, wrong type, wrong class count |
| `InvalidProbabilityError` | NaN, Infinity, negative, > 1.0, bad sum |

---

## Validation Rules

| Rule | Detail |
|------|--------|
| Exactly 4 classes | Classes 0, 1, 2, 3 — all required |
| Values must be finite | NaN and ±Infinity rejected |
| Values must be in [0, 1] | Negatives and values > 1.0 rejected |
| Sum ≈ 1.0 | Tolerance: `PROBABILITY_SUM_TOLERANCE = 1e-3` (configurable) |
| Invalid distributions are NOT silently normalised | Clear error raised |

---

## Configuration

All thresholds are in [`src/config.py`](src/config.py):

```python
# Class mapping
CLASS_MAPPING = {0: "No DR", 1: "Mild DR", 2: "Moderate DR", 3: "Severe/PDR"}
NUM_CLASSES = 4

# Validation
PROBABILITY_SUM_TOLERANCE = 1e-3   # allowable sum deviation

# Confidence levels — PROTOTYPE ENGINEERING THRESHOLDS ONLY
# These values have NOT been clinically validated.
HIGH_CONFIDENCE_THRESHOLD   = 0.80
MEDIUM_CONFIDENCE_THRESHOLD = 0.50
```

> ⚠ **Threshold disclaimer:** `HIGH_CONFIDENCE_THRESHOLD` and `MEDIUM_CONFIDENCE_THRESHOLD`
> are **prototype engineering thresholds only**. They require independent clinical validation
> before use in any medical decision-making context.

---

## Running Tests

```bash
pytest -v tests/
```

Test coverage includes:
- All 7 specification test cases
- Valid input formats (dict string keys, dict int keys, list, numpy array)
- Invalid structures (missing classes, extra classes, wrong types, empty input)
- Invalid values (NaN, ±Infinity, negative, > 1.0, bad sum)
- Class name mapping
- Non-mutation of input
- Top-2 inclusion and exclusion
- `gradcam_path` ignorance

---

## Demo

```bash
python demo.py
```

---

## Integration with Vinayak's DR Model

Vinayak will provide outputs in the format:

```json
{
  "grade": 2,
  "probabilities": {"0": 0.03, "1": 0.08, "2": 0.81, "3": 0.08},
  "gradcam_path": "outputs/gradcam/image_001.jpg"
}
```

This module consumes **only** `probabilities` and does not modify Vinayak's object.

```python
from src.confidence import calculate_confidence

# dr_result from Vinayak's module
confidence_result = calculate_confidence(vinayak_dr_result)
```

---

## Future Integration (Planned)

The output of this module is designed to extend cleanly:

```json
{
  "confidence": 0.81,
  "uncertainty": 0.21,
  "ood": false,
  "reliability_status": "acceptable"
}
```

Currently, only confidence-related fields are returned.

---

## Acceptance Checklist

- [x] 4-class probabilities accepted
- [x] Input validation works
- [x] Predicted grade calculated
- [x] Confidence calculated
- [x] Confidence stored 0–1
- [x] Percentage conversion available
- [x] Confidence level available
- [x] Thresholds configurable
- [x] Invalid distributions handled safely
- [x] Optional top-2 margin works
- [x] Unit tests pass
- [x] Demo works
- [x] README complete
- [x] No DR model code
- [x] No uncertainty code
- [x] No OOD code
- [x] No agent code
- [x] No frontend/backend dependency
- [x] Integration example documented

---

*Module Owner: Anuj — SIH26038*
