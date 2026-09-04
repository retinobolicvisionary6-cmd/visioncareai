# Reliability Engine (Module 4)

**SIH Problem ID:** SIH26038  
**Project:** Explainable AI for Diabetic Retinopathy Screening in Rural India

This repository contains **Module 4: Reliability Engine**, the orchestration layer that fuses signals from the preceding modules into a single, unified engineering reliability assessment.

## Pipeline Architecture

The final AI pipeline evaluates each fundus image as follows:
```text
Fundus Image → Image Quality Gate (D:\anuj-fundus-quality) → 4-Class DR Model → Grad-CAM / XAI

Then, the Reliability Engine orchestrates:
Confidence Module (D:\anuj-confidence)
     ↓
Uncertainty Module (D:\anuj-uncertainty)
     ↓
OOD Module (D:\anuj-ood)
     ↓
RELIABILITY ENGINE (D:\anuj-reliability)  ← THIS MODULE
```

The Reliability Engine safely manages module imports to avoid `sys.modules` namespace collisions across the sibling repositories using dynamic module swapping.

## Public Orchestration API

The single public entry point for integrating the reliability pipeline is `run_reliability_pipeline()`.

```python
from src.reliability.engine import run_reliability_pipeline

dr_result = {
    "grade": 2,
    "probabilities": {"0": 0.01, "1": 0.01, "2": 0.97, "3": 0.01}
}
image_path = "path/to/fundus.jpg"

result = run_reliability_pipeline(dr_result, image_path)

print(result["reliability_status"]) # "acceptable" | "caution" | "review_required"
print(result["review_required"])    # False
print(result["reason"])             # "High model confidence, low uncertainty and in-distribution input."
```

### Output Schema

The pipeline returns a standard, JSON-serialisable dictionary:

```json
{
  "reliability_status": "acceptable",
  "review_required": false,
  "reason": "High model confidence, low uncertainty and in-distribution input.",
  "confidence": 0.97,
  "confidence_level": "high",
  "uncertainty": 0.12,
  "uncertainty_level": "low",
  "ood": false,
  "ood_status": "in_distribution",
  "ood_score": 3.123,
  "reliability_score": 0.924
}
```

## Deterministic Rule Matrix & Priorities

The Engine answers:
> *"Considering confidence, uncertainty, and OOD together, should the model result be considered sufficiently reliable for downstream processing, or should it be flagged for additional review?"*

Rules are evaluated in this strict priority hierarchy:

1. **OOD == True** → `review_required` (HIGHEST PRIORITY)
2. **uncertainty_level == "high"** → `review_required`
3. **confidence_level == "low"** → `review_required`
4. **Intermediate Signals (medium)** → `caution`
5. **High conf + Low uncert + In-dist** → `acceptable`

### Conflicting Signals Handled Deterministically:
* **Case A (High Conf + Low Uncert + OOD=True)** → `review_required`. High confidence on unfamiliar input is unsafe (false assurance).
* **Case B (High Conf + High Uncert + OOD=False)** → `review_required`.
* **Case C (Low Conf + Low Uncert + OOD=False)** → `review_required` (or caution depending on config thresholds).
* **Case D (Medium Conf + Medium Uncert + OOD=False)** → `caution`.
* **Case E (High Conf + Low Uncert + OOD=False)** → `acceptable`.

## Non-Diagnostic Medical Safety Principles

⚠️ **IMPORTANT ENGINEERING DISCLAIMERS** ⚠️
* `confidence = 0.95` does **NOT** mean 95% medically correct.
* `ood = true` does **NOT** mean absence of disease.
* `review_required = true` is an engineering safety flag, **NOT** a diagnosis.
* This module provides an engineering classification of model reliability. It does **NOT** integrate clinical context (e.g., BP, HbA1c, age) or build the final diagnostic decision layer.
