# Multimodal Clinical Context Module
### SIH26038 — Explainable AI for Diabetic Retinopathy Screening in Rural India
### Step 5 — Anuj (Clinical Context Layer)

---

## Overview

This module collects, validates, normalises, and structures **patient-level clinical information** for downstream consumption by the Final Decision / Referral Priority layer of the DR screening pipeline.

**This module is:**
- A structured data-processing and validation layer.
- A clean interface between clinical data entry and AI-assisted DR screening.

**This module is NOT:**
- A DR prediction model.
- A clinical diagnosis tool.
- A risk-scoring engine.
- An ML model of any kind.

---

## Pipeline Position

```
Fundus Image
   |
   v
Image Quality Gate (Anuj)
   |
   v
DR Model (Vinayak)
   |
   v
Grad-CAM / XAI (Vinayak)
   |
   v
Confidence (Anuj)
   |
   v
Uncertainty (Anuj)
   |
   v
OOD (Anuj)
   |
   v
Reliability Engine (Anuj)
   |
   v
Clinical Context (Anuj) <-- THIS MODULE
   |
   v
Final Decision / Referral Priority
   |
   v
Doctor Validation
```

The Clinical Context module runs **in parallel with** (not inside) Vinayak's DR model. Clinical data does NOT enter the DR model.

---

## Clinical Fields Supported

| Field | Unit | Notes |
|-------|------|-------|
| Age | years | Integer or float accepted |
| Systolic BP | mmHg | Must be > diastolic |
| Diastolic BP | mmHg | Must be < systolic |
| HbA1c | % | Unit must be % (not IFCC mmol/mol) |
| Diabetes Duration | years | Must be <= age |
| Clinical History | structured | Optional flags (known_diabetes, previous_dr_history, notes) |

---

## Project Structure

```
anuj-clinical-context/
|
+-- src/
|   +-- __init__.py           # Public API exports
|   +-- config.py             # Configurable validation bounds and unit constants
|   +-- validation.py         # Pydantic v2 schemas and validators
|   +-- normalization.py      # Value formatting, precision, status tagging
|   +-- clinical_context.py   # Core processor and public process_clinical_context()
|
+-- tests/
|   +-- test_validation.py    # Validation unit tests
|   +-- test_normalization.py # Normalisation unit tests
|   +-- test_clinical_context.py # End-to-end integration tests
|
+-- sample_data/
|   +-- complete_patient.json
|   +-- partial_patient.json
|   +-- invalid_patient.json
|
+-- outputs/                  # Demo output JSON files
+-- requirements.txt
+-- demo.py
+-- README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

Requirements:
- `pydantic>=2.0.0`
- `pytest>=7.0.0`

No ML frameworks. No external APIs.

---

## Quick Start

```python
from src import process_clinical_context

result = process_clinical_context({
    "age": 58,
    "bp_systolic": 148,
    "bp_diastolic": 92,
    "hba1c": 8.2,
    "diabetes_duration_years": 10,
    "clinical_history": {"known_diabetes": True},
})

print(result["data_quality"]["complete"])      # True
print(result["clinical_context"]["age"])       # {"value": 58, "unit": "years", "status": "provided"}
```

---

## Public API

### `process_clinical_context(patient_data, config=DEFAULT_CONFIG) -> dict`

Main entry point. Returns structured output or a validation-error result.
Does NOT raise on invalid input — returns `validation_passed: false` instead.

### `process_clinical_context_strict(patient_data, config=DEFAULT_CONFIG) -> dict`

Strict variant. Raises `ValidationError` or `ValueError` on invalid input.

---

## Input Schema

```json
{
  "patient_id": "DEMO001",
  "age": 58,
  "bp_systolic": 148,
  "bp_diastolic": 92,
  "hba1c": 8.2,
  "diabetes_duration_years": 10,
  "clinical_history": {
    "known_diabetes": true,
    "previous_dr_history": false,
    "other_notes": ""
  }
}
```

All fields except `patient_id` are optional. Missing fields are tracked explicitly.

**Extra/unknown fields** are silently ignored (schema policy: `extra="ignore"`). This enables forward-compatible schema evolution without errors.

---

## Output Schema

### Successful result

```json
{
  "clinical_context": {
    "patient_id": "DEMO001",
    "age": {"value": 58, "unit": "years", "status": "provided"},
    "bp_systolic": {"value": 148, "unit": "mmHg", "status": "provided"},
    "bp_diastolic": {"value": 92, "unit": "mmHg", "status": "provided"},
    "hba1c": {"value": 8.2, "unit": "%", "status": "provided"},
    "diabetes_duration_years": {"value": 10.0, "unit": "years", "status": "provided"},
    "clinical_history": {"value": {"known_diabetes": true}, "status": "provided"}
  },
  "data_quality": {
    "complete": true,
    "missing_fields": [],
    "provided_fields": ["age", "bp_systolic", "bp_diastolic", "hba1c", "diabetes_duration_years", "clinical_history"],
    "clinical_context_complete": true,
    "flags": {
      "age_available": true,
      "bp_available": true,
      "hba1c_available": true,
      "diabetes_duration_available": true,
      "clinical_history_available": true
    }
  },
  "validation_passed": true,
  "validation_errors": []
}
```

### Partial result (missing fields)

Fields with status `"missing"` (key absent from input) or `"not_recorded"` (key present but null) are never imputed.

### Failed validation result

```json
{
  "clinical_context": null,
  "data_quality": {"complete": false, "missing_fields": [], ...},
  "validation_passed": false,
  "validation_errors": ["input: Value error, Clinical context validation failed:\n  * Age -5.0 is outside plausible range [0, 130] years."]
}
```

---

## Field Status Tags

| Status | Meaning |
|--------|---------|
| `provided` | Value was supplied and passed validation |
| `missing` | Key was absent from the input payload |
| `not_recorded` | Key was present but value was explicitly null |

---

## Validation Rules

### Age
- Numeric (int or float)
- Finite (not NaN or Inf)
- Range: [0, 130] years (configurable)

### Blood Pressure
- Numeric and finite
- `bp_systolic` range: [40, 300] mmHg (configurable)
- `bp_diastolic` range: [20, 200] mmHg (configurable)
- Cross-field: `bp_systolic` must be strictly greater than `bp_diastolic`

### HbA1c
- Numeric and finite
- Range: [2.0, 25.0] % (configurable)
- **Unit must be `%`** — IFCC mmol/mol values must be converted before entry

### Diabetes Duration
- Numeric, finite, and non-negative
- Range: [0, 100] years (configurable)
- Cross-field: must not exceed patient age

### Clinical History
- Structured optional fields
- Unknown sub-fields are silently ignored

> **IMPORTANT:** Validation bounds are plausible physiological ranges for data-quality checking only.
> They are NOT clinical diagnostic thresholds and carry no clinical interpretation.

---

## Configuring Bounds

```python
from src.config import ClinicalConfig
from src import process_clinical_context

custom_config = ClinicalConfig(
    age_min=18,
    age_max=100,
    hba1c_min=4.0,
    hba1c_max=20.0,
)

result = process_clinical_context(patient_data, config=custom_config)
```

---

## Running Tests

```bash
pytest tests/ -v
```

Expected: 91 tests pass.

### Test Coverage

| Test Class | What is tested |
|-----------|---------------|
| `TestCompleteValidInput` | Valid complete data round-trip |
| `TestAgeValidation` | Negative, too-large, string, None age |
| `TestBPValidation` | String BP, negative BP, diastolic > systolic |
| `TestHba1cValidation` | String, out-of-range, None HbA1c |
| `TestDiabetesDurationValidation` | Negative, exceeds age, zero |
| `TestExtraFields` | Extra keys silently ignored |
| `TestNonFiniteValues` | NaN and Inf rejected |
| `TestCustomConfig` | Custom bounds applied correctly |
| `TestNormalisedPrecision` | Rounding rules per field |
| `TestUnitConsistency` | Unit tags correct in output |
| `TestFieldStatus` | provided / missing / not_recorded |
| `TestDataQuality` | complete, missing_fields, flags |
| `TestCase1_CompleteData` | End-to-end complete case |
| `TestCase2_MissingHba1c` | Missing HbA1c not imputed |
| `TestCase3_InvalidAge` | Negative age → validation error |
| `TestCase4_InvalidBP` | String BP → validation error |
| `TestCase5_InvalidHba1c` | Non-numeric / out-of-range HbA1c |
| `TestCase6_InvalidDuration` | Negative / exceeds-age duration |
| `TestCase7_ExtraFields` | Extra fields don't appear in output |
| `TestOutputStructureInvariants` | No DR grade, no risk scores, JSON-safe |

---

## Running the Demo

```bash
python demo.py
```

Output includes four cases: complete data, partial data, invalid data, and empty data.
Results are saved to `outputs/` as JSON.

---

## Architecture Rules (Critical)

1. **This module does NOT modify Vinayak's DR grade or probabilities.**
2. **Clinical measurements are NOT fed into the DR model.**
3. **No ML model is trained or used.**
4. **No external APIs are called.**
5. **Missing data is never imputed — only explicitly tracked.**
6. **No clinical diagnosis or risk score is generated.**
7. **No invented medical thresholds are applied.**

---

## Future Decision Layer Integration

The downstream Final Decision / Referral Priority layer should consume:

```json
{
  "dr_result": { ... },
  "quality_result": { ... },
  "reliability_result": { ... },
  "clinical_context": { ... }
}
```

Where `clinical_context` is the direct output of `process_clinical_context()`.

---

## Privacy Guidelines

- Do not log raw clinical field values in production.
- Use `patient_id` for tracking — avoid storing PII in this field.
- No data is sent to external services.
- `other_notes` field has a 2000-character limit.

---

## Acceptance Criteria

- [x] Clinical schema defined (Pydantic v2)
- [x] Age validation works
- [x] BP validation works (including cross-field)
- [x] HbA1c validation works
- [x] Diabetes duration validation works (including cross-field)
- [x] History structure supported
- [x] Missing fields handled explicitly (provided / missing / not_recorded)
- [x] Units documented and enforced
- [x] Normalisation works
- [x] Invalid data rejected safely
- [x] No clinical diagnosis generated
- [x] No DR grade modification
- [x] No ML model trained
- [x] Public API works
- [x] JSON output stable and serialisable
- [x] Tests pass (91/91)
- [x] Demo works
- [x] README complete
- [x] No external API dependency
- [x] Ready for future Decision Layer integration
