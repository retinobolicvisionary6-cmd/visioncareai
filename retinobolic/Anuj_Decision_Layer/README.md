# anuj-decision — Final Decision Layer

**SIH Problem ID:** SIH26038  
**Project:** Explainable AI for Diabetic Retinopathy Screening in Rural India  
**Module:** Final Decision Layer (Deterministic Workflow Engine)  
**Version:** 1.0.0

> ⚠️ **ENGINEERING EVALUATION ONLY.**  
> This module produces **workflow-routing decisions only** — not medical diagnoses.  
> The final clinical decision rests entirely with the examining physician.

---

## Purpose

The Decision Layer is the **final deterministic orchestration module** in the SIH26038 pipeline.

It answers:

> *"Given the image quality, DR prediction, reliability signals, and available clinical context — what should the screening workflow do next?"*

This is **not** a new deep-learning model. It is transparent, configurable, deterministic software logic.

---

## Pipeline Position

```
📸 FUNDUS IMAGE
      ↓
🔍 QUALITY GATE         (anuj-fundus-quality)
      ↓
🤖 VINAYAK DR MODEL     (Vinayak — pending integration)
      ↓
💡 GRAD-CAM / XAI       (Vinayak — pending integration)
      ↓
   ┌──────────────────┐
   │ CONFIDENCE       │ (anuj-confidence)
   │ UNCERTAINTY      │ (anuj-uncertainty)
   │ OOD DETECTION    │ (anuj-ood)
   └──────────────────┘
      ↓
🔐 RELIABILITY ENGINE   (anuj-reliability)
      ↓
🩺 CLINICAL CONTEXT     (anuj-clinical-context)
      ↓
🤖 FINAL DECISION LAYER ← THIS MODULE
      ↓
   ┌────────────┬──────────────┬────────────┐
   ↓            ↓              ↓            ↓
🔄 RECAPTURE  ⚠️ REVIEW    🚨 REFER    ✅ ROUTINE
      ↓
👨‍⚕️ DOCTOR VALIDATION
      ↓
REFERRAL / FOLLOW-UP
```

---

## Quick Start

```python
from src.engine import run_screening_decision

# All inputs come from upstream modules — no recalculation here
result = run_screening_decision(
    quality_result=quality_output,       # from anuj-fundus-quality
    dr_result=dr_output,                 # from Vinayak's DR model
    reliability_result=reliability_output, # from anuj-reliability
    clinical_context=clinical_output,    # from anuj-clinical-context (optional)
)

print(result["action"])    # "routine" | "doctor_review" | "refer" | "recapture"
print(result["priority"])  # "low" | "medium" | "high" | "urgent"
print(result["reason"])    # Clear, non-diagnostic explanation
```

---

## Possible Actions

| Action | Label | Priority | Description |
|--------|-------|----------|-------------|
| `routine` | ✅ ROUTINE | `low` | Gradable, reliable, non-referable DR |
| `recapture` | 🔄 RECAPTURE IMAGE | `medium` | Image ungradable — retake required |
| `doctor_review` | ⚠️ DOCTOR REVIEW | `high` | Reliability failure — OOD, uncertainty, or low confidence |
| `refer` | 🚨 REFER | `high` / `urgent` | Reliable DR prediction meets referral threshold |

---

## Decision Priority Order

Rules are evaluated in strict safety-first order. A lower-priority rule **can never override** a higher-priority safety rule.

### Rule 1 — Image Safety (HIGHEST)
```
quality.status == "ungradable"  →  action = "recapture"
```
**Conflict example:** Grade 3, confidence 0.99, but ungradable → **recapture** (never refer).

### Rule 2 — Reliability Failure
Any of the following triggers `doctor_review`:
- `ood == True`
- `uncertainty_level == "high"`
- `confidence_level == "low"`
- `reliability_status == "review_required"`

**Conflict example:** Grade 3, OOD=True → **doctor_review** (never auto-refer).

### Rule 3 — Reliable Referable DR
```
quality is gradable  AND  reliability acceptable  AND  grade >= threshold
→  action = "refer"
```
Threshold is configurable (default: Grade 2). Grade 3 escalates to `urgent`.

### Rule 4 — Routine (Default)
```
quality is gradable  AND  reliability acceptable  AND  grade < threshold
→  action = "routine"
```

---

## Output Contract

```json
{
  "action": "refer",
  "priority": "high",
  "reason": "Reliable DR prediction (Moderate DR, Grade 2) meets the configured referral threshold (Grade >= 2). Referral for specialist assessment is warranted. Final clinical decision rests with the examining physician.",
  "dr_grade": 2,
  "reliability_status": "acceptable",
  "review_required": false,
  "evidence": {
    "quality_status": "good",
    "quality_score": 0.88,
    "confidence": 0.88,
    "confidence_level": "high",
    "uncertainty": 0.22,
    "uncertainty_level": "low",
    "ood": false,
    "ood_score": 1.55,
    "reliability_score": 0.84,
    "gradcam_path": "outputs/gradcam/patient_003.jpg",
    "clinical_context_complete": true,
    "reliability_signals": []
  },
  "metadata": {
    "rule_applied": "RULE_3_REFERABLE_DR",
    "engine_version": "1.0.0"
  }
}
```

**Recapture output** (dr_grade and reliability_status are `null`):
```json
{
  "action": "recapture",
  "priority": "medium",
  "reason": "Fundus image is ungradable due to insufficient image quality...",
  "dr_grade": null,
  "reliability_status": null,
  "review_required": false
}
```

---

## Project Structure

```
anuj-decision/
├── config/
│   └── decision_policy.yaml    # All configurable thresholds and policies
├── src/
│   ├── __init__.py             # Public API exports
│   ├── config.py               # YAML loader → DecisionPolicy dataclass
│   ├── validation.py           # Input validators (all upstream modules)
│   ├── rules.py                # 4 deterministic decision rules
│   ├── priorities.py           # Priority mapper (low/medium/high/urgent)
│   ├── reasons.py              # Non-diagnostic reason generator
│   └── engine.py               # make_final_decision / run_screening_decision
├── tests/
│   ├── test_validation.py      # 35 validation unit tests
│   ├── test_rules.py           # 24 rule unit tests
│   ├── test_decision_cases.py  # 9 required scenarios + conflict + contract tests
│   └── test_engine.py          # Integration + pipeline simulation tests
├── sample_data/
│   ├── routine.json
│   ├── recapture.json
│   ├── review.json
│   ├── refer.json
│   └── conflict_cases.json
├── outputs/                    # Demo result JSONs
├── demo.py                     # Rich CLI demo (8 scenarios)
├── requirements.txt
└── README.md
```

---

## Configuration

All decision policies are in [`config/decision_policy.yaml`](config/decision_policy.yaml).  
**Do NOT hardcode thresholds in source code.**

### Key configurable values

```yaml
referral:
  referable_grade_threshold: 2      # Grades >= 2 are referable (prototype)
  urgent_referral_grades: [3]       # Grade 3 → "urgent" priority

priority:
  recapture: medium
  doctor_review: high
  refer: high
  refer_urgent: urgent
  routine: low

reliability:
  statuses_blocking_referral: [review_required]
  caution_allows_referral: false

clinical_context:
  enable_escalation: false          # disabled by default (not clinically validated)
  require_complete_context: false   # missing context does not block workflow
```

> ⚠️ All numeric thresholds are **prototype engineering values** that have NOT been clinically validated.

---

## Running Tests

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

**126 tests — all pass.**

---

## Running the Demo

```bash
# All 8 scenarios
python -X utf8 demo.py

# Single scenario
python -X utf8 demo.py --scenario refer_grade3
python -X utf8 demo.py --scenario conflict_ungradable
```

Available scenarios: `routine`, `recapture`, `review_ood`, `review_uncertainty`,  
`refer_grade2`, `refer_grade3`, `conflict_ungradable`, `conflict_ood_confidence`

---

## Architecture Rules

| Rule | Enforced? |
|------|-----------|
| No new ML model | ✅ |
| No LLM | ✅ |
| No recalculation of upstream signals | ✅ |
| Clinical context cannot modify DR grade | ✅ |
| Missing clinical context does not create urgent flags | ✅ |
| Grad-CAM is preserved, not interpreted | ✅ |
| Ungradable always → recapture (no override) | ✅ |
| OOD always → doctor_review (no auto-referral) | ✅ |
| All thresholds are externalized to YAML | ✅ |
| No autonomous diagnosis statements in reasons | ✅ |
| Doctor remains responsible for clinical decision | ✅ |

---

## Integration Status

| Module | Status | Interface |
|--------|--------|-----------|
| `anuj-fundus-quality` | ✅ Integrated | `assess_quality()` |
| `anuj-reliability` | ✅ Integrated | `calculate_reliability()` |
| `anuj-clinical-context` | ✅ Integrated | `process_clinical_context()` |
| Vinayak DR Model | ⏳ Pending | DR result contract defined |
| Grad-CAM / XAI | ⏳ Pending | `gradcam_path` preserved in output |

---

## Acceptance Criteria

- [x] Quality result consumed
- [x] DR result consumed
- [x] Reliability result consumed
- [x] Clinical context consumed
- [x] Grad-CAM path preserved
- [x] Safety priority implemented
- [x] Ungradable → Recapture
- [x] Unreliable → Doctor Review
- [x] Reliable referable → Refer
- [x] Reliable non-referable → Routine
- [x] Conflict resolution works
- [x] Missing data handled
- [x] Reasons generated
- [x] Priority generated
- [x] Stable JSON output
- [x] English action labels
- [x] No autonomous diagnosis
- [x] No treatment advice
- [x] No new ML model
- [x] Existing modules not duplicated
- [x] Integration test passes
- [x] Unit tests pass (126/126)
- [x] README complete

---

## Safety Note

This module is part of a **screening workflow assistance system**.  
It does NOT:
- Diagnose Diabetic Retinopathy
- Prescribe treatment
- Replace a physician
- Interpret Grad-CAM as definitive lesion localization

The doctor retains full clinical responsibility for all decisions.
