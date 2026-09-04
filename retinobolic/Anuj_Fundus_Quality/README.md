# Fundus Image Quality Assessment Engine

**SIH Project:** Explainable AI for Diabetic Retinopathy Screening in Rural India  
**Problem ID:** SIH26038  
**Built by:** Anuj  
**Role in pipeline:** Image Quality Gate — runs BEFORE the DR model

> ⚠️ **Disclaimer:** This is a prototype image-quality gate and has **not been clinically validated**. All thresholds are engineering starting points and must not be used for clinical decision-making.

---

## What This Is (Read This First)

**This is a Python Module — not an app, not a website, not a backend.**

It is a single reusable function:

```python
from src.quality import assess_quality

result = assess_quality("path/to/eye_image.jpg")
```

That one call does everything: loads the image, checks quality across 5 dimensions,
tries to fix borderline images, and returns a structured dictionary with the decision.

**It has zero dependency on:**
- The DR classification model (Vinayak's code)
- Any frontend / website
- Any backend / API server
- Any database

Vinayak imports this module into his pipeline with 3 lines of code. The frontend/backend
team then wraps both modules into an API. Our job ends at `assess_quality()`.

---

## Purpose

This module answers one critical question before any DR screening happens:

> **"Is this retinal fundus image good enough for downstream DR screening?"**

It is a standalone, reusable component completely independent of the rest of the system.

---

## Architecture

```
Fundus Image (uploaded or camera-captured)
        ↓
   Preprocessing
   • Load + Validate (JPG/JPEG/PNG)
   • Resize (longest edge → 512 px)
   • Fundus ROI extraction (Hough circles + fallback ellipse)
        ↓
   ┌──────────────────────────────┐
   │  Quality Analysis Modules    │
   │                              │
   │  Focus / Sharpness           │  ← Laplacian variance + Tenengrad
   │  Illumination                │  ← Brightness, contrast, clipping
   │  Field of View               │  ← Disk fill ratio, area ratio
   │  Retinal Visibility          │  ← Edge density proxy (CLAHE + Canny)
   │  Artifact Detection          │  ← Glare ratio, noise level
   └──────────────────────────────┘
        ↓
   Hard Safety Gates
   (critical focus / illumination / FOV failure → immediate UNGRADABLE)
        ↓
   Weighted Composite Quality Score
        ↓
   ┌──────────────────────────────┐
   │  GOOD        ≥ 0.75          │  → continue to DR model
   │  BORDERLINE  0.45–0.75       │  → enhance → re-assess
   │  UNGRADABLE  < 0.45          │  → recapture
   └──────────────────────────────┘
        ↓ (BORDERLINE only)
   Enhancement Pipeline
   • CLAHE on luminance channel
   • Gamma correction (illumination normalisation)
   • Mild NLM denoising
        ↓
   Re-assessment → final decision
```

---

## Project Structure

```
anuj-fundus-quality/
│
├── src/
│   ├── __init__.py           — public API: assess_quality, result_to_json
│   ├── quality.py            — central orchestrator
│   ├── focus.py              — Laplacian + Tenengrad sharpness
│   ├── illumination.py       — brightness / contrast / clipping / uniformity
│   ├── field_of_view.py      — fundus area / disk fill ratio
│   ├── retinal_visibility.py — CLAHE + Canny edge density proxy
│   ├── artifacts.py          — glare ratio + noise level
│   ├── enhancement.py        — CLAHE → gamma → NLM denoising
│   ├── preprocessing.py      — load, validate, resize, ROI extraction
│   └── config.py             — all configurable thresholds
│
├── tests/
│   ├── conftest.py
│   ├── generate_synthetic_data.py  — create prototype test images
│   ├── test_focus.py
│   ├── test_illumination.py
│   ├── test_field_of_view.py
│   ├── test_quality.py             — integration tests
│   └── test_enhancement.py
│
├── sample_data/              — synthetic test images (auto-generated)
├── outputs/
│   ├── quality/              — JSON quality results (auto-saved)
│   └── enhanced/             — enhanced borderline images (auto-saved)
│
├── requirements.txt
├── README.md
└── demo.py
```

---

## Installation

```bash
# 1. Clone the repository / extract to the project folder
cd anuj-fundus-quality

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate synthetic test images (no real fundus images needed for testing)
python tests/generate_synthetic_data.py
```

**Dependencies:** `opencv-python`, `numpy`, `scikit-image`, `pytest`

No GPU required. No deep-learning frameworks required for this module.

---

## Quick Start

### Python API

```python
from src.quality import assess_quality

result = assess_quality("path/to/fundus_image.jpg")
print(result["status"])        # "good", "borderline", or "ungradable"
print(result["quality_score"]) # 0.0 – 1.0
print(result["action"])        # "continue", "enhance_and_recheck", "recapture"
print(result["reason"])        # human-readable explanation
```

### CLI Demo

```bash
python demo.py --image sample_data/good_fundus.jpg
```

Expected terminal output:
```
====================================================
   Fundus Image Quality Assessment
====================================================
   Image: sample_data/good_fundus.jpg
====================================================
   Running analysis …

   Status        : ✅ GOOD
   Image Quality : 82%  [████████████████░░░░]

   Component Scores
   ─────────────────────────────────────────────
   Focus          :  79%  [███████████████░░░░░]
   Illumination   :  85%  [█████████████████░░░]
   Field of View  :  90%  [██████████████████░░]
   Retinal Vis.   :  78%  [███████████████░░░░░]
   Artifacts      : Acceptable

   ─────────────────────────────────────────────
   Action  : CONTINUE
   Reason  : Image is suitable for screening.
====================================================
```

---

## Quality Metrics

| Metric | Description | Method |
|---|---|---|
| **focus_score** | Sharpness of retinal structures | Laplacian variance + Tenengrad (Sobel) |
| **illumination_score** | Brightness, contrast, uniformity | Histogram statistics on ROI |
| **field_of_view_score** | Retinal area coverage | Hough circle disk fill ratio |
| **retinal_visibility_score** | Visibility of retinal structures | CLAHE + adaptive Canny edge density |
| **artifact_score** | Absence of glare / noise artifacts | Saturated-pixel ratio + HF energy |
| **quality_score** | Weighted composite | Configurable weights in `config.py` |

### Fusion Weights (configurable in `config.py`)

| Component | Default Weight |
|---|---|
| Focus | 30% |
| Illumination | 25% |
| Field of View | 20% |
| Retinal Visibility | 15% |
| Artifacts | 10% |

---

## Output Format (stable JSON contract)

```json
{
  "status": "good",
  "quality_score": 0.82,
  "focus_score": 0.79,
  "illumination_score": 0.85,
  "field_of_view_score": 0.90,
  "retinal_visibility_score": 0.78,
  "artifact_score": 0.91,
  "reason": "Image is suitable for screening.",
  "action": "continue",
  "enhanced": false,
  "enhanced_image_path": null,
  "error": null
}
```

| Field | Values | Description |
|---|---|---|
| `status` | `good`, `borderline`, `ungradable` | Gradability decision |
| `quality_score` | 0.0 – 1.0 | Composite Image Quality Score |
| `*_score` | 0.0 – 1.0 | Per-component scores |
| `reason` | string | Human-readable explanation (from actual findings) |
| `action` | `continue`, `enhance_and_recheck`, `recapture` | Recommended next step |
| `enhanced` | bool | Whether enhancement was applied |
| `enhanced_image_path` | string or null | Path to enhanced image if created |
| `error` | string or null | Error message if loading failed |

---

## Enhancement Workflow

Enhancement is **only applied to BORDERLINE images**. It is not applied to GOOD or UNGRADABLE images.

```
BORDERLINE
    ↓
CLAHE (contrast-limited adaptive histogram equalisation on L-channel)
    ↓
Gamma correction (illumination normalisation toward mid-tone)
    ↓
Mild NLM denoising (fastNlMeansDenoisingColored)
    ↓
Save enhanced image → outputs/enhanced/<stem>_enhanced.jpg
    ↓
Re-run quality assessment
    ↓
GOOD → continue to DR model
BORDERLINE → use better of original / enhanced result
UNGRADABLE → recapture
```

> **Enhancement ≠ automatic success.** A critically degraded borderline image may remain ungradable after enhancement.

---

## Integration with Vinayak's DR Model

This module is designed to be the upstream gate for the DR classification model.

```python
from src.quality import assess_quality

def process_fundus_image(image_path: str):
    result = assess_quality(image_path)

    if result["status"] == "good":
        # ✅ Safe to pass to DR model
        dr_result = your_dr_model.predict(image_path)
        return dr_result

    elif result["status"] == "borderline":
        # 🟡 Enhancement was attempted and re-assessed
        # The result dict already contains the post-enhancement decision
        if result["status"] == "good":
            dr_result = your_dr_model.predict(result["enhanced_image_path"])
            return dr_result
        else:
            return {"status": "recapture", "reason": result["reason"]}

    else:  # ungradable
        # ❌ Do not pass to DR model
        return {"status": "recapture", "reason": result["reason"]}
```

**Important:** This module has **zero dependency** on the DR model. The integration is one-directional: quality gate → DR model.

---

## Configuration

All thresholds are in [`src/config.py`](src/config.py). Edit that file to tune:

- `QUALITY_SCORE_GOOD` / `QUALITY_SCORE_BORDERLINE_LOW` — decision boundaries
- `FOCUS_LAP_VARIANCE_*` — focus sensitivity
- `ILLUM_MEAN_*` — brightness acceptable range
- `WEIGHT_*` — fusion weights
- `CLAHE_CLIP_LIMIT`, `DENOISE_H` — enhancement aggressiveness

---

## Running Tests

```bash
# Generate synthetic test images first (if not already done)
python tests/generate_synthetic_data.py

# Run all tests
pytest tests/ -v

# Run specific module tests
pytest tests/test_quality.py -v
pytest tests/test_focus.py -v
```

---

## Limitations

1. **Not clinically validated.** Thresholds are engineering estimates.
2. **Synthetic test data only.** Test images are computer-generated, not real patient photographs.
3. **ROI detection heuristic.** Hough circle detection may fail on unusual crops; an ellipse fallback is used.
4. **Retinal visibility is a proxy.** Edge density is used as a proxy for vessel/structure visibility — it is not true anatomical segmentation.
5. **Artifact detection is basic.** Only glare and noise are detected; other artifacts (dust, eyelashes, motion blur rings) are not specifically modelled in v1.
6. **No learned quality model.** All metrics are classical CV — interpretable but may miss subtle quality failures that a trained model would catch.

---

## Future Improvements

- [ ] Replace edge density with Frangi vessel filter for better vessel visibility proxy
- [ ] Add Hough circle refinement for better ROI on unusual crops
- [ ] Integrate a lightweight learned quality score (e.g., BRISQUE, or fine-tuned EfficientNet-B0)
- [ ] Expand artifact detection (motion blur, dust, eyelashes)
- [ ] Validate thresholds against EyePACS or Kaggle DR dataset quality labels
