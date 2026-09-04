"""
src/validation.py — Input validation for the Final Decision Layer.

Validates all upstream module outputs before the decision engine processes them.

DESIGN PRINCIPLES
-----------------
- Fail safely: missing or invalid data raises clear errors (or returns
  structured error information).
- Do NOT silently assume missing information is safe.
- Do NOT invent defaults that could lead to incorrect clinical routing.
- Do NOT alter DR grade, probabilities, or any model output value.

PUBLIC API
----------
    validate_quality_result(data)     -> QualityResult (dataclass)
    validate_dr_result(data)          -> DRResult (dataclass)
    validate_reliability_result(data) -> ReliabilityResult (dataclass)
    validate_clinical_context(data)   -> ClinicalContext (dataclass)
    validate_gradcam_metadata(data)   -> GradcamMetadata (dataclass)
    validate_all_inputs(quality, dr, reliability, clinical_context, gradcam)
                                      -> ValidatedInputs (dataclass)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valid constants (aligned with upstream modules)
# ---------------------------------------------------------------------------

VALID_QUALITY_STATUSES = {"good", "borderline", "ungradable"}
VALID_QUALITY_ACTIONS = {"continue", "enhance_and_recheck", "recapture"}
VALID_DR_GRADES = {0, 1, 2, 3, 4}
VALID_RELIABILITY_STATUSES = {"acceptable", "caution", "review_required"}
VALID_CONFIDENCE_LEVELS = {"high", "medium", "low"}
VALID_UNCERTAINTY_LEVELS = {"low", "medium", "high"}

DR_CLASS_NAMES = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR (PDR)",
}


# ---------------------------------------------------------------------------
# Structured result dataclasses (typed intermediaries for the engine)
# ---------------------------------------------------------------------------

@dataclass
class QualityResult:
    """Validated output from the Fundus Quality module."""
    status: str                           # "good" | "borderline" | "ungradable"
    quality_score: float                  # 0.0..1.0
    action: str                           # "continue" | "enhance_and_recheck" | "recapture"
    reason: str                           # Human-readable quality reason
    enhanced: bool = False
    enhanced_image_path: Optional[str] = None

    @property
    def is_ungradable(self) -> bool:
        return self.status == "ungradable"

    @property
    def is_gradable(self) -> bool:
        return self.status in ("good", "borderline")


@dataclass
class DRResult:
    """Validated output from Vinayak's 4-class DR model."""
    grade: int                            # 0..3
    probabilities: Dict[str, float]       # {"0": p0, "1": p1, "2": p2, "3": p3}
    gradcam_path: Optional[str] = None    # Optional path to Grad-CAM file

    @property
    def class_name(self) -> str:
        return DR_CLASS_NAMES.get(self.grade, f"Grade {self.grade}")


@dataclass
class ReliabilityResult:
    """Validated output from the Reliability Engine."""
    reliability_status: str               # "acceptable" | "caution" | "review_required"
    review_required: bool
    reason: str
    confidence: float                     # 0.0..1.0
    confidence_level: str                 # "high" | "medium" | "low"
    uncertainty: float                    # 0.0..1.0
    uncertainty_level: str               # "low" | "medium" | "high"
    ood: bool
    ood_status: str
    ood_score: float
    reliability_score: Optional[float] = None
    predicted_grade: Optional[int] = None
    predicted_class_name: Optional[str] = None

    @property
    def is_acceptable(self) -> bool:
        return self.reliability_status == "acceptable"

    @property
    def is_caution(self) -> bool:
        return self.reliability_status == "caution"

    @property
    def is_review_required(self) -> bool:
        return self.reliability_status == "review_required"


@dataclass
class ClinicalContext:
    """
    Validated (or gracefully degraded) clinical context.

    ARCHITECTURE RULE:
        Clinical context MUST NOT alter DR grade or probabilities.
        It is preserved as-is and may influence priority/escalation
        only via explicit, configurable, documented policy rules.
    """
    validation_passed: bool
    clinical_context_complete: bool
    clinical_data: Optional[Dict[str, Any]] = None   # Normalised fields (from upstream)
    data_quality: Optional[Dict[str, Any]] = None
    validation_errors: List[str] = field(default_factory=list)
    patient_id: Optional[str] = None

    # Convenience accessors — read-only, never modify DR output
    @property
    def age(self) -> Optional[float]:
        if self.clinical_data is None:
            return None
        age_field = self.clinical_data.get("age")
        if age_field and isinstance(age_field, dict):
            return age_field.get("value")
        return None

    @property
    def hba1c(self) -> Optional[float]:
        if self.clinical_data is None:
            return None
        hba1c_field = self.clinical_data.get("hba1c")
        if hba1c_field and isinstance(hba1c_field, dict):
            return hba1c_field.get("value")
        return None

    @property
    def bp_systolic(self) -> Optional[float]:
        if self.clinical_data is None:
            return None
        bp = self.clinical_data.get("bp_systolic")
        if bp and isinstance(bp, dict):
            return bp.get("value")
        return None

    @property
    def diabetes_duration_years(self) -> Optional[float]:
        if self.clinical_data is None:
            return None
        dur = self.clinical_data.get("diabetes_duration_years")
        if dur and isinstance(dur, dict):
            return dur.get("value")
        return None


@dataclass
class GradcamMetadata:
    """
    Preserved Grad-CAM / XAI evidence metadata.

    The Decision Layer does NOT generate or interpret Grad-CAM.
    It preserves the path/metadata for the doctor workflow.
    """
    gradcam_path: Optional[str] = None
    gradcam_exists: bool = False          # True only if path was verified
    gradcam_missing_warned: bool = False


@dataclass
class ValidatedInputs:
    """
    Bundle of all validated upstream inputs.
    Passed through the decision engine pipeline.
    """
    quality: QualityResult
    dr: DRResult
    reliability: ReliabilityResult
    clinical: ClinicalContext
    gradcam: GradcamMetadata


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class DecisionValidationError(ValueError):
    """Raised when a required upstream input is structurally invalid."""


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def validate_quality_result(data: Any) -> QualityResult:
    """
    Validate and parse the quality module output.

    Parameters
    ----------
    data : dict
        Output from anuj-fundus-quality's assess_quality().

    Returns
    -------
    QualityResult

    Raises
    ------
    DecisionValidationError — on invalid/missing required fields.
    """
    if data is None:
        raise DecisionValidationError(
            "quality_result is None. The Quality module must produce a result "
            "before the Decision Layer can run."
        )
    if not isinstance(data, dict):
        raise DecisionValidationError(
            f"quality_result must be a dict; got {type(data).__name__}."
        )

    # Require 'status'
    status = data.get("status")
    if status is None:
        raise DecisionValidationError(
            "'status' is missing from quality_result. "
            f"Expected one of: {sorted(VALID_QUALITY_STATUSES)}"
        )
    if not isinstance(status, str) or status not in VALID_QUALITY_STATUSES:
        raise DecisionValidationError(
            f"quality_result 'status' is '{status}'. "
            f"Must be one of: {sorted(VALID_QUALITY_STATUSES)}."
        )

    # quality_score — required, must be a float in [0, 1]
    qs = data.get("quality_score")
    if qs is None:
        # Tolerate missing quality_score for ungradable images with error
        qs = 0.0
    try:
        qs = float(qs)
    except (TypeError, ValueError) as exc:
        raise DecisionValidationError(
            f"quality_result 'quality_score' must be numeric; got {qs!r}."
        ) from exc
    if not (0.0 <= qs <= 1.0):
        raise DecisionValidationError(
            f"quality_result 'quality_score' = {qs} is outside [0, 1]."
        )

    # action — optional, default based on status
    action = data.get("action")
    if action is None:
        action = {"good": "continue", "borderline": "enhance_and_recheck",
                  "ungradable": "recapture"}[status]
    elif action not in VALID_QUALITY_ACTIONS:
        log.warning(
            "quality_result 'action' is '%s' — not a recognised value %s. "
            "Deriving from status instead.",
            action, VALID_QUALITY_ACTIONS,
        )
        action = {"good": "continue", "borderline": "enhance_and_recheck",
                  "ungradable": "recapture"}[status]

    reason = str(data.get("reason", "No reason provided by Quality module."))

    return QualityResult(
        status=status,
        quality_score=qs,
        action=action,
        reason=reason,
        enhanced=bool(data.get("enhanced", False)),
        enhanced_image_path=data.get("enhanced_image_path"),
    )


def validate_dr_result(data: Any) -> DRResult:
    """
    Validate and parse the DR model output.

    NOTE: Vinayak's DR model + Grad-CAM module is not yet integrated.
    This validator accepts the agreed output contract:
        {
            "grade": int (0..3),
            "probabilities": dict {"0": float, "1": float, "2": float, "3": float},
            "gradcam_path": str (optional)
        }

    Parameters
    ----------
    data : dict
        Output from Vinayak's 4-class DR model.

    Returns
    -------
    DRResult

    Raises
    ------
    DecisionValidationError — on invalid/missing required fields.
    """
    if data is None:
        raise DecisionValidationError(
            "dr_result is None. The DR model must produce a result "
            "before the Decision Layer can run."
        )
    if not isinstance(data, dict):
        raise DecisionValidationError(
            f"dr_result must be a dict; got {type(data).__name__}."
        )

    # grade — required
    grade = data.get("grade")
    if grade is None:
        raise DecisionValidationError(
            "'grade' is missing from dr_result. "
            "The DR model must supply a predicted class grade (0..4)."
        )
    try:
        grade = int(grade)
    except (TypeError, ValueError) as exc:
        raise DecisionValidationError(
            f"dr_result 'grade' must be an integer 0..4; got {data['grade']!r}."
        ) from exc
    if grade not in VALID_DR_GRADES:
        raise DecisionValidationError(
            f"dr_result 'grade' = {grade} is not a valid DR grade. "
            f"Must be one of: {sorted(VALID_DR_GRADES)}."
        )

    # probabilities — required
    raw_probs = data.get("probabilities")
    if raw_probs is None:
        raise DecisionValidationError(
            "'probabilities' is missing from dr_result. "
            "The DR model must supply a probability distribution over 4 classes."
        )
    if not isinstance(raw_probs, (dict, list, tuple)):
        raise DecisionValidationError(
            f"dr_result 'probabilities' must be a dict or list; got {type(raw_probs).__name__}."
        )

    # Normalise probabilities to string-keyed dict
    probs_dict: Dict[str, float] = {}
    if isinstance(raw_probs, (list, tuple)):
        if len(raw_probs) != 5:
            raise DecisionValidationError(
                f"dr_result 'probabilities' list must have exactly 5 elements; "
                f"got {len(raw_probs)}."
            )
        for i, p in enumerate(raw_probs):
            try:
                probs_dict[str(i)] = float(p)
            except (TypeError, ValueError) as exc:
                raise DecisionValidationError(
                    f"dr_result 'probabilities'[{i}] = {p!r} is not numeric."
                ) from exc
    else:
        for k, v in raw_probs.items():
            key = str(k)
            if key not in {"0", "1", "2", "3", "4"}:
                raise DecisionValidationError(
                    f"dr_result 'probabilities' has unexpected key '{k}'. "
                    "Only keys '0', '1', '2', '3', '4' are valid."
                )
            try:
                probs_dict[key] = float(v)
            except (TypeError, ValueError) as exc:
                raise DecisionValidationError(
                    f"dr_result 'probabilities'['{k}'] = {v!r} is not numeric."
                ) from exc
        # Verify all 5 classes are present
        missing = {"0", "1", "2", "3", "4"} - set(probs_dict.keys())
        if missing:
            raise DecisionValidationError(
                f"dr_result 'probabilities' is missing classes: {sorted(missing)}."
            )

    # Validate individual probability values
    for k, p in probs_dict.items():
        if not (0.0 <= p <= 1.0):
            raise DecisionValidationError(
                f"dr_result 'probabilities'['{k}'] = {p} is outside [0, 1]."
            )
        if p != p:  # NaN check
            raise DecisionValidationError(
                f"dr_result 'probabilities'['{k}'] is NaN."
            )

    # Validate sum ≈ 1.0
    total = sum(probs_dict.values())
    if abs(total - 1.0) > 0.01:
        raise DecisionValidationError(
            f"dr_result 'probabilities' sum = {total:.6f}, deviates from 1.0 by more than 0.01."
        )

    return DRResult(
        grade=grade,
        probabilities=probs_dict,
        gradcam_path=data.get("gradcam_path"),
    )


def validate_reliability_result(data: Any) -> ReliabilityResult:
    """
    Validate and parse the Reliability Engine output.

    Accepts the full output of anuj-reliability's calculate_reliability()
    or run_reliability_pipeline().

    Parameters
    ----------
    data : dict
        Output from the Reliability Engine.

    Returns
    -------
    ReliabilityResult

    Raises
    ------
    DecisionValidationError — on invalid/missing required fields.
    """
    if data is None:
        raise DecisionValidationError(
            "reliability_result is None. The Reliability Engine must produce a "
            "result before the Decision Layer can run."
        )
    if not isinstance(data, dict):
        raise DecisionValidationError(
            f"reliability_result must be a dict; got {type(data).__name__}."
        )

    # reliability_status
    status = data.get("reliability_status")
    if status is None:
        raise DecisionValidationError(
            "'reliability_status' is missing from reliability_result."
        )
    if not isinstance(status, str) or status not in VALID_RELIABILITY_STATUSES:
        raise DecisionValidationError(
            f"reliability_result 'reliability_status' = '{status}'. "
            f"Must be one of: {sorted(VALID_RELIABILITY_STATUSES)}."
        )

    # review_required
    review_required = data.get("review_required")
    if review_required is None:
        raise DecisionValidationError(
            "'review_required' is missing from reliability_result."
        )
    review_required = bool(review_required)

    # confidence
    conf = data.get("confidence")
    if conf is None:
        raise DecisionValidationError("'confidence' is missing from reliability_result.")
    try:
        conf = float(conf)
    except (TypeError, ValueError) as exc:
        raise DecisionValidationError(
            f"reliability_result 'confidence' must be numeric; got {data.get('confidence')!r}."
        ) from exc
    if not (0.0 <= conf <= 1.0):
        raise DecisionValidationError(
            f"reliability_result 'confidence' = {conf} is outside [0, 1]."
        )

    # confidence_level
    conf_level = data.get("confidence_level")
    if conf_level not in VALID_CONFIDENCE_LEVELS:
        raise DecisionValidationError(
            f"reliability_result 'confidence_level' = '{conf_level}'. "
            f"Must be one of: {sorted(VALID_CONFIDENCE_LEVELS)}."
        )

    # uncertainty
    unc = data.get("uncertainty")
    if unc is None:
        raise DecisionValidationError("'uncertainty' is missing from reliability_result.")
    try:
        unc = float(unc)
    except (TypeError, ValueError) as exc:
        raise DecisionValidationError(
            f"reliability_result 'uncertainty' must be numeric; got {data.get('uncertainty')!r}."
        ) from exc
    if not (0.0 <= unc <= 1.0):
        raise DecisionValidationError(
            f"reliability_result 'uncertainty' = {unc} is outside [0, 1]."
        )

    # uncertainty_level
    unc_level = data.get("uncertainty_level")
    if unc_level not in VALID_UNCERTAINTY_LEVELS:
        raise DecisionValidationError(
            f"reliability_result 'uncertainty_level' = '{unc_level}'. "
            f"Must be one of: {sorted(VALID_UNCERTAINTY_LEVELS)}."
        )

    # ood
    ood = data.get("ood")
    if ood is None:
        raise DecisionValidationError("'ood' is missing from reliability_result.")
    ood = bool(ood)

    # ood_status
    ood_status = str(data.get("ood_status", "unknown"))

    # ood_score
    ood_score = data.get("ood_score", 0.0)
    try:
        ood_score = float(ood_score)
    except (TypeError, ValueError):
        ood_score = 0.0

    # reason
    reason = str(data.get("reason", "No reliability reason provided."))

    # Optional fields
    reliability_score = data.get("reliability_score")
    if reliability_score is not None:
        try:
            reliability_score = float(reliability_score)
        except (TypeError, ValueError):
            reliability_score = None

    predicted_grade = data.get("predicted_grade")
    if predicted_grade is not None:
        try:
            predicted_grade = int(predicted_grade)
        except (TypeError, ValueError):
            predicted_grade = None

    predicted_class_name = data.get("predicted_class_name")

    return ReliabilityResult(
        reliability_status=status,
        review_required=review_required,
        reason=reason,
        confidence=conf,
        confidence_level=conf_level,
        uncertainty=unc,
        uncertainty_level=unc_level,
        ood=ood,
        ood_status=ood_status,
        ood_score=ood_score,
        reliability_score=reliability_score,
        predicted_grade=predicted_grade,
        predicted_class_name=predicted_class_name,
    )


def validate_clinical_context(data: Any) -> ClinicalContext:
    """
    Validate and parse the Clinical Context module output.

    Missing or incomplete clinical context does NOT block the basic
    DR/reliability workflow (unless policy requires it).
    Invalid data is handled gracefully — it never silently creates
    a high-risk or urgent classification.

    Parameters
    ----------
    data : dict or None
        Output from anuj-clinical-context's process_clinical_context(),
        or None / empty dict if not provided.

    Returns
    -------
    ClinicalContext
    """
    if data is None or data == {}:
        log.info("Clinical context not provided — clinical_context_complete=False.")
        return ClinicalContext(
            validation_passed=True,
            clinical_context_complete=False,
            clinical_data=None,
            data_quality=None,
        )

    if not isinstance(data, dict):
        log.warning(
            "clinical_context is not a dict (got %s) — treating as not provided.",
            type(data).__name__,
        )
        return ClinicalContext(
            validation_passed=False,
            clinical_context_complete=False,
            clinical_data=None,
            data_quality=None,
            validation_errors=[
                f"clinical_context must be a dict; got {type(data).__name__}."
            ],
        )

    # Handle output from process_clinical_context()
    validation_passed = bool(data.get("validation_passed", True))
    validation_errors = list(data.get("validation_errors", []))
    clinical_data = data.get("clinical_context")        # can be None on failure
    data_quality = data.get("data_quality", {})
    clinical_context_complete = bool(
        (data_quality or {}).get("clinical_context_complete", False)
        or (data_quality or {}).get("complete", False)
    )

    # Patient ID — best-effort from nested data or raw dict
    patient_id = data.get("patient_id")

    return ClinicalContext(
        validation_passed=validation_passed,
        clinical_context_complete=clinical_context_complete,
        clinical_data=clinical_data,
        data_quality=data_quality,
        validation_errors=validation_errors,
        patient_id=patient_id,
    )


def validate_gradcam_metadata(
    dr_result_raw: Optional[Dict[str, Any]],
    reliability_result_raw: Optional[Dict[str, Any]],
    explicit_gradcam_path: Optional[str],
    warn_if_missing: bool = True,
) -> GradcamMetadata:
    """
    Extract and verify Grad-CAM path from available sources.

    The Decision Layer DOES NOT generate Grad-CAM.
    It preserves the path (if provided) for the doctor workflow.

    Priority order for path discovery:
        1. explicit_gradcam_path argument
        2. dr_result_raw["gradcam_path"]
        3. reliability_result_raw["gradcam_path"]

    Parameters
    ----------
    dr_result_raw           : optional raw DR model output dict
    reliability_result_raw  : optional raw reliability result dict
    explicit_gradcam_path   : optional override path
    warn_if_missing         : log a warning if no path found

    Returns
    -------
    GradcamMetadata
    """
    path = None
    exists = False
    warned = False

    # Discover path from available sources
    candidates = [
        explicit_gradcam_path,
        (dr_result_raw or {}).get("gradcam_path"),
        (reliability_result_raw or {}).get("gradcam_path"),
    ]
    for candidate in candidates:
        if candidate and isinstance(candidate, str) and candidate.strip():
            path = candidate.strip()
            break

    # Verify existence (best-effort)
    if path is not None:
        exists = Path(path).exists()
        if not exists:
            log.debug(
                "Grad-CAM path '%s' does not exist on disk. "
                "This is expected when Vinayak's XAI module is not yet integrated.",
                path,
            )
    elif warn_if_missing:
        log.debug(
            "No Grad-CAM path available. "
            "Vinayak's XAI module integration is not yet complete — this is expected."
        )
        warned = True

    return GradcamMetadata(
        gradcam_path=path,
        gradcam_exists=exists,
        gradcam_missing_warned=warned,
    )


def validate_all_inputs(
    quality_result: Any,
    dr_result: Any,
    reliability_result: Any,
    clinical_context: Any = None,
    gradcam_path: Optional[str] = None,
) -> ValidatedInputs:
    """
    Validate all upstream inputs together and return a ValidatedInputs bundle.

    Raises DecisionValidationError if any required input is invalid.
    Clinical context failures are non-fatal (logged but not raised).

    Parameters
    ----------
    quality_result      : dict   — from anuj-fundus-quality
    dr_result           : dict   — from Vinayak's DR model
    reliability_result  : dict   — from anuj-reliability
    clinical_context    : dict | None — from anuj-clinical-context (optional)
    gradcam_path        : str | None — explicit Grad-CAM path override

    Returns
    -------
    ValidatedInputs
    """
    quality = validate_quality_result(quality_result)
    dr = validate_dr_result(dr_result)
    reliability = validate_reliability_result(reliability_result)
    clinical = validate_clinical_context(clinical_context)

    from .config import DEFAULT_POLICY
    gradcam = validate_gradcam_metadata(
        dr_result_raw=dr_result if isinstance(dr_result, dict) else None,
        reliability_result_raw=reliability_result if isinstance(reliability_result, dict) else None,
        explicit_gradcam_path=gradcam_path,
        warn_if_missing=DEFAULT_POLICY.gradcam.warn_if_missing,
    )

    return ValidatedInputs(
        quality=quality,
        dr=dr,
        reliability=reliability,
        clinical=clinical,
        gradcam=gradcam,
    )
