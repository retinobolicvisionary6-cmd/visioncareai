"""
src/reasons.py — Reason Generator for the Final Decision Layer.

Generates clear, explainable, workflow-oriented reason strings for each
decision action.

DESIGN PRINCIPLES
-----------------
- Every reason must be clear to a healthcare worker without technical jargon.
- Reasons must be workflow-oriented, NOT diagnostic statements.
- The engine must NEVER produce statements such as:
    "Patient definitely has DR."
    "Patient is disease-free."
    "Treatment required."
- Use workflow-oriented language:
    "Routine screening follow-up."
    "Doctor review required due to ..."
    "Referral warranted."
    "Image recapture required."
- Multiple failure signals are reported together.

PUBLIC API
----------
    generate_reason(action, rule_name, trigger, quality, dr, reliability, clinical)
        -> str
"""

from __future__ import annotations

import logging
from typing import Optional

from .validation import (
    QualityResult,
    DRResult,
    ReliabilityResult,
    ClinicalContext,
    GradcamMetadata,
)
from .config import DecisionPolicy, DEFAULT_POLICY

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Base reason templates
# ---------------------------------------------------------------------------

_REASON_RECAPTURE = (
    "Fundus image is ungradable due to insufficient image quality. "
    "A new image must be captured before DR screening can proceed."
)

_REASON_REFER_BASE = (
    "Reliable DR prediction ({class_name}, Grade {grade}) meets the configured "
    "referral threshold (Grade >= {threshold}). "
    "Referral for specialist assessment is warranted. "
    "Final clinical decision rests with the examining physician."
)

_REASON_ROUTINE = (
    "Image is gradable, model reliability is acceptable, and the DR prediction "
    "({class_name}, Grade {grade}) does not meet the configured referral threshold. "
    "Standard screening follow-up is indicated. "
    "This is a screening-workflow classification, not a clinical diagnosis."
)

_REASON_REVIEW_HEADER = (
    "DR prediction cannot be automatically actioned — doctor review is required. "
    "The following issue(s) were detected:"
)

# Specific reliability failure messages
_TRIGGER_OOD = (
    "The fundus image appears to be outside the model's known/reference distribution "
    "(out-of-distribution). The DR model prediction may not be reliable for this image."
)
_TRIGGER_HIGH_UNCERTAINTY = (
    "The DR prediction has high model uncertainty. "
    "The model's probability distribution is ambiguous across multiple classes."
)
_TRIGGER_LOW_CONFIDENCE = (
    "The DR prediction has low model confidence. "
    "The model's top-class probability is below the acceptable threshold."
)
_TRIGGER_REVIEW_REQUIRED = (
    "The reliability engine has flagged this prediction as requiring review."
)


def generate_reason(
    action: str,
    rule_name: str,
    trigger: str,
    quality: QualityResult,
    dr: DRResult,
    reliability: ReliabilityResult,
    clinical: ClinicalContext,
    policy: DecisionPolicy = DEFAULT_POLICY,
    gradcam: Optional[GradcamMetadata] = None,
) -> str:
    """
    Generate a clear, human-readable reason string for the final decision.

    Parameters
    ----------
    action      : str               — "recapture" | "doctor_review" | "refer" | "routine"
    rule_name   : str               — Name of the firing rule (e.g. "RULE_1_UNGRADABLE")
    trigger     : str               — Internal trigger description from the rule
    quality     : QualityResult     — validated quality output
    dr          : DRResult          — validated DR model output
    reliability : ReliabilityResult — validated reliability output
    clinical    : ClinicalContext   — validated clinical context
    policy      : DecisionPolicy    — configurable decision policy
    gradcam     : GradcamMetadata | None — Grad-CAM metadata (optional)

    Returns
    -------
    str — A clear, non-diagnostic reason string.
    """
    if action == "recapture":
        return _reason_recapture(quality)

    if action == "doctor_review":
        return _reason_doctor_review(reliability, trigger)

    if action == "refer":
        return _reason_refer(dr, policy)

    if action == "routine":
        return _reason_routine(dr, policy)

    # Fallthrough (should not occur)
    log.warning("Unknown action '%s' in reason generator.", action)
    return f"Unrecognised action '{action}'. Please verify the decision configuration."


def _reason_recapture(quality: QualityResult) -> str:
    """
    Build the recapture reason.
    Includes the upstream quality module's own reason where available.
    """
    reason = _REASON_RECAPTURE
    if quality.reason and quality.reason not in ("No reason provided by Quality module.", ""):
        reason += f" Quality assessment detail: {quality.reason}"
    return reason


def _reason_doctor_review(reliability: ReliabilityResult, trigger: str) -> str:
    """
    Build the doctor_review reason.

    Enumerates all detected reliability issues from the trigger string
    and the reliability module's own reason.
    """
    issues = []

    # Map trigger conditions to human-facing explanations
    if "OOD" in trigger or "out-of-distribution" in trigger.lower() or reliability.ood:
        issues.append(_TRIGGER_OOD)

    if "high model uncertainty" in trigger or reliability.uncertainty_level == "high":
        issues.append(_TRIGGER_HIGH_UNCERTAINTY)

    if "low model confidence" in trigger or reliability.confidence_level == "low":
        issues.append(_TRIGGER_LOW_CONFIDENCE)

    if reliability.is_review_required and not issues:
        # Reliability engine flagged review for a reason not captured above
        issues.append(_TRIGGER_REVIEW_REQUIRED)
        # Include the upstream reason for transparency
        if reliability.reason:
            issues.append(f"Reliability engine detail: {reliability.reason}")

    if not issues:
        # Fallback: use the trigger directly
        issues.append(trigger if trigger else "Reliability issue detected.")

    # Format multi-issue reasons
    if len(issues) == 1:
        body = issues[0]
    else:
        body = "\n".join(f"  • {issue}" for issue in issues)

    return f"{_REASON_REVIEW_HEADER}\n{body}"


def _reason_refer(dr: DRResult, policy: DecisionPolicy) -> str:
    """
    Build the referral reason.
    Names the predicted class and the configured threshold explicitly.
    """
    return _REASON_REFER_BASE.format(
        class_name=dr.class_name,
        grade=dr.grade,
        threshold=policy.referral.referable_grade_threshold,
    )


def _reason_routine(dr: DRResult, policy: DecisionPolicy) -> str:
    """
    Build the routine reason.
    Names the predicted class and confirms non-referable status.
    """
    return _REASON_ROUTINE.format(
        class_name=dr.class_name,
        grade=dr.grade,
    )


# ---------------------------------------------------------------------------
# Structured multi-reason collection (used when reporting multiple signals)
# ---------------------------------------------------------------------------

def collect_reliability_signals(reliability: ReliabilityResult) -> list[str]:
    """
    Collect all active reliability failure signals as a list of short strings.

    Useful for structured output / evidence fields.
    """
    signals = []
    if reliability.ood:
        signals.append(f"OOD detected (score={reliability.ood_score:.3f})")
    if reliability.uncertainty_level == "high":
        signals.append(f"high uncertainty ({reliability.uncertainty:.3f})")
    if reliability.confidence_level == "low":
        signals.append(f"low confidence ({reliability.confidence:.3f})")
    if reliability.is_review_required and not signals:
        signals.append(f"review required (status='{reliability.reliability_status}')")
    return signals
