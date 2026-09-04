"""
src/priorities.py — Priority Mapper for the Final Decision Layer.

Maps a workflow action and supporting context to a priority label:
    low | medium | high | urgent

DESIGN PRINCIPLES
-----------------
- Priority labels are workflow labels, NOT clinical triage levels.
- All thresholds are configurable via DecisionPolicy.
- Clinical context may influence priority ONLY when escalation is
  explicitly enabled in policy (disabled by default for MVP v1.0).
- Missing clinical context NEVER automatically creates "urgent" priority.
- DR Grade 3 (Severe/PDR) escalates referral to "urgent" by default
  (configurable via policy.referral.urgent_referral_grades).

PUBLIC API
----------
    determine_priority(
        action,
        dr,
        reliability,
        clinical,
        policy
    ) -> str   # "low" | "medium" | "high" | "urgent"
"""

from __future__ import annotations

import logging
from typing import Optional

from .config import DecisionPolicy, DEFAULT_POLICY
from .validation import DRResult, ReliabilityResult, ClinicalContext

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valid priority levels
# ---------------------------------------------------------------------------

VALID_PRIORITIES = {"low", "medium", "high", "urgent"}


def determine_priority(
    action: str,
    dr: DRResult,
    reliability: ReliabilityResult,
    clinical: ClinicalContext,
    policy: DecisionPolicy = DEFAULT_POLICY,
) -> str:
    """
    Determine the workflow priority for the given action.

    Parameters
    ----------
    action      : str               — "recapture" | "doctor_review" | "refer" | "routine"
    dr          : DRResult          — validated DR model output
    reliability : ReliabilityResult — validated reliability output
    clinical    : ClinicalContext   — validated clinical context (may be incomplete)
    policy      : DecisionPolicy    — configurable decision policy

    Returns
    -------
    str — one of: "low", "medium", "high", "urgent"

    Priority Decision Logic
    -----------------------
    RECAPTURE:
        Base: policy.priority.recapture (default: "medium")

    DOCTOR_REVIEW:
        Base: policy.priority.doctor_review (default: "high")
        Clinical escalation: only when policy.clinical_context.enable_escalation=True

    REFER:
        Base: policy.priority.refer (default: "high")
        Escalated to policy.priority.refer_urgent (default: "urgent")
            if dr.grade is in policy.referral.urgent_referral_grades
        Clinical escalation: only when enable_escalation=True

    ROUTINE:
        Base: policy.priority.routine (default: "low")
        Clinical escalation: only when enable_escalation=True

    SAFETY NOTE:
        Priority labels are workflow routing labels only.
        They are NOT validated clinical triage levels.
    """
    if action == "recapture":
        priority = policy.priority.recapture
        log.debug("Priority for 'recapture': %s", priority)
        return priority

    if action == "refer":
        # Base referral priority
        if policy.referral.is_urgent(dr.grade):
            priority = policy.priority.refer_urgent
            log.debug(
                "Priority for 'refer' escalated to '%s': grade=%d is in urgent_referral_grades.",
                priority, dr.grade,
            )
        else:
            priority = policy.priority.refer
            log.debug("Priority for 'refer': %s (grade=%d)", priority, dr.grade)

        # Optional clinical escalation (disabled by default)
        if policy.clinical_context.enable_escalation:
            escalated = _apply_clinical_escalation(priority, clinical, policy)
            if escalated != priority:
                log.info(
                    "Clinical context escalated priority from '%s' to '%s' for 'refer' action.",
                    priority, escalated,
                )
            priority = escalated

        return priority

    if action == "doctor_review":
        priority = policy.priority.doctor_review
        log.debug("Priority for 'doctor_review': %s", priority)

        # Optional clinical escalation (disabled by default)
        if policy.clinical_context.enable_escalation:
            escalated = _apply_clinical_escalation(priority, clinical, policy)
            if escalated != priority:
                log.info(
                    "Clinical context escalated priority from '%s' to '%s' "
                    "for 'doctor_review' action.",
                    priority, escalated,
                )
            priority = escalated

        return priority

    if action == "routine":
        priority = policy.priority.routine
        log.debug("Priority for 'routine': %s", priority)

        # Optional clinical escalation (disabled by default)
        if policy.clinical_context.enable_escalation:
            escalated = _apply_clinical_escalation(priority, clinical, policy)
            if escalated != priority:
                log.info(
                    "Clinical context escalated priority from '%s' to '%s' "
                    "for 'routine' action.",
                    priority, escalated,
                )
            priority = escalated

        return priority

    # Unknown action — default to "medium" as a conservative fallback
    log.warning("Unknown action '%s' in priority mapper — defaulting to 'medium'.", action)
    return "medium"


def _apply_clinical_escalation(
    current_priority: str,
    clinical: ClinicalContext,
    policy: DecisionPolicy,
) -> str:
    """
    Apply clinical context escalation rules to potentially raise priority.

    IMPORTANT ARCHITECTURE RULE:
        - This function NEVER lowers priority.
        - It only escalates when:
            a) escalation is explicitly enabled in policy
            b) clinical data is available (not just missing)
            c) a specific configured threshold is exceeded

    Priority escalation order: low → medium → high → urgent

    Parameters
    ----------
    current_priority : str
    clinical         : ClinicalContext
    policy           : DecisionPolicy

    Returns
    -------
    str — the possibly-escalated priority (never lower than current)
    """
    rules = policy.clinical_context.escalation_rules

    # If clinical context is not available, do NOT escalate
    if not clinical.clinical_context_complete or clinical.clinical_data is None:
        log.debug("Clinical escalation skipped: clinical_context_complete=False.")
        return current_priority

    escalate_to: Optional[str] = None

    # HbA1c escalation
    if rules.escalate_on_high_hba1c:
        hba1c = clinical.hba1c
        if hba1c is not None and hba1c >= rules.high_hba1c_threshold:
            log.info(
                "Clinical escalation: HbA1c=%.1f >= threshold=%.1f.",
                hba1c, rules.high_hba1c_threshold,
            )
            escalate_to = _higher_priority(escalate_to or current_priority, "high")

    # Age escalation
    if rules.escalate_on_age_risk:
        age = clinical.age
        if age is not None and age >= rules.age_high_risk_threshold:
            log.info(
                "Clinical escalation: age=%.0f >= threshold=%.0f.",
                age, rules.age_high_risk_threshold,
            )
            escalate_to = _higher_priority(escalate_to or current_priority, "high")

    # Diabetes duration escalation
    if rules.escalate_on_long_duration:
        duration = clinical.diabetes_duration_years
        if duration is not None and duration >= rules.long_diabetes_duration_threshold:
            log.info(
                "Clinical escalation: diabetes_duration=%.1f >= threshold=%.1f.",
                duration, rules.long_diabetes_duration_threshold,
            )
            escalate_to = _higher_priority(escalate_to or current_priority, "high")

    return escalate_to if escalate_to is not None else current_priority


_PRIORITY_ORDER = ["low", "medium", "high", "urgent"]


def _higher_priority(a: str, b: str) -> str:
    """Return the higher of two priority strings."""
    a_idx = _PRIORITY_ORDER.index(a) if a in _PRIORITY_ORDER else 1
    b_idx = _PRIORITY_ORDER.index(b) if b in _PRIORITY_ORDER else 1
    return _PRIORITY_ORDER[max(a_idx, b_idx)]
