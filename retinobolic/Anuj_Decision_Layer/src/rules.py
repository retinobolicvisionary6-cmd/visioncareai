"""
src/rules.py — Deterministic Decision Rules for the Final Decision Layer.

Implements the 4-rule safety hierarchy:

    RULE 1 — Image Safety
        quality.status == "ungradable" → action = "recapture"

    RULE 2 — Reliability / Safety Failure
        ood == True OR uncertainty_level == "high" OR
        confidence_level == "low" OR reliability_status == "review_required"
        → action = "doctor_review"

    RULE 3 — Reliable Referable DR
        quality is gradable AND reliability is acceptable/caution-within-policy
        AND dr_grade >= referable_grade_threshold
        → action = "refer"

    RULE 4 — Routine
        All other gradable, reliable, non-referable cases
        → action = "routine"

DESIGN PRINCIPLES
-----------------
- Each rule is a separate named function — testable in isolation.
- No rule can override a higher-priority safety rule.
- No ML model, LLM, or external API is used.
- Clinical context does NOT alter DR grade or override these rules
  (it only informs priority via the priority module).
- All rule functions return (matched: bool, rule_name: str, action: str).
"""

from __future__ import annotations

import logging
from typing import NamedTuple, Optional

from .config import DecisionPolicy, DEFAULT_POLICY
from .validation import QualityResult, DRResult, ReliabilityResult, ClinicalContext

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rule result
# ---------------------------------------------------------------------------

class RuleResult(NamedTuple):
    """
    Result from a single rule evaluation.

    Attributes
    ----------
    matched   : bool   — True if this rule fired
    rule_name : str    — Identifier for the rule that fired (e.g. "RULE_1_UNGRADABLE")
    action    : str    — Workflow action ("recapture", "doctor_review", "refer", "routine")
    trigger   : str    — Human-readable description of what triggered this rule
    """
    matched: bool
    rule_name: str
    action: str
    trigger: str


# ---------------------------------------------------------------------------
# Rule implementations
# ---------------------------------------------------------------------------

def rule_1_image_safety(
    quality: QualityResult,
) -> RuleResult:
    """
    RULE 1 — Image Safety Gate (HIGHEST PRIORITY)

    If the image is ungradable, the workflow must recapture.
    No downstream signal — DR grade, confidence, OOD — can override this.

    Conflict examples:
        quality=ungradable, DR Grade=3, confidence=0.99 → recapture
        quality=ungradable, OOD=False, uncertainty=low   → recapture

    Returns
    -------
    RuleResult with matched=True if quality.status == "ungradable".
    """
    if quality.is_ungradable:
        log.debug("RULE 1 fired: quality.status='%s'", quality.status)
        return RuleResult(
            matched=True,
            rule_name="RULE_1_UNGRADABLE",
            action="recapture",
            trigger=f"Image quality status is '{quality.status}'.",
        )
    return RuleResult(matched=False, rule_name="RULE_1_UNGRADABLE", action="", trigger="")


def rule_2_reliability_safety(
    reliability: ReliabilityResult,
) -> RuleResult:
    """
    RULE 2 — Reliability / Safety Failure Gate

    Triggers doctor_review if ANY of the following is true:
        - reliability_status == "review_required"
        - ood == True
        - uncertainty_level == "high"
        - confidence_level == "low"

    The rule collects ALL triggered conditions for a transparent reason.

    This rule prevents automatic referral based on an unreliable model output.

    Returns
    -------
    RuleResult with matched=True if any reliability failure is detected.
    """
    triggers = []

    if reliability.ood:
        triggers.append(f"OOD detected (score={reliability.ood_score:.3f})")
    if reliability.uncertainty_level == "high":
        triggers.append(f"high model uncertainty ({reliability.uncertainty:.3f})")
    if reliability.confidence_level == "low":
        triggers.append(f"low model confidence ({reliability.confidence:.3f})")
    if reliability.is_review_required and not triggers:
        # review_required from reliability engine for a reason not captured above
        triggers.append(
            f"reliability engine flagged review required "
            f"(status='{reliability.reliability_status}')"
        )

    if triggers:
        trigger_str = "; ".join(triggers)
        log.debug("RULE 2 fired: %s", trigger_str)
        return RuleResult(
            matched=True,
            rule_name="RULE_2_RELIABILITY_FAILURE",
            action="doctor_review",
            trigger=trigger_str,
        )

    return RuleResult(
        matched=False, rule_name="RULE_2_RELIABILITY_FAILURE", action="", trigger=""
    )


def rule_3_referable_dr(
    quality: QualityResult,
    dr: DRResult,
    reliability: ReliabilityResult,
    policy: DecisionPolicy = DEFAULT_POLICY,
) -> RuleResult:
    """
    RULE 3 — Reliable Referable DR

    Triggers referral if ALL of the following hold:
        - quality is gradable (good or borderline)
        - reliability does NOT block referral (per policy)
        - dr.grade >= policy.referral.referable_grade_threshold

    IMPORTANT:
        This rule only runs after Rules 1 and 2 have NOT fired.
        Do NOT call this rule if Rule 2 matched — reliability failure
        must route to doctor_review, not referral.

    Returns
    -------
    RuleResult with matched=True if referral criteria are met.
    """
    if not quality.is_gradable:
        # Should not reach here (Rule 1 handles ungradable) but defensive check
        return RuleResult(
            matched=False, rule_name="RULE_3_REFERABLE_DR", action="", trigger=""
        )

    # Reliability status must permit referral
    if policy.blocks_referral(reliability.reliability_status):
        log.debug(
            "RULE 3 skipped: reliability_status='%s' blocks referral per policy.",
            reliability.reliability_status,
        )
        return RuleResult(
            matched=False, rule_name="RULE_3_REFERABLE_DR", action="", trigger=""
        )

    # Also honour caution policy
    if reliability.is_caution and policy.caution_blocks_referral():
        log.debug(
            "RULE 3 skipped: reliability_status='caution' and caution_allows_referral=False."
        )
        return RuleResult(
            matched=False, rule_name="RULE_3_REFERABLE_DR", action="", trigger=""
        )

    # DR grade threshold check
    if policy.referral.is_referable(dr.grade):
        trigger = (
            f"DR Grade {dr.grade} ({dr.class_name}) meets configured referral "
            f"threshold (>= {policy.referral.referable_grade_threshold})."
        )
        log.debug("RULE 3 fired: %s", trigger)
        return RuleResult(
            matched=True,
            rule_name="RULE_3_REFERABLE_DR",
            action="refer",
            trigger=trigger,
        )

    return RuleResult(
        matched=False, rule_name="RULE_3_REFERABLE_DR", action="", trigger=""
    )


def rule_4_routine(
    quality: QualityResult,
    dr: DRResult,
    reliability: ReliabilityResult,
) -> RuleResult:
    """
    RULE 4 — Routine / Default

    The default safe fallthrough for gradable, reliable, non-referable cases.
    Only reached when Rules 1, 2, and 3 have all NOT fired.

    Returns
    -------
    RuleResult with matched=True always (this is the safe default).
    """
    trigger = (
        f"Image is gradable (status='{quality.status}'), "
        f"reliability is acceptable (status='{reliability.reliability_status}'), "
        f"and DR Grade {dr.grade} ({dr.class_name}) is below the configured "
        "referral threshold."
    )
    log.debug("RULE 4 fired: %s", trigger)
    return RuleResult(
        matched=True,
        rule_name="RULE_4_ROUTINE",
        action="routine",
        trigger=trigger,
    )


# ---------------------------------------------------------------------------
# Rule evaluator — applies rules in priority order
# ---------------------------------------------------------------------------

def evaluate_decision_rules(
    quality: QualityResult,
    dr: DRResult,
    reliability: ReliabilityResult,
    clinical: ClinicalContext,
    policy: DecisionPolicy = DEFAULT_POLICY,
) -> RuleResult:
    """
    Evaluate all decision rules in strict priority order and return the
    first matching result.

    Priority order:
        Rule 1 (Safety: ungradable)         → highest priority
        Rule 2 (Reliability failure)
        Rule 3 (Reliable referable DR)
        Rule 4 (Routine / fallthrough)      → lowest priority

    A lower-priority rule can NEVER override a higher-priority safety rule.

    Parameters
    ----------
    quality     : QualityResult     — validated quality output
    dr          : DRResult          — validated DR model output
    reliability : ReliabilityResult — validated reliability output
    clinical    : ClinicalContext   — validated clinical context (informational)
    policy      : DecisionPolicy    — configurable decision policy

    Returns
    -------
    RuleResult — the winning rule's matched result
    """
    # RULE 1: Image safety
    r1 = rule_1_image_safety(quality)
    if r1.matched:
        return r1

    # RULE 2: Reliability safety
    r2 = rule_2_reliability_safety(reliability)
    if r2.matched:
        return r2

    # RULE 3: Reliable referable DR
    r3 = rule_3_referable_dr(quality, dr, reliability, policy)
    if r3.matched:
        return r3

    # RULE 4: Routine fallthrough (always matches)
    r4 = rule_4_routine(quality, dr, reliability)
    return r4
