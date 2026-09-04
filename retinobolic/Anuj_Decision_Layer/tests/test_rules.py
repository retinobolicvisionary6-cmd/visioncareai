"""
tests/test_rules.py — Unit tests for src/rules.py.

Tests each deterministic rule in isolation and the combined evaluate_decision_rules().
Verifies that the priority ordering is strictly maintained — a lower-priority rule
can NEVER override a higher-priority safety rule.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rules import (
    rule_1_image_safety,
    rule_2_reliability_safety,
    rule_3_referable_dr,
    rule_4_routine,
    evaluate_decision_rules,
    RuleResult,
)
from src.validation import (
    QualityResult,
    DRResult,
    ReliabilityResult,
    ClinicalContext,
)
from src.config import DecisionPolicy


# ============================================================================
# Helpers
# ============================================================================

def _quality(status: str, score: float = 0.85) -> QualityResult:
    action_map = {"good": "continue", "borderline": "enhance_and_recheck", "ungradable": "recapture"}
    return QualityResult(
        status=status,
        quality_score=score,
        action=action_map[status],
        reason="Test quality reason.",
    )


def _dr(grade: int) -> DRResult:
    base = [0.01, 0.01, 0.01, 0.01]
    base[grade] = 0.97
    probs = {str(i): p for i, p in enumerate(base)}
    return DRResult(grade=grade, probabilities=probs)


def _reliable(
    status: str = "acceptable",
    confidence_level: str = "high",
    uncertainty_level: str = "low",
    ood: bool = False,
    ood_score: float = 1.5,
) -> ReliabilityResult:
    conf_map = {"high": 0.92, "medium": 0.65, "low": 0.35}
    unc_map = {"low": 0.15, "medium": 0.52, "high": 0.78}
    return ReliabilityResult(
        reliability_status=status,
        review_required=(status == "review_required"),
        reason="Test reliability reason.",
        confidence=conf_map.get(confidence_level, 0.92),
        confidence_level=confidence_level,
        uncertainty=unc_map.get(uncertainty_level, 0.15),
        uncertainty_level=uncertainty_level,
        ood=ood,
        ood_status="review_required" if ood else "in_distribution",
        ood_score=ood_score,
    )


def _clinical(complete: bool = False) -> ClinicalContext:
    return ClinicalContext(
        validation_passed=True,
        clinical_context_complete=complete,
    )


# ============================================================================
# Rule 1 — Image Safety
# ============================================================================

class TestRule1ImageSafety:

    def test_ungradable_matches(self):
        r = rule_1_image_safety(_quality("ungradable"))
        assert r.matched is True
        assert r.action == "recapture"
        assert r.rule_name == "RULE_1_UNGRADABLE"

    def test_good_does_not_match(self):
        r = rule_1_image_safety(_quality("good"))
        assert r.matched is False

    def test_borderline_does_not_match(self):
        r = rule_1_image_safety(_quality("borderline"))
        assert r.matched is False


# ============================================================================
# Rule 2 — Reliability Safety
# ============================================================================

class TestRule2ReliabilitySafety:

    def test_ood_triggers_review(self):
        r = rule_2_reliability_safety(_reliable(status="review_required", ood=True))
        assert r.matched is True
        assert r.action == "doctor_review"
        assert "OOD" in r.trigger

    def test_high_uncertainty_triggers_review(self):
        r = rule_2_reliability_safety(_reliable(
            status="review_required", uncertainty_level="high"
        ))
        assert r.matched is True
        assert r.action == "doctor_review"
        assert "high model uncertainty" in r.trigger

    def test_low_confidence_triggers_review(self):
        r = rule_2_reliability_safety(_reliable(
            status="review_required", confidence_level="low"
        ))
        assert r.matched is True
        assert r.action == "doctor_review"
        assert "low model confidence" in r.trigger

    def test_review_required_status_triggers_review(self):
        r = rule_2_reliability_safety(_reliable(
            status="review_required", confidence_level="low"
        ))
        assert r.matched is True

    def test_acceptable_does_not_trigger(self):
        r = rule_2_reliability_safety(_reliable(
            status="acceptable", confidence_level="high", uncertainty_level="low", ood=False
        ))
        assert r.matched is False

    def test_caution_does_not_trigger(self):
        """Caution alone does NOT trigger doctor_review (policy configurable)."""
        r = rule_2_reliability_safety(_reliable(
            status="caution", confidence_level="medium", uncertainty_level="medium", ood=False
        ))
        assert r.matched is False

    def test_multiple_failures_all_reported_in_trigger(self):
        """Multiple simultaneous failures should all appear in the trigger string."""
        r = rule_2_reliability_safety(_reliable(
            status="review_required",
            confidence_level="low",
            uncertainty_level="high",
            ood=True,
        ))
        assert r.matched is True
        assert "OOD" in r.trigger
        assert "high model uncertainty" in r.trigger
        assert "low model confidence" in r.trigger

    def test_ood_overrides_high_confidence(self):
        """OOD=True must trigger review even when confidence is HIGH."""
        r = rule_2_reliability_safety(_reliable(
            status="review_required",
            confidence_level="high",
            uncertainty_level="low",
            ood=True,
        ))
        assert r.matched is True
        assert r.action == "doctor_review"


# ============================================================================
# Rule 3 — Referable DR
# ============================================================================

class TestRule3ReferableDR:

    def test_grade_2_reliable_refers(self):
        policy = DecisionPolicy()
        r = rule_3_referable_dr(
            _quality("good"), _dr(2), _reliable("acceptable"), policy
        )
        assert r.matched is True
        assert r.action == "refer"

    def test_grade_3_reliable_refers(self):
        policy = DecisionPolicy()
        r = rule_3_referable_dr(
            _quality("good"), _dr(3), _reliable("acceptable"), policy
        )
        assert r.matched is True
        assert r.action == "refer"

    def test_grade_1_reliable_does_not_refer(self):
        """Grade 1 is below default threshold of 2."""
        policy = DecisionPolicy()
        r = rule_3_referable_dr(
            _quality("good"), _dr(1), _reliable("acceptable"), policy
        )
        assert r.matched is False

    def test_grade_0_reliable_does_not_refer(self):
        policy = DecisionPolicy()
        r = rule_3_referable_dr(
            _quality("good"), _dr(0), _reliable("acceptable"), policy
        )
        assert r.matched is False

    def test_grade_2_review_required_does_not_refer(self):
        """review_required status must block referral."""
        policy = DecisionPolicy()
        r = rule_3_referable_dr(
            _quality("good"), _dr(2),
            _reliable("review_required", confidence_level="low"),
            policy,
        )
        assert r.matched is False

    def test_grade_2_caution_does_not_refer_by_default(self):
        """By default caution blocks referral (caution_allows_referral=False)."""
        policy = DecisionPolicy()
        assert policy.reliability.caution_allows_referral is False
        r = rule_3_referable_dr(
            _quality("good"), _dr(2),
            _reliable("caution", confidence_level="medium", uncertainty_level="medium"),
            policy,
        )
        assert r.matched is False

    def test_grade_2_caution_refers_when_policy_allows(self):
        """With caution_allows_referral=True, caution permits referral."""
        policy = DecisionPolicy()
        policy.reliability.caution_allows_referral = True
        r = rule_3_referable_dr(
            _quality("good"), _dr(2),
            _reliable("caution", confidence_level="medium", uncertainty_level="medium"),
            policy,
        )
        assert r.matched is True

    def test_ungradable_does_not_refer(self):
        policy = DecisionPolicy()
        r = rule_3_referable_dr(
            _quality("ungradable"), _dr(3), _reliable("acceptable"), policy
        )
        assert r.matched is False

    def test_configurable_threshold_grade_1(self):
        """When threshold is 1, Grade 1 should also refer."""
        policy = DecisionPolicy()
        policy.referral.referable_grade_threshold = 1
        r = rule_3_referable_dr(
            _quality("good"), _dr(1), _reliable("acceptable"), policy
        )
        assert r.matched is True


# ============================================================================
# Rule 4 — Routine
# ============================================================================

class TestRule4Routine:

    def test_always_matches(self):
        r = rule_4_routine(_quality("good"), _dr(0), _reliable("acceptable"))
        assert r.matched is True
        assert r.action == "routine"
        assert r.rule_name == "RULE_4_ROUTINE"


# ============================================================================
# evaluate_decision_rules — Combined priority order
# ============================================================================

class TestEvaluateDecisionRules:

    def test_rule1_beats_rule3(self):
        """Ungradable + Grade 3 + Reliable → recapture (Rule 1 wins)."""
        result = evaluate_decision_rules(
            _quality("ungradable"),
            _dr(3),
            _reliable("acceptable"),
            _clinical(),
        )
        assert result.action == "recapture"
        assert result.rule_name == "RULE_1_UNGRADABLE"

    def test_rule2_beats_rule3(self):
        """Grade 3 + OOD=True → doctor_review (Rule 2 wins over Rule 3)."""
        result = evaluate_decision_rules(
            _quality("good"),
            _dr(3),
            _reliable("review_required", ood=True),
            _clinical(),
        )
        assert result.action == "doctor_review"
        assert result.rule_name == "RULE_2_RELIABILITY_FAILURE"

    def test_rule3_fires_when_rules_1_2_clear(self):
        """Grade 2 + Reliable + Good quality → refer (Rule 3)."""
        result = evaluate_decision_rules(
            _quality("good"),
            _dr(2),
            _reliable("acceptable"),
            _clinical(),
        )
        assert result.action == "refer"
        assert result.rule_name == "RULE_3_REFERABLE_DR"

    def test_rule4_fires_as_fallthrough(self):
        """Grade 0 + Reliable + Good quality → routine (Rule 4)."""
        result = evaluate_decision_rules(
            _quality("good"),
            _dr(0),
            _reliable("acceptable"),
            _clinical(),
        )
        assert result.action == "routine"
        assert result.rule_name == "RULE_4_ROUTINE"

    def test_rule1_beats_rule2(self):
        """Ungradable + OOD=True → recapture (Rule 1 still wins)."""
        result = evaluate_decision_rules(
            _quality("ungradable"),
            _dr(1),
            _reliable("review_required", ood=True),
            _clinical(),
        )
        assert result.action == "recapture"

    def test_grade_3_high_uncertainty_doctor_review(self):
        """Grade 3 + High uncertainty → doctor_review (Rule 2 beats Rule 3)."""
        result = evaluate_decision_rules(
            _quality("good"),
            _dr(3),
            _reliable("review_required", uncertainty_level="high"),
            _clinical(),
        )
        assert result.action == "doctor_review"
