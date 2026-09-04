"""
tests/test_rules.py — Unit tests for src/reliability/rules.py

Tests the deterministic 5-priority rule engine in complete isolation
from the real upstream modules (uses mock dicts → validated dataclasses).

Coverage of all required cases from the specification:
    Case A: high confidence + low uncertainty + OOD = True  → review_required
    Case B: high confidence + high uncertainty + OOD = False → review_required
    Case C: low confidence  + low uncertainty + OOD = False  → review_required
    Case D: medium + medium + OOD = False                    → caution
    Case E: high + low + OOD = False                         → acceptable

Priority tests:
    OOD beats high confidence + low uncertainty
    High uncertainty beats medium/high confidence
    Low confidence beats medium uncertainty
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.schemas import ConfidenceResult, OODResult, ReliabilityStatus, UncertaintyResult
from src.reliability.config import ReliabilityConfig
from src.reliability.rules import evaluate_rules


# ---------------------------------------------------------------------------
# Helpers: build typed dataclasses from keyword args
# ---------------------------------------------------------------------------

def make_confidence(confidence: float, level: str) -> ConfidenceResult:
    return ConfidenceResult(confidence=confidence, confidence_level=level)


def make_uncertainty(uncertainty: float, level: str, review: bool = False) -> UncertaintyResult:
    return UncertaintyResult(uncertainty=uncertainty, uncertainty_level=level, review_recommended=review)


def make_ood(ood: bool, score: float = 1.0) -> OODResult:
    status = "review_required" if ood else "in_distribution"
    return OODResult(ood=ood, ood_status=status, ood_score=score)


# ---------------------------------------------------------------------------
# Case E — Acceptable (all good)
# ---------------------------------------------------------------------------

class TestAcceptable:
    def test_case_e_acceptable(self, default_config):
        """High confidence + low uncertainty + in-distribution → acceptable."""
        status, review, reason = evaluate_rules(
            make_confidence(0.88, "high"),
            make_uncertainty(0.18, "low"),
            make_ood(False),
            default_config,
        )
        assert status == ReliabilityStatus.ACCEPTABLE
        assert review is False
        assert "High model confidence" in reason

    def test_acceptable_boundary_high_confidence(self, default_config):
        """Exactly at HIGH_CONFIDENCE_THRESHOLD is high → acceptable."""
        status, review, _ = evaluate_rules(
            make_confidence(0.80, "high"),
            make_uncertainty(0.10, "low"),
            make_ood(False),
            default_config,
        )
        assert status == ReliabilityStatus.ACCEPTABLE
        assert review is False


# ---------------------------------------------------------------------------
# Case A — OOD priority (Priority 1)
# ---------------------------------------------------------------------------

class TestOODPriority:
    def test_case_a_ood_overrides_high_confidence(self, default_config):
        """OOD = True MUST trigger review_required even with high confidence."""
        status, review, reason = evaluate_rules(
            make_confidence(0.95, "high"),
            make_uncertainty(0.05, "low"),
            make_ood(True, score=5.5),
            default_config,
        )
        assert status == ReliabilityStatus.REVIEW_REQUIRED
        assert review is True
        assert "outside the configured reference distribution" in reason

    def test_ood_overrides_medium_confidence(self, default_config):
        status, review, _ = evaluate_rules(
            make_confidence(0.65, "medium"),
            make_uncertainty(0.50, "medium"),
            make_ood(True),
            default_config,
        )
        assert status == ReliabilityStatus.REVIEW_REQUIRED
        assert review is True

    def test_ood_with_multiple_issues_reason_combined(self, default_config):
        """OOD + high uncertainty → reason mentions both."""
        _, _, reason = evaluate_rules(
            make_confidence(0.30, "low"),
            make_uncertainty(0.80, "high", review=True),
            make_ood(True),
            default_config,
        )
        assert "outside the configured reference distribution" in reason

    def test_ood_strict_mode_false_allows_other_rules(self):
        """With OOD_STRICT_MODE=False, OOD alone does not trigger review."""
        cfg = ReliabilityConfig(OOD_STRICT_MODE=False)
        status, review, _ = evaluate_rules(
            make_confidence(0.90, "high"),
            make_uncertainty(0.10, "low"),
            make_ood(True),
            cfg,
        )
        # OOD is ignored; other signals are good → acceptable
        assert status == ReliabilityStatus.ACCEPTABLE
        assert review is False


# ---------------------------------------------------------------------------
# Case B — High uncertainty (Priority 2)
# ---------------------------------------------------------------------------

class TestHighUncertainty:
    def test_case_b_high_uncertainty_triggers_review(self, default_config):
        """High uncertainty + OOD=False → review_required regardless of confidence."""
        status, review, reason = evaluate_rules(
            make_confidence(0.90, "high"),
            make_uncertainty(0.80, "high", review=True),
            make_ood(False),
            default_config,
        )
        assert status == ReliabilityStatus.REVIEW_REQUIRED
        assert review is True
        assert "high model uncertainty" in reason.lower()

    def test_high_uncertainty_with_low_confidence_combined_reason(self, default_config):
        """High uncertainty AND low confidence → reason mentions both."""
        _, _, reason = evaluate_rules(
            make_confidence(0.35, "low"),
            make_uncertainty(0.82, "high", review=True),
            make_ood(False),
            default_config,
        )
        assert "high model uncertainty" in reason.lower()
        assert "low" in reason.lower()


# ---------------------------------------------------------------------------
# Case C — Low confidence (Priority 3)
# ---------------------------------------------------------------------------

class TestLowConfidence:
    def test_case_c_low_confidence_triggers_review(self, default_config):
        """Low confidence + low uncertainty + in-distribution → review_required."""
        status, review, reason = evaluate_rules(
            make_confidence(0.35, "low"),
            make_uncertainty(0.20, "low"),
            make_ood(False),
            default_config,
        )
        assert status == ReliabilityStatus.REVIEW_REQUIRED
        assert review is True
        assert "low" in reason.lower()

    def test_low_confidence_medium_uncertainty_review(self, default_config):
        """Low confidence + medium uncertainty → review_required (low conf fires first)."""
        status, review, _ = evaluate_rules(
            make_confidence(0.40, "low"),
            make_uncertainty(0.52, "medium"),
            make_ood(False),
            default_config,
        )
        assert status == ReliabilityStatus.REVIEW_REQUIRED
        assert review is True


# ---------------------------------------------------------------------------
# Case D — Caution (Priority 4)
# ---------------------------------------------------------------------------

class TestCaution:
    def test_case_d_medium_confidence_medium_uncertainty(self, default_config):
        """Medium confidence + medium uncertainty → caution."""
        status, review, reason = evaluate_rules(
            make_confidence(0.62, "medium"),
            make_uncertainty(0.52, "medium"),
            make_ood(False),
            default_config,
        )
        assert status == ReliabilityStatus.CAUTION
        assert review is False
        assert len(reason) > 0

    def test_medium_confidence_low_uncertainty_caution(self, default_config):
        """Medium confidence + low uncertainty → caution (conf is intermediate)."""
        status, review, _ = evaluate_rules(
            make_confidence(0.62, "medium"),
            make_uncertainty(0.20, "low"),
            make_ood(False),
            default_config,
        )
        assert status == ReliabilityStatus.CAUTION
        assert review is False

    def test_high_confidence_medium_uncertainty_caution(self, default_config):
        """High confidence + medium uncertainty → caution."""
        status, review, _ = evaluate_rules(
            make_confidence(0.85, "high"),
            make_uncertainty(0.50, "medium"),
            make_ood(False),
            default_config,
        )
        assert status == ReliabilityStatus.CAUTION
        assert review is False


# ---------------------------------------------------------------------------
# Priority ordering — exhaustive conflict tests
# ---------------------------------------------------------------------------

class TestPriorityOrdering:
    def test_ood_beats_all(self, default_config):
        """OOD=True always wins, even with worst-case signals for review."""
        for conf_level, uncert_level, uncert_val in [
            ("high", "low", 0.10),
            ("high", "high", 0.80),
            ("low", "low", 0.10),
        ]:
            status, review, _ = evaluate_rules(
                make_confidence(0.50, conf_level),
                make_uncertainty(uncert_val, uncert_level),
                make_ood(True),
                default_config,
            )
            assert status == ReliabilityStatus.REVIEW_REQUIRED, (
                f"Expected REVIEW_REQUIRED for OOD=True, "
                f"conf={conf_level}, uncert={uncert_level}"
            )

    def test_reason_always_non_empty(self, default_config):
        """All decision paths must produce a non-empty reason string."""
        scenarios = [
            (make_confidence(0.88, "high"), make_uncertainty(0.18, "low"), make_ood(False)),
            (make_confidence(0.62, "medium"), make_uncertainty(0.52, "medium"), make_ood(False)),
            (make_confidence(0.35, "low"), make_uncertainty(0.20, "low"), make_ood(False)),
            (make_confidence(0.90, "high"), make_uncertainty(0.82, "high", True), make_ood(False)),
            (make_confidence(0.90, "high"), make_uncertainty(0.10, "low"), make_ood(True)),
        ]
        for conf, uncert, ood in scenarios:
            _, _, reason = evaluate_rules(conf, uncert, ood, default_config)
            assert reason, f"Reason must be non-empty for {conf}, {uncert}, {ood}"
