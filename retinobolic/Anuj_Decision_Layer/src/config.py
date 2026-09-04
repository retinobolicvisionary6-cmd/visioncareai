"""
src/config.py — Decision Layer Configuration Loader.

Loads and exposes the YAML decision policy from config/decision_policy.yaml.

All thresholds are prototype engineering values only.
They have NOT been clinically validated.

PUBLIC API
----------
    load_policy(path=None) -> DecisionPolicy
    DEFAULT_POLICY         -> DecisionPolicy  (singleton, loaded at import time)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default policy file path
# ---------------------------------------------------------------------------
_DEFAULT_POLICY_PATH = Path(__file__).resolve().parent.parent / "config" / "decision_policy.yaml"


# ---------------------------------------------------------------------------
# Structured config objects (parsed from YAML)
# ---------------------------------------------------------------------------

@dataclass
class ReferralConfig:
    """DR referral threshold configuration."""
    referable_grade_threshold: int = 2           # Grades >= this are referable
    urgent_referral_grades: List[int] = field(default_factory=lambda: [3])

    def is_referable(self, grade: int) -> bool:
        """Return True if the given DR grade meets the referral threshold."""
        return grade >= self.referable_grade_threshold

    def is_urgent(self, grade: int) -> bool:
        """Return True if the grade warrants an urgent priority."""
        return grade in self.urgent_referral_grades


@dataclass
class PriorityConfig:
    """Workflow priority labels for each action."""
    recapture: str = "medium"
    doctor_review: str = "high"
    refer: str = "high"
    refer_urgent: str = "urgent"
    routine: str = "low"


@dataclass
class ReliabilityPolicyConfig:
    """Policy for how reliability status maps to action."""
    statuses_blocking_referral: List[str] = field(
        default_factory=lambda: ["review_required"]
    )
    caution_allows_referral: bool = False


@dataclass
class ClinicalEscalationRules:
    """Clinical context escalation rules (all disabled by default)."""
    high_hba1c_threshold: float = 9.0
    escalate_on_high_hba1c: bool = False
    age_high_risk_threshold: float = 70.0
    escalate_on_age_risk: bool = False
    long_diabetes_duration_threshold: float = 15.0
    escalate_on_long_duration: bool = False


@dataclass
class ClinicalContextConfig:
    """Clinical context policy."""
    enable_escalation: bool = False
    require_complete_context: bool = False
    escalation_rules: ClinicalEscalationRules = field(
        default_factory=ClinicalEscalationRules
    )


@dataclass
class GradcamConfig:
    """Grad-CAM / XAI policy."""
    require_gradcam_for_referral: bool = False
    warn_if_missing: bool = True


@dataclass
class EngineMetaConfig:
    """Engine versioning."""
    version: str = "1.0.0"


@dataclass
class DecisionPolicy:
    """
    Complete Decision Layer configuration.

    Loaded from config/decision_policy.yaml.
    All thresholds are prototype engineering values only.
    """
    referral: ReferralConfig = field(default_factory=ReferralConfig)
    priority: PriorityConfig = field(default_factory=PriorityConfig)
    reliability: ReliabilityPolicyConfig = field(default_factory=ReliabilityPolicyConfig)
    clinical_context: ClinicalContextConfig = field(default_factory=ClinicalContextConfig)
    gradcam: GradcamConfig = field(default_factory=GradcamConfig)
    engine: EngineMetaConfig = field(default_factory=EngineMetaConfig)

    def blocks_referral(self, reliability_status: Optional[str]) -> bool:
        """
        Return True if the reliability_status prevents referral.

        A None/missing status is treated conservatively as blocking.
        """
        if reliability_status is None:
            return True
        return reliability_status in self.reliability.statuses_blocking_referral

    def caution_blocks_referral(self) -> bool:
        """Return True if 'caution' status blocks referral (per policy)."""
        return not self.reliability.caution_allows_referral


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_policy(path: Optional[Union[str, Path]] = None) -> DecisionPolicy:
    """
    Load a DecisionPolicy from the YAML file.

    Parameters
    ----------
    path : str or Path or None
        Path to a YAML policy file.
        If None, loads from config/decision_policy.yaml (relative to project root).

    Returns
    -------
    DecisionPolicy — with values populated from YAML (or defaults if missing).

    Raises
    ------
    FileNotFoundError — if the file does not exist.
    ImportError      — if PyYAML is not installed.
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load the decision policy. "
            "Install it with: pip install pyyaml"
        ) from exc

    resolved = Path(path) if path is not None else _DEFAULT_POLICY_PATH

    if not resolved.exists():
        raise FileNotFoundError(
            f"Decision policy file not found: {resolved}\n"
            "Make sure config/decision_policy.yaml exists in the project root."
        )

    with open(resolved, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError(
            f"Decision policy YAML must be a mapping at the top level; "
            f"got {type(raw).__name__}."
        )

    policy = DecisionPolicy()

    # --- referral ---
    if "referral" in raw and isinstance(raw["referral"], dict):
        r = raw["referral"]
        if "referable_grade_threshold" in r:
            policy.referral.referable_grade_threshold = int(r["referable_grade_threshold"])
        if "urgent_referral_grades" in r:
            policy.referral.urgent_referral_grades = [
                int(g) for g in r["urgent_referral_grades"]
            ]

    # --- priority ---
    if "priority" in raw and isinstance(raw["priority"], dict):
        p = raw["priority"]
        for attr in ("recapture", "doctor_review", "refer", "refer_urgent", "routine"):
            if attr in p:
                setattr(policy.priority, attr, str(p[attr]))

    # --- reliability policy ---
    if "reliability" in raw and isinstance(raw["reliability"], dict):
        rel = raw["reliability"]
        if "statuses_blocking_referral" in rel:
            policy.reliability.statuses_blocking_referral = [
                str(s) for s in rel["statuses_blocking_referral"]
            ]
        if "caution_allows_referral" in rel:
            policy.reliability.caution_allows_referral = bool(rel["caution_allows_referral"])

    # --- clinical context ---
    if "clinical_context" in raw and isinstance(raw["clinical_context"], dict):
        cc = raw["clinical_context"]
        if "enable_escalation" in cc:
            policy.clinical_context.enable_escalation = bool(cc["enable_escalation"])
        if "require_complete_context" in cc:
            policy.clinical_context.require_complete_context = bool(
                cc["require_complete_context"]
            )
        if "escalation_rules" in cc and isinstance(cc["escalation_rules"], dict):
            er = cc["escalation_rules"]
            esc = policy.clinical_context.escalation_rules
            for attr, cast in [
                ("high_hba1c_threshold", float),
                ("age_high_risk_threshold", float),
                ("long_diabetes_duration_threshold", float),
            ]:
                if attr in er:
                    setattr(esc, attr, cast(er[attr]))
            for attr in (
                "escalate_on_high_hba1c",
                "escalate_on_age_risk",
                "escalate_on_long_duration",
            ):
                if attr in er:
                    setattr(esc, attr, bool(er[attr]))

    # --- gradcam ---
    if "gradcam" in raw and isinstance(raw["gradcam"], dict):
        gc = raw["gradcam"]
        if "require_gradcam_for_referral" in gc:
            policy.gradcam.require_gradcam_for_referral = bool(
                gc["require_gradcam_for_referral"]
            )
        if "warn_if_missing" in gc:
            policy.gradcam.warn_if_missing = bool(gc["warn_if_missing"])

    # --- engine meta ---
    if "engine" in raw and isinstance(raw["engine"], dict):
        em = raw["engine"]
        if "version" in em:
            policy.engine.version = str(em["version"])

    log.debug(
        "Decision policy loaded from %s (referral_threshold=%d, version=%s)",
        resolved,
        policy.referral.referable_grade_threshold,
        policy.engine.version,
    )
    return policy


# ---------------------------------------------------------------------------
# Default singleton
# ---------------------------------------------------------------------------
try:
    DEFAULT_POLICY: DecisionPolicy = load_policy()
except (FileNotFoundError, ImportError):
    log.warning(
        "Could not load decision_policy.yaml — using built-in default DecisionPolicy(). "
        "Ensure config/decision_policy.yaml exists and PyYAML is installed."
    )
    DEFAULT_POLICY = DecisionPolicy()
