"""
src/engine.py — Final Decision Layer Engine.

The main orchestration entry point for the Final Decision Layer.

PUBLIC API
----------
    make_final_decision(
        quality_result,
        dr_result,
        reliability_result,
        clinical_context=None,
        gradcam_path=None,
        policy=None,
        include_evidence=True,
    ) -> dict

    run_screening_decision(
        quality_result,
        dr_result,
        reliability_result,
        clinical_context=None,
        gradcam_path=None,
        policy=None,
        include_evidence=True,
    ) -> dict

PIPELINE (for make_final_decision)
-----------------------------------
    1. Validate all upstream inputs
    2. Apply decision rules (Rule 1 → Rule 2 → Rule 3 → Rule 4)
    3. Determine priority (configurable per policy)
    4. Generate reason (structured, non-diagnostic)
    5. Assemble and return stable JSON-compatible output

DESIGN RULES
------------
- This module does NOT recalculate any upstream signal.
- It does NOT contain any ML model, LLM, or deep learning code.
- It does NOT produce diagnostic statements.
- All clinical decisions remain with the examining physician.
- The output is workflow-routing information only.

SAFETY NOTE
-----------
This engine produces WORKFLOW ROUTING DECISIONS ONLY.
It is NOT a medical diagnostic tool.
The final clinical decision rests entirely with the examining physician.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .config import DecisionPolicy, DEFAULT_POLICY
from .validation import (
    validate_all_inputs,
    ValidatedInputs,
    DecisionValidationError,
)
from .rules import evaluate_decision_rules
from .priorities import determine_priority
from .reasons import generate_reason, collect_reliability_signals

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Engine version (matches config/decision_policy.yaml)
# ---------------------------------------------------------------------------
_ENGINE_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Output assembly
# ---------------------------------------------------------------------------

def _assemble_output(
    action: str,
    priority: str,
    reason: str,
    rule_name: str,
    inputs: ValidatedInputs,
    policy: DecisionPolicy,
    include_evidence: bool,
) -> Dict[str, Any]:
    """
    Assemble the stable JSON-compatible output dictionary.

    Core contract (always present):
        action            : str   "recapture" | "doctor_review" | "refer" | "routine"
        priority          : str   "low" | "medium" | "high" | "urgent"
        reason            : str   Clear, non-diagnostic workflow reason
        dr_grade          : int | None
        reliability_status: str | None
        review_required   : bool

    Optional evidence block (when include_evidence=True):
        evidence:
            quality_status          : str
            quality_score           : float
            confidence              : float
            confidence_level        : str
            uncertainty             : float
            uncertainty_level       : str
            ood                     : bool
            ood_score               : float
            reliability_score       : float | None
            gradcam_path            : str | None
            clinical_context_complete: bool
            reliability_signals     : list[str]

    Metadata block (always present):
        metadata:
            rule_applied            : str
            engine_version          : str
    """
    # Core fields
    output: Dict[str, Any] = {
        "action": action,
        "priority": priority,
        "reason": reason,
        "review_required": action == "doctor_review",
    }

    # DR grade — None for recapture (image not gradable)
    if action == "recapture":
        output["dr_grade"] = None
        output["reliability_status"] = None
    else:
        output["dr_grade"] = inputs.dr.grade
        output["reliability_status"] = inputs.reliability.reliability_status

    # Evidence block
    if include_evidence:
        reliability_signals = collect_reliability_signals(inputs.reliability)
        output["evidence"] = {
            "quality_status": inputs.quality.status,
            "quality_score": inputs.quality.quality_score,
            "confidence": inputs.reliability.confidence,
            "confidence_level": inputs.reliability.confidence_level,
            "uncertainty": inputs.reliability.uncertainty,
            "uncertainty_level": inputs.reliability.uncertainty_level,
            "ood": inputs.reliability.ood,
            "ood_score": inputs.reliability.ood_score,
            "reliability_score": inputs.reliability.reliability_score,
            "gradcam_path": inputs.gradcam.gradcam_path,
            "clinical_context_complete": inputs.clinical.clinical_context_complete,
            "reliability_signals": reliability_signals,
        }

    # Metadata
    output["metadata"] = {
        "rule_applied": rule_name,
        "engine_version": _ENGINE_VERSION,
    }

    return output


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_final_decision(
    quality_result: Any,
    dr_result: Any,
    reliability_result: Any,
    clinical_context: Optional[Any] = None,
    gradcam_path: Optional[str] = None,
    policy: Optional[DecisionPolicy] = None,
    include_evidence: bool = True,
) -> Dict[str, Any]:
    """
    Make the final screening workflow decision.

    This is the primary public entry point for the Decision Layer.

    Parameters
    ----------
    quality_result      : dict
        Output from anuj-fundus-quality's assess_quality().
        Required keys: "status", "quality_score".

    dr_result           : dict
        Output from Vinayak's 4-class DR model.
        Required keys: "grade", "probabilities".
        Optional: "gradcam_path".

        NOTE: Vinayak's DR model + Grad-CAM module is not yet integrated.
        This function accepts the agreed output contract so integration
        will be seamless when the module is ready.

    reliability_result  : dict
        Output from anuj-reliability's calculate_reliability()
        or run_reliability_pipeline().
        Required keys: "reliability_status", "review_required", "confidence",
                       "confidence_level", "uncertainty", "uncertainty_level",
                       "ood", "ood_status", "ood_score", "reason".

    clinical_context    : dict | None (optional)
        Output from anuj-clinical-context's process_clinical_context().
        Missing or None → workflow proceeds without clinical context.
        Missing context does NOT block the basic DR/reliability workflow.

    gradcam_path        : str | None (optional)
        Explicit path to the Grad-CAM output (overrides dr_result["gradcam_path"]).
        Grad-CAM is preserved in evidence but NOT interpreted as lesion localization.

    policy              : DecisionPolicy | None (optional)
        Decision policy. Uses DEFAULT_POLICY (from decision_policy.yaml) if None.

    include_evidence    : bool (default True)
        Include the upstream evidence signals in the output.
        Set False for minimal output (e.g., in production where evidence is logged
        separately).

    Returns
    -------
    dict — JSON-compatible output with keys:
        action            : str   "recapture" | "doctor_review" | "refer" | "routine"
        priority          : str   "low" | "medium" | "high" | "urgent"
        reason            : str
        dr_grade          : int | None
        reliability_status: str | None
        review_required   : bool
        evidence          : dict  (if include_evidence=True)
        metadata          : dict

    Raises
    ------
    DecisionValidationError — if any required input is structurally invalid.

    Safety Note
    -----------
    This function produces workflow-routing decisions ONLY.
    It is NOT a medical diagnostic tool.
    The final clinical decision rests with the examining physician.
    """
    cfg = policy if policy is not None else DEFAULT_POLICY

    log.info(
        "make_final_decision called | policy_version=%s | "
        "clinical_context_provided=%s | gradcam_path=%s",
        cfg.engine.version,
        clinical_context is not None,
        gradcam_path,
    )

    # -------------------------------------------------------------------
    # STEP 1: Validate all upstream inputs
    # -------------------------------------------------------------------
    try:
        inputs = validate_all_inputs(
            quality_result=quality_result,
            dr_result=dr_result,
            reliability_result=reliability_result,
            clinical_context=clinical_context,
            gradcam_path=gradcam_path,
        )
    except DecisionValidationError:
        raise  # Re-raise with original message

    log.debug(
        "Inputs validated | quality=%s | dr_grade=%d | reliability=%s | "
        "ood=%s | confidence_level=%s | uncertainty_level=%s",
        inputs.quality.status,
        inputs.dr.grade,
        inputs.reliability.reliability_status,
        inputs.reliability.ood,
        inputs.reliability.confidence_level,
        inputs.reliability.uncertainty_level,
    )

    # -------------------------------------------------------------------
    # STEP 2: Apply decision rules (priority order enforced)
    # -------------------------------------------------------------------
    rule_result = evaluate_decision_rules(
        quality=inputs.quality,
        dr=inputs.dr,
        reliability=inputs.reliability,
        clinical=inputs.clinical,
        policy=cfg,
    )

    action = rule_result.action
    rule_name = rule_result.rule_name
    trigger = rule_result.trigger

    log.info(
        "Decision rule applied | rule=%s | action=%s | trigger=%s",
        rule_name, action, trigger,
    )

    # -------------------------------------------------------------------
    # STEP 3: Determine priority
    # -------------------------------------------------------------------
    priority = determine_priority(
        action=action,
        dr=inputs.dr,
        reliability=inputs.reliability,
        clinical=inputs.clinical,
        policy=cfg,
    )

    log.info("Priority determined: %s", priority)

    # -------------------------------------------------------------------
    # STEP 4: Generate reason
    # -------------------------------------------------------------------
    reason = generate_reason(
        action=action,
        rule_name=rule_name,
        trigger=trigger,
        quality=inputs.quality,
        dr=inputs.dr,
        reliability=inputs.reliability,
        clinical=inputs.clinical,
        policy=cfg,
        gradcam=inputs.gradcam,
    )

    # -------------------------------------------------------------------
    # STEP 5: Assemble output
    # -------------------------------------------------------------------
    output = _assemble_output(
        action=action,
        priority=priority,
        reason=reason,
        rule_name=rule_name,
        inputs=inputs,
        policy=cfg,
        include_evidence=include_evidence,
    )

    log.info(
        "Final decision | action=%s | priority=%s | dr_grade=%s | "
        "review_required=%s | rule=%s",
        output["action"],
        output["priority"],
        output.get("dr_grade"),
        output["review_required"],
        rule_name,
    )

    return output


def run_screening_decision(
    quality_result: Any,
    dr_result: Any,
    reliability_result: Any,
    clinical_context: Optional[Any] = None,
    gradcam_path: Optional[str] = None,
    policy: Optional[DecisionPolicy] = None,
    include_evidence: bool = True,
) -> Dict[str, Any]:
    """
    End-to-end screening decision orchestration (alias for make_final_decision).

    This function is the high-level integration point for the complete pipeline:

        Quality → DR Model → Reliability → Clinical Context → Decision Layer

    It does NOT recalculate any upstream module — it simply orchestrates
    the already-computed outputs and returns the final routing decision.

    All parameters and return values are identical to make_final_decision().

    See make_final_decision() for full documentation.

    Example
    -------
    ::

        from src.engine import run_screening_decision

        result = run_screening_decision(
            quality_result=quality_output,    # from anuj-fundus-quality
            dr_result=dr_output,              # from Vinayak's DR model
            reliability_result=rel_output,    # from anuj-reliability
            clinical_context=clinical_output, # from anuj-clinical-context (optional)
        )

        print(result["action"])    # "routine" | "doctor_review" | "refer" | "recapture"
        print(result["priority"])  # "low" | "medium" | "high" | "urgent"
        print(result["reason"])    # clear explanation
    """
    return make_final_decision(
        quality_result=quality_result,
        dr_result=dr_result,
        reliability_result=reliability_result,
        clinical_context=clinical_context,
        gradcam_path=gradcam_path,
        policy=policy,
        include_evidence=include_evidence,
    )
