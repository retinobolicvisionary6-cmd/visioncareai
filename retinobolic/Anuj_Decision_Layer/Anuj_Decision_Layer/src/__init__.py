"""
Final Decision Layer — Public API.

SIH Problem ID: SIH26038
Explainable AI for Diabetic Retinopathy Screening in Rural India

This module is the transparent final workflow engine that converts
validated AI outputs and available clinical context into a safe,
explainable next action, while keeping the doctor in the loop.

Usage
-----
    from src import make_final_decision, run_screening_decision

    result = make_final_decision(
        quality_result=quality_output,
        dr_result=dr_output,
        reliability_result=reliability_output,
        clinical_context=clinical_output,  # optional
    )

    print(result["action"])    # "routine" | "doctor_review" | "refer" | "recapture"
    print(result["priority"])  # "low" | "medium" | "high" | "urgent"
    print(result["reason"])    # clear, non-diagnostic explanation

SAFETY NOTE
-----------
This module produces WORKFLOW ROUTING DECISIONS ONLY.
It is NOT a medical diagnostic tool.
The final clinical decision rests with the examining physician.
"""

from src.engine import make_final_decision, run_screening_decision
from src.config import DecisionPolicy, load_policy, DEFAULT_POLICY
from src.validation import (
    DecisionValidationError,
    QualityResult,
    DRResult,
    ReliabilityResult,
    ClinicalContext,
    GradcamMetadata,
    ValidatedInputs,
    validate_quality_result,
    validate_dr_result,
    validate_reliability_result,
    validate_clinical_context,
    validate_gradcam_metadata,
    validate_all_inputs,
)
from src.rules import (
    RuleResult,
    rule_1_image_safety,
    rule_2_reliability_safety,
    rule_3_referable_dr,
    rule_4_routine,
    evaluate_decision_rules,
)
from src.priorities import determine_priority
from src.reasons import generate_reason, collect_reliability_signals

__all__ = [
    # Primary public API
    "make_final_decision",
    "run_screening_decision",
    # Configuration
    "DecisionPolicy",
    "load_policy",
    "DEFAULT_POLICY",
    # Validation
    "DecisionValidationError",
    "QualityResult",
    "DRResult",
    "ReliabilityResult",
    "ClinicalContext",
    "GradcamMetadata",
    "ValidatedInputs",
    "validate_quality_result",
    "validate_dr_result",
    "validate_reliability_result",
    "validate_clinical_context",
    "validate_gradcam_metadata",
    "validate_all_inputs",
    # Rules
    "RuleResult",
    "rule_1_image_safety",
    "rule_2_reliability_safety",
    "rule_3_referable_dr",
    "rule_4_routine",
    "evaluate_decision_rules",
    # Priority & Reason
    "determine_priority",
    "generate_reason",
    "collect_reliability_signals",
]

__version__ = "1.0.0"
