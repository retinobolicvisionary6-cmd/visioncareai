"""
src/reliability/fusion.py — Signal fusion and composite scoring.

Collects validated outputs from all three upstream modules,
applies the rule engine, and assembles the final ReliabilityResult.

This is the last numeric step before the public engine API.

PUBLIC API
----------
    calculate_reliability(
        confidence_result   : dict,
        uncertainty_result  : dict,
        ood_result          : dict,
        config              : ReliabilityConfig | None = None,
    ) -> dict

IMPORTANT:
    - No module calculations are duplicated here.
    - Inputs are raw dicts from the upstream module public APIs.
    - Validation converts them to typed dataclasses before rule evaluation.
    - The optional reliability_score is mathematically defined and documented.
    - It does NOT override safety rules and is NOT a medical certainty score.

OPTIONAL RELIABILITY SCORE — mathematical definition
-----------------------------------------------------
    reliability_score is a bounded engineering heuristic in [0, 1].

    Definition:
        s_conf      = confidence           (already in [0, 1])
        s_uncert    = 1 - uncertainty      (inverted: low uncertainty = high score)
        s_ood       = 0.0 if ood else 1.0  (binary penalty)

        reliability_score = s_ood * (w_conf * s_conf + w_uncert * s_uncert)

    Default weights:
        w_conf   = 0.5
        w_uncert = 0.5

    Properties:
        - Range: [0, 1]
        - OOD = True  → score = 0.0 (regardless of other signals)
        - OOD = False → score = weighted average of (confidence, 1-uncertainty)
        - Does NOT override the deterministic status decision.
        - MUST NOT be labelled as "diagnostic accuracy" or "medical certainty".
"""

from __future__ import annotations

from typing import Optional

from .config import DEFAULT_CONFIG, ReliabilityConfig
from .rules import evaluate_rules
from ..common.schemas import ReliabilityResult
from ..common.validation import (
    validate_confidence_result,
    validate_ood_result,
    validate_uncertainty_result,
)

# Default weights for the optional composite engineering score
_WEIGHT_CONFIDENCE: float = 0.5
_WEIGHT_UNCERTAINTY: float = 0.5


def _compute_reliability_score(
    confidence: float,
    uncertainty: float,
    ood: bool,
) -> float:
    """
    Compute the optional bounded engineering reliability score.

    Formula:
        s_ood   = 0.0 if ood else 1.0
        score   = s_ood * (0.5 * confidence + 0.5 * (1 - uncertainty))

    Range: [0, 1]
    OOD = True always yields 0.0.

    This score is SUPPLEMENTARY to the deterministic status decision.
    It must NOT be interpreted as medical accuracy or diagnostic certainty.
    """
    if ood:
        return 0.0
    score = _WEIGHT_CONFIDENCE * confidence + _WEIGHT_UNCERTAINTY * (1.0 - uncertainty)
    # Hard clamp to [0, 1] to absorb floating-point imprecision
    return round(float(min(1.0, max(0.0, score))), 4)


def calculate_reliability(
    confidence_result: dict,
    uncertainty_result: dict,
    ood_result: dict,
    config: Optional[ReliabilityConfig] = None,
    include_score: bool = True,
) -> dict:
    """
    Fuse validated module outputs into a single reliability determination.

    Parameters
    ----------
    confidence_result   : dict  — output from calculate_confidence()
    uncertainty_result  : dict  — output from calculate_uncertainty()
    ood_result          : dict  — output from detect_ood()
    config              : ReliabilityConfig or None — uses DEFAULT_CONFIG if None
    include_score       : bool  — include optional reliability_score in output

    Returns
    -------
    dict — JSON-serialisable reliability result with keys:
        reliability_status  : str   "acceptable" | "caution" | "review_required"
        review_required     : bool
        reason              : str
        confidence          : float
        confidence_level    : str
        uncertainty         : float
        uncertainty_level   : str
        ood                 : bool
        ood_status          : str
        ood_score           : float
        reliability_score   : float  (only if include_score=True)
        predicted_grade     : int    (if available from confidence result)
        predicted_class_name: str    (if available from confidence result)

    Raises
    ------
    ValidationError — if any module output is structurally invalid.

    SAFETY NOTE
    -----------
    reliability_status is an engineering classification only.
    It does NOT constitute a medical diagnosis or clinical recommendation.
    """
    cfg = config if config is not None else DEFAULT_CONFIG

    # --- Step 1: Validate all three module outputs -------------------------
    conf = validate_confidence_result(confidence_result)
    uncert = validate_uncertainty_result(uncertainty_result)
    ood = validate_ood_result(ood_result)

    # --- Step 2: Apply deterministic rule engine ---------------------------
    status, review_required, reason = evaluate_rules(conf, uncert, ood, cfg)

    # --- Step 3: Optional engineering composite score ----------------------
    reliability_score: Optional[float] = None
    if include_score:
        reliability_score = _compute_reliability_score(
            conf.confidence, uncert.uncertainty, ood.ood
        )

    # --- Step 4: Assemble ReliabilityResult --------------------------------
    result = ReliabilityResult(
        reliability_status=status.value,
        review_required=review_required,
        reason=reason,
        confidence=conf.confidence,
        confidence_level=conf.confidence_level,
        uncertainty=uncert.uncertainty,
        uncertainty_level=uncert.uncertainty_level,
        ood=ood.ood,
        ood_status=ood.ood_status,
        ood_score=ood.ood_score,
        reliability_score=reliability_score,
        predicted_grade=conf.predicted_grade,
        predicted_class_name=conf.predicted_class_name,
    )

    return result.to_dict()
