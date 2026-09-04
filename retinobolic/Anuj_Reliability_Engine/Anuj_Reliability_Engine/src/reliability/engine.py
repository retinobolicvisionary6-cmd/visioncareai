"""
src/reliability/engine.py — Main orchestration entry point.

This is the single public interface for the entire Reliability Engine pipeline.

PUBLIC API
----------
    run_reliability_pipeline(
        dr_result  : dict,
        image_path : str | Path | np.ndarray,
        config     : ReliabilityConfig | None = None,
        **kwargs,
    ) -> dict

PIPELINE SEQUENCE
-----------------
    1. validate dr_result structure (light check)
    2. calculate_confidence(dr_result)          ← Confidence Module
    3. calculate_uncertainty(dr_result)         ← Uncertainty Module
    4. detect_ood(image_path)                   ← OOD Module
    5. calculate_reliability(c, u, o, config)   ← Fusion + Rules
    6. return unified dict

No module calculation is duplicated in this file.

IMPORTANT SAFETY PRINCIPLE
--------------------------
This pipeline produces an ENGINEERING reliability classification:
    acceptable | caution | review_required

It does NOT:
    - Diagnose Diabetic Retinopathy.
    - Provide treatment or referral advice.
    - Replace clinical evaluation by a qualified physician.
    - Interpret confidence as medical certainty.
    - Interpret OOD as absence of disease.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .config import DEFAULT_CONFIG, ReliabilityConfig
from .fusion import calculate_reliability
from ..common.validation import validate_dr_input

log = logging.getLogger(__name__)


def run_reliability_pipeline(
    dr_result: dict,
    image_path: Union[str, Path, "np.ndarray"],
    config: Optional[ReliabilityConfig] = None,
    include_score: bool = True,
) -> dict:
    """
    Run the full reliability pipeline for one fundus image.

    Orchestrates the three existing Anuj modules and the fusion engine:
        Confidence Module → Uncertainty Module → OOD Module → Reliability Engine

    Parameters
    ----------
    dr_result   : dict
        Output from Vinayak's 4-class DR model.
        Required key: "probabilities" (dict or list of 4 floats).
        Optional keys: "grade", "gradcam_path".

        Example::

            {
                "grade": 2,
                "probabilities": {"0": 0.03, "1": 0.08, "2": 0.81, "3": 0.08}
            }

    image_path  : str, Path, or numpy ndarray (H, W, 3) uint8
        Path to the fundus image, or a pre-loaded RGB array.
        Used by the OOD module to compute an image embedding.

    config      : ReliabilityConfig or None
        Threshold configuration.  Uses DEFAULT_CONFIG if None.

    include_score : bool (default True)
        Include optional bounded engineering reliability_score in output.

    Returns
    -------
    dict — JSON-serialisable reliability result.

        Always-present keys:
            reliability_status  (str)   "acceptable" | "caution" | "review_required"
            review_required     (bool)
            reason              (str)   human-readable explanation
            confidence          (float) in [0, 1]
            confidence_level    (str)   "high" | "medium" | "low"
            uncertainty         (float) in [0, 1]
            uncertainty_level   (str)   "low" | "medium" | "high"
            ood                 (bool)
            ood_status          (str)   "in_distribution" | "review_required"
            ood_score           (float)

        Optional keys (present when available):
            reliability_score   (float) in [0, 1]  — engineering heuristic only
            predicted_grade     (int)
            predicted_class_name(str)

    Raises
    ------
    ValidationError     — if dr_result is malformed.
    InvalidInputFormatError / InvalidProbabilityError — from Confidence Module.
    ValidationError     — from Uncertainty Module.
    FileNotFoundError, ImageLoadError, EmbeddingError — from OOD Module.

    Example
    -------
    ::

        from src.reliability.engine import run_reliability_pipeline

        dr_result = {
            "grade": 2,
            "probabilities": {"0": 0.03, "1": 0.08, "2": 0.81, "3": 0.08},
        }
        result = run_reliability_pipeline(dr_result, "path/to/fundus.jpg")
        print(result["reliability_status"])   # "acceptable"
        print(result["review_required"])      # False
        print(result["reason"])
    """
    cfg = config if config is not None else DEFAULT_CONFIG

    # Step 1: Light structural validation of DR result
    validate_dr_input(dr_result)

    # Import adapters here (lazy) to avoid circular imports and allow
    # sys.path injection to happen before import resolution.
    from ..confidence import calculate_confidence       # noqa: PLC0415
    from ..uncertainty import calculate_uncertainty     # noqa: PLC0415
    from ..ood import detect_ood                        # noqa: PLC0415

    # Step 2: Confidence Module
    log.debug("Running Confidence Module...")
    confidence_result = calculate_confidence(dr_result)
    log.debug("Confidence: %s", confidence_result.get("confidence"))

    # Step 3: Uncertainty Module
    log.debug("Running Uncertainty Module...")
    uncertainty_result = calculate_uncertainty(dr_result)
    log.debug("Uncertainty: %s", uncertainty_result.get("uncertainty"))

    # Step 4: OOD Module
    log.debug("Running OOD Module...")
    ood_result = detect_ood(image_path)
    log.debug("OOD: %s (score=%.4f)", ood_result.get("ood"), ood_result.get("ood_score", 0))

    # Step 5: Reliability Engine fusion
    log.debug("Running Reliability Engine fusion...")
    reliability = calculate_reliability(
        confidence_result=confidence_result,
        uncertainty_result=uncertainty_result,
        ood_result=ood_result,
        config=cfg,
        include_score=include_score,
    )

    log.info(
        "Reliability result | status=%s | review_required=%s | confidence=%.3f | "
        "uncertainty=%.3f | ood=%s",
        reliability.get("reliability_status"),
        reliability.get("review_required"),
        reliability.get("confidence", 0),
        reliability.get("uncertainty", 0),
        reliability.get("ood"),
    )

    return reliability
