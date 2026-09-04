"""
uncertainty.py - Core uncertainty calculations for the DR Prediction Uncertainty Engine.

Computes normalized Shannon entropy as the canonical model uncertainty score,
along with auxiliary probability margin and uncertainty level mapping.

---------------------------------------------------------------------------
SEMANTIC DISTINCTION (read carefully)
---------------------------------------------------------------------------
Confidence:
    How dominant is the top predicted class?
    = max(probabilities)
    (Produced by Module 1 - Confidence Module)

Uncertainty:
    How spread out / ambiguous is the FULL probability distribution?
    = normalized Shannon entropy over all 4 classes
    (Produced by THIS module - Module 2)

These are related but NOT identical:
  High confidence + low uncertainty  -> strong, clear prediction
  Low confidence  + high uncertainty -> ambiguous prediction

Neither value should be equated with clinical truth.
---------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np

from .config import UncertaintyConfig, DEFAULT_CONFIG
from .validation import validate_dr_input, validate_probabilities


# ---------------------------------------------------------------------------
# Core mathematical calculations
# ---------------------------------------------------------------------------

def compute_shannon_entropy(probs: np.ndarray, config: UncertaintyConfig = DEFAULT_CONFIG) -> float:
    """
    Compute numerically stable Shannon entropy in nats (natural log base).

    Formula:
        H(P) = -sum( p_i * ln(p_i) )

    Numerical stability:
        The limit  lim_{p->0} p * ln(p) = 0  is enforced by zeroing out
        terms where p_i == 0.  A tiny epsilon is added before log only for
        non-zero entries, preventing log(0) while preserving the zero-term
        convention exactly.

    Parameters
    ----------
    probs : np.ndarray
        Validated probability array of shape (4,).
    config : UncertaintyConfig
        Provides the EPSILON constant for zero-masking.

    Returns
    -------
    float
        Shannon entropy H >= 0 (in nats).
        Returns 0.0 for deterministic distributions.
        Returns ln(4) ~= 1.3863 for the uniform distribution.
        Guaranteed finite - no NaN or Infinity.
    """
    probs = np.asarray(probs, dtype=np.float64)

    # Zero-mask: only compute p_i * log(p_i) for p_i > 0
    # This correctly implements the convention 0 * log(0) = 0
    nonzero_mask = probs > 0.0
    safe_probs = probs[nonzero_mask]

    # Add epsilon only to genuinely non-zero entries (safe log)
    entropy = -np.sum(safe_probs * np.log(safe_probs + config.EPSILON))

    # Clamp to exactly 0.0 for floating-point noise near 0
    return float(max(0.0, entropy))


def compute_normalized_uncertainty(
    probs: np.ndarray,
    config: UncertaintyConfig = DEFAULT_CONFIG,
) -> float:
    """
    Compute normalized Shannon entropy (Model Uncertainty score).

    Formula:
        U = H(P) / ln(K)

    where K = NUM_CLASSES (4 for the DR model).

    This normalizes entropy to the range [0, 1]:
        U = 0.0  ->  deterministic distribution  (e.g. [0, 0, 1, 0])
        U = 1.0  ->  maximum entropy / uniform   (e.g. [0.25, 0.25, 0.25, 0.25])

    Parameters
    ----------
    probs : np.ndarray
        Validated probability array of shape (4,).
    config : UncertaintyConfig
        Provides NUM_CLASSES and EPSILON.

    Returns
    -------
    float
        Normalized uncertainty score in [0.0, 1.0].
        Guaranteed finite - no NaN or Infinity.
    """
    max_entropy = np.log(float(config.NUM_CLASSES))   # ln(4) ~= 1.3863
    raw_entropy = compute_shannon_entropy(probs, config)

    if max_entropy == 0.0:
        # Degenerate case: single class model (should never occur with NUM_CLASSES=4)
        return 0.0

    uncertainty = raw_entropy / max_entropy

    # Hard clamp to [0, 1] to absorb any residual floating-point imprecision
    return float(min(1.0, max(0.0, uncertainty)))


def compute_probability_margin(probs: np.ndarray) -> float:
    """
    Compute the probability margin as an auxiliary uncertainty signal.

    Formula:
        margin = p_(1) - p_(2)

    where p_(1) and p_(2) are the highest and second-highest probabilities.

    Interpretation:
        large margin -> clearer class separation -> lower ambiguity
        small margin -> similar top classes     -> higher ambiguity

    NOTE: Margin is an AUXILIARY signal only.
    The canonical uncertainty score is normalized entropy, not margin.

    Parameters
    ----------
    probs : np.ndarray
        Validated probability array of shape (4,).

    Returns
    -------
    float
        Probability margin in [0.0, 1.0].
    """
    sorted_probs = np.sort(probs)[::-1]   # descending
    top_1 = float(sorted_probs[0])
    top_2 = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
    return round(float(top_1 - top_2), 6)


def determine_uncertainty_level(
    uncertainty: float,
    config: UncertaintyConfig = DEFAULT_CONFIG,
) -> tuple[str, bool]:
    """
    Map a normalized uncertainty score to a human-readable level and review flag.

    Level mapping (prototype thresholds - see config.py for disclaimer):
        uncertainty <= LOW_UNCERTAINTY_MAX           -> "low",    review_recommended=False
        LOW_UNCERTAINTY_MAX < uncertainty
            < HIGH_UNCERTAINTY_MIN                   -> "medium", review_recommended=False
        uncertainty >= HIGH_UNCERTAINTY_MIN          -> "high",   review_recommended=True

    Parameters
    ----------
    uncertainty : float
        Normalized entropy score in [0, 1].
    config : UncertaintyConfig
        Provides the configurable threshold values.

    Returns
    -------
    (level, review_recommended) : tuple[str, bool]
        level              : "low" | "medium" | "high"
        review_recommended : True if uncertainty >= HIGH_UNCERTAINTY_MIN
    """
    if uncertainty <= config.LOW_UNCERTAINTY_MAX:
        return config.LEVEL_LOW, False
    elif uncertainty >= config.HIGH_UNCERTAINTY_MIN:
        return config.LEVEL_HIGH, True
    else:
        return config.LEVEL_MEDIUM, False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_uncertainty(
    dr_result: dict,
    config: UncertaintyConfig | None = None,
    confidence: float | None = None,
) -> dict:
    """
    Primary public API for the Uncertainty Engine.

    Consumes the output contract from Vinayak's 4-class DR model and returns
    a structured uncertainty analysis.

    Parameters
    ----------
    dr_result : dict
        Output from Vinayak's model. Required key: "probabilities".
        Optional key: "grade" (predicted DR grade).

        Example:
            {
                "grade": 2,
                "probabilities": {"0": 0.03, "1": 0.08, "2": 0.81, "3": 0.08},
                "gradcam_path": "outputs/gradcam/image_001.jpg"
            }

    config : UncertaintyConfig or None
        Custom configuration. Uses DEFAULT_CONFIG if None.

    confidence : float or None
        Optional confidence score from Module 1 (Confidence Engine).
        If provided, it is forwarded as metadata to facilitate later
        combination of Confidence + Uncertainty + OOD in the Reliability Engine.
        Uncertainty is ALWAYS calculated directly from probabilities regardless.

    Returns
    -------
    dict
        Uncertainty analysis result.

        Example output:
            {
                "predicted_grade": 2,
                "uncertainty": 0.3124,
                "uncertainty_level": "low",
                "review_recommended": False,
                "probability_margin": 0.73,
                # Optional fields (present only if provided):
                "confidence": 0.81      # from Module 1, if supplied
            }

    Raises
    ------
    ValidationError
        If the top-level input structure is malformed.
    InvalidProbabilityError
        If the probability distribution is invalid.

    Notes
    -----
    Confidence vs Uncertainty:
        confidence  = max(probabilities)   [Module 1 - how dominant is the top class?]
        uncertainty = normalized entropy   [Module 2 - how spread is the distribution?]

    Future extensibility:
        The output dict is designed to accept additional keys from OOD detection,
        camera reliability, and clinical context without requiring any changes to
        this function.

    Tie-breaking:
        When multiple classes share the maximum probability (e.g. uniform distribution),
        np.argmax returns the index of the FIRST occurrence. This is deterministic
        and documented behavior.
    """
    cfg = config if config is not None else DEFAULT_CONFIG

    # 1. Validate input
    probs, predicted_grade = validate_dr_input(dr_result, cfg)

    # 2. Infer grade from probabilities if not supplied by model
    if predicted_grade is None:
        predicted_grade = int(np.argmax(probs))

    # 3. Core uncertainty calculation
    uncertainty_score = compute_normalized_uncertainty(probs, cfg)
    uncertainty_score = round(uncertainty_score, 4)

    # 4. Level + review flag
    level, review_recommended = determine_uncertainty_level(uncertainty_score, cfg)

    # 5. Auxiliary signal
    margin = compute_probability_margin(probs)

    # 6. Assemble output
    result = {
        "predicted_grade": predicted_grade,
        "uncertainty": uncertainty_score,
        "uncertainty_level": level,
        "review_recommended": review_recommended,
        "probability_margin": margin,
    }

    # 7. Optional: carry confidence from Module 1 as metadata
    if confidence is not None:
        result["confidence"] = float(confidence)

    return result
