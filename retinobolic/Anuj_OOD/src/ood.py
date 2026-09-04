"""
OOD Detection Module — Core OOD Engine
========================================
Public API for out-of-distribution detection.

Primary public interface
------------------------
    detect_ood(image_path, ...)  → dict

    Returns a stable JSON-serialisable dict:
    {
        "ood":            bool,
        "ood_status":     "in_distribution" | "review_required",
        "ood_score":      float,   # computed distance score
        "threshold":      float,   # configured threshold
        "distance_metric": str,    # active metric name
        "extractor_type":  str,    # active embedding extractor
        "reason":         str,     # plain-English explanation
        "metadata":       dict,    # pass-through camera / domain metadata
    }

Score semantics
---------------
    Lower ood_score  → embedding is closer to the reference distribution
    Higher ood_score → embedding is further from the reference distribution

The score is NOT a calibrated probability. Do not label it "OOD probability"
without proper calibration against representative in/out-of-distribution data.

Threshold note
--------------
OOD_THRESHOLD (config.py) is a PROTOTYPE VALUE.
It must be calibrated on real fundus + OOD data before clinical use.
See build_reference.py for percentile-based threshold suggestions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from . import config
from .distance import compute_distance
from .embedding import extract_embedding
from .reference import ReferenceDistribution
from .utils import (
    DimensionMismatchError,
    EmbeddingError,
    ImageLoadError,
    OODError,
    ReferenceError,
    check_dimension_match,
    validate_embedding,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OOD Status constants
# ---------------------------------------------------------------------------

STATUS_IN_DISTRIBUTION = "in_distribution"
STATUS_REVIEW_REQUIRED = "review_required"

REASON_IN_DIST = (
    "Input embedding is within the configured reference distribution."
)
REASON_OOD = (
    "Input embedding is substantially distant from the reference fundus distribution."
)
REASON_BORDERLINE = (
    "Input embedding is near the OOD threshold boundary; treat with additional caution."
)

# Fraction of threshold within which a result is considered "borderline"
_BORDERLINE_FRACTION = 0.10


# ---------------------------------------------------------------------------
# OODDetector class
# ---------------------------------------------------------------------------

class OODDetector:
    """
    Stateful OOD detector that caches the reference distribution and
    extractor across multiple calls.

    Prefer the module-level detect_ood() function for one-shot use.
    Use OODDetector directly when processing many images in a loop
    (avoids reloading reference and extractor repeatedly).

    Parameters
    ----------
    reference_path  : path to reference_statistics.json, or None to use config default
    threshold       : OOD threshold; None uses config.OOD_THRESHOLD
    metric          : distance metric; None uses config.DISTANCE_METRIC
    extractor_type  : embedding extractor; None uses config.EXTRACTOR_TYPE
    """

    def __init__(
        self,
        reference_path: Optional[Union[str, Path]] = None,
        threshold: Optional[float] = None,
        metric: Optional[str] = None,
        extractor_type: Optional[str] = None,
    ) -> None:
        self.threshold: float = threshold if threshold is not None else config.OOD_THRESHOLD
        self.metric: str = metric or config.DISTANCE_METRIC
        self.extractor_type: str = extractor_type or config.EXTRACTOR_TYPE

        # Load reference distribution
        stats_path = Path(reference_path) if reference_path else config.REFERENCE_STATS_FILE
        self._reference = ReferenceDistribution.load(stats_path)

        log.info(
            "OODDetector ready | extractor=%s | metric=%s | threshold=%.4f | "
            "reference_n=%d | reference_dim=%d",
            self.extractor_type,
            self.metric,
            self.threshold,
            self._reference.n_samples,
            self._reference.embedding_dim,
        )

    def detect(
        self,
        image_path_or_array: Union[str, Path, np.ndarray],
        metadata: Optional[dict] = None,
    ) -> dict[str, Any]:
        """
        Run OOD detection on a single image.

        Parameters
        ----------
        image_path_or_array : path string, Path, or pre-loaded (H,W,3) uint8 array
        metadata            : optional camera / domain info dict (not used for scoring)

        Returns
        -------
        dict — stable JSON-serialisable OOD result (see module docstring)
        """
        metadata = metadata or {}
        source_label = (
            str(image_path_or_array)
            if not isinstance(image_path_or_array, np.ndarray)
            else "<array>"
        )
        log.debug("OOD detection: %s", source_label)

        # --- Extract embedding ---
        embedding = extract_embedding(image_path_or_array, extractor_type=self.extractor_type)
        embedding = validate_embedding(embedding, context="OODDetector.detect")

        # --- Dimension check ---
        check_dimension_match(
            embedding,
            self._reference.embedding_dim,
            context="OODDetector.detect",
        )

        # --- Compute OOD score ---
        ood_score = compute_distance(embedding, self._reference, metric=self.metric)

        # --- Apply threshold ---
        is_ood = ood_score > self.threshold

        # Determine if borderline
        border_low = self.threshold * (1.0 - _BORDERLINE_FRACTION)
        border_high = self.threshold * (1.0 + _BORDERLINE_FRACTION)
        borderline = border_low <= ood_score <= border_high

        # --- Reason string ---
        if is_ood:
            reason = REASON_OOD
        elif borderline:
            reason = REASON_BORDERLINE
        else:
            reason = REASON_IN_DIST

        status = STATUS_REVIEW_REQUIRED if is_ood else STATUS_IN_DISTRIBUTION

        result = {
            "ood": bool(is_ood),
            "ood_status": status,
            "ood_score": round(float(ood_score), 6),
            "threshold": round(float(self.threshold), 6),
            "distance_metric": self.metric,
            "extractor_type": self.extractor_type,
            "reason": reason,
            "metadata": metadata,
        }

        log.info(
            "OOD result | score=%.4f | threshold=%.4f | status=%s | source=%s",
            ood_score,
            self.threshold,
            status,
            source_label,
        )
        return result


# ---------------------------------------------------------------------------
# Module-level cached detector (lazy init)
# ---------------------------------------------------------------------------

_default_detector: Optional[OODDetector] = None


def _get_default_detector(
    reference_path: Optional[Union[str, Path]] = None,
    threshold: Optional[float] = None,
    metric: Optional[str] = None,
    extractor_type: Optional[str] = None,
) -> OODDetector:
    """
    Return the module-level detector, creating it on first call.
    Re-creates if any parameter differs from the cached instance.
    """
    global _default_detector

    # Build a key to detect config changes
    key = (
        str(reference_path or config.REFERENCE_STATS_FILE),
        threshold or config.OOD_THRESHOLD,
        metric or config.DISTANCE_METRIC,
        extractor_type or config.EXTRACTOR_TYPE,
    )

    if _default_detector is None or getattr(_default_detector, "_cache_key", None) != key:
        _default_detector = OODDetector(
            reference_path=reference_path,
            threshold=threshold,
            metric=metric,
            extractor_type=extractor_type,
        )
        _default_detector._cache_key = key  # type: ignore[attr-defined]

    return _default_detector


# ---------------------------------------------------------------------------
# Primary public API
# ---------------------------------------------------------------------------

def detect_ood(
    image_path: Union[str, Path, np.ndarray],
    reference_path: Optional[Union[str, Path]] = None,
    threshold: Optional[float] = None,
    metric: Optional[str] = None,
    extractor_type: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> dict[str, Any]:
    """
    Detect whether *image_path* is out-of-distribution relative to the
    reference fundus distribution.

    Parameters
    ----------
    image_path      : path to a fundus image (str/Path), or pre-loaded uint8 array
    reference_path  : path to reference_statistics.json  (default: config path)
    threshold       : OOD decision threshold  (default: config.OOD_THRESHOLD)
    metric          : distance metric  (default: config.DISTANCE_METRIC)
    extractor_type  : embedding extractor  (default: config.EXTRACTOR_TYPE)
    metadata        : optional camera / domain metadata dict (pass-through only)

    Returns
    -------
    dict with keys:
        ood            (bool)   — True if OOD
        ood_status     (str)    — "in_distribution" or "review_required"
        ood_score      (float)  — computed distance score
        threshold      (float)  — decision threshold applied
        distance_metric(str)    — metric used
        extractor_type (str)    — extractor used
        reason         (str)    — plain-English explanation
        metadata       (dict)   — pass-through metadata

    Raises
    ------
    FileNotFoundError  — if image_path does not exist
    ImageLoadError     — if image cannot be decoded
    EmbeddingError     — if embedding extraction fails
    ReferenceError     — if reference distribution file is missing or corrupt
    DimensionMismatchError — if embedding and reference dims do not match

    OOD Score Semantics
    -------------------
    Lower score  → closer to reference distribution (in-distribution)
    Higher score → further from reference distribution (likely OOD)

    The score is NOT a calibrated probability.

    Threshold Note
    --------------
    config.OOD_THRESHOLD is a PROTOTYPE VALUE.
    Calibrate against real in-distribution and OOD fundus data before clinical use.
    See build_reference.py --threshold_percentile for guidance.
    """
    detector = _get_default_detector(
        reference_path=reference_path,
        threshold=threshold,
        metric=metric,
        extractor_type=extractor_type,
    )
    return detector.detect(image_path, metadata=metadata)
