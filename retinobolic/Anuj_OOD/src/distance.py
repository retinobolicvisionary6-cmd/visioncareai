"""
OOD Detection Module — Distance Calculators
============================================
Numerically stable distance metrics for OOD scoring.

All distance functions accept:
    embedding   : 1-D np.ndarray (validated, finite)
    reference   : ReferenceDistribution (fitted)

And return a single float ≥ 0 representing distance from the reference
in-distribution.

Score interpretation
--------------------
For all metrics:
    Lower score  → embedding is closer to the known distribution → IN-DISTRIBUTION
    Higher score → embedding is distant from known distribution  → likely OOD

Do NOT interpret these raw scores as probabilities without proper calibration.

Available metrics
-----------------
    mahalanobis  (default) — accounts for feature covariance
    cosine       — directional distance from reference mean, range [0, 1]
    euclidean    — feature-normalised L2 distance
    nearest      — nearest-centroid distance (for multi-cluster references)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from .utils import EmbeddingError

if TYPE_CHECKING:
    from .reference import ReferenceDistribution

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Individual distance functions
# ---------------------------------------------------------------------------


def mahalanobis_distance(embedding: np.ndarray, reference: "ReferenceDistribution") -> float:
    """
    Mahalanobis distance of *embedding* from the reference distribution mean.

    Mahalanobis distance:
        d_M(x, μ) = sqrt((x - μ)ᵀ · Σ⁻¹ · (x - μ))

    This metric accounts for feature covariance and scale differences,
    making it more robust than raw Euclidean distance for high-dimensional
    embeddings.

    Returns
    -------
    float ≥ 0
    """
    dist = reference.mahalanobis_distance(embedding)
    _check_finite(dist, "mahalanobis")
    return dist


def cosine_distance(embedding: np.ndarray, reference: "ReferenceDistribution") -> float:
    """
    Cosine distance of *embedding* from the reference mean vector.
    Range: [0.0, 1.0]  (0 = same direction, 1 = orthogonal)

    Useful as a fallback when the precision matrix is ill-conditioned.

    Returns
    -------
    float in [0.0, 1.0]
    """
    dist = reference.cosine_distance(embedding)
    _check_finite(dist, "cosine")
    return dist


def euclidean_distance(embedding: np.ndarray, reference: "ReferenceDistribution") -> float:
    """
    Feature-normalised Euclidean (L2) distance from the reference mean.
    Each dimension is divided by its reference standard deviation before
    computing L2 norm, making the score partially scale-invariant.

    Returns
    -------
    float ≥ 0
    """
    dist = reference.euclidean_distance(embedding)
    _check_finite(dist, "euclidean")
    return dist


def nearest_centroid_distance(
    embedding: np.ndarray,
    reference: "ReferenceDistribution",
) -> float:
    """
    Euclidean distance to the single nearest centroid in the reference mean.
    Currently equivalent to euclidean_distance() since a single global
    centroid is maintained.

    Extension point: if future versions cluster the reference into multiple
    centroids (e.g., per-DR-grade), this function will return the minimum
    distance across all centroids.

    Returns
    -------
    float ≥ 0
    """
    # Currently single-centroid; same as feature-normalised euclidean
    return euclidean_distance(embedding, reference)


# ---------------------------------------------------------------------------
# Unified distance dispatcher
# ---------------------------------------------------------------------------

_METRIC_REGISTRY: dict[str, callable] = {
    "mahalanobis": mahalanobis_distance,
    "cosine":      cosine_distance,
    "euclidean":   euclidean_distance,
    "nearest":     nearest_centroid_distance,
}


def compute_distance(
    embedding: np.ndarray,
    reference: "ReferenceDistribution",
    metric: str = None,
) -> float:
    """
    Compute the OOD distance score using the specified *metric*.

    Parameters
    ----------
    embedding : 1-D float64 np.ndarray
    reference : ReferenceDistribution (fitted)
    metric    : str — one of "mahalanobis", "cosine", "euclidean", "nearest".
                Defaults to config.DISTANCE_METRIC.

    Returns
    -------
    float — OOD distance score (higher = more OOD).

    Raises
    ------
    EmbeddingError — if *metric* is unknown or distance computation fails.
    """
    from . import config  # local import to avoid circular

    metric = metric or config.DISTANCE_METRIC

    if metric not in _METRIC_REGISTRY:
        raise EmbeddingError(
            f"Unknown distance metric '{metric}'. "
            f"Valid options: {list(_METRIC_REGISTRY)}"
        )

    try:
        score = _METRIC_REGISTRY[metric](embedding, reference)
    except Exception as exc:
        raise EmbeddingError(
            f"Distance computation failed for metric '{metric}': {exc}"
        ) from exc

    log.debug("Distance [%s]: %.6f", metric, score)
    return score


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _check_finite(value: float, metric_name: str) -> None:
    if not np.isfinite(value):
        raise EmbeddingError(
            f"Distance metric '{metric_name}' produced a non-finite value: {value}. "
            "Check the reference distribution or embedding for degenerate values."
        )
