"""
Tests — Distance calculation
"""
import numpy as np
import pytest

from src.distance import (
    mahalanobis_distance,
    cosine_distance,
    euclidean_distance,
    compute_distance,
)
from src.reference import ReferenceDistribution
from src.utils import EmbeddingError


def _make_reference(n: int = 20, d: int = 10, seed: int = 0) -> ReferenceDistribution:
    """Build a small synthetic reference distribution."""
    rng = np.random.default_rng(seed)
    embeddings = rng.random((n, d)).astype(np.float64)
    ref = ReferenceDistribution()
    ref.fit(embeddings)
    return ref


class TestMahalanobisDistance:

    def test_mean_has_low_distance(self):
        """The reference mean itself should have a very small Mahalanobis distance."""
        ref = _make_reference()
        dist = mahalanobis_distance(ref.mean, ref)
        assert dist < 1e-6, f"Mean should have ~0 Mahalanobis distance, got {dist}"

    def test_far_vector_has_high_distance(self):
        """A vector 100× the mean norm away should have a high distance."""
        ref = _make_reference(d=5)
        far = ref.mean + 100.0  # massively shifted
        dist_far = mahalanobis_distance(far, ref)
        dist_near = mahalanobis_distance(ref.mean + 0.0001, ref)
        assert dist_far > dist_near

    def test_returns_non_negative(self):
        """Distance must always be ≥ 0."""
        ref = _make_reference()
        rng = np.random.default_rng(99)
        for _ in range(20):
            x = rng.random(ref.embedding_dim)
            d = mahalanobis_distance(x, ref)
            assert d >= 0.0

    def test_symmetric_around_mean(self):
        """Vectors equidistant from mean should have similar Mahalanobis distances
        (exact symmetry holds when covariance is symmetric, which it is)."""
        ref = _make_reference(d=5, seed=42)
        delta = np.ones(ref.embedding_dim) * 0.5
        d1 = mahalanobis_distance(ref.mean + delta, ref)
        d2 = mahalanobis_distance(ref.mean - delta, ref)
        assert abs(d1 - d2) < 1e-8


class TestCosineDistance:

    def test_identical_direction_zero(self):
        """Cosine distance from mean to mean must be 0."""
        ref = _make_reference()
        dist = cosine_distance(ref.mean, ref)
        assert dist < 1e-8

    def test_range_zero_to_one(self):
        """Cosine distance must lie in [0, 1]."""
        ref = _make_reference()
        rng = np.random.default_rng(7)
        for _ in range(30):
            x = rng.random(ref.embedding_dim)
            d = cosine_distance(x, ref)
            assert 0.0 <= d <= 1.0 + 1e-10

    def test_zero_vector_max_distance(self):
        """A zero embedding should return distance 1.0 (treated as maximally distant)."""
        ref = _make_reference()
        zero = np.zeros(ref.embedding_dim)
        dist = cosine_distance(zero, ref)
        assert dist == 1.0


class TestEuclideanDistance:

    def test_mean_has_zero_distance(self):
        """The reference mean should have near-zero normalised Euclidean distance."""
        ref = _make_reference()
        dist = euclidean_distance(ref.mean, ref)
        assert dist < 1e-6

    def test_returns_non_negative(self):
        """Euclidean distance must be ≥ 0."""
        ref = _make_reference()
        rng = np.random.default_rng(11)
        for _ in range(20):
            x = rng.random(ref.embedding_dim)
            d = euclidean_distance(x, ref)
            assert d >= 0.0


class TestComputeDistance:

    def test_dispatcher_mahalanobis(self):
        ref = _make_reference()
        d = compute_distance(ref.mean, ref, metric="mahalanobis")
        assert d < 1e-6

    def test_dispatcher_cosine(self):
        ref = _make_reference()
        d = compute_distance(ref.mean, ref, metric="cosine")
        assert d < 1e-6

    def test_dispatcher_euclidean(self):
        ref = _make_reference()
        d = compute_distance(ref.mean, ref, metric="euclidean")
        assert d < 1e-6

    def test_dispatcher_nearest(self):
        ref = _make_reference()
        d = compute_distance(ref.mean, ref, metric="nearest")
        assert d < 1e-6

    def test_unknown_metric_raises(self):
        ref = _make_reference()
        with pytest.raises(EmbeddingError):
            compute_distance(ref.mean, ref, metric="unknown_metric")


class TestNumericalStability:

    def test_singular_covariance_handled(self):
        """
        When n < d (underdetermined covariance), the matrix is singular.
        The implementation must handle this without crashing.
        """
        rng = np.random.default_rng(55)
        # 5 samples in a 10-dim space → singular covariance
        embeddings = rng.random((5, 10)).astype(np.float64)
        ref = ReferenceDistribution()
        ref.fit(embeddings)  # Should not raise
        dist = mahalanobis_distance(rng.random(10), ref)
        assert np.isfinite(dist)

    def test_near_zero_variance_dimension_handled(self):
        """A near-constant feature dimension should not crash covariance estimation."""
        rng = np.random.default_rng(66)
        embeddings = rng.random((20, 5)).astype(np.float64)
        # Make first dimension nearly constant
        embeddings[:, 0] = 0.5 + rng.random(20) * 1e-10
        ref = ReferenceDistribution()
        ref.fit(embeddings)
        dist = mahalanobis_distance(rng.random(5), ref)
        assert np.isfinite(dist)
