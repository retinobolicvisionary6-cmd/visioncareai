"""
OOD Detection Module — Reference Distribution Management
=========================================================
Handles building, serialising, and loading the reference distribution
that OOD scores are computed against.

Workflow
--------
  1. Feed reference in-distribution fundus embeddings to ReferenceDistribution.fit().
  2. Call .save() to persist statistics (mean, precision matrix, percentiles).
  3. At runtime, call ReferenceDistribution.load() — no raw images needed.
  4. Call .is_fitted to verify readiness before distance computation.

Covariance estimation
---------------------
When the number of samples n is close to or less than the embedding
dimension d, the empirical covariance matrix is singular.  We handle
this with:
  - Ledoit–Wolf shrinkage (scikit-learn) when sklearn is available.
  - Diagonal regularisation (add λI) as a universal fallback.
The precision matrix (inverse covariance) is stored for Mahalanobis distance.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from . import config
from .utils import ReferenceError

log = logging.getLogger(__name__)


class ReferenceDistribution:
    """
    Stores the statistical summary of the reference (in-distribution) embeddings
    required for Mahalanobis and centroid-based OOD scoring.

    Attributes set after fit()
    --------------------------
    mean          : 1-D array, shape (d,)
    precision     : 2-D array, shape (d, d)  — regularised inverse covariance
    covariance    : 2-D array, shape (d, d)  — regularised covariance
    distances     : 1-D array — Mahalanobis distances of reference samples
    percentiles   : dict  — p50, p90, p95, p99, p100 of reference distances
    n_samples     : int
    embedding_dim : int
    extractor_type: str
    """

    def __init__(self) -> None:
        self._fitted: bool = False
        self.mean: Optional[np.ndarray] = None
        self.precision: Optional[np.ndarray] = None
        self.covariance: Optional[np.ndarray] = None
        self.distances: Optional[np.ndarray] = None
        self.percentiles: dict[str, float] = {}
        self.n_samples: int = 0
        self.embedding_dim: int = 0
        self.extractor_type: str = ""

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(
        self,
        embeddings: np.ndarray,
        extractor_type: str = "",
    ) -> "ReferenceDistribution":
        """
        Fit the reference distribution from a (n, d) array of embeddings.

        Parameters
        ----------
        embeddings    : np.ndarray, shape (n, d)
        extractor_type: str — label for provenance / compatibility checks

        Returns
        -------
        self (for chaining)

        Raises
        ------
        ReferenceError — if n < 2 or embeddings contain NaN / Inf.
        """
        embeddings = np.asarray(embeddings, dtype=np.float64)

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        if embeddings.ndim != 2:
            raise ReferenceError(
                f"Expected 2-D embeddings array (n, d), got shape {embeddings.shape}."
            )

        n, d = embeddings.shape

        if n < 2:
            raise ReferenceError(
                f"Need at least 2 reference embeddings to estimate distribution; got {n}."
            )

        if not np.isfinite(embeddings).all():
            raise ReferenceError(
                "Reference embeddings contain NaN or Inf values. "
                "Check the embedding extractor output."
            )

        log.info("Fitting reference distribution: n=%d, d=%d.", n, d)

        self.n_samples = n
        self.embedding_dim = d
        self.extractor_type = extractor_type
        self.mean = embeddings.mean(axis=0)

        # Estimate regularised covariance
        self.covariance = self._estimate_covariance(embeddings, n, d)
        self.precision = self._invert_covariance(self.covariance)

        # Compute in-distribution distances for percentile calibration
        self.distances = self._compute_mahalanobis(embeddings)
        self.percentiles = {
            "p50":  float(np.percentile(self.distances, 50)),
            "p90":  float(np.percentile(self.distances, 90)),
            "p95":  float(np.percentile(self.distances, 95)),
            "p99":  float(np.percentile(self.distances, 99)),
            "p100": float(np.percentile(self.distances, 100)),
        }

        log.info(
            "Reference distribution fitted. "
            "Distance percentiles: p95=%.4f, p99=%.4f, max=%.4f.",
            self.percentiles["p95"],
            self.percentiles["p99"],
            self.percentiles["p100"],
        )
        self._fitted = True
        return self

    def save(
        self,
        stats_path: Path = None,
        embeddings_path: Path = None,
    ) -> None:
        """
        Persist reference statistics and embeddings to disk.

        Parameters
        ----------
        stats_path      : path for JSON statistics file
        embeddings_path : path for .npy embeddings archive
        """
        self._check_fitted("save")

        stats_path = Path(stats_path or config.REFERENCE_STATS_FILE)
        embeddings_path = Path(embeddings_path or config.REFERENCE_EMBEDDINGS_FILE)

        stats_path.parent.mkdir(parents=True, exist_ok=True)
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)

        # Save statistics JSON
        stats = {
            "n_samples": self.n_samples,
            "embedding_dim": self.embedding_dim,
            "extractor_type": self.extractor_type,
            "mean": self.mean.tolist(),
            "precision": self.precision.tolist(),
            "covariance": self.covariance.tolist(),
            "distances": self.distances.tolist(),
            "percentiles": self.percentiles,
        }
        with open(stats_path, "w") as fh:
            json.dump(stats, fh, indent=2)

        # Save raw embeddings array (optional; used for nearest-neighbour methods)
        np.save(str(embeddings_path), self.distances)

        log.info("Reference statistics saved to %s.", stats_path)
        log.info("Reference distance array saved to %s.", embeddings_path)

    @classmethod
    def load(
        cls,
        stats_path: Path = None,
        embeddings_path: Path = None,
    ) -> "ReferenceDistribution":
        """
        Load a previously saved reference distribution.

        Raises
        ------
        ReferenceError — if the file is missing or corrupt.
        """
        stats_path = Path(stats_path or config.REFERENCE_STATS_FILE)

        if not stats_path.exists():
            raise ReferenceError(
                f"Reference statistics file not found: {stats_path}. "
                "Run 'python build_reference.py' to build the reference distribution first."
            )

        try:
            with open(stats_path) as fh:
                stats = json.load(fh)
        except Exception as exc:
            raise ReferenceError(
                f"Cannot parse reference statistics file '{stats_path}': {exc}"
            ) from exc

        ref = cls()
        try:
            ref.n_samples = int(stats["n_samples"])
            ref.embedding_dim = int(stats["embedding_dim"])
            ref.extractor_type = stats.get("extractor_type", "")
            ref.mean = np.array(stats["mean"], dtype=np.float64)
            ref.precision = np.array(stats["precision"], dtype=np.float64)
            ref.covariance = np.array(stats["covariance"], dtype=np.float64)
            ref.distances = np.array(stats["distances"], dtype=np.float64)
            ref.percentiles = stats.get("percentiles", {})
        except (KeyError, ValueError) as exc:
            raise ReferenceError(
                f"Reference statistics file is incomplete or corrupt: {exc}"
            ) from exc

        ref._fitted = True
        log.info(
            "Reference distribution loaded: n=%d, d=%d, extractor='%s'.",
            ref.n_samples,
            ref.embedding_dim,
            ref.extractor_type,
        )
        return ref

    def mahalanobis_distance(self, embedding: np.ndarray) -> float:
        """
        Compute the Mahalanobis distance of *embedding* from the reference mean.

        Parameters
        ----------
        embedding : 1-D array, shape (d,)

        Returns
        -------
        float — Mahalanobis distance ≥ 0.
        """
        self._check_fitted("mahalanobis_distance")
        x = np.asarray(embedding, dtype=np.float64)
        diff = x - self.mean
        dist_sq = float(diff @ self.precision @ diff)
        return float(np.sqrt(max(dist_sq, 0.0)))

    def cosine_distance(self, embedding: np.ndarray) -> float:
        """
        Cosine distance of *embedding* from the reference mean vector.
        Returns a value in [0, 1] (0 = identical direction, 1 = orthogonal).
        """
        self._check_fitted("cosine_distance")
        x = np.asarray(embedding, dtype=np.float64)
        mu = self.mean

        norm_x = np.linalg.norm(x)
        norm_mu = np.linalg.norm(mu)

        if norm_x < 1e-12 or norm_mu < 1e-12:
            return 1.0  # treat zero-vector as maximally distant

        cos_sim = float(np.dot(x, mu) / (norm_x * norm_mu))
        cos_sim = float(np.clip(cos_sim, -1.0, 1.0))
        return 1.0 - cos_sim

    def euclidean_distance(self, embedding: np.ndarray) -> float:
        """
        Normalised Euclidean distance from the reference mean.
        Normalised by the mean of reference standard deviations to be
        roughly scale-invariant across feature dimensions.
        """
        self._check_fitted("euclidean_distance")
        x = np.asarray(embedding, dtype=np.float64)
        diff = x - self.mean
        # Normalise by feature-wise std derived from covariance diagonal
        stds = np.sqrt(np.diag(self.covariance) + 1e-12)
        normalised_diff = diff / stds
        return float(np.linalg.norm(normalised_diff))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_mahalanobis(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute Mahalanobis distances for a batch of embeddings."""
        diff = embeddings - self.mean  # (n, d)
        # (n, d) @ (d, d) = (n, d); element-wise × diff then sum over d
        dist_sq = np.einsum("nd,nd->n", diff @ self.precision, diff)
        dist_sq = np.clip(dist_sq, 0.0, None)
        return np.sqrt(dist_sq)

    @staticmethod
    def _estimate_covariance(
        embeddings: np.ndarray,
        n: int,
        d: int,
    ) -> np.ndarray:
        """
        Estimate a regularised covariance matrix using Ledoit–Wolf shrinkage
        (when scikit-learn is available) or diagonal regularisation.
        """
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf(assume_centered=False)
            lw.fit(embeddings)
            cov = lw.covariance_
            log.debug("Covariance estimated with Ledoit–Wolf shrinkage.")
        except Exception:
            # Fallback: empirical + diagonal regularisation
            cov = np.cov(embeddings.T, ddof=1)
            if cov.ndim == 0:
                cov = cov.reshape(1, 1)
            cov += config.COVARIANCE_REGULARISATION * np.eye(d)
            log.debug(
                "Covariance estimated empirically with λ=%.1e regularisation.",
                config.COVARIANCE_REGULARISATION,
            )
        return cov

    @staticmethod
    def _invert_covariance(cov: np.ndarray) -> np.ndarray:
        """
        Invert the covariance matrix with a pseudoinverse fallback for
        near-singular cases.
        """
        try:
            precision = np.linalg.inv(cov)
            # Symmetrise to reduce floating-point asymmetry
            precision = (precision + precision.T) / 2.0
        except np.linalg.LinAlgError:
            log.warning(
                "Covariance matrix is singular; using pseudoinverse. "
                "Consider increasing COVARIANCE_REGULARISATION."
            )
            precision = np.linalg.pinv(cov)
            precision = (precision + precision.T) / 2.0
        return precision

    def _check_fitted(self, method_name: str) -> None:
        if not self._fitted:
            raise ReferenceError(
                f"ReferenceDistribution.{method_name}() called before fit() or load(). "
                "Run 'python build_reference.py' to build the reference first."
            )
