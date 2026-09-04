"""
Tests — Input validation and edge cases
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.utils import (
    ImageLoadError,
    EmbeddingError,
    DimensionMismatchError,
    load_image,
    load_and_preprocess,
    validate_embedding,
    check_dimension_match,
)
from src.reference import ReferenceDistribution
from src.utils import ReferenceError


class TestImageLoading:

    def test_load_valid_png(self):
        """Valid PNG should load as (H, W, 3) uint8 array."""
        imgs = sorted(Path("sample_data/in_distribution").glob("*.png"))
        arr = load_image(imgs[0])
        assert arr.ndim == 3
        assert arr.shape[2] == 3
        assert arr.dtype == np.uint8

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_image("does_not_exist.png")

    def test_unsupported_extension_raises(self):
        with pytest.raises(ImageLoadError, match="Unsupported"):
            load_image("sample_data/in_distribution/fundus_01.xyz")

    def test_corrupt_file_raises(self):
        """A text file renamed to .png should raise ImageLoadError."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"not an image at all --- corrupt data ###")
            tmp = Path(f.name)
        try:
            with pytest.raises(ImageLoadError):
                load_image(tmp)
        finally:
            tmp.unlink(missing_ok=True)

    def test_preprocess_output_shape_and_range(self):
        """Preprocessed image must be float32 (H, W, 3) in [0, 1]."""
        imgs = sorted(Path("sample_data/in_distribution").glob("*.png"))
        arr = load_and_preprocess(imgs[0], target_size=(224, 224))
        assert arr.shape == (224, 224, 3)
        assert arr.dtype == np.float32
        assert arr.min() >= 0.0
        assert arr.max() <= 1.0 + 1e-6


class TestEmbeddingValidation:

    def test_valid_embedding_passes(self):
        emb = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        result = validate_embedding(emb)
        assert result.ndim == 1

    def test_none_raises(self):
        with pytest.raises(EmbeddingError, match="None"):
            validate_embedding(None)

    def test_empty_raises(self):
        with pytest.raises(EmbeddingError, match="empty"):
            validate_embedding(np.array([]))

    def test_multidimensional_raises(self):
        with pytest.raises(EmbeddingError, match="1-D"):
            validate_embedding(np.ones((3, 4)))

    def test_nan_raises(self):
        emb = np.array([1.0, float("nan"), 2.0])
        with pytest.raises(EmbeddingError, match="non-finite"):
            validate_embedding(emb)

    def test_inf_raises(self):
        emb = np.array([1.0, float("inf"), 2.0])
        with pytest.raises(EmbeddingError, match="non-finite"):
            validate_embedding(emb)

    def test_neg_inf_raises(self):
        emb = np.array([float("-inf"), 1.0])
        with pytest.raises(EmbeddingError, match="non-finite"):
            validate_embedding(emb)


class TestDimensionMismatch:

    def test_matching_dim_passes(self):
        emb = np.ones(71, dtype=np.float64)
        check_dimension_match(emb, 71)  # should not raise

    def test_mismatched_dim_raises(self):
        emb = np.ones(50, dtype=np.float64)
        with pytest.raises(DimensionMismatchError):
            check_dimension_match(emb, 71)


class TestReferenceDistribution:

    def test_fit_with_valid_data(self):
        rng = np.random.default_rng(0)
        embeddings = rng.random((10, 5)).astype(np.float64)
        ref = ReferenceDistribution()
        ref.fit(embeddings)
        assert ref.is_fitted
        assert ref.embedding_dim == 5
        assert ref.n_samples == 10

    def test_fit_requires_at_least_2_samples(self):
        with pytest.raises(ReferenceError, match="at least 2"):
            ref = ReferenceDistribution()
            ref.fit(np.random.rand(1, 5))

    def test_fit_rejects_nan(self):
        embeddings = np.random.rand(10, 5)
        embeddings[3, 2] = float("nan")
        with pytest.raises(ReferenceError, match="NaN or Inf"):
            ReferenceDistribution().fit(embeddings)

    def test_save_and_load_roundtrip(self):
        """Saved reference distribution must load with identical statistics."""
        rng = np.random.default_rng(42)
        embeddings = rng.random((15, 8)).astype(np.float64)
        ref = ReferenceDistribution()
        ref.fit(embeddings)

        with tempfile.TemporaryDirectory() as tmpdir:
            stats_path = Path(tmpdir) / "stats.json"
            emb_path = Path(tmpdir) / "embs.npy"
            ref.save(stats_path, emb_path)

            ref2 = ReferenceDistribution.load(stats_path, emb_path)

        assert ref2.is_fitted
        np.testing.assert_allclose(ref2.mean, ref.mean)
        np.testing.assert_allclose(ref2.precision, ref.precision, rtol=1e-5)
        assert ref2.n_samples == ref.n_samples
        assert ref2.embedding_dim == ref.embedding_dim

    def test_load_missing_file_raises(self):
        with pytest.raises(ReferenceError, match="not found"):
            ReferenceDistribution.load("nonexistent/stats.json")

    def test_load_corrupt_json_raises(self):
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False
        ) as f:
            f.write("{bad json{{")
            tmp = Path(f.name)
        try:
            with pytest.raises(ReferenceError, match="Cannot parse"):
                ReferenceDistribution.load(tmp)
        finally:
            tmp.unlink(missing_ok=True)

    def test_load_incomplete_json_raises(self):
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False
        ) as f:
            json.dump({"n_samples": 5}, f)
            tmp = Path(f.name)
        try:
            with pytest.raises(ReferenceError, match="incomplete or corrupt"):
                ReferenceDistribution.load(tmp)
        finally:
            tmp.unlink(missing_ok=True)

    def test_unfitted_raises_on_distance(self):
        ref = ReferenceDistribution()
        with pytest.raises(ReferenceError, match="fit\\(\\) or load\\(\\)"):
            ref.mahalanobis_distance(np.ones(5))

    def test_percentiles_populated(self):
        rng = np.random.default_rng(7)
        embeddings = rng.random((20, 6)).astype(np.float64)
        ref = ReferenceDistribution()
        ref.fit(embeddings)
        for key in ("p50", "p90", "p95", "p99", "p100"):
            assert key in ref.percentiles
            assert np.isfinite(ref.percentiles[key])
