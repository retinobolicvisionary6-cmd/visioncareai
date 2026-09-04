"""
Tests — Embedding extraction
"""
import numpy as np
import pytest
from pathlib import Path

from src.embedding import (
    extract_embedding,
    get_extractor,
    ClassicalFundusExtractor,
    PretrainedTorchExtractor,
)
from src.utils import EmbeddingError, ImageLoadError

SAMPLE_ID = Path("sample_data/in_distribution")
ID_IMAGES = sorted(SAMPLE_ID.glob("*.png"))


# ---------------------------------------------------------------------------
# ClassicalFundusExtractor unit tests
# ---------------------------------------------------------------------------

class TestClassicalExtractor:

    def test_output_shape(self):
        """Embedding must be 1-D of length 71."""
        extractor = ClassicalFundusExtractor()
        dummy = np.random.rand(224, 224, 3).astype(np.float32)
        emb = extractor(dummy)
        assert emb.ndim == 1
        assert emb.shape[0] == 71

    def test_output_dtype(self):
        """Embedding must be float64."""
        extractor = ClassicalFundusExtractor()
        dummy = np.zeros((224, 224, 3), dtype=np.float32)
        emb = extractor(dummy)
        assert emb.dtype == np.float64

    def test_finite_output(self):
        """Embedding must be fully finite (no NaN / Inf)."""
        extractor = ClassicalFundusExtractor()
        dummy = np.random.rand(224, 224, 3).astype(np.float32)
        emb = extractor(dummy)
        assert np.isfinite(emb).all(), "Embedding contains NaN or Inf"

    def test_deterministic(self):
        """Same input must yield identical embedding."""
        extractor = ClassicalFundusExtractor()
        dummy = np.random.default_rng(123).random((224, 224, 3)).astype(np.float32)
        emb1 = extractor(dummy)
        emb2 = extractor(dummy)
        np.testing.assert_array_equal(emb1, emb2)

    def test_different_images_differ(self):
        """Different images should produce different embeddings."""
        extractor = ClassicalFundusExtractor()
        img1 = np.zeros((224, 224, 3), dtype=np.float32)
        img2 = np.ones((224, 224, 3), dtype=np.float32)
        emb1 = extractor(img1)
        emb2 = extractor(img2)
        assert not np.allclose(emb1, emb2), "Distinct images produced identical embeddings"

    def test_consistent_on_saved_samples(self):
        """Extract embeddings from saved sample images without error."""
        extractor = ClassicalFundusExtractor()
        for img_path in ID_IMAGES[:3]:
            from src.utils import load_and_preprocess
            arr = load_and_preprocess(img_path)
            emb = extractor(arr)
            assert emb.shape[0] == 71
            assert np.isfinite(emb).all()


# ---------------------------------------------------------------------------
# extract_embedding() public API tests
# ---------------------------------------------------------------------------

class TestExtractEmbeddingAPI:

    def test_from_path(self):
        """extract_embedding accepts a file path."""
        emb = extract_embedding(ID_IMAGES[0], extractor_type="classical")
        assert emb.ndim == 1
        assert emb.size > 0

    def test_from_string_path(self):
        """extract_embedding accepts a string path."""
        emb = extract_embedding(str(ID_IMAGES[0]), extractor_type="classical")
        assert emb.ndim == 1

    def test_from_uint8_array(self):
        """extract_embedding accepts a pre-loaded uint8 numpy array."""
        dummy = np.full((224, 224, 3), 128, dtype=np.uint8)
        emb = extract_embedding(dummy, extractor_type="classical")
        assert emb.ndim == 1

    def test_from_float32_array(self):
        """extract_embedding accepts a float32 numpy array in [0,1]."""
        dummy = np.random.rand(224, 224, 3).astype(np.float32)
        emb = extract_embedding(dummy, extractor_type="classical")
        assert emb.ndim == 1

    def test_missing_file_raises(self):
        """Missing file must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            extract_embedding("nonexistent/image.png", extractor_type="classical")

    def test_invalid_extension_raises(self):
        """Unsupported extension must raise ImageLoadError."""
        with pytest.raises(ImageLoadError):
            extract_embedding("sample_data/in_distribution/fake.xyz", extractor_type="classical")

    def test_unknown_extractor_raises(self):
        """Unknown extractor type must raise EmbeddingError."""
        with pytest.raises(EmbeddingError):
            extract_embedding(ID_IMAGES[0], extractor_type="does_not_exist")

    def test_extractor_caching(self):
        """get_extractor() returns the same instance on repeated calls."""
        e1 = get_extractor("classical")
        e2 = get_extractor("classical")
        assert e1 is e2, "Extractor instance was not cached"
