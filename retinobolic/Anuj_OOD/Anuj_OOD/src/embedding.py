"""
OOD Detection Module — Embedding Extraction
============================================
Modular feature extraction interface for OOD detection.

Three extractor tiers (in priority order):
  1. DRModelExtractorHook  — plugs into Vinayak's trained DR model feature layer.
  2. PretrainedTorchExtractor — lightweight PyTorch backbone (MobileNet-V2 / ResNet-18).
  3. ClassicalFundusExtractor — multi-feature statistical extractor, zero deep-learning deps.

Active extractor is controlled by config.EXTRACTOR_TYPE.

Public interface
----------------
    extract_embedding(image_path_or_array, extractor_type=None) -> np.ndarray

The returned embedding is always a 1-D float64 NumPy array.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

from . import config
from .utils import (
    ImageLoadError,
    EmbeddingError,
    load_and_preprocess,
    load_image,
    preprocess_image,
    validate_embedding,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BaseEmbeddingExtractor(ABC):
    """
    All extractor implementations must produce a fixed-length 1-D float64
    embedding from an (H, W, 3) float32 image array in [0, 1].
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for logging and reference metadata."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the produced embedding."""

    @abstractmethod
    def _extract(self, image_array: np.ndarray) -> np.ndarray:
        """
        Core extraction method.  Receives a (H, W, 3) float32 image normalised
        to [0, 1].  Must return a 1-D float64 array of length self.embedding_dim.
        """

    def __call__(self, image_array: np.ndarray) -> np.ndarray:
        embedding = self._extract(image_array)
        return validate_embedding(embedding, context=self.name)


# ---------------------------------------------------------------------------
# Tier 3 — Classical multi-feature extractor (no deep-learning deps)
# ---------------------------------------------------------------------------

class ClassicalFundusExtractor(BaseEmbeddingExtractor):
    """
    Produces a rich, fixed-length feature vector from fundus-relevant
    image statistics without requiring PyTorch or any trained model.

    Feature groups
    --------------
    1. Global colour moments (mean, std, skewness per RGB channel) — 9 dims
    2. HSV colour moments (mean, std per H,S,V channel) — 6 dims
    3. Contrast / brightness statistics — 4 dims
    4. Radial profile (mean intensity at 8 concentric rings in each RGB channel) — 24 dims
    5. Gradient statistics (Sobel magnitude mean, std, percentile) — 3 dims
    6. High-frequency energy (Laplacian variance proxy) — 1 dim
    7. Colour histogram (8 bins per channel, L1-normalised) — 24 dims

    Total: 71 dims
    """

    _DIM = 71

    @property
    def name(self) -> str:
        return "ClassicalFundusExtractor"

    @property
    def embedding_dim(self) -> int:
        return self._DIM

    def _extract(self, image_array: np.ndarray) -> np.ndarray:  # (H, W, 3) float32
        features: list[np.ndarray] = []

        # --- 1. Global colour moments per RGB channel (9 dims) ---
        for c in range(3):
            ch = image_array[:, :, c].astype(np.float64)
            mean = ch.mean()
            std = ch.std() + 1e-8
            skew = float(np.mean(((ch - mean) / std) ** 3))
            features.append(np.array([mean, std, skew]))

        # --- 2. HSV colour moments (6 dims) ---
        pil_hsv = Image.fromarray((image_array * 255).astype(np.uint8)).convert("HSV")
        hsv = np.asarray(pil_hsv, dtype=np.float64) / 255.0
        for c in range(3):
            ch = hsv[:, :, c]
            features.append(np.array([ch.mean(), ch.std() + 1e-8]))

        # --- 3. Contrast / brightness statistics (4 dims) ---
        gray = 0.2989 * image_array[:, :, 0] + \
               0.5870 * image_array[:, :, 1] + \
               0.1140 * image_array[:, :, 2]
        gray64 = gray.astype(np.float64)
        features.append(np.array([
            float(gray64.mean()),
            float(gray64.std() + 1e-8),
            float(np.percentile(gray64, 5)),
            float(np.percentile(gray64, 95)),
        ]))

        # --- 4. Radial profile — 8 rings × 3 channels (24 dims) ---
        H, W = image_array.shape[:2]
        cy, cx = H / 2.0, W / 2.0
        max_r = min(cy, cx)
        ys, xs = np.mgrid[0:H, 0:W]
        r = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2) / (max_r + 1e-8)
        n_rings = 8
        ring_feats = np.zeros(n_rings * 3, dtype=np.float64)
        for i in range(n_rings):
            r_lo = i / n_rings
            r_hi = (i + 1) / n_rings
            mask = (r >= r_lo) & (r < r_hi)
            for c in range(3):
                vals = image_array[:, :, c][mask]
                ring_feats[i * 3 + c] = vals.mean() if vals.size > 0 else 0.0
        features.append(ring_feats)

        # --- 5. Gradient statistics — Sobel approximation (3 dims) ---
        # Sobel-x and Sobel-y via simple finite differences
        gray_f = gray.astype(np.float64)
        gx = np.abs(np.diff(gray_f, axis=1, prepend=gray_f[:, :1]))
        gy = np.abs(np.diff(gray_f, axis=0, prepend=gray_f[:1, :]))
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)
        features.append(np.array([
            float(grad_mag.mean()),
            float(grad_mag.std() + 1e-8),
            float(np.percentile(grad_mag, 90)),
        ]))

        # --- 6. High-frequency energy — Laplacian variance (1 dim) ---
        lap = (
            -gray_f[:-2, 1:-1] - gray_f[2:, 1:-1]
            - gray_f[1:-1, :-2] - gray_f[1:-1, 2:]
            + 4 * gray_f[1:-1, 1:-1]
        )
        features.append(np.array([float(lap.var())]))

        # --- 7. Colour histograms — 8 bins × 3 channels, L1-normalised (24 dims) ---
        n_bins = 8
        hist_feats = np.zeros(n_bins * 3, dtype=np.float64)
        for c in range(3):
            h, _ = np.histogram(image_array[:, :, c].ravel(), bins=n_bins, range=(0.0, 1.0))
            h = h.astype(np.float64)
            total = h.sum() + 1e-8
            hist_feats[c * n_bins:(c + 1) * n_bins] = h / total
        features.append(hist_feats)

        embedding = np.concatenate(features).astype(np.float64)
        assert embedding.shape[0] == self._DIM, (
            f"ClassicalFundusExtractor dim mismatch: {embedding.shape[0]} != {self._DIM}"
        )
        return embedding


# ---------------------------------------------------------------------------
# Tier 2 — Pretrained PyTorch encoder (optional, graceful fallback)
# ---------------------------------------------------------------------------

class PretrainedTorchExtractor(BaseEmbeddingExtractor):
    """
    Extracts a global average-pooled feature vector from a pretrained
    MobileNet-V2 or ResNet-18 backbone (torchvision).

    Falls back to ClassicalFundusExtractor at *instantiation time* if
    PyTorch / torchvision are not available.
    """

    def __init__(
        self,
        model_name: str = None,
        layer_name: str = None,
    ) -> None:
        model_name = model_name or config.PRETRAINED_MODEL_NAME
        layer_name = layer_name or config.PRETRAINED_FEATURE_LAYER

        try:
            import torch
            import torchvision.models as tv_models
            import torchvision.transforms as T
        except ImportError:
            log.warning(
                "PyTorch / torchvision not installed. "
                "Falling back to ClassicalFundusExtractor."
            )
            self._fallback = ClassicalFundusExtractor()
            self._torch_available = False
            return

        self._torch_available = True
        self._torch = torch
        self._model_name = model_name

        # Build backbone
        if model_name == "mobilenet_v2":
            backbone = tv_models.mobilenet_v2(
                weights=tv_models.MobileNet_V2_Weights.IMAGENET1K_V1
            )
            self._features = backbone.features
            self._dim = 1280
        elif model_name == "resnet18":
            backbone = tv_models.resnet18(
                weights=tv_models.ResNet18_Weights.IMAGENET1K_V1
            )
            # Strip classifier — keep up to avgpool
            self._features = torch.nn.Sequential(*list(backbone.children())[:-1])
            self._dim = 512
        else:
            raise EmbeddingError(f"Unsupported pretrained model: {model_name}")

        self._features.eval()

        # ImageNet normalisation
        self._transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        log.info("PretrainedTorchExtractor: loaded %s (%d-dim).", model_name, self._dim)

    @property
    def name(self) -> str:
        if getattr(self, "_torch_available", False):
            return f"PretrainedTorchExtractor({self._model_name})"
        return f"PretrainedTorchExtractor(fallback→{self._fallback.name})"

    @property
    def embedding_dim(self) -> int:
        if getattr(self, "_torch_available", False):
            return self._dim
        return self._fallback.embedding_dim

    def _extract(self, image_array: np.ndarray) -> np.ndarray:
        if not getattr(self, "_torch_available", False):
            return self._fallback._extract(image_array)

        import torch

        pil_img = Image.fromarray((image_array * 255).astype(np.uint8))
        tensor = self._transform(pil_img).unsqueeze(0)  # (1, 3, H, W)

        with torch.no_grad():
            feat = self._features(tensor)
            # Global average pool to 1-D
            if feat.ndim == 4:
                feat = feat.mean(dim=[2, 3])
            embedding = feat.squeeze().cpu().numpy().astype(np.float64)

        return embedding


# ---------------------------------------------------------------------------
# Tier 1 — DR Model hook (Vinayak integration placeholder)
# ---------------------------------------------------------------------------

class DRModelExtractorHook(BaseEmbeddingExtractor):
    """
    Pluggable adapter that extracts embeddings from Vinayak's trained DR model.

    When the model checkpoint is available (config.DR_MODEL_CHECKPOINT is set),
    this extractor registers a forward hook on config.DR_MODEL_FEATURE_LAYER
    and returns the resulting activation as the embedding.

    Falls back to PretrainedTorchExtractor → ClassicalFundusExtractor when
    the checkpoint is not available.

    -----------------------------------------------------------------------
    INTEGRATION NOTE FOR VINAYAK
    -----------------------------------------------------------------------
    When the DR model is ready:
      1. Set config.DR_MODEL_CHECKPOINT to the .pth path.
      2. Set config.DR_MODEL_FEATURE_LAYER to the desired layer name.
      3. Ensure the model architecture is importable or provide a factory fn.
    -----------------------------------------------------------------------
    """

    def __init__(self) -> None:
        checkpoint = config.DR_MODEL_CHECKPOINT

        if not checkpoint:
            log.warning(
                "DR model checkpoint not configured (config.DR_MODEL_CHECKPOINT is empty). "
                "Falling back to PretrainedTorchExtractor."
            )
            self._fallback = PretrainedTorchExtractor()
            self._dr_available = False
            return

        try:
            import torch
            state = torch.load(checkpoint, map_location="cpu")
            log.info("DRModelExtractorHook: checkpoint loaded from %s.", checkpoint)
            # TODO: reconstruct model architecture here when Vinayak's model
            # interface is finalised.
            raise NotImplementedError(
                "DRModelExtractorHook requires Vinayak's model class. "
                "Please update this extractor once the DR model interface is available."
            )
        except (FileNotFoundError, NotImplementedError) as exc:
            log.warning("DRModelExtractorHook unavailable (%s). Falling back.", exc)
            self._fallback = PretrainedTorchExtractor()
            self._dr_available = False

    @property
    def name(self) -> str:
        if getattr(self, "_dr_available", False):
            return "DRModelExtractorHook"
        return f"DRModelExtractorHook(fallback→{self._fallback.name})"

    @property
    def embedding_dim(self) -> int:
        if getattr(self, "_dr_available", False):
            return self._dim  # set when model is loaded
        return self._fallback.embedding_dim

    def _extract(self, image_array: np.ndarray) -> np.ndarray:
        if not getattr(self, "_dr_available", False):
            return self._fallback._extract(image_array)
        raise NotImplementedError("DR model hook not yet wired.")


# ---------------------------------------------------------------------------
# Extractor registry and factory
# ---------------------------------------------------------------------------

_EXTRACTOR_REGISTRY: dict[str, type[BaseEmbeddingExtractor]] = {
    "classical": ClassicalFundusExtractor,
    "pretrained": PretrainedTorchExtractor,
    "dr_model": DRModelExtractorHook,
}

# Module-level singleton cache — avoids reloading models on repeated calls
_extractor_cache: dict[str, BaseEmbeddingExtractor] = {}


def get_extractor(extractor_type: str = None) -> BaseEmbeddingExtractor:
    """
    Return a (cached) instance of the requested extractor.

    Parameters
    ----------
    extractor_type : str or None
        One of "classical", "pretrained", "dr_model".
        Defaults to config.EXTRACTOR_TYPE.
    """
    extractor_type = extractor_type or config.EXTRACTOR_TYPE
    if extractor_type not in _EXTRACTOR_REGISTRY:
        raise EmbeddingError(
            f"Unknown extractor type '{extractor_type}'. "
            f"Valid options: {list(_EXTRACTOR_REGISTRY)}"
        )

    if extractor_type not in _extractor_cache:
        log.info("Instantiating extractor: %s", extractor_type)
        _extractor_cache[extractor_type] = _EXTRACTOR_REGISTRY[extractor_type]()

    return _extractor_cache[extractor_type]


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def extract_embedding(
    image_path_or_array: Union[str, Path, np.ndarray],
    extractor_type: str = None,
) -> np.ndarray:
    """
    Extract a 1-D float64 embedding from an image.

    Parameters
    ----------
    image_path_or_array : str | Path | np.ndarray
        Path to a fundus image, or a pre-loaded (H, W, 3) uint8 array.
    extractor_type : str or None
        Override the extractor.  Defaults to config.EXTRACTOR_TYPE.

    Returns
    -------
    np.ndarray
        1-D float64 embedding vector.

    Raises
    ------
    ImageLoadError   — if the image cannot be loaded or decoded.
    EmbeddingError   — if extraction fails or produces invalid output.
    """
    extractor = get_extractor(extractor_type)

    if isinstance(image_path_or_array, np.ndarray):
        if image_path_or_array.dtype == np.uint8:
            image_array = preprocess_image(image_path_or_array)
        else:
            image_array = image_path_or_array.astype(np.float32)
    else:
        image_array = load_and_preprocess(image_path_or_array)

    embedding = extractor(image_array)
    return embedding
