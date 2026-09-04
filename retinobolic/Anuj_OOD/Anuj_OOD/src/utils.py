"""
OOD Detection Module — Input Validation & Utilities
====================================================
Robust image loading, format validation, normalisation,
and numerical sanity checks used throughout the pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, UnidentifiedImageError

from . import config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exception hierarchy
# ---------------------------------------------------------------------------

class OODError(Exception):
    """Base exception for all OOD module errors."""


class ImageLoadError(OODError):
    """Raised when an image cannot be loaded or decoded."""


class EmbeddingError(OODError):
    """Raised when embedding extraction fails or produces invalid output."""


class ReferenceError(OODError):
    """Raised when the reference distribution is missing, corrupt, or incompatible."""


class DimensionMismatchError(OODError):
    """Raised when embedding dimensionality does not match the reference."""


# ---------------------------------------------------------------------------
# Image loading and validation
# ---------------------------------------------------------------------------

def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    Load an image from *image_path* and return it as an (H, W, 3) uint8 NumPy
    array in RGB colour order.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    ImageLoadError
        If the file exists but cannot be decoded as an image, or the image
        has an unsupported colour mode.
    """
    path = Path(image_path)

    if path.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
        raise ImageLoadError(
            f"Unsupported file extension '{path.suffix}'. "
            f"Supported: {config.SUPPORTED_EXTENSIONS}"
        )

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    try:
        img = Image.open(path)
        img.verify()  # checks for truncation / corruption
    except (UnidentifiedImageError, Exception) as exc:
        raise ImageLoadError(f"Cannot decode image '{path}': {exc}") from exc

    # Re-open after verify() (verify() consumes the file handle)
    try:
        img = Image.open(path).convert("RGB")
    except Exception as exc:
        raise ImageLoadError(f"Cannot convert image to RGB '{path}': {exc}") from exc

    return np.asarray(img, dtype=np.uint8)


def preprocess_image(
    image_array: np.ndarray,
    target_size: tuple[int, int] = None,
) -> np.ndarray:
    """
    Resize an (H, W, 3) uint8 image to *target_size* and return a float32
    array normalised to [0, 1].

    Parameters
    ----------
    image_array : np.ndarray
        Raw uint8 RGB image array.
    target_size : (int, int) or None
        (height, width) to resize to.  Defaults to config.IMAGE_SIZE.

    Returns
    -------
    np.ndarray of shape (H, W, 3), dtype float32, values in [0.0, 1.0].
    """
    if target_size is None:
        target_size = config.IMAGE_SIZE

    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ImageLoadError(
            f"Expected (H, W, 3) array, got shape {image_array.shape}"
        )

    h, w = target_size
    pil_img = Image.fromarray(image_array).resize((w, h), Image.BILINEAR)
    arr = np.asarray(pil_img, dtype=np.float32) / 255.0
    return arr


def load_and_preprocess(
    image_path: Union[str, Path],
    target_size: tuple[int, int] = None,
) -> np.ndarray:
    """
    Convenience: load image from disk and return preprocessed float32 array.
    Combines load_image() and preprocess_image().
    """
    raw = load_image(image_path)
    return preprocess_image(raw, target_size)


# ---------------------------------------------------------------------------
# Embedding sanity checks
# ---------------------------------------------------------------------------

def validate_embedding(embedding: np.ndarray, context: str = "") -> np.ndarray:
    """
    Validate that *embedding* is a 1-D finite float array.

    Raises
    ------
    EmbeddingError
        If the embedding is empty, multidimensional, or contains NaN / Inf.
    """
    prefix = f"[{context}] " if context else ""

    if embedding is None:
        raise EmbeddingError(f"{prefix}Embedding is None.")

    embedding = np.asarray(embedding, dtype=np.float64)

    if embedding.ndim != 1:
        raise EmbeddingError(
            f"{prefix}Embedding must be 1-D, got shape {embedding.shape}."
        )

    if embedding.size == 0:
        raise EmbeddingError(f"{prefix}Embedding is empty (zero-length vector).")

    if not np.isfinite(embedding).all():
        n_bad = int(np.sum(~np.isfinite(embedding)))
        raise EmbeddingError(
            f"{prefix}Embedding contains {n_bad} non-finite value(s) (NaN / Inf)."
        )

    return embedding


def check_dimension_match(
    embedding: np.ndarray,
    reference_dim: int,
    context: str = "",
) -> None:
    """
    Raise DimensionMismatchError if *embedding* does not match *reference_dim*.
    """
    prefix = f"[{context}] " if context else ""
    if embedding.shape[0] != reference_dim:
        raise DimensionMismatchError(
            f"{prefix}Embedding dimension {embedding.shape[0]} does not match "
            f"reference dimension {reference_dim}. "
            "Re-build the reference with the same extractor type."
        )
