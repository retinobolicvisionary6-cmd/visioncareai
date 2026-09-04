"""
OOD Detection Module — src package
===================================
Public API surface for the OOD module.
"""

from .ood import detect_ood, OODDetector
from .utils import OODError, ImageLoadError, EmbeddingError, ReferenceError, DimensionMismatchError

__all__ = [
    "detect_ood",
    "OODDetector",
    "OODError",
    "ImageLoadError",
    "EmbeddingError",
    "ReferenceError",
    "DimensionMismatchError",
]
