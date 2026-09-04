"""
Grad-CAM Module for Retinobolic.
Generates attention/evidence heatmap overlay for visual explainability.
"""
from src.gradcam import (
    GradCAM,
    generate_gradcam_overlay
)

__all__ = [
    "GradCAM",
    "generate_gradcam_overlay"
]
