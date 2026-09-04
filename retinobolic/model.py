"""
Model Architecture Module for Retinobolic.
Transfer learning backbones (EfficientNet-B0 baseline, ResNet, MobileNet) with 4-class output head.
"""
from src.model import (
    DiabeticRetinopathyClassifier,
    build_model
)

__all__ = [
    "DiabeticRetinopathyClassifier",
    "build_model"
]
