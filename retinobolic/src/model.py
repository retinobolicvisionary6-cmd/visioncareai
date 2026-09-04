"""
Model Factory — VISIONARY6 VINAYAK Module.

Implements a modular transfer-learning classifier for 4-class DR grading.

Supported architectures:
  - efficientnet_b0   (default — balanced accuracy/size)
  - resnet50          (ResNet baseline)
  - mobilenet_v3_small (lightweight/mobile experiments)

Key design principles:
  - Never built from scratch; always uses pretrained ImageNet weights.
  - The final classification head is replaced for 4-class output.
  - Target layer for Grad-CAM is automatically selected per architecture.
  - Model is fully replaceable without touching inference or training code.
"""
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils import get_logger, print_model_summary

log = get_logger(__name__)


class DiabeticRetinopathyClassifier(nn.Module):
    """
    4-Class Transfer Learning Classifier for Diabetic Retinopathy.

    Wraps a pretrained backbone (EfficientNet / ResNet / MobileNet) and
    replaces its final classification head with a dropout + linear layer
    for 4-class DR grading.

    The number of output classes is configurable (default: 4).

    Args:
        architecture: One of 'efficientnet_b0', 'resnet50', 'mobilenet_v3_small'.
        num_classes:  Number of output classes (should always be 4 per contract).
        pretrained:   If True, loads ImageNet weights for the backbone.
        dropout_rate: Dropout probability applied before the final linear layer.
    """

    SUPPORTED_ARCHITECTURES = {
        "efficientnet_b0",
        "resnet50",
        "mobilenet_v3_small",
    }

    def __init__(
        self,
        architecture: str = "efficientnet_b0",
        num_classes: int = 4,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.architecture = architecture.lower().strip()
        self.num_classes = num_classes

        if self.architecture not in self.SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"Unsupported architecture: '{architecture}'. "
                f"Choose from: {sorted(self.SUPPORTED_ARCHITECTURES)}"
            )

        if "efficientnet" in self.architecture:
            self.backbone, self._target_layer = self._build_efficientnet(
                pretrained, num_classes, dropout_rate
            )
        elif "resnet" in self.architecture:
            self.backbone, self._target_layer = self._build_resnet(
                pretrained, num_classes, dropout_rate
            )
        elif "mobilenet" in self.architecture:
            self.backbone, self._target_layer = self._build_mobilenet(
                pretrained, num_classes, dropout_rate
            )

    # ------------------------------------------------------------------
    # Architecture builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_efficientnet(
        pretrained: bool, num_classes: int, dropout_rate: float
    ) -> tuple[nn.Module, nn.Module]:
        """EfficientNet-B0 with custom 4-class head."""
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, num_classes),
        )
        target_layer = backbone.features[-1]   # last convolutional block
        return backbone, target_layer

    @staticmethod
    def _build_resnet(
        pretrained: bool, num_classes: int, dropout_rate: float
    ) -> tuple[nn.Module, nn.Module]:
        """ResNet-50 with custom 4-class head."""
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes),
        )
        target_layer = backbone.layer4[-1]     # last residual block
        return backbone, target_layer

    @staticmethod
    def _build_mobilenet(
        pretrained: bool, num_classes: int, dropout_rate: float
    ) -> tuple[nn.Module, nn.Module]:
        """MobileNet-V3-Small with custom 4-class head."""
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)
        in_features = backbone.classifier[3].in_features
        backbone.classifier[3] = nn.Linear(in_features, num_classes)
        target_layer = backbone.features[-1]   # last convolutional block
        return backbone, target_layer

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass through the backbone."""
        return self.backbone(x)

    # ------------------------------------------------------------------
    # Grad-CAM hook point
    # ------------------------------------------------------------------

    def get_target_layer_for_gradcam(self) -> nn.Module:
        """
        Returns the final convolutional feature layer suitable for Grad-CAM.

        This layer is architecture-specific and is automatically selected
        during model construction.
        """
        return self._target_layer


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_model(
    architecture: str = "efficientnet_b0",
    num_classes: int = 4,
    pretrained: bool = True,
    dropout_rate: float = 0.3,
    device: Optional[str] = None,
    verbose: bool = True,
) -> DiabeticRetinopathyClassifier:
    """
    Instantiates a DiabeticRetinopathyClassifier and moves it to the target device.

    Args:
        architecture: Model backbone name.
        num_classes:  Number of output classes (4 per contract).
        pretrained:   Load ImageNet weights for backbone.
        dropout_rate: Dropout before classification head.
        device:       'cuda' or 'cpu'. Auto-detected if None.
        verbose:      If True, prints model summary.

    Returns:
        Model on the specified device, in train mode.
    """
    if device is None:
        from configs.config import DEVICE
        device = DEVICE

    log.info("Building model: %s | classes=%d | pretrained=%s | device=%s",
             architecture, num_classes, pretrained, device)

    model = DiabeticRetinopathyClassifier(
        architecture=architecture,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
    )
    model = model.to(device)

    if verbose:
        print_model_summary(model, architecture)

    return model
