"""
Grad-CAM (Gradient-weighted Class Activation Mapping) — VISIONARY6 VINAYAK Module.

Phase 10 — Grad-CAM visualization pipeline:
  1. Original / reference image
  2. Heatmap (standalone)
  3. Overlay (heatmap superimposed on fundus image)

All three outputs are saved to outputs/gradcam/.

IMPORTANT DISCLAIMER:
  Grad-CAM visualizations represent model attention and evidence regions that
  influenced the DR grade prediction. They are provided for clinical review aid
  and educational purposes ONLY.

  Do NOT interpret Grad-CAM as:
    - An exact lesion detector
    - Proof of causality between highlighted regions and the diagnosis
    - A clinically validated diagnostic tool

  Correct wording: "AI model attention / evidence visualization"
"""
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.config import GRADCAM_OUTPUT_DIR, IMAGE_SIZE, get_logger
from src.preprocess import crop_retina_circle

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Core Grad-CAM implementation
# ---------------------------------------------------------------------------

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for PyTorch models.

    Registers forward and backward hooks on the specified target layer to
    capture activations and gradients, then computes a weighted heatmap.

    Args:
        model:        PyTorch model (must be in eval mode for inference,
                      but grad must be enabled during heatmap generation).
        target_layer: The convolutional layer to hook. Use
                      model.get_target_layer_for_gradcam() for auto-selection.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model        = model
        self.target_layer = target_layer
        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None
        self._handles: list = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module, inp, out):
            self._activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach()

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self._handles = [h1, h2]

    def remove_hooks(self) -> None:
        """Cleans up registered hooks to prevent memory leaks."""
        for h in self._handles:
            h.remove()
        self._handles = []

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generates a 2D Grad-CAM heatmap normalized to [0.0, 1.0].

        Args:
            input_tensor: Preprocessed image tensor of shape [1, C, H, W].
            target_class: Class index to visualize. If None, uses argmax.

        Returns:
            2D float32 numpy array (spatial_h, spatial_w) normalized to [0, 1].
        """
        # Enable grad for backward pass (even during eval)
        input_tensor = input_tensor.requires_grad_(True)

        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = int(output.argmax(dim=1).item())

        # Backward pass for target class only
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()

        if self._gradients is None or self._activations is None:
            raise RuntimeError(
                "Grad-CAM hooks did not capture gradients/activations. "
                "Ensure target_layer is correct for this architecture."
            )

        # Global average pooling of gradients → channel importance weights
        weights = self._gradients.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]

        # Weighted sum of activation maps
        weighted_activations = (weights * self._activations).sum(dim=1, keepdim=True)  # [1, 1, h, w]
        heatmap = weighted_activations.squeeze().cpu().numpy()

        # ReLU (only positive contributions)
        heatmap = np.maximum(heatmap, 0)

        # Normalize to [0, 1]
        max_val = heatmap.max()
        if max_val > 0:
            heatmap /= max_val

        return heatmap.astype(np.float32)

    def __del__(self):
        self.remove_hooks()


# ---------------------------------------------------------------------------
# Overlay generator
# ---------------------------------------------------------------------------

def generate_gradcam_overlay(
    image_path: str | Path,
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    output_dir: Path = GRADCAM_OUTPUT_DIR,
    alpha: float = 0.4,
) -> str:
    """
    Generates and saves three Grad-CAM outputs:
      1. {stem}_original.jpg   — cropped fundus reference image
      2. {stem}_heatmap.jpg    — standalone Grad-CAM heatmap
      3. {stem}_overlay.jpg    — heatmap superimposed on fundus (primary output)

    Args:
        image_path:   Path to the original fundus image.
        model:        Loaded PyTorch model (must expose get_target_layer_for_gradcam()).
        input_tensor: Preprocessed tensor [1, 3, H, W] — same used for prediction.
        target_class: The class index to generate the heatmap for.
        output_dir:   Directory to save outputs.
        alpha:        Heatmap blend weight (0=only image, 1=only heatmap). Default 0.4.

    Returns:
        Path to the overlay image (primary Grad-CAM output) as a string.

    Raises:
        ValueError: If the image cannot be loaded.
        RuntimeError: If Grad-CAM hooks fail.
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Compute heatmap ---
    target_layer = model.get_target_layer_for_gradcam()
    gradcam = GradCAM(model, target_layer)
    try:
        heatmap = gradcam.generate_heatmap(input_tensor, target_class=target_class)
    finally:
        gradcam.remove_hooks()

    # --- Load and prepare original image ---
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise ValueError(f"Could not load image for Grad-CAM overlay: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    cropped_rgb = crop_retina_circle(img_rgb)
    display_w, display_h = IMAGE_SIZE
    cropped_rgb = cv2.resize(cropped_rgb, (display_w, display_h))

    # --- Resize heatmap to match display size ---
    heatmap_resized = cv2.resize(heatmap, (display_w, display_h))
    heatmap_uint8  = np.uint8(255 * heatmap_resized)
    heatmap_color  = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb    = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # --- Blend overlay ---
    overlay_rgb = np.uint8((1 - alpha) * cropped_rgb.astype(np.float32) +
                            alpha      * heatmap_rgb.astype(np.float32))

    # --- Save all three outputs ---
    stem = image_path.stem

    def _save(img_rgb: np.ndarray, suffix: str) -> Path:
        out_path = output_dir / f"{stem}{suffix}"
        cv2.imwrite(str(out_path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        return out_path

    _save(cropped_rgb,  "_original.jpg")
    _save(heatmap_rgb,  "_heatmap.jpg")
    overlay_path = _save(overlay_rgb, "_overlay.jpg")

    log.info(
        "Grad-CAM saved: original / heatmap / overlay for '%s' → class %d",
        image_path.name, target_class,
    )
    return str(overlay_path)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from configs.config import CHECKPOINT_PATH, MODEL_NAME, NUM_CLASSES, DROPOUT_RATE
    from src.model import build_model
    from src.preprocess import preprocess_single_image

    parser = argparse.ArgumentParser(description="Generate Grad-CAM for a fundus image.")
    parser.add_argument("image_path", help="Path to fundus image")
    parser.add_argument("--target_class", type=int, default=None,
                        help="Class index to visualize (default: argmax of prediction)")
    args = parser.parse_args()

    from configs.config import DEVICE
    model = build_model(
        architecture=MODEL_NAME, num_classes=NUM_CLASSES,
        pretrained=False, dropout_rate=DROPOUT_RATE, device=DEVICE, verbose=False,
    )
    if CHECKPOINT_PATH.exists():
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        print(f"Loaded checkpoint: {CHECKPOINT_PATH}")
    else:
        print(f"[WARNING] Checkpoint not found: {CHECKPOINT_PATH} — using base weights")

    tensor = preprocess_single_image(args.image_path).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze().cpu().numpy()
    pred_class = int(probs.argmax())
    target = args.target_class if args.target_class is not None else pred_class

    print(f"Prediction: grade={pred_class} | probs={probs.round(3)}")
    overlay = generate_gradcam_overlay(
        image_path=args.image_path,
        model=model,
        input_tensor=tensor,
        target_class=target,
    )
    print(f"Grad-CAM overlay saved: {overlay}")
    print("NOTE: This is model evidence visualization — NOT an exact lesion detector.")
