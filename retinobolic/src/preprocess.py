"""
Preprocessing Pipeline for Retinal Fundus Images — VINAYAK Module.

Operations:
  - Black-border circular crop (remove dark padding around fundus)
  - Resize to configurable target size
  - ImageNet pixel normalization
  - Clinically-appropriate data augmentation (training only)

IMPORTANT:
  - Validation and test images receive ONLY deterministic transforms (resize + normalize).
  - All augmentation is configurable via AUGMENTATION_CONFIG in configs/config.py.
  - Unrealistic distortions are explicitly excluded.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.config import (
    IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, AUGMENTATION_CONFIG
)

# ---------------------------------------------------------------------------
# CLAHE Preprocessing (Accuracy Boost for Microaneurysms)
# ---------------------------------------------------------------------------

def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
    This highlights blood vessels, microaneurysms, and exudates in fundus images.
    
    Args:
        image: RGB numpy array (H, W, 3).
    Returns:
        RGB numpy array with enhanced contrast.
    """
    if image.ndim == 2:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        return clahe.apply(image)
        
    # Convert RGB to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    cl = clahe.apply(l)
    
    # Merge channels and convert back to RGB
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    return final

# ---------------------------------------------------------------------------
# Black-border / circular retina cropping
# ---------------------------------------------------------------------------

def crop_retina_circle(image: np.ndarray, tol: int = 7) -> np.ndarray:
    """
    Removes the dark black border surrounding the circular fundus area.

    The APTOS dataset often has large black padded borders that carry no
    clinical information. This crop retains only the circular retinal region.

    Args:
        image: RGB numpy array (H, W, 3) or grayscale (H, W).
        tol:   Intensity threshold below which pixels are considered border.

    Returns:
        Cropped numpy array. If the crop would produce an empty result,
        the original image is returned unchanged.
    """
    if image.ndim == 2:
        # Grayscale path
        mask = image > tol
        rows_ok = mask.any(axis=1)
        cols_ok = mask.any(axis=0)
        if not rows_ok.any() or not cols_ok.any():
            return image
        return image[np.ix_(rows_ok, cols_ok)]

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    rows_ok = mask.any(axis=1)
    cols_ok = mask.any(axis=0)

    # Guard against a fully-dark or tiny image
    if not rows_ok.any() or not cols_ok.any():
        return image

    cropped_shape = image[:, :, 0][np.ix_(rows_ok, cols_ok)].shape
    if cropped_shape[0] == 0 or cropped_shape[1] == 0:
        return image

    ch0 = image[:, :, 0][np.ix_(rows_ok, cols_ok)]
    ch1 = image[:, :, 1][np.ix_(rows_ok, cols_ok)]
    ch2 = image[:, :, 2][np.ix_(rows_ok, cols_ok)]
    return np.stack([ch0, ch1, ch2], axis=-1)


# ---------------------------------------------------------------------------
# Transform factories
# ---------------------------------------------------------------------------

def get_train_transforms(image_size: tuple[int, int] = IMAGE_SIZE) -> T.Compose:
    """
    Returns the training augmentation pipeline.

    Augmentations are moderate and clinically appropriate:
      - Horizontal flip: valid (fundus can be imaged from either eye)
      - Vertical flip:   valid (fundus symmetric around horizontal axis)
      - Small rotation:  ±15° (fundus orientation varies)
      - Colour jitter:   subtle brightness/contrast adjustment only
      - No elastic/perspective distortion (alters lesion morphology)

    Configuration is driven by AUGMENTATION_CONFIG in configs/config.py.
    """
    aug = AUGMENTATION_CONFIG
    transforms_list = [T.Resize(image_size)]

    if aug.get("random_horizontal_flip", True):
        transforms_list.append(T.RandomHorizontalFlip(p=0.5))

    if aug.get("random_vertical_flip", True):
        transforms_list.append(T.RandomVerticalFlip(p=0.5))

    rot = aug.get("rotation_degrees", 15)
    if rot > 0:
        transforms_list.append(T.RandomRotation(degrees=rot))

    brightness = aug.get("brightness_jitter", 0.1)
    contrast   = aug.get("contrast_jitter", 0.1)
    saturation = aug.get("saturation_jitter", 0.1)
    hue        = aug.get("hue_jitter", 0.0)
    if any(v > 0 for v in [brightness, contrast, saturation, hue]):
        transforms_list.append(
            T.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
            )
        )

    transforms_list += [
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return T.Compose(transforms_list)


def get_val_transforms(image_size: tuple[int, int] = IMAGE_SIZE) -> T.Compose:
    """
    Returns deterministic preprocessing for validation and test sets.

    Only resize and normalize — NO random augmentation.
    This ensures repeatable, comparable evaluation results.
    """
    return T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Single-image inference preprocessor
# ---------------------------------------------------------------------------

def preprocess_single_image(
    image_path: str | Path,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> torch.Tensor:
    """
    Loads, crops, and transforms a single fundus image for inference.

    Args:
        image_path: Path to the fundus image file.
        image_size: Target (width, height) for resizing.

    Returns:
        Float tensor of shape [1, 3, H, W] — ready for model input.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError:        If the image cannot be decoded.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise ValueError(f"Unable to decode image (corrupt or unsupported format): {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    cropped = crop_retina_circle(img_rgb)
    clahe_img = apply_clahe(cropped)
    pil_img = Image.fromarray(clahe_img)

    transform = get_val_transforms(image_size)
    tensor = transform(pil_img).unsqueeze(0)   # → [1, 3, H, W]
    return tensor
