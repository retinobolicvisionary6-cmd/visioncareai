"""
fundus_validator.py — Domain verification gate to ensure an image is a genuine retinal fundus scan.
Detects and rejects arbitrary non-retinal images (e.g. personal photos, faces, buildings, natural scenes).
"""

from __future__ import annotations
import cv2
import numpy as np
from typing import Dict, Any, Tuple


_FACE_CASCADE = None
_PROFILE_CASCADE = None
_UPPER_CASCADE = None

def _get_cascades():
    global _FACE_CASCADE, _PROFILE_CASCADE, _UPPER_CASCADE
    if _FACE_CASCADE is None:
        try:
            _FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            _PROFILE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
            _UPPER_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
        except Exception:
            pass
    return _FACE_CASCADE, _PROFILE_CASCADE, _UPPER_CASCADE


def verify_fundus_image(img_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Validates whether an input image possesses optical and chromatic characteristics of a retinal fundus scan.
    
    Parameters
    ----------
    img_bgr : np.ndarray
        BGR image array (uint8).
        
    Returns
    -------
    dict:
        is_fundus: bool
        is_human_image: bool
        confidence: float (0.0 to 1.0)
        reasons: list of str describing why the image failed or passed
        metrics: dict of computed color and geometric values
    """
    if img_bgr is None or img_bgr.size == 0:
        return {
            "is_fundus": False,
            "is_human_image": False,
            "confidence": 0.0,
            "reasons": ["Image is empty or unreadable."],
            "metrics": {}
        }

    h, w = img_bgr.shape[:2]

    # 0. Human Face / Person Detection Safety Gate
    # A retinal fundus scan is an internal ocular image and never contains human faces or bodies.
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    face_casc, profile_casc, upper_casc = _get_cascades()
    human_detected = False
    detected_features = []

    if face_casc is not None and not face_casc.empty():
        faces = face_casc.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(45, 45))
        if len(faces) > 0:
            human_detected = True
            detected_features.append(f"{len(faces)} human face(s)")

    if not human_detected and profile_casc is not None and not profile_casc.empty():
        profiles = profile_casc.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(45, 45))
        if len(profiles) > 0:
            human_detected = True
            detected_features.append(f"{len(profiles)} profile face(s)")

    if not human_detected and upper_casc is not None and not upper_casc.empty():
        uppers = upper_casc.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if len(uppers) > 0:
            human_detected = True
            detected_features.append(f"{len(uppers)} human upper body structure(s)")

    if human_detected:
        feature_str = ", ".join(detected_features)
        return {
            "is_fundus": False,
            "is_human_image": True,
            "confidence": 0.0,
            "reasons": ["A photo of a person was detected instead of an eye retina scan. Please insert a genuine retinal fundus image."],
            "metrics": {"human_detected": True, "features": feature_str}
        }
    
    # 1. Corner Brightness (Circular fundus aperture check)
    # Most fundus cameras produce a circular field with black / dark outer corners.
    cw, ch = max(1, int(w * 0.08)), max(1, int(h * 0.08))
    corners = np.concatenate([
        img_bgr[:ch, :cw].reshape(-1, 3),
        img_bgr[:ch, -cw:].reshape(-1, 3),
        img_bgr[-ch:, :cw].reshape(-1, 3),
        img_bgr[-ch:, -cw:].reshape(-1, 3)
    ])
    corner_brightness = float(np.mean(corners))
    
    # 2. Central Region Colorimetric Analysis (Retinal Chromatic Profile)
    # The retina is heavily vascularized and backed by choroidal melanin and hemoglobin,
    # which absorb blue light strongly and reflect orange/red wavelengths.
    center = img_bgr[int(h * 0.20):int(h * 0.80), int(w * 0.20):int(w * 0.80)]
    if center.size == 0:
        center = img_bgr
        
    b_ch = center[:, :, 0].astype(np.float64)
    g_ch = center[:, :, 1].astype(np.float64)
    r_ch = center[:, :, 2].astype(np.float64)
    
    r_mean = float(np.mean(r_ch))
    g_mean = float(np.mean(g_ch))
    b_mean = float(np.mean(b_ch))
    
    rb_ratio = float(r_mean / (b_mean + 1e-5))
    rg_ratio = float(r_mean / (g_mean + 1e-5))
    
    # 3. HSV Retinal Hue Spectrum Check
    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    
    # Retinal hue in OpenCV scale [0, 180]: Orange/Red/Amber is typically [0, 26] or [165, 180]
    retinal_hue_mask = ((hue <= 26) | (hue >= 165)) & (sat >= 25) & (val >= 20)
    retinal_hue_pct = float(np.mean(retinal_hue_mask) * 100.0)
    
    # 4. Color variance / multi-color entropy
    # Natural photos (clothes, face with background, sky, objects) have wide hue spreads.
    non_dark_mask = (val > 25) & (sat > 20)
    if np.sum(non_dark_mask) > 50:
        hue_std = float(np.std(hue[non_dark_mask]))
    else:
        hue_std = 0.0

    reasons: list[str] = []
    hard_fail = False

    # Check A: Red to Blue Dominance
    if rb_ratio < 1.85:
        reasons.append("non-retinal color tones")
        hard_fail = True
    elif rb_ratio < 2.2:
        reasons.append("marginal color balance")

    # Check B: Blue Channel Absorption
    if b_mean > 85.0 and rb_ratio < 2.5:
        reasons.append("unnatural lighting for an eye scan")
        hard_fail = True

    # Check C: Retinal Hue Concentration
    if retinal_hue_pct < 65.0:
        reasons.append("lacks retinal tissue appearance")
        hard_fail = True

    # Check D: Rectangular Full-Scene vs Circular Retinal Aperture
    if corner_brightness > 35.0 and (rb_ratio < 2.8 or hue_std > 35.0):
        reasons.append("regular camera photo detected")
        hard_fail = True

    is_fundus = not hard_fail
    
    # Calculate confidence score
    confidence_factors = []
    if rb_ratio >= 3.0:
        confidence_factors.append(1.0)
    elif rb_ratio >= 2.0:
        confidence_factors.append((rb_ratio - 2.0) / 1.0)
    else:
        confidence_factors.append(0.0)

    if retinal_hue_pct >= 85.0:
        confidence_factors.append(1.0)
    elif retinal_hue_pct >= 65.0:
        confidence_factors.append((retinal_hue_pct - 65.0) / 20.0)
    else:
        confidence_factors.append(0.0)

    confidence = float(np.clip(np.mean(confidence_factors), 0.0, 1.0))

    if not is_fundus:
        if "regular camera photo detected" in reasons:
            reasons = ["A standard camera photo was detected instead of an eye retina scan. Please insert a genuine retinal fundus image."]
        else:
            reasons = ["This photo is not an eye retina scan. Please upload a genuine retinal fundus image taken with an eye camera."]
    elif not reasons:
        reasons.append("Genuine retinal fundus scan verified.")

    return {
        "is_fundus": is_fundus,
        "is_human_image": False,
        "confidence": round(confidence, 3),
        "reasons": reasons,
        "metrics": {
            "rb_ratio": round(rb_ratio, 2),
            "rg_ratio": round(rg_ratio, 2),
            "b_mean": round(b_mean, 1),
            "r_mean": round(r_mean, 1),
            "g_mean": round(g_mean, 1),
            "corner_brightness": round(corner_brightness, 1),
            "retinal_hue_pct": round(retinal_hue_pct, 1),
            "hue_std": round(hue_std, 1),
        }
    }
