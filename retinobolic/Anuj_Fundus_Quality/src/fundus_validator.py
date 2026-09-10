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

def _get_cascades():
    global _FACE_CASCADE, _PROFILE_CASCADE
    if _FACE_CASCADE is None:
        try:
            _FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            _PROFILE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        except Exception:
            pass
    return _FACE_CASCADE, _PROFILE_CASCADE


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

    # 1. Corner Brightness (Ophthalmoscope circular lens aperture check)
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
    center = img_bgr[int(h * 0.15):int(h * 0.85), int(w * 0.15):int(w * 0.85)]
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
    # In OpenCV HSV scale [0, 180]:
    # Red, Orange, Amber, and Golden Yellow span [0, 35] and [165, 180].
    # Evaluated across visible (non-black) pixels so dark clinical fundus scans are not penalized.
    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    
    vis_mask = (val >= 8) & (sat >= 10)
    vis_count = int(np.sum(vis_mask))
    
    if vis_count > 50:
        retinal_hue_mask = ((hue[vis_mask] <= 35) | (hue[vis_mask] >= 165))
        retinal_hue_pct = float(np.mean(retinal_hue_mask) * 100.0)
        hue_std = float(np.std(hue[vis_mask]))
    else:
        retinal_hue_pct = 0.0
        hue_std = 0.0

    # 4. Human Face / Person Detection Safety Gate
    # Genuine personal photos (selfies, portraits, hospital staff photos) are regular rectangular
    # camera photos with bright corners and human skin tones.
    # In contrast, ophthalmic retinal fundus scans have dark aperture corners and monochromatic
    # retinal tissue (where optic disc and blood vessels might otherwise trigger weak Haar edge false alarms).
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    face_casc, profile_casc = _get_cascades()
    human_detected = False
    detected_features = []

    # If the image has dark circular corners and strong retinal hue, it is an ocular scan
    is_circular_retina = (corner_brightness <= 25.0) and (retinal_hue_pct >= 85.0)

    if not is_circular_retina and face_casc is not None and not face_casc.empty():
        faces = face_casc.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(50, 50))
        for (fx, fy, fw, fh) in faces:
            fcrop = img_bgr[fy:fy+fh, fx:fx+fw]
            f_bm = float(fcrop[:, :, 0].mean())
            f_rm = float(fcrop[:, :, 2].mean())
            f_rb = f_rm / (f_bm + 1e-5)
            # Filter out false alarms on deep monochromatic red retinal tissue
            if corner_brightness <= 25.0 and (f_rb > 2.2 or f_bm < 30.0):
                continue
            human_detected = True
            detected_features.append(f"{len(faces)} human face(s)")
            break

    if not human_detected and corner_brightness > 25.0 and profile_casc is not None and not profile_casc.empty():
        profiles = profile_casc.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(50, 50))
        if len(profiles) > 0:
            human_detected = True
            detected_features.append(f"{len(profiles)} profile face(s)")

    if human_detected:
        feature_str = ", ".join(detected_features)
        return {
            "is_fundus": False,
            "is_human_image": True,
            "confidence": 0.0,
            "reasons": ["A photo of a person was detected instead of an eye retina scan. Please insert a genuine retinal fundus image."],
            "metrics": {"human_detected": True, "features": feature_str}
        }

    reasons: list[str] = []
    hard_fail = False

    # Check 1: Blank or pitch-dark non-image
    if r_mean < 6.0 and g_mean < 6.0 and b_mean < 6.0:
        reasons.append("Image is pitch dark or blank. Please upload a clear retinal fundus photograph.")
        hard_fail = True

    # Check 2: Red channel dominance (Retinal tissue always reflects red > blue and red >= green)
    elif rb_ratio < 1.05:
        reasons.append("This photo is not an eye retina scan. Please upload a genuine retinal fundus image taken with an eye camera.")
        hard_fail = True
    elif rg_ratio < 0.90:
        reasons.append("This photo is not an eye retina scan. Please upload a genuine retinal fundus image taken with an eye camera.")
        hard_fail = True

    # Check 3: Retinal Hue Concentration
    elif vis_count > 100 and retinal_hue_pct < 35.0:
        reasons.append("This photo is not an eye retina scan. Please upload a genuine retinal fundus image taken with an eye camera.")
        hard_fail = True

    # Check 4: Regular rectangular camera photo (high corner brightness + wide color spread or low retinal hue)
    elif corner_brightness > 35.0 and (retinal_hue_pct < 70.0 or hue_std > 28.0 or rb_ratio < 1.20):
        reasons.append("A standard camera photo was detected instead of an eye retina scan. Please insert a genuine retinal fundus image.")
        hard_fail = True

    # Check 5: Solid artificial / synthetic color check
    elif vis_count > 100:
        green_std = float(np.std(center[:, :, 1]))
        if green_std < 1.5:
            reasons.append("This photo is not an eye retina scan. Please upload a genuine retinal fundus image taken with an eye camera.")
            hard_fail = True

    is_fundus = not hard_fail

    # Confidence calculation
    if is_fundus:
        rb_factor = float(np.clip((rb_ratio - 1.05) / 1.5, 0.5, 1.0))
        hue_factor = float(np.clip((retinal_hue_pct - 35.0) / 45.0, 0.5, 1.0))
        corner_factor = 1.0 if corner_brightness <= 25.0 else 0.8
        confidence = float(np.clip(0.4 * rb_factor + 0.4 * hue_factor + 0.2 * corner_factor, 0.70, 0.99))
        reasons = ["Genuine retinal fundus scan verified."]
    else:
        confidence = 0.0
        if not reasons:
            reasons.append("This photo is not an eye retina scan. Please upload a genuine retinal fundus image taken with an eye camera.")

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
