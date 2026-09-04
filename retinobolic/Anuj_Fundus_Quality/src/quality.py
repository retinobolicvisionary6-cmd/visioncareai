"""
quality.py — Central quality orchestrator for the Fundus Image Quality Engine.

Public API
----------
    assess_quality(image_path: str) -> dict
        Run the full quality pipeline and return a standardised result dict.

Pipeline
--------
    load + validate → preprocess → ROI extraction
    → focus + illumination + FOV + retinal visibility + artifacts
    → hard safety gates → weighted composite score → Good/Borderline/Ungradable
    → if borderline: enhance → re-assess → final decision

Output contract (stable for Vinayak's integration):
    {
        "status": "good" | "borderline" | "ungradable",
        "quality_score": float,          # 0.0–1.0  (Image Quality Score)
        "focus_score": float,
        "illumination_score": float,
        "field_of_view_score": float,
        "retinal_visibility_score": float,
        "artifact_score": float,
        "reason": str,
        "action": "continue" | "enhance_and_recheck" | "recapture",
        # optional extra detail
        "enhanced": bool,
        "enhanced_image_path": str | None,
        "component_details": dict,
        "error": str | None,
    }
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import config
from .preprocessing import preprocess, PreprocessError, resize_for_analysis, extract_fundus_roi, apply_mask
from .focus import assess_focus
from .illumination import assess_illumination
from .field_of_view import assess_field_of_view
from .retinal_visibility import assess_retinal_visibility
from .artifacts import assess_artifacts
from .enhancement import enhance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_reason(
    focus: dict,
    illum: dict,
    fov: dict,
    retvis: dict,
    art: dict,
    status: str,
) -> str:
    """Build a human-readable reason string from detected conditions."""
    issues: list[str] = []

    if focus["is_hard_fail"] or focus["focus_score"] < 0.5:
        issues.append("low image sharpness")

    if illum["is_hard_fail"]:
        conds = illum.get("conditions", [])
        if "critically_dark" in conds:
            issues.append("critically dark image")
        elif "critically_bright" in conds:
            issues.append("critically overexposed image")
        elif "severe_clipping" in conds:
            issues.append("severe pixel clipping")
        else:
            issues.append("critical illumination problem")
    elif illum["illumination_score"] < 0.6:
        conds = illum.get("conditions", [])
        if conds:
            issues.append(conds[0].replace("_", " "))

    if fov["is_hard_fail"]:
        issues.append("insufficient retinal field of view")
    elif fov["field_of_view_score"] < 0.6:
        issues.append("limited field of view")

    if retvis["retinal_visibility_score"] < 0.4:
        issues.append("poor retinal structure visibility")

    if art["artifact_score"] < 0.5:
        afs = art.get("artifacts_found", [])
        if afs:
            issues.append(afs[0].replace("_", " "))

    if not issues:
        if status == "good":
            return "Image is suitable for screening."
        elif status == "borderline":
            return "Image quality is marginal and may benefit from enhancement."
        else:
            return "Overall image quality is insufficient for reliable screening."

    issue_str = ", ".join(issues)
    if status == "good":
        return f"Image is suitable for screening despite minor {issue_str}."
    elif status == "borderline":
        caps = issue_str[0].upper() + issue_str[1:]
        return f"{caps} may affect screening quality."
    else:
        caps = issue_str[0].upper() + issue_str[1:]
        return f"{caps} — image is not suitable for reliable screening."


def _compute_status(
    focus: dict,
    illum: dict,
    fov: dict,
    quality_score: float,
) -> str:
    """Apply hard safety gates first, then fall back to composite score."""
    # Hard gates → immediate ungradable.
    if focus["is_hard_fail"]:
        return "ungradable"
    if illum["is_hard_fail"]:
        return "ungradable"
    if fov["is_hard_fail"]:
        return "ungradable"

    # Composite score thresholds.
    if quality_score >= config.QUALITY_SCORE_GOOD:
        return "good"
    elif quality_score >= config.QUALITY_SCORE_BORDERLINE_LOW:
        return "borderline"
    else:
        return "ungradable"


def _run_analysis(bgr: np.ndarray) -> dict:
    """Run all metric modules on *bgr* (already resized BGR).

    Returns the raw sub-results dict.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    fundus_mask, fundus_circle = extract_fundus_roi(bgr)

    focus_r = assess_focus(gray, fundus_mask)
    illum_r = assess_illumination(gray, fundus_mask)
    fov_r = assess_field_of_view(gray, fundus_mask, fundus_circle)
    retvis_r = assess_retinal_visibility(gray, fundus_mask)
    art_r = assess_artifacts(bgr, gray, fundus_mask)

    return {
        "gray": gray,
        "fundus_mask": fundus_mask,
        "focus": focus_r,
        "illumination": illum_r,
        "field_of_view": fov_r,
        "retinal_visibility": retvis_r,
        "artifacts": art_r,
    }


def _scores_to_composite(focus_s, illum_s, fov_s, retvis_s, art_s) -> float:
    """Weighted combination of component scores."""
    return float(np.clip(
        config.WEIGHT_FOCUS * focus_s
        + config.WEIGHT_ILLUMINATION * illum_s
        + config.WEIGHT_FIELD_OF_VIEW * fov_s
        + config.WEIGHT_RETINAL_VISIBILITY * retvis_s
        + config.WEIGHT_ARTIFACT * art_s,
        0.0, 1.0
    ))


def _package_result(
    status: str,
    quality_score: float,
    focus_r: dict,
    illum_r: dict,
    fov_r: dict,
    retvis_r: dict,
    art_r: dict,
    enhanced: bool = False,
    enhanced_image_path: Optional[str] = None,
) -> dict:
    """Build the standardised output dictionary."""
    action_map = {
        "good": "continue",
        "borderline": "enhance_and_recheck",
        "ungradable": "recapture",
    }

    reason = _build_reason(focus_r, illum_r, fov_r, retvis_r, art_r, status)

    return {
        "status": status,
        "quality_score": round(quality_score, 4),
        "focus_score": focus_r["focus_score"],
        "illumination_score": illum_r["illumination_score"],
        "field_of_view_score": fov_r["field_of_view_score"],
        "retinal_visibility_score": retvis_r["retinal_visibility_score"],
        "artifact_score": art_r["artifact_score"],
        "reason": reason,
        "action": action_map[status],
        "enhanced": enhanced,
        "enhanced_image_path": enhanced_image_path,
        "component_details": {
            "focus": focus_r,
            "illumination": illum_r,
            "field_of_view": fov_r,
            "retinal_visibility": retvis_r,
            "artifacts": art_r,
        },
        "error": None,
    }


def _save_enhanced(enhanced_bgr: np.ndarray, original_path: str) -> str:
    """Save the enhanced image to outputs/enhanced/ and return its path."""
    orig = Path(original_path)
    out_dir = Path("outputs") / "enhanced"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{orig.stem}_enhanced{orig.suffix}"
    cv2.imwrite(str(out_path), enhanced_bgr)
    return str(out_path)


def _save_json(result: dict, original_path: str) -> str:
    """Save the quality result JSON to outputs/quality/."""
    orig = Path(original_path)
    out_dir = Path("outputs") / "quality"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{orig.stem}_quality.json"
    # Exclude non-serialisable raw arrays from component_details before saving.
    safe_result = {k: v for k, v in result.items() if k != "component_details"}
    safe_result["component_details"] = {
        k: {kk: vv for kk, vv in v.items() if not isinstance(vv, np.ndarray)}
        for k, v in result.get("component_details", {}).items()
    }
    with open(out_path, "w") as f:
        json.dump(safe_result, f, indent=2)
    return str(out_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def assess_quality(image_path: str) -> dict:
    """Assess the gradability of a fundus image.

    This is the single public entry point for the module.

    Parameters
    ----------
    image_path : str
        Path to a JPG, JPEG, or PNG fundus image.

    Returns
    -------
    dict
        Standardised quality result (see module docstring for full schema).
        On error, returns ``{"error": "<message>", "status": "ungradable",
        "action": "recapture", ...}``.

    Notes
    -----
    * The original image is never modified.
    * Enhancement is only applied to BORDERLINE images.
    * If an enhanced image still fails, the result is "ungradable".
    """
    # ------------------------------------------------------------------ load
    try:
        data = preprocess(image_path)
    except PreprocessError as exc:
        return {
            "status": "ungradable",
            "quality_score": 0.0,
            "focus_score": 0.0,
            "illumination_score": 0.0,
            "field_of_view_score": 0.0,
            "retinal_visibility_score": 0.0,
            "artifact_score": 0.0,
            "reason": f"Image could not be loaded: {exc}",
            "action": "recapture",
            "enhanced": False,
            "enhanced_image_path": None,
            "component_details": {},
            "error": str(exc),
        }

    resized = data["resized"]

    # ------------------------------------------------------- initial analysis
    raw = _run_analysis(resized)
    f, ill, fov, rv, art = (
        raw["focus"], raw["illumination"],
        raw["field_of_view"], raw["retinal_visibility"], raw["artifacts"],
    )

    quality_score = _scores_to_composite(
        f["focus_score"], ill["illumination_score"], fov["field_of_view_score"],
        rv["retinal_visibility_score"], art["artifact_score"],
    )
    status = _compute_status(f, ill, fov, quality_score)

    # ----------------------------------------------------------- enhancement
    if status == "borderline":
        enhanced_bgr = enhance(resized)
        enhanced_path = _save_enhanced(enhanced_bgr, image_path)

        # Re-analyse the enhanced image.
        raw2 = _run_analysis(enhanced_bgr)
        f2, ill2, fov2, rv2, art2 = (
            raw2["focus"], raw2["illumination"],
            raw2["field_of_view"], raw2["retinal_visibility"], raw2["artifacts"],
        )

        quality_score2 = _scores_to_composite(
            f2["focus_score"], ill2["illumination_score"], fov2["field_of_view_score"],
            rv2["retinal_visibility_score"], art2["artifact_score"],
        )
        status2 = _compute_status(f2, ill2, fov2, quality_score2)

        # Use whichever result is better (original or enhanced).
        if quality_score2 >= quality_score:
            result = _package_result(
                status2, quality_score2, f2, ill2, fov2, rv2, art2,
                enhanced=True, enhanced_image_path=enhanced_path,
            )
        else:
            # Enhancement made things worse; use original assessment.
            result = _package_result(
                status, quality_score, f, ill, fov, rv, art,
                enhanced=False, enhanced_image_path=None,
            )
    else:
        result = _package_result(
            status, quality_score, f, ill, fov, rv, art,
        )

    # ------------------------------------------------------------- save JSON
    _save_json(result, image_path)
    return result


def result_to_json(result: dict, indent: int = 2) -> str:
    """Serialise an ``assess_quality`` result to a JSON string.

    Strips NumPy arrays (not JSON-serialisable) from the output.
    """
    safe = {k: v for k, v in result.items() if k != "component_details"}
    if "component_details" in result:
        safe["component_details"] = {
            k: {kk: vv for kk, vv in v.items() if not isinstance(vv, np.ndarray)}
            for k, v in result["component_details"].items()
        }
    return json.dumps(safe, indent=indent, default=str)
