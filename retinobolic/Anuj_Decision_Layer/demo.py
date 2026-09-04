"""
demo.py — Final Decision Layer Demo
=====================================
SIH Problem ID: SIH26038
Explainable AI for Diabetic Retinopathy Screening in Rural India

Demonstrates the Decision Layer across 8 realistic screening scenarios.

Usage:
    python demo.py
    python demo.py --scenario all
    python demo.py --scenario routine
    python demo.py --scenario recapture
    python demo.py --scenario review_ood
    python demo.py --scenario review_uncertainty
    python demo.py --scenario refer_grade2
    python demo.py --scenario refer_grade3
    python demo.py --scenario conflict_ungradable
    python demo.py --scenario conflict_ood_confidence

IMPORTANT:
    This demo is for engineering evaluation ONLY.
    It does NOT constitute a medical diagnosis or clinical recommendation.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Force UTF-8 output on Windows to handle emoji/unicode symbols
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Make src importable from project root
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engine import run_screening_decision

# ---------------------------------------------------------------------------
# Terminal colours (graceful degradation)
# ---------------------------------------------------------------------------
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    GREEN  = Fore.GREEN
    YELLOW = Fore.YELLOW
    RED    = Fore.RED
    CYAN   = Fore.CYAN
    MAGENTA = Fore.MAGENTA
    BLUE   = Fore.BLUE
    BOLD   = Style.BRIGHT
    DIM    = Style.DIM
    RESET  = Style.RESET_ALL
except ImportError:
    GREEN = YELLOW = RED = CYAN = MAGENTA = BLUE = BOLD = DIM = RESET = ""

# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

_SEP = "─" * 60
_THICK = "═" * 60

_ACTION_ICONS = {
    "routine":      f"{GREEN}✅  ROUTINE{RESET}",
    "recapture":    f"{YELLOW}🔄  RECAPTURE IMAGE{RESET}",
    "doctor_review":f"{MAGENTA}⚠️   DOCTOR REVIEW{RESET}",
    "refer":        f"{RED}🚨  REFER{RESET}",
}

_PRIORITY_COLOURS = {
    "low":    GREEN,
    "medium": YELLOW,
    "high":   RED,
    "urgent": f"{BOLD}{RED}",
}


def _action_label(action: str) -> str:
    return _ACTION_ICONS.get(action, action.upper())


def _priority_label(priority: str) -> str:
    colour = _PRIORITY_COLOURS.get(priority, "")
    return f"{colour}{priority.upper()}{RESET}"


def _yn(val: bool) -> str:
    return f"{GREEN}YES{RESET}" if val else f"{YELLOW}NO{RESET}"


def _bar(score: float, width: int = 20) -> str:
    filled = max(0, min(width, int(round(score * width))))
    return f"[{'#' * filled}{'.' * (width - filled)}]"


def _fmt_score(score: Optional[float]) -> str:
    if score is None:
        return "N/A"
    return f"{score * 100:.1f}%  {_bar(score)}"


def _wrap(text: str, width: int = 58, indent: int = 4) -> str:
    """Word-wrap a string."""
    words = text.split()
    lines = []
    line = " " * indent
    for word in words:
        if len(line) + len(word) + 1 > width:
            lines.append(line)
            line = " " * indent + word + " "
        else:
            line += word + " "
    if line.strip():
        lines.append(line)
    return "\n".join(lines)


def print_scenario_result(title: str, result: Dict[str, Any]) -> None:
    print(f"\n{BOLD}{_THICK}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{_SEP}")

    # Action + Priority
    print(f"  Action   : {_action_label(result['action'])}")
    print(f"  Priority : {_priority_label(result['priority'])}")
    print(f"  Review   : {_yn(result['review_required'])}")

    # DR grade
    if result.get("dr_grade") is not None:
        grade = result["dr_grade"]
        grade_names = {0: "No DR", 1: "Mild DR", 2: "Moderate DR", 3: "Severe / PDR"}
        print(f"  DR Grade : {CYAN}{grade} — {grade_names.get(grade, '?')}{RESET}")
    else:
        print(f"  DR Grade : {DIM}N/A (image ungradable){RESET}")

    # Reliability
    rel_status = result.get("reliability_status")
    if rel_status:
        rel_colours = {"acceptable": GREEN, "caution": YELLOW, "review_required": RED}
        rc = rel_colours.get(rel_status, "")
        print(f"  Reliability: {rc}{rel_status.upper()}{RESET}")
    else:
        print(f"  Reliability: {DIM}N/A{RESET}")

    # Evidence
    ev = result.get("evidence", {})
    if ev:
        print(f"\n{_SEP}")
        print(f"  {BOLD}Evidence{RESET}")
        print(f"  Quality       : {ev.get('quality_status', '?').upper()} "
              f"({_fmt_score(ev.get('quality_score'))})")
        print(f"  Confidence    : {_fmt_score(ev.get('confidence'))} "
              f"({ev.get('confidence_level', '?').upper()})")
        print(f"  Uncertainty   : {_fmt_score(ev.get('uncertainty'))} "
              f"({ev.get('uncertainty_level', '?').upper()})")
        print(f"  OOD           : {_yn(ev.get('ood', False))} "
              f"(score={ev.get('ood_score', 0.0):.3f})")
        if ev.get("reliability_score") is not None:
            print(f"  Rel. Score    : {_fmt_score(ev.get('reliability_score'))}")
        gp = ev.get("gradcam_path")
        if gp:
            print(f"  Grad-CAM      : {CYAN}{gp}{RESET}")
        else:
            print(f"  Grad-CAM      : {DIM}Not available (XAI module pending){RESET}")
        print(f"  Clinical Ctx  : {_yn(ev.get('clinical_context_complete', False))}")
        sigs = ev.get("reliability_signals", [])
        if sigs:
            print(f"  Rel. Signals  : {YELLOW}{', '.join(sigs)}{RESET}")

    # Rule applied
    meta = result.get("metadata", {})
    if meta:
        print(f"\n{_SEP}")
        print(f"  {BOLD}Decision Metadata{RESET}")
        print(f"  Rule Applied  : {meta.get('rule_applied', '?')}")
        print(f"  Engine Vers.  : {meta.get('engine_version', '?')}")

    # Reason
    print(f"\n{_SEP}")
    print(f"  {BOLD}Decision Reason{RESET}")
    reason = result.get("reason", "")
    print(_wrap(reason, width=60, indent=4))

    print(f"{_THICK}\n")


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

def scenario_routine() -> tuple[str, Dict[str, Any]]:
    """TEST 1: Good quality, reliable, No DR → routine."""
    quality = {
        "status": "good", "quality_score": 0.91,
        "action": "continue", "reason": "Image is suitable for screening.",
        "enhanced": False, "enhanced_image_path": None, "error": None,
    }
    dr = {
        "grade": 0,
        "probabilities": {"0": 0.92, "1": 0.05, "2": 0.02, "3": 0.01},
        "gradcam_path": None,
    }
    rel = {
        "reliability_status": "acceptable", "review_required": False,
        "reason": "High model confidence, low uncertainty and in-distribution input.",
        "confidence": 0.92, "confidence_level": "high",
        "uncertainty": 0.18, "uncertainty_level": "low",
        "ood": False, "ood_status": "in_distribution",
        "ood_score": 1.42, "reliability_score": 0.87,
    }
    clinical = {
        "validation_passed": True, "validation_errors": [],
        "clinical_context": {
            "age": {"value": 42, "unit": "years", "status": "provided"},
            "hba1c": {"value": 6.2, "unit": "%", "status": "provided"},
        },
        "data_quality": {"complete": True, "clinical_context_complete": True, "missing_fields": []},
    }
    result = run_screening_decision(quality, dr, rel, clinical)
    return "SCENARIO 1 — Routine (Good Quality, No DR, Reliable)", result


def scenario_recapture() -> tuple[str, Dict[str, Any]]:
    """TEST 2: Ungradable image → recapture (even with high Grade 3)."""
    quality = {
        "status": "ungradable", "quality_score": 0.18,
        "action": "recapture",
        "reason": "Low image sharpness, critically dark image — not suitable for screening.",
        "enhanced": False, "enhanced_image_path": None, "error": None,
    }
    dr = {
        "grade": 3,
        "probabilities": {"0": 0.01, "1": 0.01, "2": 0.01, "3": 0.97},
        "gradcam_path": None,
    }
    rel = {
        "reliability_status": "acceptable", "review_required": False,
        "reason": "High model confidence, low uncertainty and in-distribution input.",
        "confidence": 0.97, "confidence_level": "high",
        "uncertainty": 0.08, "uncertainty_level": "low",
        "ood": False, "ood_status": "in_distribution",
        "ood_score": 1.30, "reliability_score": 0.95,
    }
    result = run_screening_decision(quality, dr, rel)
    return "SCENARIO 2 — Recapture (Ungradable — Safety Override)", result


def scenario_review_ood() -> tuple[str, Dict[str, Any]]:
    """TEST 4 & 6: OOD=True overrides high confidence → doctor_review."""
    quality = {
        "status": "good", "quality_score": 0.87,
        "action": "continue", "reason": "Image is suitable for screening.",
        "error": None,
    }
    dr = {
        "grade": 2,
        "probabilities": {"0": 0.02, "1": 0.03, "2": 0.93, "3": 0.02},
        "gradcam_path": None,
    }
    rel = {
        "reliability_status": "review_required", "review_required": True,
        "reason": "Input appears outside the configured reference distribution.",
        "confidence": 0.93, "confidence_level": "high",
        "uncertainty": 0.10, "uncertainty_level": "low",
        "ood": True, "ood_status": "review_required",
        "ood_score": 5.80, "reliability_score": 0.0,
    }
    result = run_screening_decision(quality, dr, rel)
    return "SCENARIO 3 — Doctor Review (OOD Overrides High Confidence)", result


def scenario_review_uncertainty() -> tuple[str, Dict[str, Any]]:
    """TEST 3: High uncertainty → doctor_review (even with high DR grade)."""
    quality = {
        "status": "good", "quality_score": 0.85,
        "action": "continue", "reason": "Image is suitable for screening.",
        "error": None,
    }
    dr = {
        "grade": 3,
        "probabilities": {"0": 0.10, "1": 0.20, "2": 0.25, "3": 0.45},
        "gradcam_path": None,
    }
    rel = {
        "reliability_status": "review_required", "review_required": True,
        "reason": "Prediction has high model uncertainty. Model confidence is low.",
        "confidence": 0.45, "confidence_level": "low",
        "uncertainty": 0.82, "uncertainty_level": "high",
        "ood": False, "ood_status": "in_distribution",
        "ood_score": 1.75, "reliability_score": 0.12,
    }
    result = run_screening_decision(quality, dr, rel)
    return "SCENARIO 4 — Doctor Review (High Uncertainty, Grade 3 Blocked)", result


def scenario_refer_grade2() -> tuple[str, Dict[str, Any]]:
    """TEST 5: Reliable Moderate DR (Grade 2) → refer."""
    quality = {
        "status": "good", "quality_score": 0.88,
        "action": "continue", "reason": "Image is suitable for screening.",
        "error": None,
    }
    dr = {
        "grade": 2,
        "probabilities": {"0": 0.02, "1": 0.04, "2": 0.88, "3": 0.06},
        "gradcam_path": "outputs/gradcam/patient_003.jpg",
    }
    rel = {
        "reliability_status": "acceptable", "review_required": False,
        "reason": "High model confidence, low uncertainty and in-distribution input.",
        "confidence": 0.88, "confidence_level": "high",
        "uncertainty": 0.22, "uncertainty_level": "low",
        "ood": False, "ood_status": "in_distribution",
        "ood_score": 1.55, "reliability_score": 0.84,
    }
    clinical = {
        "validation_passed": True, "validation_errors": [],
        "clinical_context": {
            "age": {"value": 55, "unit": "years", "status": "provided"},
            "hba1c": {"value": 7.8, "unit": "%", "status": "provided"},
            "diabetes_duration_years": {"value": 8, "unit": "years", "status": "provided"},
        },
        "data_quality": {"complete": True, "clinical_context_complete": True, "missing_fields": []},
    }
    result = run_screening_decision(quality, dr, rel, clinical)
    return "SCENARIO 5 — Refer (Moderate DR, Grade 2, Reliable)", result


def scenario_refer_grade3() -> tuple[str, Dict[str, Any]]:
    """Grade 3 reliable → refer with URGENT priority."""
    quality = {
        "status": "good", "quality_score": 0.90,
        "action": "continue", "reason": "Image is suitable for screening.",
        "error": None,
    }
    dr = {
        "grade": 3,
        "probabilities": {"0": 0.01, "1": 0.01, "2": 0.02, "3": 0.96},
        "gradcam_path": "outputs/gradcam/patient_007.jpg",
    }
    rel = {
        "reliability_status": "acceptable", "review_required": False,
        "reason": "High model confidence, low uncertainty and in-distribution input.",
        "confidence": 0.96, "confidence_level": "high",
        "uncertainty": 0.09, "uncertainty_level": "low",
        "ood": False, "ood_status": "in_distribution",
        "ood_score": 1.28, "reliability_score": 0.94,
    }
    result = run_screening_decision(quality, dr, rel)
    return "SCENARIO 6 — URGENT Refer (Severe DR / PDR, Grade 3, Reliable)", result


def scenario_conflict_ungradable() -> tuple[str, Dict[str, Any]]:
    """CONFLICT A: Grade 3, 0.99 confidence, OOD=False BUT ungradable → recapture."""
    quality = {
        "status": "ungradable", "quality_score": 0.19,
        "action": "recapture", "reason": "Insufficient field of view.",
        "error": None,
    }
    dr = {
        "grade": 3,
        "probabilities": {"0": 0.01, "1": 0.01, "2": 0.01, "3": 0.97},
    }
    rel = {
        "reliability_status": "acceptable", "review_required": False,
        "reason": "High model confidence, low uncertainty and in-distribution input.",
        "confidence": 0.97, "confidence_level": "high",
        "uncertainty": 0.08, "uncertainty_level": "low",
        "ood": False, "ood_status": "in_distribution",
        "ood_score": 1.30, "reliability_score": 0.95,
    }
    result = run_screening_decision(quality, dr, rel)
    return "SCENARIO 7 — Conflict: Ungradable + Grade 3 + High Confidence → RECAPTURE", result


def scenario_conflict_ood_confidence() -> tuple[str, Dict[str, Any]]:
    """CONFLICT C: Grade 2, High Confidence, Low Uncertainty BUT OOD=True → doctor_review."""
    quality = {
        "status": "good", "quality_score": 0.88,
        "action": "continue", "reason": "Image is suitable for screening.",
        "error": None,
    }
    dr = {
        "grade": 2,
        "probabilities": {"0": 0.02, "1": 0.04, "2": 0.88, "3": 0.06},
    }
    rel = {
        "reliability_status": "review_required", "review_required": True,
        "reason": "Input appears outside the configured reference distribution.",
        "confidence": 0.88, "confidence_level": "high",
        "uncertainty": 0.20, "uncertainty_level": "low",
        "ood": True, "ood_status": "review_required",
        "ood_score": 5.50, "reliability_score": 0.0,
    }
    result = run_screening_decision(quality, dr, rel)
    return (
        "SCENARIO 8 — Conflict: Grade 2 + High Confidence + OOD → DOCTOR REVIEW",
        result,
    )


# ---------------------------------------------------------------------------
# All scenarios registry
# ---------------------------------------------------------------------------

SCENARIOS = {
    "routine":              scenario_routine,
    "recapture":            scenario_recapture,
    "review_ood":           scenario_review_ood,
    "review_uncertainty":   scenario_review_uncertainty,
    "refer_grade2":         scenario_refer_grade2,
    "refer_grade3":         scenario_refer_grade3,
    "conflict_ungradable":  scenario_conflict_ungradable,
    "conflict_ood_confidence": scenario_conflict_ood_confidence,
}


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(results: list[tuple[str, str, str]]) -> None:
    """Print a summary table: (scenario_name, action, priority)."""
    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  DECISION LAYER DEMO — SUMMARY{RESET}")
    print(f"{'=' * 60}")
    print(f"  {'SCENARIO':<38} {'ACTION':<16} {'PRIORITY'}")
    print(f"  {'─' * 38} {'─' * 16} {'─' * 10}")
    action_colours = {
        "routine": GREEN,
        "recapture": YELLOW,
        "doctor_review": MAGENTA,
        "refer": RED,
    }
    for name, action, priority in results:
        ac = action_colours.get(action, "")
        print(f"  {name:<38} {ac}{action:<16}{RESET} {priority}")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_demo(scenario_name: str = "all") -> None:
    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  FINAL DECISION LAYER — Demo{RESET}")
    print(f"{BOLD}  SIH26038: Explainable AI for DR Screening{RESET}")
    print(f"{'=' * 60}")
    print()
    print(f"  {RED}⚠  ENGINEERING EVALUATION ONLY.{RESET}")
    print(f"  {RED}⚠  NOT a medical diagnosis tool.{RESET}")
    print(f"  {RED}⚠  All decisions must be reviewed by a physician.{RESET}")
    print()

    if scenario_name == "all":
        scenarios_to_run = list(SCENARIOS.keys())
    elif scenario_name in SCENARIOS:
        scenarios_to_run = [scenario_name]
    else:
        print(f"Unknown scenario: '{scenario_name}'")
        print(f"Available: {', '.join(SCENARIOS.keys())} or 'all'")
        sys.exit(1)

    summary_rows = []
    for key in scenarios_to_run:
        fn = SCENARIOS[key]
        try:
            title, result = fn()
            print_scenario_result(title, result)
            summary_rows.append((key, result["action"], result["priority"]))

            # Save to outputs/
            out_dir = ROOT / "outputs"
            out_dir.mkdir(exist_ok=True)
            out_file = out_dir / f"result_{key}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"  {CYAN}→ Result saved: {out_file}{RESET}\n")

        except Exception as exc:
            print(f"  {RED}ERROR in scenario '{key}': {exc}{RESET}\n")
            summary_rows.append((key, "ERROR", "—"))

    if len(scenarios_to_run) > 1:
        print_summary(summary_rows)

    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"  DEMO COMPLETE")
    print(f"{'=' * 60}")
    print()
    print("  The Final Decision Layer integrates outputs from:")
    print(f"    • {CYAN}anuj-fundus-quality{RESET}      — Image Quality Assessment")
    print(f"    • {CYAN}Vinayak DR Model{RESET}          — 4-Class DR Prediction (pending)")
    print(f"    • {CYAN}Grad-CAM / XAI{RESET}            — Visual Evidence (pending)")
    print(f"    • {CYAN}anuj-confidence{RESET}           — Model Confidence")
    print(f"    • {CYAN}anuj-uncertainty{RESET}          — Model Uncertainty")
    print(f"    • {CYAN}anuj-ood{RESET}                  — Out-of-Distribution")
    print(f"    • {CYAN}anuj-reliability{RESET}          — Reliability Fusion")
    print(f"    • {CYAN}anuj-clinical-context{RESET}     — Clinical Context")
    print()
    print(f"  {RED}⚠  Output is for engineering evaluation only.{RESET}")
    print(f"  {RED}⚠  NOT a medical diagnosis or clinical recommendation.{RESET}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Final Decision Layer Demo — SIH26038"
    )
    parser.add_argument(
        "--scenario",
        default="all",
        choices=list(SCENARIOS.keys()) + ["all"],
        help="Which scenario to run (default: all)",
    )
    args = parser.parse_args()
    run_demo(args.scenario)


if __name__ == "__main__":
    main()
