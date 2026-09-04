"""
demo.py -- Clinical Context Module demonstration script.

Run:
    python demo.py

Demonstrates:
  1. Complete patient data -- all fields provided.
  2. Partial patient data -- missing HbA1c and diabetes duration.
  3. Invalid patient data -- catches validation errors cleanly.

PRIVACY NOTE:
  Full clinical values are printed here for demonstration only.
  In production deployments, do NOT log raw clinical values.
  Use get_summary() for privacy-safe logging.
"""

import json
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent))

from src import process_clinical_context

# ---------------------------------------------------------------------------
# Colour helpers (optional -- graceful degradation on non-ANSI terminals)
# ---------------------------------------------------------------------------
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    GREEN  = Fore.GREEN
    YELLOW = Fore.YELLOW
    RED    = Fore.RED
    CYAN   = Fore.CYAN
    BOLD   = Style.BRIGHT
    RESET  = Style.RESET_ALL
except ImportError:
    GREEN = YELLOW = RED = CYAN = BOLD = RESET = ""


# ---------------------------------------------------------------------------
# Display helper
# ---------------------------------------------------------------------------
def _val(field_dict) -> str:
    """Format a normalised field dict for display."""
    if field_dict is None:
        return "Not provided"
    v = field_dict.get("value")
    u = field_dict.get("unit", "")
    s = field_dict.get("status", "")
    if v is None:
        return f"{YELLOW}Not provided{RESET} [{s}]"
    if u:
        return f"{GREEN}{v} {u}{RESET}"
    return f"{GREEN}{v}{RESET}"


def _history_str(history_field) -> str:
    if history_field is None:
        return "Not provided"
    v = history_field.get("value")
    if not v:
        return f"{YELLOW}Not provided{RESET}"
    parts = []
    if v.get("known_diabetes") is not None:
        parts.append(f"Known diabetes: {v['known_diabetes']}")
    if v.get("previous_dr_history") is not None:
        parts.append(f"Previous DR history: {v['previous_dr_history']}")
    if v.get("other_notes"):
        parts.append(f"Notes: {v['other_notes']}")
    return " | ".join(parts) if parts else "(empty)"


def print_result(label: str, result: dict):
    """Pretty-print a clinical context result."""
    sep = "-" * 50

    print(f"\n{BOLD}{CYAN}{'=' * 50}{RESET}")
    print(f"{BOLD}{CYAN} CLINICAL CONTEXT -- {label}{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 50}{RESET}")

    if not result["validation_passed"]:
        print(f"\n{RED}!  VALIDATION FAILED{RESET}")
        for err in result["validation_errors"]:
            print(f"   {RED}* {err}{RESET}")
        print()
        return

    ctx = result["clinical_context"]
    dq  = result["data_quality"]

    print(f"\n{sep}")
    print(f"  Age                : {_val(ctx.get('age'))}")
    print(f"  Blood Pressure     : ", end="")

    sys_v = ctx.get("bp_systolic", {}).get("value")
    dia_v = ctx.get("bp_diastolic", {}).get("value")
    sys_s = ctx.get("bp_systolic", {}).get("status", "missing")
    dia_s = ctx.get("bp_diastolic", {}).get("status", "missing")

    if sys_v is not None and dia_v is not None:
        print(f"{GREEN}{sys_v}/{dia_v} mmHg{RESET}")
    elif sys_v is not None:
        print(f"{YELLOW}{sys_v}/-- mmHg (diastolic {dia_s}){RESET}")
    else:
        print(f"{YELLOW}Not provided [{sys_s}]{RESET}")

    print(f"  HbA1c              : {_val(ctx.get('hba1c'))}")
    print(f"  Diabetes Duration  : {_val(ctx.get('diabetes_duration_years'))}")
    print(f"  Clinical History   : {_history_str(ctx.get('clinical_history'))}")
    print(sep)

    status_label = f"{GREEN}COMPLETE{RESET}" if dq["complete"] else f"{YELLOW}INCOMPLETE{RESET}"
    print(f"\n  Data Quality       : {status_label}")

    if dq["missing_fields"]:
        print(f"  Missing Fields     :")
        for mf in dq["missing_fields"]:
            print(f"    {YELLOW}* {mf}{RESET}")
    else:
        print(f"  Missing Fields     : {GREEN}None{RESET}")

    # Context flags
    flags = dq.get("flags", {})
    if flags:
        print(f"\n  Availability Flags :")
        for k, v in flags.items():
            icon = f"{GREEN}Y{RESET}" if v else f"{YELLOW}N{RESET}"
            print(f"    {icon} {k}")

    print()


def save_result(label: str, result: dict):
    """Save result JSON to outputs/ directory."""
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    safe_label = label.lower().replace(" ", "_").replace("/", "_")
    out_path = out_dir / f"result_{safe_label}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  {CYAN}→ Saved to {out_path}{RESET}")


# ---------------------------------------------------------------------------
# Demo cases
# ---------------------------------------------------------------------------
def main():
    print(f"\n{BOLD}{'=' * 50}")
    print(f"  SIH26038 -- Clinical Context Module -- Demo")
    print(f"{'=' * 50}{RESET}")
    print("  Module purpose: Collect, validate, normalise,")
    print("  and structure patient clinical data for the")
    print("  downstream Final Decision / Referral Priority layer.")
    print(f"\n  {RED}NOT a diagnosis tool. NOT a DR prediction model.{RESET}")

    # -----------------------------------------------------------------------
    # DEMO 1 -- Complete patient
    # -----------------------------------------------------------------------
    complete_data = {
        "patient_id": "DEMO001",
        "age": 58,
        "bp_systolic": 148,
        "bp_diastolic": 92,
        "hba1c": 8.2,
        "diabetes_duration_years": 10,
        "clinical_history": {
            "known_diabetes": True,
            "previous_dr_history": False,
            "other_notes": "Annual DR screening.",
        },
    }
    result1 = process_clinical_context(complete_data)
    print_result("Complete Data (DEMO001)", result1)
    save_result("demo001_complete", result1)

    # -----------------------------------------------------------------------
    # DEMO 2 -- Partial patient (missing HbA1c and duration)
    # -----------------------------------------------------------------------
    partial_data = {
        "patient_id": "DEMO002",
        "age": 45,
        "bp_systolic": 130,
        "bp_diastolic": 85,
        "clinical_history": {
            "known_diabetes": True,
        },
    }
    result2 = process_clinical_context(partial_data)
    print_result("Partial Data (DEMO002)", result2)
    save_result("demo002_partial", result2)

    # -----------------------------------------------------------------------
    # DEMO 3 -- Invalid patient
    # -----------------------------------------------------------------------
    invalid_data = {
        "patient_id": "DEMO_INVALID",
        "age": -5,
        "bp_systolic": 148,
        "bp_diastolic": 92,
    }
    result3 = process_clinical_context(invalid_data)
    print_result("Invalid Data (DEMO_INVALID)", result3)
    save_result("demo_invalid", result3)

    # -----------------------------------------------------------------------
    # DEMO 4 -- Empty data (all missing)
    # -----------------------------------------------------------------------
    result4 = process_clinical_context({})
    print_result("Empty Data (no fields)", result4)
    save_result("demo_empty", result4)

    print(f"{BOLD}{'=' * 50}{RESET}")
    print(f"  Demo complete. Results saved to outputs/")
    print(f"{BOLD}{'=' * 50}{RESET}\n")


if __name__ == "__main__":
    main()

