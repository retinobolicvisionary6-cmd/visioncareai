"""
clinical_context.py — Clinical Context Module: Core processor and public API.

Public API:
    process_clinical_context(patient_data) -> dict

This is the single entry point for the clinical context layer.
It validates, normalises, and structures patient clinical data
for consumption by the downstream Final Decision / Referral Priority layer.

ARCHITECTURE RULES:
  - Input: patient clinical fields (age, BP, HbA1c, duration, history).
  - Output: structured dict with "clinical_context" and "data_quality".
  - Does NOT modify DR grade or DR model probabilities.
  - Does NOT generate clinical diagnoses or risk scores.
  - Does NOT call external APIs or ML models.
  - Missing data is preserved as-is — never imputed.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Union

from pydantic import ValidationError

from src.config import ClinicalConfig, DEFAULT_CONFIG
from src.validation import ClinicalContextInput, validate_clinical_input
from src.normalization import normalise_clinical_input

# ---------------------------------------------------------------------------
# Module logger — do NOT log sensitive clinical values in production
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result Wrapper
# ---------------------------------------------------------------------------
class ClinicalContextResult:
    """
    Wrapper around the processed clinical context output.

    Attributes:
        clinical_context (dict): Normalised field values with units and status.
        data_quality (dict): Completeness summary and missing field list.
        validation_passed (bool): True if input validation succeeded.
        validation_errors (list[str]): List of validation error messages, if any.
    """

    def __init__(
        self,
        clinical_context: dict,
        data_quality: dict,
        validation_passed: bool = True,
        validation_errors: list[str] | None = None,
        patient_id: str | None = None,
    ):
        self.clinical_context = clinical_context
        self.data_quality = data_quality
        self.validation_passed = validation_passed
        self.validation_errors = validation_errors or []
        self.patient_id = patient_id

    def to_dict(self) -> dict:
        """Return the full result as a plain dictionary."""
        return {
            "clinical_context": self.clinical_context,
            "data_quality": self.data_quality,
            "validation_passed": self.validation_passed,
            "validation_errors": self.validation_errors,
        }

    def to_json(self, indent: int = 2) -> str:
        """Return the full result as a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def get_summary(self) -> str:
        """
        Return a brief, privacy-safe human-readable summary.
        Does NOT print raw clinical values — only completeness status.
        """
        pid = f"[{self.patient_id}]" if self.patient_id else "[no-id]"
        status = "PASS" if self.validation_passed else "FAIL"
        complete = "COMPLETE" if self.data_quality.get("complete") else "INCOMPLETE"
        missing = self.data_quality.get("missing_fields", [])
        lines = [
            f"Patient ID: {pid}",
            f"Validation: {status}",
            f"Context completeness: {complete}",
        ]
        if missing:
            lines.append(f"Missing fields: {', '.join(missing)}")
        if self.validation_errors:
            lines.append("Validation errors:")
            for e in self.validation_errors:
                lines.append(f"  • {e}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core Processor
# ---------------------------------------------------------------------------
def process_clinical_context(
    patient_data: Union[dict, ClinicalContextInput],
    config: ClinicalConfig = DEFAULT_CONFIG,
) -> dict:
    """
    Validate, normalise, and structure patient clinical context data.

    This is the public API for the Clinical Context Module.

    Args:
        patient_data: Raw patient data as a dict or ClinicalContextInput.
                      Expected keys (all optional except schema compliance):
                        - patient_id (str, optional)
                        - age (float)
                        - bp_systolic (float, mmHg)
                        - bp_diastolic (float, mmHg)
                        - hba1c (float, %)
                        - diabetes_duration_years (float)
                        - clinical_history (dict with optional flags)
        config: Optional ClinicalConfig to override validation bounds.

    Returns:
        dict with keys:
          - "clinical_context": normalised field values with units and status.
          - "data_quality": completeness summary.
          - "validation_passed": bool.
          - "validation_errors": list of error strings (empty if valid).

    Raises:
        TypeError: If patient_data is not a dict or ClinicalContextInput.

    Notes:
        - This function does NOT raise on validation failure by default;
          it returns validation_passed=False and populates validation_errors.
        - Call process_clinical_context_strict() if you need exceptions raised.
    """
    if not isinstance(patient_data, (dict, ClinicalContextInput)):
        raise TypeError(
            f"patient_data must be a dict or ClinicalContextInput, "
            f"got {type(patient_data).__name__}"
        )

    # Extract raw dict and key set for status tracking
    if isinstance(patient_data, ClinicalContextInput):
        raw_dict = patient_data.model_dump()
        raw_keys = set(raw_dict.keys())
        parsed = patient_data
        validation_errors: list[str] = []
        validation_passed = True
    else:
        raw_dict = patient_data
        raw_keys = set(raw_dict.keys())
        parsed = None
        validation_errors = []
        validation_passed = True

        try:
            parsed = validate_clinical_input(raw_dict, config)
        except ValidationError as exc:
            validation_passed = False
            validation_errors = [
                f"{err['loc'][0] if err['loc'] else 'input'}: {err['msg']}"
                for err in exc.errors()
            ]
            logger.warning(
                "Clinical context validation failed (patient_id=%s): %d error(s)",
                raw_dict.get("patient_id", "unknown"),
                len(validation_errors),
            )
        except ValueError as exc:
            validation_passed = False
            validation_errors = [str(exc)]
            logger.warning(
                "Clinical context validation failed (patient_id=%s): %s",
                raw_dict.get("patient_id", "unknown"),
                str(exc),
            )

    # If validation failed, return a structured error result
    if not validation_passed or parsed is None:
        return {
            "clinical_context": None,
            "data_quality": {
                "complete": False,
                "missing_fields": [],
                "provided_fields": [],
                "clinical_context_complete": False,
                "flags": {},
            },
            "validation_passed": False,
            "validation_errors": validation_errors,
        }

    # Normalise
    output = normalise_clinical_input(parsed, raw_keys)

    # Log summary only — no clinical values
    pid = raw_dict.get("patient_id", "unknown") if isinstance(raw_dict, dict) else "unknown"
    logger.info(
        "Clinical context processed (patient_id=%s): complete=%s, missing=%s",
        pid,
        output["data_quality"]["complete"],
        output["data_quality"]["missing_fields"],
    )

    output["validation_passed"] = True
    output["validation_errors"] = []
    return output


def process_clinical_context_strict(
    patient_data: Union[dict, ClinicalContextInput],
    config: ClinicalConfig = DEFAULT_CONFIG,
) -> dict:
    """
    Strict variant of process_clinical_context that raises on validation errors.

    Args:
        patient_data: Same as process_clinical_context.
        config: Optional ClinicalConfig.

    Returns:
        Same output dict as process_clinical_context.

    Raises:
        ValidationError: If Pydantic schema validation fails.
        ValueError: If cross-field sanity checks fail.
    """
    if isinstance(patient_data, ClinicalContextInput):
        parsed = patient_data
        raw_keys = set(patient_data.model_dump().keys())
    else:
        parsed = validate_clinical_input(patient_data, config)
        raw_keys = set(patient_data.keys())

    output = normalise_clinical_input(parsed, raw_keys)
    output["validation_passed"] = True
    output["validation_errors"] = []
    return output
