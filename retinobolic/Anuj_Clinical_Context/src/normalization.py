"""
normalization.py — Clinical Context Module: Value formatting and field status.

Responsible for:
  - Standardising numeric precision for each clinical field.
  - Tagging each field with its data availability status.
  - Producing unit-aware structured output.

DESIGN RULES:
  - Missing fields are explicitly marked — never imputed or invented.
  - No clinical thresholds or classifications are applied here.
  - No DR grade modifications occur here.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from src.config import (
    AGE_UNIT,
    BP_UNIT,
    HBA1C_UNIT,
    DIABETES_DURATION_UNIT,
)
from src.validation import ClinicalContextInput, ClinicalHistory


# ---------------------------------------------------------------------------
# Field Status Enum
# ---------------------------------------------------------------------------
class FieldStatus(str, Enum):
    """
    Explicit data availability status for each clinical field.

    PROVIDED    — value was supplied and passed validation.
    MISSING     — field was absent from the input payload.
    NOT_RECORDED— field was explicitly set to null/None by the caller,
                  indicating it was known to be unavailable at time of entry.
    """
    PROVIDED = "provided"
    MISSING = "missing"
    NOT_RECORDED = "not_recorded"


# ---------------------------------------------------------------------------
# Normalised Value Containers
# ---------------------------------------------------------------------------
def _field_status(value: Any, original_key_present: bool = True) -> FieldStatus:
    """
    Determine the status of a field based on its value and whether the
    key existed in the original payload.

    Args:
        value: The parsed field value (None = absent or null).
        original_key_present: True if the key existed in the raw dict.
    """
    if value is not None:
        return FieldStatus.PROVIDED
    if original_key_present:
        return FieldStatus.NOT_RECORDED
    return FieldStatus.MISSING


def _round_or_none(value: Optional[float], decimals: int) -> Optional[float]:
    """Round a float to the given decimal places, or return None."""
    if value is None:
        return None
    return round(float(value), decimals)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def normalise_clinical_input(
    parsed: ClinicalContextInput,
    raw_keys: set[str],
) -> dict:
    """
    Convert a validated ClinicalContextInput into a normalised, structured
    output dictionary with field-level status tags.

    Args:
        parsed: Validated ClinicalContextInput instance.
        raw_keys: Set of keys that were present in the raw input dict
                  (used to distinguish MISSING vs NOT_RECORDED).

    Returns:
        Dictionary with two top-level keys:
          - "clinical_context": normalised field values and units.
          - "data_quality": completeness summary and missing field list.
    """

    # -----------------------------------------------------------------------
    # Normalise each field with precision and units
    # -----------------------------------------------------------------------
    age_val = _round_or_none(parsed.age, 0)
    if age_val is not None:
        age_val = int(age_val)

    bp_sys_val = _round_or_none(parsed.bp_systolic, 0)
    if bp_sys_val is not None:
        bp_sys_val = int(bp_sys_val)

    bp_dia_val = _round_or_none(parsed.bp_diastolic, 0)
    if bp_dia_val is not None:
        bp_dia_val = int(bp_dia_val)

    hba1c_val = _round_or_none(parsed.hba1c, 1)
    dur_val = _round_or_none(parsed.diabetes_duration_years, 1)

    # Normalise history — convert Pydantic model to plain dict if present
    history_val: Optional[dict] = None
    if parsed.clinical_history is not None:
        history_val = parsed.clinical_history.model_dump(exclude_none=True)

    # -----------------------------------------------------------------------
    # Determine field statuses
    # -----------------------------------------------------------------------
    statuses = {
        "age":                    _field_status(age_val,    "age" in raw_keys),
        "bp_systolic":            _field_status(bp_sys_val, "bp_systolic" in raw_keys),
        "bp_diastolic":           _field_status(bp_dia_val, "bp_diastolic" in raw_keys),
        "hba1c":                  _field_status(hba1c_val,  "hba1c" in raw_keys),
        "diabetes_duration_years":_field_status(dur_val,    "diabetes_duration_years" in raw_keys),
        "clinical_history":       _field_status(history_val,"clinical_history" in raw_keys),
    }

    # -----------------------------------------------------------------------
    # Identify missing fields (MISSING or NOT_RECORDED, i.e. value is None)
    # -----------------------------------------------------------------------
    missing_fields = [
        fname for fname, status in statuses.items()
        if status != FieldStatus.PROVIDED
    ]
    is_complete = len(missing_fields) == 0

    # -----------------------------------------------------------------------
    # Availability flags (non-diagnostic, data-quality signals only)
    # -----------------------------------------------------------------------
    context_flags = {
        "age_available":                  statuses["age"] == FieldStatus.PROVIDED,
        "bp_available":                   (
            statuses["bp_systolic"] == FieldStatus.PROVIDED
            and statuses["bp_diastolic"] == FieldStatus.PROVIDED
        ),
        "hba1c_available":                statuses["hba1c"] == FieldStatus.PROVIDED,
        "diabetes_duration_available":    statuses["diabetes_duration_years"] == FieldStatus.PROVIDED,
        "clinical_history_available":     statuses["clinical_history"] == FieldStatus.PROVIDED,
    }

    # -----------------------------------------------------------------------
    # Build output
    # -----------------------------------------------------------------------
    clinical_context = {
        "patient_id": parsed.patient_id,
        "age": {
            "value": age_val,
            "unit": AGE_UNIT,
            "status": statuses["age"].value,
        },
        "bp_systolic": {
            "value": bp_sys_val,
            "unit": BP_UNIT,
            "status": statuses["bp_systolic"].value,
        },
        "bp_diastolic": {
            "value": bp_dia_val,
            "unit": BP_UNIT,
            "status": statuses["bp_diastolic"].value,
        },
        "hba1c": {
            "value": hba1c_val,
            "unit": HBA1C_UNIT,
            "status": statuses["hba1c"].value,
        },
        "diabetes_duration_years": {
            "value": dur_val,
            "unit": DIABETES_DURATION_UNIT,
            "status": statuses["diabetes_duration_years"].value,
        },
        "clinical_history": {
            "value": history_val,
            "status": statuses["clinical_history"].value,
        },
    }

    data_quality = {
        "complete": is_complete,
        "missing_fields": missing_fields,
        "provided_fields": [
            fname for fname, status in statuses.items()
            if status == FieldStatus.PROVIDED
        ],
        "clinical_context_complete": is_complete,
        "flags": context_flags,
    }

    return {
        "clinical_context": clinical_context,
        "data_quality": data_quality,
    }
