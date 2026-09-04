"""
validation.py — Clinical Context Module: Pydantic schemas and validators.

Validates raw patient-level clinical input for:
  - Type correctness
  - Plausible physiological range (configurable via ClinicalConfig)
  - Cross-field sanity (systolic > diastolic; duration <= age)

IMPORTANT DESIGN RULES:
  - This module does NOT generate clinical diagnoses.
  - This module does NOT compute DR grades or risk scores.
  - Validation bounds are NOT diagnostic thresholds.
  - Missing fields are allowed; they are tracked explicitly — never imputed.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.config import ClinicalConfig, DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Clinical History Sub-Schema
# ---------------------------------------------------------------------------
class ClinicalHistory(BaseModel):
    """
    Structured optional clinical history flags.

    Extra/unknown fields are silently ignored (not an error).
    This allows forward-compatible schema evolution.
    """

    model_config = ConfigDict(extra="ignore")

    known_diabetes: Optional[bool] = Field(
        default=None,
        description="Whether the patient has a known diabetes diagnosis.",
    )
    previous_dr_history: Optional[bool] = Field(
        default=None,
        description="Whether the patient has a history of Diabetic Retinopathy.",
    )
    other_notes: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Free-text clinical notes. Keep brief; do not include PII.",
    )


# ---------------------------------------------------------------------------
# Main Input Schema
# ---------------------------------------------------------------------------
class ClinicalContextInput(BaseModel):
    """
    Raw patient clinical context input.

    All clinical measurement fields are optional — the module handles
    missing data explicitly without imputation.

    Extra/unknown top-level fields are ignored to allow schema evolution.
    Document this behaviour: callers should not rely on extra fields being
    passed through; only declared fields are processed.
    """

    model_config = ConfigDict(extra="ignore")

    # Patient identifier — optional for privacy-sensitive deployments
    patient_id: Optional[str] = Field(
        default=None,
        description="Optional patient/session identifier. Do not include PII.",
    )

    # --- Clinical Measurements ---
    age: Optional[float] = Field(
        default=None,
        description="Patient age in years.",
    )
    bp_systolic: Optional[float] = Field(
        default=None,
        description="Systolic blood pressure in mmHg.",
    )
    bp_diastolic: Optional[float] = Field(
        default=None,
        description="Diastolic blood pressure in mmHg.",
    )
    hba1c: Optional[float] = Field(
        default=None,
        description="Glycated haemoglobin (HbA1c) in percent (%). "
                    "Unit must be % — do not enter IFCC mmol/mol values without conversion.",
    )
    diabetes_duration_years: Optional[float] = Field(
        default=None,
        description="Duration of diabetes diagnosis in years.",
    )
    clinical_history: Optional[ClinicalHistory] = Field(
        default=None,
        description="Structured optional clinical history flags.",
    )

    # -------------------------------------------------------------------
    # Individual field validators
    # -------------------------------------------------------------------

    @model_validator(mode="before")
    @classmethod
    def _coerce_and_check_types(cls, values: Any) -> Any:
        """
        Pre-validation: ensure numeric fields are actually numeric.
        Rejects strings like 'abc' before Pydantic tries float coercion,
        producing a clear error message instead of a Pydantic parse error.
        """
        numeric_fields = [
            "age", "bp_systolic", "bp_diastolic",
            "hba1c", "diabetes_duration_years",
        ]
        for fname in numeric_fields:
            val = values.get(fname) if isinstance(values, dict) else getattr(values, fname, None)
            if val is None:
                continue
            # Accept int and float; reject str, list, dict, etc.
            if not isinstance(val, (int, float)):
                raise ValueError(
                    f"Field '{fname}' must be a numeric value (int or float). "
                    f"Received type '{type(val).__name__}': {val!r}"
                )
            # Reject NaN and Inf
            if math.isnan(float(val)) or math.isinf(float(val)):
                raise ValueError(
                    f"Field '{fname}' must be a finite number. "
                    f"Received non-finite value: {val!r}"
                )
        return values

    @model_validator(mode="after")
    def _validate_bounds_and_cross_fields(self) -> "ClinicalContextInput":
        """
        Post-parse: check plausible physiological ranges and cross-field
        sanity constraints using the default configuration.
        """
        return _apply_bounds_validation(self, DEFAULT_CONFIG)


# ---------------------------------------------------------------------------
# Bounds Validation (separated so config can be injected)
# ---------------------------------------------------------------------------
def _apply_bounds_validation(
    data: ClinicalContextInput,
    config: ClinicalConfig,
) -> ClinicalContextInput:
    """
    Apply configurable bounds and cross-field sanity checks.
    Called by the model_validator and also usable standalone for testing.
    """
    errors: list[str] = []

    # Age
    if data.age is not None:
        if data.age < config.age_min or data.age > config.age_max:
            errors.append(
                f"Age {data.age} is outside plausible range "
                f"[{config.age_min}, {config.age_max}] years."
            )
        if data.age < 0:
            errors.append("Age must be non-negative.")

    # BP Systolic
    if data.bp_systolic is not None:
        if not (config.bp_systolic_min <= data.bp_systolic <= config.bp_systolic_max):
            errors.append(
                f"bp_systolic {data.bp_systolic} mmHg is outside plausible range "
                f"[{config.bp_systolic_min}, {config.bp_systolic_max}] mmHg."
            )

    # BP Diastolic
    if data.bp_diastolic is not None:
        if not (config.bp_diastolic_min <= data.bp_diastolic <= config.bp_diastolic_max):
            errors.append(
                f"bp_diastolic {data.bp_diastolic} mmHg is outside plausible range "
                f"[{config.bp_diastolic_min}, {config.bp_diastolic_max}] mmHg."
            )

    # Cross-field: systolic > diastolic
    if data.bp_systolic is not None and data.bp_diastolic is not None:
        if data.bp_systolic <= data.bp_diastolic:
            errors.append(
                f"bp_systolic ({data.bp_systolic}) must be greater than "
                f"bp_diastolic ({data.bp_diastolic})."
            )

    # HbA1c
    if data.hba1c is not None:
        if not (config.hba1c_min <= data.hba1c <= config.hba1c_max):
            errors.append(
                f"hba1c {data.hba1c}% is outside plausible range "
                f"[{config.hba1c_min}, {config.hba1c_max}]%. "
                "Unit must be % — do not enter IFCC mmol/mol values."
            )

    # Diabetes Duration
    if data.diabetes_duration_years is not None:
        if data.diabetes_duration_years < config.diabetes_duration_min:
            errors.append(
                f"diabetes_duration_years {data.diabetes_duration_years} must be non-negative."
            )
        if data.diabetes_duration_years > config.diabetes_duration_max:
            errors.append(
                f"diabetes_duration_years {data.diabetes_duration_years} exceeds "
                f"plausible upper bound of {config.diabetes_duration_max} years."
            )

    # Cross-field: duration <= age (sanity only, allowed to differ e.g. if age unknown)
    if data.diabetes_duration_years is not None and data.age is not None:
        if data.diabetes_duration_years > data.age:
            errors.append(
                f"diabetes_duration_years ({data.diabetes_duration_years}) cannot exceed "
                f"patient age ({data.age})."
            )

    if errors:
        raise ValueError(
            "Clinical context validation failed:\n" + "\n".join(f"  • {e}" for e in errors)
        )

    return data


# ---------------------------------------------------------------------------
# Convenience: validate with custom config
# ---------------------------------------------------------------------------
def validate_clinical_input(
    raw_data: dict,
    config: ClinicalConfig = DEFAULT_CONFIG,
) -> ClinicalContextInput:
    """
    Validate raw dictionary input against the clinical schema.

    Args:
        raw_data: Dictionary matching ClinicalContextInput schema.
        config: Optional custom ClinicalConfig to override bounds.

    Returns:
        Validated ClinicalContextInput instance.

    Raises:
        pydantic.ValidationError: If the input fails type or bounds validation.
        ValueError: If cross-field sanity checks fail.
    """
    # Parse with Pydantic (runs pre + post validators)
    parsed = ClinicalContextInput.model_validate(raw_data)

    # If a non-default config was provided, re-run bounds with it
    if config is not DEFAULT_CONFIG:
        _apply_bounds_validation(parsed, config)

    return parsed
