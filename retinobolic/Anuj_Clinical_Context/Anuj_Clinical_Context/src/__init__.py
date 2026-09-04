"""
Clinical Context Module — Public API.

Usage:
    from src import process_clinical_context

    result = process_clinical_context({
        "age": 58,
        "bp_systolic": 148,
        "bp_diastolic": 92,
        "hba1c": 8.2,
        "diabetes_duration_years": 10,
        "clinical_history": {"known_diabetes": True},
    })
"""

from src.clinical_context import (
    process_clinical_context,
    process_clinical_context_strict,
    ClinicalContextResult,
)
from src.validation import ClinicalContextInput, ClinicalHistory, validate_clinical_input
from src.normalization import FieldStatus
from src.config import ClinicalConfig, DEFAULT_CONFIG

__all__ = [
    # Primary public API
    "process_clinical_context",
    "process_clinical_context_strict",
    "ClinicalContextResult",
    # Schema types
    "ClinicalContextInput",
    "ClinicalHistory",
    "validate_clinical_input",
    # Enums & config
    "FieldStatus",
    "ClinicalConfig",
    "DEFAULT_CONFIG",
]
