"""
test_validation.py — Unit tests for ClinicalContextInput schema validation.

Tests cover:
  - Valid complete input
  - Out-of-range age (negative, too large)
  - Invalid BP types and out-of-range values
  - Diastolic >= systolic cross-field failure
  - Out-of-range HbA1c and non-numeric HbA1c
  - Negative and out-of-bounds diabetes duration
  - Duration > age cross-field failure
  - Extra/unknown top-level fields (ignored)
  - NaN / Inf values rejected
"""

import math
import pytest
from pydantic import ValidationError

from src.validation import ClinicalContextInput, validate_clinical_input
from src.config import ClinicalConfig


# ---------------------------------------------------------------------------
# TEST 1 — Complete valid data
# ---------------------------------------------------------------------------
class TestCompleteValidInput:
    def test_complete_valid_passes(self):
        data = {
            "age": 58,
            "bp_systolic": 148,
            "bp_diastolic": 92,
            "hba1c": 8.2,
            "diabetes_duration_years": 10,
            "clinical_history": {"known_diabetes": True, "previous_dr_history": False},
        }
        parsed = validate_clinical_input(data)
        assert parsed.age == 58
        assert parsed.bp_systolic == 148
        assert parsed.bp_diastolic == 92
        assert parsed.hba1c == 8.2
        assert parsed.diabetes_duration_years == 10
        assert parsed.clinical_history.known_diabetes is True

    def test_float_age_accepted(self):
        """Float age like 58.5 should be accepted."""
        data = {"age": 58.5, "bp_systolic": 120, "bp_diastolic": 80}
        parsed = validate_clinical_input(data)
        assert parsed.age == 58.5

    def test_optional_patient_id(self):
        data = {"patient_id": "DEMO001", "age": 40}
        parsed = validate_clinical_input(data)
        assert parsed.patient_id == "DEMO001"

    def test_all_fields_none_is_valid(self):
        """Fully missing input should still parse without error."""
        data = {}
        parsed = validate_clinical_input(data)
        assert parsed.age is None
        assert parsed.hba1c is None


# ---------------------------------------------------------------------------
# TEST 3 — Invalid Age
# ---------------------------------------------------------------------------
class TestAgeValidation:
    def test_negative_age_fails(self):
        with pytest.raises((ValidationError, ValueError)):
            validate_clinical_input({"age": -5})

    def test_zero_age_accepted(self):
        parsed = validate_clinical_input({"age": 0})
        assert parsed.age == 0

    def test_age_too_large_fails(self):
        with pytest.raises((ValidationError, ValueError)):
            validate_clinical_input({"age": 200})

    def test_age_exactly_at_max_passes(self):
        parsed = validate_clinical_input({"age": 130})
        assert parsed.age == 130

    def test_string_age_fails(self):
        with pytest.raises((ValidationError, ValueError)):
            validate_clinical_input({"age": "fifty"})

    def test_none_age_passes(self):
        """None age = missing, should not raise."""
        parsed = validate_clinical_input({"age": None})
        assert parsed.age is None


# ---------------------------------------------------------------------------
# TEST 4 — Invalid Blood Pressure
# ---------------------------------------------------------------------------
class TestBPValidation:
    def test_string_bp_systolic_fails(self):
        with pytest.raises((ValidationError, ValueError)):
            validate_clinical_input({"bp_systolic": "abc", "bp_diastolic": 80})

    def test_string_bp_diastolic_fails(self):
        with pytest.raises((ValidationError, ValueError)):
            validate_clinical_input({"bp_systolic": 120, "bp_diastolic": "high"})

    def test_negative_bp_systolic_fails(self):
        with pytest.raises((ValidationError, ValueError)):
            validate_clinical_input({"bp_systolic": -10, "bp_diastolic": 80})

    def test_bp_out_of_range_fails(self):
        with pytest.raises((ValidationError, ValueError)):
            validate_clinical_input({"bp_systolic": 500, "bp_diastolic": 80})

    def test_diastolic_greater_than_systolic_fails(self):
        """Diastolic must be less than systolic."""
        with pytest.raises((ValidationError, ValueError)):
            validate_clinical_input({"bp_systolic": 80, "bp_diastolic": 120})

    def test_diastolic_equal_to_systolic_fails(self):
        with pytest.raises((ValidationError, ValueError)):
            validate_clinical_input({"bp_systolic": 100, "bp_diastolic": 100})

    def test_valid_bp_passes(self):
        parsed = validate_clinical_input({"bp_systolic": 120, "bp_diastolic": 80})
        assert parsed.bp_systolic == 120
        assert parsed.bp_diastolic == 80

    def test_only_systolic_provided_passes(self):
        """Partial BP (only systolic) is allowed — no cross-field check."""
        parsed = validate_clinical_input({"bp_systolic": 130})
        assert parsed.bp_systolic == 130
        assert parsed.bp_diastolic is None


# ---------------------------------------------------------------------------
# TEST 5 — Invalid HbA1c
# ---------------------------------------------------------------------------
class TestHba1cValidation:
    def test_string_hba1c_fails(self):
        with pytest.raises((ValidationError, ValueError)):
            validate_clinical_input({"hba1c": "high"})

    def test_hba1c_below_min_fails(self):
        with pytest.raises((ValidationError, ValueError)):
            validate_clinical_input({"hba1c": 0.5})

    def test_hba1c_above_max_fails(self):
        with pytest.raises((ValidationError, ValueError)):
            validate_clinical_input({"hba1c": 30.0})

    def test_valid_hba1c_passes(self):
        parsed = validate_clinical_input({"hba1c": 8.2})
        assert parsed.hba1c == 8.2

    def test_none_hba1c_passes(self):
        parsed = validate_clinical_input({"hba1c": None})
        assert parsed.hba1c is None


# ---------------------------------------------------------------------------
# TEST 6 — Invalid Diabetes Duration
# ---------------------------------------------------------------------------
class TestDiabetesDurationValidation:
    def test_negative_duration_fails(self):
        with pytest.raises((ValidationError, ValueError)):
            validate_clinical_input({"diabetes_duration_years": -3, "age": 50})

    def test_duration_exceeds_age_fails(self):
        with pytest.raises((ValidationError, ValueError)):
            validate_clinical_input({"diabetes_duration_years": 60, "age": 40})

    def test_duration_equals_age_passes(self):
        """Duration exactly equal to age should pass."""
        parsed = validate_clinical_input({"diabetes_duration_years": 40, "age": 40})
        assert parsed.diabetes_duration_years == 40

    def test_zero_duration_passes(self):
        parsed = validate_clinical_input({"diabetes_duration_years": 0})
        assert parsed.diabetes_duration_years == 0

    def test_excessive_duration_fails(self):
        with pytest.raises((ValidationError, ValueError)):
            validate_clinical_input({"diabetes_duration_years": 150})

    def test_valid_duration_passes(self):
        parsed = validate_clinical_input({"diabetes_duration_years": 10, "age": 58})
        assert parsed.diabetes_duration_years == 10


# ---------------------------------------------------------------------------
# TEST 7 — Extra / Unknown Fields
# ---------------------------------------------------------------------------
class TestExtraFields:
    def test_extra_top_level_fields_ignored(self):
        """
        Unknown top-level keys are silently ignored (extra='ignore').
        Only declared fields are processed. This is documented behaviour.
        """
        data = {
            "age": 50,
            "unknown_field": "should_be_ignored",
            "random_key": 999,
        }
        parsed = validate_clinical_input(data)
        assert parsed.age == 50
        # Extra fields must not appear on the model
        assert not hasattr(parsed, "unknown_field")
        assert not hasattr(parsed, "random_key")

    def test_extra_history_fields_ignored(self):
        """Unknown keys in clinical_history are also ignored."""
        data = {
            "clinical_history": {
                "known_diabetes": True,
                "future_field": "not_yet_defined",
            }
        }
        parsed = validate_clinical_input(data)
        assert parsed.clinical_history.known_diabetes is True


# ---------------------------------------------------------------------------
# Non-finite value tests
# ---------------------------------------------------------------------------
class TestNonFiniteValues:
    def test_nan_age_fails(self):
        with pytest.raises((ValidationError, ValueError)):
            validate_clinical_input({"age": float("nan")})

    def test_inf_bp_fails(self):
        with pytest.raises((ValidationError, ValueError)):
            validate_clinical_input({"bp_systolic": float("inf")})

    def test_negative_inf_hba1c_fails(self):
        with pytest.raises((ValidationError, ValueError)):
            validate_clinical_input({"hba1c": float("-inf")})


# ---------------------------------------------------------------------------
# Custom config tests
# ---------------------------------------------------------------------------
class TestCustomConfig:
    def test_custom_bounds_applied(self):
        """Custom config with tighter age bounds should reject age=130."""
        tight_config = ClinicalConfig(age_min=18, age_max=100)
        with pytest.raises((ValidationError, ValueError)):
            validate_clinical_input({"age": 130}, config=tight_config)

    def test_custom_bounds_pass(self):
        tight_config = ClinicalConfig(age_min=18, age_max=100)
        parsed = validate_clinical_input({"age": 65}, config=tight_config)
        assert parsed.age == 65
