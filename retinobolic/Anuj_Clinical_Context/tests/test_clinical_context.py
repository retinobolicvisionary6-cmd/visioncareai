"""
test_clinical_context.py — End-to-end integration tests for process_clinical_context().

Covers all 7 master-prompt test cases plus edge cases:
  1. Complete valid data
  2. Missing HbA1c
  3. Invalid age (negative)
  4. Invalid BP (string type)
  5. Invalid HbA1c (non-numeric / out-of-range)
  6. Invalid diabetes duration (negative / exceeds age)
  7. Extra/unknown fields handling
  + Additional: Privacy safety, JSON serialisability, output structure invariants.
"""

import json
import pytest

from src.clinical_context import process_clinical_context, process_clinical_context_strict
from src.validation import ClinicalContextInput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _complete_data():
    return {
        "age": 58,
        "bp_systolic": 148,
        "bp_diastolic": 92,
        "hba1c": 8.2,
        "diabetes_duration_years": 10,
        "clinical_history": {"known_diabetes": True, "previous_dr_history": False},
    }


# ---------------------------------------------------------------------------
# TEST 1 — Complete valid data
# ---------------------------------------------------------------------------
class TestCase1_CompleteData:
    def test_returns_valid_true(self):
        result = process_clinical_context(_complete_data())
        assert result["validation_passed"] is True

    def test_complete_true(self):
        result = process_clinical_context(_complete_data())
        assert result["data_quality"]["complete"] is True

    def test_missing_fields_empty(self):
        result = process_clinical_context(_complete_data())
        assert result["data_quality"]["missing_fields"] == []

    def test_validation_errors_empty(self):
        result = process_clinical_context(_complete_data())
        assert result["validation_errors"] == []

    def test_clinical_context_values_correct(self):
        result = process_clinical_context(_complete_data())
        ctx = result["clinical_context"]
        assert ctx["age"]["value"] == 58
        assert ctx["bp_systolic"]["value"] == 148
        assert ctx["bp_diastolic"]["value"] == 92
        assert ctx["hba1c"]["value"] == 8.2
        assert ctx["diabetes_duration_years"]["value"] == 10.0

    def test_output_is_json_serialisable(self):
        result = process_clinical_context(_complete_data())
        serialised = json.dumps(result)
        assert isinstance(serialised, str)
        parsed_back = json.loads(serialised)
        assert parsed_back["validation_passed"] is True


# ---------------------------------------------------------------------------
# TEST 2 — Missing HbA1c
# ---------------------------------------------------------------------------
class TestCase2_MissingHba1c:
    def _data(self):
        data = _complete_data()
        del data["hba1c"]
        return data

    def test_validation_passed(self):
        result = process_clinical_context(self._data())
        assert result["validation_passed"] is True

    def test_complete_false(self):
        result = process_clinical_context(self._data())
        assert result["data_quality"]["complete"] is False

    def test_hba1c_in_missing_fields(self):
        result = process_clinical_context(self._data())
        assert "hba1c" in result["data_quality"]["missing_fields"]

    def test_hba1c_value_is_none_not_invented(self):
        """Critical: HbA1c must NOT be imputed or invented."""
        result = process_clinical_context(self._data())
        hba1c = result["clinical_context"]["hba1c"]
        assert hba1c["value"] is None

    def test_hba1c_status_is_missing(self):
        result = process_clinical_context(self._data())
        assert result["clinical_context"]["hba1c"]["status"] == "missing"


# ---------------------------------------------------------------------------
# TEST 3 — Invalid Age (negative)
# ---------------------------------------------------------------------------
class TestCase3_InvalidAge:
    def test_negative_age_validation_fails(self):
        result = process_clinical_context({"age": -5})
        assert result["validation_passed"] is False

    def test_negative_age_has_error_messages(self):
        result = process_clinical_context({"age": -5})
        assert len(result["validation_errors"]) > 0

    def test_negative_age_clinical_context_is_none(self):
        """On validation failure, clinical_context must be None — no partial data leaks."""
        result = process_clinical_context({"age": -5})
        assert result["clinical_context"] is None

    def test_strict_negative_age_raises(self):
        from pydantic import ValidationError
        with pytest.raises((ValidationError, ValueError)):
            process_clinical_context_strict({"age": -5})


# ---------------------------------------------------------------------------
# TEST 4 — Invalid BP (string)
# ---------------------------------------------------------------------------
class TestCase4_InvalidBP:
    def test_string_bp_systolic_fails(self):
        result = process_clinical_context({"bp_systolic": "abc", "bp_diastolic": 80})
        assert result["validation_passed"] is False

    def test_string_bp_has_errors(self):
        result = process_clinical_context({"bp_systolic": "abc", "bp_diastolic": 80})
        assert len(result["validation_errors"]) > 0

    def test_string_bp_context_is_none(self):
        result = process_clinical_context({"bp_systolic": "abc", "bp_diastolic": 80})
        assert result["clinical_context"] is None

    def test_list_bp_fails(self):
        result = process_clinical_context({"bp_systolic": [120, 130], "bp_diastolic": 80})
        assert result["validation_passed"] is False


# ---------------------------------------------------------------------------
# TEST 5 — Invalid HbA1c
# ---------------------------------------------------------------------------
class TestCase5_InvalidHba1c:
    def test_string_hba1c_fails(self):
        result = process_clinical_context({"hba1c": "high"})
        assert result["validation_passed"] is False

    def test_too_high_hba1c_fails(self):
        result = process_clinical_context({"hba1c": 50.0})
        assert result["validation_passed"] is False

    def test_too_low_hba1c_fails(self):
        result = process_clinical_context({"hba1c": 0.1})
        assert result["validation_passed"] is False

    def test_hba1c_none_passes(self):
        """None HbA1c = missing, must not raise."""
        result = process_clinical_context({"hba1c": None})
        assert result["validation_passed"] is True


# ---------------------------------------------------------------------------
# TEST 6 — Invalid Diabetes Duration
# ---------------------------------------------------------------------------
class TestCase6_InvalidDuration:
    def test_negative_duration_fails(self):
        result = process_clinical_context({"diabetes_duration_years": -3})
        assert result["validation_passed"] is False

    def test_duration_exceeds_age_fails(self):
        result = process_clinical_context({"diabetes_duration_years": 60, "age": 40})
        assert result["validation_passed"] is False

    def test_duration_exceeds_age_has_error_message(self):
        result = process_clinical_context({"diabetes_duration_years": 60, "age": 40})
        errors_text = " ".join(result["validation_errors"])
        assert "duration" in errors_text.lower() or "age" in errors_text.lower()

    def test_zero_duration_passes(self):
        result = process_clinical_context({"diabetes_duration_years": 0})
        assert result["validation_passed"] is True


# ---------------------------------------------------------------------------
# TEST 7 — Extra / Unknown Fields
# ---------------------------------------------------------------------------
class TestCase7_ExtraFields:
    def test_extra_fields_ignored_not_error(self):
        """
        Default schema policy: extra unknown fields are ignored (not an error).
        Only declared fields are processed and returned.
        """
        data = {
            "age": 50,
            "bp_systolic": 120,
            "bp_diastolic": 80,
            "unknown_clinical_field": "should_be_ignored",
            "random_key": 999,
        }
        result = process_clinical_context(data)
        assert result["validation_passed"] is True

    def test_extra_fields_not_in_output(self):
        data = {
            "age": 50,
            "future_field": "not_yet_implemented",
        }
        result = process_clinical_context(data)
        ctx = result["clinical_context"]
        assert "future_field" not in ctx


# ---------------------------------------------------------------------------
# Output structure invariants
# ---------------------------------------------------------------------------
class TestOutputStructureInvariants:
    def test_output_always_has_required_top_level_keys(self):
        for data in [
            _complete_data(),
            {"age": -99},  # invalid
            {},            # empty
        ]:
            result = process_clinical_context(data)
            assert "clinical_context" in result
            assert "data_quality" in result
            assert "validation_passed" in result
            assert "validation_errors" in result

    def test_no_dr_grade_in_output(self):
        """Clinical context output must never contain DR grade fields."""
        result = process_clinical_context(_complete_data())
        serialised = json.dumps(result)
        assert "dr_grade" not in serialised
        assert "grade" not in serialised

    def test_no_risk_score_in_output(self):
        """No invented medical risk scores should appear in output."""
        result = process_clinical_context(_complete_data())
        serialised = json.dumps(result).lower()
        assert "risk_score" not in serialised
        assert "dr_risk" not in serialised
        assert "high risk" not in serialised

    def test_accepts_clinicalcontextinput_directly(self):
        """Public API must accept a ClinicalContextInput object."""
        parsed = ClinicalContextInput(age=50, hba1c=7.0)
        result = process_clinical_context(parsed)
        assert result["validation_passed"] is True

    def test_wrong_type_raises_typeerror(self):
        with pytest.raises(TypeError):
            process_clinical_context("not a dict")

    def test_complete_flag_true_only_when_all_fields_provided(self):
        result = process_clinical_context(_complete_data())
        assert result["data_quality"]["complete"] is True

    def test_complete_flag_false_when_any_field_missing(self):
        data = _complete_data()
        del data["hba1c"]
        result = process_clinical_context(data)
        assert result["data_quality"]["complete"] is False
