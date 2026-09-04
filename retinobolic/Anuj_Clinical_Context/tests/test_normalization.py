"""
test_normalization.py — Unit tests for normalisation layer.

Tests cover:
  - Correct numeric precision per field
  - Unit consistency in output
  - FieldStatus tags for provided, missing, and not-recorded fields
  - Clinical history normalisation
"""

import pytest

from src.validation import validate_clinical_input
from src.normalization import normalise_clinical_input, FieldStatus
from src.config import AGE_UNIT, BP_UNIT, HBA1C_UNIT, DIABETES_DURATION_UNIT


class TestNormalisedPrecision:
    def _normalise(self, data: dict) -> dict:
        raw_keys = set(data.keys())
        parsed = validate_clinical_input(data)
        return normalise_clinical_input(parsed, raw_keys)

    def test_age_rounded_to_integer(self):
        result = self._normalise({"age": 58.7})
        assert result["clinical_context"]["age"]["value"] == 59
        assert isinstance(result["clinical_context"]["age"]["value"], int)

    def test_hba1c_rounded_to_one_decimal(self):
        result = self._normalise({"hba1c": 8.267})
        assert result["clinical_context"]["hba1c"]["value"] == 8.3

    def test_bp_rounded_to_integer(self):
        result = self._normalise({"bp_systolic": 148.6, "bp_diastolic": 92.4})
        assert result["clinical_context"]["bp_systolic"]["value"] == 149
        assert result["clinical_context"]["bp_diastolic"]["value"] == 92

    def test_diabetes_duration_rounded_to_one_decimal(self):
        result = self._normalise({"diabetes_duration_years": 10.567, "age": 58})
        assert result["clinical_context"]["diabetes_duration_years"]["value"] == 10.6


class TestUnitConsistency:
    def _normalise(self, data: dict) -> dict:
        raw_keys = set(data.keys())
        parsed = validate_clinical_input(data)
        return normalise_clinical_input(parsed, raw_keys)

    def test_age_unit_is_years(self):
        result = self._normalise({"age": 58})
        assert result["clinical_context"]["age"]["unit"] == AGE_UNIT

    def test_bp_unit_is_mmhg(self):
        result = self._normalise({"bp_systolic": 130, "bp_diastolic": 85})
        assert result["clinical_context"]["bp_systolic"]["unit"] == BP_UNIT
        assert result["clinical_context"]["bp_diastolic"]["unit"] == BP_UNIT

    def test_hba1c_unit_is_percent(self):
        result = self._normalise({"hba1c": 7.5})
        assert result["clinical_context"]["hba1c"]["unit"] == HBA1C_UNIT

    def test_duration_unit_is_years(self):
        result = self._normalise({"diabetes_duration_years": 5, "age": 40})
        assert result["clinical_context"]["diabetes_duration_years"]["unit"] == DIABETES_DURATION_UNIT


class TestFieldStatus:
    def _normalise(self, data: dict) -> dict:
        raw_keys = set(data.keys())
        parsed = validate_clinical_input(data)
        return normalise_clinical_input(parsed, raw_keys)

    def test_provided_field_has_provided_status(self):
        result = self._normalise({"age": 58})
        assert result["clinical_context"]["age"]["status"] == FieldStatus.PROVIDED.value

    def test_key_absent_has_missing_status(self):
        """Key not present in payload → MISSING."""
        result = self._normalise({})
        assert result["clinical_context"]["age"]["status"] == FieldStatus.MISSING.value

    def test_key_present_null_has_not_recorded_status(self):
        """Key present but value null → NOT_RECORDED."""
        result = self._normalise({"age": None})
        assert result["clinical_context"]["age"]["status"] == FieldStatus.NOT_RECORDED.value


class TestDataQuality:
    def _normalise(self, data: dict) -> dict:
        raw_keys = set(data.keys())
        parsed = validate_clinical_input(data)
        return normalise_clinical_input(parsed, raw_keys)

    def test_complete_data_is_complete_true(self):
        data = {
            "age": 58, "bp_systolic": 148, "bp_diastolic": 92,
            "hba1c": 8.2, "diabetes_duration_years": 10,
            "clinical_history": {"known_diabetes": True},
        }
        result = self._normalise(data)
        assert result["data_quality"]["complete"] is True
        assert result["data_quality"]["missing_fields"] == []

    def test_missing_hba1c_not_in_provided(self):
        result = self._normalise({"age": 58, "bp_systolic": 120, "bp_diastolic": 80})
        assert "hba1c" not in result["data_quality"]["provided_fields"]

    def test_missing_fields_listed_correctly(self):
        result = self._normalise({"age": 58})
        assert "hba1c" in result["data_quality"]["missing_fields"]
        assert "diabetes_duration_years" in result["data_quality"]["missing_fields"]

    def test_flags_bp_available_true_when_both_provided(self):
        result = self._normalise({"bp_systolic": 130, "bp_diastolic": 85})
        assert result["data_quality"]["flags"]["bp_available"] is True

    def test_flags_bp_available_false_when_only_systolic(self):
        result = self._normalise({"bp_systolic": 130})
        assert result["data_quality"]["flags"]["bp_available"] is False

    def test_clinical_context_complete_flag_mirrors_complete(self):
        result = self._normalise({})
        assert result["data_quality"]["clinical_context_complete"] is False


class TestHistoryNormalisation:
    def _normalise(self, data: dict) -> dict:
        raw_keys = set(data.keys())
        parsed = validate_clinical_input(data)
        return normalise_clinical_input(parsed, raw_keys)

    def test_history_normalised_to_dict(self):
        data = {"clinical_history": {"known_diabetes": True, "previous_dr_history": False}}
        result = self._normalise(data)
        hist = result["clinical_context"]["clinical_history"]["value"]
        assert isinstance(hist, dict)
        assert hist["known_diabetes"] is True

    def test_empty_history_dict_normalised(self):
        result = self._normalise({"clinical_history": {}})
        hist = result["clinical_context"]["clinical_history"]["value"]
        assert isinstance(hist, dict)
        assert hist == {}
