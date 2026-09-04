"""
test_validation.py - Deterministic tests for the probability validation layer.

Every test must pass without network access, trained models, or external data.
All expected outputs are derived analytically.
"""

import math
import pytest
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.validation import (
    validate_probabilities,
    validate_dr_input,
    InvalidProbabilityError,
    ValidationError,
)
from src.config import UncertaintyConfig, DEFAULT_CONFIG


# ===========================================================================
# Helper
# ===========================================================================

VALID_DICT = {"0": 0.03, "1": 0.08, "2": 0.81, "3": 0.08}
VALID_LIST = [0.03, 0.08, 0.81, 0.08]


# ===========================================================================
# 1. Valid input formats
# ===========================================================================

class TestValidInputFormats:

    def test_dict_string_keys(self):
        probs = validate_probabilities({"0": 0.03, "1": 0.08, "2": 0.81, "3": 0.08})
        assert probs.shape == (4,)
        assert isinstance(probs, np.ndarray)
        assert probs.dtype == np.float64

    def test_dict_integer_keys(self):
        probs = validate_probabilities({0: 0.03, 1: 0.08, 2: 0.81, 3: 0.08})
        np.testing.assert_allclose(probs, [0.03, 0.08, 0.81, 0.08])

    def test_list_input(self):
        probs = validate_probabilities([0.03, 0.08, 0.81, 0.08])
        assert probs.shape == (4,)

    def test_tuple_input(self):
        probs = validate_probabilities((0.25, 0.25, 0.25, 0.25))
        assert probs.shape == (4,)

    def test_numpy_array_input(self):
        arr = np.array([0.03, 0.08, 0.81, 0.08])
        probs = validate_probabilities(arr)
        np.testing.assert_allclose(probs, arr)

    def test_sum_tolerance_accepted(self):
        # 0.9999 is within the default tolerance of 1e-3
        probs = validate_probabilities([0.9999, 0.0, 0.0, 0.0])
        assert probs is not None

    def test_sum_very_close_to_1(self):
        # Floating-point accumulation: values sum to 1.0000000000000002
        vals = [0.25, 0.25, 0.25, 0.25]
        # numpy sum may produce tiny rounding
        probs = validate_probabilities(vals)
        assert probs is not None

    def test_integer_probabilities_cast(self):
        # 0 and 1 are valid probabilities
        probs = validate_probabilities([1, 0, 0, 0])
        assert probs[0] == pytest.approx(1.0)


# ===========================================================================
# 2. Invalid: wrong class count
# ===========================================================================

class TestWrongClassCount:

    def test_too_few_list(self):
        with pytest.raises(InvalidProbabilityError, match="Expected 4"):
            validate_probabilities([0.5, 0.5])

    def test_too_many_list(self):
        with pytest.raises(InvalidProbabilityError, match="Expected 4"):
            validate_probabilities([0.2, 0.2, 0.2, 0.2, 0.2])

    def test_dict_missing_key(self):
        with pytest.raises(InvalidProbabilityError, match="Missing class key"):
            validate_probabilities({"0": 0.5, "1": 0.3, "2": 0.2})

    def test_dict_extra_key(self):
        with pytest.raises(InvalidProbabilityError, match="Unexpected class key"):
            validate_probabilities({"0": 0.2, "1": 0.2, "2": 0.2, "3": 0.2, "4": 0.2})

    def test_empty_dict(self):
        with pytest.raises(InvalidProbabilityError, match="empty"):
            validate_probabilities({})

    def test_empty_list(self):
        with pytest.raises(InvalidProbabilityError, match="empty"):
            validate_probabilities([])

    def test_empty_array(self):
        with pytest.raises(InvalidProbabilityError, match="empty"):
            validate_probabilities(np.array([]))


# ===========================================================================
# 3. Invalid: bad values
# ===========================================================================

class TestInvalidValues:

    def test_negative_probability(self):
        with pytest.raises(InvalidProbabilityError, match="negative"):
            validate_probabilities([-0.1, 0.3, 0.4, 0.4])

    def test_probability_exceeds_1(self):
        with pytest.raises(InvalidProbabilityError, match="exceeds 1"):
            validate_probabilities([1.2, 0.0, 0.0, 0.0])

    def test_nan_value(self):
        with pytest.raises(InvalidProbabilityError, match="NaN"):
            validate_probabilities([float("nan"), 0.33, 0.33, 0.34])

    def test_positive_infinity(self):
        with pytest.raises(InvalidProbabilityError, match="infinity"):
            validate_probabilities([float("inf"), 0.0, 0.0, 0.0])

    def test_negative_infinity(self):
        with pytest.raises(InvalidProbabilityError, match="infinity"):
            validate_probabilities([float("-inf"), 0.5, 0.5, 0.0])

    def test_none_element(self):
        with pytest.raises(InvalidProbabilityError, match="None"):
            validate_probabilities([None, 0.33, 0.33, 0.34])

    def test_string_element(self):
        with pytest.raises(InvalidProbabilityError):
            validate_probabilities(["0.5", 0.2, 0.2, 0.1])

    def test_none_input(self):
        with pytest.raises(InvalidProbabilityError, match="None"):
            validate_probabilities(None)

    def test_integer_input_type(self):
        """Completely wrong type (not dict/list/tuple/ndarray)."""
        with pytest.raises(ValidationError):
            validate_probabilities(42)

    def test_string_input_type(self):
        with pytest.raises(ValidationError):
            validate_probabilities("0.25, 0.25, 0.25, 0.25")


# ===========================================================================
# 4. Invalid: sum mismatch
# ===========================================================================

class TestSumMismatch:

    def test_sum_too_high(self):
        # [0.4, 0.4, 0.4, 0.1] sums to 1.3
        with pytest.raises(InvalidProbabilityError, match="sum"):
            validate_probabilities([0.4, 0.4, 0.4, 0.1])

    def test_sum_too_low(self):
        with pytest.raises(InvalidProbabilityError, match="sum"):
            validate_probabilities([0.1, 0.1, 0.1, 0.1])

    def test_all_zeros(self):
        with pytest.raises(InvalidProbabilityError, match="sum"):
            validate_probabilities([0.0, 0.0, 0.0, 0.0])

    def test_custom_tolerance_tighter(self):
        """With a tighter tolerance, even small deviations should fail."""
        tight_config = UncertaintyConfig(PROB_SUM_TOLERANCE=1e-6)
        with pytest.raises(InvalidProbabilityError, match="sum"):
            validate_probabilities([0.9999, 0.0, 0.0, 0.0], config=tight_config)

    def test_custom_tolerance_looser(self):
        """With a looser tolerance, a slightly wrong sum should pass."""
        loose_config = UncertaintyConfig(PROB_SUM_TOLERANCE=0.05)
        # Sum = 0.98, deviation = 0.02 < 0.05
        probs = validate_probabilities([0.90, 0.04, 0.02, 0.02], config=loose_config)
        assert probs is not None


# ===========================================================================
# 5. validate_dr_input
# ===========================================================================

class TestValidateDrInput:

    def test_full_valid_input(self):
        dr_result = {
            "grade": 2,
            "probabilities": {"0": 0.03, "1": 0.08, "2": 0.81, "3": 0.08},
            "gradcam_path": "outputs/gradcam/image_001.jpg",
        }
        probs, grade = validate_dr_input(dr_result)
        assert grade == 2
        assert probs.shape == (4,)

    def test_missing_probabilities_key(self):
        with pytest.raises(ValidationError, match="probabilities"):
            validate_dr_input({"grade": 2})

    def test_not_a_dict(self):
        with pytest.raises(ValidationError):
            validate_dr_input([0.25, 0.25, 0.25, 0.25])

    def test_grade_optional(self):
        dr_result = {"probabilities": {"0": 0.1, "1": 0.2, "2": 0.6, "3": 0.1}}
        probs, grade = validate_dr_input(dr_result)
        assert grade is None

    def test_float_grade_converted(self):
        dr_result = {"grade": 2.0, "probabilities": {"0": 0.1, "1": 0.2, "2": 0.6, "3": 0.1}}
        _, grade = validate_dr_input(dr_result)
        assert grade == 2
        assert isinstance(grade, int)
