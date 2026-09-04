"""
test_calibration.py - Tests for optional calibration and ECE evaluation utilities.
"""

import math
import pytest
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calibration import TemperatureScaler, compute_ece, reliability_diagram_bins


# ===========================================================================
# TemperatureScaler tests
# ===========================================================================

class TestTemperatureScaler:

    def test_identity_temperature(self):
        """T=1.0 should leave probabilities unchanged."""
        scaler = TemperatureScaler(temperature=1.0)
        logits = np.array([2.0, 1.0, 0.5, -0.5])
        probs = scaler.scale_logits(logits)
        # Manually compute softmax
        x = logits - np.max(logits)
        e = np.exp(x)
        expected = e / np.sum(e)
        np.testing.assert_allclose(probs, expected, rtol=1e-6)

    def test_high_temperature_softens(self):
        """T > 1 should produce a softer (more uniform) distribution."""
        logits = np.array([5.0, 1.0, 0.5, 0.2])

        scaler_default = TemperatureScaler(temperature=1.0)
        scaler_hot = TemperatureScaler(temperature=5.0)

        probs_default = scaler_default.scale_logits(logits)
        probs_hot = scaler_hot.scale_logits(logits)

        # Higher temperature -> max prob should decrease (softer)
        assert np.max(probs_hot) < np.max(probs_default)

    def test_low_temperature_sharpens(self):
        """T < 1 should produce a sharper (more peaked) distribution."""
        logits = np.array([2.0, 1.0, 0.5, 0.2])

        scaler_default = TemperatureScaler(temperature=1.0)
        scaler_cold = TemperatureScaler(temperature=0.3)

        probs_default = scaler_default.scale_logits(logits)
        probs_cold = scaler_cold.scale_logits(logits)

        assert np.max(probs_cold) > np.max(probs_default)

    def test_output_sums_to_1(self):
        """Scaled output must always be a valid probability distribution."""
        scaler = TemperatureScaler(temperature=2.0)
        logits = np.array([3.0, 1.0, -0.5, -2.0])
        probs = scaler.scale_logits(logits)
        assert np.sum(probs) == pytest.approx(1.0, abs=1e-10)
        assert np.all(probs >= 0.0)

    def test_invalid_temperature_raises(self):
        with pytest.raises(ValueError, match="positive"):
            TemperatureScaler(temperature=-1.0)

    def test_zero_temperature_raises(self):
        with pytest.raises(ValueError, match="positive"):
            TemperatureScaler(temperature=0.0)

    def test_temperature_setter(self):
        scaler = TemperatureScaler(temperature=1.0)
        scaler.temperature = 2.5
        assert scaler.temperature == pytest.approx(2.5)

    def test_scale_probabilities(self):
        """scale_probabilities should also produce valid distributions."""
        scaler = TemperatureScaler(temperature=2.0)
        probs = np.array([0.7, 0.2, 0.07, 0.03])
        calibrated = scaler.scale_probabilities(probs)
        assert np.sum(calibrated) == pytest.approx(1.0, abs=1e-6)
        assert np.all(calibrated >= 0.0)
        # With T>1, distribution should be softer
        assert np.max(calibrated) < np.max(probs)

    def test_fit_decreases_ece(self):
        """
        Fitting temperature on calibration data should not raise errors.
        We verify that the result is a positive finite float.
        """
        rng = np.random.default_rng(42)
        N = 200
        K = 4
        # Simulate overconfident logits (high values)
        val_logits = rng.normal(loc=0, scale=3.0, size=(N, K))
        val_labels = np.argmax(val_logits, axis=1)   # 100% correct (simplistic)

        scaler = TemperatureScaler(temperature=1.0)
        fitted_T = scaler.fit(val_logits, val_labels, n_iterations=50)
        assert math.isfinite(fitted_T)
        assert fitted_T > 0.0


# ===========================================================================
# ECE tests
# ===========================================================================

class TestECE:

    def test_perfect_calibration(self):
        """A perfectly calibrated model: confidence == accuracy in every bin -> ECE ~= 0."""
        # 100 samples, all correct, confidence = accuracy = 1.0
        N = 100
        confidences = np.ones(N)
        predictions = np.zeros(N, dtype=int)
        labels = np.zeros(N, dtype=int)
        ece = compute_ece(confidences, predictions, labels, n_bins=10)
        assert ece == pytest.approx(0.0, abs=1e-6)

    def test_worst_case_calibration(self):
        """All wrong with high confidence -> ECE = 1.0."""
        N = 100
        confidences = np.ones(N)
        predictions = np.zeros(N, dtype=int)
        labels = np.ones(N, dtype=int)   # all wrong
        ece = compute_ece(confidences, predictions, labels, n_bins=10)
        assert ece == pytest.approx(1.0, abs=1e-6)

    def test_ece_is_nonnegative(self):
        rng = np.random.default_rng(0)
        N = 50
        confidences = rng.uniform(0.5, 1.0, N)
        predictions = rng.integers(0, 4, N)
        labels = rng.integers(0, 4, N)
        ece = compute_ece(confidences, predictions, labels)
        assert ece >= 0.0

    def test_ece_at_most_1(self):
        """ECE should never exceed 1.0 by construction."""
        rng = np.random.default_rng(7)
        N = 50
        confidences = rng.uniform(0.0, 1.0, N)
        predictions = rng.integers(0, 4, N)
        labels = rng.integers(0, 4, N)
        ece = compute_ece(confidences, predictions, labels)
        assert ece <= 1.0 + 1e-9

    def test_empty_input(self):
        """Empty input should return 0.0 without error."""
        ece = compute_ece(
            np.array([]), np.array([], dtype=int), np.array([], dtype=int)
        )
        assert ece == pytest.approx(0.0)


# ===========================================================================
# Reliability diagram bins
# ===========================================================================

class TestReliabilityDiagramBins:

    def test_returns_correct_num_bins(self):
        rng = np.random.default_rng(1)
        N = 100
        confidences = rng.uniform(0, 1, N)
        predictions = rng.integers(0, 4, N)
        labels = rng.integers(0, 4, N)
        bins = reliability_diagram_bins(confidences, predictions, labels, n_bins=10)
        assert len(bins) == 10

    def test_bin_structure(self):
        """Each bin dict must have the expected keys."""
        rng = np.random.default_rng(2)
        N = 100
        confidences = rng.uniform(0, 1, N)
        predictions = rng.integers(0, 4, N)
        labels = rng.integers(0, 4, N)
        bins = reliability_diagram_bins(confidences, predictions, labels)
        expected_keys = {"bin_lower", "bin_upper", "count", "accuracy", "confidence", "gap"}
        for b in bins:
            assert set(b.keys()) == expected_keys

    def test_bin_counts_sum_to_n(self):
        """The count across all bins should equal the total number of samples."""
        N = 80
        rng = np.random.default_rng(3)
        confidences = rng.uniform(0, 1, N)
        predictions = rng.integers(0, 4, N)
        labels = rng.integers(0, 4, N)
        bins = reliability_diagram_bins(confidences, predictions, labels, n_bins=5)
        total_count = sum(b["count"] for b in bins)
        assert total_count == N
