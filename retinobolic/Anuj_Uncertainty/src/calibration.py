"""
calibration.py - Optional calibration and evaluation utilities.

IMPORTANT: This module is OPTIONAL for the 3-day MVP.
The core uncertainty engine (uncertainty.py) works fully without it.

Contents
--------
1. TemperatureScaler
   Post-hoc temperature scaling to calibrate overconfident neural network outputs.
   Must be applied to logits or raw probabilities BEFORE passing to the uncertainty
   engine. Calibration is separate from inference.

2. compute_ece()
   Expected Calibration Error - an EVALUATION metric only.
   Do NOT use ECE as a real-time uncertainty score.

3. reliability_diagram_bins()
   Bin statistics for plotting reliability diagrams (offline evaluation).

Architecture (when using calibration):
    Raw logits / probabilities
          |
    TemperatureScaler.scale()
          |
    Calibrated probabilities
          |
    uncertainty.calculate_uncertainty()

WARNING:
    Temperature calibration requires a held-out calibration/validation dataset.
    Never fit temperature on the test set or during inference.
    Never fit temperature during clinical operation.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Temperature Scaling
# ---------------------------------------------------------------------------

class TemperatureScaler:
    """
    Post-hoc temperature scaling for neural network probability calibration.

    Neural networks are often overconfident (their maximum probability is
    systematically higher than their empirical accuracy). Temperature scaling
    divides the logits by a learned scalar T > 1 to produce a softer, better-
    calibrated distribution.

    Formula:
        P_calibrated(y=k | z) = softmax(z / T)[k]
                               = exp(z_k / T) / sum_j( exp(z_j / T) )

    Where:
        z  = raw logits (pre-softmax)
        T  = temperature (T > 1 -> softer/more uncertain, T < 1 -> sharper)

    Usage
    -----
    # Option A: use a pre-determined temperature
    scaler = TemperatureScaler(temperature=1.5)
    calibrated_probs = scaler.scale_logits(raw_logits)

    # Option B: fit temperature on a calibration dataset (offline, NOT during inference)
    scaler = TemperatureScaler()
    scaler.fit(val_logits, val_labels)           # one-time offline step
    calibrated_probs = scaler.scale_logits(raw_logits)

    # Option C: work with probabilities instead of logits (less precise)
    calibrated_probs = scaler.scale_probabilities(raw_probs)
    """

    def __init__(self, temperature: float = 1.0) -> None:
        """
        Parameters
        ----------
        temperature : float
            Initial temperature value. T=1.0 means no scaling (identity).
            T > 1 softens the distribution (more uncertainty).
            T < 1 sharpens the distribution (more confidence).
        """
        if temperature <= 0.0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        self._temperature = float(temperature)

    @property
    def temperature(self) -> float:
        """Current temperature value."""
        return self._temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError(f"Temperature must be positive, got {value}")
        self._temperature = float(value)

    def scale_logits(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling to raw logits and return calibrated probabilities.

        Parameters
        ----------
        logits : np.ndarray
            Raw pre-softmax logits. Shape (num_classes,) or (batch, num_classes).

        Returns
        -------
        np.ndarray
            Calibrated probability distribution(s). Same shape as input.
        """
        logits = np.asarray(logits, dtype=np.float64)
        scaled = logits / self._temperature
        return self._softmax(scaled)

    def scale_probabilities(self, probs: np.ndarray) -> np.ndarray:
        """
        Apply approximate temperature scaling to already-softmaxed probabilities.

        NOTE: This is less precise than working with raw logits because the
        inversion of softmax (computing log-probs) loses information about the
        original scale. Prefer scale_logits() when raw logits are available.

        Parameters
        ----------
        probs : np.ndarray
            Softmax probabilities. Shape (num_classes,) or (batch, num_classes).

        Returns
        -------
        np.ndarray
            Approximately calibrated probabilities.
        """
        probs = np.asarray(probs, dtype=np.float64)
        # Recover approximate logits via log (inverse of softmax up to a constant)
        eps = 1e-12
        log_probs = np.log(np.clip(probs, eps, 1.0))
        return self.scale_logits(log_probs)

    def fit(
        self,
        val_logits: np.ndarray,
        val_labels: np.ndarray,
        n_iterations: int = 100,
        learning_rate: float = 0.01,
    ) -> float:
        """
        Fit temperature by minimizing Negative Log-Likelihood on a calibration set.

        MUST be called offline on a held-out validation dataset.
        NEVER call this during inference or on the test set.

        Parameters
        ----------
        val_logits : np.ndarray
            Logits from the validation set. Shape (N, num_classes).
        val_labels : np.ndarray
            True class labels (integer). Shape (N,).
        n_iterations : int
            Number of gradient descent steps.
        learning_rate : float
            Step size for gradient descent.

        Returns
        -------
        float
            The fitted temperature value.
        """
        val_logits = np.asarray(val_logits, dtype=np.float64)
        val_labels = np.asarray(val_labels, dtype=np.int64)

        T = self._temperature

        for _ in range(n_iterations):
            scaled = val_logits / T
            probs = self._softmax(scaled)                    # (N, K)
            N = len(val_labels)
            # NLL gradient w.r.t. T
            # dNLL/dT = (1/N) * sum_i [ -logit_y_i / T^2 + sum_k p_k * logit_k / T^2 ]
            correct_logits = val_logits[np.arange(N), val_labels]  # (N,)
            expected_logits = np.sum(probs * val_logits, axis=1)    # (N,)
            grad = np.mean((-correct_logits + expected_logits)) / (T ** 2)
            T = T - learning_rate * grad
            T = max(1e-3, T)   # prevent collapse to zero or negative

        self._temperature = T
        return T

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x_shifted = x - np.max(x)
            e = np.exp(x_shifted)
            return e / np.sum(e)
        else:
            x_shifted = x - np.max(x, axis=1, keepdims=True)
            e = np.exp(x_shifted)
            return e / np.sum(e, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Expected Calibration Error (ECE) - Evaluation utility only
# ---------------------------------------------------------------------------

def compute_ece(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE measures how well a model's confidence corresponds to its empirical
    accuracy. A perfectly calibrated model has ECE = 0.

    Formula:
        ECE = sum_b ( |B_b| / N ) * |acc(B_b) - conf(B_b)|

    Where:
        B_b  = set of samples whose confidence falls in bin b
        N    = total number of samples
        acc  = fraction of correct predictions in the bin
        conf = mean confidence in the bin

    IMPORTANT: ECE is an EVALUATION METRIC only.
    Do NOT use ECE as a real-time uncertainty score in the pipeline.

    Parameters
    ----------
    confidences : np.ndarray
        Maximum predicted class probability (confidence score). Shape (N,).
    predictions : np.ndarray
        Predicted class indices. Shape (N,).
    labels : np.ndarray
        True class labels (integer). Shape (N,).
    n_bins : int
        Number of equal-width bins in [0, 1].

    Returns
    -------
    float
        Expected Calibration Error in [0, 1]. Lower is better.
    """
    confidences = np.asarray(confidences, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)

    n = len(confidences)
    if n == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Include right endpoint in last bin
        if i < n_bins - 1:
            mask = (confidences >= lo) & (confidences < hi)
        else:
            mask = (confidences >= lo) & (confidences <= hi)

        bin_size = np.sum(mask)
        if bin_size == 0:
            continue

        bin_acc = np.mean(predictions[mask] == labels[mask])
        bin_conf = np.mean(confidences[mask])
        ece += (bin_size / n) * abs(bin_acc - bin_conf)

    return float(ece)


def reliability_diagram_bins(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> list[dict]:
    """
    Compute per-bin statistics for plotting a reliability diagram.

    A reliability diagram plots mean confidence (x-axis) vs. empirical
    accuracy (y-axis) per bin. A perfectly calibrated model lies on the
    diagonal y = x.

    Parameters
    ----------
    confidences : np.ndarray
        Maximum predicted class probability. Shape (N,).
    predictions : np.ndarray
        Predicted class indices. Shape (N,).
    labels : np.ndarray
        True class labels. Shape (N,).
    n_bins : int
        Number of equal-width confidence bins.

    Returns
    -------
    list of dict
        One dict per bin with keys:
            "bin_lower"  : float - lower edge of confidence bin
            "bin_upper"  : float - upper edge of confidence bin
            "count"      : int   - number of samples in bin
            "accuracy"   : float - fraction correct in bin
            "confidence" : float - mean confidence in bin
            "gap"        : float - confidence - accuracy (positive = overconfident)
    """
    confidences = np.asarray(confidences, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < n_bins - 1:
            mask = (confidences >= lo) & (confidences < hi)
        else:
            mask = (confidences >= lo) & (confidences <= hi)

        count = int(np.sum(mask))
        if count == 0:
            bins.append({
                "bin_lower": lo,
                "bin_upper": hi,
                "count": 0,
                "accuracy": None,
                "confidence": None,
                "gap": None,
            })
            continue

        acc = float(np.mean(predictions[mask] == labels[mask]))
        conf = float(np.mean(confidences[mask]))
        bins.append({
            "bin_lower": lo,
            "bin_upper": hi,
            "count": count,
            "accuracy": acc,
            "confidence": conf,
            "gap": conf - acc,
        })

    return bins
