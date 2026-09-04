"""
conftest.py — Shared pytest fixtures for the Reliability Engine test suite.

Fixtures provided:
    dr_result_high_confidence  — high-confidence, low-uncertainty DR output
    dr_result_high_uncertainty — nearly-uniform distribution (high entropy)
    dr_result_low_confidence   — low max probability
    dr_result_medium           — intermediate signals
    dr_result_all_failures     — worst-case: low conf, high uncertainty
    mock_confidence_high       — pre-built confidence result dict
    mock_uncertainty_low       — pre-built uncertainty result dict
    mock_ood_in_distribution   — pre-built OOD result dict (not OOD)
    mock_ood_out_of_distribution — pre-built OOD result dict (OOD)
    default_config             — default ReliabilityConfig
"""

import sys
from pathlib import Path

import pytest

# Ensure project root is on path so 'src' is importable
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# DR result fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dr_result_high_confidence():
    """Grade 2 — Moderate DR, high confidence (0.88), low entropy."""
    return {
        "grade": 2,
        "probabilities": {"0": 0.01, "1": 0.01, "2": 0.97, "3": 0.01},
    }


@pytest.fixture
def dr_result_high_uncertainty():
    """Nearly uniform distribution → high Shannon entropy."""
    return {
        "grade": 1,
        "probabilities": {"0": 0.26, "1": 0.28, "2": 0.24, "3": 0.22},
    }


@pytest.fixture
def dr_result_low_confidence():
    """Grade 0 — max probability 0.35, all classes compete."""
    return {
        "grade": 0,
        "probabilities": {"0": 0.35, "1": 0.25, "2": 0.20, "3": 0.20},
    }


@pytest.fixture
def dr_result_medium():
    """Medium confidence (~0.55), medium uncertainty scenario."""
    return {
        "grade": 2,
        "probabilities": {"0": 0.15, "1": 0.20, "2": 0.55, "3": 0.10},
    }


@pytest.fixture
def dr_result_all_failures():
    """Nearly uniform: low confidence + high uncertainty."""
    return {
        "grade": 0,
        "probabilities": {"0": 0.27, "1": 0.24, "2": 0.25, "3": 0.24},
    }


# ---------------------------------------------------------------------------
# Pre-built module result fixtures (for unit-testing fusion/rules in isolation)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_confidence_high():
    return {
        "predicted_grade": 2,
        "predicted_class_name": "Moderate DR",
        "confidence": 0.88,
        "confidence_percent": 88.0,
        "confidence_level": "high",
        "margin": 0.83,
    }


@pytest.fixture
def mock_confidence_medium():
    return {
        "predicted_grade": 2,
        "predicted_class_name": "Moderate DR",
        "confidence": 0.62,
        "confidence_percent": 62.0,
        "confidence_level": "medium",
        "margin": 0.42,
    }


@pytest.fixture
def mock_confidence_low():
    return {
        "predicted_grade": 0,
        "predicted_class_name": "No DR",
        "confidence": 0.35,
        "confidence_percent": 35.0,
        "confidence_level": "low",
        "margin": 0.10,
    }


@pytest.fixture
def mock_uncertainty_low():
    return {
        "predicted_grade": 2,
        "uncertainty": 0.18,
        "uncertainty_level": "low",
        "review_recommended": False,
        "probability_margin": 0.83,
    }


@pytest.fixture
def mock_uncertainty_medium():
    return {
        "predicted_grade": 2,
        "uncertainty": 0.52,
        "uncertainty_level": "medium",
        "review_recommended": False,
        "probability_margin": 0.42,
    }


@pytest.fixture
def mock_uncertainty_high():
    return {
        "predicted_grade": 1,
        "uncertainty": 0.82,
        "uncertainty_level": "high",
        "review_recommended": True,
        "probability_margin": 0.02,
    }


@pytest.fixture
def mock_ood_in_distribution():
    return {
        "ood": False,
        "ood_status": "in_distribution",
        "ood_score": 1.50,
        "threshold": 3.17,
        "distance_metric": "mahalanobis",
        "extractor_type": "classical",
        "reason": "Input embedding is within the configured reference distribution.",
        "metadata": {},
    }


@pytest.fixture
def mock_ood_out_of_distribution():
    return {
        "ood": True,
        "ood_status": "review_required",
        "ood_score": 5.80,
        "threshold": 3.17,
        "distance_metric": "mahalanobis",
        "extractor_type": "classical",
        "reason": "Input embedding is substantially distant from the reference fundus distribution.",
        "metadata": {},
    }


@pytest.fixture
def default_config():
    from src.reliability.config import DEFAULT_CONFIG
    return DEFAULT_CONFIG
