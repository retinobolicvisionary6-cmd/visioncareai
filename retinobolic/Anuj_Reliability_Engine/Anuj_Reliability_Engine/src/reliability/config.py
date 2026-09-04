"""
src/reliability/config.py — Reliability Engine configuration.

All thresholds are externalized here so they can be tuned without
touching the rules or fusion logic.

IMPORTANT — PROTOTYPE THRESHOLDS
---------------------------------
All numeric thresholds in this file are PROTOTYPE ENGINEERING VALUES.
They have NOT been clinically validated and MUST NOT be used for
medical decision-making without independent clinical evaluation.

The thresholds mirror those in the upstream modules (confidence / uncertainty)
so that the Reliability Engine makes consistent level-based decisions.

External override:
    Load from config/thresholds.json using load_config().
    The JSON file may define a subset of fields; unspecified fields
    retain their default values.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


@dataclass
class ReliabilityConfig:
    """
    Centralised, externalizable configuration for the Reliability Engine.

    Decision thresholds (aligned with upstream modules)
    ---------------------------------------------------
    Confidence thresholds mirror anuj-confidence/src/config.py:
        HIGH_CONFIDENCE_THRESHOLD  = 0.80
        LOW_CONFIDENCE_THRESHOLD   = 0.50  (below this → "low")

    Uncertainty thresholds mirror anuj-uncertainty/src/config.py:
        LOW_UNCERTAINTY_MAX        = 0.35
        HIGH_UNCERTAINTY_MIN       = 0.70

    OOD
    ---
        OOD_STRICT_MODE = True  — OOD always triggers review_required
                                  regardless of other signals.

    Future extensibility
    --------------------
        Additional signal thresholds (camera reliability, clinical context)
        can be added here without touching the rules engine.
    """

    # --- Confidence level thresholds (must match anuj-confidence config) ---
    HIGH_CONFIDENCE_THRESHOLD: float = 0.80
    LOW_CONFIDENCE_THRESHOLD: float = 0.50   # strictly below → "low"

    # --- Uncertainty level thresholds (must match anuj-uncertainty config) ---
    LOW_UNCERTAINTY_MAX: float = 0.35        # at or below → "low"
    HIGH_UNCERTAINTY_MIN: float = 0.70       # at or above → "high"

    # --- OOD behaviour ---
    OOD_STRICT_MODE: bool = True             # OOD always → review_required

    # --- Caution / acceptable boundary ---
    # When confidence is "medium" and uncertainty is "medium" → caution
    # No additional numeric thresholds needed: level strings drive decisions.

    def validate_self(self) -> None:
        """Sanity-check configuration values. Raises ValueError on inconsistency."""
        if not (0.0 <= self.LOW_CONFIDENCE_THRESHOLD < self.HIGH_CONFIDENCE_THRESHOLD <= 1.0):
            raise ValueError(
                f"Confidence thresholds must satisfy "
                f"0 <= LOW ({self.LOW_CONFIDENCE_THRESHOLD}) "
                f"< HIGH ({self.HIGH_CONFIDENCE_THRESHOLD}) <= 1"
            )
        if not (0.0 <= self.LOW_UNCERTAINTY_MAX < self.HIGH_UNCERTAINTY_MIN <= 1.0):
            raise ValueError(
                f"Uncertainty thresholds must satisfy "
                f"0 <= LOW_MAX ({self.LOW_UNCERTAINTY_MAX}) "
                f"< HIGH_MIN ({self.HIGH_UNCERTAINTY_MIN}) <= 1"
            )


# ---------------------------------------------------------------------------
# Default config singleton
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = ReliabilityConfig()


# ---------------------------------------------------------------------------
# JSON config loader
# ---------------------------------------------------------------------------

def load_config(
    path: Optional[Union[str, Path]] = None,
) -> ReliabilityConfig:
    """
    Load a ReliabilityConfig from a JSON file.

    The JSON file may define any subset of ReliabilityConfig fields.
    Fields not present in the file retain their default values.

    Parameters
    ----------
    path : str or Path or None
        Path to a JSON configuration file.
        If None, the default config/thresholds.json in this project is used.

    Returns
    -------
    ReliabilityConfig — with values overridden by JSON where present.

    Raises
    ------
    FileNotFoundError — if the specified path does not exist.
    json.JSONDecodeError — if the file is not valid JSON.
    ValueError — if loaded values fail validate_self().
    """
    if path is None:
        # Default: config/thresholds.json relative to project root
        project_root = Path(__file__).resolve().parent.parent.parent
        path = project_root / "config" / "thresholds.json"

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Reliability Engine config file not found: {path}\n"
            "Use the default ReliabilityConfig() or provide a valid path."
        )

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a JSON object; got {type(data).__name__}.")

    cfg = ReliabilityConfig()

    # Apply overrides for known fields only (ignore unknown keys)
    float_fields = [
        "HIGH_CONFIDENCE_THRESHOLD",
        "LOW_CONFIDENCE_THRESHOLD",
        "LOW_UNCERTAINTY_MAX",
        "HIGH_UNCERTAINTY_MIN",
    ]
    bool_fields = ["OOD_STRICT_MODE"]

    for field_name in float_fields:
        if field_name in data:
            setattr(cfg, field_name, float(data[field_name]))

    for field_name in bool_fields:
        if field_name in data:
            setattr(cfg, field_name, bool(data[field_name]))

    cfg.validate_self()
    return cfg
