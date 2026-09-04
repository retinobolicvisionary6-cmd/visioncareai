"""
config.py — Clinical Context Module: Configurable bounds and units.

All numeric bounds represent plausible physiological ranges used for
input validation only. They are NOT diagnostic thresholds and do NOT
imply any clinical classification.

Units:
    Age              → years
    Blood Pressure   → mmHg
    HbA1c            → %
    Diabetes Duration→ years
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Unit Constants
# ---------------------------------------------------------------------------
AGE_UNIT: str = "years"
BP_UNIT: str = "mmHg"
HBA1C_UNIT: str = "%"
DIABETES_DURATION_UNIT: str = "years"


# ---------------------------------------------------------------------------
# Default Configurable Bounds
# ---------------------------------------------------------------------------
@dataclass
class ClinicalConfig:
    """
    Configurable validation bounds for clinical input fields.

    These ranges represent plausible physiological values used for
    sanity-checking data entry. They are NOT clinical diagnostic thresholds.
    Override any bound by instantiating with different values.
    """

    # Age bounds (years)
    age_min: int = 0
    age_max: int = 130

    # Blood pressure bounds (mmHg)
    bp_systolic_min: float = 40.0
    bp_systolic_max: float = 300.0
    bp_diastolic_min: float = 20.0
    bp_diastolic_max: float = 200.0

    # HbA1c bounds (%) — unit must be explicitly % before entry
    hba1c_min: float = 2.0
    hba1c_max: float = 25.0

    # Diabetes duration bounds (years)
    diabetes_duration_min: float = 0.0
    diabetes_duration_max: float = 100.0


# Module-level default instance (used by validation layer)
DEFAULT_CONFIG = ClinicalConfig()
