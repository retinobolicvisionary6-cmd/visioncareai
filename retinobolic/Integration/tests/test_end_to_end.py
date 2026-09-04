import pytest
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from Integration.pipeline.screening_pipeline import run_pipeline

SAMPLE_IMAGE = str(PROJECT_ROOT / "Vinayak_DR_gradCam_Xai" / "tests" / "sample_test_retina.jpg")

@pytest.mark.skipif(not os.path.exists(SAMPLE_IMAGE), reason="Sample image not found")
def test_pipeline_runs_without_error():
    result = run_pipeline(
        image_path=SAMPLE_IMAGE,
        age=58,
        bp_systolic=148,
        bp_diastolic=92,
        hba1c=8.2,
        diabetes_duration_years=10
    )
    
    assert "quality" in result
    assert "final_decision" in result
    
    # If it's gradable, these should exist
    if result["quality"]["status"] != "ungradable":
        assert "dr_result" in result
        assert "reliability" in result
        assert "clinical_context" in result

        dr = result["dr_result"]
        assert "grade" in dr
        assert "probabilities" in dr
        assert "gradcam_path" in dr
