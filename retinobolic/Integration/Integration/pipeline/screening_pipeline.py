"""
SIH26038 DR Screening Pipeline - Main Integration Layer
Orchestrates the flow from Image Quality through DR inference to Final Decision.
"""
import os
import sys
import importlib
from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def _import_from_subproject(project_name: str, module_path: str, function_name: str):
    """
    Safely imports a function from an independent sub-project without polluting the global `src` namespace.
    This resolves the conflict where multiple sub-projects use the `src` module name.
    """
    sub_path = str(PROJECT_ROOT / project_name)
    
    # Temporarily add the sub-project to sys.path
    sys.path.insert(0, sub_path)
    
    # Backup and clear existing 'src' modules from the global state
    saved_src_modules = {}
    for k in list(sys.modules.keys()):
        if k == 'src' or k.startswith('src.'):
            saved_src_modules[k] = sys.modules.pop(k)
            
    try:
        # Import the target module and extract the function
        mod = importlib.import_module(module_path)
        func = getattr(mod, function_name)
        
        # We must keep the loaded modules around so the function can run, 
        # but we rename them internally to avoid conflicts.
        # Actually, python functions retain references to their globals, 
        # but importing another `src` later will overwrite sys.modules['src'].
        # To make it perfectly robust, we can wrap the function to swap sys.modules during execution.
        
        sub_src_modules = {}
        for k in list(sys.modules.keys()):
            if k == 'src' or k.startswith('src.'):
                sub_src_modules[k] = sys.modules.pop(k)
                
        def wrapper(*args, **kwargs):
            # Backup current
            current_src = {}
            for k in list(sys.modules.keys()):
                if k == 'src' or k.startswith('src.'):
                    current_src[k] = sys.modules.pop(k)
            # Inject sub-project modules
            sys.path.insert(0, sub_path)
            sys.modules.update(sub_src_modules)
            try:
                return func(*args, **kwargs)
            finally:
                sys.path.pop(0)
                # Save any newly imported
                for k in list(sys.modules.keys()):
                    if k == 'src' or k.startswith('src.'):
                        sub_src_modules[k] = sys.modules.pop(k)
                # Restore current
                sys.modules.update(current_src)
                
        return wrapper
        
    finally:
        sys.path.pop(0)
        # Restore the original 'src' modules to sys.modules
        sys.modules.update(saved_src_modules)

# Safely import all entry points
assess_quality = _import_from_subproject("Anuj_Fundus_Quality", "src.quality", "assess_quality")
dr_predict = _import_from_subproject("Vinayak_DR_gradCam_Xai", "src.inference", "predict")
make_final_decision = _import_from_subproject("Anuj_Decision_Layer", "src.engine", "make_final_decision")
process_clinical_context = _import_from_subproject("Anuj_Clinical_Context", "src.clinical_context", "process_clinical_context")

# The Reliability Engine handles Confidence, Uncertainty, and OOD internally.
run_reliability_pipeline = _import_from_subproject("Anuj_Reliability_Engine", "src.reliability.engine", "run_reliability_pipeline")

def run_pipeline(
    image_path: str,
    age: Optional[int] = None,
    bp_systolic: Optional[int] = None,
    bp_diastolic: Optional[int] = None,
    hba1c: Optional[float] = None,
    diabetes_duration_years: Optional[int] = None,
    dr_model_checkpoint: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes the complete SIH26038 DR screening workflow.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # 1. Quality Gate
    quality_result = assess_quality(image_path)
    
    if quality_result.get("status") == "ungradable":
        clinical_result = process_clinical_context({
            "age": age,
            "bp_systolic": bp_systolic,
            "bp_diastolic": bp_diastolic,
            "hba1c": hba1c,
            "diabetes_duration_years": diabetes_duration_years
        })
        final_decision = make_final_decision(
            quality_result=quality_result,
            dr_result={},  
            reliability_result={},
            clinical_context=clinical_result
        )
        return {
            "quality": quality_result,
            "dr_result": None,
            "reliability": None,
            "clinical_context": clinical_result,
            "final_decision": final_decision
        }

    final_image_path = quality_result.get("enhanced_image_path") or image_path

    # 2. DR Model & Grad-CAM
    dr_result = dr_predict(
        image_path=final_image_path,
        checkpoint_path=dr_model_checkpoint,
        generate_gradcam=True
    )
    gradcam_path = dr_result.get("gradcam_path")

    # 3-6. Confidence, Uncertainty, OOD, and Reliability Engine
    reliability_result = run_reliability_pipeline(
        dr_result=dr_result,
        image_path=final_image_path
    )

    # 7. Clinical Context
    clinical_result = process_clinical_context({
        "age": age,
        "bp_systolic": bp_systolic,
        "bp_diastolic": bp_diastolic,
        "hba1c": hba1c,
        "diabetes_duration_years": diabetes_duration_years
    })

    # 8. Final Decision Layer
    final_decision = make_final_decision(
        quality_result=quality_result,
        dr_result=dr_result,
        reliability_result=reliability_result,
        clinical_context=clinical_result,
        gradcam_path=gradcam_path
    )

    return {
        "quality": quality_result,
        "dr_result": dr_result,
        "reliability": reliability_result,
        "clinical_context": clinical_result,
        "final_decision": final_decision
    }
