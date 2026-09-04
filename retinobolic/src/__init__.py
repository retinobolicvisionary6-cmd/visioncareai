"""
VISIONARY6 — VINAYAK Module
AI-Assisted Diabetic Retinopathy Screening & Grad-CAM Explainability

Primary integration interface:
    from src.inference import predict
    result = predict("path/to/fundus_image.jpg")

Output contract:
    {
        "grade": int,                    # 0=No DR, 1=Mild, 2=Moderate, 3=Severe/PDR
        "probabilities": {"0": float, "1": float, "2": float, "3": float},
        "gradcam_path": str              # path to Grad-CAM overlay image
    }
"""
