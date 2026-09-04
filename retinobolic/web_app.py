import os
import sys
import base64
import json
import tempfile
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS

# Add project root to sys path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from Integration.pipeline.screening_pipeline import run_pipeline

app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "templates"),
    static_folder=str(PROJECT_ROOT / "static")
)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max upload

SAMPLE_IMAGES = {
    0: str(PROJECT_ROOT / "data" / "processed_384" / "9c893e16c055.jpg"),
    1: str(PROJECT_ROOT / "data" / "processed_384" / "9eaf735cf01f.jpg"),
    2: str(PROJECT_ROOT / "data" / "processed_384" / "9c088d2d1559.jpg"),
    3: str(PROJECT_ROOT / "data" / "processed_384" / "4b618537d52f.jpg"),
    4: str(PROJECT_ROOT / "data" / "processed_384" / "07122e268a1d.jpg")
}

def img_to_base64(img_path: str) -> str:
    if not img_path or not os.path.exists(img_path):
        return ""
    try:
        with open(img_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
            ext = Path(img_path).suffix.lower()
            mime = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
            return f"data:{mime};base64,{encoded}"
    except Exception as e:
        print(f"Error encoding image {img_path}: {e}")
        return ""

@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/sample/<int:grade>", methods=["GET"])
def get_sample(grade):
    if grade not in SAMPLE_IMAGES:
        return jsonify({"error": "Invalid sample grade"}), 400
    img_path = SAMPLE_IMAGES[grade]
    b64 = img_to_base64(img_path)
    return jsonify({
        "grade": grade,
        "image_b64": b64,
        "filename": os.path.basename(img_path)
    })

@app.route("/api/analyze", methods=["POST"])
def analyze():
    temp_file = None
    try:
        image_path = None
        target_grade = None
        
        # 1. Check if uploaded file (Highest Priority)
        if "image" in request.files and request.files["image"].filename != "":
            file = request.files["image"]
            suffix = Path(file.filename).suffix or ".jpg"
            fd, temp_file = tempfile.mkstemp(suffix=suffix)
            os.close(fd)
            file.save(temp_file)
            image_path = temp_file
            print(f"[WebAPI] Custom file uploaded: {file.filename} -> {temp_file}")
            
        # 2. Check if sample grade requested
        elif request.form.get("sample_grade") is not None and request.form.get("sample_grade") != "":
            grade = int(request.form.get("sample_grade"))
            if grade in SAMPLE_IMAGES:
                image_path = SAMPLE_IMAGES[grade]
                target_grade = grade
                print(f"[WebAPI] Preset sample requested: Grade {grade} -> {image_path}")

        if not image_path or not os.path.exists(image_path):
            return jsonify({"error": "No valid fundus image provided."}), 400

        # Clinical parameters
        def parse_val(val, type_fn):
            if val is not None and str(val).strip() != "":
                try:
                    return type_fn(val)
                except Exception:
                    return None
            return None

        age = parse_val(request.form.get("age"), int)
        bp_systolic = parse_val(request.form.get("bp_systolic"), int)
        bp_diastolic = parse_val(request.form.get("bp_diastolic"), int)
        hba1c = parse_val(request.form.get("hba1c"), float)
        diabetes_duration = parse_val(request.form.get("diabetes_duration"), int)

        # Run real-time screening pipeline
        result = run_pipeline(
            image_path=image_path,
            age=age,
            bp_systolic=bp_systolic,
            bp_diastolic=bp_diastolic,
            hba1c=hba1c,
            diabetes_duration_years=diabetes_duration,
            target_grade=target_grade
        )

        # Encode input and gradcam images for instantaneous UI rendering
        input_b64 = img_to_base64(image_path)
        
        gradcam_overlay_b64 = ""
        gradcam_heatmap_b64 = ""
        gradcam_orig_b64 = ""
        
        dr_res = result.get("dr_result") or {}
        gc_path = dr_res.get("gradcam_path")
        if gc_path and os.path.exists(gc_path):
            gradcam_overlay_b64 = img_to_base64(gc_path)
            # Check sibling heatmap and original
            p = Path(gc_path)
            stem = p.name.replace("_overlay.jpg", "")
            heatmap_p = p.parent / f"{stem}_heatmap.jpg"
            orig_p = p.parent / f"{stem}_original.jpg"
            if heatmap_p.exists():
                gradcam_heatmap_b64 = img_to_base64(str(heatmap_p))
            if orig_p.exists():
                gradcam_orig_b64 = img_to_base64(str(orig_p))

        response_payload = {
            "success": True,
            "images": {
                "input": input_b64,
                "gradcam_overlay": gradcam_overlay_b64,
                "gradcam_heatmap": gradcam_heatmap_b64,
                "gradcam_original": gradcam_orig_b64
            },
            "pipeline": result
        }
        return jsonify(response_payload)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception:
                pass

if __name__ == "__main__":
    print("=" * 60)
    print("  RETINOBOLIC AI Diagnostic Web Server")
    print("  Theme: Cyber Red & Obsidian Black")
    port = int(os.environ.get("PORT", 5000))
    print(f"  Running at: http://127.0.0.1:{port}")
    print("=" * 60)
    app.run(host="0.0.0.0", port=port, debug=False)
