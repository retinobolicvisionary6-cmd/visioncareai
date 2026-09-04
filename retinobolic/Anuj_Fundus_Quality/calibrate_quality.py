import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(r"D:\anuj-fundus-quality")))
import src.focus as focus
from src.preprocessing import preprocess
import src.config as config

def calibrate(image_dir, max_images=300):
    image_dir = Path(image_dir)
    images = list(image_dir.glob("*.png"))[:max_images]
    
    print(f"Calibrating Quality Gate on {len(images)} APTOS images...")
    lap_vars = []
    grad_means = []
    
    for i, img_path in enumerate(images):
        if i % 50 == 0:
            print(f"  Processing {i}/{len(images)}...")
        try:
            p_dict = preprocess(str(img_path))
            gray = p_dict["gray"]
            mask = p_dict["fundus_mask"]
            metrics = focus.assess_focus(gray, mask)
            lap_vars.append(metrics["laplacian_var"])
            grad_means.append(metrics["tenengrad_mean"])
        except Exception as e:
            print(f"Failed {img_path.name}: {e}")
            
    lap_vars = np.array(lap_vars)
    grad_means = np.array(grad_means)
    
    print("\n--- FOCUS CALIBRATION (LAPLACIAN VARIANCE) ---")
    print(f"Current Config Ungradable: {config.FOCUS_LAP_VARIANCE_UNGRADABLE}")
    print(f"APTOS 1st percentile (worst 1%): {np.percentile(lap_vars, 1):.2f}")
    print(f"APTOS 10th percentile: {np.percentile(lap_vars, 10):.2f}")
    print(f"APTOS 50th percentile (median): {np.median(lap_vars):.2f}")
    
    print("\n--- FOCUS CALIBRATION (TENENGRAD MEAN) ---")
    print(f"Current Config Ungradable: {config.FOCUS_GRAD_UNGRADABLE}")
    print(f"APTOS 1st percentile (worst 1%): {np.percentile(grad_means, 1):.2f}")
    print(f"APTOS 10th percentile: {np.percentile(grad_means, 10):.2f}")
    print(f"APTOS 50th percentile (median): {np.median(grad_means):.2f}")

if __name__ == "__main__":
    calibrate(r"E:\retinobolic\data\raw\train_images", 300)
