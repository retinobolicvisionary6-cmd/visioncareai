import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(r"D:\anuj-ood")))
from src.embedding import extract_embedding
from src.reference import ReferenceDistribution
import src.config as config

def calibrate(image_dir, max_images=300):
    image_dir = Path(image_dir)
    images = list(image_dir.glob("*.png"))[:max_images]
    
    print(f"Calibrating OOD Module on {len(images)} APTOS images...")
    
    embeddings = []
    
    for i, img_path in enumerate(images):
        if i % 50 == 0:
            print(f"  Processing image {i}/{len(images)}...")
        try:
            emb = extract_embedding(str(img_path), extractor_type="classical")
            embeddings.append(emb)
        except Exception as e:
            print(f"Failed {img_path.name}: {e}")
            
    embeddings = np.array(embeddings)
    print(f"\nExtracted shape: {embeddings.shape}")
    
    ref = ReferenceDistribution()
    ref.fit(embeddings, extractor_type="classical")
    ref.save(config.REFERENCE_STATS_FILE, config.REFERENCE_EMBEDDINGS_FILE)
    
    print(f"\nCalibration complete!")
    print(f"Mean shape: {ref.mean.shape}")
    print(f"95th percentile distance threshold: {ref.percentiles['p95']:.4f}")
    
if __name__ == "__main__":
    calibrate(r"E:\retinobolic\data\raw\train_images", 300)
