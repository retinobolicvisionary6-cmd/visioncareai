import os
import sys
from pathlib import Path

root = Path(r'c:\Users\asus\Desktop\html\c3lite\retinobolic')
print(f"Checking codebase integrity at: {root}\n")

expected_files = [
    'configs/config.py',
    'src/__init__.py',
    'src/utils.py',
    'src/preprocess.py',
    'src/dataset.py',
    'src/model.py',
    'src/train.py',
    'src/evaluate.py',
    'src/compare_models.py',
    'src/inference.py',
    'src/gradcam.py',
    'data/prepare_dataset.py',
    'data/splits/train.csv',
    'data/splits/val.csv',
    'data/splits/test.csv',
    'data/metadata/aptos2019_metadata.json',
    'models/checkpoints/best_model.pth',
    'models/class_mapping.json',
    'models/training_config.json',
    'tests/test_inference.py',
    'test_manual_pipeline.py',
    'README.md',
    'requirements.txt',
    'train.py',
    'evaluate.py',
    'inference.py',
]

missing = []
for rel in expected_files:
    p = root / rel
    exists = p.exists()
    status = 'OK' if exists else 'MISSING'
    size = f"{p.stat().st_size / 1024:.1f} KB" if exists else "N/A"
    print(f"  [{status:<7s}] {rel:<40s} ({size})")
    if not exists:
        missing.append(rel)

print('\n' + '='*60)
if missing:
    print(f"WARNING: {len(missing)} expected file(s) are missing!")
    for m in missing:
        print(f"  - {m}")
else:
    print("ALL EXPECTED FILES ARE PRESENT AND INTACT!")
print('='*60)
